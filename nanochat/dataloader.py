"""
Distributed seq2seq dataloaders for T5 pretraining.

Each document is turned into one T5 denoising example:
- encoder input starts with the document-start token
- masked spans are replaced by `<extra_id_*>` sentinels in the encoder input
- decoder target emits sentinel/span chunks plus EOS
- batches are padded to fixed shape and returned as dicts ready for T5
"""

import random

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import DENOISING_EXTRA_ID_OFFSET, EXTRA_ID_COUNT


NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0
MAX_DENOISING_SPANS = EXTRA_ID_COUNT - DENOISING_EXTRA_ID_OFFSET - 1

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def _random_segmentation(num_items, num_segments, rng):
    assert 1 <= num_segments <= num_items
    if num_segments == 1:
        return [num_items]
    cuts = sorted(rng.sample(range(1, num_items), num_segments - 1))
    lengths = []
    start = 0
    for end in cuts + [num_items]:
        lengths.append(end - start)
        start = end
    return lengths


def _random_spans_noise_mask(length, noise_density, mean_noise_span_length, rng):
    num_noise_tokens = min(max(int(round(length * noise_density)), 1), length - 1)
    num_nonnoise_tokens = length - num_noise_tokens
    num_noise_spans = min(
        max(int(round(num_noise_tokens / mean_noise_span_length)), 1),
        num_noise_tokens,
        num_nonnoise_tokens,
        MAX_DENOISING_SPANS,
    )
    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, rng)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, rng)
    mask = []
    for nonnoise_len, noise_len in zip(nonnoise_span_lengths, noise_span_lengths, strict=True):
        mask.extend([False] * nonnoise_len)
        mask.extend([True] * noise_len)
    return mask[:length]


def _build_denoising_example(tokenizer, body, max_input_tokens, max_target_tokens, rng):
    document_start = tokenizer.get_document_start_token_id()
    eos = tokenizer.get_eos_token_id()
    unk = tokenizer.get_unk_token_id()
    body = list(body)
    if len(body) < 2:
        body = body + [unk]
    max_body_tokens = min(
        int((max_input_tokens - 2) / (1.0 - NOISE_DENSITY)),
        max_target_tokens * 4,
    )
    body = body[:max(2, max_body_tokens)]
    for length in range(len(body), 1, -1):
        tokens = body[:length]
        noise_mask = _random_spans_noise_mask(length, NOISE_DENSITY, MEAN_NOISE_SPAN_LENGTH, rng)
        input_ids = [document_start]
        target_ids = []
        span_idx = 0
        i = 0
        while i < length:
            if noise_mask[i]:
                sentinel = tokenizer.get_denoising_token_id(span_idx)
                input_ids.append(sentinel)
                target_ids.append(sentinel)
                while i < length and noise_mask[i]:
                    target_ids.append(tokens[i])
                    i += 1
                span_idx += 1
            else:
                input_ids.append(tokens[i])
                i += 1
        target_ids.append(tokenizer.get_denoising_token_id(span_idx))
        target_ids.append(eos)
        if len(input_ids) <= max_input_tokens and len(target_ids) <= max_target_tokens:
            return input_ids, target_ids
    sentinel = tokenizer.get_denoising_token_id(0)
    return [document_start, sentinel], [sentinel, body[0], tokenizer.get_denoising_token_id(1), eos]


def tokenizing_distributed_seq2seq_loader_with_state(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
):
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1
    rng = random.Random(1234 + get_dist_info()[1])
    pad_token = tokenizer.get_pad_token_id()
    input_ids = torch.full((B, T), pad_token, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    targets = torch.full((B, T), -1, dtype=torch.long, device=device)

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, num_threads=tokenizer_threads)
        doc_buffer.extend(token_lists)

    while True:
        input_ids.fill_(pad_token)
        attention_mask.zero_()
        targets.fill_(-1)
        for row_idx in range(B):
            while len(doc_buffer) < buffer_size:
                refill_buffer()
            doc_tokens = doc_buffer.pop(0)
            source_ids, target_ids = _build_denoising_example(tokenizer, doc_tokens, T, T, rng)
            source_len = min(len(source_ids), T)
            target_len = min(len(target_ids), T)
            if source_len > 0:
                input_ids[row_idx, :source_len] = torch.tensor(source_ids[:source_len], dtype=torch.long, device=device)
                attention_mask[row_idx, :source_len] = True
            if target_len > 0:
                targets[row_idx, :target_len] = torch.tensor(target_ids[:target_len], dtype=torch.long, device=device)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        batch = {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "targets": targets.clone(),
        }
        yield batch, state_dict


def tokenizing_distributed_seq2seq_loader(*args, **kwargs):
    for batch, state_dict in tokenizing_distributed_seq2seq_loader_with_state(*args, **kwargs):
        yield batch
