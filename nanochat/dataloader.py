"""
Distributed seq2seq dataloaders for T5 pretraining.

Each document is turned into one T5 denoising example:
- encoder input starts with the document-start token
- masked spans are replaced by `<extra_id_*>` sentinels in the encoder input
- decoder target emits sentinel/span chunks plus EOS
- batches are padded to fixed shape and returned as dicts ready for T5
"""

import os
import queue
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import DENOISING_EXTRA_ID_OFFSET, EXTRA_ID_COUNT


NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 3.0
MAX_DENOISING_SPANS = EXTRA_ID_COUNT - DENOISING_EXTRA_ID_OFFSET - 1
DEFAULT_PREFETCH_BATCHES = 4
DEFAULT_CORRUPTION_DRAWS_PER_DOC = 4


@dataclass
class _CachedDoc:
    token_ids: list[int]
    draws_left: int


class _ProducerError:
    def __init__(self, error):
        self.error = error


def _normalize_loader_state(state):
    if isinstance(state, dict):
        return dict(state)
    pq_idx, rg_idx, epoch = state
    return {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

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


def _resolve_tokenizer_workers(tokenizer_threads, tokenizer_workers):
    if tokenizer_workers is None:
        tokenizer_workers = min(max(1, tokenizer_threads), 4)
    tokenizer_workers = max(1, min(tokenizer_workers, tokenizer_threads))
    threads_per_worker = max(1, tokenizer_threads // tokenizer_workers)
    return tokenizer_workers, threads_per_worker


class _PrefetchingSeq2SeqLoader:
    def __init__(
        self,
        tokenizer,
        B,
        T,
        split,
        tokenizer_threads=4,
        tokenizer_workers=None,
        tokenizer_batch_size=128,
        device="cuda",
        resume_state_dict=None,
        buffer_size=1000,
        corruption_draws_per_doc=DEFAULT_CORRUPTION_DRAWS_PER_DOC,
        prefetch_batches=DEFAULT_PREFETCH_BATCHES,
    ):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        assert tokenizer_threads >= 1, "tokenizer_threads must be >= 1"
        assert buffer_size >= 1, "buffer_size must be >= 1"
        assert corruption_draws_per_doc >= 1, "corruption_draws_per_doc must be >= 1"
        assert prefetch_batches >= 1, "prefetch_batches must be >= 1"

        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.device = torch.device(device)
        self.use_pinned_memory = self.device.type == "cuda"
        self.buffer_size = buffer_size
        self.corruption_draws_per_doc = corruption_draws_per_doc
        self.prefetch_batches = prefetch_batches
        self.rng = random.Random(1234 + get_dist_info()[1])
        self.pad_token = tokenizer.get_pad_token_id()
        self.doc_batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
        self.doc_cache = deque()
        self.tokenization_futures = deque()
        self.last_resolved_state = {"pq_idx": 0, "rg_idx": 0, "epoch": 1}
        self.stop_event = threading.Event()
        self.batch_queue = queue.Queue(maxsize=prefetch_batches)
        self.tokenizer_workers, self.threads_per_worker = _resolve_tokenizer_workers(tokenizer_threads, tokenizer_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.tokenizer_workers, thread_name_prefix="t5tok")
        self.producer_thread = threading.Thread(target=self._producer_loop, name="t5-batch-producer", daemon=True)
        self.producer_thread.start()

    def _current_staged_docs(self):
        pending_docs = sum(size for _, _, size in self.tokenization_futures)
        return len(self.doc_cache) + pending_docs

    def _submit_tokenization(self):
        doc_batch, state_dict = next(self.doc_batches)
        future = self.executor.submit(self.tokenizer.encode, doc_batch, num_threads=self.threads_per_worker)
        self.tokenization_futures.append((future, _normalize_loader_state(state_dict), len(doc_batch)))

    def _schedule_tokenization(self):
        while (
            not self.stop_event.is_set()
            and len(self.tokenization_futures) < self.tokenizer_workers
            and self._current_staged_docs() < self.buffer_size
        ):
            self._submit_tokenization()

    def _collect_tokenized_docs(self, block=False):
        while self.tokenization_futures:
            future, state_dict, _ = self.tokenization_futures[0]
            if not block and not future.done():
                break
            self.tokenization_futures.popleft()
            token_lists = future.result()
            self.last_resolved_state = state_dict
            self.doc_cache.extend(
                _CachedDoc(token_ids=token_ids, draws_left=self.corruption_draws_per_doc)
                for token_ids in token_lists
            )
            block = False

    def _ensure_cached_docs(self, minimum_docs):
        while len(self.doc_cache) < minimum_docs:
            self._schedule_tokenization()
            self._collect_tokenized_docs(block=True)
        self._schedule_tokenization()
        self._collect_tokenized_docs(block=False)

    def _pop_cached_doc(self):
        self._ensure_cached_docs(1)
        cached_doc = self.doc_cache.popleft()
        token_ids = cached_doc.token_ids
        cached_doc.draws_left -= 1
        if cached_doc.draws_left > 0:
            self.doc_cache.append(cached_doc)
        return token_ids

    def _build_cpu_batch(self):
        self._ensure_cached_docs(self.B)
        input_ids = torch.full((self.B, self.T), self.pad_token, dtype=torch.long, pin_memory=self.use_pinned_memory)
        attention_mask = torch.zeros((self.B, self.T), dtype=torch.bool, pin_memory=self.use_pinned_memory)
        targets = torch.full((self.B, self.T), -1, dtype=torch.long, pin_memory=self.use_pinned_memory)

        for row_idx in range(self.B):
            doc_tokens = self._pop_cached_doc()
            source_ids, target_ids = _build_denoising_example(self.tokenizer, doc_tokens, self.T, self.T, self.rng)
            source_len = min(len(source_ids), self.T)
            target_len = min(len(target_ids), self.T)
            if source_len > 0:
                input_ids[row_idx, :source_len] = torch.tensor(source_ids[:source_len], dtype=torch.long)
                attention_mask[row_idx, :source_len] = True
            if target_len > 0:
                targets[row_idx, :target_len] = torch.tensor(target_ids[:target_len], dtype=torch.long)

        state_dict = dict(self.last_resolved_state)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
        }
        return batch, state_dict

    def _producer_loop(self):
        try:
            self._schedule_tokenization()
            self._collect_tokenized_docs(block=True)
            while not self.stop_event.is_set():
                batch = self._build_cpu_batch()
                while not self.stop_event.is_set():
                    try:
                        self.batch_queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as error:
            while not self.stop_event.is_set():
                try:
                    self.batch_queue.put(_ProducerError(error), timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _move_to_device(self, batch):
        if self.device.type == "cpu":
            return batch
        return {
            key: value.to(device=self.device, non_blocking=self.use_pinned_memory)
            for key, value in batch.items()
        }

    def __iter__(self):
        return self

    def __next__(self):
        item = self.batch_queue.get()
        if isinstance(item, _ProducerError):
            self.close()
            raise item.error
        batch, state_dict = item
        return self._move_to_device(batch), state_dict

    def close(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.producer_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        self.close()


def tokenizing_distributed_seq2seq_loader_with_state(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_workers=None, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000,
    corruption_draws_per_doc=DEFAULT_CORRUPTION_DRAWS_PER_DOC,
    prefetch_batches=DEFAULT_PREFETCH_BATCHES,
):
    loader = _PrefetchingSeq2SeqLoader(
        tokenizer,
        B,
        T,
        split,
        tokenizer_threads=tokenizer_threads,
        tokenizer_workers=tokenizer_workers,
        tokenizer_batch_size=tokenizer_batch_size,
        device=device,
        resume_state_dict=resume_state_dict,
        buffer_size=buffer_size,
        corruption_draws_per_doc=corruption_draws_per_doc,
        prefetch_batches=prefetch_batches,
    )
    try:
        while True:
            yield next(loader)
    finally:
        loader.close()


def tokenizing_distributed_seq2seq_loader(*args, **kwargs):
    loader = tokenizing_distributed_seq2seq_loader_with_state(*args, **kwargs)
    try:
        for batch, state_dict in loader:
            yield batch
    finally:
        loader.close()
