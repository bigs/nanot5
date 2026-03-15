import random
import threading

from nanochat import dataloader
from nanochat.dataloader import _build_denoising_example
from nanochat.tokenizer import SentencePieceTokenizer


def build_tokenizer():
    return SentencePieceTokenizer.train_from_iterator(
        iter([
            "The capital of France is Paris.",
            "The capital of Germany is Berlin.",
            "Rivers flow to the sea.",
        ] * 32),
        vocab_size=512,
    )


def test_build_denoising_example_uses_denoising_sentinels():
    tokenizer = build_tokenizer()
    doc_tokens = tokenizer.encode("The capital of France is Paris and Germany is Berlin.")
    input_ids, target_ids = _build_denoising_example(tokenizer, doc_tokens, 32, 32, random.Random(0))

    chat_token_ids = set(tokenizer.chat_token_ids.values())
    assert input_ids[0] == tokenizer.get_document_start_token_id()
    assert target_ids[-1] == tokenizer.get_eos_token_id()
    assert tokenizer.get_denoising_token_id(0) in input_ids
    assert tokenizer.get_denoising_token_id(0) in target_ids
    assert not any(token in chat_token_ids for token in input_ids[1:])
    assert not any(token in chat_token_ids for token in target_ids[:-1])


def test_seq2seq_loader_reuses_tokenized_docs_before_retokenizing(monkeypatch):
    base_tokenizer = build_tokenizer()

    class CountingTokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.encode_calls = 0
            self.lock = threading.Lock()

        def encode(self, *args, **kwargs):
            with self.lock:
                self.encode_calls += 1
            return self.tokenizer.encode(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.tokenizer, name)

    tokenizer = CountingTokenizer(base_tokenizer)

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        assert split == "train"
        docs = [
            "The capital of France is Paris.",
            "The capital of Germany is Berlin.",
        ]
        while True:
            yield docs, (0, 0, 1)

    monkeypatch.setattr(dataloader, "_document_batches", fake_document_batches)
    loader = dataloader.tokenizing_distributed_seq2seq_loader_with_state(
        tokenizer,
        B=1,
        T=32,
        split="train",
        tokenizer_threads=1,
        tokenizer_workers=1,
        tokenizer_batch_size=2,
        device="cpu",
        buffer_size=2,
        corruption_draws_per_doc=3,
        prefetch_batches=1,
    )
    try:
        for _ in range(3):
            batch, state = next(loader)
            assert batch["input_ids"].shape == (1, 32)
            assert batch["targets"].shape == (1, 32)
            assert state["epoch"] == 1
        assert tokenizer.encode_calls == 1
    finally:
        loader.close()
