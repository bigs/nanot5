from dataclasses import asdict

import pytest
import torch

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.t5 import T5, T5Config
from nanochat.tokenizer import SentencePieceTokenizer, build_token_bytes, get_tokenizer_fingerprint


def write_repo_tokenizer(base_dir):
    tokenizer = SentencePieceTokenizer.train_from_iterator(
        iter([
            "Respond with the answer only.",
            "What is 2 + 2?",
            "4",
            "What is the capital of France?",
            "paris",
        ] * 32),
        vocab_size=512,
    )
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer.save(tokenizer_dir)
    with open(tokenizer_dir / "token_bytes.pt", "wb") as f:
        torch.save(build_token_bytes(tokenizer, device="cpu"), f)
    return tokenizer


def write_t5_checkpoint(base_dir, tokenizer, meta_overrides=None):
    config = T5Config(
        sequence_len=16,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=16,
        ff_mult=2.0,
        pad_token_id=tokenizer.get_pad_token_id(),
        eos_token_id=tokenizer.get_eos_token_id(),
    )
    model = T5(config)
    model.init_weights()
    checkpoint_dir = base_dir / "chatsft_checkpoints" / "tiny"
    meta = {
        "model_type": "t5",
        "step": 0,
        "model_config": asdict(config),
        "tokenizer_fingerprint": get_tokenizer_fingerprint(tokenizer),
    }
    if meta_overrides:
        meta.update(meta_overrides)
    save_checkpoint(checkpoint_dir, 0, model.state_dict(), None, meta, rank=0)


def test_load_model_accepts_matching_t5_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tokenizer = write_repo_tokenizer(tmp_path)
    write_t5_checkpoint(tmp_path, tokenizer)

    model, loaded_tokenizer, meta = load_model("sft", torch.device("cpu"), phase="eval", model_tag="tiny", step=0)

    assert meta["model_type"] == "t5"
    assert model.config.vocab_size == tokenizer.get_vocab_size()
    assert loaded_tokenizer.get_fingerprint() == tokenizer.get_fingerprint()


def test_load_model_rejects_tokenizer_fingerprint_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tokenizer = write_repo_tokenizer(tmp_path)
    write_t5_checkpoint(tmp_path, tokenizer, meta_overrides={"tokenizer_fingerprint": "bad-fingerprint"})

    with pytest.raises(ValueError, match="Tokenizer fingerprint mismatch"):
        load_model("sft", torch.device("cpu"), phase="eval", model_tag="tiny", step=0)


def test_load_model_rejects_legacy_gpt_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    tokenizer = write_repo_tokenizer(tmp_path)
    write_t5_checkpoint(
        tmp_path,
        tokenizer,
        meta_overrides={
            "model_type": None,
            "model_config": {"vocab_size": tokenizer.get_vocab_size(), "window_pattern": "L"},
        },
    )

    with pytest.raises(ValueError, match="Legacy GPT checkpoints"):
        load_model("sft", torch.device("cpu"), phase="eval", model_tag="tiny", step=0)
