"""
Tiny real T5 base-training run using the actual base_train script.

This creates repo-style tokenizer and parquet artifacts in a temporary base dir,
runs `scripts.base_train` for a couple of CPU steps, and verifies that the saved
base checkpoint reloads through the normal checkpoint manager.

Run:
    python -m scripts.t5_base_tiny_run
"""

import os
import subprocess
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_seq2seq_loader_with_state
from nanochat.tokenizer import SentencePieceTokenizer, build_token_bytes, get_tokenizer, get_tokenizer_fingerprint


TRAIN_DOCS = [
    "Paris is the capital of France and Berlin is the capital of Germany.",
    "The sky is blue on a clear day and the ocean can also look blue.",
    "Two plus two equals four and three plus three equals six.",
    "Apples grow on trees and strawberries grow near the ground.",
] * 32

VAL_DOCS = [
    "Cats purr and dogs bark.",
    "The moon orbits the earth and the earth orbits the sun.",
] * 8


def write_repo_tokenizer(base_dir):
    tokenizer = SentencePieceTokenizer.train_from_iterator(iter(TRAIN_DOCS + VAL_DOCS), vocab_size=512)
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer.save(tokenizer_dir)
    with open(os.path.join(tokenizer_dir, "token_bytes.pt"), "wb") as f:
        torch.save(build_token_bytes(tokenizer, device="cpu"), f)


def write_dataset(base_dir):
    data_dir = os.path.join(base_dir, "base_data_climbmix")
    os.makedirs(data_dir, exist_ok=True)
    train_table = pa.Table.from_pydict({"text": TRAIN_DOCS})
    val_table = pa.Table.from_pydict({"text": VAL_DOCS})
    pq.write_table(train_table, os.path.join(data_dir, "shard_00000.parquet"), row_group_size=8)
    pq.write_table(val_table, os.path.join(data_dir, "shard_00001.parquet"), row_group_size=8)


def inspect_denoising_loader():
    tokenizer = get_tokenizer()
    loader = tokenizing_distributed_seq2seq_loader_with_state(tokenizer, B=2, T=32, split="train", device="cpu")
    batch, state = next(loader)
    first_input = batch["input_ids"][0].tolist()
    first_target = batch["targets"][0].tolist()
    first_target = [token for token in first_target if token >= 0]
    assert tokenizer.get_denoising_token_id(0) in first_input
    assert tokenizer.get_denoising_token_id(0) in first_target
    assert state["epoch"] >= 1
    return first_input, first_target


def run_base_train(base_dir):
    env = os.environ.copy()
    env["NANOCHAT_BASE_DIR"] = base_dir
    cmd = [
        sys.executable,
        "-m",
        "scripts.base_train",
        "--device-type", "cpu",
        "--run", "dummy",
        "--model-tag", "tiny_base",
        "--depth", "1",
        "--aspect-ratio", "32",
        "--head-dim", "16",
        "--max-seq-len", "32",
        "--device-batch-size", "2",
        "--total-batch-size", "64",
        "--num-iterations", "2",
        "--target-flops", "-1",
        "--target-param-data-ratio", "-1",
        "--eval-every", "-1",
        "--core-metric-every", "-1",
        "--sample-every", "-1",
        "--save-every", "-1",
    ]
    subprocess.run(cmd, cwd=os.getcwd(), env=env, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix="nanochat-t5-base-") as base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = base_dir
        import nanochat.dataset as dataset_module
        dataset_module.base_dir = base_dir
        dataset_module.DATA_DIR = os.path.join(base_dir, "base_data_climbmix")
        write_repo_tokenizer(base_dir)
        write_dataset(base_dir)
        first_input, first_target = inspect_denoising_loader()
        print(f"input_has_sentinel={first_input[:12]}")
        print(f"target_has_sentinel={first_target[:12]}")

        run_base_train(base_dir)

        model, tokenizer, meta = load_model("base", torch.device("cpu"), phase="eval", model_tag="tiny_base")
        assert meta["model_type"] == "t5"
        assert meta["tokenizer_fingerprint"] == get_tokenizer_fingerprint(tokenizer)
        loader = tokenizing_distributed_seq2seq_loader_with_state(tokenizer, B=2, T=32, split="val", device="cpu")
        batch, _ = next(loader)
        loss = model(**batch)
        assert torch.isfinite(loss)
        print(f"checkpoint_step={meta['step']}")
        print(f"val_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
