"""
Tiny real T5 chat/SFT run.

This is a small end-to-end plumbing check:
1. Build tokenizer artifacts in a repo-style base dir.
2. Reload them through the normal tokenizer helpers.
3. Train a tiny T5 on a tiny synthetic chat dataset.
4. Save a real checkpoint triplet (model/meta/optim).
5. Reload the checkpoint into a fresh T5.
6. Greedy-generate and assert expected answers.

Run:
    python -m scripts.t5_tiny_run
"""

import os
import tempfile
from dataclasses import asdict

import torch

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from nanochat.t5 import T5, T5Config
from nanochat.tokenizer import SentencePieceTokenizer, build_token_bytes, get_token_bytes, get_tokenizer, get_tokenizer_fingerprint


def synthetic_dataset():
    return [
        {
            "messages": [
                {"role": "system", "content": "Respond with just the answer."},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ],
            "expected": "4",
        },
        {
            "messages": [
                {"role": "system", "content": "Respond with just the answer."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "paris"},
            ],
            "expected": "paris",
        },
        {
            "messages": [
                {"role": "system", "content": "Respond with just the answer."},
                {"role": "user", "content": "What color is the sky on a clear day?"},
                {"role": "assistant", "content": "blue"},
            ],
            "expected": "blue",
        },
    ]


def build_tokenizer_corpus(dataset):
    texts = [
        "Respond with just the answer.",
        "Solve the problem briefly.",
        "Short answer only.",
    ]
    for example in dataset:
        for message in example["messages"]:
            texts.append(message["content"])
    return texts * 32


def save_repo_tokenizer(base_dir, tokenizer):
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer.save(tokenizer_dir)
    token_bytes = build_token_bytes(tokenizer, device="cpu")
    with open(os.path.join(tokenizer_dir, "token_bytes.pt"), "wb") as f:
        torch.save(token_bytes, f)


def build_seq2seq_examples(tokenizer, dataset):
    examples = []
    for example in dataset:
        examples.append({
            "example": tokenizer.render_sft_example({"messages": example["messages"]}),
            "expected": example["expected"],
        })
    return examples


def collate(tokenizer, batch):
    pad = tokenizer.get_pad_token_id()
    max_encoder = max(len(example["example"][0]) for example in batch)
    max_decoder = max(len(example["example"][1]) for example in batch)

    input_ids = torch.full((len(batch), max_encoder), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_encoder), dtype=torch.bool)
    decoder_input_ids = torch.full((len(batch), max_decoder), pad, dtype=torch.long)
    decoder_attention_mask = torch.zeros((len(batch), max_decoder), dtype=torch.bool)
    targets = torch.full((len(batch), max_decoder), -1, dtype=torch.long)

    for i, example in enumerate(batch):
        prompt_ids, decoder_ids, target_ids, decoder_mask = example["example"]
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        decoder_ids = torch.tensor(decoder_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        input_ids[i, :len(prompt_ids)] = prompt_ids
        attention_mask[i, :len(prompt_ids)] = True
        decoder_input_ids[i, :len(decoder_ids)] = decoder_ids
        decoder_attention_mask[i, :len(decoder_ids)] = torch.tensor(decoder_mask, dtype=torch.bool)
        targets[i, :len(target_ids)] = target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "targets": targets,
    }


def build_model(tokenizer, examples):
    max_encoder = max(len(example["example"][0]) for example in examples)
    max_decoder = max(len(example["example"][1]) for example in examples) + 4
    config = T5Config(
        encoder_sequence_len=max_encoder,
        decoder_sequence_len=max_decoder,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=96,
        ff_mult=2.0,
        dropout=0.0,
    )
    model = T5(config)
    model.init_weights()
    return model, config


def generate_answer(engine, tokenizer, prompt_ids, max_tokens):
    generated, _ = engine.generate_batch(prompt_ids, num_samples=1, max_tokens=max_tokens, temperature=0.0)
    completion = generated[0][len(prompt_ids):]
    return tokenizer.decode(completion).strip().lower()


def train_tiny_model(model, tokenizer, examples):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    max_steps = 500
    batch = collate(tokenizer, examples)

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch)
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == max_steps:
            model.eval()
            engine = Engine(model, tokenizer)
            outputs = [
                generate_answer(engine, tokenizer, example["example"][0], max_tokens=len(example["example"][2]) + 2)
                for example in examples
            ]
            print(f"step={step:03d} loss={loss.item():.4f} outputs={outputs}")
            if all(example["expected"] in output for example, output in zip(examples, outputs, strict=True)):
                return optimizer, step
            model.train()

    raise AssertionError("Tiny real T5 run failed to learn the expected answers")

def main():
    torch.manual_seed(0)
    dataset = synthetic_dataset()

    with tempfile.TemporaryDirectory(prefix="nanochat-t5-tiny-") as base_dir:
        os.environ["NANOCHAT_BASE_DIR"] = base_dir

        tokenizer = SentencePieceTokenizer.train_from_iterator(iter(build_tokenizer_corpus(dataset)), vocab_size=512)
        save_repo_tokenizer(base_dir, tokenizer)

        tokenizer = get_tokenizer()
        token_bytes = get_token_bytes()
        assert token_bytes.shape[0] == tokenizer.get_vocab_size()

        examples = build_seq2seq_examples(tokenizer, dataset)
        model, config = build_model(tokenizer, examples)
        optimizer, step = train_tiny_model(model, tokenizer, examples)

        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", "t5_tiny")
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            model_data=model.state_dict(),
            optimizer_data=optimizer.state_dict(),
            meta_data={
                "model_type": "t5",
                "step": step,
                "model_config": asdict(config),
                "tokenizer_fingerprint": get_tokenizer_fingerprint(tokenizer),
            },
            rank=0,
        )

        reloaded_model, reloaded_tokenizer, meta = load_model("sft", torch.device("cpu"), phase="eval", model_tag="t5_tiny", step=step)
        assert meta["model_type"] == "t5"
        reloaded_engine = Engine(reloaded_model, reloaded_tokenizer)

        outputs = {}
        for example in examples:
            output = generate_answer(
                reloaded_engine,
                reloaded_tokenizer,
                example["example"][0],
                max_tokens=len(example["example"][2]) + 2,
            )
            outputs[example["expected"]] = output
            print(f"expected={example['expected']!r} generated={output!r}")
            assert example["expected"] in output, f"Expected {example['expected']!r} in generated output {output!r}"

        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"step={step}")
        print(f"outputs={outputs}")


if __name__ == "__main__":
    main()
