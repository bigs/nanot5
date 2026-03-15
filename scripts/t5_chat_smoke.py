"""
Throwaway T5 chat smoke test.

Trains a tiny SentencePiece tokenizer and a tiny T5 on a couple of synthetic
chat examples, then greedy-generates the answers back through the current
chat rendering path. This is intended as a quick end-to-end sanity check, not
as a quality benchmark or a permanent regression test.

Run:
    python -m scripts.t5_chat_smoke
"""

import torch

from nanochat.t5 import T5, T5Config
from nanochat.tokenizer import SentencePieceTokenizer


def build_tokenizer():
    corpus = [
        "What is 2 + 2?",
        "4",
        "The answer is 4.",
        "What is the capital of France?",
        "paris",
        "Paris is the capital of France.",
        "Solve the problem briefly.",
        "Respond with just the answer.",
    ] * 32
    return SentencePieceTokenizer.train_from_iterator(iter(corpus), vocab_size=512)


def build_examples(tokenizer):
    conversations = [
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
    ]

    examples = []
    assistant_end = tokenizer.get_chat_token_id("assistant_end")
    for example in conversations:
        full_ids, _ = tokenizer.render_conversation({"messages": example["messages"]})
        prompt_ids = tokenizer.render_for_completion({
            "messages": example["messages"][:-1] + [{"role": "assistant", "content": ""}],
        })
        assert full_ids[:len(prompt_ids)] == prompt_ids
        target_ids = full_ids[len(prompt_ids):]
        assert target_ids[-1] == assistant_end
        examples.append({
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "expected": example["expected"],
        })
    return examples


def collate_examples(tokenizer, examples):
    pad = tokenizer.get_pad_token_id()
    max_encoder = max(len(example["prompt_ids"]) for example in examples)
    max_decoder = max(len(example["target_ids"]) for example in examples)

    input_ids = torch.full((len(examples), max_encoder), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), max_encoder), dtype=torch.bool)
    targets = torch.full((len(examples), max_decoder), -1, dtype=torch.long)

    for i, example in enumerate(examples):
        prompt_ids = torch.tensor(example["prompt_ids"], dtype=torch.long)
        target_ids = torch.tensor(example["target_ids"], dtype=torch.long)
        input_ids[i, :len(prompt_ids)] = prompt_ids
        attention_mask[i, :len(prompt_ids)] = True
        targets[i, :len(target_ids)] = target_ids

    return input_ids, attention_mask, targets


def build_model(tokenizer, input_ids, targets):
    decoder_room = max(4, targets.size(1) + 2)
    config = T5Config(
        encoder_sequence_len=input_ids.size(1),
        decoder_sequence_len=decoder_room,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        ff_mult=2.0,
        dropout=0.0,
    )
    model = T5(config)
    model.init_weights()
    return model


def generate_answer(model, tokenizer, prompt_ids, max_tokens):
    assistant_end = tokenizer.get_chat_token_id("assistant_end")
    generated = []
    for token in model.generate(prompt_ids, max_tokens=max_tokens, temperature=0.0):
        if token == assistant_end:
            break
        generated.append(token)
    return tokenizer.decode(generated).strip().lower()


def main():
    torch.manual_seed(0)
    tokenizer = build_tokenizer()
    examples = build_examples(tokenizer)
    input_ids, attention_mask, targets = collate_examples(tokenizer, examples)
    model = build_model(tokenizer, input_ids, targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)

    model.train()
    max_steps = 300
    check_every = 25
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, targets=targets)
        loss.backward()
        optimizer.step()

        if step % check_every == 0 or step == max_steps:
            model.eval()
            decoded = [
                generate_answer(model, tokenizer, example["prompt_ids"], max_tokens=targets.size(1) + 2)
                for example in examples
            ]
            print(f"step={step:03d} loss={loss.item():.4f} outputs={decoded}")
            if all(example["expected"] in output for example, output in zip(examples, decoded, strict=True)):
                break
            model.train()
    else:
        raise AssertionError("Tiny T5 smoke run failed to learn the expected answers")

    for example in examples:
        output = generate_answer(model, tokenizer, example["prompt_ids"], max_tokens=targets.size(1) + 2)
        print(f"prompt={example['expected']!r} generated={output!r}")
        assert example["expected"] in output, f"Expected {example['expected']!r} in generated output {output!r}"


if __name__ == "__main__":
    main()
