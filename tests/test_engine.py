"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

from dataclasses import dataclass

import torch

from nanochat.engine import Engine


@dataclass
class MockConfig:
    decoder_sequence_len: int = 128


class MockModel:
    def __init__(self, vocab_size=263):
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def encode(self, input_ids, attention_mask=None):
        return torch.zeros(input_ids.size(0), input_ids.size(1), 8)

    def forward(
        self,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        B, T = decoder_input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size)
        if use_cache:
            return logits, past_key_values, encoder_hidden_states
        return logits


class ScriptedModel(MockModel):
    def __init__(self, script, vocab_size=263):
        super().__init__(vocab_size=vocab_size)
        self.script = script
        self.step = 0

    def forward(self, decoder_input_ids=None, encoder_hidden_states=None, past_key_values=None, use_cache=False, **kwargs):
        B, T = decoder_input_ids.shape
        logits = torch.full((B, T, self.vocab_size), -1e9)
        next_token = self.script[self.step] if self.step < len(self.script) else 260
        logits[:, -1, next_token] = 0.0
        self.step += 1
        if use_cache:
            return logits, past_key_values, encoder_hidden_states
        return logits


class ByteTokenizer:
    def __init__(self):
        self._special_tokens = {
            "python_start": 256,
            "python_end": 257,
            "output_start": 258,
            "output_end": 259,
            "assistant_end": 260,
            "document_start": 261,
            "eos": 262,
        }
        self._document_start = 261
        self._eos = 262

    def get_chat_token_id(self, name):
        return self._special_tokens[name]

    def get_document_start_token_id(self):
        return self._document_start

    def get_decoder_start_token_id(self):
        return self._document_start

    def get_pad_token_id(self):
        return self._document_start

    def get_eos_token_id(self):
        return self._eos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


def test_multi_sample_first_token_diversity():
    model = MockModel(vocab_size=263)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)
    prompt_tokens = [261, 72, 101, 108, 108, 111]

    first_tokens = []
    for token_column, _ in engine.generate(
        prompt_tokens,
        num_samples=16,
        max_tokens=1,
        temperature=1.0,
        seed=42,
    ):
        first_tokens = token_column

    assert len(set(first_tokens)) > 1


def test_seed_reproducibility():
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for seed in [1, 42, 123, 999]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r3, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2 == r3


def test_temperature_zero_determinism():
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=42)
    r3, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=123)
    assert r1 == r2 == r3


def test_max_tokens_respected():
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        assert len(results[0]) - len(prompt) <= max_tokens


def test_num_samples_count():
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=3)
        assert len(results) == num_samples


def test_different_seeds_introduce_variation_when_temperature_nonzero():
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    outputs = set()
    for seed in range(10):
        results, _ = engine.generate_batch(prompt, max_tokens=4, temperature=1.0, seed=seed)
        outputs.add(tuple(results[0]))
    assert len(outputs) > 1


def test_python_tool_forces_output_tokens():
    tokenizer = ByteTokenizer()
    script = [
        tokenizer.get_chat_token_id("python_start"),
        *tokenizer.encode("2+2"),
        tokenizer.get_chat_token_id("python_end"),
        tokenizer.get_chat_token_id("assistant_end"),
    ]
    model = ScriptedModel(script)
    engine = Engine(model, tokenizer)
    prompt = [tokenizer.get_document_start_token_id(), *tokenizer.encode("calc")]

    results, masks = engine.generate_batch(prompt, max_tokens=16, temperature=0.0)
    completion = results[0][len(prompt):]
    assert tokenizer.get_chat_token_id("output_start") in completion
    assert tokenizer.encode("4")[0] in completion
    assert tokenizer.get_chat_token_id("output_end") in completion
    forced_mask_idx = completion.index(tokenizer.get_chat_token_id("output_start")) + len(prompt)
    assert masks[0][forced_mask_idx] == 0
