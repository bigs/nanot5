"""
SentencePiece tokenizer used throughout nanot5.

The repo is T5-only, so we train/load a single SentencePiece model and use
native T5 special tokens. A few `<extra_id_*>` sentinels are repurposed for the
existing chat/tool formatting so the rest of the stack can stay lightweight.
"""

import copy
import hashlib
import io
import os
from functools import lru_cache


EXTRA_ID_COUNT = 100
T5_SPECIAL_TOKENS = ["<pad>", "</s>", "<unk>"] + [f"<extra_id_{i}>" for i in range(EXTRA_ID_COUNT)]

CHAT_TOKEN_PIECES = {
    "user_start": "<extra_id_0>",
    "user_end": "<extra_id_1>",
    "assistant_start": "<extra_id_2>",
    "assistant_end": "<extra_id_3>",
    "python_start": "<extra_id_4>",
    "python_end": "<extra_id_5>",
    "output_start": "<extra_id_6>",
    "output_end": "<extra_id_7>",
}
DENOISING_EXTRA_ID_OFFSET = len(CHAT_TOKEN_PIECES)


class SentencePieceTokenizer:
    """Light wrapper around SentencePiece for nanot5."""

    def __init__(self, processor, model_proto=None):
        self.processor = processor
        self.model_proto = model_proto
        self.pad_token_id = self._require_token("<pad>")
        self.eos_token_id = self._require_token("</s>")
        self.unk_token_id = self._require_token("<unk>")
        self.special_tokens = [token for token in T5_SPECIAL_TOKENS if self._lookup_token(token) is not None]
        self.chat_token_ids = {
            name: self._require_token(piece)
            for name, piece in CHAT_TOKEN_PIECES.items()
        }

    @classmethod
    def from_file(cls, model_path):
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_file=model_path)
        with open(model_path, "rb") as f:
            model_proto = f.read()
        return cls(processor, model_proto=model_proto)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        candidates = [
            os.path.join(tokenizer_dir, "tokenizer.model"),
            os.path.join(tokenizer_dir, "spiece.model"),
            os.path.join(tokenizer_dir, "sentencepiece.model"),
        ]
        for model_path in candidates:
            if os.path.exists(model_path):
                return cls.from_file(model_path)
        raise FileNotFoundError(f"No SentencePiece model found in {tokenizer_dir}")

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        import sentencepiece as spm

        model_buffer = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=text_iterator,
            model_writer=model_buffer,
            model_type="unigram",
            vocab_size=vocab_size,
            pad_id=0,
            eos_id=1,
            unk_id=2,
            bos_id=-1,
            byte_fallback=True,
            user_defined_symbols=[f"<extra_id_{i}>" for i in range(EXTRA_ID_COUNT)],
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            add_dummy_prefix=False,
            hard_vocab_limit=False,
        )
        model_proto = model_buffer.getvalue()
        processor = spm.SentencePieceProcessor(model_proto=model_proto)
        return cls(processor, model_proto=model_proto)

    def _lookup_token(self, text):
        token_id = self.processor.piece_to_id(text)
        if token_id < 0:
            return None
        return token_id if self.processor.id_to_piece(token_id) == text else None

    def _require_token(self, text):
        token_id = self._lookup_token(text)
        assert token_id is not None, f"Required token missing from SentencePiece model: {text}"
        return token_id

    def get_vocab_size(self):
        return self.processor.vocab_size()

    def get_special_tokens(self):
        return self.special_tokens

    def id_to_token(self, token_id):
        return self.processor.id_to_piece(token_id)

    @lru_cache(maxsize=128)
    def encode_special(self, piece):
        return self._require_token(piece)

    def get_pad_token_id(self):
        return self.pad_token_id

    def get_eos_token_id(self):
        return self.eos_token_id

    def get_unk_token_id(self):
        return self.unk_token_id

    def get_decoder_start_token_id(self):
        return self.pad_token_id

    def get_document_start_token_id(self):
        return self.pad_token_id

    def get_chat_token_id(self, name):
        return self.chat_token_ids[name]

    def get_extra_id_token_id(self, index):
        return self._require_token(f"<extra_id_{index}>")

    def get_denoising_token_id(self, span_index):
        return self.get_extra_id_token_id(DENOISING_EXTRA_ID_OFFSET + span_index)

    def get_fingerprint(self):
        assert self.model_proto is not None, "Tokenizer fingerprint requires serialized SentencePiece model bytes"
        return hashlib.sha256(self.model_proto).hexdigest()

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        assert isinstance(text, str)
        ids = list(self.processor.encode(text, out_type=int, num_threads=num_threads))
        if prepend is not None:
            ids.insert(0, prepend)
        if append is not None:
            ids.append(append)
        return ids

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        if isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.processor.decode(ids)

    def save(self, tokenizer_dir):
        assert self.model_proto is not None, "Tokenizer was not initialized from a serialized SentencePiece model"
        os.makedirs(tokenizer_dir, exist_ok=True)
        model_path = os.path.join(tokenizer_dir, "tokenizer.model")
        with open(model_path, "wb") as f:
            f.write(self.model_proto)
        print(f"Saved SentencePiece model to {model_path}")

    def _normalize_messages(self, conversation):
        if conversation["messages"][0]["role"] != "system":
            return conversation["messages"]
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[1]["role"] == "user", "System message must be followed by a user message"
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        return messages[1:]

    def _encode_assistant_content(self, content):
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        assistant_end = self.get_chat_token_id("assistant_end")
        python_start = self.get_chat_token_id("python_start")
        python_end = self.get_chat_token_id("python_end")
        output_start = self.get_chat_token_id("output_start")
        output_end = self.get_chat_token_id("output_end")

        if isinstance(content, str):
            add_tokens(self.encode(content), 1)
        elif isinstance(content, list):
            for part in content:
                value_ids = self.encode(part["text"])
                if part["type"] == "text":
                    add_tokens(value_ids, 1)
                elif part["type"] == "python":
                    add_tokens(python_start, 1)
                    add_tokens(value_ids, 1)
                    add_tokens(python_end, 1)
                elif part["type"] == "python_output":
                    add_tokens(output_start, 0)
                    add_tokens(value_ids, 0)
                    add_tokens(output_end, 0)
                else:
                    raise ValueError(f"Unknown part type: {part['type']}")
        else:
            raise ValueError(f"Unknown content type: {type(content)}")
        add_tokens(assistant_end, 1)
        return ids, mask

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single chat conversation and return:
        - ids: token ids
        - mask: supervision mask (1 where assistant text should be learned)
        """
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        messages = self._normalize_messages(conversation)
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        document_start = self.get_document_start_token_id()
        user_start = self.get_chat_token_id("user_start")
        user_end = self.get_chat_token_id("user_end")
        assistant_start = self.get_chat_token_id("assistant_start")

        add_tokens(document_start, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are expected to be strings"
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
                continue

            add_tokens(assistant_start, 0)
            content_ids, content_mask = self._encode_assistant_content(content)
            add_tokens(content_ids, 0)
            mask[-len(content_mask):] = content_mask

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.id_to_token(token_id)
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()
        ids, _ = self.render_conversation(conversation)
        ids.append(self.get_chat_token_id("assistant_start"))
        return ids

    def render_sft_example(self, conversation, max_input_tokens=2048, max_target_tokens=2048):
        prompt_ids = self.render_for_completion(conversation)
        assistant_content = conversation["messages"][-1]["content"]
        target_ids, target_mask = self._encode_assistant_content(assistant_content)
        if len(prompt_ids) > max_input_tokens:
            prompt_ids = [self.get_document_start_token_id()] + prompt_ids[-(max_input_tokens - 1):]
        if len(target_ids) > max_target_tokens:
            target_ids = target_ids[:max_target_tokens]
            target_mask = target_mask[:max_target_tokens]
        decoder_input_ids = [self.get_decoder_start_token_id()] + target_ids[:-1]
        targets = [token_id if mask_val else -1 for token_id, mask_val in zip(target_ids, target_mask)]
        decoder_attention_mask = [1] * len(decoder_input_ids)
        return prompt_ids, decoder_input_ids, targets, decoder_attention_mask


def get_tokenizer():
    from nanochat.common import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return SentencePieceTokenizer.from_directory(tokenizer_dir)


def get_tokenizer_fingerprint(tokenizer=None):
    tokenizer = get_tokenizer() if tokenizer is None else tokenizer
    return tokenizer.get_fingerprint()


def build_token_bytes(tokenizer, device="cpu"):
    import torch

    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32, device=device)
    for token_id in range(vocab_size):
        token_piece = tokenizer.id_to_token(token_id)
        if token_piece not in special_set:
            token_bytes[token_id] = len(tokenizer.decode([token_id]).encode("utf-8"))
    return token_bytes


def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
