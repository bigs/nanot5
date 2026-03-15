import random

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
