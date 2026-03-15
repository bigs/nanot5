import itertools

from nanochat.tokenizer import SentencePieceTokenizer


def build_tokenizer():
    corpus = [
        "Hello world",
        "  hi  there  ",
        "emoji 🌍",
        "你好世界",
        "User question\nAssistant answer",
        "```python\n2+2\n```",
    ]
    cycling_iter = itertools.islice(itertools.cycle(corpus), 128)
    return SentencePieceTokenizer.train_from_iterator(cycling_iter, vocab_size=400)


def test_sentencepiece_roundtrip_and_specials():
    tokenizer = build_tokenizer()
    text = "  spaced text  \nemoji 🌍 and 中文"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.get_document_start_token_id() == tokenizer.get_pad_token_id()
    assert tokenizer.get_chat_token_id("assistant_start") != tokenizer.get_chat_token_id("assistant_end")


def test_sentencepiece_chat_rendering():
    tokenizer = build_tokenizer()
    conversation = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "python", "text": "2+2"},
                    {"type": "python_output", "text": "4"},
                    {"type": "text", "text": "It is 4."},
                ],
            },
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation)
    assert len(ids) == len(mask)
    assert ids[0] == tokenizer.get_document_start_token_id()
    assert tokenizer.get_chat_token_id("assistant_end") in ids

    completion_prompt = tokenizer.render_for_completion({
        "messages": conversation["messages"] + [{"role": "assistant", "content": ""}]
    })
    assert completion_prompt[-1] == tokenizer.get_chat_token_id("assistant_start")
