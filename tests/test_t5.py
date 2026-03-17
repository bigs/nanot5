import torch
from torch.testing import assert_close

import nanochat.t5 as t5_module
from nanochat.t5 import Attention, T5, T5Config


def test_t5_config_defaults_to_13_layers_per_stack():
    config = T5Config()
    assert config.n_layer == 13
    assert config.n_encoder_layer == 13
    assert config.n_decoder_layer == 13
    with torch.device("meta"):
        model = T5(config)
    assert len(model.encoder.h) == 13
    assert len(model.decoder.h) == 13


def build_model(tie_word_embeddings=True):
    config = T5Config(
        sequence_len=16,
        encoder_sequence_len=12,
        decoder_sequence_len=10,
        vocab_size=96,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        ff_mult=2.0,
        feed_forward="gated-gelu",
        tie_word_embeddings=tie_word_embeddings,
    )
    model = T5(config)
    model.init_weights()
    return model


def test_t5_attention_requires_selected_backend_when_fast_attention_is_active(monkeypatch):
    config = T5Config(
        sequence_len=8,
        encoder_sequence_len=8,
        decoder_sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=16,
        ff_hidden_size=32,
        dropout=0.0,
    )
    attn = Attention(config)
    calls = []

    def fake_flash_attn_func(q, k, v, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(q)

    monkeypatch.setattr(t5_module.flash_attention_module, "FAST_ATTN_BACKEND", "fa4")
    monkeypatch.setattr(t5_module.flash_attn, "flash_attn_func", fake_flash_attn_func)

    x = torch.randn(2, 4, config.n_embd)
    position_bias = torch.randn(1, config.n_head, 4, 4)

    attn(x, position_bias=position_bias)

    assert calls[0]["require_selected_backend"] is True


def test_t5_forward_loss():
    model = build_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    targets = torch.randint(0, model.config.vocab_size, (2, 5))
    targets[0, -1] = -1
    loss = model(input_ids=input_ids, targets=targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss2d = model(input_ids=input_ids, targets=targets, loss_reduction="none")
    assert loss2d.shape == targets.shape


def test_t5_forward_logits_and_cache():
    model = build_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
    decoder_input_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    logits, past_key_values, encoder_hidden_states = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        use_cache=True,
    )
    assert logits.shape == (2, 4, model.config.vocab_size)
    assert len(past_key_values) == model.config.n_decoder_layer
    assert encoder_hidden_states.shape == (2, 6, model.config.n_embd)


def test_t5_generate_and_optimizer():
    model = build_model(tie_word_embeddings=False)
    optimizer = model.setup_optimizer()
    assert len(optimizer.param_groups) > 0

    prompt = [3, 4, 5, 6]
    generated = list(model.generate(prompt, max_tokens=4, temperature=0.0))
    assert len(generated) <= 4
    assert all(isinstance(token, int) for token in generated)


def test_t5_encoder_padding_mask_ignores_masked_tokens():
    model = build_model()
    model.eval()
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    input_a = torch.tensor([[7, 8, 9, 0, 0, 0]])
    input_b = torch.tensor([[7, 8, 9, 10, 11, 12]])

    hidden_a = model.encode(input_a, attention_mask=attention_mask)
    hidden_b = model.encode(input_b, attention_mask=attention_mask)

    assert_close(hidden_a[:, :3], hidden_b[:, :3], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(hidden_a[:, 3:]) == 0
    assert torch.count_nonzero(hidden_b[:, 3:]) == 0


def test_t5_default_encoder_mask_keeps_first_document_start_token():
    model = build_model()
    model.eval()
    input_a = torch.tensor([[0, 7, 8, 9]])
    input_b = torch.tensor([[0, 7, 8, 9]])
    hidden_a = model.encode(input_a)
    hidden_b = model.encode(input_b, attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.bool))
    assert_close(hidden_a, hidden_b, atol=1e-5, rtol=1e-5)


def test_t5_decoder_padding_mask_ignores_masked_suffix():
    model = build_model()
    model.eval()
    input_ids = torch.tensor([[3, 4, 5, 6]])
    decoder_attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    decoder_a = torch.tensor([[0, 13, 14, 0, 0]])
    decoder_b = torch.tensor([[0, 13, 14, 21, 22]])

    logits_a = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_a,
        decoder_attention_mask=decoder_attention_mask,
    )
    logits_b = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_b,
        decoder_attention_mask=decoder_attention_mask,
    )

    assert_close(logits_a[:, :3], logits_b[:, :3], atol=1e-5, rtol=1e-5)


def test_t5_cross_attention_mask_ignores_masked_encoder_positions():
    model = build_model()
    model.eval()
    encoder_attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    encoder_hidden_a = torch.randn(1, 5, model.config.n_embd)
    encoder_hidden_b = encoder_hidden_a.clone()
    encoder_hidden_b[:, 3:] = torch.randn_like(encoder_hidden_b[:, 3:])
    decoder_input_ids = torch.tensor([[0, 15, 16]])
    decoder_attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

    logits_a = model(
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_a,
        encoder_attention_mask=encoder_attention_mask,
    )
    logits_b = model(
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_b,
        encoder_attention_mask=encoder_attention_mask,
    )

    assert_close(logits_a, logits_b, atol=1e-5, rtol=1e-5)
