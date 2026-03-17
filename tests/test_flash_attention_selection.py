import types

import pytest
import torch

import nanochat.common as common
import nanochat.flash_attention as fa_module


@pytest.fixture(autouse=True)
def restore_fast_attention_state():
    original_backends = fa_module._fast_attn_backends
    original_override = fa_module._override_impl
    original_dtype = common.COMPUTE_DTYPE
    yield
    fa_module._fast_attn_backends = original_backends
    fa_module._override_impl = original_override
    common.COMPUTE_DTYPE = original_dtype
    fa_module._callable_accepts_kwarg.cache_clear()
    fa_module._refresh_fast_attention_state()


def configure_backends(monkeypatch, backends, dtype=torch.bfloat16, override=None):
    monkeypatch.setattr(common, "COMPUTE_DTYPE", dtype)
    monkeypatch.setattr(fa_module, "_fast_attn_backends", backends)
    fa_module._override_impl = override
    fa_module._refresh_fast_attention_state()


def make_fake_backend(fill_value, *, accept_dropout=True, error=None):
    calls = {"count": 0, "kwargs": []}

    if accept_dropout:
        def flash_attn_func(q, k, v, *, causal=False, window_size=(-1, -1), dropout_p=0.0):
            calls["count"] += 1
            calls["kwargs"].append(
                {"causal": causal, "window_size": window_size, "dropout_p": dropout_p}
            )
            if error is not None:
                raise error
            return torch.full_like(q, fill_value)
    else:
        def flash_attn_func(q, k, v, *, causal=False, window_size=(-1, -1)):
            calls["count"] += 1
            calls["kwargs"].append(
                {"causal": causal, "window_size": window_size}
            )
            if error is not None:
                raise error
            return torch.full_like(q, fill_value)

    return types.SimpleNamespace(flash_attn_func=flash_attn_func), calls


class TestBackendSelection:
    def test_classify_cuda_arch(self):
        assert fa_module._classify_cuda_arch((9, 0)) == "hopper"
        assert fa_module._classify_cuda_arch((10, 0)) == "blackwell"
        assert fa_module._classify_cuda_arch((11, 0)) == "blackwell"
        assert fa_module._classify_cuda_arch((12, 0)) == "blackwell"
        assert fa_module._classify_cuda_arch((8, 9)) is None

    def test_fa4_supported_capability_matches_current_package(self):
        assert fa_module._fa4_supports_cuda_capability((10, 0)) is True
        assert fa_module._fa4_supports_cuda_capability((11, 0)) is True
        assert fa_module._fa4_supports_cuda_capability((12, 0)) is False

    def test_fa4_loader_skips_unsupported_capability(self, monkeypatch):
        import_calls = {"count": 0}

        def fake_import(name):
            import_calls["count"] += 1
            return types.SimpleNamespace(flash_attn_func=object())

        monkeypatch.setattr(fa_module, "_get_cuda_capability", lambda: (12, 0))
        monkeypatch.setattr(fa_module, "import_module", fake_import)

        assert fa_module._load_flash_attention_4() is None
        assert import_calls["count"] == 0

    def test_auto_prefers_fa3_when_both_are_available(self, monkeypatch):
        fa3_backend, _ = make_fake_backend(3.0)
        fa4_backend, _ = make_fake_backend(4.0)
        configure_backends(monkeypatch, {"fa3": fa3_backend, "fa4": fa4_backend})

        assert fa_module.HAS_FA3 is True
        assert fa_module.HAS_FA4 is True
        assert fa_module.FAST_ATTN_BACKEND == "fa3"
        assert fa_module.USE_FA3 is True
        assert fa_module.USE_FA4 is False

    def test_auto_selects_fa4_for_blackwell_dtype(self, monkeypatch):
        fa4_backend, _ = make_fake_backend(4.0)
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float16)

        assert fa_module.HAS_FA4 is True
        assert fa_module.FAST_ATTN_BACKEND == "fa4"
        assert fa_module.USE_FA4 is True
        assert fa_module.USE_FAST_ATTENTION is True

    def test_auto_skips_fa4_for_float32(self, monkeypatch):
        fa4_backend, _ = make_fake_backend(4.0)
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float32)

        assert fa_module.HAS_FA4 is True
        assert fa_module.FAST_ATTN_BACKEND is None
        assert fa_module.USE_FAST_ATTENTION is False


class TestAttentionRouting:
    def test_fa4_path_handles_cross_attention_lengths(self, monkeypatch):
        fa4_backend, calls = make_fake_backend(4.0)
        configure_backends(monkeypatch, {"fa4": fa4_backend})

        q = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
        k = torch.randn(2, 5, 4, 8, dtype=torch.bfloat16)
        v = torch.randn(2, 5, 4, 8, dtype=torch.bfloat16)

        out = fa_module.flash_attn.flash_attn_func(q, k, v, causal=False)

        assert calls["count"] == 1
        assert torch.equal(out, torch.full_like(q, 4.0))

    def test_fa4_path_adapts_to_current_api_without_dropout_kwarg(self, monkeypatch):
        fa4_backend, calls = make_fake_backend(4.0, accept_dropout=False)
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float16)

        q = torch.randn(2, 3, 4, 8, dtype=torch.float32)
        k = torch.randn(2, 3, 4, 8, dtype=torch.float32)
        v = torch.randn(2, 3, 4, 8, dtype=torch.float32)

        out = fa_module.flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        assert calls["count"] == 1
        assert calls["kwargs"] == [{"causal": True, "window_size": (None, None)}]
        assert torch.equal(out, torch.full_like(q, 4.0))

    def test_fa4_dropout_falls_back_to_sdpa_when_backend_lacks_dropout(self, monkeypatch):
        fa4_backend, calls = make_fake_backend(4.0, accept_dropout=False)
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float32, override="fa4")

        q = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        k = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        v = torch.randn(1, 3, 2, 4, dtype=torch.float32)

        out = fa_module.flash_attn.flash_attn_func(q, k, v, causal=True, dropout_p=0.1)

        assert calls["count"] == 0
        assert out.shape == q.shape

    def test_masks_force_sdpa_even_when_fa4_is_selected(self, monkeypatch):
        fa4_backend, calls = make_fake_backend(4.0)
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float32, override="fa4")

        q = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        k = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        v = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        attn_mask = torch.tensor([[True, True, False]], dtype=torch.bool).view(1, 1, 1, 3)

        out = fa_module.flash_attn.flash_attn_func(q, k, v, causal=False, attn_mask=attn_mask)

        assert calls["count"] == 0
        assert out.shape == q.shape

    def test_require_selected_backend_raises_instead_of_mask_fallback(self, monkeypatch):
        fa4_backend, calls = make_fake_backend(4.0)
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float32, override="fa4")

        q = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        k = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        v = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        attn_mask = torch.tensor([[True, True, False]], dtype=torch.bool).view(1, 1, 1, 3)

        with pytest.raises(RuntimeError, match="Selected backend fa4 cannot handle this attention call"):
            fa_module.flash_attn.flash_attn_func(
                q,
                k,
                v,
                causal=False,
                attn_mask=attn_mask,
                require_selected_backend=True,
            )

        assert calls["count"] == 0

    def test_fa3_still_falls_back_for_cross_attention_lengths(self, monkeypatch):
        fa3_backend, calls = make_fake_backend(3.0)
        configure_backends(monkeypatch, {"fa3": fa3_backend}, override="fa3")

        q = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        k = torch.randn(1, 5, 2, 4, dtype=torch.float32)
        v = torch.randn(1, 5, 2, 4, dtype=torch.float32)

        out = fa_module.flash_attn.flash_attn_func(q, k, v, causal=False)

        assert calls["count"] == 0
        assert out.shape == q.shape

    def test_kvcache_uses_sdpa_when_fa4_is_selected(self, monkeypatch):
        fa4_backend, calls = make_fake_backend(4.0)
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float32, override="fa4")

        q = torch.randn(1, 2, 2, 4, dtype=torch.float32)
        k = torch.randn(1, 2, 2, 4, dtype=torch.float32)
        v = torch.randn(1, 2, 2, 4, dtype=torch.float32)
        k_cache = torch.zeros(1, 8, 2, 4, dtype=torch.float32)
        v_cache = torch.zeros(1, 8, 2, 4, dtype=torch.float32)
        cache_seqlens = torch.zeros(1, dtype=torch.int32)

        out = fa_module.flash_attn.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
            window_size=(8, 0),
        )

        assert calls["count"] == 0
        assert torch.equal(k_cache[:, :2], k)
        assert torch.equal(v_cache[:, :2], v)
        assert out.shape == q.shape

    def test_fa4_runtime_unsupported_error_disables_backend_and_falls_back(self, monkeypatch):
        fa4_backend, calls = make_fake_backend(
            4.0,
            accept_dropout=False,
            error=AssertionError("Unsupported compute capability. Supported: 9.x, 10.x, 11.x"),
        )
        configure_backends(monkeypatch, {"fa4": fa4_backend}, dtype=torch.float32, override="fa4")

        q = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        k = torch.randn(1, 3, 2, 4, dtype=torch.float32)
        v = torch.randn(1, 3, 2, 4, dtype=torch.float32)

        out = fa_module.flash_attn.flash_attn_func(q, k, v, causal=False)
        out_second = fa_module.flash_attn.flash_attn_func(q, k, v, causal=False)

        assert calls["count"] == 1
        assert out.shape == q.shape
        assert out_second.shape == q.shape
        assert fa_module.HAS_FA4 is False
        assert fa_module.FAST_ATTN_BACKEND is None
        assert fa_module.USE_FAST_ATTENTION is False
        assert fa_module._override_impl is None
