"""
Unified Flash Attention interface with automatic FA3/FA4/SDPA switching.

Exports `flash_attn` with the small API surface we use in the repo, while
falling back to PyTorch SDPA when the fast kernel cannot represent the
requested masking or bias semantics.
"""
from importlib import import_module
from functools import lru_cache
import inspect
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load the right fast-attention backend for the active GPU
# =============================================================================
_HOPPER_CAPABILITY_MAJORS = {9}
_BLACKWELL_CAPABILITY_MAJORS = {10, 11, 12}
_FA4_SUPPORTED_CAPABILITY_MAJORS = {10, 11}
_FA4_UNSUPPORTED_RUNTIME_SUBSTRINGS = (
    "Unsupported compute capability",
)


def _get_cuda_capability():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except Exception:
        return None


def _classify_cuda_arch(capability=None):
    """Map CUDA capability to the architecture family relevant for FA selection."""
    if capability is None:
        capability = _get_cuda_capability()
    if capability is None:
        return None

    major, _ = capability
    if major in _HOPPER_CAPABILITY_MAJORS:
        return "hopper"
    if major in _BLACKWELL_CAPABILITY_MAJORS:
        return "blackwell"
    return None


def _normalize_fa3_backend(module):
    """Normalize the slightly different FA3 module layouts we may load."""
    if hasattr(module, "flash_attn_func") and hasattr(module, "flash_attn_with_kvcache"):
        return SimpleNamespace(
            flash_attn_func=module.flash_attn_func,
            flash_attn_with_kvcache=module.flash_attn_with_kvcache,
        )

    interface = getattr(module, "flash_attn_interface", None)
    if interface is not None and hasattr(interface, "flash_attn_func") and hasattr(interface, "flash_attn_with_kvcache"):
        return SimpleNamespace(
            flash_attn_func=interface.flash_attn_func,
            flash_attn_with_kvcache=interface.flash_attn_with_kvcache,
        )

    raise AttributeError("FA3 module does not expose flash_attn_func/flash_attn_with_kvcache")


def _load_flash_attention_3():
    """Try to load Flash Attention 3 for Hopper GPUs."""
    if _classify_cuda_arch() != "hopper":
        return None

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from kernels import get_kernel
    except Exception:
        return None

    for repo_id in ("kernels-community/flash-attn3", "varunneal/flash-attention-3"):
        try:
            return _normalize_fa3_backend(get_kernel(repo_id))
        except Exception:
            continue
    return None


def _load_flash_attention_4():
    """Try to load FlashAttention-4 for Blackwell GPUs."""
    if _classify_cuda_arch() != "blackwell":
        return None
    if not _fa4_supports_cuda_capability():
        return None

    try:
        module = import_module("flash_attn.cute")
        if not hasattr(module, "flash_attn_func"):
            return None
        return SimpleNamespace(
            flash_attn_func=module.flash_attn_func,
            flash_attn_with_kvcache=None,
        )
    except Exception:
        return None


def _fa4_supports_cuda_capability(capability=None):
    """Current flash-attn-4 beta supports SM100/SM110, but not SM120 yet."""
    if capability is None:
        capability = _get_cuda_capability()
    if capability is None:
        return False
    major, _ = capability
    return major in _FA4_SUPPORTED_CAPABILITY_MAJORS


def _load_fast_attention_backends():
    arch = _classify_cuda_arch()
    if arch == "hopper":
        candidates = (("fa3", _load_flash_attention_3),)
    elif arch == "blackwell":
        candidates = (("fa4", _load_flash_attention_4),)
    else:
        candidates = ()

    backends = {}
    for name, loader in candidates:
        backend = loader()
        if backend is not None:
            backends[name] = backend
    return backends


_fast_attn_backends = _load_fast_attention_backends()

# Override for testing: set to 'fa3', 'fa4', 'sdpa', or None (auto)
_override_impl = None


def _backend_supports_dtype(backend_name):
    """Whether auto-selection should use the backend for the current compute dtype."""
    from nanochat.common import COMPUTE_DTYPE

    if backend_name == "fa3":
        # Keep the existing conservative FA3 gate: bf16 is the validated path in this repo.
        return COMPUTE_DTYPE == torch.bfloat16
    if backend_name == "fa4":
        return COMPUTE_DTYPE in (torch.float16, torch.bfloat16)
    return False


def _resolve_fast_attention_backend():
    """Pick the active backend name ('fa3', 'fa4', or None for SDPA)."""
    if _override_impl in {"fa3", "fa4"}:
        assert _override_impl in _fast_attn_backends, f"Cannot override to {_override_impl}: not available on this hardware"
        return _override_impl
    if _override_impl == "sdpa":
        return None

    for name in _fast_attn_backends:
        if _backend_supports_dtype(name):
            return name
    return None


def _refresh_fast_attention_state():
    global HAS_FA3, HAS_FA4, HAS_FAST_ATTENTION
    global FAST_ATTN_BACKEND, USE_FA3, USE_FA4, USE_FAST_ATTENTION

    HAS_FA3 = "fa3" in _fast_attn_backends
    HAS_FA4 = "fa4" in _fast_attn_backends
    HAS_FAST_ATTENTION = bool(_fast_attn_backends)
    FAST_ATTN_BACKEND = _resolve_fast_attention_backend()
    USE_FA3 = FAST_ATTN_BACKEND == "fa3"
    USE_FA4 = FAST_ATTN_BACKEND == "fa4"
    USE_FAST_ATTENTION = FAST_ATTN_BACKEND is not None


def _disable_fast_attention_backend(backend_name):
    global _fast_attn_backends, _override_impl

    if backend_name not in _fast_attn_backends:
        return
    _fast_attn_backends = {name: backend for name, backend in _fast_attn_backends.items() if name != backend_name}
    if _override_impl == backend_name:
        _override_impl = None
    _refresh_fast_attention_state()


def _resolve_use_fa3():
    """Backward-compatible helper retained for existing FA3-focused tests."""
    return _resolve_fast_attention_backend() == "fa3"


_refresh_fast_attention_state()


@lru_cache(maxsize=None)
def _callable_accepts_kwarg(func, kwarg_name):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    return kwarg_name in signature.parameters


def _normalize_fa4_window_size(window_size):
    left, right = window_size
    left = None if left is not None and left < 0 else left
    right = None if right is not None and right < 0 else right
    return left, right


def _fa4_runtime_is_unsupported_error(error):
    if not isinstance(error, (AssertionError, RuntimeError)):
        return False
    message = str(error)
    return any(substr in message for substr in _FA4_UNSUPPORTED_RUNTIME_SUBSTRINGS)


def _call_fast_attention_func(backend_name, q, k, v, *, causal, window_size, dropout_p):
    backend = _fast_attn_backends[backend_name]
    backend_window_size = _normalize_fa4_window_size(window_size) if backend_name == "fa4" else window_size
    kwargs = {
        "causal": causal,
        "window_size": backend_window_size,
    }
    if dropout_p != 0.0:
        kwargs["dropout_p"] = dropout_p
    return backend.flash_attn_func(q, k, v, **kwargs)


# =============================================================================
# SDPA helpers
# =============================================================================
def _to_additive_mask(mask, dtype):
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        zeros = torch.zeros((), dtype=dtype, device=mask.device)
        neg = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=mask.device)
        return torch.where(mask, zeros, neg)
    return mask.to(dtype=dtype)


def _build_structural_mask(Tq, Tk, device, causal, window_size):
    left, right = window_size
    if not causal and left < 0 and right < 0:
        return None

    q_idx = torch.arange(Tq, device=device).unsqueeze(1)
    k_idx = torch.arange(Tk, device=device).unsqueeze(0)
    if causal:
        q_idx = q_idx + (Tk - Tq)
        mask = k_idx <= q_idx
    else:
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=device)
    if left >= 0:
        mask = mask & ((q_idx - k_idx) <= left)
    if right >= 0:
        mask = mask & ((k_idx - q_idx) <= right)
    return mask


def _sdpa_attention(q, k, v, causal=False, window_size=(-1, -1), attn_mask=None, attn_bias=None, dropout_p=0.0):
    """
    SDPA attention with causal/bidirectional masking, sliding windows,
    additive attention bias, and key padding masks.

    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    enable_gqa = q.size(1) != k.size(1)
    structural_mask = _build_structural_mask(Tq, Tk, q.device, causal, window_size)

    if structural_mask is None and attn_mask is None and attn_bias is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p, enable_gqa=enable_gqa)

    additive_mask = _to_additive_mask(structural_mask, q.dtype)
    user_mask = _to_additive_mask(attn_mask, q.dtype)
    if user_mask is not None:
        additive_mask = user_mask if additive_mask is None else additive_mask + user_mask
    if attn_bias is not None:
        bias = attn_bias.to(dtype=q.dtype)
        additive_mask = bias if additive_mask is None else additive_mask + bias

    return F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask, dropout_p=dropout_p, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(
    q,
    k,
    v,
    causal=False,
    window_size=(-1, -1),
    attn_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    require_selected_backend=False,
):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
        attn_mask: Optional key mask or additive mask, broadcastable to (B, H, Tq, Tk).
        attn_bias: Optional additive attention bias, broadcastable to (B, H, Tq, Tk).
        dropout_p: Attention dropout probability.
        require_selected_backend: If True, raise instead of silently falling back when
            the currently selected fast-attention backend cannot service this call.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    backend_name = FAST_ATTN_BACKEND
    incompatibilities = []
    if backend_name is not None:
        if attn_mask is not None:
            incompatibilities.append("attn_mask is set")
        if attn_bias is not None:
            incompatibilities.append("attn_bias is set")
        if dropout_p != 0.0 and not _callable_accepts_kwarg(_fast_attn_backends[backend_name].flash_attn_func, "dropout_p"):
            incompatibilities.append(f"{backend_name} backend does not accept dropout_p")
        if backend_name == "fa3" and not (q.size(1) == k.size(1) == v.size(1)):
            incompatibilities.append("fa3 requires q, k, and v to have matching head counts")

    use_fast_attention = backend_name is not None and not incompatibilities
    if require_selected_backend and backend_name is not None and incompatibilities:
        details = ", ".join(incompatibilities)
        raise RuntimeError(f"Selected backend {backend_name} cannot handle this attention call: {details}")

    if use_fast_attention:
        try:
            return _call_fast_attention_func(
                backend_name,
                q,
                k,
                v,
                causal=causal,
                window_size=window_size,
                dropout_p=dropout_p,
            )
        except Exception as error:
            if backend_name == "fa4" and _fa4_runtime_is_unsupported_error(error):
                _disable_fast_attention_backend("fa4")
                if require_selected_backend:
                    raise RuntimeError(f"Selected backend {backend_name} failed at runtime: {error}") from error
            else:
                raise

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = _sdpa_attention(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
    )
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1), attn_mask=None, attn_bias=None, dropout_p=0.0):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
        attn_mask: Optional key mask or additive mask, broadcastable to (B, H, Tq, Tk).
        attn_bias: Optional additive attention bias, broadcastable to (B, H, Tq, Tk).
        dropout_p: Attention dropout probability.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3 and attn_mask is None and attn_bias is None:
        return _fast_attn_backends["fa3"].flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size, dropout_p=dropout_p
        )
    # FA4's simple training/inference API is `flash_attn_func`; keep KV-cache decode
    # on the SDPA fallback until we wire up and validate the CuTe paged-KV path.

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    y_sdpa = _sdpa_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        causal=causal,
        window_size=window_size,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
    )

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
