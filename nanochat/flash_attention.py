"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` with the small API surface we use in the repo, while
falling back to PyTorch SDPA when FA3 cannot represent the requested masking
or bias semantics.
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


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
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), attn_mask=None, attn_bias=None, dropout_p=0.0):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.
        attn_mask: Optional key mask or additive mask, broadcastable to (B, H, Tq, Tk).
        attn_bias: Optional additive attention bias, broadcastable to (B, H, Tq, Tk).
        dropout_p: Attention dropout probability.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    use_fa3 = USE_FA3 and attn_mask is None and attn_bias is None and q.size(1) == k.size(1) == v.size(1)
    if use_fa3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size, dropout_p=dropout_p)

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
    use_fa3 = USE_FA3 and attn_mask is None and attn_bias is None
    if use_fa3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size, dropout_p=dropout_p
        )

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
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
