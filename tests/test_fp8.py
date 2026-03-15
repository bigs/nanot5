import torch
import pytest

from nanochat.fp8 import _Float8Matmul


REQUIRES_FP8_SCALED_MM = not (
    hasattr(torch, "_scaled_mm")
    and hasattr(torch, "float8_e4m3fn")
    and hasattr(torch, "float8_e5m2")
)


@pytest.mark.skipif(REQUIRES_FP8_SCALED_MM, reason="requires torch float8 _scaled_mm support")
def test_float8_backward_meta_accepts_row_major_grad_output():
    # The meta kernel enforces the same row-major / col-major contract as CUDA.
    x = torch.randn(16, 16, device="meta", requires_grad=True)
    w = torch.randn(16, 16, device="meta", requires_grad=True)
    out = _Float8Matmul.apply(x, w)

    grad_output = torch.randn(16, 16, device="meta").contiguous()
    grad_input, grad_weight = torch.autograd.grad(out, (x, w), grad_outputs=grad_output)

    assert grad_input.shape == x.shape
    assert grad_weight.shape == w.shape


@pytest.mark.skipif(REQUIRES_FP8_SCALED_MM, reason="requires torch float8 _scaled_mm support")
def test_float8_backward_meta_accepts_column_major_grad_output():
    x = torch.randn(16, 16, device="meta", requires_grad=True)
    w = torch.randn(16, 16, device="meta", requires_grad=True)
    out = _Float8Matmul.apply(x, w)

    # This stride pattern previously triggered:
    # RuntimeError: self must be row_major, got stride (1, 16)
    grad_output = torch.randn(16, 16, device="meta").t().contiguous().t()
    assert grad_output.stride() == (1, 16)

    grad_input, grad_weight = torch.autograd.grad(out, (x, w), grad_outputs=grad_output)

    assert grad_input.shape == x.shape
    assert grad_weight.shape == w.shape
