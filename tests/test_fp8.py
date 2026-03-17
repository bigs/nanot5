import ast
from pathlib import Path

import torch
import pytest
import torch.nn as nn

from nanochat.fp8 import (
    Float8Linear,
    Float8LinearConfig,
    _Float8MatmulOpaque,
    _Float8MatmulTraceable,
    _get_float8_matmul_op,
    convert_to_float8_training,
)


REQUIRES_FP8_SCALED_MM = not (
    hasattr(torch, "_scaled_mm")
    and hasattr(torch, "float8_e4m3fn")
    and hasattr(torch, "float8_e5m2")
)


@pytest.mark.skipif(REQUIRES_FP8_SCALED_MM, reason="requires torch float8 _scaled_mm support")
@pytest.mark.parametrize("matmul_op", [_Float8MatmulOpaque, _Float8MatmulTraceable])
def test_float8_backward_meta_accepts_row_major_grad_output(matmul_op):
    # The meta kernel enforces the same row-major / col-major contract as CUDA.
    x = torch.randn(16, 16, device="meta", requires_grad=True)
    w = torch.randn(16, 16, device="meta", requires_grad=True)
    config = Float8LinearConfig(opaque_autograd=matmul_op is _Float8MatmulOpaque)
    out = matmul_op.apply(x, w, config)

    grad_output = torch.randn(16, 16, device="meta").contiguous()
    grad_input, grad_weight = torch.autograd.grad(out, (x, w), grad_outputs=grad_output)

    assert grad_input.shape == x.shape
    assert grad_weight.shape == w.shape


@pytest.mark.skipif(REQUIRES_FP8_SCALED_MM, reason="requires torch float8 _scaled_mm support")
@pytest.mark.parametrize("matmul_op", [_Float8MatmulOpaque, _Float8MatmulTraceable])
def test_float8_backward_meta_accepts_column_major_grad_output(matmul_op):
    x = torch.randn(16, 16, device="meta", requires_grad=True)
    w = torch.randn(16, 16, device="meta", requires_grad=True)
    config = Float8LinearConfig(opaque_autograd=matmul_op is _Float8MatmulOpaque)
    out = matmul_op.apply(x, w, config)

    # This stride pattern previously triggered:
    # RuntimeError: self must be row_major, got stride (1, 16)
    grad_output = torch.randn(16, 16, device="meta").t().contiguous().t()
    assert grad_output.stride() == (1, 16)

    grad_input, grad_weight = torch.autograd.grad(out, (x, w), grad_outputs=grad_output)

    assert grad_input.shape == x.shape
    assert grad_weight.shape == w.shape


def test_get_float8_matmul_op_respects_opaque_autograd_flag():
    assert _get_float8_matmul_op(Float8LinearConfig()) is _Float8MatmulOpaque
    assert _get_float8_matmul_op(Float8LinearConfig(opaque_autograd=False)) is _Float8MatmulTraceable


def test_convert_to_float8_training_propagates_config():
    model = nn.Sequential(
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
    )
    config = Float8LinearConfig(
        opaque_autograd=False,
        forward_fast_accum=False,
        grad_input_fast_accum=True,
        grad_weight_fast_accum=True,
        fix_grad_input_layout=False,
        fix_grad_weight_layout=False,
    )

    convert_to_float8_training(model, config=config)

    fp8_layers = [module for module in model.modules() if isinstance(module, Float8Linear)]
    assert len(fp8_layers) == 2
    assert all(layer.fp8_config == config for layer in fp8_layers)


def test_base_train_cli_flags_feed_float8_config_keywords():
    source = Path("scripts/base_train.py").read_text()
    module = ast.parse(source)

    float8_config_calls = [
        node for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Float8LinearConfig"
    ]
    assert float8_config_calls, "expected Float8LinearConfig(...) call in scripts/base_train.py"

    config_call = None
    for call in float8_config_calls:
        keyword_names = {kw.arg for kw in call.keywords}
        if "opaque_autograd" in keyword_names:
            config_call = call
            break
    assert config_call is not None, "expected ablation Float8LinearConfig(...) call in scripts/base_train.py"

    actual = {}
    for kw in config_call.keywords:
        if kw.arg is None:
            continue
        assert isinstance(kw.value, ast.Attribute)
        assert isinstance(kw.value.value, ast.Name)
        assert kw.value.value.id == "args"
        actual[kw.arg] = kw.value.attr

    expected = {
        "opaque_autograd": "fp8_opaque_autograd",
        "forward_fast_accum": "fp8_forward_fast_accum",
        "grad_input_fast_accum": "fp8_grad_input_fast_accum",
        "grad_weight_fast_accum": "fp8_grad_weight_fast_accum",
        "fix_grad_input_layout": "fp8_fix_grad_input_layout",
        "fix_grad_weight_layout": "fp8_fix_grad_weight_layout",
    }
    assert actual == expected
