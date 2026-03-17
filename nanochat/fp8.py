"""Minimal FP8 training for nanochat — tensorwise dynamic scaling only.

Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines.
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
subclass dispatch tables, etc.)

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

How this differs from torchao's approach
========================================
torchao uses a "tensor subclass" architecture: Float8TrainingTensor is a subclass
of torch.Tensor that bundles FP8 data + scale + metadata. It implements
__torch_dispatch__ with a dispatch table that intercepts every aten op (mm, t,
reshape, clone, ...) and handles it in FP8-aware fashion. When you call
  output = input @ weight.T
the @ operator dispatches to aten.mm, which gets intercepted and routed to
torch._scaled_mm behind the scenes. This is ~2000 lines of code because you need
a handler for every tensor operation that might touch an FP8 tensor.

We take a simpler approach: a single autograd.Function (_Float8Matmul) that takes
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
opaque node rather than trying to trace inside.

The trade-off is in how torch.compile sees the two approaches:
  - torchao: compile decomposes the tensor subclass (via __tensor_flatten__) and
    sees every individual op (amax, scale, cast, _scaled_mm) as separate graph
    nodes. Inductor can fuse these with surrounding operations (e.g. fuse the
    amax computation with the preceding layer's activation function).
  - ours: compile sees a single opaque call. It can optimize everything around
    the FP8 linear (attention, norms, etc.) but cannot fuse across the boundary.

Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
compared to the matmul. In practice this means our version is slightly faster
(less compilation overhead, no tensor subclass dispatch cost) but can produce
subtly different floating-point rounding paths under torch.compile, since Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


@dataclass(frozen=True)
class Float8LinearConfig:
    """Minimal config matching torchao's API plus local ablation switches."""

    opaque_autograd: bool = True
    forward_fast_accum: bool = True
    grad_input_fast_accum: bool = False
    grad_weight_fast_accum: bool = False
    fix_grad_input_layout: bool = True
    fix_grad_weight_layout: bool = True

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def _float8_matmul_forward(ctx, input_2d, weight, config):
    # Quantize both operands to e4m3 (higher precision format)
    input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
    weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
    ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)
    ctx.fp8_config = config

    # output = input @ weight.T
    # input_fp8 is [B, K] contiguous = row-major (good for first arg)
    # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
    # strides (1, K) = column-major (good for second arg, no copy needed!)
    return torch._scaled_mm(
        input_fp8,
        weight_fp8.t(),
        scale_a=input_inv,
        scale_b=weight_inv,
        out_dtype=input_2d.dtype,
        use_fast_accum=config.forward_fast_accum,
    )


def _float8_matmul_backward(ctx, grad_output):
    config = ctx.fp8_config
    in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

    # === GEMM 1: grad_input = grad_output @ weight ===
    # Shapes: [B, N] @ [N, K] -> [B, K]
    # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
    # grad_output may arrive with non-row-major strides from upstream views,
    # but _scaled_mm requires its first operand to be row-major.
    grad_input_lhs = grad_output.contiguous() if config.fix_grad_input_layout else grad_output
    go_fp8, go_inv = _to_fp8(grad_input_lhs, torch.float8_e5m2)
    # go_fp8 is [B, N] contiguous = row-major, good for first arg
    # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
    grad_input_rhs = _to_col_major(w_fp8) if config.fix_grad_input_layout else w_fp8
    grad_input = torch._scaled_mm(
        go_fp8,
        grad_input_rhs,
        scale_a=go_inv,
        scale_b=w_inv,
        out_dtype=grad_output.dtype,
        use_fast_accum=config.grad_input_fast_accum,
    )

    # === GEMM 2: grad_weight = grad_output.T @ input ===
    # Shapes: [N, B] @ [B, K] -> [N, K]
    # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
    # Transposing gives column-major, but first arg needs row-major,
    # so we may need .contiguous() to physically rearrange the memory.
    grad_weight_lhs = go_fp8.t().contiguous() if config.fix_grad_weight_layout else go_fp8.t()
    grad_weight_rhs = _to_col_major(in_fp8) if config.fix_grad_weight_layout else in_fp8
    grad_weight = torch._scaled_mm(
        grad_weight_lhs,
        grad_weight_rhs,
        scale_a=go_inv,
        scale_b=in_inv,
        out_dtype=grad_output.dtype,
        use_fast_accum=config.grad_weight_fast_accum,
    )

    return grad_input, grad_weight, None


@torch._dynamo.allow_in_graph
class _Float8MatmulOpaque(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer."""

    @staticmethod
    def forward(ctx, input_2d, weight, config):
        return _float8_matmul_forward(ctx, input_2d, weight, config)

    @staticmethod
    def backward(ctx, grad_output):
        return _float8_matmul_backward(ctx, grad_output)


class _Float8MatmulTraceable(torch.autograd.Function):
    """Same FP8 autograd, but without forcing an opaque torch.compile boundary."""

    @staticmethod
    def forward(ctx, input_2d, weight, config):
        return _float8_matmul_forward(ctx, input_2d, weight, config)

    @staticmethod
    def backward(ctx, grad_output):
        return _float8_matmul_backward(ctx, grad_output)


# Backwards compatibility for older tests/imports.
_Float8Matmul = _Float8MatmulOpaque


def _get_float8_matmul_op(config):
    return _Float8MatmulOpaque if config.opaque_autograd else _Float8MatmulTraceable


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute."""

    def __init__(self, *args, fp8_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp8_config = Float8LinearConfig() if fp8_config is None else fp8_config

    def forward(self, input):
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this.
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _get_float8_matmul_op(self.fp8_config).apply(input_2d, self.weight, self.fp8_config)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod, *, config=None):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias."""
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False, fp8_config=config)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are converted. Common use: skip layers with dims not divisible by 16
            (hardware requirement for FP8 matmuls on H100).
    """
    config = Float8LinearConfig() if config is None else config

    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child, config=config))

    _convert(module)
    return module
