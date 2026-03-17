# FP8 throughput results on 1x RTX PRO 6000 Blackwell (2026-03-16)

Goal: compare short bf16 vs fp8 throughput on a single Blackwell RTX PRO 6000 using the nanot5 T5 training stack.

Scope:
- model depth: 24
- seq len: 2048
- device batch size: 1
- num iterations: 6
- compare steps 2-5
- one train shard + one val shard
- tokenizer reused from local artifact
- environment used PyTorch 2.9.1 + cu128 on Vast.ai
- attention path: SDPA fallback, not FA3

Important caveat:
- these numbers are not "best possible Blackwell" numbers
- Flash Attention 3 was not active in this environment
- this is a short throughput sanity check, not a quality benchmark

## Summary

Result:
- fp8 worked after fixing the `_scaled_mm` row-major backward bug in `nanochat/fp8.py`
- fp8 showed a small but real throughput win over bf16 on this stack
- fp8 also reduced peak memory

Measured steady-state comparison (steps 2-5):
- bf16 avg tok/sec: 5430
- fp8 avg tok/sec: 5541
- fp8 speedup: about 2.0%
- bf16 avg dt: about 193.09 s
- fp8 avg dt: about 189.21 s
- bf16 peak memory: 68.29 GiB
- fp8 peak memory: 63.26 GiB
- fp8 memory reduction: about 5.03 GiB

Interpretation:
- the fp8 fix did not destroy performance
- fp8 is operational on Blackwell in this setup
- end-to-end gains are modest here, likely because the stack is still bottlenecked by SDPA fallback / non-FP8 portions of the training path

## Runs

### BF16 baseline

Source run:
- single Blackwell RTX PRO 6000 box
- batch size 1
- 6 steps

Logged steps:
- step 0: dt 265080.20 ms, tok/sec 3955
- step 1: dt 192798.34 ms, tok/sec 5438
- step 2: dt 193062.42 ms, tok/sec 5431
- step 3: dt 193125.12 ms, tok/sec 5429
- step 4: dt 193082.49 ms, tok/sec 5430
- step 5: dt 193096.01 ms, tok/sec 5430

Steady-state average over steps 2-5:
- avg tok/sec = (5431 + 5429 + 5430 + 5430) / 4 = 5430
- avg dt = (193062.42 + 193125.12 + 193082.49 + 193096.01) / 4 = 193091.51 ms

Peak memory:
- 68292.34 MiB (~68.29 GiB)

### FP8 rerun after fix

Fresh instance used for rerun:
- Vast instance: 32983716
- GPU reported: NVIDIA RTX PRO 6000 Blackwell Server Edition
- compute capability: (12, 0)

Startup confirmation:
- `✓ FP8 training enabled (tensorwise scaling) - converted 432/432 linear layers, skipped 0 (too small)`

Logged steps:
- step 0: dt 394647.27 ms, tok/sec 2656
- step 1: dt 191924.01 ms, tok/sec 5463
- step 2: dt 189282.20 ms, tok/sec 5539
- step 3: dt 189342.66 ms, tok/sec 5537
- step 4: dt 188855.28 ms, tok/sec 5552
- step 5: dt 189342.48 ms, tok/sec 5537

Steady-state average over steps 2-5:
- avg tok/sec = (5539 + 5537 + 5552 + 5537) / 4 = 5541.25
- avg dt = (189282.20 + 189342.66 + 188855.28 + 189342.48) / 4 = 189205.66 ms

Peak memory:
- 63258.73 MiB (~63.26 GiB)

## Comparison

### Throughput

bf16 steady-state tok/sec:
- 5430.00

fp8 steady-state tok/sec:
- 5541.25

Relative speedup:
- 5541.25 / 5430.00 = 1.02049x
- about +2.05%

### Step time

bf16 steady-state dt:
- 193091.51 ms

fp8 steady-state dt:
- 189205.66 ms

Relative improvement:
- 193091.51 / 189205.66 = 1.02054x
- about 2.05% faster per step

### Memory

bf16 peak memory:
- 68.29 GiB

fp8 peak memory:
- 63.26 GiB

Difference:
- about 5.03 GiB lower with fp8

## Bug encountered and fix

Original fp8 attempt failed before producing throughput numbers.

Failure:
- `_scaled_mm` backward layout error
- `RuntimeError: self must be row_major, got stride (1, 2048)`

Root cause:
- in `nanochat/fp8.py`, `_Float8Matmul.backward()` assumed `grad_output` was row-major
- under compiled training, `grad_output` could arrive column-major / non-contiguous
- `_to_fp8()` preserved that layout
- first backward `_scaled_mm` requires row-major first operand

Minimal fix:
- add `grad_output = grad_output.contiguous()` before quantizing it in backward

Regression test added:
- `tests/test_fp8.py`

## Operational notes

Instances used:
- initial instance: 32978210
- rerun instance: 32983716

Status:
- both instances were destroyed after the experiment

Repo state used for successful fp8 rerun:
- bookmark `feat/t5`
- commit `ea728c4f`

## Bottom line

For this nanot5 T5 training stack on 1x Blackwell RTX PRO 6000, with SDPA fallback active:
- fp8 is now working
- fp8 gives a modest throughput win (~2%)
- fp8 saves a modest amount of memory (~5 GiB)
- observed gains are below the stronger fp8 gains documented elsewhere in nanochat's prior H100/d26 notes, likely because this stack is not yet on the ideal fast path
