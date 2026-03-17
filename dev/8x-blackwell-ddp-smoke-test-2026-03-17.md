# 8x RTX PRO 6000 Blackwell DDP smoke test (2026-03-17)

Goal: verify that nanot5 can launch a short distributed training run across 8x RTX PRO 6000 Blackwell GPUs on Vast.ai, complete a few steps, and checkpoint successfully.

## Summary

Result: success.

Confirmed:
- 8/8 GPUs visible
- all 8 GPUs reported as `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- all 8 GPUs reported compute capability `(12, 0)`
- `torchrun --standalone --nproc_per_node=8` initialized correctly
- distributed world size was `8`
- all 8 GPUs were actively utilized during training
- 2 training steps completed successfully
- checkpoint and optimizer states were written successfully
- no NCCL/DDP startup failure observed

Instance lifecycle:
- offer used: `32916017`
- instance created: `32995214`
- instance destroyed after the test

## Environment

Vast.ai offer:
- `8x RTX_PRO_6000_S`
- CUDA `13.1`
- driver `590.48.01`
- region: Maryland, US

Container image:
- `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`

Repo state:
- checked out `feat/t5`
- reset to `origin/feat/t5`
- commit lineage included the fp8 backward layout fix, but this smoke run used bf16 only

Dataset / tokenizer setup:
- tokenizer artifact copied from local cache
- downloaded exactly 1 train shard and 1 val shard via `python -m nanochat.dataset -n 1`

Attention path:
- SDPA fallback, not FA3

## GPU verification

Observed before training:
- `cuda_available = True`
- `device_count = 8`
- all devices named `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- all devices capability `(12, 0)`

## Smoke-test command

```bash
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --run=dummy \
  --model-tag=ddp-smoke-8x-bf16 \
  --depth=24 \
  --device-batch-size=1 \
  --num-iterations=2 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1
```

Notes:
- bf16 only
- intentionally minimal step count
- no eval/sample overhead
- this is a parallelism smoke test, not a throughput benchmark

## Key log confirmations

Distributed startup:
- `Distributed world size: 8`

Model/runtime summary:
- depth: 24
- sequence length: 2048
- per-rank device batch size: 1
- tokens / micro-batch / rank: 2048
- tokens / micro-batch total: 16384
- total batch size target: 1048576
- gradient accumulation steps: 64

Observed step logs:
- `step 00000/00002 ... dt: 132675.47ms | tok/sec: 7,903`
- `step 00001/00002 ... dt: 26484.40ms | tok/sec: 39,592`

Checkpointing completed:
- optimizer shards written for ranks 0-7
- `model_000002.pt` written
- `meta_000002.json` written

Peak memory reported:
- `41680.85 MiB`

Live GPU state during run:
- all 8 GPUs at roughly `97-100%` utilization
- each GPU around `~53.8 GiB` memory used during active run observation

## Interpretation

This test establishes that:
- 8-rank DDP initialization works on 8x Blackwell RTX PRO 6000s
- the nanot5 training stack can execute real distributed forward/backward/update steps on this hardware
- all ranks participate successfully
- checkpoint writing works under this 8x configuration

This does not establish:
- optimal throughput
- fp8 behavior on 8x
- FA4/attention fast-path behavior
- long-run training stability
- interconnect quality beyond a basic short-run smoke check

## Bottom line

The 8x Blackwell GPU parallelism smoke test passed.

nanot5 can successfully launch and complete a short 8-GPU distributed training run on Vast.ai using 8x `RTX_PRO_6000_S` cards.
