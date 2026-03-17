# Vast.ai Throughput Test Runbook (1x RTX PRO 6000 / Blackwell)

Goal: compare one short bf16 run vs one short FP8 run on a single Blackwell RTX PRO 6000, then destroy the machine.

## 0. GPU requirement

We want a Blackwell RTX PRO 6000.

We do not want `RTX_6000Ada`.

The Ada card is not the Blackwell Pro card.

Search examples:

```bash
vastai search offers 'reliability>0.97 num_gpus=1 rented=False disk_space>250 gpu_name in [RTX_PRO_6000,RTX_PRO_6000_S]' --limit 20 -o 'dph'
```

If Vast naming is inconsistent, try a broader search and inspect manually:

```bash
vastai search offers 'reliability>0.97 num_gpus=1 rented=False disk_space>250' --limit 50 -o 'dph'
```

Before creating the instance, verify the offer is the Blackwell RTX PRO 6000, not Ada.

## 1. Launch the instance

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
  --disk 250
```

Get the instance id and SSH info:

```bash
vastai show instances
vastai ssh-url INSTANCE_ID
```

## 2. SSH into the machine

```bash
ssh -i ~/.ssh/id_ed25519 root@HOST -p PORT
```

## 3. Basic machine setup

```bash
apt-get update
apt-get install -y git curl tmux rsync htop
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

## 4. Clone and install nanot5

```bash
mkdir -p ~/Code
cd ~/Code
git clone https://github.com/bigs/nanot5.git
cd nanot5
uv venv
uv sync --extra gpu
source .venv/bin/activate
```

## 5. Verify the machine before spending time on setup

```bash
nvidia-smi
python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
print('device_count=', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name=', torch.cuda.get_device_name(0))
PY
```

Expected:
- `cuda_available= True`
- `device_count= 1`
- GPU name should clearly be Blackwell RTX PRO 6000 class, not Ada

If the GPU is wrong, destroy the instance immediately and start over.

## 6. Set artifact/cache dir

Prefer a concrete path so later copy commands are unambiguous.

```bash
export NANOCHAT_BASE_DIR=/workspace/nanochat-cache
mkdir -p /workspace/nanochat-cache
mkdir -p /workspace/nanochat-cache/base_checkpoints
```

If `/workspace` is unavailable:

```bash
export NANOCHAT_BASE_DIR=/root/.cache/nanochat
mkdir -p /root/.cache/nanochat
mkdir -p /root/.cache/nanochat/base_checkpoints
```

## 7. W&B

Optional.

If you want W&B:

```bash
wandb login
```

If not, use `--run dummy` below.

## 8. Fetch exactly one train shard plus one val shard

This is a short throughput test, not a quality run.

```bash
python -m nanochat.dataset -n 1
```

This should give:
- 1 train shard
- 1 val shard

## 9. Tokenizer artifact

Best option: reuse a known-good tokenizer artifact for this repo if you have one.

If you do not have one on the machine yet, train it once:

```bash
python -m scripts.tok_train --max-chars 250000000 --vocab-size 32768
```

Do this once before the throughput runs. Do not include tokenizer time in throughput comparison.

## 10. BF16 throughput run

Keep flags identical between bf16 and FP8 except for `--fp8`.

Use a short run. Ignore early warmup steps when comparing throughput.

```bash
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --run=dummy \
  --model-tag=throughput-bf16 \
  --depth=24 \
  --target-param-data-ratio=8 \
  --device-batch-size=16 \
  --num-iterations=20 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1
```

Capture from logs:
- startup line showing GPU / dtype
- whether Flash Attention 3 or SDPA fallback was used
- per-step `dt`
- per-step `tok/sec`
- MFU
- any OOM / NaN / kernel errors

Comparison rule: discard the first 5 steps, then average steps 6-20.

## 11. FP8 throughput run

Same setup, only add `--fp8`.

```bash
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --run=dummy \
  --model-tag=throughput-fp8 \
  --depth=24 \
  --target-param-data-ratio=8 \
  --device-batch-size=16 \
  --num-iterations=20 \
  --fp8 \
  --eval-every=-1 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1
```

Also capture:
- startup line confirming FP8 conversion enabled
- whether Flash Attention 3 or SDPA fallback was used
- per-step `dt`
- per-step `tok/sec`
- MFU
- any OOM / NaN / kernel errors

Comparison rule: discard the first 5 steps, then average steps 6-20.

## 12. Manual monitoring during runs

```bash
watch -n 2 nvidia-smi
```

Record any obvious memory delta between bf16 and FP8.

## 13. Success/fail gates

BF16 pass:
- no crash
- no NaN/Inf issues
- steady logging through all 20 steps

FP8 pass:
- no crash
- no NaN/Inf issues
- startup confirms FP8 conversion enabled
- steady logging through all 20 steps

Abort / investigate if:
- wrong GPU
- CUDA unavailable
- FP8 path is ignored
- repeated kernel/runtime errors
- OOM at the chosen batch size

## 14. Results template

Record:

```text
GPU:
Torch/CUDA image:
Tokenizer source: reused or freshly trained
Attention path: FA3 or SDPA

BF16 avg dt (steps 6-20):
BF16 avg tok/sec (steps 6-20):
BF16 avg MFU (steps 6-20):

FP8 avg dt (steps 6-20):
FP8 avg tok/sec (steps 6-20):
FP8 avg MFU (steps 6-20):

FP8 speedup vs bf16:
Memory observations:
Stability notes:
```

## 15. Optional: copy artifacts back

If you used `/workspace/nanochat-cache`:

```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519 -p PORT" \
  root@HOST:/workspace/nanochat-cache/base_checkpoints/ \
  ~/Downloads/nanot5-throughput-checkpoints/
```

If you used `/root/.cache/nanochat`:

```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519 -p PORT" \
  root@HOST:/root/.cache/nanochat/base_checkpoints/ \
  ~/Downloads/nanot5-throughput-checkpoints/
```

## 16. Destroy the machine

Destroy, do not stop:

```bash
vastai destroy instance INSTANCE_ID
```

Verify:

```bash
vastai show instances
```

## Notes

- This is an end-to-end throughput sanity check, not a quality benchmark.
- One train shard is enough for bf16 vs FP8 comparison.
- Keep all flags identical between the two runs except `--fp8`.
- Record whether the run used FA3 or SDPA; that matters for interpreting results.
- If bf16 is unstable, do not trust FP8 conclusions yet.
