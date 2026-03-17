# Vast.ai 8x RTX 6000 Pro Runbook

## 0. Assumptions

- Local machine has `vastai` installed and authenticated
- SSH key exists at `~/.ssh/id_ed25519`
- Repo is `https://github.com/bigs/nanot5.git` or equivalent SSH remote
- Target box: 8x RTX 6000-class GPUs, 250GB disk
- We want to stop the instance at the end, not destroy it

## 1. Find an offer

Important: `RTX_6000Ada` is NOT the same GPU as the Blackwell `RTX PRO 6000` / `RTX_PRO_6000_S`. For this runbook, we do **not** want the Ada card.

Example searches:

```bash
vastai search offers 'reliability>0.97 num_gpus=8 gpu_name=RTX_PRO_6000_S rented=False disk_space>250' --limit 20 -o 'dph'
```

Broader Blackwell-only search:

```bash
vastai search offers 'num_gpus=8 gpu_name in [RTX_PRO_6000_S,RTX_PRO_6000] rented=False disk_space>250' --limit 20 -o 'dph'
```

Pick an `OFFER_ID`.

## 2. Launch the instance

Check exact CLI options first if needed:

```bash
vastai create instance --help
```

Typical launch:

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
  --disk 250
```

Record the returned `INSTANCE_ID`.

## 3. Wait for the instance to be ready

```bash
vastai show instances
```

Get SSH info:

```bash
vastai ssh-url INSTANCE_ID
```

Or inspect directly:

```bash
vastai show instance INSTANCE_ID
```

## 4. SSH into the machine

Using the host/port from `vastai ssh-url`:

```bash
ssh -i ~/.ssh/id_ed25519 root@HOST -p PORT
```

## 5. Basic machine setup

On the remote box:

```bash
apt-get update
apt-get install -y git curl tmux rsync htop
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

If needed:

```bash
source ~/.bashrc || true
```

## 6. Clone the repo

HTTPS:

```bash
mkdir -p ~/Code
cd ~/Code
git clone https://github.com/bigs/nanot5.git
cd nanot5
```

SSH:

```bash
mkdir -p ~/Code
cd ~/Code
git clone git@github.com:bigs/nanot5.git
cd nanot5
```

## 7. Install Python dependencies

```bash
uv venv
uv sync --extra gpu
source .venv/bin/activate
```

## 8. Verify GPUs

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected: `True 8`

## 9. Set cache / artifact directory

Preferred:

```bash
export NANOCHAT_BASE_DIR=/workspace/nanochat-cache
mkdir -p "$NANOCHAT_BASE_DIR"
```

Fallback:

```bash
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
mkdir -p "$NANOCHAT_BASE_DIR"
```

## 10. Fetch dataset subset

Nanochat-style subset for a real run:

```bash
python -m nanochat.dataset -n 170
```

## 11. Authenticate Weights & Biases (optional but recommended)

If you want training metrics in W&B, authenticate on the GPU VM before launching runs:

```bash
wandb login
```

If you skip this, keep `--run dummy` or use another dummy/local run name.

## 12. Train tokenizer

```bash
python -m scripts.tok_train
```

Optional:

```bash
python -m scripts.tok_eval
```

## 13. First training run (recommended bf16 burn-in)

Run inside `tmux` if desired:

```bash
tmux new -s train
```

Suggested first run:

```bash
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --run=bf16-burnin \
  --model-tag=rtx6000pro-bf16-burnin \
  --depth=24 \
  --target-param-data-ratio=8 \
  --device-batch-size=16 \
  --core-metric-every=-1 \
  --sample-every=-1
```

## 13. Optional FP8 follow-up run

Only after bf16 is stable:

```bash
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --run=fp8-burnin \
  --model-tag=rtx6000pro-fp8-burnin \
  --depth=24 \
  --target-param-data-ratio=8 \
  --device-batch-size=16 \
  --fp8 \
  --core-metric-every=-1 \
  --sample-every=-1
```

## 14. Monitor training

Useful commands:

```bash
watch -n 2 nvidia-smi
```

If using `tmux`, detach with `Ctrl-b d`.

Artifacts live under:

```bash
$NANOCHAT_BASE_DIR
```

## 15. Copy checkpoints back to local machine

From your local machine:

```bash
mkdir -p ~/Downloads/nanot5-checkpoints
scp -r -i ~/.ssh/id_ed25519 -P PORT \
  root@HOST:$NANOCHAT_BASE_DIR/base_checkpoints/rtx6000pro-bf16-burnin \
  ~/Downloads/nanot5-checkpoints/
```

Or copy all base checkpoints:

```bash
scp -r -i ~/.ssh/id_ed25519 -P PORT \
  root@HOST:$NANOCHAT_BASE_DIR/base_checkpoints \
  ~/Downloads/nanot5-checkpoints/
```

Rsync alternative:

```bash
rsync -avz -e "ssh -i ~/.ssh/id_ed25519 -p PORT" \
  root@HOST:$NANOCHAT_BASE_DIR/base_checkpoints/ \
  ~/Downloads/nanot5-checkpoints/
```

## 16. Stop the instance (do not delete)

From local:

```bash
vastai stop instance INSTANCE_ID
```

Verify:

```bash
vastai show instances
```

## 17. Restart later

```bash
vastai start instance INSTANCE_ID
```

Then SSH back in and reuse:

- cached dataset
- tokenizer artifacts
- repo checkout
- checkpoints

## Notes

- Data is fetched once and cached locally; training is not streaming from the internet
- Approx dataset subset size for ~170 shards: ~16GB
- 250GB disk is plenty for dataset subset + tokenizer + checkpoints + logs
- Recommended run order:
  1. bf16 burn-in
  2. fp8 run if stable
