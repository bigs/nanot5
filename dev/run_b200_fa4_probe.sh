#!/usr/bin/env bash
set -euo pipefail

OFFER_ID="${1:-31929962}"
INSTANCE_ID=""
WORKDIR_REMOTE="/workspace"
REMOTE_REPO_DIR="$WORKDIR_REMOTE/nanot5"
REMOTE_BASE_DIR="$WORKDIR_REMOTE/nanochat-cache"
LOGDIR_LOCAL="/home/ubuntu/Code/nanot5/dev/b200-probe-logs"
mkdir -p "$LOGDIR_LOCAL"
RUNSTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY_PATH="$LOGDIR_LOCAL/summary-$RUNSTAMP.json"
TOKENIZER_LOCAL="/home/ubuntu/.cache/nanochat/tokenizer/tokenizer.model"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

cleanup() {
  local exit_code=$?
  if [[ -n "$INSTANCE_ID" ]]; then
    echo "[cleanup] destroying Vast instance $INSTANCE_ID"
    vastai destroy instance "$INSTANCE_ID" >/dev/null 2>&1 || true
  fi
  echo "[cleanup] exit_code=$exit_code"
}
trap cleanup EXIT

wait_for_running() {
  for _ in $(seq 1 120); do
    local line
    line="$(vastai show instances | awk -v id="$INSTANCE_ID" '$1==id {print}')"
    echo "status: ${line:-missing}"
    if [[ -n "$line" ]] && grep -qi 'running' <<<"$line"; then
      return 0
    fi
    sleep 5
  done
  return 1
}

wait_for_ssh() {
  local host="$1"
  local port="$2"
  for _ in $(seq 1 120); do
    if ssh "${SSH_OPTS[@]}" -p "$port" "$host" 'echo ssh_ready' >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "instance did not accept ssh in time" >&2
  return 1
}

echo "[1/8] creating 1x B200 instance from offer $OFFER_ID"
CREATE_OUT="$(vastai create instance "$OFFER_ID" --image nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 --disk 80 --ssh --direct 2>&1)"
echo "$CREATE_OUT"
INSTANCE_ID="$(printf '%s\n' "$CREATE_OUT" | grep -Eo 'New instance [0-9]+' | awk '{print $3}' | tail -n1 || true)"
if [[ -z "$INSTANCE_ID" ]]; then
  INSTANCE_ID="$(printf '%s\n' "$CREATE_OUT" | grep -Eo "'new_contract': [0-9]+" | awk '{print $2}' | tail -n1 || true)"
fi
[[ -n "$INSTANCE_ID" ]] || { echo "failed to parse instance id"; exit 1; }
echo "instance_id=$INSTANCE_ID"

echo "[2/8] waiting for instance to become reachable"
wait_for_running
SSH_URL="$(vastai ssh-url "$INSTANCE_ID" | tail -n1)"
SSH_HOST="$(printf '%s' "$SSH_URL" | sed -E 's#^ssh://##; s#:[0-9]+$##')"
SSH_PORT="$(printf '%s' "$SSH_URL" | sed -En 's#.*:([0-9]+)$#\1#p')"
[[ -n "$SSH_HOST" && -n "$SSH_PORT" ]] || { echo "failed to resolve ssh host/port from: $SSH_URL"; exit 1; }
echo "ssh_host=$SSH_HOST"
echo "ssh_port=$SSH_PORT"
wait_for_ssh "$SSH_HOST" "$SSH_PORT"

echo "[3/8] preparing remote environment"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" "bash -s" <<EOF
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git curl build-essential python3 python3-venv python3-pip openssh-client
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
mkdir -p "$WORKDIR_REMOTE" "$REMOTE_BASE_DIR/tokenizer"
cd "$WORKDIR_REMOTE"
if [[ ! -d nanot5/.git ]]; then
  git clone https://github.com/bigs/nanot5.git
fi
cd nanot5
git fetch origin feat/t5
git checkout feat/t5
git reset --hard origin/feat/t5
uv sync --extra gpu --extra fa4
. .venv/bin/activate
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('capability', torch.cuda.get_device_capability())
print('device', torch.cuda.get_device_name(0))
PY
EOF

echo "[4/8] copying tokenizer"
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$TOKENIZER_LOCAL" "$SSH_HOST:$REMOTE_BASE_DIR/tokenizer/tokenizer.model"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" "bash -s" <<EOF
set -euo pipefail
cd "$REMOTE_REPO_DIR"
export PATH=/root/.local/bin:$PATH
uv run python - <<'PY'
import os
import torch
from nanochat.tokenizer import SimpleTokenizer, build_token_bytes

tokenizer_dir = os.path.join("$REMOTE_BASE_DIR", "tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer = SimpleTokenizer.from_directory(tokenizer_dir)
token_bytes = build_token_bytes(tokenizer, device="cpu").cpu()
torch.save(token_bytes, os.path.join(tokenizer_dir, "token_bytes.pt"))
print(f"wrote {tokenizer_dir}/token_bytes.pt")
PY
EOF

echo "[5/8] downloading minimal dataset shards"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" "bash -s" <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "$REMOTE_REPO_DIR"
. .venv/bin/activate
export NANOCHAT_BASE_DIR="$REMOTE_BASE_DIR"
python -m nanochat.dataset -n 2 -w 4
EOF

echo "[6/8] running throughput probes"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" "bash -s" <<EOF | tee "$SUMMARY_PATH"
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "$REMOTE_REPO_DIR"
. .venv/bin/activate
export NANOCHAT_BASE_DIR="$REMOTE_BASE_DIR"
mkdir -p "$WORKDIR_REMOTE/probe-logs"

run_probe() {
  local name="\$1"; shift
  echo "===== \$name ====="
  python -m scripts.base_train --run=dummy --device-type=cuda --num-iterations=20 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 "\$@" 2>&1 | tee "$WORKDIR_REMOTE/probe-logs/\${name}.log"
}

run_probe d24_bf16 --depth=24 --max-seq-len=2048 --device-batch-size=1
run_probe d24_fp8  --depth=24 --max-seq-len=2048 --device-batch-size=1 --fp8
run_probe d13_bf16 --depth=13 --max-seq-len=2048 --device-batch-size=2
run_probe d13_fp8  --depth=13 --max-seq-len=2048 --device-batch-size=2 --fp8

python - <<'PY'
import json, os, re, statistics
logdir = '/workspace/probe-logs'
summary = {}
step_re = re.compile(r'tok/sec: ([0-9,]+).*?bf16_mfu: ([0-9.]+)')
mem_re = re.compile(r'Peak memory usage: ([0-9.]+)MiB')
for fn in sorted(os.listdir(logdir)):
    if not fn.endswith('.log'):
        continue
    text = open(os.path.join(logdir, fn)).read()
    vals = [(int(t.replace(',', '')), float(m)) for t, m in step_re.findall(text)]
    steady = vals[10:] if len(vals) > 10 else vals
    toks = [v[0] for v in steady]
    mfus = [v[1] for v in steady]
    mem = mem_re.findall(text)
    summary[fn] = {
        'samples': len(steady),
        'tok_per_sec_mean': round(statistics.mean(toks), 2) if toks else None,
        'tok_per_sec_max': max(toks) if toks else None,
        'mfu_mean': round(statistics.mean(mfus), 2) if mfus else None,
        'peak_mem_mib': float(mem[-1]) if mem else None,
    }
print(json.dumps(summary, indent=2, sort_keys=True))
PY
EOF

echo "[7/8] retrieving raw logs"
mkdir -p "$LOGDIR_LOCAL/$RUNSTAMP"
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$SSH_HOST:$WORKDIR_REMOTE/probe-logs/d24_bf16.log" "$LOGDIR_LOCAL/$RUNSTAMP/"
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$SSH_HOST:$WORKDIR_REMOTE/probe-logs/d24_fp8.log" "$LOGDIR_LOCAL/$RUNSTAMP/"
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$SSH_HOST:$WORKDIR_REMOTE/probe-logs/d13_bf16.log" "$LOGDIR_LOCAL/$RUNSTAMP/"
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$SSH_HOST:$WORKDIR_REMOTE/probe-logs/d13_fp8.log" "$LOGDIR_LOCAL/$RUNSTAMP/"

echo "[8/8] finished; summary saved to $SUMMARY_PATH"
cat "$SUMMARY_PATH"
