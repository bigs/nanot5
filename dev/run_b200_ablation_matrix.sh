#!/usr/bin/env bash
set -euo pipefail

SSH_PORT="${SSH_PORT:-11494}"
SSH_HOST="${SSH_HOST:-ssh9.vast.ai}"
SSH_USER="${SSH_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/nanot5}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_DIR/dev/b200_ablation_logs}"

ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH=/root/.local/bin:$PATH
cd /root/nanot5
mkdir -p dev/b200_ablation_logs

run_cfg() {
  local name="$1"
  local backend="$2"
  shift 2
  local log="dev/b200_ablation_logs/${name}.log"
  echo "=== RUN ${name} (backend=${backend}; extra=$*) ==="
  CUDA_VISIBLE_DEVICES=0 /root/.local/bin/uv run python - "$backend" "$name" "$@" <<'PY' 2>&1 | tee "$log"
import runpy
import sys

backend = sys.argv[1]
name = sys.argv[2]
extra = sys.argv[3:]

import nanochat.flash_attention as fa
fa._override_impl = backend
fa._refresh_fast_attention_state()

sys.argv = [
    "base_train.py",
    "--run", "dummy",
    "--device-type", "cuda",
    "--num-iterations", "10",
    "--eval-every", "-1",
    "--sample-every", "-1",
    "--save-every", "-1",
    "--core-metric-every", "-1",
    "--device-batch-size", "4",
    "--total-batch-size", "32768",
    "--model-tag", name,
] + extra
runpy.run_module("scripts.base_train", run_name="__main__")
PY
}

run_cfg b200_bf16_sdpa sdpa
run_cfg b200_bf16_fa4  fa4
run_cfg b200_fp8_sdpa  sdpa --fp8
run_cfg b200_fp8_fa4   fa4 --fp8

python3 - <<'PY'
import json
import pathlib
import re
import statistics as stats

log_dir = pathlib.Path('dev/b200_ablation_logs')
step_re = re.compile(r"step\s+(\d+)/(\d+).*?tok/sec:\s*([0-9,]+).*?bf16_mfu:\s*([0-9.]+)")
peak_re = re.compile(r"Peak memory usage:\s*([0-9.]+)MiB")
backend_re = re.compile(r"(WARNING: No compatible Flash Attention backend available|Flash Attention 4 available \(Blackwell optimized\))")

results = []
for path in sorted(log_dir.glob('b200_*.log')):
    text = path.read_text()
    steps = []
    for m in step_re.finditer(text):
        steps.append({
            'step': int(m.group(1)),
            'tok_per_sec': int(m.group(3).replace(',', '')),
            'mfu': float(m.group(4)),
        })
    steady = [s for s in steps if s['step'] >= 1]
    peak = peak_re.search(text)
    backend = backend_re.search(text)
    results.append({
        'name': path.stem,
        'backend_status': backend.group(1) if backend else None,
        'steady_tok_per_sec_mean': round(stats.mean([s['tok_per_sec'] for s in steady]), 1) if steady else None,
        'steady_tok_per_sec_median': round(stats.median([s['tok_per_sec'] for s in steady]), 1) if steady else None,
        'steady_mfu_mean': round(stats.mean([s['mfu'] for s in steady]), 2) if steady else None,
        'steady_steps': len(steady),
        'peak_mem_mib': float(peak.group(1)) if peak else None,
        'raw_steps': steps,
    })

summary_path = log_dir / 'summary.json'
summary_path.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print(f"Wrote {summary_path}")
PY
REMOTE
