#!/usr/bin/env bash
set -euo pipefail

SSH_PORT="${SSH_PORT:-11494}"
SSH_HOST="${SSH_HOST:-ssh9.vast.ai}"
SSH_USER="${SSH_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/nanot5}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$REMOTE_DIR/dev/b200_fp8_impl_logs}"

ssh -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH=/root/.local/bin:$PATH
cd /root/nanot5
mkdir -p dev/b200_fp8_impl_logs

run_cfg() {
  local name="$1"
  shift 1
  local log="dev/b200_fp8_impl_logs/${name}.log"
  echo "=== RUN ${name} (extra=$*) ==="
  set +e
  CUDA_VISIBLE_DEVICES=0 /root/.local/bin/uv run python - "fa4" "$name" "$@" <<'PY' 2>&1 | tee "$log"
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
    "--fp8",
] + extra
runpy.run_module("scripts.base_train", run_name="__main__")
PY
  rc=${PIPESTATUS[0]}
  set -e
  echo "EXIT_CODE=${rc}" | tee -a "$log"
}

run_cfg fp8_default
run_cfg fp8_traceable --no-fp8-opaque-autograd
run_cfg fp8_grad_input_fastaccum --fp8-grad-input-fast-accum
run_cfg fp8_grad_weight_fastaccum --fp8-grad-weight-fast-accum
run_cfg fp8_bwd_fastaccum --fp8-grad-input-fast-accum --fp8-grad-weight-fast-accum
run_cfg fp8_traceable_bwd_fastaccum --no-fp8-opaque-autograd --fp8-grad-input-fast-accum --fp8-grad-weight-fast-accum

python3 - <<'PY'
import json
import pathlib
import re
import statistics as stats

log_dir = pathlib.Path('dev/b200_fp8_impl_logs')
step_re = re.compile(r"step\s+(\d+)/(\d+).*?tok/sec:\s*([0-9,]+).*?bf16_mfu:\s*([0-9.]+)")
peak_re = re.compile(r"Peak memory usage:\s*([0-9.]+)MiB")
exit_re = re.compile(r"EXIT_CODE=(\d+)")

results = []
for path in sorted(log_dir.glob('fp8_*.log')):
    text = path.read_text()
    steps = [
        {
            'step': int(m.group(1)),
            'tok_per_sec': int(m.group(3).replace(',', '')),
            'mfu': float(m.group(4)),
        }
        for m in step_re.finditer(text)
    ]
    steady = [s for s in steps if s['step'] >= 1]
    tail5 = [s for s in steps if s['step'] >= 5]
    peak = peak_re.search(text)
    exit_match = exit_re.search(text)
    results.append({
        'name': path.stem,
        'exit_code': int(exit_match.group(1)) if exit_match else None,
        'steady_tok_per_sec_mean': round(stats.mean([s['tok_per_sec'] for s in steady]), 1) if steady else None,
        'tail5_tok_per_sec_mean': round(stats.mean([s['tok_per_sec'] for s in tail5]), 1) if tail5 else None,
        'steady_mfu_mean': round(stats.mean([s['mfu'] for s in steady]), 2) if steady else None,
        'tail5_mfu_mean': round(stats.mean([s['mfu'] for s in tail5]), 2) if tail5 else None,
        'peak_mem_mib': float(peak.group(1)) if peak else None,
        'warmup_tok_per_sec': steps[0]['tok_per_sec'] if steps else None,
    })

summary_path = log_dir / 'summary.json'
summary_path.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print(f'Wrote {summary_path}')
PY
REMOTE
