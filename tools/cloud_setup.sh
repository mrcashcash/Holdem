#!/usr/bin/env bash
# Provision a rented single-GPU pod (vast.ai / RunPod) to run this project's
# training, evaluation or serving workloads.
#
# Run ON the pod, from the repo root:
#   bash tools/cloud_setup.sh
#
# It is idempotent and restartable: every step re-checks before doing work, and
# everything is appended to a durable log (logs/cloud-setup.log) rather than
# only stdout, per the project's observability rule.
#
# What it deliberately does NOT do: change any quality dial. It installs the
# SAME torch/numpy/numba versions the local box is validated on, then runs the
# trusted guards. A pod that cannot pass test_gpu_convergence and the null
# duels is not allowed to produce a number.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs
LOG="logs/cloud-setup.log"

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG"; }

log "=== cloud_setup start (repo=$REPO_ROOT) ==="

# ---------------------------------------------------------------- pinned deps
# Matched to the validated local environment. torch cu128 wheels run on any
# driver >= 525, including CUDA 13.x images (driver ABI is backward compatible),
# so the cu128 build is correct even on a CUDA 13.2 base image.
TORCH_SPEC="torch==2.7.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
PINS=(
  "numpy==2.2.6"
  "numba==0.61.2"
  "llvmlite==0.44.0"
  "fastapi==0.116.1"
  "uvicorn==0.35.0"
  "pydantic==2.11.7"
  "websockets==16.1"
  "requests==2.34.2"
)

log "--- host ---"
log "python: $(python -V 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$LOG"
else
  log "WARNING: nvidia-smi not found; this pod may have no GPU"
fi

log "--- dependencies ---"
if python -c "import torch; assert torch.__version__.startswith('2.7.0')" 2>/dev/null; then
  log "torch already at 2.7.0; skipping install"
else
  log "installing $TORCH_SPEC from $TORCH_INDEX"
  pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX" 2>&1 | tee -a "$LOG"
fi

for pin in "${PINS[@]}"; do
  name="${pin%%==*}"
  want="${pin##*==}"
  have="$(python -c "import importlib.metadata as m;print(m.version('$name'))" 2>/dev/null || echo "")"
  if [ "$have" = "$want" ]; then
    log "$name==$want already present"
  else
    log "installing $pin (had '${have:-none}')"
    pip install -q "$pin" 2>&1 | tee -a "$LOG"
  fi
done

log "--- GPU identity as torch sees it ---"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch

print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: torch cannot see a CUDA device")
props = torch.cuda.get_device_properties(0)
total_mib = props.total_memory / 1024**2
free, total = torch.cuda.mem_get_info()
print(f"device 0: {props.name}")
print(f"  capability : sm_{props.major}{props.minor}")
print(f"  SMs        : {props.multi_processor_count}")
print(f"  VRAM total : {total_mib:,.0f} MiB")
print(f"  VRAM free  : {free / 1024**2:,.0f} MiB  (nothing else should be resident)")
PY

log "--- recommended env profile for a HEADLESS card ---"
# These raise the ceilings that exist only because the local 3060 also drives a
# monitor. They are printed, NOT exported into a profile, because the flop node
# budget is a LATENCY guard as much as a memory guard -- raising it must be
# justified by tools/benchmark_resolver_latency.py on this card, not assumed.
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch

total_mib = int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
# Headless: reserve ~700 MiB for driver/context instead of the 2 GiB a desktop
# compositor needs, and allow a higher resident fraction.
budget = int((total_mib - 700) * 0.92)
headroom = 1024
showdown = 384 if total_mib < 14000 else 768
print("# memory ceilings only -- safe to export")
print(f"export HOLDEM_RESOLVER_MAX_VRAM_MB={budget}")
print(f"export HOLDEM_RESOLVER_VRAM_HEADROOM_MB={headroom}")
print(f"export HOLDEM_SHOWDOWN_WORKSPACE_MB={showdown}")
print()
print("# LATENCY guard -- do NOT raise without measuring on this card first:")
print("#   python tools/benchmark_resolver_latency.py")
print("# export HOLDEM_FLOP_NODE_BUDGET=12000   (default; raising it costs seconds/decision)")
PY

log "--- trusted guards (a pod failing these may not produce numbers) ---"
GUARDS=(
  tests.test_gpu_convergence
  tests.test_duel_null
  tests.test_lbr_validation
)
FAILED=0
for guard in "${GUARDS[@]}"; do
  log "running $guard"
  if python -m unittest "$guard" >>"$LOG" 2>&1; then
    log "  PASS $guard"
  else
    log "  FAIL $guard  <-- see $LOG"
    FAILED=1
  fi
done

if [ "$FAILED" -ne 0 ]; then
  log "=== cloud_setup FINISHED WITH FAILING GUARDS ==="
  log "Do not trust any measurement from this pod until the guards pass."
  exit 1
fi

log "=== cloud_setup OK: deps pinned, GPU visible, guards green ==="
log "Full log: $LOG"
