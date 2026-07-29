#!/usr/bin/env bash
# Launch river CFV datagen on a rented multi-GPU box (RunPod / vast.ai).
#
# Sizing comes from measurements in docs/PLAN_V2_STRONGEST_PLAYER.md:
#
#   * this workload is LATENCY-bound (123x above its bandwidth floor), so GPU
#     model barely matters and GPU COUNT is everything;
#   * one worker per GPU. Two workers on ONE device measured a 3x REGRESSION
#     (0.4 vs 1.20 rows/s) because separate CUDA contexts time-slice instead of
#     interleaving. Never oversubscribe a device;
#   * CPU solving is 43x slower per solve, so spare vCPUs are worth ~+40%
#     total. Useful if already paid for, never worth choosing an instance for;
#   * OMP_NUM_THREADS=1 on CPU workers, or each grabs every core and they thrash.
#
# Usage on the pod:
#   bash tools/cloud_datagen.sh 1000000 6 60
#     ^ target rows, GPU workers (= GPU count), CPU workers
#
# Collect afterwards: copy the whole OUT directory back and merge with the local
# one. Shards are worker-tagged (river-w{N}-*.npz) with per-worker manifests, so
# `backend.cfv.river_net.load_shards` merges them with no collisions.

set -euo pipefail

TARGET_ROWS="${1:-1000000}"
GPU_WORKERS="${2:-6}"
CPU_WORKERS="${3:-0}"
OUT="${OUT:-data/river}"
ITERATIONS="${ITERATIONS:-200}"
PER_CELL="${PER_CELL:-200}"

TOTAL_WORKERS=$((GPU_WORKERS + CPU_WORKERS))
mkdir -p "$OUT" logs

echo "=== river CFV datagen ==="
echo "target rows   : $TARGET_ROWS"
echo "GPU workers   : $GPU_WORKERS (one per device)"
echo "CPU workers   : $CPU_WORKERS"
echo "total workers : $TOTAL_WORKERS"
echo "output        : $OUT"

if ! python -c "import torch, numpy, numba" 2>/dev/null; then
  echo "installing dependencies..."
  pip install -q numpy 'numba>=0.61,<0.62' || pip install -q numpy numba
  python -c "import torch" 2>/dev/null || \
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
fi

python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | devices {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

launch() {  # index, device, extra env
  local index="$1" device="$2"
  local log="logs/worker-${index}.out"
  if [ "$device" = "cuda" ]; then
    CUDA_VISIBLE_DEVICES="$3" nohup python tools/generate_river_cfv.py \
      --samples "$TARGET_ROWS" --out "$OUT" --iterations "$ITERATIONS" --emit 0 \
      --per-cell "$PER_CELL" --worker "$index" --workers "$TOTAL_WORKERS" \
      --device cuda --report-every 200 > "$log" 2>&1 &
  else
    # One thread each: otherwise every CPU worker grabs all cores and thrashes.
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
      nohup python tools/generate_river_cfv.py \
      --samples "$TARGET_ROWS" --out "$OUT" --iterations "$ITERATIONS" --emit 0 \
      --per-cell "$PER_CELL" --worker "$index" --workers "$TOTAL_WORKERS" \
      --device cpu --report-every 50 > "$log" 2>&1 &
  fi
}

index=0
for device_id in $(seq 0 $((GPU_WORKERS - 1))); do
  launch "$index" cuda "$device_id"
  index=$((index + 1))
done
for _ in $(seq 1 "$CPU_WORKERS"); do
  [ "$CPU_WORKERS" -eq 0 ] && break
  launch "$index" cpu
  index=$((index + 1))
done

echo
echo "launched $index workers. progress:"
echo "  tail -f $OUT/datagen-w0.log"
echo "  python -c \"import sys;sys.path.insert(0,'.');from pathlib import Path;from backend.cfv.river_net import dataset_rows;print(dataset_rows(Path('$OUT')))\""
echo
echo "rows are flushed to disk every 2 min, so a crash or a pod stop costs"
echo "minutes at most, and relaunching the same command resumes."
