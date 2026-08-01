#!/usr/bin/env bash
# Provision a FRESH rented GPU pod (vast.ai / RunPod) from scratch.
#
# Bootstraps from nothing -- it clones the repo itself, so it can run before the
# repo exists on the pod:
#   curl -fsSL https://raw.githubusercontent.com/mrcashcash/Holdem/master/tools/cloud_setup.sh | bash
# or, if the file is already on the pod:
#   bash cloud_setup.sh
#
#   apt update/upgrade -> system packages -> clone repo -> python deps
#   -> GPU operational check -> 60s GPU workload
#
# Env overrides:
#   HOLDEM_DIR      clone target      (default /workspace/Holdem, else ~/Holdem)
#   BRANCH          branch to clone   (default master)
#   SMOKE_SECONDS   length of the closing GPU workload (default 60)
#   GITHUB_TOKEN    fine-grained PAT with Contents:read+write. REQUIRED to push
#                   results back off an ephemeral pod. Set it once on your Vast
#                   account and every future instance inherits it:
#                     vastai create env-var GITHUB_TOKEN github_pat_xxx
#   GIT_USER_NAME / GIT_USER_EMAIL   commit identity (defaults below)

set -euo pipefail

REPO_URL="https://github.com/mrcashcash/Holdem"
BRANCH="${BRANCH:-master}"
SMOKE_SECONDS="${SMOKE_SECONDS:-60}"

LOG="${HOME:-/root}/cloud-setup.log"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n=== %s %s\n' "$(date -u '+%H:%M:%SZ')" "$*"; }

say "cloud_setup start"

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ------------------------------------------------------------------- system
say "apt update / upgrade"
export DEBIAN_FRONTEND=noninteractive

# Pin the container's GPU userspace first. The driver is injected by the host,
# so an upgrade that swaps libnvidia/cuda out from under it leaves torch unable
# to see the card -- on a rented pod that is a paid-for brick.
held="$(dpkg-query -W -f='${Package}\n' 2>/dev/null | grep -E 'nvidia|libcuda|^cuda' || true)"
[ -n "$held" ] && $SUDO apt-mark hold $held >/dev/null 2>&1 || true

$SUDO apt-get update -qq

# The full upgrade is OPT-IN. Measured 2026-08-01: it spent 3-5 min upgrading ~18
# packages the job never touches (vim, rsyslog, libxml2), and because the set
# included openssh-server it RESTARTED sshd and killed the live SSH session
# mid-provision -- which looks exactly like a dead pod. The base image is already
# current enough to build and run this project. Set FULL_UPGRADE=1 if a CVE or a
# broken system package actually requires it.
if [ "${FULL_UPGRADE:-0}" = "1" ]; then
  say "apt full upgrade (FULL_UPGRADE=1; may restart sshd and drop your session)"
  $SUDO apt-get -y -qq -o Dpkg::Options::=--force-confold upgrade
else
  echo "skipping full apt upgrade (set FULL_UPGRADE=1 to force)"
fi

say "system packages"
$SUDO apt-get install -y -qq \
  git curl ca-certificates build-essential \
  python3 python3-dev python3-pip python3-venv \
  tmux

# --------------------------------------------------------------------- repo
if [ -z "${HOLDEM_DIR:-}" ]; then
  if [ -d /workspace ]; then HOLDEM_DIR=/workspace/Holdem; else HOLDEM_DIR="$HOME/Holdem"; fi
fi

say "git identity and push credentials"
git config --global user.name "${GIT_USER_NAME:-holdem-pod}"
git config --global user.email "${GIT_USER_EMAIL:-pod@holdem.local}"
git config --global safe.directory '*'   # repo dir is often owned by another uid

# Vast injects account env-vars (`vastai create env-var`) into the CONTAINER's
# environment -- i.e. PID 1 -- and sshd does NOT pass them into ssh sessions.
# Verified 2026-08-01 on vastai/pytorch: GITHUB_TOKEN is present in
# /proc/1/environ but empty in `ssh host '...'`, in `bash -l`, and in every
# profile file. So `ssh pod 'bash cloud_setup.sh'` sees no token unless we go
# get it. Recover it from PID 1 rather than making the user re-export by hand.
if [ -z "${GITHUB_TOKEN:-}" ] && [ -r /proc/1/environ ]; then
  GITHUB_TOKEN="$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^GITHUB_TOKEN=//p' | head -1)"
  export GITHUB_TOKEN
  [ -n "$GITHUB_TOKEN" ] && echo "recovered GITHUB_TOKEN from the container environment"
fi

if [ -n "${GITHUB_TOKEN:-}" ]; then
  # Read the token from the environment at each git call instead of baking it
  # into the remote URL or ~/.git-credentials -- the secret then never touches
  # the pod's disk, where the host operator could read it after you disconnect.
  git config --global credential.helper \
    '!f() { if [ "$1" = get ]; then printf "username=x-access-token\npassword=%s\n" "$GITHUB_TOKEN"; fi; }; f'
  echo "GITHUB_TOKEN present -> push enabled"
else
  echo "WARNING: no GITHUB_TOKEN -- this pod can pull but CANNOT push."
  echo "         Nothing it produces will survive the instance being destroyed."
  echo "         Fix: vastai create env-var GITHUB_TOKEN github_pat_xxx"
fi

if [ -d "$HOLDEM_DIR/.git" ]; then
  say "repo already at $HOLDEM_DIR; fast-forwarding $BRANCH"
  git -C "$HOLDEM_DIR" pull --ff-only origin "$BRANCH"
else
  # NOT --depth 1: pushing from a shallow clone is fragile, and on a pod with no
  # persistent disk, pushing is the only way results survive.
  #
  # NOT --filter either, by default. MEASURED 2026-08-01 on a 3090 pod:
  # --filter=blob:limit=1m saved 193MB -> 156MB (19%) but the clone took
  # **26 minutes at ~1 Mbps**, while torch pulled 2.5GB in 11 min (~30 Mbps) over
  # the same link. The link is fine; a filtered clone makes GitHub compute a
  # custom pack server-side, off the cached path, and that is what crawls. The
  # same filter then made every PUSH re-send ~77MB and take ~11 min even for a
  # one-line file, because thin-pack negotiation against a promisor remote cannot
  # tell what the remote already has. A plain clone hits GitHub's cached pack and
  # pushes normally. Set FILTER_BLOBS=1 to restore the old behaviour on a pod
  # where disk, not time, is the binding constraint.
  FILTER_ARGS=""
  [ "${FILTER_BLOBS:-0}" = "1" ] && FILTER_ARGS="--filter=blob:limit=1m"

  say "cloning into $HOLDEM_DIR${FILTER_ARGS:+ (filtered)}"
  if [ "${SPARSE_NO_DATA:-0}" = "1" ]; then
    # Training from scratch and not evaluating against existing blueprints:
    # skips backend/data entirely. Note --sampler-init needs a champion.npz, so
    # do not combine this with a histogram run unless you fetch one separately.
    git clone $FILTER_ARGS --no-checkout --branch "$BRANCH" "$REPO_URL" "$HOLDEM_DIR"
    git -C "$HOLDEM_DIR" sparse-checkout set --no-cone '/*' '!/backend/data/**'
    git -C "$HOLDEM_DIR" checkout "$BRANCH"
  else
    git clone $FILTER_ARGS --branch "$BRANCH" "$REPO_URL" "$HOLDEM_DIR"
  fi
fi
cd "$HOLDEM_DIR"

# ------------------------------------------------------------- python deps
# Vast.ai images keep Python in a venv (default /venv/main) that a
# NON-INTERACTIVE shell does not activate -- `ssh host 'python ...'` would
# otherwise hit the bare system interpreter and install into the wrong place.
VENV="${VENV:-/venv/${ACTIVE_VENV:-main}}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
else
  [ -d .venv ] || python3 -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
say "python: $(python -V 2>&1) at $(command -v python)"

PIP="pip"
command -v uv >/dev/null 2>&1 && PIP="uv pip"

# Versions matched to the validated local environment. The torch cu128 wheels
# run on any driver >= 525, including CUDA 13.x images (driver ABI is backward
# compatible), so cu128 is correct even on a CUDA 13.2 base image.
#
# Deliberately NOT installed: opencv, mss, rapidocr, resvg_py, tkinter,
# windows_capture. Those serve the Windows-only screen-scraper and GUI overlay;
# every solver, eval and serving import works without them.
say "python dependencies"

# Skip the ~2.5GB torch download when the image already satisfies the pin. The
# vastai/pytorch images usually ship a recent torch, and reinstalling cost 11 min
# of measured provision time on 2026-08-01 for no change. Match on the version
# prefix AND on CUDA support, since a CPU-only wheel of the right version would
# pass a naive check and then fail at the first kernel launch.
if python - <<'PY' 2>/dev/null
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.__version__.startswith("2.7.0") and torch.version.cuda else 1)
PY
then
  echo "torch already satisfies the pin ($(python -c 'import torch;print(torch.__version__)')); skipping download"
else
  $PIP install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
fi

$PIP install -q \
  numpy==2.2.6 numba==0.61.2 llvmlite==0.44.0 \
  fastapi==0.116.1 uvicorn==0.35.0 pydantic==2.11.7 \
  websockets==16.1 requests==2.34.2 python-dotenv==1.2.2

# ---------------------------------------------------------------------- GPU
say "GPU"
if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
  echo "FATAL: nvidia-smi unavailable -- this pod has no usable GPU"
  exit 1
fi

say "${SMOKE_SECONDS}s GPU workload"
SMOKE_SECONDS="$SMOKE_SECONDS" python - <<'PY'
import os
import time

import torch

if not torch.cuda.is_available():
    raise SystemExit("FATAL: torch cannot see a CUDA device")

props = torch.cuda.get_device_properties(0)
print(
    f"torch {torch.__version__} (CUDA {torch.version.cuda}) -> {props.name} "
    f"sm_{props.major}{props.minor}, {props.multi_processor_count} SMs, "
    f"{props.total_memory / 1024**2:,.0f} MiB"
)

# A card that is visible but computing wrong is worse than one that is absent.
probe = torch.randn(2048, 2048, device="cuda")
torch.testing.assert_close(
    probe @ torch.eye(2048, device="cuda"), probe, rtol=1e-4, atol=1e-4
)
print("matmul sanity: OK")

from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

tree = BettingTree(GpuActionConfig(max_raises_per_street=2, stack_bb=20.0))
solver = VectorCFR(tree, DealSampler(), device="cuda", seed=0, batch_boards=4)
print(f"tree: {len(tree.kind):,} nodes")

solver.run(2)  # warm the kernels outside the clock
budget = float(os.environ.get("SMOKE_SECONDS", "60"))
start, iters = time.time(), 0
while time.time() - start < budget:
    solver.run(10)
    iters += 10
elapsed = time.time() - start

print(f"{iters} CFR iterations in {elapsed:.1f}s -> {iters / elapsed:.1f} it/s")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:,.0f} MiB")
PY

say "READY: repo at $HOLDEM_DIR, log at $LOG"
