#!/usr/bin/env bash
# Get results OFF an ephemeral pod before the instance is destroyed.
#
# A rented pod has no persistent disk: when you `vastai destroy instance`, every
# log, checkpoint and metric on it is gone. This commits the run's output and
# pushes it to a per-pod branch on GitHub.
#
#   bash tools/cloud_push.sh "20bb menu sweep, 40k iters"
#
# It pushes to a POD BRANCH (pod/<id>-<date>), never straight to master, so a
# pod and your local work can never reject each other's pushes. Merge locally
# when you have looked at what came back.
#
# logs/ and artifacts/ are gitignored on purpose -- they are force-added here,
# because on a pod they are the only reason the run happened.
#
# Env overrides:
#   PUSH_BRANCH           override the generated branch name
#   INCLUDE_CHECKPOINTS   1 to also push *.npz blueprints (see the warning below)

set -euo pipefail

# Locate the clone. Prefer the caller's cwd, then this script's parent. Resolving
# only script-relative means a copy parked outside the repo (say /root) silently
# cd's to / and every git command then fails with a baffling "not a git
# repository" -- fail with a usable message instead.
if repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  cd "$repo_root"
elif [ -d "$(dirname "${BASH_SOURCE[0]}")/../.git" ]; then
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  echo "FATAL: not inside the Holdem clone. cd into it first, or keep this"
  echo "       script at <repo>/tools/cloud_push.sh."
  exit 1
fi

MESSAGE="${1:-pod run}"
POD_ID="${CONTAINER_ID:-${VAST_CONTAINERLABEL:-$(hostname)}}"
BRANCH="${PUSH_BRANCH:-pod/${POD_ID}-$(date -u +%m%d-%H%M)}"

# Vast puts account env-vars in the CONTAINER environment (PID 1) and sshd does
# not forward them into ssh sessions -- see the long note in cloud_setup.sh.
# Without this, a push driven over ssh fails even though the token is set.
if [ -z "${GITHUB_TOKEN:-}" ] && [ -r /proc/1/environ ]; then
  GITHUB_TOKEN="$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^GITHUB_TOKEN=//p' | head -1)"
  export GITHUB_TOKEN
fi

# The credential helper installed by cloud_setup.sh reads $GITHUB_TOKEN at each
# git call, so it must be exported in THIS process too, not just at setup time.
git config --global credential.helper \
  '!f() { if [ "$1" = get ]; then printf "username=x-access-token\npassword=%s\n" "$GITHUB_TOKEN"; fi; }; f'

# Fail before doing work if the push cannot possibly succeed -- discovering this
# after a 6-hour run, with the results still only on the pod, is the bad outcome.
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "FATAL: GITHUB_TOKEN is not set; this pod cannot push."
  echo "  On your workstation:  vastai create env-var GITHUB_TOKEN github_pat_xxx"
  echo "  Then, without destroying this pod, export it here and re-run:"
  echo "    export GITHUB_TOKEN=github_pat_xxx"
  exit 1
fi

echo "=== staging results onto $BRANCH"
git checkout -B "$BRANCH"   # -B not -b: a same-minute re-run must not hard-fail

git add -A                                    # tracked code/config changes
git add -f logs/ 2>/dev/null || true          # normally ignored; the run's record
git add -f artifacts/ 2>/dev/null || true

if [ "${INCLUDE_CHECKPOINTS:-0}" = "1" ]; then
  git add -f 'backend/data/**/*.npz' 2>/dev/null || true
else
  # Every .npz is 18-33MB of incompressible binary that git keeps in history
  # FOREVER, and this repo's .git is already ~190MB. Opt in deliberately.
  git reset -q -- '*.npz' 2>/dev/null || true
  echo "note: *.npz excluded (set INCLUDE_CHECKPOINTS=1 to push blueprints)"
fi

if git diff --cached --quiet; then
  echo "nothing to push -- no changes, no logs, no artifacts."
  exit 0
fi

# GitHub hard-rejects any single file over 100MB, and the rejection kills the
# whole push, so catch it here rather than after uploading everything else.
oversized=0
while IFS= read -r f; do
  [ -f "$f" ] || continue
  bytes="$(stat -c %s "$f" 2>/dev/null || echo 0)"
  if [ "$bytes" -gt 94371840 ]; then
    echo "FATAL: $f is $((bytes / 1048576))MB; GitHub rejects files over 100MB."
    oversized=1
  fi
done < <(git diff --cached --name-only)
[ "$oversized" -eq 0 ] || { echo "Unstage it or move it to object storage."; exit 1; }

echo "--- staged ---"
git diff --cached --stat | tail -20

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "no gpu")"
git commit -q -m "$MESSAGE

Pod: $POD_ID on $gpu
Pushed from an ephemeral instance by tools/cloud_push.sh."

echo "=== pushing $BRANCH"
git push -u origin "$BRANCH"

echo
echo "=== SAFE TO DESTROY THE POD"
echo "Branch: $BRANCH"
echo "Locally:  git fetch origin && git log --stat origin/$BRANCH"
