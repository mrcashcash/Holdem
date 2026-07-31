---
name: gpu-box
description: Connect to and run work on the rented Vast.ai GPU box (RTX 4070 Ti SUPER) for this project — SSH, repo sync, provisioning, and the persistence/GPU rules that apply there. Use when asked to run training, evals, datagen or benchmarks "on the cloud/online/remote GPU", to sync local↔remote, or when a local GPU job would otherwise have to queue.
---

# Remote GPU box (Vast.ai)

A second, independent GPU. This matters because the project's **"never overlap GPU
work" rule is per-machine** — so a long job here runs concurrently with a local one
without the multi-process CUDA contention that measured a 3x regression locally.

## 1. Connect

```bash
ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    -o BatchMode=yes -p 31356 root@171.231.24.241
```

`BatchMode=yes` matters: without it a key failure hangs waiting for a password
prompt that a non-interactive shell can never answer.

| item | value |
|---|---|
| public IP | **171.231.24.241** (Static) |
| SSH port | **31356** → container 22 |
| instance ID | 46338962 |
| forwarded port range | 31307–31374 |
| container 8080 | → external **31333** (not 8080) |
| machine copy port | 31400 |
| RTT from local | ~251 ms |

### When the port or IP changes

It already changed once (31203 → 31356, plus a new IP). **Re-derive it, never
assume it.** Read *Open Ports* in the instance's **IP & Port Info** panel and take
the line mapping to `22/tcp`, or:

```bash
vastai show instances     # prints ssh_host / ssh_port
```

Diagnosing a failure to connect:

- **`Connection refused` on every port, but ping succeeds** → the container is not
  running. The IP is the *physical host*, which answers ICMP regardless, so a
  successful ping says nothing about your instance. Start it in the UI.
- **`Permission denied (publickey)`** → the key is not in the instance's
  `authorized_keys`. Attach it:
  ```bash
  vastai attach ssh 46338962 "$(cat ~/.ssh/id_ed25519.pub)"
  ```
  Adding a key account-wide sometimes only affects *new* instances.

## 2. Environment — verified 2026-07-31

| item | value |
|---|---|
| GPU | RTX 4070 Ti SUPER, **15,937 MiB**, 66 SMs, driver 595.71.05 |
| CUDA (driver) | 13.2 — **torch cu128 wheels work on it** (driver ABI is backward compatible) |
| torch | 2.7.0+cu128, matching local |
| python | 3.12.13 at **`/venv/main`** |
| repo | **`/workspace/Holdem`** |
| disk | 32 G total, ~25 G free |
| installer | `uv` present — `uv pip install` is much faster than pip |

### The venv trap

Vast keeps Python in a venv that a **non-interactive shell does not activate**. So
`ssh host 'python -c ...'` silently hits the bare system interpreter, which has no
torch. Always activate first:

```bash
ssh ... 'bash -lc "cd /workspace/Holdem && source /venv/main/bin/activate && python ..."'
```

`tools/cloud_setup.sh` handles this itself (`VENV=${VENV:-/venv/$ACTIVE_VENV}`).

## 3. Persistence — read this before writing any output

`/workspace` is on **overlay storage, NOT a mounted volume** (this instance has no
volume). Therefore:

- **stop/start** → everything survives;
- **recycle or destroy** → **the whole filesystem is wiped**, `/workspace` included.

So **never leave a result only on the box.** Pull anything you care about back to
local as soon as it exists (§4). Long jobs must also checkpoint to disk so an
interruption costs minutes, not hours — same rule as locally.

## 4. Repo sync

The GitHub repo is **public**, which makes the inbound direction credential-free.

**local → box (works, no secret needed):**

```bash
# once per fresh box: the checked-out remote may be the SSH form, which has no key
ssh ... 'cd /workspace/Holdem && git remote set-url origin https://github.com/mrcashcash/Holdem.git'
# then, after pushing from local:
ssh ... 'cd /workspace/Holdem && GIT_TERMINAL_PROMPT=0 git pull --ff-only origin master'
```

**box → local: use `scp`, not `git push`.** The box has **no** GitHub credential
(only `authorized_keys` for inbound SSH), so `git push` fails with
`could not read Username`. Don't fix that by storing a PAT on the box: it is
rented hardware whose filesystem is wiped on recycle. Pull results down instead —
initiated from local, which already holds the key:

```bash
scp -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -P 31356 \
    root@171.231.24.241:/workspace/Holdem/backend/data/evaluations/RESULT.json ./
```

`GIT_TERMINAL_PROMPT=0` is worth keeping on every remote git call — otherwise an
auth failure blocks on a username prompt instead of erroring.

## 5. Provisioning a fresh box

```bash
ssh ... 'cd /workspace/Holdem && mkdir -p logs && \
  setsid nohup bash tools/cloud_setup.sh > logs/cloud-setup.out 2>&1 < /dev/null &'
```

Three things that bite here:

1. **`mkdir -p logs` first.** `logs/` is gitignored, so it does not exist on a
   fresh clone, and the redirect fails *before* the script runs — leaving no log
   and no job. Verify a launch by watching output GROW, never by exit status.
2. **`setsid nohup … < /dev/null &`** so the job survives the SSH session.
3. The script **refuses to bless the box** unless `test_gpu_convergence`,
   `test_duel_null` and `test_lbr_validation` all pass. Do not trust a number from
   a box that has not printed `cloud_setup OK`. It caught a real gap on first run:
   `python-dotenv` was missing, which made the promotion gate uncollectable.

Verified green on this box 2026-07-31: all three guards PASS.

Deliberately **not** installed (Windows/GUI-only, irrelevant headless): `cv2`,
`mss`, `rapidocr`, `resvg_py`, `tkinter`, `windows_capture`.

## 6. Headless VRAM ceilings

The local resolver limits exist only because the 3060 also drives a monitor. This
card is headless with 16 GB, so the memory ceilings can be raised **by environment
variable, with no code change**:

```bash
export HOLDEM_RESOLVER_MAX_VRAM_MB=14017     # printed by cloud_setup.sh on this card
export HOLDEM_RESOLVER_VRAM_HEADROOM_MB=1024
export HOLDEM_SHOWDOWN_WORKSPACE_MB=768
```

**Do not raise `HOLDEM_FLOP_NODE_BUDGET`.** It is a *latency* guard as much as a
memory one — more nodes means seconds more per decision. Raising it requires
measuring `tools/benchmark_resolver_latency.py` on this card first.

## 7. What this box is good for

Good fits — long, GPU-bound, resumable, few round trips:

- LBR re-measurement at scale (`backend/eval/lbr.py`, 20k pairs ≈ 50 min/depth)
- paired A/B gates (`tools/lbr_guard_gate.py`)
- blueprint training / CFV datagen (resumable by design)
- resolver work that benefits from the headless 16 GB ceiling

Poor fits:

- **Anything latency-sensitive or chatty** — 251 ms RTT.
- **Serving**, if the instance is interruptible (spot pricing): preemption lands
  mid-hand. Confirm the pricing model before serving from here.
- **Multi-worker GPU datagen.** `tools/cloud_datagen.sh` is written for a 6-GPU
  box because that workload is latency-bound and GPU *count* is what matters. This
  box has **one** GPU, and two workers on one device measured a **3x regression**.

## 8. The instance's own agent guide

The SSH banner points at **`/etc/vast-agents-guide.md`** (also `./AGENTS.md`) and
asks agents to read it before acting. Read it — it is the authority on this image,
not this file. Highlights: it is an unprivileged container (no Docker-in-Docker, no
kernel modules, no `perf`/eBPF); long-running services belong under **supervisor**;
web apps are fronted by **Caddy** with TLS and token auth. `vast-capabilities`
prints live state as JSON (also `/etc/vast_capabilities.json`, or
`curl -s localhost:11111/capabilities`).
