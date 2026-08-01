---
name: vast-gpu-jobs
description: Run this project's training, solver, eval, benchmark or test workloads on a rented Vast.ai GPU pod end to end — search an offer, launch, provision with tools/cloud_setup.sh, run the job, push results to GitHub with tools/cloud_push.sh, verify the push landed, destroy the instance. Use for any prompt about running Holdem work on a cloud/rented GPU, "train on vast", "run this on a cloud GPU", "spin up a pod", "run the 20bb/50bb/100bb/200bb sweep remotely", or getting results off a pod. Vast pods have NO persistent disk — this skill exists because unpushed results are lost on destroy.
---

# Running Holdem jobs on a Vast.ai pod

## The one rule

**A Vast pod has no persistent disk. `vastai destroy` erases everything on it.**
A run that produced numbers but did not push them produced nothing. Treat "results
are on GitHub and I verified it" as the definition of a finished job — not "the
job exited 0".

Two independent ways this silently fails, both of which have actually happened here:

1. The output path is in `.gitignore`, so `git add` skips it and the commit is empty.
2. The push fails auth, and the failure is noticed after the pod is gone.

Steps 5–6 below exist entirely to close those two holes. Do not skip them.

Use the `vastai` skill for CLI syntax and instance-state semantics. This skill is
the project-specific procedure layered on top of it.

## One-time setup

```bash
vastai show ssh-keys --raw                  # local ~/.ssh/id_ed25519.pub must be listed
vastai create env-var GITHUB_TOKEN github_pat_xxx
```

The PAT must be fine-grained, scoped to `mrcashcash/Holdem` only, with
**Contents: Read and write**. Nothing else is needed.

**Set the env-var BEFORE launching.** Vast injects account env-vars at instance
*creation*; a pod already running when you add one never sees it. Verify with
`vastai show env-vars --raw` (values are always masked — that is normal).

## The loop

### 1. Search

```bash
vastai search offers 'num_gpus=1 compute_cap>=750 direct_port_count>=1 disk_space>=40 inet_down>=200 reliability>0.9 rentable=true' -o dph_total --limit 12 --raw
```

- `compute_cap>=750` (Turing+) is **required**: the pinned `torch==2.7.0+cu128`
  wheels have no kernels for older cards, and the failure is a runtime
  `no kernel image is available`, not a clean install error.
- `inet_down>=200` matters more than it looks — provisioning pulls ~2–3 GB of torch.
- **`inet_down` does NOT predict GitHub speed, and the repo clone is the real cost.**
  Measured on a 3090 advertising 533 Mbps down: the `git clone` ran at **~1 Mbps**
  (interface RX, 30 s sample) and took **>15 min** for the 156 MB filtered clone,
  while torch pulled at full speed. `--filter=blob:limit=1m` makes GitHub compute
  the pack server-side and that path is throttled; the advertised figure is a
  benchmark against Vast's own endpoints. Budget provisioning at ~20 min, not 9,
  unless you use `SPARSE_NO_DATA=1`.
- **If the job needs only one or two artifacts from `backend/data`, prefer
  `SPARSE_NO_DATA=1` plus `scp`.** A 9.8 MB clone finishes in seconds; copying the
  single 32 MB `champion.npz` a `--sampler-init` run needs beats waiting 15 min for
  all 156 MB.
- Cheap 8 GB cards are fine for 20bb/50bb work; check VRAM against the profile first.

### 2. Launch

```bash
vastai create instance <OFFER_ID> --image vastai/pytorch:@vastai-automatic-tag \
  --disk 40 --ssh --direct --cancel-unavail --label <job-name> --raw
```

Returns `"new_contract": <INSTANCE_ID>`. `--disk 40` fits image + torch + a
156 MB clone with room for checkpoints; raise it for long checkpoint sweeps.

### 3. Connect — use the direct path, not the proxy

`ssh_host`/`ssh_port` from `show instance` are the **Vast proxy**. That tunnel can
be broken on a perfectly healthy instance (seen: `Error: remote port forwarding
failed for listen port NNNNN` looping every 2s, while the pod itself was fine).
Because we always launch with `--direct`, prefer:

```bash
vastai show instance <ID> --raw     # read: public_ipaddr, ports["22/tcp"][0].HostPort
ssh -o StrictHostKeyChecking=no -p <HostPort> root@<public_ipaddr>
```

**`Connection refused` right after launch is normal** — sshd starts only after the
image's own onstart work finishes. Retry for a few minutes. If it persists, run
`vastai logs <ID>` **before** destroying or relaunching (the rejection reason is
in there). A dead proxy is not a dead pod.

### 4. Provision

`cloud_setup.sh` clones the repo itself, so it runs before the repo exists. Copy
it over and run it — do not `curl | bash` from GitHub unless master already has
the version you want to test.

```bash
ssh $SSHOPTS root@$IP 'cat > /root/cloud_setup.sh' < tools/cloud_setup.sh
ssh $SSHOPTS root@$IP 'bash /root/cloud_setup.sh'
```

Expect **~3–4 minutes** with the current defaults: apt update ~20s, system packages
~30s, unfiltered clone ~1–2m, torch skipped when the image already satisfies the
pin, 60s GPU workload. Budget ~14m if the image ships an incompatible torch.

That is after three fixes made on 2026-08-01, when a run took **40 minutes**:

- the blob filter is now **off** by default (it made the clone take 26m at ~1 Mbps
  and every push re-send ~77 MB) — `FILTER_BLOBS=1` restores it
- the full `apt upgrade` is now **opt-in** (3–5m of irrelevant packages, and it
  restarted sshd mid-provision, dropping the live session) — `FULL_UPGRADE=1`
- torch is **skipped** when the installed version already matches the pin *and*
  has CUDA support (11m saved; the CUDA check matters because a CPU-only wheel of
  the right version passes a naive version test and fails at the first kernel)

Other overrides: `SPARSE_NO_DATA=1` (skips `backend/data` entirely — do **not**
combine with a `--sampler-init` run, which needs a `champion.npz`),
`SMOKE_SECONDS=5`, `HOLDEM_DIR`, `BRANCH`.

It must print `GITHUB_TOKEN present -> push enabled`. If it prints the
`WARNING: no GITHUB_TOKEN` block instead, **stop and fix that before starting a
long job** — the pod cannot return results.

### 5. Run the job, then confirm the output is pushable

Run the workload from the clone (`/root/Holdem` when the image has no `/workspace`).
Use `tmux` or `nohup` so an SSH drop does not kill it.

Before committing to a long run, confirm where the job writes and whether git will
take it:

```bash
ssh $SSHOPTS root@$IP 'cd /root/Holdem && git check-ignore -v logs/ artifacts/ <your-output-path>'
```

Anything it prints **is ignored and will not be committed by a plain `git add`.**
This repo's `.gitignore` covers `logs/`, `artifacts/`, `tmp/`, and `backend/data/*`
(with narrow `champion.npz` exceptions). `cloud_push.sh` force-adds `logs/` and
`artifacts/` for exactly this reason. If your job writes anywhere else that is
ignored, add it to the force-add list in `cloud_push.sh` or the commit will be
empty and the script will exit with `nothing to push`.

**Smoke-test the push before the long run.** Write one throwaway file to `logs/`,
run step 6, confirm the branch appears, then start the real job. Ten seconds now
beats discovering broken auth after six hours.

### 6. Push — and verify it actually landed

```bash
# The clone only contains cloud_push.sh once it is committed to master.
# Until then, copy it in:
ssh $SSHOPTS root@$IP 'cat > /root/Holdem/tools/cloud_push.sh' < tools/cloud_push.sh

ssh $SSHOPTS root@$IP 'cd /root/Holdem && bash tools/cloud_push.sh "20bb sweep, 50k iters"'
```

It pushes to a **pod branch** (`pod/<id>-<date>`), never master, so a pod and your
local work can never reject each other. Checkpoints are excluded by default —
pass `INCLUDE_CHECKPOINTS=1` to ship `*.npz`, but see the warning below.

Now verify **from your workstation**, not from the pod's own exit code:

```bash
git ls-remote --heads origin 'pod/*'                  # branch must be listed
git fetch origin && git log --stat origin/pod/<branch>   # inspect what arrived
```

Only after the branch is confirmed present with the expected files is the job done.

### 7. Destroy

```bash
vastai destroy instance <ID> -y                       # -y is mandatory
vastai show instances-v1 --raw --limit 25             # must report instances_found: 0
```

Confirm the count. `exited`, `unknown` and `offline` are terminal states that still
accrue disk charges — destroy those too rather than waiting for recovery.

## Checkpoint policy

`*.npz` blueprints are 18–33 MB of incompressible binary and `.git` is already
~190 MB. Git keeps every pushed version forever, so routinely pushing checkpoints
bloats the repo permanently. `cloud_push.sh` excludes them by default on purpose.

- One-off promotion candidate → `INCLUDE_CHECKPOINTS=1`, fine.
- Every-run checkpoint shipping → wrong tool. Use R2/S3 (`vastai cloud copy`) or
  accept that only metrics come back over git.
- GitHub hard-rejects any file >100 MB and the rejection kills the whole push;
  `cloud_push.sh` pre-checks at 90 MB so this fails locally, cheaply.

## Knowing when a remote phase has finished

Do not poll. A `for i in $(seq 1 10); do ...; sleep 90; done` loop always outlives
the tool's foreground timeout, so every attempt leaves a lingering background shell,
and it still cannot tell you the moment the thing finished. Nine stray shells
accumulated this way in one session.

Wait on a **condition** with an explicit failure branch, backgrounded once, so the
harness notifies you the instant it resolves:

```bash
ssh $SSHOPTS -p $PORT root@$IP '
while true; do
  if grep -q "READY:" /root/provision.out 2>/dev/null; then echo "COMPLETE"; break; fi
  if ! pgrep -f cloud_setup.sh >/dev/null 2>&1; then echo "DIED (no process, no READY)"; break; fi
  sleep 20
done
grep -E "^=== " /root/provision.out
grep -E "GITHUB_TOKEN present|WARNING: no GITHUB_TOKEN" /root/provision.out'
```

The second branch is the point: a poll loop spins happily against a script that
already died. Exit on failure too, or the notification never fires.

Three usable completion signals for the clone specifically, strongest first:
the next `=== <time> <phase>` line appearing after `cloning into`; `tmp_pack_*`
being replaced by `pack-<sha>.pack`; and `READY:` for the whole script.

## Gotchas (all measured on real pods, 2026-08-01)

**`du -sk .git` shows zero growth during a healthy clone.** Git streams into
`.git/objects/pack/tmp_pack_XXXXXX` and only renames at the end, and `du`'s
rounding hides it. A frozen `du` number plus a frozen log is NOT evidence of a
hang — this looked exactly like a stall for several minutes while the transfer was
fine. Check the real signals instead: `ls -l .git/objects/pack/tmp_pack_*` growing,
and `/sys/class/net/eth0/statistics/rx_bytes` differenced over 30 s.

**Measure CPU on the child, not the wrapper.** `ps -o cputime= -p $(pgrep -f
cloud_setup.sh)` reads `00:00:00` on a perfectly busy provision, because the shell
sleeps while `git`/`apt`/`pip` do the work. Look at the child process or at
filesystem/network counters.

**"Template not found" in the web UI is cosmetic.** Launching with `--image` leaves
`template_id: None`, so the panel has no template record to name. Confirm health
from `actual_status: running` and `status_msg: success, running <image>` instead.

**Env-vars do not reach SSH sessions.** Vast puts account env-vars in the
container's PID 1 environment only. `GITHUB_TOKEN` is empty in
`ssh pod 'command'`, empty in `bash -l`, and in no profile file — while present in
`/proc/1/environ`. Both scripts recover it from PID 1. If you write your own
remote script that needs a Vast env-var, you must do the same.

**`python` does not exist in a non-interactive shell.** The image keeps it in
`/venv/main`, which `ssh host 'python ...'` never activates. `cloud_setup.sh`
sources it; ad-hoc commands should use `/venv/main/bin/python` explicitly.

**No `/workspace` on some images.** `cloud_setup.sh` falls back to `$HOME/Holdem`.
Read the `READY:` line for the actual path rather than assuming.

**Shell scripts must stay LF.** `core.autocrlf=true` here; `.gitattributes` pins
`*.sh text eol=lf`. A CRLF copy dies on the pod with `bad interpreter: bash^M`.

**cu128 on CUDA 13.x images is fine.** Verified: `torch 2.7.0+cu128` ran on a
`pytorch_cuda-13.2.1` image with driver 595.71.05. Don't "fix" the pin.

**Headless is faster.** Same RTX 3060: 4.7 it/s on a pod vs 3.4 it/s locally
(identical 635 MiB peak VRAM), because the local card also drives a display.
Do not compare a pod number against a local baseline without noting this.

## Cost reference

Measured: a full provision + 60s workload + a push-verification pod cost **$0.09
total**. Cheapest suitable single-GPU offers run $0.05–0.09/hr.

**`--disk 40` costs more than it looks. Read `dph_total` back after launch.**
Measured on a 3090: the offer quoted **$0.113/hr**, and the created instance
reported `dph_total` **$0.155/hr** — the disk added **$0.042/hr**, i.e. **+37%**,
not the ~$0.02 assumed earlier. Storage is priced per host, so the multiplier
varies; always confirm with `vastai show instance <ID> --raw` and cost the run off
that figure rather than the search result.

Rough per-workload costs at ~$0.155/hr (measured throughput, RTX 3090):

| job | time | cost |
|---|---|---|
| provision only (throttled clone) | ~20 min | ~$0.05 |
| 200bb blueprint, 20k iterations, 147k-node tree | ~11 h | ~$1.70 |
| 200bb blueprint, 40k iterations | ~23 h | ~$3.60 |

A 40k run therefore consumes an entire $3.60 balance. Check credit **before**
launching a sweep, not after.

Check headroom before a long run — credit is credit-only here, with no card on
file, so a run that outlives the balance just dies:

```bash
vastai show user --raw          # read "credit"
```

## Failure triage

| Symptom | Cause | Action |
|---|---|---|
| `Connection refused` after launch | sshd not up yet | retry a few minutes, then `vastai logs <ID>` |
| `Connection closed by <ip>` | proxy tunnel broken | use `public_ipaddr` + `ports["22/tcp"]` |
| `WARNING: no GITHUB_TOKEN` | env-var added after launch | fix before running the job, not after |
| `nothing to push` | output path is gitignored | `git check-ignore -v <path>`, extend force-add list |
| `not a git repository` | script run outside the clone | `cd` into it first |
| `no kernel image is available` | `compute_cap < 750` | pick a Turing+ offer |
| push rejected, file >100 MB | checkpoint too large | object storage, not git |
