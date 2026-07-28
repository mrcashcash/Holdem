# GPU-CFR: Dense-Tensor Blueprint Training

**Date:** 2026-07-17 · **Status:** Implemented & verified (2026-07-21: solver convergence
proven correct by independent best response — see `docs/STATUS.md` §2). Production
trainer for all blueprints. Current state/ops: `docs/STATUS.md`; next steps:
`docs/RESEARCH_ROADMAP.md`.
**Goal:** Replace the per-node Python traversal (~40 iters/s on 4 CPU workers) with
vectorized, chance-sampled range CFR on the RTX 3060 — targeting hundreds of
equivalent iterations per second.

## Why the current trainer is slow

MCCFR walks one sampled line at a time through Python objects, and 70% of the
time goes to bucket assignment on equity-cache misses. A GPU cannot run that
loop — but it can run the *range-based* formulation used by DeepStack and all
modern postflop solvers: traverse the **betting tree once per sampled board**,
carrying *vectors over all 1,326 private combos* through every node.

## Design

### 1. The public betting tree is enumerated once (`gpu/tree.py`)

Chip amounts are a deterministic function of the action sequence, so the
action-abstracted betting tree is a finite tree, flattened into arrays:
`street[n]`, `actor[n]`, `pot/commit[n]`, `children[n, a]`, `terminal kind
[fold|showdown]`, `matched pot`. The GPU blueprint uses a **coarser action
menu** than the CPU blueprint (2 raise sizes + all-in, raise cap 3) to keep
the tree in the 10^4–10^5 node range; richness at play time comes from
re-solving (river now, turn later), following the Pluribus/Modicum philosophy:
coarse blueprint, fine search.

### 2. Strategy tensors are dense over (node, bucket)

`regret[n, b, a]` and `strategy_sum[n, b, a]` as float32 CUDA tensors, where
`b` is the acting player's street bucket (169 preflop / K flop / K turn / K
river). Sizing example: 50k nodes x 200 buckets x 6 actions x 4 B x 2 tensors
≈ 480 MB — comfortably inside 12 GB, leaving room for batching boards.

### 3. Buckets are computed per sampled board for all combos (`gpu/deals.py`)

The insight that makes this fast: **on a fixed board, the equity of all combos
is one sort of their 7-card scores** (equity = normalized rank, ties averaged).
So per sampled runout: score all 1,326 combos (Numba batch, ~1 ms), then
- river bucket = equity quantile (from the sort),
- turn bucket = quantile of mean equity over sampled rivers,
- flop bucket = quantile of mean equity over sampled runouts,
- preflop = the lossless 169 classes.

This is a scalar-equity abstraction (weaker per-bucket than the CPU
blueprint's potential-aware EMD k-means, compensated by lossless combo ranges
inside the iteration and far more iterations per hour). GPU artifacts are
therefore **separate** from the CPU blueprint's (`backend/data/gpu_blueprint/`).

### 4. Vectorized chance-sampled CFR (`gpu/cfr.py`)

Per iteration (batched over B boards):
1. Sample board(s); build bucket tensors + card-collision masks.
2. **Forward** (topological order): reach-probability vectors per player,
   `reach[n, combo]`, gathered through regret-matched strategies
   `sigma[n, bucket(combo), a]`.
3. **Terminals:** fold nodes pay ±matched pot x opponent reach (O(n) with the
   per-card collision correction); showdown nodes use the score-sort trick:
   values for every combo against the opponent's whole reach vector in
   O(n log n) instead of O(n^2), with standard per-card and same-combo
   corrections.
4. **Backward:** counterfactual values roll up; regret increments and
   strategy sums scatter-add into the dense tensors with linear-CFR weights.

All steps are tensor ops → the 3060 does the work; the CPU only samples
boards and orchestrates.

### 5. Serving and evaluation

`GpuBlueprintAgent` mirrors the existing serving contract; bucket lookup at
play time uses the same sort-based equity (fast). The eval harness
(styles benchmark, LBR) applies unchanged. River re-solving carries over.

## Phases

1. **Tree enumeration + parity tests** — flatten the betting tree; verify
   pot/stack/legality parity against `AbstractHoldem` on random lines.
2. **Deals module** — per-board combo scores, sort-based equity, bucket ids,
   collision masks; validated against `river_equity` ground truth.
3. **CFR core** — forward/backward over a tiny config on CPU torch first
   (exactness debuggable), then CUDA; validate by benchmarking the resulting
   strategy vs the scripted styles and comparing convergence against the CPU
   MCCFR blueprint at equal wall-clock.
4. **Trainer CLI + serving integration** — checkpoints, telemetry,
   `--device cuda`, GpuBlueprintAgent behind the existing API.
5. **Scale-up** — batch boards, grow buckets/sizes to VRAM budget, retire the
   CPU blueprint when the GPU one dominates it on the eval suite.

## Risks

- **Correctness of the showdown/collision math** — the classic source of
  solver bugs; mitigated by exhaustive small-case tests vs brute force.
- **float32 regret growth** — linear CFR sums grow ~t^2; mitigate with
  periodic DCFR-style discounting (also improves convergence).
- **Scalar-equity turn/river buckets are weaker** than potential-aware EMD —
  accepted for v1; upgradeable later by porting histogram k-means to torch.
