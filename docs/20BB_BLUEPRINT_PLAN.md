# Native 20bb Blueprint Implementation

**Date:** 2026-07-29
**Status:** Training complete through 50,000; 5,000 remains champion

## Objective

Replace the current 20bb fallback to the 100bb blueprint with a native 20bb
GPU-CFR blueprint. Exact-card flop/turn/river resolving remains enabled above
the blueprint; the native model supplies the missing preflop policy, continual
range prior, and fail-closed fallback.

The baseline to beat is the deployed 100bb histogram champion played at 20bb:
LBR exploitability **+130.31 bb/100** with 95% CI **[+95.22,+165.40]**.

## Implemented configuration

`backend.solver.gpu.train.BLUEPRINT_CONFIG_20` is selected automatically by
`--stack-bb 20` unless an explicit ruleset is requested:

| Item | Value |
|---|---|
| Preflop fractions | 0.50, 0.75 (2bb and 2.5bb opening sizes) |
| Postflop fractions | 0.33, 0.66, 1.00, 1.50 |
| Raises per street | 2 |
| Donk bets | Disabled for the OOP caller in single-raised pots |
| Card abstraction | histogram-EMD |
| Artifact directory | `backend/data/gpu_blueprint_20bb/` |

`tools/size_blueprint_menus.py` measured this tree at 36,906 total nodes,
13,706 decision nodes, and approximately 176 MiB after the no-donk rule.
strategy tables. The durable result is
`backend/data/evaluations/blueprint-menu-sizing.json`.

Checkpoints at 5,000, 10,000, 20,000, and every later 20,000 iterations are
retained automatically.

## Verification before training

Run the focused guards:

```bash
.venv/bin/python -m unittest \
  tests.test_20bb_blueprint \
  tests.test_gate_bootstrap \
  tests.test_multistack_routing
```

Reproduce menu sizing:

```bash
.venv/bin/python tools/size_blueprint_menus.py \
  --depths 20 \
  --output backend/data/evaluations/blueprint-menu-sizing.json
```

Do not run GPU tests or other GPU jobs concurrently with training.

## Training sequence

Train the first 5,000-iteration milestone:

```bash
.venv/bin/python -m backend.solver.gpu.train \
  --stack-bb 20 \
  --abstraction histogram \
  --iterations 5000 \
  --save-every 200 \
  --batch-boards 1 \
  --device cuda \
  --seed 20260729
```

This creates `backend/data/gpu_blueprint_20bb/checkpoint.npz`; it does not
create or replace a champion.

The final two-size preflop configuration reached iteration 5,000 on 2026-07-29
in 1,433 seconds. The checkpoint SHA-256 is
`888f38ac9bf75bf706ff928fdb705347202151f9d3a54e86015690a1f4ee345c`;
`checkpoint-5000.npz` is retained beside the live training checkpoint.
The bootstrap gate then passed:

- screen: +29.38 bb/100, 95% CI [+5.60,+53.15], 750 pairs;
- disjoint confirmation: +32.08 bb/100, 95% CI [+20.16,+44.01], 3,000 pairs;
- LBR: +22.02 bb/100 [-25.85,+69.90], versus the deployed fallback's
  +116.88 [+80.03,+153.72], 400 pairs each; and
- mapping: pass (0.31% fallback, translation gap slightly below incumbent).

The gate marked the checkpoint eligible but the read-only run did not install
it. The trainer's `--iterations` argument is an increment, not an absolute
target.

The eligible 5k checkpoint was subsequently promoted. Training continued to
50k with saves every 1,000 iterations and full gates every 10k:

| Iteration | Screen bb/100 | Confirmation bb/100 | Decision |
|---:|---:|---:|---|
| 10k | −3.65 [−25.94,+18.63] | not run | tie; keep 5k |
| 20k | −18.22 [−39.47,+3.03] | not run | tie; keep 5k |
| 30k | +9.46 [−12.33,+31.25] | +0.48 [−10.20,+11.17] | tie; keep 5k |
| 40k | −24.68 [−46.23,−3.13] | not run | significant regression |
| 50k | +3.61 [−17.89,+25.11] | +0.32 [−10.28,+10.91] | tie; keep 5k |

At 50k, LBR was +9.14 bb/100 [−34.85,+53.12] versus the 5k champion's
+23.08 [−18.97,+65.14] on the shared block; the difference was not
statistically decisive. Mapping passed with zero fallback. The repeated ties
and one significant regression establish a plateau under this configuration:
do not continue identical training beyond 50k.

At each milestone:

1. Confirm checkpoint configuration, iteration, telemetry, and CUDA memory.
2. Run legal-action and routing tests.
3. Run blueprint-only LBR at 20bb.
4. Run the promotion-grade gate with fresh/disjoint seeds.
5. Continue only if mapping, fallback, LBR, and head-to-head results do not
   regress. Stop after two statistically tied milestones.

## First-depth bootstrap gate

The first native 20bb checkpoint has no same-depth champion. Evaluator v4
therefore supports a narrowly guarded cross-depth incumbent:

```bash
.venv/bin/python -m backend.eval.gate \
  --data-dir backend/data/gpu_blueprint_20bb \
  --stack-bb 20 \
  --incumbent backend/data/gpu_blueprint/champion.npz \
  --allow-bootstrap-incumbent \
  --screen-pairs 750 \
  --confirm-pairs 3000 \
  --lbr-pairs 400
```

The mode requires the challenger to be native 20bb, requires the incumbent to
be cross-depth, records all three depths in the report, and refuses to run if a
20bb champion already exists. After the first promotion, every gate must use a
same-depth incumbent and omit `--allow-bootstrap-incumbent`.

Add `--promote` only after reviewing a completed read-only report. Promotion
requires a positive confirmatory confidence interval plus retained-model,
mapping, and relative-LBR gates.

## Serving and routing

`MultiStackBlueprintAgent` already discovers
`backend/data/gpu_blueprint_20bb/champion.npz`. No server change is needed.
Routing is locked for the hand; equal-distance ties choose the shallower
blueprint because depths are sorted. Consequently, after promotion:

- 20bb routes to 20bb;
- 50bb, equidistant from 20 and 100, routes to 20bb;
- depths above 50bb route to 100bb until the 100/200 boundary.

Before release, verify `/api/health` reports stack depths `[20, 100, 200]`, and
run the four-depth continual-resolver smoke. A resolver admission failure must
fall back to the native 20bb champion.

## Promotion and completion criteria

Promote only when:

- the 3,000-pair confirmatory interval against the deployed fallback clears
  zero;
- 400-pair LBR materially improves on the recorded 20bb baseline;
- fallback and translation gates pass;
- no legality, stack-geometry, projection, or routing regression appears; and
- the checkpoint and evaluation manifests identify the exact code, sampler,
  action configuration, hashes, and seeds.

After promotion, update `docs/STATUS.md` and `docs/SERVING.md` with the actual
iteration, checkpoint hash, LBR result, confirmatory result, routing coverage,
and resolver smoke result. Do not describe the blueprint as serving before
`champion.npz` exists and health confirms it loaded.
