# CFV Network: Depth-Limited Search on All Streets

**Date:** 2026-07-23 · **Status:** BUILT, MEASURED, PARKED (2026-07-25)

> Outcome: every pipeline stage works and is verified (horizon plumbing
> bit-identical to trusted kernels; bucket-level net 9.3bb val MAE vs 24bb
> zero-baseline; end-to-end flop solves ~7s warm). But the honest A/B — real
> solves, NULL-tested harness — reads −65 bb/100 [−157,+27] vs no flop search:
> the v0 net's value error (bounded by 4-runout target noise, not sample
> count: learning curve flat 4.5k→7.7k samples) is too large to beat the
> blueprint's own flop play. Revival: regenerate targets with 16+ runouts
> (datagen is resumable), retrain, re-A/B. Key transferable findings: raw-combo
> I/O scores below a zero-predictor at feasible sample counts (bucket I/O is
> mandatory); datagen needs SPR-grid solver/graph reuse + coarse menus.
**Goal:** a counterfactual-value network that evaluates turn states, so the
flop (and later preflop) can be re-solved depth-limited in real time — the
DeepStack / ReBeL architecture, the last structural gap between this agent and
the published superhuman systems.

## Why now

- Blueprint iteration count is an exhausted lever (proven twice by honest
  head-to-head duels at both depths).
- Search is the one proven big lever (+511 bb/100 relative uplift, turn/river
  only today). Extending it to the flop requires evaluating turn states
  without solving to the end of the game — a value function.
- Our GPU subgame solver is verified-correct and fast (turn solve ~3s with
  CUDA graphs) — exactly the data generator DeepStack needed a cluster for.

## Architecture (v1 = turn network)

**Function:** f(pot, board4, range_p0, range_p1) -> (cfv_p0, cfv_p1)
per-combo counterfactual values at a turn root, zero-sum enforced.

- **Input encoding:** pot (normalized by starting stack), board as 4x52
  one-hots (52 floats summed), ranges as two 1326-dim reach vectors
  (normalized). Total ~2708 floats.
- **Body:** MLP, 7 hidden layers x 500 units, PReLU (DeepStack's shape; fits
  easily on the 3060).
- **Output:** 2x1326 CFVs + DeepStack's zero-sum outer layer: subtract the
  range-weighted violation so sum_c r0[c] cfv0[c] + sum_c r1[c] cfv1[c] = 0.
- **Loss:** Huber on CFVs (per-combo, masked to valid combos), reported in
  pot-normalized units.

## Data generation (the GPU job)

Per sample:
1. Random turn situation: board (4 cards), pot from a recursive bet-history
   sampler, both ranges from the pseudo-random recursive range generator
   (DeepStack supplement) so range shapes match what re-solving produces.
2. Solve turn+river to the end with VectorCFR (GraphRunner, 500 iters).
3. Extract per-combo root CFVs for both players (the `_last_root_values`
   machinery, averaged over final iterations).
4. Store (encoding, cfv targets) — float16 on disk.

Budget: ~50k-200k samples (DeepStack used 10M for its licence-level net; the
literature after it — Supremus — showed far fewer, better-generated samples
work; we start at 50k ≈ 2 GPU-days and scale by measured validation loss).

## Deployment (flop re-solving)

Flop subgame tree truncated at the turn transition: chance deals the turn
card, then a pseudo-terminal node calls the net for both players' CFVs
instead of continuing. The safe-resolve gadget (already built) wraps the same
way. River re-solve stays exact (no net). Latency target: <2s per flop
decision (one net batch per chance card x betting states).

## Phases

1. **P1 (this build):** dataset generator + net + training loop + unit tests
   (encoding roundtrip, zero-sum layer exactness, tiny-overfit sanity).
2. **P2 (GPU):** generate 50k turn samples (interleaved with serving; after
   the hist run's honest plateau), train, report validation CFV error vs a
   held-out solved set.
3. **P3:** depth-limited flop solve using the net; A/B vs blueprint-only flop
   play at 3000 hands/style + head-to-head duel (NULL-tested harness).
4. **P4 (later):** preflop net or turn-net bootstrapping (ReBeL-style),
   decision-variate AIVAT using the same net.

## Risks

- Net error compounds through re-solves — measured by P3's A/B before any
  serving change; the safe gadget bounds the damage vs the blueprint.
- Range-generator mismatch (train vs deployment distributions) — mitigate by
  also harvesting ranges from actual blueprint play.
- 3060 VRAM: batches sized to <2GB; training runs when the GPU is otherwise
  free.
