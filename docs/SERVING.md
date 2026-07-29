# Serving the strongest agent

**Start it:**

```powershell
.\tools\serve_best.ps1
```

That script sets the resolve menus before launching uvicorn. Starting uvicorn bare
serves the module defaults instead, which are not the measured configuration.

Then `GET /api/health` reports exactly what is deciding hands:

```json
{
  "serving_agent": "MultiStackBlueprintAgent",
  "iteration": 30000,
  "stack_depths_bb": [100.0, 200.0],
  "search": {
    "continual_search": true,
    "resolve_streets": ["flop", "turn", "river"],
    "continual_iterations": 120,
    "continual_min_iterations": 60,
    "continual_budget_ms": 45000,
    "bucketed_subgame_search": false,
    "river_net_requested": false,
    "river_net": false,
    "resolver_resources": {
      "max_vram_mib": 9500,
      "required_headroom_mib": 2048,
      "flop_node_budget": 12000,
      "showdown_workspace_mib": 384
    }
  }
}
```

Check that endpoint before believing anything about strength. The previous default
silently enabled the **retired** bucketed resolver (below), while the docs
described the champion as search-off.

## What is served, and why each choice

| component | state | evidence |
|---|---|---|
| Depth-routed blueprints (100bb, 200bb) | ON | Slumbot −18.33 bb/100 [−40.41, +3.74] over 20,000 hands |
| Exact-card **river** resolving | ON | fixes a documented leak class; 1,244/1,244 resolves, 0 fallbacks |
| Exact-card **turn** resolving | ON | affordable once the menu is capped: 890 nodes at 3 sizes/cap2 vs 3,032 at 5/cap2 |
| Exact-card **flop** resolving | ON when admitted | richest safe menu is selected from a resource ladder; otherwise the promoted blueprint plays |
| ALL-IN geometry guard | **OFF by default** | measured **−268.82 bb/100** (200bb) and **−124.00** (100bb) vs a min-raiser; never measured helpful |
| Bucketed subgame search | OFF | measured **−31 bb/100** [−95, +33] at 120 iters, **−86** [−150, −22] at 500 |
| River value net | OFF | **fails** its gate: 38% action agreement, policy L1 1.15 vs ~0.21 noise |

**Why all three streets, and why the menus are resource-adaptive.** Latency and
memory scale with tree size, and tree size is driven by the bet menu far more
than by the street. The requested menu is tried first when the SPR is shallow
enough. Deeper flops start at a two-size tier, and every candidate must pass the
12,000-node and VRAM estimates before CUDA allocation.

| street | nodes @5 sizes/cap2 | nodes @3 sizes/cap2 |
|---|---:|---:|
| flop (100bb) | 132,107 | geometry-dependent; 5,139-9,443 in the resource sweep at two sizes |
| turn (100bb) | 3,032 | 890 |

Capping own sizes does **not** cap responses: observed opponent bet sizes are
inserted into the tree regardless, so the agent still answers any size it faces
exactly. Only its own raise sizes narrow.

**What the exact river resolve buys.** Every solve gives each of the 1,326 private
combos its own bucket instead of sharing 30. On the documented exhibit — 7s5s on
4s5c6sKc, a 19-out draw — the blueprint folds **83%** to a 0.7-pot bet it needs
29% equity to call; the exact solve folds **3.2%**, while trash still folds
0.92–0.99. That is card-level discrimination, not looseness.

Honest caveat: the resolver's on/off duel is **inconclusive**, not proven
(+17.82 / +54.45 / +28.12 bb/100 at 60/240/240-CRN iterations, every interval
spanning zero). Nothing has measured it as harmful, the mechanism is sound, and
the arms tested were handicapped — 60 iterations sits L1 0.418 from converged. It
is served because quality is the priority and the evidence points one way, not
because the gate passed.

## Earlier latency sample (superseded)

| street | source | mean | max |
|---|---|---:|---:|
| preflop | blueprint | 0.01 s | 0.02 s |
| flop | blueprint | 0.11 s | 0.36 s |
| turn | blueprint | 0.04 s | 0.05 s |
| river | exact resolve | **1.84 s** | **2.16 s** |

That sample predates flop/turn serving and the VRAM-safe ladder, so it is not a
claim about the current end-to-end profile. Current decisions record
`decision_elapsed_ms`, selected menu, tree nodes, resource estimate, projection
coverage, continuation reuse, and any fallback in `server-debug.jsonl`.

## Tuning

| variable | default | effect |
|---|---|---|
| `HOLDEM_RESOLVE_STREETS` | `flop,turn,river` | which streets the exact resolver handles |
| `HOLDEM_FLOP_SIZES` / `HOLDEM_TURN_SIZES` | `0.33,0.75,1.4` | the agent's own raise menu when resolving |
| `HOLDEM_FLOP_CAP` / `HOLDEM_TURN_CAP` | `2` | raises per street in the resolve tree |
| `HOLDEM_CONTINUAL_ITERS` | `120` | solve quality per decision |
| `HOLDEM_CONTINUAL_MIN_ITERS` | `60` | quality floor; lower requested values are clamped |
| `HOLDEM_CONTINUAL_BUDGET_MS` | `45000` | hard per-decision ceiling |
| `HOLDEM_FLOP_NODE_BUDGET` | `12000` | latency and allocation guard for flop trees |
| `HOLDEM_RESOLVER_MAX_VRAM_MB` | `9500` | maximum resolver process allocation on a 12 GiB card |
| `HOLDEM_RESOLVER_VRAM_HEADROOM_MB` | `2048` | VRAM kept free for Windows, display, and other processes |
| `HOLDEM_SHOWDOWN_WORKSPACE_MB` | `384` | blocker-correction workspace ceiling |
| `HOLDEM_FLOP_RICH_MAX_SPR` | `4.0` | skips the predictably explosive rich tier above this SPR |
| `HOLDEM_CONTINUAL` | `1` | `0` serves the blueprint alone |
| `HOLDEM_SAFETY_PRICE_GRAPH` | `1` | graph-captures the exact same opponent safety-price traversals |
| `HOLDEM_RESOLVER_PREFETCH` | `1` | prepares sampled runouts in order while CUDA replays the prior one |
| `HOLDEM_SESSION_RUNOUT_CACHE` | `1` | reuses exact deal/bucket work within the same hand |
| `HOLDEM_RESOLVER_WARMUP` | `1` | pays CUDA/scorer initialization at server startup |

## Zero-quality latency pass (2026-07-29)

The safety gadget has two solve phases. The final gadget was already CUDA-graph
captured, but the preceding opponent safety-price best response still ran 40
eager traversals. Serving now captures that phase too. Its runout-aware
blueprint projection reads from fixed bucket buffers refilled before each replay,
so turn/flop chance resampling and the sequence drawn from the fixed seed are
unchanged.

The same pass adds ordered background runout preparation, reusable pinned host
buffers for graph inputs, per-session flop/turn sampler and blueprint-bucket
caches, and scorer/CUDA startup warmup. None changes iterations, ranges, menus,
buckets, precision, or the safe-gadget equations.

A fixed-seed eager-vs-optimized flop control produced **max root-policy
difference 0.0**. At 12 iterations on the same 9,274-node tree, end-to-end time
fell from 6.151 s to 5.576 s; capture overhead dominates such a deliberately
short solve, while the 120-iteration serving profile amortizes it.

Representative fresh 120-iteration measurements on the RTX 3060:

| street | nodes | total | setup | safety price | gadget | cleanup |
|---|---:|---:|---:|---:|---:|---:|
| flop | 9,274 | **14.23 s** | 0.51 s | 3.31 s | 10.34 s | 0.07 s |
| turn | 3,926 | **6.90 s** | 0.29 s | 1.74 s | 4.79 s | 0.09 s |
| river | 400 | **1.70 s** | 0.28 s | 0.62 s | 0.72 s | 0.07 s |

These are geometry-specific single-decision probes, not universal averages.
They are nevertheless conservative relative to the earlier live sample because
their trees are larger (earlier: 8,037 / 3,130 / 369 nodes at 22.78 / 10.64 /
2.45 s). Use `tools/benchmark_resolver_latency.py` for repeatable local probes.

Every successful resolve now records `stage_ms`, safety/gadget graph capture and
replay times, prefetch state, sampler reuse, and blueprint-bucket cache hits in
`server-debug.jsonl`.

## VRAM-safe flop resolving (2026-07-29)

The old admission check counted public nodes but did not estimate CUDA memory.
It could admit a full flop-to-river solve, allocate dense
`[nodes, 1326, actions]` baseline/output tensors, capture a second private CUDA
graph pool, and then retain the graph through cleanup. On the RTX 3060 this
reached 11.8/12.0 GiB dedicated VRAM, spilled into shared memory, and often
timed out.

Serving now applies these controls before constructing a solver:

1. Build the richest plausible action tier for the live pot/stack geometry.
2. Reject it when the tree exceeds 12,000 nodes, when estimated peak process
   allocation exceeds 9.5 GiB, or when it would consume the 2 GiB headroom.
3. Try progressively smaller own-bet menus; observed opponent sizes are still
   inserted exactly.
4. If no tier fits, use the frozen blueprint without allocating a solver.

Deep-SPR flops start at the two-size tier instead of spending seconds building a
known-oversized rich tree. In the implementation sweep, admitted 20-200bb
examples contained 987-10,429 nodes with conservative estimated peaks of
970-1,856 MiB. A real 200bb passive-entry flop produced 11,991 nodes at the
two-size tier.

The solver also keeps the loaded blueprints compact (20.6 and 27.1 MiB for the
current 200bb and 100bb strategies), exports only the acting node plus its
immediate response frontier, caps showdown card-channel temporaries at 384 MiB,
and destroys graph pools before emptying the CUDA cache.

`/api/health` reports both the configured limits and current CUDA
allocated/reserved/free memory. On Windows, `cuda_free_mib` may not perfectly
match Task Manager's global WDDM number, so the process allocation ceiling is
the primary guard.

## Resolver quality and fallback order

The live order is:

1. exact-card solve using the richest admitted menu;
2. a compact exact-card menu at the same real pot/stack geometry;
3. the promoted frozen blueprint if resource admission or the deadline fails.

The failed river CFV network is never activated merely by setting
`HOLDEM_RIVER_NET=1`. It must have a `river-net-acceptance` report with at least
0.90 action agreement and at most 0.30 policy L1. The current report is 0.3766
agreement / 1.1474 L1, so health reports the failure and the turn resolver
continues exactly to showdown.

**The budget is deliberately generous.** A blown deadline fails *closed* — the
session is marked failed and the whole remaining hand reverts to the frozen
blueprint, because resuming with a half-advanced belief would condition every
later solve on a range that never existed. One slow turn used to silently disable
the river resolver for that hand; that is exactly how the 15 s budget was found to
be too tight.

**Study profile** (analysis, not live play — expect 30–40 s per turn decision):

```powershell
$env:HOLDEM_RESOLVE_STREETS = "turn,river"
$env:HOLDEM_CONTINUAL_ITERS = "240"
$env:HOLDEM_CONTINUAL_BUDGET_MS = "120000"
```

## The ALL-IN geometry guard (off, and why)

A live hand on 2026-07-29 had the agent jam 3,980 chips into a 166-chip pot - 24x
the pot, from a blueprint whose widest own bet is 1.0x pot. Root cause: ALL-IN was
the only size-bearing action executed **literally**. Every `raise` re-derives its
chip amount from the real pot; all-in shoved whatever was behind. `_locate` matches
a live hand to an abstract node by action sequence and never compares pot/stack
geometry, so opponent min-raises that exhaust the tree's 3-raise cap land the hand
on a node where jamming is a sane 2-3x-pot action - which becomes 24x pot once
translated back.

`GpuBlueprintAgent._all_in_size` repairs the translation (preserve the abstract
jam's multiple of the matched pot, bounded at 6x) and cuts the worst jam from 15.4x
to 4.6x pot. **It is off by default anyway**, because the only A/B taken says it
costs 124-269 bb/100 against the min-raiser whose lines trigger it: against a
station that calls any jam, shoving 199bb is correct exploitation, not a bug. It is
neutral (-3.46 / -13.33) against everything else, and fires zero times in self-play.

Turn it on per agent with `agent.all_in_geometry_guard = True`, and tune with
`all_in_max_pot_multiple` / `all_in_geometry_tolerance`.

The structural fix is the resolver, not the guard: at the reported spot an exact
turn solve puts **0.16%** on all-in because its tree is built on the real geometry,
so there is no translation to distort. The gap the guard covers is therefore
**preflop**, where half the observed overbets occurred and where no resolving is
planned before P5. See docs/PLAN_V2_STRONGEST_PLAYER.md for the full write-up,
including the LBR measurement that would settle it and has not been taken.

## Depth coverage

Routing picks the nearest trained depth per hand, then resolves at the *exact*
stack, so an odd depth is absorbed by the resolver rather than mis-served.

| depth | blueprint | LBR exploitability |
|---|---|---:|
| 20bb | none — routes to 100bb | **+130.31** [+95.22, +165.40] |
| 50bb | none — routes to 100bb | +85.05 [−12.80, +182.90] |
| 100bb | histogram@30k | +137.58 [−3.57, +278.73] |
| 200bb | scalar@118k | +291.23 [+79.10, +503.36] |

**20bb is the weakest depth and has no blueprint of its own** — its interval
clears zero decisively, because a 100bb-trained blueprint plays far too many small
bets at a depth that is nearly push-fold. It is also the cheapest to fix: an exact
flop-to-river tree there is only 5,303 nodes, so 20bb can be played exactly on
every postflop street with no value net at all
(`backend/search/exact_flop.py`). That path is implemented and resource-gated;
the blueprint remains the fail-closed fallback.
