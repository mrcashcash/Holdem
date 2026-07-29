# Plan V2 — The Strongest Possible HUNL Player On This Hardware

**Date:** 2026-07-27
**Hardware:** RTX 3060 12GB (28 SM, ~360 GB/s), 8 cores, 32 GB RAM, torch 2.7+cu128
**Supersedes as the strategic document:** `RESEARCH_ROADMAP.md`, `TRAINING_QUALITY_OPTIMIZATION_PLAN.md`
(their Phase 2/3 abstraction lines are explicitly retired below; `STATUS.md` stays
as the living experiment record).

---

## 0. The one-paragraph answer

Stop trying to make a *bucketed blueprint* stronger. That line is measured-out and
capped by hardware you do not have. The strongest player reachable on one 3060 is
the **DeepStack/Supremus architecture**: no card abstraction anywhere in the played
strategy, and **continual depth-limited re-solving with exact 1,326-combo ranges on
every street**, where the depth limit is priced by a hierarchy of counterfactual-value
networks (river → turn → flop → preflop). Until 2025 that architecture needed a
cluster (DeepStack: 175 core-years for turn data alone). Three things now make it
feasible on a desktop: (a) **single-solve/multi-iteration target generation**
(TurboReBeL, ~250× belief-learning speedup), (b) adding a **river net** so turn
solves stop expanding 44 river subtrees, and (c) **situation-batched solving** on a
GPU that is currently ~5% utilized on small trees. This is not a from-scratch
rewrite: the verified CFR kernels, tree builder, graph runner, exact-river resolver
and eval harness all carry over. What gets deleted is the card-abstraction research
programme. **And nothing starts until the player has an honest absolute number,
which today it does not have.**

---

## 1. Diagnosis — what is actually wrong

### 1.1 The player's strength is unmeasured (the single biggest problem)

Every absolute number in the docs (`~+130 bb/100` at 100bb, `~+290` at 200bb) comes
from a field of **scripted style bots** in `backend/styles.py`. Beating scripted
opponents by 130 bb/100 is evidence of nothing about equilibrium quality — a
maximally-exploitable rock can post those numbers. The two instruments that *would*
measure real strength are both non-functional:

- `backend/eval/slumbot.py` exists but **has apparently never produced a number**
  (docs call Slumbot "parked"). It also has real defects: `client_incr()` collapses
  the agent's rich menu into a 4-way choice with a hard-coded `0.5`-pot default raise
  (line 145), so even if run today it would measure a crippled agent.
- `backend/eval/lbr.py` is admitted in the docs to be untrustworthy.

Consequence: three instrument bugs already invalidated an entire month of A/B results
(`STATUS.md` §4). Repeating experiments against scripted bots and self-duels cannot
detect a fourth. **You are optimizing an unobserved quantity.**

### 1.2 You are paying exact-card compute for bucketed-strategy quality

This is the structural insight the docs miss. `VectorCFR._iterate` carries
`reach[2, nodes, 1326]` — **full exact-combo vectors at every node** — and then
indexes regrets by 150–384 buckets. The abstraction buys **memory, not speed**. So:

| | cost paid | quality received |
|---|---|---|
| current blueprint | exact-card (1,326-wide tensors) | 150-bucket flop/turn, **30-bucket river** |
| exact subgame solve | exact-card (1,326-wide tensors) | exact (1,326 identity buckets) |

The measured throughput is **0.77 iterations/second** (`backend/data/*/telemetry.json`,
consistent across 45 and 117 checkpoints). At 143k nodes × 1,326 combos × fp32, one
iteration moves ~10 GB of VRAM traffic, so the solver is bandwidth- and
launch-bound, not compute-bound. 118k iterations ≈ 2.5 GPU-days bought **118k public
chance samples** over a tree whose river strategy is stored in 30 buckets.

Same compute spent on *small exact-card subgames* buys 1–2 orders of magnitude more
strategic resolution. That is the whole thesis of this plan.

### 1.3 The abstraction race is unwinnable on this box

Slumbot — the free public benchmark, and the weakest agent worth calling a benchmark —
is an abstraction blueprint computed on a 64-core/512 GB machine over months, with
orders of magnitude more buckets and betting sequences. 12 GB of VRAM caps you at
~150k nodes (`STATUS.md` §6) and the 0.77 it/s ceiling caps chance coverage. Every
plateau result in `STATUS.md` §2.2 is this wall, correctly measured. More
potential-aware clustering (the planned Phase 2 line) is optimizing the wrong
quantity: it makes a better *blueprint*, and the blueprint should stop being the
player.

### 1.4 Search failed for five fixable engineering reasons, not conceptual ones

The retired-search verdict in `STATUS.md` §3.1 is correct about *that*
implementation and wrong as a general conclusion. Every superhuman HUNL agent ever
built searches. What was actually broken:

1. **Bucketed subgames.** `gpu_subgame.py` re-solves with the same 150/30 buckets as
   the blueprint. A re-solve at the blueprint's own resolution has **no information
   edge** — it can only re-derive the blueprint's answer with fewer samples. The
   −86 bb/100 at 500 iterations is exactly what "converging harder onto a strictly
   less-informed problem" looks like.
2. **Self-range inconsistency** — correctly diagnosed in the docs.
3. **The value net was 169-bucket I/O** (`NetEvaluator.BUCKETS = 169`,
   `depth_limited.py:113`). DeepStack and Supremus both use **1,000 clusters per
   player** (net input 2,001 = 1000+1000+pot). 169 buckets cannot express the value
   differences that drive flop play. This alone is disqualifying.
4. **Targets were 4 sampled runouts × 500 iterations** (`cfv/dataset.py:128`, `:98`).
   Supremus used **4,000 DCFR+ iterations per player**. Runouts should be
   *enumerated*, not sampled — 4-runout Monte Carlo noise is what the docs correctly
   identified as the binding constraint.
5. **7,750 training samples.** DeepStack: 10M turn / 1M flop. Supremus: 20M turn /
   50M river / 5M flop. You were three to four orders of magnitude short.

### 1.5 Two conclusions in the current docs should be reversed

- **"Raw-combo net I/O cannot generalize"** (`STATUS.md` §3.4). This was measured at
  7.7k samples, where *nothing* generalizes to 1,326 outputs. DeepStack and Supremus
  both run ~1,000-way per-player I/O successfully. The finding is a sample-size
  artifact and should not be allowed to force 169-bucket nets.
- **"Bucketed re-solving does not help, therefore search is retired."** Replace with:
  *re-solving at the blueprint's own resolution cannot help by construction; exact-card
  re-solving is a different intervention and is the core of the new architecture.*
  The Phase 4 exact-river result (+7.62 bb/100, tie) is consistent with that: it is
  the only search variant that wasn't negative, and it was crippled by a 1.32%
  projection-failure rate and a blueprint baseline it could not improve on the river
  alone.

### 1.6 What is genuinely good and must be preserved

- `VectorCFR` terminal math: sort-based showdown with inclusion–exclusion blocker
  correction, verified against a Kuhn/Leduc-validated best response at 0.0 mbb.
  This is the hard part of a poker solver and it is correct.
- `BettingTree` with `start_street` / `start_pot` / `start_stacks` / `end_street` /
  exact mid-street roots — everything needed for arbitrary subgame re-rooting.
- `GraphRunner` CUDA-graph capture, the `HORIZON` node kind + evaluator hook,
  `frozen_average` evaluation passes (exactly the primitive TurboReBeL needs),
  `ExactRiverSampler` (identity 1,326 buckets — proof the kernels are
  abstraction-agnostic), `GadgetCFR` safe re-solving, `duel.py`/`gate.py`
  NULL-tested promotion machinery, `ChanceCorrector` (AIVAT).
- The whole serving/frontend/screen-scraper stack is orthogonal and untouched.

**Verdict on "start from scratch": no.** Roughly 2,500 lines of solver core and
1,500 lines of eval carry over intact. What is retired is a *research programme*
(card abstraction), not a codebase.

---

## 2. What the literature says wins

| Agent | Architecture | vs Slumbot | Notes |
|---|---|---|---|
| Slumbot 2017 | big abstraction blueprint, no search | — (the benchmark) | cluster-scale offline compute |
| DeepStack (2017) | continual re-solving, exact cards, turn+flop CFV nets | **−63 ± 40 mbb/g** (Supremus' reimpl.) | 175 core-years turn data |
| Libratus (2017) | blueprint + nested safe subgame solving, exact cards | superhuman vs pros | 25M core-hours |
| ReBeL (2020) | self-play RL + search over public belief states | superhuman | 4.5B samples, ~2M GPU-hours |
| Supremus (2020) | DeepStack + better nets/menus/iterations | **+176 ± 44 mbb/g** | 20M turn + 50M river + 5M flop |
| RL-CFR (2024) | learned per-state bet-size abstraction | **+84 ± 17 mbb/h** | on top of a ReBeL-class base |
| GTO Wizard AI (2022/26) | real-time deep + equilibrium | **+19.4 ± 4.1 bb/100** | current public bar |
| TurboReBeL (2025) | ReBeL + single-solve/multi-iteration targets | ≈ReBeL at **450× fewer samples** | **the enabler for desktop scale** |

Four load-bearing conclusions:

1. **Every superhuman agent re-solves in real time with exact cards.** Card
   abstraction survives only inside offline blueprints and inside *value function
   inputs* — never in the played strategy.
2. **Net quality is a hard gate, not a dial.** A DeepStack-grade architecture with
   insufficient net quality **loses to Slumbot by 63 mbb/g**. Supremus's +176 came
   from the same architecture with 2× turn data, a new river net, 8× solver
   iterations, and ~2× richer bet menus. So each net must clear a gate before it is
   allowed into serving — a half-trained net makes the agent *worse* than the
   blueprint it replaces. This is exactly what your v0 A/B (−65 bb/100) measured, and
   it agrees with the literature rather than contradicting it.
3. **Action abstraction matters at play time, cheaply.** Supremus's first-action menu
   is `F, C, 0.33, 0.5, 0.75, 1.0, 1.25, 2.0, A`. Your live re-solve menu is
   `(0.33, 0.75, 1.5, 2.5)` with cap 3, and the *datagen* menu is `(0.5, 1.0)` cap 2.
   Search trees are small; spend nodes there, not in the blueprint.
4. **LBR is the cheap exploitability probe that actually discriminates.** It crushed
   ACPC-class abstraction agents for thousands of mbb/g; resolving agents beat it
   (Supremus +951, DeepStack-reimpl +536). A correct LBR is the fastest way to know
   whether the current agent is even in the game.

---

## 3. Target architecture

```
                     played strategy: EXACT 1,326 combos, no card abstraction
  ┌──────────┬──────────────┬──────────────┬───────────────┬──────────────────┐
  │ street   │ solve         │ depth limit  │ horizon priced by │ latency target │
  ├──────────┼──────────────┼──────────────┼───────────────┼──────────────────┤
  │ preflop  │ offline, once │ flop         │ flop net       │ 0 (table lookup) │
  │ flop     │ real time     │ turn         │ turn net       │ ≤ 8 s            │
  │ turn     │ real time     │ river        │ river net      │ ≤ 5 s            │
  │ river    │ real time     │ showdown     │ exact terminal │ ≤ 3 s            │
  └──────────┴──────────────┴──────────────┴───────────────┴──────────────────┘
        continual re-solving: own range from the policy ACTUALLY played,
        opponent CFV constraints via the safe re-solve gadget (already built),
        observed off-tree bet sizes inserted into the tree (nested solving)
```

Key properties:

- **Preflop card abstraction is lossless.** 169 isomorphism classes *are* the exact
  preflop information partition. So a preflop table solved against a flop-net horizon
  is not an approximation of the preflop strategy — it is exact up to net error. The
  blueprint's legitimate final role is precisely this.
- **No blueprint in the played postflop strategy at all.** Ranges come from the
  actual chain of solutions, which structurally removes the self-range inconsistency
  that killed v1 search.
- **Value nets never touch the strategy space.** They are consulted only at horizon
  terminals, per player, at ~1,000-cluster resolution, with an explicit zero-sum
  projection layer (already implemented in `cfv/model.py:zero_sum_project`).
- **A river net is a departure from DeepStack and a necessity here.** DeepStack
  solved turns to the end of the game; that is what made turn datagen cost 175
  core-years. Pricing the turn's horizon with a river net makes each turn solve
  ~40× cheaper and is what brings turn datagen onto a 3060. Supremus validated a
  river net (50M samples) independently.

### Interim states are playable

The migration is street-by-street, and there is a working, gate-tested agent at every
step. The blueprint's role shrinks from the bottom up:

| stage | preflop | flop | turn | river |
|---|---|---|---|---|
| today | blueprint | blueprint | blueprint | blueprint |
| P1 | blueprint | blueprint | **exact resolve** | **exact resolve** |
| P4 | blueprint | **exact resolve (turn net)** | exact resolve | exact resolve |
| P5 | **exact 169 table (flop net)** | exact resolve | exact resolve | exact resolve |

---

## 4. Feasibility budget on the 3060

The whole plan lives or dies on subgame-solve throughput. Current measured points:
bucketed turn solve ≈ 3 s / 500 iters (graph-captured); eager flop solve ≈ 7 s;
exact-river resolve ≈ 1.9 s mean. Required throughput and how to get there:

| dataset | samples wanted | solves needed | per-solve budget | GPU-time |
|---|---|---|---|---|
| river net | 5–10 M | 20–50 k × T | ≤ 0.05 s | ~0.5 day |
| turn net | 2–5 M | 20–50 k × T | ≤ 0.5 s | ~0.5–1 day |
| flop net | 1–3 M | 10–30 k × T | ≤ 2 s | ~1–2 days |
| preflop aux | 1755 isomorphic flops × situations | enumerate | — | hours |

`× T` is the **TurboReBeL multiplier**: one CFR solve emits one sample per iterate
instead of one sample total. Fix the subgame policy to the CFR *average* σ̄, then
price each intermediate belief β₁…β_T against σ̄ with a frozen evaluation pass. You
already have that primitive — `VectorCFR._iterate(frozen_average=σ̄, frozen_player=None)`
is exactly it, and `cfv/dataset.py:129` already calls it. One 100-iterate solve
becomes ~100 training rows at a few percent extra cost. Diversity comes from the
number of *solves* (distinct boards/pots/ranges), so target ≥20k solves per net and
take T for free.

Three throughput levers, in order of payoff:

1. **Situation batching (the big one).** Small subgame trees leave the 3060 almost
   idle — the solve is kernel-launch bound, not FLOP bound. `_iterate` already folds
   a board axis into the combo axis (`width = batch * NUM_COMBOS`). Extend the same
   trick to a *situation* axis: N different root ranges/pots sharing one tree
   topology. `cfv/dataset.py:snap_pot` already quantizes pots onto a grid, so
   same-tree batching is natural. Expect ~10–30× datagen throughput.
2. **Depth-limit everything during datagen.** Turn solves with a river-net horizon
   never build river subtrees (~40× fewer nodes). Flop solves with a turn-net horizon
   likewise.
3. **Targeted kernel work only after measuring.** fp16/bf16 reach vectors,
   fusing the per-level gather/scatter, and a Triton kernel for the showdown
   prefix-sum are the candidates. Do not rewrite in CUDA C++ speculatively —
   `nsys`/`torch.profiler` first.

**Total datagen ≈ 3–7 GPU-days on the 3060, if per-solve budgets are hit.** That is
the difference between this plan and the 2017 papers. If those budgets slip by more
than ~3×, rent GPU: ~200–400 hours of 4090/A100 (≈ $150–600) collapses the datagen
phases to days and is the single highest-leverage money in the project.

---

## 5. The plan

Each phase has a **deliverable**, a **gate**, and an honest **effort** estimate.
No phase is allowed to ship into serving without passing its gate. The standing
NULL-test rule from `STATUS.md` §4 applies to every new instrument.

### P0 — Honest measurement (BLOCKING, ~1 week)

Nothing else in this document is worth doing before this. You cannot evaluate an
architecture change with an instrument that reads scripted-bot winnings.

1. **Repair the Slumbot harness** (`backend/eval/slumbot.py`).
   - Replace `client_incr()`'s 4-way collapse with the serving agent's real action
     set, including its actual raise fractions; make the mapping shared with normal
     serving (this is the *same* defect class as the Phase 4 projection bug — one
     translation policy, one code path).
   - Handle Slumbot's exact protocol edge cases: `b<total>` is street-cumulative,
     all-in detection, the 200bb/20,000-chip/50-100-blind geometry.
   - Session resume + a persistent JSONL hand log (so a 20k-hand run survives
     restarts and can be re-analysed offline).
   - Add a **null arm**: play a fixed always-fold/always-call policy and assert the
     measured rate matches the analytic value. This is the harness NULL test.
2. **Run 20,000+ hands vs Slumbot** with AIVAT on, at 200bb (Slumbot's native depth).
   Report raw and AIVAT bb/100 with CIs. *This number is the project baseline.*

   **Precision: see §9 P0.2 for the MEASURED figures**, which superseded two
   successive estimates in this document. Short version: σ = 16.34 bb/hand at
   200bb, so 20k hands gives ±227 mbb/h and even 300k gives ±58. AIVAT chance
   variates are off (5.4% measured variance reduction, a net loss per unit
   wall-clock). Slumbot is a milestone anchor; LBR and the duel gate are the
   working instruments. Wall-clock ≈ 3.1 s/hand, so budget ~14–17 h for 20k in
   the background.
3. **Rebuild LBR properly** (`backend/eval/lbr.py`): exact 1,326-combo opponent
   belief tracking, full-starting-stack baseline, the serving GPU agent (not the
   retired CPU one), and a real bet-size probe set. Validate it on a deliberately
   exploitable agent (always-call, always-min-raise) where the exploit is analytic.
4. **Freeze the current champions as permanent baselines** and record their Slumbot
   and LBR numbers in `STATUS.md`.

**Gate:** a Slumbot number with a CI, a null-tested harness, and an LBR number that
correctly ranks three deliberately-broken reference agents.
**Expect:** the honest Slumbot number to be negative, possibly by hundreds of mbb/h.
That is not a failure — it is the first true datum in the project.

### P1 — Exact-card continual resolving, turn + river (~2–3 weeks)

Delete the bucketed subgame resolver as the search path and generalize the Phase 4
exact-river machinery upward to the turn.

1. **Finish the Phase 4 projection repair** already scoped in
   `TRAINING_QUALITY_OPTIMIZATION_PLAN.md` §Phase 4 (one shared translation policy,
   a complete projected baseline policy, safe defaults instead of aborts, mismatch
   logging, plus shallow-stack/all-in/raise-cap/off-tree regression tests). The 1.32%
   fallback rate must go to 0%.
2. **`ExactTurnSampler`** — identity 1,326 buckets on a 4-card board, mirroring
   `ExactRiverSampler`. Turn+river exact solve; measure VRAM and latency. Budget:
   turn tree with a rich menu ~5–20k nodes × 1,326 × A × 4 B × 2 tables ≈ 1–3 GB.
   If it doesn't fit, cap postflop raises before coarsening cards — **never trade
   card exactness for tree depth again.**
3. **A real continual-resolving session object.** One object owns, for the whole
   hand: both players' exact per-combo ranges, the opponent CFV vector, and the
   chain of solutions actually played. Own-range updates come from the solution that
   actually chose each move (never from the blueprint). Opponent updates are Bayesian
   on exact per-combo likelihoods. This is the structural fix for the diagnosed
   self-range inconsistency, and it replaces `gpu_subgame.gpu_blueprint_range`.
4. **Nested solving for off-tree sizes** (Libratus): insert the observed bet fraction
   into the re-solve tree rather than translating it away. Phase 4 already does this
   retrospectively; make it the general rule.
5. **Richer live menus.** Move the search menu toward Supremus:
   `0.33, 0.5, 0.75, 1.0, 1.5, 2.0` + all-in on the first action, `0.25, 0.5, 1.0`
   + all-in on later ones, tuned to the latency budget.

**Gate:** vs Slumbot (P0 harness, ≥10k hands, AIVAT) the P1 agent must beat the
frozen blueprint champion with a CI clearing zero; LBR must not regress; zero
resolve fallbacks; latency inside budget; model-vs-itself null run centered at zero.
**Expected gain:** this is where the first large jump should appear — exact-card
turn+river play against a 30-bucket-river blueprint is a genuine information edge.
This is also the point at which the agent stops being "a blueprint" and becomes
"Libratus-lite".

### P2 — Solve throughput (the enabler, ~2 weeks, overlaps P1)

1. Profile a turn solve and a flop solve (`torch.profiler`, then `nsys`). Publish a
   table: launch overhead vs bandwidth vs compute, per phase of `_iterate`.
2. **Situation batching** (§4 lever 1) — an N-situation axis on top of the existing
   board axis, with graph capture retained.
3. **fp16/bf16 reach and value tensors** with fp32 regret accumulation. The
   float32-saturation guard in `_discount()` shows precision is already being
   managed explicitly; keep regrets fp32.
4. **CFR variant upgrade:** implement **DCFR+** (Supremus' linear `max(0, t−100)`
   averaging weight) and **PDCFR+** (predictive/optimistic regret matching). Cheap
   code, measurable convergence win per iteration, and the same regret tables.
   Validate on Kuhn/Leduc against the existing exploitability harness before trusting
   it on HUNL.
5. Publish a throughput scoreboard (solves/sec by street and menu) and keep it as a
   regression test — datagen budgets depend on it.

**Gate:** ≥10× turn-solve throughput at unchanged solution quality (identical
exploitability on a fixed Leduc/Kuhn control and unchanged duel result on a fixed
HUNL control).

### P3 — Value-net stack, built bottom-up (~4–6 weeks + GPU-days)

Each net is trained, gated, and only then used to price the next street's horizon.
Shared design rules, all of which are corrections to v0:

- **I/O resolution ~1,000 clusters per player per street** (DeepStack/Supremus),
  not 169. Cluster per canonical board. Include an explicit board encoding
  (52-dim one-hot, or per-card embeddings) *in addition to* the clusters — cheap,
  and strictly more information than the papers had.
- **Depth-agnostic normalization:** express pot and all commitments in units of
  effective remaining stack, so one net serves 100bb and 200bb (and everything
  between). Feed pot fraction and SPR explicitly.
- **Targets from 2,000–4,000 DCFR+ iterations per player** (Supremus), never 500.
- **Enumerate runouts** for target CFVs (all 44/48 cards), never 4 samples. This was
  the identified binding constraint.
- **TurboReBeL multi-iterate emission:** one solve → T rows.
- **Range distributions must match deployment**: mix the recursive pseudo-random
  generator (DeepStack supplement) with ranges harvested from actual resolving play.
- **Zero-sum output projection** (already implemented) plus **target repeatability
  measurement** before fitting: solve the same situation with different seeds and
  report target variance. If target noise exceeds the value differences that change
  actions, no amount of data helps — that is the v0 lesson, quantified.
- **Report action-changing error, not only MAE**: what fraction of decisions flip
  when the horizon is priced by the net vs by an exact solve?

**P3a — River net.** Root: 5-card board, both exact ranges, pot. Targets are exact
(river solves terminate at showdown, no runouts to enumerate) — so this is the
cleanest, cheapest, highest-signal net, and the right one to debug the whole pipeline
on. 5–10M rows.
*Gate:* net-priced turn solves must produce the same action as exact turn+river
solves on ≥95% of a held-out decision set, and a turn-resolve A/B with the net
horizon must not regress vs exact-river resolving.

**P3b — Turn net.** Turn solves use the P3a river net at the horizon (40× cheaper).
2–5M rows, ≥20k distinct situations.
*Gate:* same structure, one street up: net-priced flop solves must match
turn-net-free reference solves on a held-out decision set.

**P3c — Flop net.** Flop solves use the P3b turn net. 1–3M rows.
*Gate:* as above, plus the P4 end-to-end gate below.

**P3d — Preflop auxiliary values.** Enumerate the **1,755 strategically distinct
flops** (suit isomorphism) rather than DeepStack's 22,100, averaging flop-net CFVs.
This is hours of compute, not days.

### P4 — Flop resolving (~1–2 weeks after P3b)

Replace blueprint flop play with a depth-limited exact-card flop re-solve, turn-net
horizon, safe gadget seeded by the preflop/flop CFVs carried in the resolving session.
`DepthLimitedCFR` + `HORIZON` + `NetEvaluator` already exist; they need the
1,000-cluster net, the exact-card sampler, and the session integration.

**Gate:** Slumbot A/B (≥10k hands, AIVAT) with a CI clearing zero vs the P1 agent;
LBR non-regression; latency ≤ 8 s at the 99th percentile.
**This is the phase with the largest downside risk** — see §7. If the flop net cannot
clear its gate, P1's turn+river agent remains the shipped player and the flop stays
on the blueprint. That is an acceptable, strong, resting state.

### P5 — Exact preflop, architecture complete (~1–2 weeks)

Solve the preflop tree (169 lossless classes, rich menu, high iteration count)
against a flop-net horizon; ship it as the preflop table. At this point the entire
card abstraction is gone from the player and the DeepStack architecture is complete:
preflop table → flop resolve → turn resolve → river resolve, exact cards throughout.

**Gate:** full-stack Slumbot run (≥20k hands) + LBR + head-to-head vs every retained
champion.
**Target:** beat Slumbot with a CI clearing zero. Then push toward the published
bars: Supremus +176 mbb/g, GTO Wizard +194 mbb/g.

### P6 — Supremus-grade and beyond (open-ended)

Only once P5 is green, in descending expected value:

1. **More iterations at play time** (Supremus: 1,000 iterations in 0.8 s — 6× faster
   than DeepStack). Raw solve speed converts directly into strength here.
2. **Richer menus** everywhere the latency budget allows.
3. **Learned action abstraction (RL-CFR, +84 mbb/h over a fixed menu).** This is
   the correct home for the retired Phase 3 work — per-state *dynamic* sizing chosen
   by a learned policy inside a small search tree, not a static profile baked into a
   giant blueprint tree.
4. **AIVAT decision variates** — now unblocked, because P3 provides the dependable
   value function they were always waiting on. Cuts evaluation cost for everything
   after.
5. **ReBeL-style self-play refinement**: regenerate net targets from the agent's own
   resolving play, iterating value net and policy together (with TurboReBeL's
   emission trick, this is affordable here).
6. **Multi-depth**: with stack-normalized nets, one agent covers all stack depths;
   retire the per-depth champion machinery.

### P7 — Robustness (continuous, from P1 onward)

- LBR-hardening: run LBR with progressively nastier probe menus; each new exploit
  found becomes a regression test.
- Off-tree stress: opponents using sizes, limps, and stack depths never seen.
- Latency degradation ladder: iteration count as a function of remaining time budget,
  with a *proven-safe* fallback (the safe gadget bounds loss vs the baseline policy).
- Keep the `duel.py`/`gate.py` promotion discipline exactly as-is. It works.

---

## 6. What to stop doing, and what to delete

| Item | Action | Why |
|---|---|---|
| More blueprint iterations at any depth | **Stop** | Plateau measured twice at both depths |
| Potential-aware / v3 recursive abstraction (planned Phase 2) | **Retire** | Improves a component that stops being the player; the 384/192 configuration is still ~30× too coarse to matter |
| `phase3-static-v1` action profiles in the blueprint tree | **Retire** | Regressed −94.92 bb/100; the idea belongs in search (RL-CFR, P6.3) |
| Bucketed turn/river subgame search (`gpu_subgame.py` as a search path) | **Delete** | No information edge by construction; superseded by exact-card resolving |
| 169-bucket CFV nets (`NetEvaluator.BUCKETS`) | **Delete** | Contradicted by DeepStack/Supremus; the measured failure mode |
| `cfv/dataset.py` v0 datagen (500 iters, 4 runouts, 7.7k samples) | **Rewrite** | Under-resourced on every axis simultaneously |
| "Styles field" absolute bb/100 as a strength claim | **Stop quoting** | Not evidence about equilibrium quality; keep only as a smoke test |
| Multi-stack per-depth champions | **Retire after P1** | Stack-normalized nets (§8.3) make this obsolete; keep the frozen champions as measurement baselines only |

Keep and extend: `VectorCFR` kernels, `BettingTree`, `GraphRunner`,
`DepthLimitedCFR`/`HORIZON`, `safe_subgame.GadgetCFR`, `exact_river.py`,
`duel.py`/`gate.py`, `aivat.py`, the serving/frontend/scraper stack.

---

## 7. Risks and honest expectations

| Risk | Severity | Mitigation |
|---|---|---|
| **The honest Slumbot number is very negative** | high likelihood, low harm | This is information, not damage. It reframes every prior "+130 bb/100" claim and makes future gates real. |
| **Net quality never reaches usefulness** (the DeepStack-reimpl. −63 mbb/g failure mode) | high | Bottom-up gating: river net first (exact targets, cheapest), each net gated on *action-changing* error before use. Measure target repeatability before fitting. Fallback: ship P1 (turn+river exact resolving on a blueprint preflop/flop) — a genuinely strong resting state. |
| **Solve throughput misses budget by >3×** | medium | Rent GPU for datagen phases (~$150–600). Decide after P2's scoreboard, not before. |
| **12 GB VRAM caps exact-card turn trees** | medium | Cap postflop raises and menu width before ever coarsening cards; per-street shards already exist; stream cold tables to the 32 GB host RAM. |
| **Latency incompatible with live play** | medium | Nets exist precisely to cut depth; add the P7 degradation ladder. If live play needs sub-second decisions, the served policy becomes a distilled network trained on the resolver's own output — a separate, later project. |
| **Continual-resolving bugs are subtle and silent** | high | The project has already been burned three times. Every component gets an oracle test: exact-card resolve on a tiny game vs a full-tree solve; `ShowdownOracle` equivalence (already the pattern); Kuhn/Leduc exploitability controls; independent cross-check of postflop solutions against an open-source solver (e.g. TexasSolver) on fixed spots. |
| **Scope: this is a 3–6 month programme** | certain | Phase gates are designed so each one leaves a stronger shippable player than the last. |

**Calibrated expectation.** P0 tells you where you stand. P1 is the highest
confidence gain in the plan (exact-card turn+river vs a 30-bucket river). P3–P5 are
the ceiling-raiser and carry real risk of not clearing their gates on this hardware.
A realistic good outcome is a player that beats Slumbot by a clear margin; matching
Supremus/GTO Wizard (+176/+194 mbb/g) would require net quality at the upper end of
what 3–7 GPU-days can produce, and is the stretch target, not the plan of record.

---

## 8. Decisions taken (2026-07-27, user)

1. **Latency: ≤8 s/decision now, sub-second later — resolver first, distillation
   after.** The resolver is the reference player and the sole strength target through
   P5. Live scraper autoplay is served by a distilled policy network (new **P8**
   below), which cannot start before P4/P5 exist because it can only be as good as
   what it copies. Consequence for P1–P5: **do not compromise the resolver's strength
   for latency.** Keep the P7 degradation ladder for the ≤8 s path only.
2. **Rented GPU: decide after P2's throughput scoreboard.** Budget locally for now;
   rent (~$150–600 of 4090/A100) only if P2 misses the §4 per-solve budgets by more
   than ~3×. P2's scoreboard is therefore a **decision gate**, not just an
   optimization report — it must publish measured solves/sec per street and menu, and
   the implied GPU-days for each P3 dataset.
3. **One stack-normalized agent.** All nets and resolves normalize pot and
   commitments by effective remaining stack, with pot fraction and SPR as explicit
   inputs. Consequences:
   - P3 trains **one** net per street covering all depths (halves net-training cost
     versus per-depth nets).
   - The per-depth champion machinery (`--stack-bb` artifact directories,
     `MultiStackBlueprintAgent` depth routing, per-depth promotion gates) is retired
     once P1 ships — but **not before**: the frozen 100bb/200bb champions remain the
     P0/P1 measurement baselines.
   - P0 measures at 200bb (Slumbot's native depth) and additionally records the 100bb
     champion's LBR number, so both retired champions have a permanent datum.
   - Every net's validation split must be stratified **by stack depth** as well as by
     board/pot/range-source/seed, or depth-agnosticism is assumed rather than measured.

### P8 — Distillation for live play (starts only after P4 or P5 is green)

Train a fast policy network on the resolver's own decisions (state → action
distribution), sampled over self-play and resolving play, and serve *that* in the
scraper autoplay path.

- Inputs: the same stack-normalized public state plus the agent's own exact hand.
- Targets: the resolver's full mixed strategy at that infoset, not its sampled action —
  the mixture is the whole point, and a deterministic policy is trivially exploitable.
- Strictly weaker than the resolver, by construction. Gate it on both a head-to-head
  duel against the resolver (measure the loss, in bb/100, that live play costs) and
  an LBR run (distillation is where exploitability creeps back in).
- Keep the resolver as the browser/study player regardless of how good P8 gets.

---

## 9. Implementation log

### P0.1 Slumbot harness — DONE (2026-07-27)

`backend/eval/slumbot.py` rewritten. The defect that mattered: the old
`client_incr()` reimplemented the agent's action mapping and collapsed it into a
4-way choice with a hard-coded 0.5-pot raise, so any number it produced described
a crippled agent. It now calls `agent.select()` + `agent.execute()` — the same
path `duel.py` and live serving use — and *derives* the wire token from the
engine event the agent produced. Also:

- **Position is protocol-derived,** not read from `client_pos`. An empty action
  string on `new_hand` means we act first, i.e. we hold the button. This was
  worth doing: public sources contradict each other, and a live response
  (`{'action': 'b200', 'client_pos': 0}`) settles it as *client_pos=0 = big
  blind* — the opposite of what one reference client's documentation says. The
  harness reports the observed mapping so it can never be silently wrong.
- **Blind double-posting bug found and fixed.** Seating the button by calling
  `new_hand()` a second time posts blinds twice out of already blind-reduced
  stacks (19,850 instead of 19,950/19,900), corrupting stack depth and SPR on
  every out-of-position hand. `button_offset` now goes in at construction, and
  the constructor asserts `sum(stacks) + pot == 2 * STACK`. This was latent in
  the original harness — the same accounting-bias family as the +75 bb/100
  artifact.
- **Broken hands are excluded, not folded and banked** (the old path folded and
  still counted the winnings, biasing the mean toward losses).
- **Deck sanitization:** the engine's phantom deals can no longer duplicate a
  real Slumbot card.
- Per-hand JSONL logging, session-token resume, retry/backoff, and per-position
  reporting.

`backend/eval/null_agents.py` (new): `ScriptedAgent` with always-fold /
always-call / always-min-raise / always-all-in, plus the analytic values. Shared
with the LBR validation.

**NULL tests: `tests/test_slumbot_harness.py`, 12 tests, all passing.** A
`FakeSlumbot` speaks the real wire protocol over a real rules engine, so
`play_match` runs its true code path with only the transport swapped — the
harness is fully verifiable offline, with no API calls. The anchor is exact:
always-fold from the button reads **−50.0000 bb/100 with zero variance**. Other
tests cover protocol-derived position (inverting the `client_pos` convention
changes nothing), 250 random-agent hands with zero exclusions and zero board
desyncs, street-cumulative bet encoding, all-in encoding, deck sanitization, and
the exclusion-not-banking rule.

**Integration check:** the real `MultiStackBlueprintAgent` (100bb@30k +
200bb@118k) played 150 hands against the fake server with **0 exclusions, 0 board
desyncs**.

### P0.1b AIVAT defect — FIXED (2026-07-27)

`ChanceCorrector.observe` scaled corrections by `engine.pot`, which the engine
zeroes when awarding the pot. An all-in run-out deals the remaining streets and
awards inside one call, so the correction was silently multiplied by zero on
exactly the highest-variance hands AIVAT exists to correct. Now falls back to
`last_pot`. `tests/test_aivat.py` still green.

### P0.3 LBR — UPGRADED AND VALIDATED (2026-07-27)

The docs' §4 critique of LBR was already stale: the Phase 1 work had fixed the
full-starting-stack baseline, the GPU serving agent and the blueprint belief
updates. The two real gaps are now closed:

- **A real probe menu.** LBR tried exactly one raise size (pot). It now sweeps
  `0.25, 0.5, 0.75, 1.0, 1.5, 2.0` pot plus all-in, deduped after clamping. A
  one-size probe understates exploitability, because a real exploiter picks the
  size the victim's abstraction handles worst.
- **Bucket memoization** (`_board_buckets`), since the menu multiplied
  fold-response queries ~7× per decision and each recomputed the same board
  bucketing.

**Validation: `tests/test_lbr_validation.py`, 3 tests, all passing.** A real
agent's strategy table is replaced by a `ConstantStrategy`, so every LBR path
(node location, translation, range posterior, fold response) runs unmodified
against an opponent whose exploitability is known. The anchor is exact: against
always-fold, LBR reads **+75.0000 bb/100 with zero variance** (button +100,
blind-defender +50, averaged over the duplicate pair). Plus: a calling station
must measure both >+100 bb/100 and strictly more exploitable than the champion —
a probe that cannot separate those two is useless as a guard.

Also: `--hands 500 --gpu` is now the script-GUI Slumbot default (it was measuring
the *retired CPU* blueprint), and a one-click "Slumbot harness NULL check" entry
was added.

### P0.4 First honest exploitability baseline — MEASURED (2026-07-27)

Both serving champions probed with the repaired LBR, search off, 400 duplicate
pairs / 800 hands each. Positive means exploitable.

| champion | LBR bb/100 | 95% CI | fallback |
|---|---:|---:|---:|
| 200bb scalar@118k | **+291.23** | [+79.10, +503.36] | 1.11% |
| 100bb histogram@30k | **+137.58** | [−3.57, +278.73] | 0.58% |

So the probe wins roughly **2,910 mbb/hand** against the 200bb agent and
**1,376 mbb/hand** against the 100bb agent. Read against the literature:

| | LBR result |
|---|---:|
| Supremus | LBR **loses** by 951 ± 96 mbb/g |
| DeepStack reimplementation | LBR **loses** by 536 ± 68 mbb/g |
| 2016 ACPC-class abstraction agents | LBR **wins** by thousands of mbb/g |
| **this champion** | **LBR wins by ~2,910 mbb/hand** |

The champion is in the 2016-abstraction-agent class, not the
resolving-agent class. Three qualifications, all of which make the picture worse
rather than better:

1. **LBR is a lower bound.** It is a restricted response over a fixed probe
   menu, so true exploitability is ≥ 291 bb/100. A stronger probe finds more.
2. **The probe's continuation model is pessimistic for the probe**, not for the
   victim: it prices a call as "showdown at current equity", which understates
   what a real exploiter extracts on later streets.
3. The instrument reads its analytic anchors exactly (§P0.3), so the readings are
   not harness artifacts.

**Two things this measurement does NOT establish**, stated because it would be
easy to over-read:

- **It does not credit the histogram abstraction.** The 100bb figure is half the
  200bb one, but depth and abstraction changed together: deeper stacks mean more
  betting rounds and more room for a probe to extract, so depth alone could
  account for the gap. The CIs also overlap heavily. Attributing the difference
  to bucketing would need both models probed at the same depth.
- **LBR precision at 400 pairs is only ±140 bb/100** — the 100bb CI includes
  zero, so at this sample size the probe cannot even establish that the 100bb
  champion is exploitable at p<0.05, despite a +137 point estimate. This
  corrects an over-confident earlier note: LBR is *not* a 4-minute fine-grained
  diagnostic. It is a 4-minute detector of changes worth *hundreds* of bb/100,
  which is the size P1 should produce. For inner-loop use on smaller changes,
  run ≥1,600 pairs (≈16 min) for ±70 bb/100, and keep the duel gate for anything
  finer.

This retires the last defence of the current architecture. It also reframes the
"~+130 bb/100 at 100bb / ~+290 at 200bb" styles-field numbers in `STATUS.md` §1:
an agent losing ~3 bb/hand to a simple local best response can still beat
scripted opponents comfortably, because scripted opponents do not probe. The
two facts are consistent, and only one of them is about equilibrium quality.

**Consequence for the plan:** P1's expected gain is now quantified as a target
rather than a hope — exact-card turn+river resolving must move this number
substantially, and LBR (cheap, 400 pairs ≈ 4 min) becomes the fast inner-loop
diagnostic for every P1 change, with Slumbot reserved for milestone anchoring.

### P0 COMPLETE: the project's first honest absolute number (2026-07-28)

**Slumbot, 20,000 hands: −18.33 bb/100, 95% CI [−40.41, +3.74].** Zero exclusions,
zero board desyncs, σ = 15.93 bb/hand, 11.6 h. Button −11.43, big blind −25.24.

The serving champion is **statistically tied with Slumbot, leaning slightly
negative**. Set beside LBR's verdict that the same agent is ≥291 bb/100
exploitable, the pair is the most useful thing measured in this project: *a
non-probing opponent cannot collect what a probe can*. Slumbot is itself an
abstraction blueprint, so two mutually exploitable agents sit near even.

Instrument validations from the run: the `client_pos` mapping came out exactly
balanced (10,000 each way), and **99.0% of wall-clock was spent waiting on the
API** (41,253 s of 41,653; agent 371 s, mirroring 11 s), so the harness adds
nothing measurable — and that is why it ran alongside GPU datagen without
interference. Precision landed at ±22.1 bb/100 against the ±22.7 predicted from
σ, so the sample-size model in this document is calibrated.

**The discipline paid for itself.** The 400-hand screen read **+84.88**; the truth
is **−18.33** — a 103 bb/100 swing, larger than any effect this project is
hunting. Reporting that screen as a win would have poisoned everything after it.

### P0.2 The 400-hand screen (superseded by the 20k run above) (2026-07-27)

400 hands, real API, serving champion (200bb scalar@118k), search off,
**0 exclusions, 0 board desyncs**, 1,236.9 s (**3.09 s/hand**).

| | bb/100 | 95% CI |
|---|---:|---:|
| raw | **+84.88** | [−75.30, +245.05] |
| AIVAT | +91.00 | [−64.75, +246.75] |
| button (200 hands) | +66.50 | [−162.83, +295.83] |
| big blind (200 hands) | +103.25 | [−120.96, +327.46] |

`client_pos` mapping confirmed on a balanced 200/200 split: **client_pos=0 = big
blind, client_pos=1 = button.**

**Do not read the +85 as a win.** The CI spans zero by ±160 bb/100, and the
outcome distribution says plainly what happened: **median −0.5 bb**, worst hands
−60/−48/−45, best hands +200/+200. The mean is two won all-in pots. The
per-100-hand trajectory (−48, +0.9, −1.2, +84.9) is the same story. The honest
statement is *"400 hands, indistinguishable from a tie."*

#### Three corrections this forces

**1. Slumbot is a much coarser instrument than planned.** Measured spread is
**σ = 16.34 bb/hand** at 200bb — two to three times the 5–8 bb/hand the §P0
table assumed, because a single all-in at 200bb depth swings ±200 bb. Corrected:

| hands | 95% CI (measured σ) |
|---:|---:|
| 400 | ±1,602 mbb/h |
| 20,000 | **±227 mbb/h** |
| 50,000 | ±143 mbb/h |
| 300,000 | ±58 mbb/h |
| 1,000,000 | ±32 mbb/h |

So 20k hands resolves to ±227 mbb/h, not the ±64–102 previously tabulated. Even
300k hands cannot cleanly separate Supremus' +176 from GTO Wizard's +194.
Slumbot is a **milestone sanity anchor, not a measurement instrument**, until
AIVAT decision variates (P6.4) exist. LBR and the duel gate do the real work.

**2. AIVAT chance variates are switched OFF for Slumbot sessions.** Measured
variance reduction on real data: **5.4%** — a third of the 15.6% seen on the
local fixture, because the value function (hero equity vs a *uniform* opponent)
poorly predicts realized luck against a real opponent's actual ranges. Per unit
wall-clock it is a net loss at any plausible cost: even at 0.6 s/hand it widens
the CI by 8%, and at 1.5 s/hand by 36%, because the hands it buys are worth more
than the variance it removes. Revisit only with a better value function — which
is precisely what P3 produces.

**3. LBR and Slumbot disagree, and both are right.** LBR says the champion is
≥291 bb/100 exploitable; Slumbot says roughly even. These are different
quantities and the gap is the single most useful thing measured today:

> Slumbot is itself a non-probing abstraction agent. Two mutually exploitable
> agents can sit near even head-to-head, because neither one hunts the other's
> abstraction holes. Head-to-head result against a fixed opponent is **not**
> distance from equilibrium.

**Consequence for the plan's success criteria.** "Beat Slumbot" is necessary but
not sufficient, and it is a weak target: it can be satisfied by an agent that a
real exploiter dismantles. **LBR exploitability is the better north star**, with
Slumbot as the external sanity check that we have not fooled ourselves. Every
phase gate from P1 onward should lead with LBR and cite Slumbot second.

slumbot.com reachable. A 400-hand validation session against the real API is
running with the serving champion, search off, logging to
`backend/data/slumbot/baseline-400.jsonl`. Purpose: confirm the live protocol
end-to-end and settle the `client_pos` mapping empirically. **400 hands cannot
measure strength** (§P0 precision table: expect roughly ±300–500 mbb/h at that
count) — it is an engineering screen. The 20k-hand baseline run follows once it
is clean.

### P1.1 Phase 4 projection repair — DONE (2026-07-27)

The defect: `_project_blueprint` assumed the exact resolver tree and the coarse
blueprint tree keep the same actor/street topology downstream. They legitimately
diverge — the exact tree uses real stacks, a richer size menu and a live
mid-street root — so in shallow-stack, all-in, raise-cap and off-tree branches
the blueprint can terminate while the exact tree still has a decision. Any such
divergence raised, which cost the whole hand's exact-card resolving and produced
all 25 fallbacks (1.32%) in the 3,000-pair confirmation.

The repair, in `backend/search/exact_river.py`:

1. **Divergence detaches instead of aborting.** A node with no usable blueprint
   counterpart takes the serving agent's safe-default policy
   (`_safe_default_policy`, mirroring `GpuBlueprintAgent._safe_default`), and its
   descendants inherit that. The projection is only the *baseline* that prices
   the safe gadget's opponent opt-out, so a locally-approximate baseline in a
   rare subtree is vastly better than losing the resolve. The four former abort
   paths are now four distinguishable detach reasons.
2. **One translation policy, shared with serving.** `_map_action` disagreed with
   `GpuBlueprintAgent._translate_event` in three ways, all fixed: check/call fell
   back to *fold* instead of the smallest legal raise; an untranslatable raise
   always became an all-in (serving requires the size to exceed 1.5 pot first);
   an all-in was remapped onto the largest sized raise.
3. **The mismatch is logged with actionable detail** — node, actor, street, pot,
   legal mask and stack depth on the exact side, plus kind/actor/street/pot on
   the blueprint side, for the first five root causes per solve, with counts by
   reason and a detached-fraction summary in every solve's diagnostics.

**Tests: `tests/test_exact_river_projection.py`, 8 passing.** Six unit tests pin
the translation precedence against serving's. Two robustness tests project 36
(exact tree, blueprint root) combinations spanning deep, shallow, sub-blind,
raise-capped, asymmetric-stack and off-tree roots, asserting the projection never
raises and always yields a valid distribution — zero mass on illegal actions or
blocked combos, sums to 1 on valid combos. 594 of 666 decision nodes detached
across that adversarial set, i.e. **the old code would have aborted almost all of
it**. The suite also asserts that detachment actually occurs, so it cannot
silently stop testing the repair.

**Engineering screen: `tools/phase4_projection_screen.py`** (new). It reports
fallbacks and detachment *together*, because zero fallbacks with heavy detachment
would be a weaker guarantee than it looks. Depth sweep, 120 hands each:

| stack | attempts | resolved | fallbacks | detach mean | detach max | fully attached | latency mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 200bb | 43 | 43 | **0** | 1.77% | 42.9% | 41/43 | 1,240 / 2,457 ms |
| 40bb | 45 | 45 | **0** | 0.00% | 0.00% | 45/45 | 1,225 / 1,936 ms |
| 20bb | 42 | 42 | **0** | 0.00% | 0.00% | 42/42 | 1,045 / 2,195 ms |

**130/130 resolves, zero fallbacks**, all latencies far inside the 6 s budget.
The gate ("zero projection failures and acceptable latency") passes.

Two findings worth keeping:

- **Divergence is driven by menu richness and depth, not shallowness** —
  correcting the working hypothesis. Detachment appears only at 200bb (2 of 43
  solves) and never at 40bb or 20bb, because shallow trees are small on both
  sides while a deep exact tree with the richer river menu creates nodes the
  coarse blueprint has no counterpart for. That matches where the original 25
  fallbacks were observed (200bb), and the 2/43 rate sits in the same band as the
  original 1.32%. So the repair is hitting the real failure mode, not a
  hypothetical one.
- **The logged cause is now fully explanatory.** Example root cause: exact node
  with pot 240bb, SPR 0.33, legal = {fold, check/call} only, versus a blueprint
  node of kind SHOWDOWN at matched 120bb. The blueprint considers the hand over
  (both all-in, betting closed) while the exact tree — built from real stacks —
  still has one player owing a call. `240 = 2 x 120` cross-validates the two pot
  readings.

Fixing that log field took two attempts and is itself worth recording:
`matched_pot` is populated only on SHOWDOWN/HORIZON nodes (a fold's amount lives
in `fold_loser_committed`) and stores the *winner's gain*, i.e. `min(committed)`,
so the pot is twice it. The naive read returns 0.0 at every decision node — a
diagnostic that always logs zero looks like data. `_node_matched_pot` now derives
it correctly and `tests/test_exact_river_projection.py::NodePotTests` pins it,
including "no decision node reports a non-positive pot" at three stack depths.

### P1.2 Exact-card turn sampler + viability measurement — DONE (2026-07-27)

`backend/search/exact_turn.py`: `ExactTurnSampler` gives every private combo its
own bucket on **both** the turn and the river, so the played turn+river strategy
carries no card abstraction. `VectorCFR` already propagates exact per-combo reach
at every node — the abstraction only ever lived in how regret/strategy rows were
indexed — so exact cards cost memory, not a new kernel.

Two properties the design leans on, both pinned by tests:

* **A turn board admits only 48 rivers**, so every deal the solver can ever see
  is enumerated and cached at construction. Sampling is a dict lookup and there
  is no per-iteration bucketing work at all (the blueprint sampler spends
  ~25-40 ms/iteration on exactly that).
* **A combo's turn bucket is its own index in every runout.** Turn regrets
  therefore accumulate per combo across river cards rather than smearing, which
  is what makes the turn strategy learnable.

`tests/test_exact_turn.py`, 10 passing: identity bucketing, validity matching
independent scoring, `C(47,2) = 1081` live combos per runout, board/river
blocking, river-independence of turn buckets, deal caching, and a real
exact-card turn+river solve whose every decision node is a normalized
distribution with zero mass on illegal actions.

#### The viability answer: memory is free, throughput is the constraint

`tools/exact_turn_probe.py` (new) sweeps menu x raise cap x depth on the 3060:

| config | nodes | turn / river decisions | tables | peak VRAM | eager s/it | graph s/it | speedup | iters in 2 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 sizes cap2 100bb | 414 | 28 / 124 | 7.7 MiB | 249 MiB | 0.122 | 0.0098 | 12.5x | **162** |
| 2 sizes cap2 200bb | 726 | 28 / 228 | 12.9 MiB | 453 MiB | 0.128 | 0.0135 | 9.5x | **116** |
| 2 sizes cap3 200bb | 1,068 | 56 / 328 | 19.4 MiB | 687 MiB | 0.121 | 0.0200 | 6.0x | 79 |
| 3 sizes cap2 200bb | 1,934 | 52 / 612 | 40.3 MiB | 1,188 MiB | 0.131 | 0.0299 | 4.4x | 49 |
| 3 sizes cap3 200bb | 3,746 | 120 / 1,160 | 77.7 MiB | 2,421 MiB | 0.154 | 0.0515 | 3.0x | 25 |
| 4 sizes cap2 200bb | 4,614 | 84 / 1,496 | 111.9 MiB | 2,781 MiB | 0.174 | 0.0620 | 2.8x | 23 |
| 4 sizes cap3 200bb | 10,602 | 248 / 3,408 | 258.9 MiB | 991 MiB | 0.284 | 0.1087 | 2.6x | 11 |

Findings:

1. **VRAM is not the constraint, and the §4 worry was misplaced.** The richest
   configuration needs 259 MiB of tables and under 3 GiB peak against 12 GiB.
   The plan's rule "cap menus before ever coarsening cards" still holds, but for
   *throughput* reasons, not memory.
2. **CUDA-graph capture is mandatory, not an optimization.** Eager mode gives
   9-18 iterations in a 2 s budget — useless. Capture buys 2.6-12.5x, and the
   speedup is largest on the smallest trees, confirming these solves are
   launch-bound rather than compute-bound. Capture costs 0.40-0.74 s once per
   solve and is charged against the budget in the table above.
3. **Exact turn resolving is viable at 2 sizes**: 116 iterations in 2 s at 200bb
   (162 at 100bb), comparable to the river resolver's working 40-80. This is the
   configuration P1 should carry.
4. **The menu/cap tradeoff is now priced.** From 2 sizes cap2: adding a third
   size costs ~2.4x iterations, a fourth ~5x, and raising the cap to 3 ~1.5x.
   Spend that budget only where an A/B says it pays.
5. Peak VRAM is non-monotonic (4 sizes cap3 uses *less* than cap2) because the
   showdown blocker kernel's 2 GB guard switches to the per-card loop on big
   trees, trading memory for time — which also explains its worse s/iteration.

**Open question for P1.3**, deliberately not assumed: 116 iterations is far from
converged by Supremus' standard (4,000/player). The claim being tested is only
that a moderately-converged *exact-card* solve beats a fully-converged
*150-bucket* one. That is what the P1 gate measures, and P2's throughput work is
the lever if it falls short.

#### Decision-level evidence: the documented draw-fold leak is fixed

`docs/RESEARCH_ROADMAP.md` records a concrete exhibit (decision log, hand #222):
7s5s on 4s 5c 6s Kc — pair + OESD + flush draw, ~19 outs, ~43% vs top pair —
**folded 83%** to a 0.7-pot turn bet needing 29%, because the scalar bucket
merged that combo-draw with static ~45%-equity hands that correctly fold to
polarized aggression. An exact-card turn solve of the same spot (400 iterations,
uniform live ranges):

| hand | fold | call | raise |
|---|---:|---:|---:|
| **7s5s (the documented hand)** | **0.032** | 0.926 | 0.042 |
| 8s7s (OESD + flush draw) | 0.000 | 0.950 | 0.050 |
| As2s (nut flush draw) | 0.124 | 0.661 | 0.215 |
| 8h7h (OESD) | 0.000 | 0.942 | 0.058 |
| Qd2h (control) | 0.945 | 0.049 | 0.005 |
| Jh3d (control) | 0.917 | 0.049 | 0.035 |
| 9d2d (control) | 0.986 | 0.011 | 0.003 |

Fold rate 0.83 → **0.032** on the documented hand. The test cuts both ways by
design: trash still folds 0.92-0.99, a discrimination gap of **+0.910**, so this
is card-level discrimination and not mere looseness. As2s prefers *semi-bluff
raising* (0.215) over the pair+draw's flat call, which is the correct structural
treatment of a nut draw versus a hand that wants to realize equity cheaply.

#### Design constraint found while reading the session code (shapes P1.3 and P3a)

In a turn-rooted tree, river decision rows are indexed by **combo only** — there
is no river-card axis — so one turn solve learns a single river strategy per
combo, *averaged over all 48 runouts*. The turn solve is therefore **card-blind on
the river**, and its river subtree acts as a crude value estimator rather than a
playable river strategy. Two consequences:

1. **P1.3 must re-solve at the river with the actual card** rather than reuse the
   turn solution's river rows. Phase 4's river resolver already does exactly
   this, so the session's job is to carry *ranges* forward into it — which is the
   real upgrade over Phase 4, where river ranges are re-derived from the
   blueprint at the street boundary instead of from the turn solves actually
   played.
2. **This is precisely the hole a river value network fills** (P3a). Adding a
   river-card axis instead would cost 48x the river rows (~580 MiB at 228 river
   decisions — affordable) but would need ~48x the samples to learn, which the
   116-iteration budget cannot supply. So the interim card-blind horizon is the
   stand-in for a river net, and P3a's priority as the *first* net is confirmed
   by an independent route.

Pinned as regression tests (`tests/test_exact_turn.py::DrawFoldLeakTests`).
**Caveats, stated because it would be easy to over-read:** ranges are uniform
over live combos rather than blueprint-tracked, so this is a mechanism
demonstration, not a strength measurement; it is one spot on one board; and the
absolute mixes would shift under real ranges. What it does establish is that
exact cards no longer inherit bucket-mates' folds — the specific mechanism the
plan claims is the ceiling. It is the same evidence standard used to validate the
histogram abstraction (hand #326: fold 97.8% → call 95.9%).

### P1.3 Continual turn+river resolving — BUILT (2026-07-27)

`backend/search/continual.py` (`ContinualSession`, `open_session`,
`resolve_decision`, `register_selected_action`) plus
`backend/search/exact_turn_resolve.py` (`resolve_turn_at`). The river resolver was
parameterized by street rather than duplicated: `_root_state`, `_config`,
`_project_blueprint` and `_blueprint_ranges` now take the street, so both
resolvers share one projection, one translation policy and one gadget driver.

**A correction to the plan's premise.** P1.3 was described as the structural fix
for self-range inconsistency, but reading the code shows Phase 4 *already* tracks
own-range correctly: `register_selected_action` records the per-combo likelihood
from the solution that actually chose the move. What was actually missing is
**range continuity across the street boundary** — Phase 4 re-derives river ranges
from the blueprint when the river card lands, which is the same falsified-history
error one street later. The session now enters at the turn and carries ranges
through into the river resolver; the blueprint is consulted exactly once, to seed.

#### A silent bug this nearly shipped

`_GadgetGraphRunner` captures a CUDA graph that reads its deal from **fixed
buffers filled once at construction**, and `_blueprint_alt_values` sampled one
deal outside its loop. Both are exactly correct for the river, where
`ExactRiverSampler` has precisely one possible deal. Reused unchanged for the
turn — which has 48 river runouts — they would have frozen a single river card for
the whole solve: **the agent would have played the turn as if it already knew the
river**, and nothing about the output would look wrong.

Both now take `resample`, defaulting to False so the validated river path stays
bit-identical, and the turn resolver passes True (refilling the graph's input
buffers per iteration, the same pattern the blueprint trainer's `GraphRunner`
uses). With resampling the opt-out values also average over 8 runouts instead of
trusting one. Pinned by
`tests/test_continual_resolving.py::ChanceResamplingTests`: >10 distinct rivers
drawn with the flag on, <=1 with it off.

#### A pre-existing broken test suite, found and fixed

`tests/test_safe_subgame.py` — the *only* coverage of the safe gadget, which both
resolvers depend on — had been failing on every run since Phase 2's compact
storage landed. It called `gpu_subgame.average_from_sums`, which assumes the old
dense `[nodes, buckets, actions]` layout, while `VectorCFR.strategy_sums` is now a
2-D compact table, so it raised `AxisError` on every call. The helper was dead in
production (tests were its only caller), so it is deleted with a note, and the
tests now use `solver.average_strategy_tables()` / `average_strategy_tensor()`,
which understand the compact layout. 4/4 passing again.

This is worth recording as a process point: the plan leans on these suites as the
safety net for P1, and one load-bearing suite had been silently red. Search-module
regression run is now **41 tests green**, and the Phase 4 river screen still
reports 23/23 resolves with 0 fallbacks after the resample refactor.

#### Serving integration — DONE, off by default

`GpuBlueprintAgent.continual_search` (`HOLDEM_CONTINUAL=1`,
`HOLDEM_CONTINUAL_ITERS`, `HOLDEM_CONTINUAL_BUDGET_MS`) routes turn and river
decisions through `resolve_decision`, samples from the exact per-combo policy,
and calls `register_selected_action` so the agent's own range advances from the
policy actually played. It supersedes the river-only Phase 4 path when enabled and
fails closed to the frozen blueprint. Smoke pass over 20 hands:
**16/16 resolved, 0 fallbacks** (11 turn, 5 river), latency 3,152 ms mean /
6,799 ms max, with `own_policy_updates` and `opponent_policy_updates` both firing
— i.e. ranges really are carried from played solutions and from retrospective
Bayesian updates.

A real bug the smoke pass caught: `resolve_turn_at` read `game.community`
directly, so a **retrospective** turn resolve during a river decision saw the live
5-card board and raised. Belief catch-up replays turn events while the hand may
already be on the river, and a retrospective turn resolve must see the board as it
was then. It now uses the 4-card prefix. Without the smoke pass this would have
shown up only as a silent fallback rate on river decisions.

#### The P1 gate instrument, and why it is a duel rather than LBR

LBR's per-pair spread at 200bb is ~15 bb/hand and the two arms decorrelate
immediately (different actions, different trajectories), so even a *paired* LBR
comparison needs ~850 pairs to resolve a 1 bb/hand effect. The duel plays the arms
against each other on duplicate seat-swapped deals, which cancels card luck by
construction.

`tools/continual_search_gate.py` (new; `tools/lbr_search_gate.py` also added, with
per-pair samples now exposed by `local_best_response_probe` for paired analysis).

**NULL run first, per the standing rule.** Off-vs-off, 3,000 pairs:
**+6.99 bb/100 [−19.46, +33.44]** — centred on zero. Note that unlike
`tests/test_duel_null.py` (deterministic `HeuristicAgent`, cancels *exactly*), two
stochastic blueprint agents consume their RNGs at different rates, so a duplicate
pair does not cancel exactly; the residual is variance, not bias. Measured
σ ≈ 7.4 bb/pair, consistent with the ±29 bb/100 the docs report at 3,000 pairs.

#### P1 gate, 300-pair screen at 60 resolve iterations — RELIABLE, STRENGTH INCONCLUSIVE

**+17.82 bb/100, 95% CI [−98.79, +134.42]** over 600 hands. Verdict `KEEP`.

* **417/417 resolves, 0 fallbacks** (275 turn, 142 river) — reliability is not the
  issue.
* Latency 3,308 ms mean / 4,886 ms p90 / 6,321 ms max, inside the 8 s budget.
* The point estimate is weakly positive and **is not evidence**. It is the same
  shape as Phase 4's river-only +7.62 [−21.73, +36.97]: adding the turn moved the
  estimate from +7.6 to +17.8, both inconclusive.

**The measurement cost, not the implementation, is now the bottleneck.** Measured
σ = 10.30 bb/pair with the resolver on (higher than the null's 7.4 — resolving
adds variance):

| effect to resolve | pairs needed | wall-clock at 9.2 s/pair |
|---:|---:|---:|
| 100 bb/100 | 408 | ~1 h |
| 50 bb/100 | 1,632 | ~4.2 h |
| 30 bb/100 | 4,532 | ~11.6 h |
| 20 bb/100 | 10,198 | ~26 h |

So making the effect *bigger* is ~9x cheaper than measuring a small one. That
reframes the next step as improving the resolver rather than buying pairs.

#### Why it is inconclusive: the gate ran an under-converged solver

A convergence probe on the documented leak board (414-node tree, L1 distance of
the root policy from a 960-iteration reference, averaged over live combos, max
2.0):

| iterations | L1 vs converged | L1 vs next rung | solve time |
|---:|---:|---:|---:|
| 30 | 0.564 | 0.289 | 1.0 s |
| **60 (the gate's setting)** | **0.418** | 0.194 | 0.9 s |
| 120 | 0.323 | 0.158 | 1.5 s |
| 240 | 0.213 | 0.134 | 2.7 s |
| 480 | 0.118 | 0.118 | 5.0 s |

At 60 iterations the average live combo's policy differs from the converged answer
by ~21% of its probability mass. **The P1 gate measured a solver that had not
converged**, which settles the P1.2 open question in favour of "under-converged"
rather than "small effect".

A nuance that explains the near-zero duel result: the documented draw's fold
probability is **0.000 at every rung, including 30 iterations**. Exactness fixes
*gross* errors immediately; only the *fine mixtures* need many iterations. So the
likely story is that rare draw spots gain while under-converged mixtures leak EV
in common spots, roughly cancelling. That predicts the effect grows with
iterations — being tested now with a 240-iteration re-run on identical seeds.

**Consequence for the plan:** this is an evidence-backed handoff from P1 to **P2
(throughput)**. A converged resolve costs ~5 s at 480 iterations for one solve, and
a decision can need 1–3 solves, so converged play does not fit an 8 s budget
today. P2's CUDA-graph/batching/PDCFR+ work is what makes P1 pay, exactly as the
plan sequenced it — but the reason is now measured rather than assumed.

#### 240-iteration re-run, and a methodology failure that voids both readings

At 240 iterations on identical seeds: **+54.45 bb/100 [−78.03, +186.93]**,
404/404 resolves, 0 fallbacks, latency **10,356 ms mean / 22,488 ms max** — so 240
iterations is emphatically *not* servable, confirming the P2 dependency. The point
estimate tripled from +17.82, directionally consistent with under-convergence.

**Then an off-vs-off NULL run at 100bb read +34.91 bb/100 [+12.27, +57.55],
verdict PROMOTE, with `resolves: 0/0`** — the harness claiming a policy beats an
identical copy of itself. Diagnosis across three seeds (500 pairs each):

| config | seed 51009 | seed 777 | seed 31337 | spread | lag-1 ρ |
|---|---:|---:|---:|---:|---:|
| two objects (the gate) | +49.10 | −18.65 | +8.06 | 67.75 | ≈0 |
| one shared object (repo null form) | +4.60 | −12.40 | −16.40 | 21.00 | ≈0 |

Conclusions, including two corrections to earlier claims in this document:

1. **The CI formula is sound.** Lag-1 autocorrelation is ≈0, so pair samples are
   effectively independent. An initial serial-correlation hypothesis (the agents
   carry `self._rng` across hands where `lbr.py` reseeds per hand) was **wrong**.
2. **The null is not biased.** Other seeds give −18.65 and +8.06; +34.91 was a
   seed-specific ~5% fluctuation that happened to clear zero. Note the 500-pair
   and 3,000-pair runs on seed 51009 are *nested* subsets of the same deals, so
   they are not independent confirmations of each other.
3. **The real error was methodological, and it is mine.** Every gate reading used
   a *single seed with no disjoint confirmation* — precisely the mistake the
   project's own protocol exists to prevent. `STATUS.md` records the identical
   pattern: a challenger screening at **+75.52** then failing its disjoint
   confirmation at **−12.78**. That history was available and was not applied.
4. **An earlier claim here that the harness "is sound" because the 200bb null read
   +6.99 was too strong.** One passing null does not establish soundness; the same
   harness at another depth then failed. Single nulls are necessary, not
   sufficient.
5. **Two independent agent objects add real variance** (spread 67.75 vs 21.00 for
   the shared-object form) because their RNG streams decorrelate the arms.

**Status of the P1 strength question: unresolved and unsupported.** +17.82 and
+54.45 are not evidence — not because they are wrong, but because a ±116/±132
single-seed interval cannot support any conclusion. What *does* stand, because it
does not route through the duel: resolve reliability (821/821 across both gates,
0 fallbacks), the latency characterisation, the convergence ladder (direct policy
comparison), and the draw-fold leak fix (direct strategy inspection).

#### Common random numbers: fixes the null exactly, does nothing for the gate

`duel.py` gained `common_random_numbers` (default off), which reseeds both agents'
action-sampling RNG identically per hand. Results:

* **Null: exactly +0.00 bb/100, σ = 0.00, every seed.** Identical policies drawing
  identical variates produce mirror-image duplicate hands that cancel exactly —
  the property `test_duel_null.py` previously got only from a *deterministic*
  agent. `tests/test_duel_null.py::test_stochastic_self_duel_reads_exactly_zero_under_crn`
  now pins it for stochastic agents, closing the gap that let a 100bb null report
  +34.91 [+12.27, +57.55].
* **Real gate: no benefit at all.** 240 iterations, seed 51009, 300 pairs:
  **+28.12 bb/100 [−104.79, +161.04]**, margin ±132.9 versus ±132.5 without CRN.

The prediction that CRN would help the gate (by making pre-turn hands cancel) was
**wrong**, and the error is worth recording: *the two arms are opponents within one
hand, not parallel copies of one player*. Coupling their dice cannot make them
play alike — they hold different cards in different seats. CRN removes
"same policy, different dice" variance, which is the null's entire variance and
almost none of the gate's.

#### P1 strength: three inconclusive readings, and the decision that follows

All on seed 51009, 300 pairs, 200bb:

| configuration | result | latency mean/max |
|---|---:|---:|
| 60 iterations | +17.82 [−98.79, +134.42] | 3,308 / 6,321 ms |
| 240 iterations | +54.45 [−78.03, +186.93] | 10,356 / 22,488 ms |
| 240 iterations + CRN | +28.12 [−104.79, +161.04] | 10,750 / 23,239 ms |

Every point estimate is positive; none clears zero; the ~37 bb/100 spread between
them is noise inside their own ±130 intervals. Resolve reliability across all
three: **1,244/1,244, zero fallbacks.**

Resolving a 50 bb/100 effect at 240 iterations needs ~1,800 pairs ≈ **7.6 h** —
spent measuring a configuration already known to be under-converged (L1 = 0.418 at
60 iterations) and unservable (10.7 s/decision against an 8 s budget). **That is
the wrong purchase.** P2 raises iterations per second, which grows the effect and
shrinks the cost of detecting it simultaneously; the same 7.6 h buys a better
answer after P2 than before it.

**Decision: stop measuring P1 at this configuration and start P2.** P1's
deliverables stand on their own evidence — reliability counts, the latency
characterisation, the convergence ladder, and the draw-fold leak fix — none of
which route through the duel.

**Required protocol for any future P1 gate**, non-negotiable:
- multi-seed replication, with a **disjoint confirmation block** on fresh seeds;
- a same-seed off-vs-off null subtracted from each arm, since the null offset is a
  property of the seed;
- enough pairs for the target effect (§ the sample-size table above);
- common random numbers between arms if the variance test below supports it.

### P2.1 Throughput diagnosis — DONE (2026-07-28)

Profiled before optimizing, per the plan. Three measurements, each of which
refuted the preceding hypothesis:

**1. Not bandwidth-bound.** A graph-captured 726-node exact turn solve moves
~39 MB per iteration; at 360 GB/s that is a 0.11 ms floor against 13.5 ms
measured — **123x above the floor**.

**2. Not (mainly) occupancy-bound.** `batch_boards` folds B boards into the combo
axis, making every kernel B x wider for free. If occupancy were the limit this
should scale near-linearly. Measured (`tools/p2_throughput_scoreboard.py`):

| batch | s/iter | iters/s | chance samples/s | speedup | peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.01403 | 71.3 | 71.3 | 1.00x | 427 MiB |
| 2 | 0.02243 | 44.6 | 89.2 | 1.25x | 846 MiB |
| 4 | 0.03978 | 25.1 | 100.6 | 1.41x | 1,645 MiB |
| 8 | 0.07644 | 13.1 | 104.7 | 1.47x | 3,233 MiB |
| 16 | 0.10703 | 9.3 | 149.5 | 2.10x | 672 MiB |
| 24 | 0.15632 | 6.4 | 153.5 | 2.15x | 994 MiB |
| 48 | 0.30121 | 3.3 | 159.4 | 2.24x | 1,880 MiB |

Only **1.47x** across batch 1→8. (Rows at batch ≥16 are not comparable: the
showdown blocker kernel's 2 GB guard silently switches to the per-card loop,
which is why VRAM *drops* from 3,233 to 672 MiB. Worth knowing — it makes naive
batch sweeps misleading.)

**3. It is latency-bound on a serial chain of small dependent kernels.** Counting
the ops `_iterate` issues on this tree: 11 levels, ~265 ops per traversal,
**~530 dependent GPU ops per iteration at ~26.5 µs each**. That product is the
13.5 ms. Widening kernels cannot help a chain whose cost is per-op; only issuing
*fewer, larger* ops can.

**The target is therefore op fusion, not batching, precision, or custom kernels.**
192 of the 265 forward ops come from the per-action × per-player reach loops in
`_iterate`:

```
for action_plan in plan["actions"]:        # up to 6 sized actions
    for player in (0, 1):                  # x2
        reach[player, actor_children] += reach[player, actor_nodes] * strategy[...]
        reach[player, other_children] += reach[player, other_nodes]
```

Replacing that with one `index_add_` over precomputed concatenated indices should
cut the forward chain from ~192 ops to ~10-20, i.e. a plausible 5-10x. This is a
delicate change to a *verified* kernel, so it must ship with a bit-identical
equivalence test against the current implementation on a fixed seed, plus the
existing Kuhn/Leduc exploitability controls.

### P2.2 Forward-pass fusion — DONE, +1.26x (2026-07-28)

The per-action × per-player reach loops in `_iterate` (192 of 265 forward ops per
traversal) are replaced by two `index_add_` calls per level over a precomputed
flat edge list, with both players encoded in one index over a `[2*nodes, width]`
view. Enabled by default; `fused_forward=False` is retained as the control arm.

**Measured: 13.60 → 10.78 ms/iteration, 73.5 → 92.8 iterations/s (1.26x).** Real
and permanent — it applies to blueprint training too — but far below the 5-10x
projected from the op count. The reason is that fusion trades many small ops for
fewer *larger* ops that materialize `[edges, width]` intermediates, so some of the
saved per-op cost returns as memory traffic.

`tests/test_gpu_fused_forward.py`, 9 passing, plus the ground-truth suite (Kuhn
converges to Nash, Leduc converges, graph-replay-matches-eager, batching
exactness) all green.

#### The equivalence test paid for itself twice

It first reported deltas of 7.8e-4 and 1.4e-2 — apparently a broken fusion.
Running the **original** path against *itself* showed deltas of the same
magnitude (9.6e-4 and 1.4e-2), which located the real cause:

> **The solver is non-deterministic run-to-run on CUDA whenever multiple combos
> share a bucket row.** The backward pass's `regrets.index_add_` then has repeated
> destination indices, so CUDA accumulates with atomics in arbitrary order. This
> is a pre-existing property of the verified kernel, not of the fusion, and it
> means bit-identical reproducibility is unavailable for *any* bucketed
> configuration on GPU — including every blueprint training run.

Bit-identity holds exactly where the path is deterministic: identity buckets
(exact turn, batch 1) on CUDA, and all configurations on CPU. The test now asserts
equality there, and elsewhere bounds fusion by the measured nondeterminism floor.
Had the test used a tolerance from the start, it would have passed while hiding
both facts.

#### A third silently-broken test suite, found and fixed

`GpuActionConfig.__post_init__` rejected an **empty** raise menu, but a push-fold
tree legitimately has no sized raises (fold / check-call / all-in only, and
`num_actions` already handles 3 + 0). Phase 3's validation therefore broke
`test_gpu_cfr.PushFoldConvergenceTests` and the entire `test_gpu_exploit` suite
the moment it landed. Empty menus are now legal; non-positive sizes still error.

Running tally for this session: `test_safe_subgame` (compact-storage API drift),
these two push-fold suites, and the stochastic-agent gap in `test_duel_null` —
**three separate holes in the safety net the plan depends on**, none of which any
running process would have surfaced.

#### Rent-a-GPU decision gate: currently INDICATED, pending the fusion attempt

The plan defers the rent decision to this scoreboard, with the rule "rent if the
§4 per-solve budgets are missed by more than ~3x". A 240-iteration turn solve
costs ~3.4 s against a **≤0.5 s** budget for turn-net datagen — **~7x over**.
Fusion (1.26x) and board batching (1.47x) do not close that, and they stack
poorly. **Verdict: rent for datagen.**

Two separate problems, needing two different remedies — conflating them would
waste the money:

* **Datagen scale → rent.** An earlier caveat in this document worried that a
  faster GPU helps a *latency-bound* job less than a compute-bound one. True per
  GPU — but datagen is **embarrassingly parallel across independent solves**, so N
  rented GPUs give ~Nx regardless of per-GPU inefficiency. Renting works here.
  Estimated $150-600 for the whole net stack.
* **Serving latency → a fused Triton/CUDA tree walk.** Renting cannot help a
  single decision return in under 8 s. At 240 iterations a decision costs 10.7 s
  today, and the workload sits ~30x above its bandwidth floor purely in
  framework overhead. The plan's rule was "do not write custom kernels
  speculatively"; that condition is now discharged by profiling rather than
  hunch, so this is the sanctioned next engineering item.

**Blueprint retraining is explicitly NOT on this list.** More blueprint iterations
is the one lever measured dead (plateau at both depths), and in the target
architecture the blueprint is deleted rather than retrained — its two remaining
roles (seeding session ranges, pricing the gadget opt-out) are far less sensitive
to its quality than playing with it was.

### P3a River value net — datagen built and characterised (2026-07-28)

Chosen over a Triton kernel rewrite on measurement: truncating a turn tree at the
river with a value-net horizon takes it from **726 nodes to 81 (9.0x)**, whereas
realistic partial Triton fusion was worth ~1.5-2x for multi-week, high-risk work
on the verified kernel. Depth-limiting *is* the throughput fix — the same reason
DeepStack ran in ~5 s on 2017 hardware.

`backend/cfv/river_dataset.py`: recursive pseudo-random ranges (DeepStack
supplement — flat Dirichlet ranges look nothing like what re-solving meets),
pot/stack grids so one captured CUDA graph serves many situations, exact
per-combo I/O, and TurboReBeL multi-iterate emission.

#### Three measured facts that decide the phase

**1. Targets are essentially exact, which removes v0's binding constraint.**
River solves are *deterministic* — one deal, no chance left to sample — so there
is no Monte-Carlo noise floor at all. Convergence error against a
4,000-iteration reference (mean |delta| / mean |value|, 3 situations):

| iterations | 100 | 200 | 500 | 1000 | 2000 |
|---|---:|---:|---:|---:|---:|
| target error | 1.26% | 0.93% | 0.60% | 0.20% | 0.07% |

**200 iterations suffices**, versus Supremus' 4,000. CFV v0's net scored 9.3 bb
validation MAE against a 24 bb zero-baseline because its targets came from 4
sampled runouts; that failure mode cannot occur on the river.

*A methodological note:* the plan asked for a **repeatability** check before
fitting. Run as written it returned 0.00% at every iteration count — a vacuous
result, because a deterministic solve gives byte-identical output for any seed.
The meaningful diagnostic for a deterministic target is *convergence*, which is
what the table above measures.

**2. TurboReBeL's multiplier is real.** One solve, priced against its own fixed
average strategy for extra beliefs: 0.55 -> 6.23 samples/s at emit=12
(**11x** for ~15% extra cost), 11.15/s at emit=30, saturating ~12/s at emit=60
(each extra row costs ~23 ms). At 200 iterations with emit=30, ~20 samples/s.

**3. A target-quality bug found and fixed.** Board-colliding combos carried fold
values up to 2.32 (the fold kernel weights by opponent mass without masking the
hero's own combo — `cfr.py` warns about this). Their reach is zero so play is
unaffected, but stored as targets they are pure noise for the net to chase. Now
zeroed explicitly. Zero-sum of the range-weighted CFVs verified at 1.1e-4.

#### Rent decision: NOT YET

| samples | local wall-clock |
|---:|---:|
| 1M | ~14 h |
| 5M | ~2.9 days |

A first **1M-sample river net is an overnight local run**, so nothing needs to be
bought to find out whether the net works. Renting (~$150-600, and effective
because datagen parallelises across independent solves regardless of per-GPU
inefficiency) is the right move only *after* the net clears its gate and more
data is the demonstrated bottleneck — which is the plan's own staged-scaling
rule.

### Scope change (2026-07-28): four target depths, autonomous operation

The player must be strong at **20, 50, 100 and 200bb**, quality takes priority
over latency, cloud GPU is available, and work proceeds without approval gates.
Two consequences landed immediately:

**1. The shallow depths had no blueprint at all.** 20bb and 50bb hands route to
the 100bb model, which is badly mismatched. Shallow trees are much smaller, so
those depths can afford a *richer* menu than the deep ones
(`tools/size_blueprint_menus.py`):

| depth | richest affordable menu | nodes | decisions | tables |
|---|---|---:|---:|---:|
| 20bb | rich — 4 preflop / 4 postflop sizes, cap 2 | 121,885 | 44,808 | 558 MiB |
| 50bb | mid+cap3 — 3/3 sizes, cap 3 | 188,333 | 68,428 | 736 MiB |
| 50bb | (rich would be 694,945 nodes / 2,899 MiB) | — | — | too big |

At 20bb the game is close enough to solvable that a rich-menu blueprint may be
strong outright, which is a cheaper win than anything on the net path.

**2. The river datagen grid was 50/100/200bb with absolute pot sizes.** A 160bb
pot is meaningless at 20bb, so pots are now a FRACTION of the effective stack and
the stack grid includes 20bb. Verified: zero-sum holds on 24/24 situations with
coverage across all four depths.

#### Three self-inflicted incidents worth recording

These cost hours and all three were failures of *verification*, not of design:

1. **Three datagen processes ran simultaneously.** A `taskkill /FI` filter printed
   "No tasks running with the specified criteria" — meaning *the filter matched
   nothing* — and that was read as *nothing was running*. Each relaunch stacked.
   The resulting contention produced a fake 3.7x throughput collapse
   (0.39 vs 1.43 solves/s idle) and 9.3 GB of VRAM, which was then misdiagnosed
   as "too many cached solvers" and "the desktop is using the GPU". Both wrong.
2. **Long jobs were repeatedly launched through `tail` or into `/dev/null`**, so
   logs looked empty while jobs ran fine — and one relaunch died silently with a
   syntax error that went unseen for minutes.
3. **A 20k-row shard interval** meant a crash could lose ~4 h of generation.

Standing rules now: every script writes a timestamped log **next to its output in
the repo**; checkpoints are frequent enough that a crash costs minutes (shards
every 2k rows); training saves the best net plus per-epoch telemetry JSONL; and a
job is confirmed running by watching its **output grow over an interval**, never
by process presence or a kill command's exit text.

#### "Only if we make sure this is helpful": the data-scaling gate

A full river dataset is 1-3 days of local generation, so
`tools/river_net_scaling.py` decides whether that is worth spending *before* it is
spent. It trains on nested subsets and reports held-out MAE as a ratio to a
zero-predictor:

* ratio ≈ 1.00 → the net learned nothing; **stop**, more data cannot fix a
  representation problem (this is exactly how CFV v0 failed, unmeasured);
* ratio falling with N → data-limited, **scale**;
* ratio flat but < 1 → capacity- or target-limited; spend on the net or the
  targets, **not** on more rows.

Sequencing: let generation reach ~15-20k rows (~5 h at the measured 1.0-1.4
solves/s), run the probe, and only then commit the remaining days — or pivot the
GPU to the 20bb/50bb blueprints, which are the concrete alternative.

### Standard postflop menu, and LBR baselines at all four depths (2026-07-28)

**Postflop sizing is now 0.33 / 0.5 / 0.75 / 1.0 / 1.4 pot** on every postflop
street, applied to the turn, river and subgame resolvers alike
(`TURN_FRACTIONS`, `RIVER_FRACTIONS`, `SUBGAME_FRACTIONS`, `FLOP_FRACTIONS`).

*Does the blueprint's different training menu break this?* Measured, not assumed:
28/28 resolves, **0 fallbacks, 1.79% mean detachment** — statistically unchanged
from the 1.77% seen with the old menu, and the one detachment was the usual
all-in/showdown topology divergence, not a sizing mismatch. `_map_action` maps
each blueprint size to the nearest resolver size (1.5 → 1.4, 0.33 → 0.33) and
renormalises. The projected baseline is coarser than the resolver's menu, which
loosens the gadget's safety bound; it does not corrupt the played strategy,
because that comes from the exact-card solve.

Cost of the wider menu, and why it strengthens the case for the river net:

| depth | turn+river nodes | turn only (river net) | net gain |
|---|---:|---:|---:|
| 20bb | 60 | 21 | 2.9x |
| 50bb | 774 | 159 | 4.9x |
| 100bb | 3,032 | 333 | 9.1x |
| 200bb | 8,204 | 369 | **22.2x** |

Deep turn trees are 11x bigger than under the 2-size menu, so the river net moves
from helpful to **essential** at 100/200bb. Datagen cost rose only 16%
(1.43 → 1.20 solves/s) despite 3x more nodes — latency-bound again.

**LBR baselines now cover every target depth** (400 pairs each, 0% fallback):

| depth | served today by | LBR bb/100 | 95% CI |
|---|---|---:|---|
| **20bb** | 100bb histogram champion | **+130.31** | [+95.22, +165.40] |
| 50bb | 100bb histogram champion | +85.05 | [−12.80, +182.90] |
| 100bb | 100bb histogram champion | +137.58 | [−3.57, +278.73] |
| 200bb | 200bb scalar champion | +291.23 | [+79.10, +503.36] |

20bb is the **worst-served depth with the tightest evidence** — its interval
clears zero decisively (shallow play has lower variance), because a 100bb-trained
blueprint plays far too many small bets at a depth that is nearly push-fold.

### P3b Exact FLOP resolving for shallow stacks — no value net required

Sizing exact flop-to-river trees under the standard menu changed the shallow-depth
plan entirely:

| depth | pot 6bb | pot 12bb | exact-combo tables | verdict |
|---|---:|---:|---|---|
| 20bb | 5,303 | 987 | ~0.4 GiB | **feasible** |
| 50bb | 41,135 | 9,771 | 0.8–3.3 GiB | feasible at medium+ pots |
| 100bb | 132,107 | 41,135 | ~10.5 GiB | too big |
| 200bb | 339,433 | 132,107 | ~27 GiB | impossible |

So **20bb can be played by solving flop, turn and river exactly to showdown, with
no value network at all** — DeepStack-class play at that depth using machinery
that already exists, attacking the +130 bb/100 exploitability directly. The nets
remain necessary for 100/200bb.

Built: `backend/search/exact_flop.py` (`ExactFlopSampler` with identity buckets on
flop/turn/river, sampled runouts since a flop has 1,176 completions rather than
the turn's 48, plus `exact_flop_is_affordable` as the depth guard);
`resolve_postflop_at` generalises the turn resolver to any postflop street; and
`continual.py` accepts a flop entry street. The serving agent picks flop entry
when the tree fits and turn entry otherwise, falling through to the blueprint for
flop decisions at depths where no exact path exists.
`tests/test_exact_flop.py`, 10 passing, including the affordability guard
refusing 100/200bb.

**Not yet done:** the four-depth smoke of the flop path (it was launched on top of
a running datagen and correctly killed — one GPU job at a time), and the 20bb/50bb
blueprints that would give those depths a matched preflop policy and range seed.

### Live defect: the 24x-pot shove — ROOT-CAUSED; the fix is NOT a clear win (2026-07-29)

Reported from live play: on `T(c) J(s) K(h) A(c)` holding `K(d) 5(s)`, the agent
moved in for **3,980 chips into a 166-chip pot — 24x the pot**, ~199bb effective.
The widest own-bet size anywhere in the system is 1.4x pot, so no strategy in the
project contains that action.

#### The instrument was wrong first

`log_agent_decision` reported `blueprint-only` for this hand, but it only ever
consulted `subgame_search` / `exact_river_search` — it had **no knowledge of
`continual_search`**. So every exact-resolver decision was mislabelled as
blueprint-only, and its `actions` field always showed the BLUEPRINT's mix rather
than the acting one, which made "why did it do X?" unanswerable for the engine
that was actually playing. Added `decided_by`, `resolver.acting_mix`, and renamed
the old field to `blueprint_actions`. The log now reads e.g.
`DECIDED BY: exact-resolver` with the resolver betting 86% on `6(d) J(c) 6(h)`
where the blueprint checks 98.8%.

#### Neither policy wanted to shove

| source | mix at that spot |
|---|---|
| blueprint, node 104350, turn bucket 12 | check/call 72.6%, raise 0.5x 27.2%, **all-in 0.0%** |
| exact turn solve, 3,210 nodes, 200 iters | `K(d)5(s)` checks **94.2%**, raise 0.33x 4.9%; **all-in 0.16%** across the whole range |

So the shove was not a strategic choice by either engine.

#### Root cause: ALL-IN was the one size-bearing action executed literally

Every `raise` re-derives its chip amount from the REAL pot in
`_raise_target_for_choice`. ALL-IN just shoved whatever was behind. And `_locate`
maps a live hand onto an abstract node **by translated action sequence alone — it
never compares pot/stack geometry**. A few repeated opponent min-raises exhaust
the tree's 3-raise cap, so the real state lands on an abstract node whose pot is
far larger, and where jamming the stack is a sane 2-3x-pot action. Translated back
into a 166-chip pot, the same trained action becomes 24x pot.

Measured with `tools/overbet_audit.py` (200bb, 200 hands, resolver OFF; jams
counted only where a smaller raise was legal, so a forced short-stack shove never
counts):

| opponent | decisions | overbets >=3x pot | worst |
|---|---|---|---|
| always-call | 758 | 0 | — |
| self-play | 460 | 0 | — |
| always-min-raise | 640 | **8 (1.25%)** | **15.4x pot** |

Only the min-raiser triggers it, because only it exhausts the raise cap. **Four of
the eight were preflop**, which no amount of postflop resolving can ever fix.

Supporting structure in the 200bb champion: **77-79% of decision nodes offer no
raise below all-in** and carry ~30% all-in mass, but the abstract jam is <=3x pot
at 14,492 of those 15,188 nodes. The mass is trained and reasonable; only the
translation is wrong.

#### The fix, and why it is not obviously right

`GpuBlueprintAgent._all_in_size` gives ALL-IN the same action translation raises
already get: preserve the size relative to the MATCHED POT that the abstract jam
represented (`all_in_geometry_tolerance = 1.5`), bounded absolutely
(`all_in_max_pot_multiple = 6.0`). Both ratios are computed identically — total
committed after shoving, over the matched pot before — so units cancel and a
genuine short-stack jam is untouched. Worst jam **15.4x -> 4.6x pot**; the guard
fires 11 times in 677 decisions vs the min-raiser and **0 times** vs a calling
station or in self-play.

Then the A/B (blueprint only, CRN coupled, 3,000 hands/arm):

| depth | opponent | guard ON | guard OFF | delta |
|---|---|---|---|---|
| 200bb | always-min-raise | +80.58 [-13.91, +175.07] | +349.40 [+195.49, +503.31] | **-268.82** |
| 200bb | always-call | +199.17 [+152.77, +245.57] | +202.63 [+154.88, +250.39] | -3.46 |
| 100bb | always-min-raise | +104.65 [+23.89, +185.42] | +228.65 [+128.96, +328.34] | **-124.00** |
| 100bb | always-call | +189.52 [+135.67, +243.36] | +202.85 [+145.90, +259.80] | -13.33 |

**The guard costs 124-269 bb/100 against the opponent whose lines trigger it, and
is neutral against everything else.** That is not a defect in the fix — against a
station that calls any jam, shoving 199bb IS correct exploitation. So the reported
hand is a leak against a *thinking* opponent and a moneymaker against a loose one,
and the blueprint has no opponent model to distinguish them.

By this project's own standard — never serve what has been measured harmful and
never measured helpful — the guard does not yet earn its place on. It stays
implemented, tested, and behind per-agent flags (`all_in_geometry_guard`,
`all_in_max_pot_multiple`, `all_in_geometry_tolerance`) so the decision rests on a
measurement rather than on taste.

**The measurement that would settle it was NOT taken:** LBR (a best responder,
hence the right judge of whether the jam is exploitable) guard-on vs guard-off. A
first attempt crashed on wrong result keys (`mean_bb_per_100` instead of
`lbr_bb_per_100`) and the rerun was stopped by user instruction. Until that number
exists, "is the huge jam a leak in absolute terms?" is unanswered.

Covered by `tests/test_all_in_translation.py` — 7 tests: exact-arithmetic anchors
(700 and 1,100 chips, derived not guessed), a matching-geometry no-op, a
short-stack no-op, the disable switch the A/B depends on, and an end-to-end
serving-path assertion that no jam exceeds the cap over 120 hands vs a min-raiser.

Resizes are reported in the decision log as `all_in_rescaled`, so a bet that is
neither a menu size nor the stack is explainable rather than looking like a new bug.

#### What this says about the architecture

The exact resolver puts **0.16%** on all-in at the reported spot because its tree
is built on the REAL geometry — there is no translation step to distort. That is
the structural fix, and it is already on for flop/turn/river. The guard only
matters on paths the resolver does not cover, and the largest of those is
**preflop**, where half the observed overbets occurred and where nothing is planned
before P5. Raise-cap exhaustion is a translation problem, so the durable answers
are a richer preflop menu or preflop resolving, not a sizing heuristic.

### Instrument defect: CRN was silently OFF for the serving agent (2026-07-29)

`backend.eval.duel.head_to_head`'s common-random-numbers coupling reseeds
`agent._rng` before every hand so two arms draw identical variates at identical
infosets and diverge ONLY where their policies differ. It guards with
`hasattr(target, "_rng")` — and `MultiStackBlueprintAgent`, **the serving agent
every real comparison uses**, never had an `_rng`. The guard silently skipped it.
The documented "off-vs-off null reads exactly +0.00 bb/100" was measured on a
single-depth `GpuBlueprintAgent`, so it never covered the router.

Null duel of two identical routers, 600 hands, 200bb:

| coupling | result |
|---|---|
| CRN off | **+33.15 bb/100 [-48.71, +115.01]** |
| CRN on, after the fix | **+0.00 bb/100 [0.00, 0.00]** |

Fixed with an `_rng` property on the router that fans a fresh generator (copied
state, not a shared object) out to each depth — sharing one generator would let a
hand routed to 100bb advance the 200bb stream, reintroducing exactly the desync
CRN exists to remove.

**Consequence for past results:** any router duel labelled "CRN on" before this
date was uncoupled and carried that ~+/-80 bb/100 of avoidable noise. The resolver
on/off duels in section 8 (+17.82 / +54.45 / +28.12, all spanning zero) are in
that class, which is one more reason their intervals were uninformative.


## 9.9 2026-07-29 implementation: bounded-memory exact resolving

The live flop incident changed the resolver constraint from "tree appears small
enough" to "the complete CUDA peak is admitted before allocation." The
implementation now follows this order:

1. Estimate compact CFR tables, frozen policy, tree/static tensors, traversal,
   tiled showdown work, graph-private work, and returned strategy memory.
2. Require the candidate to fit the 12,000-node latency ceiling, 9.5 GiB process
   allocation ceiling, and 2 GiB display/OS headroom.
3. Select the richest admitted flop menu. Above SPR 4, skip the known-explosive
   rich tier and begin with two sizes; fall to one size only if required.
4. Allocate the solver only after admission. If every tier fails, keep playing
   the promoted frozen blueprint.

The memory model is intentionally conservative. Representative 20-200bb generic
flop roots admitted 987-10,429 nodes and estimated 970-1,856 MiB peaks. The
current real passive-entry 200bb diagnostic builds 11,991 nodes at the two-size
tier.

Quality-preserving changes accompany the hard limits:

- legacy dense blueprint checkpoints are compacted as they load;
- future-street blueprint buckets are recomputed for each sampled runout;
- structural projection crosses street-end nodes instead of orphaning all
  descendants;
- only the acting node and immediate response frontier leave the GPU;
- stored response frontiers replace retrospective opponent solves where the
  policy is already known;
- a retrospective solve's child policy becomes the current policy when it
  already reaches the live node;
- 120 iterations remain the serving target and 60 is the default quality floor.

The optional river CFV horizon is promotion-gated. It cannot be enabled by
configuration alone: its report must show at least 0.90 range-weighted
top-action agreement and at most 0.30 policy L1. The existing checkpoint
(0.3766 / 1.1474) stays disabled, so no failed neural approximation was
introduced to solve the memory problem.


## 10. References

- DeepStack — https://arxiv.org/pdf/1701.01724 · supplement: https://poker.cs.ualberta.ca/publications/17science-supplementary.pdf
- Supremus (Deep CFV networks, DCFR+) — https://arxiv.org/pdf/2007.10442
- Safe and Nested Subgame Solving (Libratus' method) — https://arxiv.org/pdf/1705.02955
- ReBeL — https://arxiv.org/pdf/2007.13544
- TurboReBeL (single-solve/multi-iteration, 250×) — https://openreview.net/forum?id=yMo7Z670f6
- RL-CFR (learned action abstraction) — https://arxiv.org/abs/2403.04344
- Discounted CFR — https://arxiv.org/abs/1809.04040
- PDCFR+ / weighted CFR with optimistic mirror descent — https://arxiv.org/abs/2404.13891
- LBR ("Equilibrium Approximation Quality of Current No-Limit Poker Bots") — https://arxiv.org/abs/1612.07547
- AIVAT — https://arxiv.org/abs/1612.06915
- GTO Wizard benchmark — https://arxiv.org/html/2603.23660v1
- AlphaHoldem (end-to-end RL alternative, considered and not chosen) — https://ojs.aaai.org/index.php/AAAI/article/view/20394
- Search in Imperfect-Information Games (survey) — https://arxiv.org/pdf/2111.05884
