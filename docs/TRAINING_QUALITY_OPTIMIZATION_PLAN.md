# Holdem Training-Quality Optimization Plan

**Date:** 2026-07-25; experiment record updated 2026-07-27  
**Status:** Phases 1–4 implemented and evaluated; Phase 4 projection repair is the next task  
**Goal:** Improve the actual playing strength, robustness, and measurability of the 100bb and 200bb HUNL agents within the RTX 3060 12GB constraint.

## Executive conclusion

The strongest path is not more iterations on the existing abstraction. The
current player has reached an abstraction ceiling. The next meaningful gains
should come from:

1. trustworthy evaluation and promotion;
2. higher-resolution, genuinely potential-aware hand abstraction;
3. state-dependent bet sizing;
4. exact-card river continual resolving with correct range tracking; and
5. a CFV network only after target quality is high enough.

## Current baseline

- The verified serving champions are histogram@30k for 100bb and scalar@118k
  for 200bb, with live search disabled.
- More iterations have stopped improving the existing abstractions. The 200bb
  model plateaued from approximately 47k to 118k.
- The clean 200bb no-limp histogram experiment completed gates at 10k, 20k,
  and 40k. Its confirmation estimates remained statistical ties with the
  scalar@118k champion, so the checkpoint was not promoted and further
  identical training was stopped.
- Histogram bucketing is the only recent change with a verified qualitative
  gain. Bucketed turn/river search and CFV flop search did not improve play
  under corrected evaluation.
- The combined v3 + Phase 3 challenger failed its 5k screen and was retired.
  Exact-card Phase 4 passed its latency target but did not establish a strength
  gain and exposed a blueprint-projection defect.

## Audit findings

### 1. The histogram abstraction is not fully potential-aware

The implementation samples final river-equity histograms from the flop or
turn. It does not recursively represent how a flop transitions into
strategically different turn clusters and then river clusters. This is
distribution-aware, but the potential-aware method models trajectories through
all future rounds.

Reference:
[Potential-Aware Imperfect-Recall Abstraction](https://www.cs.cmu.edu/~sandholm/www/potential-aware_imperfect-recall.aaai14.pdf)

### 2. The river abstraction is especially coarse

River hands are reduced to 30 scalar-equity buckets against a generic range.
This loses blocker effects, nut advantage, and equity against polarized ranges.
It is likely a major contributor to the measured tight-aggressive weakness.

### 3. VRAM is spent on low-value table entries

The active 200bb no-limp tree has 143,396 nodes, but only 50,656 are decision
nodes. Regret and strategy tensors nevertheless allocate 169 buckets for every
node and street, including terminal nodes and river nodes that currently use
only 30 buckets. Street-sharded, decision-node-only storage can fund higher
flop, turn, and river resolution without widening the betting tree.

### 4. LBR is not a trustworthy absolute signal yet

The current probe measures from post-blind stacks, loads the old CPU agent
rather than the serving GPU champion, and uses a weak uniform-range model.
It must be repaired before it can serve as an exploitability guard.

Reference:
[Equilibrium Approximation Quality of Current No-Limit Poker Bots](https://arxiv.org/abs/1612.07547)

### 5. The promotion gate is only relative

The duel is NULL-tested, but its default invocation repeatedly uses the same
seed block and compares only against one incumbent. Repeated tuning can
overfit that block, while single-opponent promotion can miss non-transitive
regressions.

### 6. CFV v0 was target-noise and data limited

Current targets use 500 solving iterations and four evaluation runouts. The
dataset contains 7,750 solved situations. Supremus used 4,000 iterations per
player and millions of solved situations; its gains came from lower
target/network error, more search iterations, and a finer action space.

Reference:
[Unlocking the Potential of Deep Counterfactual Value Networks](https://arxiv.org/pdf/2007.10442)

## Phase 0 — Protect the current experiment

1. Snapshot the 200bb no-limp model at 10k.
2. Run a screening duel against the current 200bb champion.
3. Run the confirmation duel on a new, disjoint seed block.
4. Continue to 20k only if the model is non-regressing and decision logs prove
   the intended no-limp and sizing behavior is active.
5. Stop after two consecutive statistically tied milestones unless another
   trusted diagnostic shows continued improvement.
6. Resize the queued 100bb no-limp configuration before launch.

Training milestones are maximum budgets, not automatic targets.

## Phase 1 — Make evaluation promotion-grade

Build one evaluation orchestrator that provides:

- corrected GPU-compatible LBR with full starting-stack baselines;
- duplicate, seat-swapped LBR deals;
- GPU-strategy belief updates after observed actions;
- fresh screening seeds and a separate confirmatory holdout;
- cross-play against the current champion and retained prior champions;
- per-street, pot, SPR, position, off-tree, and sizing diagnostics;
- exact-node, fallback, and translation-distance measurements; and
- experiment manifests containing code revision, model hash, abstraction hash,
  configuration, seeds, and evaluator version.

Promotion requires:

- a positive confirmatory head-to-head confidence interval;
- no significant regression against retained champions;
- no material LBR regression;
- no unexplained fallback or translation increase; and
- an independent external benchmark for major releases.

Complete AIVAT decision variates once a dependable value function exists.

Reference:
[AIVAT](https://aaai.org/papers/11481-aivat-a-new-variance-reduction-technique-for-agent-evaluation-in-imperfect-information-games/)

### Phase 1 implementation

Implemented on 2026-07-25:

- `backend/eval/gate.py` now runs fresh-seed screening, disjoint confirmation,
  retained-champion cross-play, serving diagnostics, a relative GPU-LBR guard,
  and optional promotion.
- `backend/eval/duel.py` now reports exact-node/fallback rates, off-tree and
  sizing-translation gaps, and decision distributions by street, position,
  pot band, and SPR band.
- `backend/eval/lbr.py` now evaluates the serving GPU blueprint on duplicate
  seat-swapped deals, uses the full starting-stack baseline, updates
  blueprint beliefs from public actions, and estimates equity against the
  resulting range.
- Every gate report records fresh or explicit seeds, evaluation budgets, code
  revision and dirty state, checkpoint hashes, abstraction hashes, model
  configuration, sampler configuration, and evaluator version.

Run the full gate after a suitable challenger checkpoint is available:

```powershell
python -m backend.eval.gate `
  --data-dir backend/data/gpu_blueprint_200bb_nolimp `
  --stack-bb 200 `
  --promote
```

Omit `--promote` for a read-only evaluation. Reports are saved under the
selected data directory's `evaluations/` folder. A standalone LBR comparison
can be run with:

```powershell
python -m backend.eval.lbr `
  --data-dir backend/data/gpu_blueprint_200bb_nolimp `
  --checkpoint backend/data/gpu_blueprint_200bb_nolimp/checkpoint.npz `
  --stack-bb 200 `
  --pairs 250
```

The independent external-bot benchmark remains a release-process dependency;
it needs a selected external opponent and adapter. AIVAT decision variates
remain intentionally deferred until a dependable value function exists.

### 10k no-limp gate result

The evaluator-v3 gate ran on the intentional 10k checkpoint using 750
screening pairs and a disjoint 3,000-pair confirmation block:

- screening: **+5.50 bb/100**, 95% CI **[-61.27, +72.28]** (`KEEP`);
- confirmation: **+1.57 bb/100**, 95% CI **[-26.73, +29.87]** (`KEEP`);
- exact-node rate: 99.88% for the challenger and 99.97% for the incumbent;
- fallback rate: 0.12% for the challenger and 0.03% for the incumbent;
- mapping and relative LBR guards passed;
- the no-limp challenger emitted zero open limps, while it observed 93
  incumbent limps that required no-limp-tree translation;
- no retained backup champions were available for cross-play; and
- the challenger was **not promoted** because the confirmatory confidence
  interval did not clear zero.

The result is non-regressing and verifies that the intended no-limp behavior
is active, so it satisfies the plan's conditions for an optional continuation
to the 20k milestone. It is not evidence that the 10k challenger is stronger.

Full report:
`backend/data/gpu_blueprint_200bb_nolimp/evaluations/gate-20260725T135428Z-iter10000.json`

### 20k and 40k no-limp continuation results

The clean no-limp checkpoint was continued in isolation with search disabled.
Both later milestones used the same evaluator-v3 screen/confirmation structure
and were read-only: neither run could promote automatically.

| Milestone | Screen bb/100 (95% CI) | Confirmation bb/100 (95% CI) | Mapping | Relative LBR | Eligible |
|---|---:|---:|---:|---:|---:|
| 20k | **+75.52** [+14.28,+136.75] | **−12.78** [−44.27,+18.71] | Pass | Pass | No |
| 40k | **+67.56** [−2.17,+137.29] | **−9.58** [−41.32,+22.15] | Pass | Pass | No |

The positive 20k screen did not replicate on its disjoint confirmation block.
The 20k and 40k confirmation estimates are nearly unchanged and both include
zero. This satisfies the stop-on-plateau rule: the clean no-limp checkpoint is
retained as an experiment artifact but is not a verified upgrade, and no
further identical training is planned.

Full reports:

- `backend/data/gpu_blueprint_200bb_nolimp/evaluations/gate-20260726T024805Z-iter20000.json`
- `backend/data/gpu_blueprint_200bb_nolimp/evaluations/gate-20260726T141817Z-iter40000.json`

## Phase 2 — Build blueprint-v3 hand abstraction

1. Replace the single dense table with decision-node-only, per-street tables.
2. Reinvest memory in approximately 300–500 flop clusters, 300–500 turn
   clusters, and 100–300 opponent-aware river clusters.
3. Build recursive potential-aware abstraction:
   - cluster rivers first;
   - represent turns by transition distributions over river clusters;
   - represent flops by transition distributions over turn clusters; and
   - cluster those distributions with EMD.
4. Make bucket assignment deterministic through suit/public-board
   canonicalization and enumerated or stratified common runouts.
5. Replace scalar river equity with OCHS-style equity-versus-range, blocker,
   nut, and board-texture features.
6. Experiment with limited prior-street recall.

A recent preprint formalizes historical-information loss as an abstraction
resolution ceiling, but validates the proposal on simplified Hold'em. Treat it
as a measured experiment rather than an assumed win.

Reference:
[Beyond Outcome-Based Imperfect-Recall](https://arxiv.org/abs/2510.15094)

### Phase 2 implementation

Implemented on 2026-07-25:

- CFR regrets and strategy sums now use decision-node-only, per-street compact
  shards concatenated into ordinary 2-D tensors. CUDA graph discounting,
  cloning, zeroing, and checkpoint copying remain in-place operations.
- Each street stores exactly its configured bucket count. Terminal nodes and
  unused river bucket columns no longer consume table VRAM.
- Legacy dense checkpoints are migrated into the compact layout in memory and
  are written back in compact-v2 format on the next save.
- Serving loads both legacy dense and compact-v2 checkpoints. Compact
  strategies preserve the existing `strategy[node, bucket]` lookup contract
  without expanding the full blueprint into a dense array.
- Blueprint-v3 fits river clusters first from opponent-range equity, blocker,
  nut, and board-texture features. It then clusters turn-to-river and
  flop-to-turn transition distributions using EMD-style CDF landmarks.
- Public boards are canonicalized across all 24 suit permutations. Every
  private combo on a public board uses the same deterministic stratified
  future-card set.
- Two-bin prior-street recall is encoded at the turn and river. The resulting
  configured resolution is **169 preflop / 384 flop / 384 turn / 192 river**.
- Abstraction centroids, recall configuration, compact storage layout, and
  table-memory statistics are persisted in checkpoints and telemetry.
- The expensive one-time v3 fit is saved at iteration zero before CFR
  training starts.
- CFV-v1 now rejects the wider v3 turn abstraction explicitly; support belongs
  to the planned CFV-v2 phase.

Start an isolated 200bb no-limp v3 experiment with:

```powershell
python -m backend.solver.gpu.train `
  --iterations 10000 `
  --device cuda `
  --stack-bb 200 `
  --ruleset nolimp `
  --abstraction v3 `
  --tag v3_nolimp `
  --batch-boards 1 `
  --save-every 1000
```

The first invocation fits and saves the recursive abstraction before
training. Subsequent invocations resume the compact checkpoint and restore
the exact fitted centroids.

The approved 5k/10k experiment, now deferred until Phase 3 is present, is
automated by:

```powershell
python tools/run_v3_experiment.py
```

The runner is resumable and owns the complete sequence: train to 5k, run the
Phase 1 gate, train to 10k, and run the second gate. It emits one monitor event
per 1,000 training iterations and persists:

Use `python tools/run_v3_experiment.py --stop-after 5000` to stop cleanly
after the first gate. A later default invocation reuses that checkpoint and
completed gate before continuing to 10k.

- `experiment_state.json` for stage-level resume;
- `experiment_events.jsonl` for machine-readable milestones; and
- `experiment.log` for complete child-process output.

All three files are written inside
`backend/data/gpu_blueprint_200bb_v3_phase3_nolimp/`. The runner never
promotes a checkpoint automatically; both gates are read-only comparisons
against the current 200bb no-limp champion.

Acceptance procedure:

1. Record the printed compact-table MiB and row-reduction percentage.
2. Measure v3 bucketing and CFR throughput separately.
3. Screen at 1k only for crashes, fallback drift, and gross regression.
4. Run the Phase 1 promotion gate at 5k and 10k.
5. Promote only if the disjoint confirmation interval clears zero and the
   mapping/LBR guards remain acceptable.

This implementation is not itself evidence that v3 is stronger. The
blueprint-v3 checkpoint must win the Phase 1 A/B gate before replacing any
champion.

## Phase 3 — Improve bet sizing without tree explosion

Use state-dependent menus with two or three raises per node. Classify public
states by street, position, SPR, pot size, raise number, board texture, range
advantage, and nut advantage. On sampled high-reach states, solve richer local
menus and keep only the actions with the largest marginal EV.

After the static state-dependent version clears evaluation, consider RL-CFR.

Reference:
[RL-CFR, ICML 2024](https://proceedings.mlr.press/v235/li24t.html)

EVPA is an exploratory option, but its published hardware scale is much larger
than this desktop.

Reference:
[EVPA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c1b5863a6b0f925617b917bb2f55be0-Paper-Conference.pdf)

### Phase 3 implementation

Implemented on 2026-07-25:

- The action id space now contains stable candidate pools: **0.5, 0.75, 1.0,
  1.5, 2.25 pot preflop** and **0.25, 0.33, 0.5, 0.75, 1.0, 1.5 pot
  postflop**.
- Only two or three sized raises are legal at any structural public node;
  fold/check-call/all-in remain separate actions. The selector uses street,
  button/out-of-position status, SPR, pot band, raise number, and facing-bet
  pressure.
- The initial `phase3-static-v1` policy changes menus deterministically across
  those state classes while keeping the tree card-independent and compact.
- Rich local-solve measurements can override individual state classes. The
  offline compiler greedily retains the sizes with the largest
  reach-weighted marginal EV. Each sample remains separate by board texture,
  range advantage, and nut advantage, so those strategic contexts influence
  the fixed structural menu without forcing a card-dependent superset tree.
- Compiled overrides and the source profile SHA-256 are embedded in checkpoint
  configuration. Training, serving, retained-model loading, and evaluation
  therefore reconstruct exactly the same action tree without an external
  runtime file.
- Gate translation diagnostics now compare observed raises only with the
  sizes legal at the receiver's exact pre-action node, rather than the entire
  candidate pool.
- Phase 3 starts with zero regrets and strategy sums under the isolated
  `v3_phase3_nolimp` tag. It may import the already-fitted v3 card sampler from
  the cancelled iteration-zero experiment; no incompatible strategy values
  are migrated.

Compile local-solve action measurements from JSONL:

```powershell
python -m backend.solver.gpu.action_profile `
  --samples backend/data/action_profiles/phase3_local_evs.jsonl `
  --output backend/data/action_profiles/phase3_v1.json `
  --max-sizes 3 `
  --min-third-gain-bb 0.01
```

Each input row contains the structural state, reach weight, strategic
context, and candidate action values:

```json
{
  "street": 2,
  "actor": 1,
  "pot": 18.5,
  "to_call": 0.0,
  "stack_behind": 172.0,
  "raises": 0,
  "reach": 0.0062,
  "board_texture": "wet-connected",
  "range_advantage": "oop-small",
  "nut_advantage": "btn-large",
  "candidate_evs": {
    "0.25": 1.31,
    "0.33": 1.39,
    "0.5": 1.42,
    "0.75": 1.40,
    "1.0": 1.33,
    "1.5": 1.22
  }
}
```

Start a combined v3 + Phase 3 run with the built-in static profile:

```powershell
python -m backend.solver.gpu.train `
  --iterations 10000 `
  --device cuda `
  --stack-bb 200 `
  --ruleset nolimp `
  --abstraction v3 `
  --phase3-actions `
  --tag v3_phase3_nolimp `
  --sampler-init backend/data/gpu_blueprint_200bb_v3_nolimp/checkpoint.npz `
  --batch-boards 1 `
  --save-every 1000
```

Add `--action-profile backend/data/action_profiles/phase3_v1.json` to train a
compiled local-EV profile. The 5k/10k orchestrator uses the built-in static
profile first, as required by the plan. A compiled profile is a separate
challenger and must use a separate tag.

### Combined v3 + Phase 3 experiment result

The combined model was trained from iteration zero and screened at 5k. It
measured **−94.92 bb/100**, 95% CI **[−166.36,−23.48]**, against its incumbent.
The screen classified it as a significant regression, the mapping guard also
failed, and the confirmation block was correctly skipped. The experiment was
stopped at 5k, never promoted, and is retired in its present form.

Full report:
`backend/data/gpu_blueprint_200bb_v3_phase3_nolimp/evaluations/gate-20260725T175229Z-iter5000.json`

## Phase 4 — Exact-card river continual resolving

Build a river resolver that:

- uses exact private-card combinations;
- tracks both players' ranges after every real action;
- updates the agent's own range using the policy actually played by prior
  solutions;
- inserts observed off-tree opponent sizes directly into the subgame;
- uses the safe resolving gadget;
- re-solves after each action; and
- has a strict latency budget and blueprint fallback.

Do not revive bucketed turn search before this exact-card river design passes.

Reference:
[Safe and Nested Subgame Solving](https://papers.nips.cc/paper_files/paper/2017/hash/7fe1f8abaad094e0b5cb1b01d712f708-Abstract.html)

### Phase 4 implementation

Implemented on 2026-07-26, behind an opt-in serving flag:

- `BettingTree` accepts an exact mid-street root containing the real actor,
  total commitments, river commitments, remaining stacks, prior actions,
  raise count, and minimum raise increment. Every river decision therefore
  re-roots at the live state instead of replaying from a street-boundary
  approximation.
- `ExactRiverSampler` uses one identity bucket for each of the 1,326 private
  combinations. Board-colliding combinations have zero reach; terminal
  showdown and blocker correction continue to use the trusted `VectorCFR`
  kernels.
- Both players begin the river with exact-combo blueprint reaches. After an
  action, Bayesian range updates use its exact per-combo likelihood. The
  agent's own update is taken from the solution that actually chose the move.
- When an opponent uses an off-tree river size, that exact fraction is added
  to a retrospective solve rooted immediately before the observed action.
  Its likelihood is therefore measured directly; pseudo-harmonic translation
  is not used for the Phase 4 belief update.
- The safe gadget's per-combo opponent opt-out values are derived by
  projecting the loaded blueprint into the exact river tree and computing an
  opponent best response to that frozen policy.
- A single wall-clock budget covers belief catch-up and the fresh current
  solve. A timeout, projection failure, or zero-mass belief disables Phase 4
  for the rest of that hand and falls back to the frozen blueprint.
- Enabling Phase 4 in normal serving disables the legacy bucketed turn/river
  resolver, keeping the experiment isolated.
- Every solve records mode, tree size, exact combo count, iterations, latency,
  blueprint baseline source, inserted observed size, and fallback status.

Phase 4 is **off by default**. Enable it only for evaluation:

```powershell
$env:HOLDEM_PHASE4_RIVER = "1"
$env:HOLDEM_PHASE4_ITERS = "80"
$env:HOLDEM_PHASE4_BUDGET_MS = "6000"
```

Run the isolated on/off gate against one frozen checkpoint:

```powershell
python -m backend.eval.river_search_ab `
  --checkpoint backend/data/gpu_blueprint_200bb/champion.npz `
  --stack-bb 200 `
  --pairs 100 `
  --iterations 80 `
  --budget-ms 6000
```

The screen is only an engineering check. Confirmation must use a
pre-registered larger duplicate-pair sample and requires:

1. the 95% confidence interval for Phase 4 minus blueprint to clear zero;
2. at least 99% of attempted river decisions to resolve without fallback;
3. the latency distribution to fit the serving budget; and
4. a model-vs-itself/search-off null run to remain centered at zero.

Do not promote a checkpoint from this gate: checkpoint strength and resolver
value are independent decisions. After a new no-limp blueprint passes its
search-off gate, repeat the same Phase 4 on/off confirmation on that exact
checkpoint.

Engineering screen completed on 2026-07-26 against the frozen 200bb
scalar@118k champion:

- Phase 4 on minus off: **+26.88 bb/100**, 95% CI
  **[−45.10,+98.85]**, 200 hands. This is a non-regression screen, not proof
  of improvement.
- **71/71** attempted exact river decisions resolved with no fallback.
- Mean full-decision latency was **2.84 seconds**; maximum was **4.62
  seconds** under the strict 6-second budget.
- The max-margin gadget iteration is CUDA-graph captured. This preserves the
  exact-card/safe-gadget math while removing Windows GPU kernel-launch
  overhead.

### Phase 4 larger confirmation result

A fresh-seed, 3,000-pair / 6,000-hand on/off confirmation completed on
2026-07-26 against the same frozen scalar@118k checkpoint:

- Phase 4 on minus off: **+7.62 bb/100**, 95% CI
  **[−21.73,+36.97]** (`KEEP`), so the strength requirement did not pass.
- Exact river resolving succeeded on **1,864/1,889** attempts.
- The **25 fallbacks (1.32%)** exceeded the ≤1% eligibility limit.
- Mean full-decision latency improved to **1.93 seconds**; maximum was
  **5.10 seconds**, still within the strict 6-second budget.
- Every fallback reported the same cause:
  `blueprint projection reached an incompatible public state`.
- Phase 4 was not eligible and remains off in normal serving.

The defect is in the blueprint adapter. The exact resolver uses real stacks
and a richer size menu, while the coarse blueprint uses translated public
states. `_project_blueprint` currently assumes both downstream trees retain
the same actor/street topology. It also maps unavailable raises differently
from normal serving. In shallow-stack, all-in, raise-cap, or off-tree branches,
the coarse blueprint can terminate while the exact tree still has a decision.

Repair requirements:

1. share one action-translation policy between serving and Phase 4;
2. replace the identical-topology assumption with a complete projected
   baseline policy;
3. when the coarse blueprint path ends, use the serving agent's legal
   safe-default policy for the remaining exact subtree instead of aborting;
4. log the exact node/action/stack/SPR mismatch; and
5. add shallow-stack, all-in, raise-cap, and off-tree regression tests.

This repair does not require blueprint retraining. After it lands, run a small
engineering screen requiring zero projection failures and acceptable latency;
repeat a large confirmation only if the screen passes.

Full report:
`backend/data/gpu_blueprint_200bb/evaluations/phase4-confirm-3000pairs-6s.json`

## Phase 5 — CFV network v2

Proceed only after the blueprint and river resolver improve:

- add a reproducible bucket-network training command;
- enumerate legal river cards for target evaluation;
- raise target solves toward 2,000–4,000 iterations per player;
- measure target repeatability before fitting the network;
- generate most states from actual blueprint/search public belief states;
- split validation by board, pot/SPR, range source, and solver seed;
- evaluate regret-weighted and action-changing error, not only MAE;
- add uncertainty-aware blueprint fallback; and
- scale data in stages only while held-out and end-to-end results improve.

Reference:
[ReBeL](https://arxiv.org/pdf/2007.13544)

The acceptance criterion is a positive confirmatory real-solve duel, not a
lower validation loss by itself.

## Phase 6 — Solver schedule and efficiency experiments

- Compare current DCFR averaging with delayed linear averaging.
- Compare LCFR for chance-sampled blueprint training.
- Try stratified/canonical public-board sampling.
- Evaluate multiple independent training seeds and robust finalist mixtures.
- Reduce transient reach/value memory after table sharding.
- Automate gates, resource guards, and stop-on-plateau behavior.

Reference:
[Discounted CFR](https://arxiv.org/abs/1809.04040)

## Scope

In scope:

- 100bb and 200bb heads-up no-limit Hold'em;
- the no-limp agent policy;
- the RTX 3060 12GB resource constraint;
- blueprint training, evaluation, exact river search, and eventual CFV search.

Out of scope until the preceding phases pass:

- returning to the PPO trainer;
- more bucketed turn search;
- preflop neural search;
- opponent-specific exploitation; and
- globally widening the action menu without a memory/EV analysis.

## Priority

1. Evaluation repair.
2. True potential-aware/OCHS abstraction.
3. Exact-card river continual resolving.
4. State-dependent action abstraction.
5. CFV v2.
6. CFR schedule tuning.

## Expected implementation areas

- `backend/solver/gpu/train.py`
- `backend/solver/gpu/cfr.py`
- `backend/solver/gpu/deals.py`
- `backend/solver/gpu/tree.py`
- `backend/eval/duel.py`
- `backend/eval/lbr.py`
- `backend/eval/aivat.py`
- `backend/agents/gpu_blueprint_agent.py`
- `backend/search/`
- `backend/cfv/`
- a unified experiment/gate orchestrator under `backend/eval/`

Final frontend build command after implementation:

```powershell
Set-Location frontend
npm run build
```
