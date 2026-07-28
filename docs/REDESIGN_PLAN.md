# Holdem Agent Redesign Plan

**Date:** 2026-07-17
**Status:** Implemented 2026-07-17 (Phases 0-5 built and tested; blueprint training ongoing)

> **Superseded for current state (2026-07-22):** this plan was executed and then
> superseded by the dense GPU trainer (`docs/GPU_CFR_PLAN.md`). For the live
> architecture, verified-correctness table, strength numbers, and operations, see
> **`docs/STATUS.md`**; for what we build next and why, see
> **`docs/RESEARCH_ROADMAP.md`**. Notable deltas vs this document: the CPU MCCFR
> blueprint/serving path became the fallback (GPU blueprint serves); river-only
> unsafe re-solving grew into CUDA-graph turn/river re-solving but is **disabled**
> — measured net regression, safe re-solving is the planned replacement.

> **Implementation notes (2026-07-17).** All phases landed in one pass:
> - Phase 0: bf16 autocast + logit repair in `learning.py`; improvement-loop skill frozen.
> - Phase 1: `backend/solver/` — Linear MCCFR (external sampling, iteration-scaled pruning),
>   exact best response; validated on Kuhn (<0.005 exploitability, value = -1/18) and Leduc.
> - Phase 2: `backend/abstraction/` — 169 lossless preflop, EMD k-means flop/turn (Numba equity
>   kernels ~0.7 ms/hand), river equity quantiles, action menu + pseudo-harmonic translation.
> - Phase 3: `backend/solver/holdem.py` + `blueprint.py` — abstracted 50 bb HUNL, checkpointable
>   trainer (`python -m backend.solver.blueprint --iterations N`), ~10 iters/s single-process.
> - Phase 4: `backend/agents/blueprint_agent.py` served through `main.py` automatically when
>   `backend/data/blueprint/` artifacts exist (`/api/health` reports which agent serves).
> - Phase 5: `backend/search/` — blueprint range tracking + exact-cards river CFR re-solve,
>   on by default in the agent (unsafe re-solving v1; safe depth-limited turn solving is next).
> - Eval: `backend/eval/` — duplicate-deal style benchmark (`python -m backend.eval.benchmark`)
>   and LBR probe (`python -m backend.eval.lbr`).
> - Deviations from the letter of the plan: the blueprint trainer is single-process for now
>   (multiprocess merge is the next optimization); AIVAT is approximated by duplicate deals;
>   the Slumbot API harness is not yet wired.
> - Next: keep training the blueprint (hundreds of thousands to millions of iterations),
>   benchmark per checkpoint, then multiprocess traversals and safe turn re-solving.
**Verdict:** Replace the PPO self-play trainer with a CFR-based blueprint + search architecture. Keep the game engine, vectorized kernels, evaluation styles, and API contract.

---

## 1. Diagnosis — why the current agent is not improving

### 1.1 Evidence from training reports (48 reports analyzed)

| Signal | Observation |
|---|---|
| Adversarial LCB (bb/100) | Stuck between −65 and −345 across ~1,263 updates; never positive |
| Restricted best-response proxy | Always negative (−4 to −942); persistent exploitability |
| Champion promotions | **1 promotion in the entire training history**; every other eval held by gates |
| Composite quality (latest run) | Decayed monotonically 0.10 → 0.0 across updates 3→39 — training makes the model worse |
| July 17 reset | Update counter 1263 → 22, champion version 1 → 0 (fresh restart), followed by two crashes |
| Crashes | `FloatingPointError: Policy logits contain non-finite values` ×2, plus earlier safety stops and rollout cache errors |

### 1.2 Immediate crash cause (mechanical)

- Training runs under **fp16** autocast (`learning.py:3895`); the GRU + transformer trunk can overflow fp16's ~65504 range.
- `masked_distribution` (`learning.py:960`) **hard-raises** on any non-finite logit. It is called from **6 sites**; the KL-guard/rollback/retry machinery protects only the post-step path — the **entry forward of every PPO minibatch (`:5338`) and all inference paths (`:1071/:1112/:1147/:1582/:3056`) are unguarded**, including the live `/api/game/action` serving path.
- Masked `logsumexp` numerators in the teacher losses are not uniformly guarded (`-inf − (-inf) = NaN`).

### 1.3 Structural cause (why metrics decline even when it doesn't crash)

The PPO master loss (`learning.py:5513`) is **13 weighted terms in one line** — ten of them hand-authored preflop teacher/contrastive shaping penalties — followed by **8 separate auxiliary optimizer passes** (Deep-CFR heads, subgame policy, oracle distillation, search value, counterfactual value, self-imitation, belief, hard-spot) all writing into one 3.6M-parameter network, defended only by a tiny entropy coefficient (0.002–0.022). **Four distinct "teacher" families** push the policy in different directions. The shaping cocktail has overwhelmed the actual win-EV objective. This grew from an automated per-report-warning patching loop (`.codex/skills/holdem-training-improvement`) — each patch addressed a symptom and added a subsystem.

### 1.4 Fundamental cause (algorithm choice)

The literature is unambiguous about vanilla PPO self-play in poker:

1. **No last-iterate convergence** — self-play gradient dynamics orbit the equilibrium; the *average* strategy converges, the current network does not. CFR's entire design is that time-averaged strategy → Nash; PPO discards the average. (Wang et al. ICLR 2025, arXiv:2408.00751)
2. **Non-stationarity** — each update changes the opponent; the agent chases exploits of its recent self (rock-paper-scissors cycling). (Heinrich & Silver 2016, arXiv:1603.01121)
3. **Extreme reward variance** — NLHE outcomes are luck-dominated; PG estimates need enormous batches. AlphaHoldem needed "trinal-clip" PPO + a K-best opponent pool precisely because vanilla PPO diverges. (AAAI 2022)
4. **Self-play winrate hides exploitability** — LBR showed bots with strong head-to-head records exploitable for >1,000 mbb/h. (Lisý & Bowling, arXiv:1612.07547)

**The observed instability is the textbook outcome, not a bug.** Every consumer-budget system that beat strong benchmarks used the same recipe: **abstraction + Linear MCCFR blueprint + depth-limited search** —
- **Pluribus** (Science 2019): superhuman 6-max, blueprint on 64 CPU cores in 8 days (~$144), **zero GPUs**.
- **Modicum** (NeurIPS 2018): beat Slumbot-class on a **4-core CPU + 16 GB RAM** (700 core-hours).
- **DecisionHoldem** (arXiv:2201.11580, open source): +730 mbb/h vs Slumbot, 3–4 days on 48 CPU cores.

This recipe is **CPU/RAM-bound**. The RTX 3060 is best used for abstraction clustering (equity rollouts + k-means) and, later, a search value network.

---

## 2. Target architecture (v2)

```
┌────────────────────────────────────────────────────────────────┐
│                       serving (main.py, unchanged API)          │
│   /api/game/*  →  BlueprintAgent (+ SearchAgent when enabled)   │
└──────────────▲─────────────────────────────▲───────────────────┘
               │                             │
   ┌───────────┴───────────┐     ┌───────────┴────────────┐
   │  blueprint/            │     │  search/                │
   │  strategy store        │     │  river exact re-solve   │
   │  (avg strategy per     │     │  → safe depth-limited   │
   │   infoset bucket)      │     │    turn/flop re-solve   │
   └───────────▲───────────┘     └───────────▲────────────┘
               │                             │
   ┌───────────┴─────────────────────────────┴────────────┐
   │  solver/  Linear MCCFR (external sampling, pruning)   │
   │  multiprocess CPU workers, checkpointable             │
   └───────────▲───────────────────────────────────────────┘
               │
   ┌───────────┴───────────────────────────────────────────┐
   │  abstraction/                                          │
   │  cards: 169 preflop lossless · EMD k-means flop/turn   │
   │         · OCHS river · suit-isomorphism indexing       │
   │  actions: fold/call/{0.33,0.5,0.75,1,2}×pot/all-in     │
   │         + pseudo-harmonic translation for off-tree     │
   └───────────▲───────────────────────────────────────────┘
               │
   ┌───────────┴───────────────────────────────────────────┐
   │  engine (KEPT): poker.py · vectorized_engine.py ·      │
   │  rollout_arena.py · benchmark styles                   │
   └────────────────────────────────────────────────────────┘
   ┌────────────────────────────────────────────────────────┐
   │  eval/ (NEW): AIVAT variance reduction · LBR probe ·   │
   │  fixed-style suite (kept) · Slumbot API harness        │
   └────────────────────────────────────────────────────────┘
```

### 2.1 Keep / delete / build

| Keep as-is | Delete / archive | Build new |
|---|---|---|
| `poker.py` (rules engine) | `learning.py` trainer internals (7.7k lines) | `abstraction/` card + action abstraction |
| `vectorized_engine.py` (Numba kernels + parity tests) | 13-term loss, 8 auxiliary passes, 4 teacher families | `solver/` Linear MCCFR blueprint trainer |
| `rollout_arena.py` | populations/league/exploiters/promotion gates | `search/` river re-solve → depth-limited solving |
| `main.py` API layer (9 endpoints, ~10 symbols) | fp16 PPO machinery, trust regions, recovery anchors | `eval/` AIVAT + LBR + Slumbot harness |
| Fixed-style bots (as eval opponents) | 180-dim observation w/ embedded MC equity (`rl_env.py`) | `BlueprintAgent` serving adapter |
| `tests/` parity tests | `abstract_solver.py` (already disabled/dead) | infoset bucket indexer (Waugh isomorphism) |
| `abstract_cfr.py` (reference CFR+, reusable for unit tests) | checkpoint migration machinery (v25/26) | new checkpoint format (regret + strategy tables) |

The frontend is insulated: it depends only on the `GameState` snapshot shape and the training-status JSON. `run.outcome`/`telemetry` report fields can be preserved with a slimmed schema (bump `schema_version`).

### 2.2 Card abstraction (GPU-accelerated — this is where the 3060 earns its keep)

- **Preflop:** 169 lossless strategically-distinct hands.
- **Flop/turn:** potential-aware k-means with **earth-mover's distance on next-street equity distributions** (not scalar E[HS] — it conflates made hands with draws). Ganzfried & Sandholm AAAI-14.
- **River:** k-means on OCHS vectors (equity vs ~8 opponent-range clusters).
- **Indexing:** suit-isomorphism (Waugh) to collapse the deal space.
- **Bucket counts** (DecisionHoldem used 169/50k/5k/1k; start smaller): `169 / 5k / 5k / 1k`, sized to fit regret tables in RAM (see §4 budget).
- Equity rollouts batched on GPU via existing Numba/torch kernels; k-means in PyTorch.

### 2.3 Action abstraction

- Postflop: fold, check/call, bets {0.33, 0.5, 0.75, 1.0, 2.0}×pot, all-in; fewer sizes deeper in tree.
- Preflop: richer raise grid (2, 2.5, 3, 4×BB opens; 3-bet/4-bet/5-bet sizes; all-in).
- **Pseudo-harmonic action translation** (Ganzfried & Sandholm) to map opponent off-tree bets; later superseded on turn/river by re-solving.

### 2.4 Blueprint solver

- **External-sampling Linear CFR** with regret-based pruning (skip actions with very negative cumulative regret ~95% of iterations, à la Pluribus).
- Multiprocess CPU workers over shared-memory regret/strategy arrays (numpy memmap or shared_memory); Numba for the traversal inner loop.
- Checkpoint = regret + average-strategy tables; **monotone, resumable convergence** — the opposite of the current instability.
- Target: 100M–1B traversal iterations over 1–3 weeks of desktop CPU time.

### 2.5 Real-time search (biggest strength multiplier — phased)

1. **Phase A — river exact re-solving:** rivers are small enough to solve exactly with CFR+ given both ranges; reference implementations: TexasSolver, desktop-postflop, noambrown/poker_solver.
2. **Phase B — safe depth-limited turn/flop re-solving (Modicum):** at the depth limit the opponent chooses among k≈4 continuation strategies (blueprint, always-fold-biased, always-call-biased, always-raise-biased); nested safe re-solving so exploitability cannot increase (Brown & Sandholm, arXiv:1705.02955).
3. **Phase C (stretch) — neural value net at the depth limit** (DeepStack/Ruse direction): train a turn/river counterfactual value network on solved situations; this is the only path from "beats Slumbot" toward Ruse/GTO-Wizard-class play, and where the GPU returns.

### 2.6 Evaluation harness (non-negotiable — the current gates measure the wrong thing)

- **AIVAT** variance reduction (arXiv:1612.06915): >10× fewer hands for significance; on this budget, evaluation hands are as scarce as training hands.
- **LBR probe** (arXiv:1612.07547): cheap exploitability lower bound; the sanity check that head-to-head wins aren't hiding a paper tiger.
- **Fixed-style suite:** keep the existing 8 scripted styles as regression smoke tests (report bb/100 with CIs, as today).
- **Slumbot API** (slumbot.com): the standard free external benchmark; track mbb/hand per checkpoint.
- Small-game **ground truth**: run the solver on Kuhn and Leduc first and compare exploitability to known values (OpenSpiel reference numbers).

---

## 3. Phased roadmap

### Phase 0 — Stop the bleeding (0.5–1 day)
Keep the app serving while v2 is built:
- Guard all 6 `masked_distribution` call sites: sanitize non-finite logits (replace with uniform-over-legal + log) instead of raising on inference paths; keep the raise only as a training-side diagnostic.
- Switch autocast fp16 → **bf16** (RTX 3060 = Ampere, native bf16); removes the overflow class.
- Freeze the improvement-loop skill (`.codex/skills/holdem-training-improvement`) — no more per-warning patches to the old trainer.
- **Do not** invest further tuning in the old loss stack.

### Phase 1 — Scaffolding + ground truth (week 1)
- New packages: `backend/abstraction/`, `backend/solver/`, `backend/search/`, `backend/eval/`, `backend/agents/`.
- Implement external-sampling Linear CFR generically; validate on **Kuhn** (exact Nash known) and **Leduc** (exploitability curve vs OpenSpiel reference). Acceptance: Kuhn exploitability < 1 mbb; Leduc curve matches published DCFR shape.
- Port the existing `abstract_cfr.py` tests as regression anchors.

### Phase 2 — Abstraction pipeline (weeks 1–3)
- Suit-isomorphism infoset indexer (unit-tested against brute-force enumeration on small boards).
- GPU equity-distribution rollouts; EMD k-means flop/turn; OCHS river clusters.
- Acceptance: bucket quality audit — within-bucket equity-distribution variance, spot-check known hand pairs (e.g. KQs vs 66 must separate); memory budget check (§4).

### Phase 3 — HUNL blueprint training (weeks 3–6, mostly wall-clock)
- Linear MCCFR with pruning on the abstraction, multiprocess, checkpointed.
- Continuous eval: fixed-style suite + LBR probe per checkpoint; convergence dashboard (reuse the existing report JSON schema, slimmed).
- Acceptance gates (in order):
  1. Positive bb/100 with CI vs **all 8 fixed styles** (the current model loses to 7 of 8).
  2. LBR exploitability trending down across checkpoints.
  3. Slumbot harness connected; establish baseline mbb/hand.

### Phase 4 — Serving integration (week 6)
- `BlueprintAgent` implementing the `NeuralAgent.select/execute/observe_completed_hand` contract; pseudo-harmonic translation for off-tree opponent bets; sample from average strategy.
- Keep `strategic_champion.pt` path semantics or introduce a new artifact name behind `import-model`.
- Frontend unchanged. Old checkpoints archived (`checkpoint_archives/`).

### Phase 5 — Search (weeks 6–10)
- River exact re-solve (CFR+ on the actual river subgame with tracked ranges) → measure uplift vs blueprint-only on the eval suite.
- Safe depth-limited turn re-solve with k continuation strategies.
- Acceptance: **positive vs Slumbot**, the documented outcome of this recipe at this budget (Modicum +11±9, DecisionHoldem +730 mbb/h).

### Phase 6 (optional/stretch) — Neural value net + exploitation
- Train turn CFV network on solved situations (DeepStack-lite / Ruse direction) to deepen search.
- Opponent modeling for *exploitative* play vs the fixed styles (safe-exploitation literature: Ganzfried & Sandholm TEAC 2015) — only after the equilibrium core is strong.

### Parallel exploratory track (only if the deep-RL stack should survive)
If a neural end-to-end agent remains a goal: strip the trainer to **plain PPO + MMD magnet regularization** (KL toward an EMA reference policy, annealed — Sokota et al. ICLR 2023; Rudolph et al. arXiv:2502.08938) + trinal-clip + K-best checkpoint league (AlphaHoldem), evaluated by the Phase-3 harness. Expected outcome: intermediate strength, no convergence guarantee. **Do not run this instead of Phases 1–5; it is strictly secondary.**

---

## 4. Resource budget

| Resource | Use | Budget check |
|---|---|---|
| RAM | Regret + strategy tables: (#infoset buckets × actions) × 2 arrays × 8 bytes. 169 + 5k·(flop) + 5k·(turn) + 1k·(river) buckets × betting sequences — target ≤ 32–64 GB via bucket counts and half-precision strategy sums | Size before training; shrink buckets if needed |
| CPU | MCCFR traversals (the real budget): 8–16 cores × 1–3 weeks | Numba inner loop; measure iterations/sec in Phase 1 |
| GPU (3060 12GB) | Abstraction equity rollouts + k-means; Phase 6 value net | Not on the critical path for strength |
| Disk | Checkpoints are tables, not 500MB torch blobs; a few GB | Replaces 622MB `.pt` pairs |

---

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Abstraction bugs silently cap strength | Phase-1 ground truth on Kuhn/Leduc; bucket audits; LBR probe catches gross exploitability |
| RAM blow-up at chosen bucket counts | Compute table sizes analytically in Phase 2 before training; imperfect recall keeps tree bounded |
| Windows multiprocessing friction | Use `spawn` + shared_memory arrays (already proven in current rollout pool); Numba nopython workers |
| Slumbot API availability/limits | Fixed-style suite + LBR remain the primary continuous signals; Slumbot for milestone checks |
| Temptation to re-add shaping terms | Rule: any new loss/subsystem requires an A/B ablation showing eval uplift with CIs before merging |
| Old model regression during transition | Phase 0 guards keep the current champion serving until Phase 4 swaps it |

---

## 6. Key references

- Pluribus — Brown & Sandholm, Science 2019 (blueprint recipe, $144 compute)
- Modicum — Brown, Sandholm, Amos, NeurIPS 2018, arXiv:1805.08195 (depth-limited solving on a 4-core CPU)
- DecisionHoldem — arXiv:2201.11580 + github.com/AI-Decision/DecisionHoldem (open-source full pipeline, beat Slumbot)
- Discounted/Linear CFR — Brown & Sandholm, AAAI 2019, arXiv:1809.04040
- Potential-aware EMD abstraction — Ganzfried & Sandholm, AAAI 2014
- Safe & nested subgame solving — Brown & Sandholm, NeurIPS 2017, arXiv:1705.02955
- AIVAT — Burch et al., AAAI 2018, arXiv:1612.06915; LBR — Lisý & Bowling, arXiv:1612.07547
- MMD — Sokota et al., ICLR 2023, arXiv:2206.05825; PG reevaluation — Rudolph et al., arXiv:2502.08938
- AlphaHoldem — Zhao et al., AAAI 2022 (trinal-clip PPO, K-best league)
- Deep CFR / SD-CFR — arXiv:1811.00164 / arXiv:1901.07621 + github.com/EricSteinberger (PokerRL)
- Reference code: ozzi7/Poker-MCCFRM (Pluribus-recipe pipeline), b-inary/desktop-postflop (DCFR solver), bupticybee/TexasSolver
