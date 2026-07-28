# Agent Status & Conclusions — living document

**Last updated:** 2026-07-27

## 1. What is serving right now (the best verified player)

`MultiStackBlueprintAgent`, **blueprint-only** (search off — see §4):

| Depth | Model | Why it holds the slot |
|---|---|---|
| 100bb | **histogram@30k** (`gpu_blueprint/champion.npz`) | Ties the fully-trained scalar@100k head-to-head (+7 [−16,+31]) AND fixes the draw-fold leak class (hand #326: fold 97.8% → call 95.9%) |
| 200bb | **scalar@118k** (`gpu_blueprint_200bb/champion.npz`) | Plateau champion; all checkpoints 47k–118k are statistical ties |

Honest absolute strength (styles field, corrected for the +75 artifact): ~+130 bb/100
mean at 100bb, ~+290 at 200bb. Known weakness: tight-aggressive opponents
(−160..−280) — an abstraction ceiling, not a training gap.

### FROZEN BASELINES — measured 2026-07-27 against real instruments

These champions are now permanent measurement baselines. The styles-field numbers
above are retained for continuity but are **not** evidence about equilibrium
quality: scripted opponents do not probe. See
`docs/PLAN_V2_STRONGEST_PLAYER.md` §9 for method and full caveats.

| instrument | 200bb scalar@118k | 100bb histogram@30k |
|---|---:|---:|
| **LBR exploitability** (400 pairs, multi-size probe) | **+291.23 bb/100** [+79.10, +503.36] | **+137.58 bb/100** [−3.57, +278.73] |
| LBR probe fallback rate | 1.11% | 0.58% |
| **Slumbot, 20,000 hands** (real API, search off) | **−18.33 bb/100** [−40.41, +3.74] | not run (Slumbot is 200bb native) |

LBR at the shallower target depths, both served today by the 100bb champion:
**20bb +130.31** [+95.22, +165.40] (interval clears zero) and **50bb +85.05**
[−12.80, +182.90].

#### The completed Slumbot baseline (2026-07-28)

20,000 hands, **0 exclusions, 0 board desyncs**, σ = 15.93 bb/hand, 11.6 h.
Button −11.43, big blind −25.24. Precision ±22.1 bb/100, matching the ±22.7
predicted from the measured spread.

Two instrument validations worth keeping: the `client_pos` mapping came out
exactly balanced (10,000 each way), and the timing split shows **99.0% of the run
was waiting on the Slumbot API** (41,253 s of 41,653; agent compute 371 s,
mirroring 11 s) — the harness contributes nothing measurable, which is also why it
ran alongside GPU datagen without interference.

**This vindicates refusing the early reads.** The 400-hand screen said
**+84.88 bb/100**; the truth is **−18.33** — a 103 bb/100 swing. Every downstream
decision would have been built on a number that was wrong by more than the entire
effect anyone is hunting.

Three things to carry forward:

1. **LBR wins ~2,910 mbb/hand against the serving champion.** Positive means
   exploitable, and LBR is a *restricted* response, so true exploitability is
   ≥ that. This is the abstraction ceiling as a number: the agent is in the
   2016-ACPC class, not the resolving-agent class (Supremus *beats* LBR by 951).
2. **The Slumbot +84.88 is a tie, not a win.** Median hand −0.5 bb; the mean is
   two won all-in pots out of 400 hands. σ = 16.34 bb/hand at 200bb, so the CI is
   ±160 bb/100 at this sample size.
3. **LBR and Slumbot disagreeing is informative, not contradictory.** Slumbot is
   also a non-probing abstraction agent, so two mutually exploitable agents sit
   near even head-to-head. Head-to-head result ≠ distance from equilibrium.
   **LBR is the north star from here; Slumbot is the external sanity check.**

Instrument status: the Slumbot harness is NULL-tested offline
(`tests/test_slumbot_harness.py`, 12 tests; always-fold from the button reads
exactly −50.0000 bb/100) and LBR is validated against four deliberately broken
reference agents (`tests/test_lbr_validation.py`; always-fold reads exactly
+75.0000 bb/100). Two bugs were fixed in the process: blind double-posting in the
Slumbot mirror (corrupted stack depth on every out-of-position hand) and an AIVAT
correction that was silently zeroed on all-in run-outs.

**Experiment state:** no training or evaluation run is active, and no
challenger was promoted.

| Experiment | Screening result | Confirmatory result | Decision |
|---|---:|---:|---|
| Clean 200bb no-limp @10k | +5.50 [−61.27,+72.28] | +1.57 [−26.73,+29.87] | Keep training; no promotion |
| Clean 200bb no-limp @20k | +75.52 [+14.28,+136.75] | −12.78 [−44.27,+18.71] | Statistical tie; no promotion |
| Clean 200bb no-limp @40k | +67.56 [−2.17,+137.29] | −9.58 [−41.32,+22.15] | Plateau; stop this run |
| Combined v3 + Phase 3 @5k | −94.92 [−166.36,−23.48] | Not run after failed screen | Retired |
| Phase 4 exact river, 3,000 pairs | — | +7.62 [−21.73,+36.97] | Inconclusive; repair projection |

The clean no-limp challenger passed mapping and relative-LBR guards at 20k
and 40k, but its disjoint confirmation blocks did not show improvement over
the unrestricted 118k champion. The similar 20k and 40k estimates are evidence
of a plateau under this configuration, so more identical training is not the
next step.

## 2. Audited conclusions — what actually works

Everything below survived NULL-tested, timing-sane instruments (§5).

1. **The GPU vector-CFR solver is correct** — proven by an independent
   Kuhn/Leduc-validated best response reading 0.0 mbb on a converged control
   (2026-07-21), regression-guarded (`tests/test_gpu_convergence.py`).
2. **Blueprint training converges and then stops paying.** At both depths,
   head-to-head duels show plateau (200bb: 47k ≈ 118k, −6 [−34,+22]). More
   iterations past plateau buy nothing. The 10k-gate + stop-on-plateau loop is
   the right way to train (`backend/eval/duel.py` + monitors).
3. **Abstraction quality is a real lever.** Histogram-EMD bucketing
   (`DealSampler(histogram=True)`) reached parity with a 3.3x-trained scalar
   model and fixed a whole leak class (draws folding to correct-odds bets:
   hands #222/#245/#27/#326). It is the default abstraction now.
4. **Rule/menu changes are config-level.** `no_limp`, `preflop_raise_cap`,
   sizing menus — the tree builder enforces house rules structurally; the
   translator maps off-tree opponent actions (e.g. limps) to nearest branches.

## 3. Audited conclusions — what does NOT work (and why)

1. **Bucketed subgame re-solving (turn/river "live search") does not help.**
   Real-solve duels vs the same blueprint: 120 iters −31 [−95,+33] (wash);
   **500 iters −86 [−150,−22] (significant regression)** — deeper convergence
   onto the re-solved subgame plays WORSE. Since the duel opponent literally
   played the blueprint, opponent-range error is excluded; the prime suspect is
   **self-range inconsistency**: the range tracker assumes the agent's own past
   actions followed the blueprint, so a searching agent solves subgames
   conditioned on a falsified own-history. Design retired; any successor needs
   (a) self-range tracked through actual past solutions and (b) a genuine
   information edge over the buckets — e.g. **exact-card river re-solving**.
   Phase 4 now implements that successor behind an off-by-default switch.
2. **Phase 4 exact-card river resolving has not cleared its gate yet.** The
   3,000-pair confirmation estimated +7.62 bb/100 [−21.73,+36.97], a
   statistical tie. It resolved 1,864 of 1,889 river attempts; all 25
   fallbacks came from the blueprint projection reaching an incompatible
   public state. The 1.32% fallback rate misses the ≤1% eligibility rule.
   This is a projection/translation defect, not a timeout or retraining issue:
   mean latency was 1.93s and maximum latency was 5.10s under the 6s budget.
   Repair the adapter and re-screen before any further large confirmation.
3. **CFV-net flop search (v0) does not help**: −65 [−157,+27] with verified
   real solves. The pipeline is mechanically proven (horizon plumbing
   bit-identical to trusted kernels; net val MAE 9.3bb vs 24bb zero-baseline)
   but target noise (4-runout CFV evaluation) caps net quality below
   usefulness. Revival path: regenerate targets with more runouts —
   datagen is resumable. Parked.
4. **Raw-combo net I/O cannot generalize** at feasible sample counts (scored
   below the zero-predictor); bucket-level I/O (2×169) is the workable
   representation (+58% vs zero baseline on identical data).
5. **The +511 bb/100 "search uplift" and every pre-2026-07-23 absolute eval
   number were artifacts** — see §5.

## 4. The great eval corrections (why numbers before 2026-07-24 are suspect)

Three instrument bugs, all user-instinct-triggered ("too consistent", "are you
sure it's running?"), all fixed and regression-guarded:

1. **+75 bb/100 blind inflation** (duel.py + benchmark.py measured winnings
   from post-blind stacks). Symptom: ten straight gates reading +70..+104. A
   model-vs-ITSELF duel read +78.8. Fix: baseline = full starting stack;
   guard: `tests/test_duel_null.py` (self-duel must read 0; old code reads +75).
2. **Stale subgame cache via recycled `id(game)`** — eval loops hit dead
   engines' solutions; "search" arms ran FASTER than no-search arms (the tell).
   Fix: monotonic `game._search_uid`. This poisoned the entire 2026-07-23
   search A/B (+347/+858/+752 — retracted).
3. **Silent flop-solve discard** — street-1 decisions fell into the river
   bucket branch → every flop solve paid for and thrown away. Fix + the first
   flop A/B (6-minute "search" run = blueprint speed) exposed it.

**Standing rules:** every new harness gets a NULL test before its numbers are
believed; runtimes must be consistent with claimed work (a search eval that
finishes at blueprint speed didn't search); decision-level verification (logs
prove the choice came from the claimed component). Also: `CUDA_VISIBLE_DEVICES=""`
does NOT hide GPUs on this Windows/torch — never rely on it.

## 5. Infrastructure that exists and is trusted

- **Duel gate**: `python -m backend.eval.duel --data-dir D --stack-bb N --pairs 3000 [--promote]`
  — NULL-tested head-to-head with auto-promotion; the only promotion signal.
- **Per-decision logger**: every served move in `backend/data/server-debug.jsonl`
  (node, bucket, exact_match, full mix, search_active) — answers "why did it do X?".
- **Monitors**: training runs get 10k gates + stop-on-plateau + VRAM/RAM guards.
- **Depth-limited solving**: HORIZON trees + evaluators (proven plumbing) —
  ready for any future value-function work.
- **Safe re-solve gadget (v1/v2) + AIVAT chance-variates**: built and tested;
  value gated on search being worth anything at all.
- **CFV pipeline**: situations→solve→bucketize→train (7,750 solved turn
  samples in `backend/data/cfv/turn.npz`, net in `bucket_net.pt`).

## 6. VRAM engineering facts (RTX 3060 12GB)

- Tree memory ≈ persistent (nodes×169×A×8B) + transients — empirically
  ~60-75KB/node total; **budget ≤ ~150k nodes** for comfort with the server up.
- Full 4-size no-limp menu at 200bb = 1.3M nodes (~40GB): impossible; 310k
  nodes measured 12GB+spill (overflow incident 2026-07-25). Per-street raise
  caps are the fix (preflop depth is cheap; postflop explodes).
- Batching (`batch_boards`) multiplies transients; run big trees at batch=1.
- No test suites or GPU side-jobs during training (starvation incidents
  2026-07-19/22/23); heavy CPU evals one at a time.

## 7. Roadmap position

- Done: solver, serving blueprints, histogram abstraction, honest eval stack,
  house-rules capability, search post-mortem, clean no-limp gates through 40k,
  the v3 + Phase 3 5k screen, and the 3,000-pair Phase 4 confirmation.
- Next engineering task: make Phase 4 blueprint projection use the same action
  translation semantics as normal serving and provide a complete safe-default
  policy when coarse and exact river trees have different downstream topology.
  Add shallow-stack, all-in, raise-cap, and off-tree regression tests.
- Next evaluation: after the projection repair, run a small engineering screen
  requiring zero projection failures and acceptable latency. Repeat a
  3,000-pair confirmation only if that screen passes. Phase 4 remains disabled
  in normal serving.
- Stopped: further iterations on the current clean no-limp checkpoint and the
  combined v3 + Phase 3 challenger. Neither is promotion-eligible.
- Parked with revival criteria: CFV net (better targets), bucketed turn
  search, AIVAT decision-variates, and Slumbot rematch (worth doing once any
  new model or full stack clears its gates).
