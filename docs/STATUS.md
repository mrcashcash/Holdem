# Agent Status & Conclusions — living document

**Last updated:** 2026-07-29

## 1. What is serving right now (the best verified player)

`MultiStackBlueprintAgent` with **exact-card continual resolving ON** for flop,
turn and river (`tools/serve_best.ps1`; see docs/SERVING.md for the full config and
`GET /api/health` for what is actually loaded). The §4 warning about search being
off applied to the *retired bucketed* resolver, which stays off; exact-card
resolving is a different mechanism and is served.

### 2026-07-31 FIRST EXTERNAL PROBING MEASUREMENT — and it questions the serving default

The GTO Wizard AI harness (§5) gives the first number from an opponent that both
**probes** and is **variance-reduced**. 500 hands per arm at 200bb, AIVAT-adjusted,
bootstrap intervals (the distribution is heavy-tailed, so the normal-approximation
interval the harness prints understates uncertainty):

| arm | hands | mean AIVAT bb/100 | bootstrap 95% | median | sd bb/hand |
|---|---:|---:|---|---:|---:|
| resolver **OFF** (blueprint only) | 500 | **−17.00** | [−47.71, +21.22] | −7.50 | 3.92 |
| resolver **ON** (exact-card continual) | 497 | **−52.89** | [−80.04, −30.18] | −14.66 | 2.83 |
| **difference (ON − OFF)** | | **−35.89** | **[−81.38, +3.27]** | | |

`P(ON worse than OFF) = 0.962`. Zero board desyncs in both arms; 3 hands excluded
in the ON arm, all transient API 503s, excluded rather than folded-and-scored.

**Two things this establishes.**

1. **The agent is far closer to a strong resolving opponent than LBR implies.**
   LBR says +252 bb/100 exploitable, yet blueprint-only play is only ~17–31 bb/100
   behind an agent that beat Slumbot by 19.4. Consistent with the §5 point that
   head-to-head result is not distance from equilibrium — but the gap is larger
   than expected.
2. **Exact-card resolving measured WORSE, not better.** It is currently served ON,
   justified in §3.2 "on mechanism and reliability, NOT on a passed gate." This is
   the first external evidence pointing the other way, and it is not cheap: the ON
   arm spent **9,302 s of GPU compute** (18.7 s/hand) versus 12 s for OFF, for a
   worse result.

**What it does NOT yet establish, and why the default should not be flipped on it.**

- The interval includes zero (+3.27), so this is 96% suggestive, not significant.
- The arms are **unpaired** — GTO Wizard deals its own hands and the external API
  admits no common random numbers. Pairing is what made the LBR gate 9x more
  sensitive.
- It is **outlier-sensitive**: dropping each arm's single largest |AIVAT| hand
  moves OFF to −30.74 and ON to −45.45, shrinking the difference from −35.89 to
  **−14.71**. The OFF arm contains a +68.4 bb hand against a 3.92 sd — 17.5σ.

Supporting the direction: the medians (outlier-immune) also favour OFF, −7.50
versus −14.66, and the sign held across all six ON-arm milestones.

**The deciding measurement was then taken and CONFIRMS the regression.**
`tools/continual_search_gate.py --crn`, 300 seat-swapped duplicate pairs at 200bb,
120 resolve iterations (the serving target):

| | result |
|---|---|
| on minus off | **−58.61 bb/100 [−99.85, −17.37]** |
| verdict | **REGRESSION** — the interval clears zero |
| resolves | **208/208, 0 fallbacks** |
| latency ms mean/p90/max | 1425.8 / 1783.8 / 1918.7 |
| CRN null (off vs off) | **+0.00 [+0.00, +0.00]** |

So two independent instruments agree: external probing opponent −35.89 (P=0.962)
and internal paired duel **−58.61 (significant)**. This is not a reliability
failure — every resolve succeeded and it still lost ~59 bb/100.

The §3.2 duels (+17.82 / +54.45 / +28.12) all predate the CRN fix and were
uncoupled. Adding proper pairing did not merely tighten the interval, it
**flipped the sign**.

**A serving-truth correction: at 200bb only the RIVER resolves.** §1 and §7 claim
resolving is on "for flop, turn and river." The street counter reports
`{'street3:resolved': N}` and nothing else, and latency confirms it independently —
max 1918.8 ms cannot contain a 14.23 s flop or 6.90 s turn solve. Flop and turn are
never admitted at 200bb, almost certainly the 12,000-node budget rejecting
deep-stack flop trees. So the measured −58.61 is the cost of **river-only**
resolving at 200bb, and the documented three-street resolver does not exist at that
depth.

That counter could not see flop resolves at all until 2026-07-31 (it watched only
streets 2 and 3), the same blind spot as §4.3 and §4.4. Fixed.

**Also corrected: low GPU utilisation during resolving is NORMAL, not a fault.**
A 200bb river tree is **369 nodes** and holds ~138 MiB; an instrumented probe
confirms `VectorCFR` is constructed with `device='cuda'`. Utilisation near 0% is
what PLAN_V2 already measured ("latency-bound, 123x above its bandwidth floor, GPU
~5% utilized on small trees"). The resolver's cost is the **CPU-side** pipeline —
blueprint projection, bucket computation, showdown scoring — not the GPU kernels.
Consequence: a faster GPU buys little here, and P4 (Parallel CFR) should target the
serial CPU pipeline rather than the card. A 10x latency inflation observed on the
rented box was CPU contention from three concurrent LBR jobs, violating the §6 rule
that heavy CPU evals run one at a time.

### 2026-07-31 the served configuration's honest number

After depth-gating the resolver off at 200bb, the served config was re-measured
against GTO Wizard AI. Because that benchmark plays **only 200bb**, and the gated
config is blueprint-only at 200bb, the "resolver off everywhere" run and the
"depth-gated" run sample the **identical policy** — so they are pooled:

| config | hands | mean AIVAT | bootstrap 95% | median |
|---|---:|---:|---|---:|
| resolver ON at every depth | 497 | −52.89 | [−80.04, −30.18] | −14.66 |
| run 1, off at 200bb | 500 | −17.00 | [−48.05, +21.03] | −7.50 |
| run 2, depth-gated | 499 | −21.36 | [−51.34, −1.17] | −8.25 |
| **pooled — what serves today** | **999** | **−19.18** | **[−40.41, +2.83]** | **−7.84** |

`P(the agent is losing) = 0.961`. **Depth-gating beat resolver-ON-everywhere by
+31.53 bb/100 with P(better) = 0.954**, independently corroborating the internal
paired duel's −58.61 [−99.85, −17.37]. Two unrelated instruments, one conclusion.

Two caveats that bound the claim:

- **This benchmark cannot validate the gating decision as a whole.** GTO Wizard is
  200bb-only, so the resolver-ON choices at 20bb and 100bb rest solely on the
  internal duel (+31.83 at 100bb, −7.11 at 20bb).
- **Accidental replication, and it passed.** The two runs above measure the same
  policy and read −17.00 and −21.36 — a 4.4 bb/100 spread, well inside their
  intervals. That is a free reproducibility check on the harness.

Instrument defect found and fixed in the process: `tools/gtowizard_benchmark.py`
assigned the ROUTER's `continual_search`, whose setter fans out to every
sub-agent, so a `--resolver on` run silently re-enabled resolving at 200bb and
measured a configuration that is not served — nine times slower, for a number
already known. It now defaults to `--resolver default`, which leaves the served
config alone, and logs `resolver by depth` so every run states what actually
plays. This is the same fan-out hazard `backend/agents/serving.py` already warns
about beside the depth-gating pass.

### 2026-07-29 live flop VRAM repair

The exact flop resolver could saturate an RTX 3060 (11.8/12.0 GiB dedicated
VRAM plus shared-memory spill) and then time out. The failure was cumulative:
admission did not model the whole CUDA peak, multi-street projection and final
strategy export were dense, graph pools outlived cleanup, and some opponent
actions caused two consecutive full solves.

The implemented repair adds:

- pre-allocation admission with a 12,000 flop-node ceiling, conservative peak
  estimate, 9.5 GiB process ceiling, and 2 GiB headroom;
- a richest-safe action-menu ladder, with deep-SPR roots starting at the
  compact tier;
- compact loaded blueprints (27.1 MiB at 100bb, 20.6 MiB at 200bb), root plus
  response-frontier-only strategy export, and a 384 MiB showdown workspace cap;
- explicit graph/gadget destruction before CUDA cache cleanup;
- runout-aware cross-street blueprint projection (98.37% detached before,
  7.85% in the new real 200bb passive-entry diagnostic);
- stored response frontiers and child-policy reuse to avoid redundant solves;
- health and decision telemetry for limits, estimates, menu choice, projection,
  reuse, CUDA allocation, and fallback reasons.

### 2026-07-29 exact-equivalent latency pass

The opponent safety-price best response is now CUDA-graph captured instead of
running 40 eager traversals before every safe solve. Runouts are prepared in
the identical RNG order on a producer thread and copied through reusable pinned
buffers; flop/turn samplers and runout-aware blueprint bucket rows are reused
within a continual session. CUDA and the scorer are warmed at server startup.

No quality dial changed: serving remains exact-card FP32 with the same 120
gadget iterations, safety-price iterations, action trees, ranges, and seeds. A
fixed-seed eager/optimized control had max root-policy difference **0.0**.

Representative fresh solves on the RTX 3060 now measure **14.23 s flop
(9,274 nodes), 6.90 s turn (3,926), and 1.70 s river (400)**. The earlier live
sample was 22.78 / 10.64 / 2.45 s on slightly smaller trees. Per-stage timing is
now emitted in every successful resolver diagnostic; see `docs/SERVING.md`.

The river CFV horizon is connected but action-gated, and the checkpoint remains
OFF. The reason it is off has been re-derived: the old figures (0.3766 agreement /
1.1474 policy L1 against a 0.90 / 0.30 bar) came from a gate whose **null passes
it** — an all-zero net scores 0.9269 agreement. Those numbers are retracted as a
pass/fail signal. Under the rebuilt null-anchored gate
(`tools/river_net_gate_v2.py`) the checkpoint stays off for a measured reason: it
**adds** policy error relative to pricing the river at zero. See the 2026-07-31
entries below.

The three frozen blueprints below are the depth-routed base policy. The 20bb
artifact is promoted on disk; the API server was not running during the
2026-07-30 documentation audit, so verify `/api/health` after startup before
claiming a live process loaded it.

| Depth | Model | Why it holds the slot |
|---|---|---|
| 20bb | **histogram@5k** (`gpu_blueprint_20bb/champion.npz`) | Native shallow policy beat the former 100bb fallback by +32.08 [+20.16,+44.01]; 10k–50k were ties or regression |
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
| **LBR, 20,000 pairs** (2026-07-30, supersedes the row above at 200bb) | **+252.45 bb/100** [+223.34, +281.55] | **+118.64** [+99.40, +137.87] |
| LBR probe fallback rate | 1.11% (400 pairs) / 1.36% (20,000) | 0.58% |
| **Slumbot, 20,000 hands** (real API, search off) | **−18.33 bb/100** [−40.41, +3.74] | not run (Slumbot is 200bb native) |

#### Exploitability by depth at 20,000 pairs (2026-07-31)

The full depth curve, each arm 20,000 duplicate pairs, blueprint only. The 20/50/100bb
rows were run on a rented instance that has since been **destroyed**, and
`/workspace` there was overlay storage rather than a volume, so the JSON artifacts
are unrecoverable — these are transcribed from the run logs. Weaker provenance than
a checked-in artifact; re-run before treating any single figure as load-bearing.
The 200bb row was run locally and its artifact survives at
`backend/data/evaluations/lbr-guard-gate-200bb-20k.json`.

| depth | blueprint | LBR exploitability | probe fallback |
|---|---|---:|---:|
| 20bb | native 20bb histogram@5k | **+13.34** [+6.93, +19.75] | 0.35% |
| 50bb | 20bb champion (serving routes 50→20) | **+19.12** [+6.54, +31.70] | 0.79% |
| 100bb | histogram@30k | **+118.64** [+99.40, +137.87] | 0.12% |
| 200bb | scalar@118k | **+252.45** [+223.34, +281.55] | 1.36% |

**Exploitability is overwhelmingly a deep-stack problem.** The probe wins 133
mbb/hand at 20bb and **2,525 at 200bb** — a 19x spread. For scale, Supremus *beats*
LBR by 951 mbb/hand. Shallow play is within sight of the resolving-agent class;
200bb is not. Any effort aimed at overall strength belongs at 100-200bb, and the
50bb figure also vindicates routing 50bb to the native 20bb blueprint rather than
the 100bb one (the old 400-pair figure with the 100bb champion was +85.05).

The all-in geometry guard was A/B'd in the same runs (paired, ON minus OFF,
negative means less exploitable):

| depth | paired delta | pairs differing | verdict |
|---|---:|---:|---|
| 20bb | −0.09 [−0.31, +0.12] | 37 / 20,000 | inconclusive, effect ~nil |
| 50bb | −1.84 [−11.65, +7.98] | **3,014 / 20,000** | inconclusive |
| 100bb | **−5.65 [−10.30, −1.00]** | 164 / 20,000 | **reduces exploitability** |
| 200bb | −4.74 [−10.63, +1.14] | 69 / 20,000 | same direction |

100bb is the guard's first significant positive result, and no depth is harmed. It
still ships OFF because the head-to-head cost (−139 to −244 against a min-raiser)
outweighs it — see §3.6. Two oddities worth chasing: 50bb had **20x more** differing
pairs than any other depth, consistent with real 50bb geometry against a 20bb
abstract tree being the most strained translation; and 50bb's guard-ON fallback rose
to **3.45%** from 0.79%, above the ≤1% rule, which is unexplained.

**The 400-pair LBR figures are imprecise and should not be quoted as headline
numbers.** At 400 pairs the 200bb interval was [+79.10, +503.36] — a width of 424
bb/100. Re-measured at 20,000 pairs it is **+252.45 [+223.34, +281.55]**, a 7.3x
narrower interval, and the point estimate moved 39 bb/100. Every other LBR number
in this document is still a 400-pair reading and carries the same imprecision;
treat their point estimates as indicative, not settled. This does not change the
qualitative conclusion — LBR still wins ~2,500 mbb/hand, so the agent remains in
the 2016-ACPC class, not the resolving-agent class.

Historical shallow-depth LBR with the 100bb champion was **20bb +130.31**
[+95.22,+165.40] and **50bb +85.05** [−12.80,+182.90]. The promoted native
20bb champion measured +22.02 [−25.85,+69.90] on its bootstrap block versus
the fallback's +116.88 [+80.03,+153.72]. These blocks are not interchangeable
absolute estimates; the shared-block difference is the useful comparison.

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
   That successor exists and is now served: `backend/search/continual.py` advances
   the agent's own range from the policy that actually chose each action (fixing
   (a)), and identity 1,326-combo buckets are a genuine information edge over 150
   turn / 30 river buckets (fixing (b)). The retired bucketed path stays off.
2. **Exact-card resolving is served on mechanism and reliability, NOT on a
   passed gate — and that distinction matters.** Its on/off duels are
   *inconclusive*: +17.82 / +54.45 / +28.12 bb/100 at 60/240/240 iterations, every
   interval spanning zero — and all three were run before the CRN coupling bug
   (§4.5) was found, so they were noisier than labelled. What IS established:
   the projection defect is repaired (the 1.32% fallback rate that missed the ≤1%
   rule is now **130/130 resolves with zero fallbacks** across three depths and
   both entry streets, and 1,244/1,244 in the wider run), and it fixes a
   documented leak class (a 19-out draw the 150-bucket blueprint folds 99.1% is
   folded 17.9% by an exact solve, while trash still folds 0.92–0.99 — so it is
   card-level discrimination, not looseness). Nothing has ever measured it as
   harmful. It is served because quality is the stated priority and the mechanism
   is sound, not because a gate passed. Re-running those duels with working CRN is
   an open task.
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
6. **Action translation is a live leak source, and sizing heuristics do not
   patch it** (2026-07-29). `_locate` maps a real hand onto an abstract node by
   translated action sequence and **never compares pot/stack geometry**. ALL-IN
   was the only size-bearing action executed literally (every `raise` re-derives
   its amount from the real pot), so opponent min-raises that exhaust the tree's
   3-raise cap produced a **24x-pot shove in live play** — 3,980 chips into a 166
   pot. Structural context: **77–79% of the 200bb champion's decision nodes offer
   no raise below all-in**, carrying ~30% all-in mass, though the abstract jam is
   ≤3x pot at 14,492 of those 15,188 nodes — so the trained mass is reasonable and
   only the translation is wrong. Measured rate (`tools/overbet_audit.py`, 200bb):
   8 overbets in 640 decisions vs a min-raiser (worst 15.4x), **zero** vs a
   calling station or in self-play.

   **RETRACTED 2026-07-31: the leak is not preflop, it is the river.** The claim
   that "four of the eight were preflop, which no postflop resolving can reach"
   came from `tools/overbet_audit.py`, whose criterion is `amount / pot_before >=
   3.0` — an ABSOLUTE pot multiple, the same assumption that made the guard fire on
   89.6% false positives. Preflop the matched pot is one or two big blinds, so a
   correct 200bb shove is inherently 100-200x pot and trips any absolute bound.

   `tools/overbet_distortion_audit.py` classifies by DISTORTION instead —
   `ratio_real / ratio_abstract`, i.e. how far the real geometry has drifted from
   the node the action was trained on — over 200 hands at 200bb versus the same
   min-raiser, guard off, behaviour byte-identical to serving:

   | street | translated all-ins | flagged by the ≥3x rule | genuinely distorted |
   |---|---:|---:|---:|
   | preflop | 4 | **4** | **0** |
   | turn | 1 | 1 | 1 |
   | river | 6 | 5 | **6** |
   | total | 11 | 10 | **7** |

   Every preflop jam is undistorted: real ratio equals abstract ratio, so these are
   legitimate shoves. All six river all-ins ARE distorted, worst 6.25x (the abstract
   node expected roughly a 1x-pot bet). So **`PLAN_V3` P3's 1-2 week preflop budget
   is aimed at a leak that does not exist**, and the real one sits on streets that
   postflop resolving *can* reach.

   That creates a tension worth stating plainly: the river distortion is exactly
   what the exact-card resolver removes, because its tree is built on the real
   geometry — and resolving is now switched OFF at 200bb because it measured −58.61
   bb/100 there. Both results stand: the resolver fixes translation and still loses
   overall, so it must be introducing a larger error elsewhere. The guard also
   addresses this distortion and is off for its own measured reason (§3.6 below).
   With both mitigations rejected on measurement, the honest conclusion is that a
   real river-translation leak is currently **unmitigated**, and the remaining fix
   is structural: `_locate` matches a live hand onto an abstract node by translated
   action sequence and **never compares pot/stack geometry**. Making it
   geometry-aware attacks the cause rather than the symptom, and it is the work P3's
   budget should be pointed at. The real fix is the
   resolver (0.16% on all-in at that spot, since its tree uses the real geometry)
   and, for preflop, a richer menu or preflop resolving — not a sizing heuristic.

   **The guard's −268.82 bb/100 was a DEFECT, not a GTO-versus-exploitation
   tradeoff (corrected 2026-07-30).** The earlier reading was attributed to "a
   station calls any jam, so shoving into it is correct exploitation." That
   explanation was wrong. The trigger was
   `allowed = min(ratio_abstract * tolerance, cap)`, so whenever the abstract jam
   was itself larger than the 6.0x cap, `allowed` collapsed to the cap and the
   geometry test became **unreachable** — the guard fired on jams that had
   translated perfectly. Measured over 150 LBR pairs at 200bb with the guard on:

   | firing cause | count | share |
   |---|---:|---:|
   | genuine mismatch (the intended trigger) | 67 | 10.4% |
   | **cap-only — geometry fine, only the absolute bound objected** | **575** | **89.6%** |

   483 of the 575 had `real == abstract` **exactly**. At 200bb the matched preflop
   pot is one big blind, so *every* preflop shove reads as a ~200x overbet to an
   absolute pot-multiple bound, and the guard trimmed it to 6x pot — turning an
   all-in into a small raise. That is a strategy change, not a translation fix.

   Fixed by making the cap bound the **correction** and never the **trigger**
   (`backend/agents/gpu_blueprint_agent.py`). Firings fell 642 → 67 with every
   intended trigger preserved and **zero** false positives. Two tests had encoded
   the defect as intended behaviour (`test_cap_binds_when_the_abstract_jam_is_itself_huge`
   asserted that a *matched* 19.5x jam be trimmed; the end-to-end test asserted no
   jam may exceed the cap on **any** street) and were replaced with
   matched-geometry, preflop, and genuine-distortion regressions —
   `tests/test_all_in_translation.py`, 10 tests, all passing.

   **The deciding measurement was taken (2026-07-30): the fixed guard is
   near-neutral on exploitability.** `tools/lbr_guard_gate.py`, paired and
   chunk-checkpointed, 20,000 duplicate pairs per arm at 200bb, blueprint only:

   | arm | LBR bb/100 | 95% CI | fallback |
   |---|---:|---|---:|
   | guard OFF | +252.45 | [+223.34, +281.55] | 1.36% |
   | guard ON | +247.70 | [+218.13, +277.27] | 1.44% |
   | **paired delta (ON − OFF)** | **−4.74** | **[−10.63, +1.14]** | — |

   Negative means less exploitable. The interval still spans zero, so the verdict
   is formally INCONCLUSIVE — but it now *bounds* the effect: the guard costs at
   most **+1.14 bb/100** and may gain up to **−10.63**. That is a different
   statement from the −268.82 that originally condemned it, which was the defect
   above rather than the mechanism.

   Only **69 of 20,000 pairs (0.345%)** differed at all. The guard fires often
   (~67 genuine mismatches per 150 pairs) but rarely changes the *outcome*,
   because a best responder answers a 5.6x-pot bet and an 11x-pot bet the same
   way. An earlier 400-pair attempt read −26.50 [−78.44, +25.44] with 399/400
   pairs identical — effective sample size 1, which is why it resolved nothing.

   **The head-to-head was re-run with the fixed guard, and the cost did NOT go
   away** (`tools/guard_headtohead_gate.py`, 3,000 pairs, blueprint-only, CRN):

   | depth | opponent | before the fix | after the fix |
   |---|---|---:|---:|
   | 200bb | always-min-raise | −268.82 | **−244.14** |
   | 100bb | always-min-raise | −124.00 | **−139.46** |
   | either | always-call | ~0 | **+0.00** (never fires) |

   So the earlier reading that the −268.82 was "the defect, not the mechanism" is
   **wrong** and is retracted. The defect inflated the firing count 10x but not
   the cost: the fix removed the *cap-only* firings, which were overwhelmingly
   preflop shoves the opponent folds to anyway, while the expensive firings are
   the **genuine mismatches** the guard is designed to correct. Trimming a truly
   distorted jam is correct by the guard's own logic, and against an opponent who
   calls too wide, jamming was the profitable error.

   **Ship/no-ship: the guard stays OFF, and §3.6's original decision was right.**

   | | magnitude |
   |---|---:|
   | exploitability gained (LBR paired, 100bb, significant) | **−5.65 bb/100** |
   | head-to-head cost against a min-raiser | **−139 to −244 bb/100** |

   A 25–43x unfavourable ratio. The two numbers are not on one utility scale — LBR
   measures what a best responder extracts, the duel measures value against one
   artificial script — but the asymmetry is far too large for the trade to be
   worth taking without an opponent model to switch between them. That switch is
   what this agent lacks, and what the AlphaExploitem line addresses
   (docs/PLAN_V3_LITERATURE_ALIGNED.md).

   The fix is still worth keeping even with the guard off: without it the guard
   mangles every preflop all-in, so anyone who ever enables the flag now gets the
   intended behaviour rather than a 90%-false-positive one.

### 2026-07-31 the river-net acceptance gate fails its own null test

`tools/river_net_gate.py` had never been null-tested, which the standing rule in
§4 requires before any of its numbers are believed. Gating an **all-zero net** —
weights zeroed, so it predicts 0 CFVs everywhere and is definitionally the
zero-predictor — against the same 12 situations x 160 iterations as the recorded
run:

| net | action agreement | policy L1 |
|---|---:|---:|
| **all-zero NULL net** | **0.9269** | 0.4657 |
| trained net (ratio 0.4864 vs baseline) | **0.3766** | 1.1474 |
| gate requirement | ≥ 0.90 | ≤ 0.30 |

**The null net PASSES the agreement criterion and the trained net fails it**, and
the null is better on *both* criteria. Two readings, and they are not exclusive:

1. **The 0.90 threshold is uninformative.** It sits *below* what predicting
   nothing scores (0.9269), so agreement cannot separate a useful net from an
   empty one. STATUS.md §7 already flagged that "the 0.1 acceptance threshold was
   assumed and never measured" — it is now measured, and it is wrong. Any future
   criterion has to be stated relative to the null floor, not as an absolute.
2. **A partially-wrong horizon may be genuinely worse than none.** Pricing the
   river at zero is neutral and leaves the solve's relative action values roughly
   intact, whereas a half-learned horizon actively misprices and flips top
   actions. On that reading the gate is working and its verdict is stronger than
   recorded: this net is not merely unhelpful, it is harmful.

Either way the roadmap consequence is the same. **The CFV line is blocked on its
acceptance instrument, not only on datagen throughput.** Do not spend GPU-days
generating rows for a net whose gate cannot tell learning from nothing. Rebuild the
criterion first, null-anchored, and re-derive the threshold from measurement.

Also settled while investigating, both cheaply and both retiring hypotheses from
`docs/PLAN_V3_LITERATURE_ALIGNED.md` §P5.1:

- **The EV-versus-CFV target variant is a mathematical no-op here.** Measured
  range mass per player is **exactly 1.0000** (min 0.9998, max 1.0002, std
  0.0000), so the counterfactual value already *is* the expected value.
- **Rebucketing to Supremus's 1,000 is a no-op too.** Exactly **1,081** combos are
  live on every river board, so 1,326-raw, 1,081-live and 1,000-bucket are the
  same width. The genuine contrast is the v0 **169**-bucket run (a 6.4x reduction,
  `backend/data/cfv/bucket_net.pt`, input 391 = 52 board + 2x169 + 1 pot,
  val MAE 9.324 bb), not 1,000.

The "ratio 0.301 with strength-ordered inputs" result quoted in §7 and in
`backend/agents/serving.py` has **no code in the repo** — it is unreproducible as
recorded, and should not be relied on until someone reconstructs it.

### 2026-07-31 THE 200bb CHAMPION IS RUNNING A STALE CARD ABSTRACTION

Found while a menu experiment failed to start: `--sampler-init` from the 200bb
champion was rejected with

    sampler bucket counts do not match: (169, 20, 20, 20) != (169, 150, 150, 30)

Reading all three champions' stored samplers:

| depth | abstraction | postflop buckets | LBR exploitability |
|---|---|---|---:|
| 20bb | histogram-EMD | (169, **150, 150, 30**) | **+13.34** |
| 100bb | histogram-EMD | (169, **150, 150, 30**) | **+118.64** |
| **200bb** | **scalar (legacy)** | (169, **20, 20, 20**) | **+252.45** |

**The 200bb champion is 7.5x coarser on flop and turn than either other depth**, and
1.5x coarser on the river. §2.3 of this document already records histogram-EMD as
"the default abstraction now", crediting it with reaching parity against a
3.3x-trained scalar model and fixing a whole leak class — but `scalar@118k` predates
that switch and was never retrained. The deepest stack, where the agent is by far
the weakest, is the one depth still on the superseded abstraction.

This is a better candidate explanation for the deep-stack weakness than the betting
menu, and it plausibly explains something else that had been left unexplained: why
exact-card resolving measured **helpful at 100bb (+31.83) and harmful at 200bb
(−58.61)**. The resolver projects blueprint ranges into its exact solve, so at 100bb
it starts from 150-bucket ranges and at 200bb from 20-bucket ones. A resolver
seeded with a much coarser prior is a resolver solving the wrong subgame.

A 200bb histogram blueprint was trained with **the deployed menu left unchanged**
(0.5/1.0, cap 3, 147,349 nodes) and the 100bb champion's fitted sampler imported, so
the card abstraction is the only variable. First gate at 5,000 iterations
(`tools/abstraction_duel.py`, 3,000 seat-swapped duplicate pairs, CRN, null exactly
**+0.00**):

| | value |
|---|---|
| histogram@5k minus scalar@118k | **−17.79 bb/100 [−50.04, +14.47]** |
| verdict | **INCONCLUSIVE — not promoted** |

The number that matters here is not the point estimate but the pairing: **5,000
iterations against 118,000**, a **23.6x training deficit**, and the interval still
spans zero. §2.3 already records histogram "reaching parity with a 3.3x-trained
scalar model"; this is the same effect at seven times the deficit, which is
consistent with the abstraction being materially better rather than worthless.

So the read is undertraining, not a dead end, and training was continued from 5,000
toward 20,000 iterations (~10.4h at the measured 0.40 iters/s). Gate again at 10k
and 20k. Promotion still needs a positive disjoint confirmatory interval, LBR at
20,000 pairs against the frozen +252.45 [+223.34, +281.55], and the mapping and
fallback checks — a duel win alone is necessary, not sufficient.

Measured training cost at this tree size, for planning: **0.385–0.405 iterations/s**,
flat across fourteen 500-iteration chunks, i.e. ~21 min per 500. Progress is
visible only in `telemetry.json` and the checkpoint files — the trainer's stdout
prints nothing after startup, so a healthy run and a hung one look identical from the
log alone.

Note for anyone repeating this: two OOMs preceded the discovery and both are
informative. A GPU OOM in `_showdown_values` needed
`HOLDEM_SHOWDOWN_WORKSPACE_MB=128` (the tiling is mathematically identical at any
tile size, so this costs speed and not quality), and `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` is **not supported on Windows** — torch warns and ignores
it. A host-RAM OOM then came from refitting histogram centroids from scratch;
`--sampler-init` avoids it, and is also the better experiment because it holds the
card abstraction fixed across runs.

### 2026-07-31 river-net gate rebuilt, and the failure is BIMODAL not uniform

`tools/river_net_gate_v2.py` replaces the retired v1 criterion. It computes the
null arm instead of assuming it, and scores on the horizon-SENSITIVE subset — the
combos whose top action actually differs between a full solve and a zero-priced
horizon — where the floor is 0 by construction rather than ~0.93.

Validation that only the metric changed: v2 reproduces v1's figures on the same
population almost exactly (agreement net **0.3759** vs the recorded 0.3766, null
**0.9268** vs 0.9269).

12 situations x 160 iterations, current checkpoint:

| metric | value |
|---|---|
| mean horizon-sensitive mass | **7.3%** |
| sensitive-set accuracy (primary) | **0.3029** |
| policy L1, net vs null | **1.1482** vs **0.4657** |
| L1 skill vs null | **−1.4658** |

That 7.3% is the whole explanation for v1: **93% of range mass has a top action no
horizon can change**, so v1's statistic was measuring insensitive decisions and its
floor was pinned near 0.93.

**Both gates fed the net a range distribution it was never trained on.** v1's
`_situation` sets both ranges to `live / live.sum()` — perfectly uniform — and v2
inherited it to keep the metric change isolated. But `random_range`, which the
datagen uses, exists precisely to avoid that: its docstring says uniform weights are
"far too flat to look like anything re-solving actually meets." Measured on one
river board with 1,081 live combos:

| ranges | effective support | max weight | top-1% mass |
|---|---:|---:|---:|
| uniform (what both gates fed it) | **1081.0** | 0.0009 | 0.009 |
| `random_range` (what it was trained on) | **31–102** | 0.07–0.32 | 0.39–0.60 |

10–35x outside its training support on every concentration statistic. `--ranges
polarized` was added to sample the way training and real play do. Re-running the
whole gate on that population, 12 situations x 160 iterations:

| metric | uniform | **polarized (in-distribution)** |
|---|---:|---:|
| mean horizon-sensitive mass | 0.0732 | 0.0695 |
| sensitive-set accuracy | 0.3029 | **0.1313** |
| policy L1, net | 1.1482 | 0.5038 |
| policy L1, null | 0.4657 | 0.2258 |
| L1 skill vs null | −1.4658 | **−1.2306** |
| situations where the net works (acc ≥ 0.5) | 5 / 12 | **1 / 12** |
| situations where the net beats the null on L1 | 2 / 12 | **0 / 12** |

**The distribution mismatch was not the explanation.** On its own training
distribution the net is worse by every measure that matters — accuracy on the
decisions the horizon moves falls to 0.13, and it never once beats a zero horizon.
Only the absolute L1 magnitudes shrink, for both arms, because polarized ranges are
lower-entropy so all policies sit closer together.

**Two claims made earlier on 2026-07-31 are retracted:**

1. *"The failure is bimodal — the architecture can demonstrably learn this, 0.70
   accuracy where it works is not noise."* That 5/12 success rate was measured on
   the out-of-distribution uniform population, where occasional agreement is
   extrapolation luck. In-distribution it is 1/12. There is no demonstrated
   sub-population where this net works.
2. *"Polarized ranges make it 7x less bad (L1 skill −0.20)."* That came from a
   **3-situation** smoke and did not survive n=12, where the figure is −1.23. Over-
   reading an underpowered run is the exact failure this document warns about
   throughout, and it happened here.

Also checked and dead: `(stack, pot)` coverage does not explain anything. The gate
drew pots {14, 22, 34} and both the working and collapsing groups span all of them
(means 21.2 vs 22.0), so the datagen grid's spacing is not implicated.

**Net conclusion for the CFV line.** The river net is worse than pricing the river at
zero, on every population tested, by a wide margin. That is now an instrument-clean
result rather than an artifact: the gate is null-anchored, sensitivity-restricted,
and run in-distribution. Whether more rows would fix it is untested and should not be
assumed in either direction — but nothing measured so far supports spending 482
GPU-days to find out.

**No replacement threshold is proposed, deliberately.** v1's 0.90 was assumed and
never measured (§7 said so); inventing a new constant would repeat that with fresh
confidence. v2 reports position on two null-anchored scales, and the bar should be
derived from an accuracy-versus-strength curve once one exists. What v2 does settle
is that the CFV branch is now **decidable** — the gate can distinguish a learning
net from an empty one, which it could not this morning.

### 2026-07-31 DCFR+ averaging: implemented, measured, NOT adopted

Supremus (arXiv 2007.10442) weights iteration t in the average policy by
max{0, t - d} rather than t**gamma. Implemented in `VectorCFR._discount` behind
`dcfr_plus_delay` (opt-in; default None keeps the quadratic averaging every
champion was trained under) and threaded through `build_solver`, `train` and
`--dcfr-plus-delay`.

Measured by `tools/dcfr_plus_duel.py`: two 20bb blueprints trained from scratch at
5,000 iterations, identical solver seed, abstraction and tree, differing ONLY in
the averaging rule, then duelled at 3,000 seat-swapped duplicate pairs with CRN.

| | result |
|---|---|
| CRN null, control vs itself | **+0.00 [+0.00, +0.00]** |
| DCFR+ minus DCFR | **+2.80 bb/100 [−2.56, +8.16]** |
| verdict | **INCONCLUSIVE — do not change the default** |

The interval is tight, so this is a real null rather than an underpowered test: the
effect at this depth is bounded small in both directions. The flag stays in the
codebase, off, because Supremus's related finding — that simultaneous regret
updates beat alternating ones — holds specifically *when a value network is in the
loop*, a regime this project cannot enter until the CFV line is unblocked. Retest
there, not here.

**Why exploitability could not be used, and why the duel had to be:**
`abstract_exploitability_mbb` reads exactly 0.00 mbb for BOTH rules at 1,000 and
4,000 iterations on the fixed-river control game, raises IndexError below ~500
iterations on an under-trained strategy, and raises IndexError outright on a larger
tree. It is a converged-strategy instrument with no dynamic range for a
convergence-speed claim.

**Two confounds this measurement had to remove, both of which faked a large win:**

1. **Unequal averaging warmup.** `build_solver` hardcodes `averaging_delay=1000`,
   so pairing it against Supremus's d=100 changes how many early iterations are
   discarded AND the weighting rule. At 300 iterations the control accumulates
   *nothing* (300 < 1000) while DCFR+ averages 200 iterations — that alone read
   **+142.12 bb/100**, roughly 50x the real effect. The gate now defaults d to 1000
   to match, and refuses to run unless iterations >= 3d.
2. **Unequal training.** A killed run left the control at 5,000 iterations and the
   challenger at 4,500, and the resume check tested checkpoint *existence* rather
   than iteration count. 500 fewer iterations of training is indistinguishable from
   a worse averaging rule. Resume is now iteration-aware (training only the
   shortfall, since the trainer's `--iterations` is an increment), and a pre-duel
   assertion refuses to produce any number unless both arms sit at exactly the
   requested count.

Caveat on scope: this tests linear-versus-quadratic weighting at a *matched*
1,000-iteration warmup, which isolates the rule but is not literally Supremus's
d=100 configuration. And it is one seed at 20bb — the depth where the agent is
already strongest (+13.34 exploitability) with trees 4x smaller than the 200bb case
that actually needs help (+252.45).

## 4. The great eval corrections (why numbers before 2026-07-24 are suspect)

Five instrument bugs, all user-instinct-triggered ("too consistent", "are you
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
4. **The decision log could not see the resolver** (2026-07-29) —
   `log_agent_decision` consulted only `subgame_search` / `exact_river_search`, so
   every exact-resolver decision was logged as `blueprint-only` and its `actions`
   field always showed the BLUEPRINT's mix rather than the acting one. "Why did it
   do X?" was unanswerable for the component that was actually playing, which is a
   violation of the standing decision-level-verification rule below. Fix:
   `decided_by`, `resolver.acting_mix`, `blueprint_actions`, `all_in_rescaled`.
5. **Common random numbers were silently OFF for the serving agent** (2026-07-29)
   — `head_to_head`'s CRN coupling reseeds `agent._rng` and guards with
   `hasattr(target, "_rng")`. `MultiStackBlueprintAgent`, the agent every real
   comparison uses, had no `_rng`, so the guard skipped it and the arms desynced.
   A null duel of two identical routers read **+33.15 bb/100 [−48.71, +115.01]**;
   coupled it reads **+0.00 [0.00, 0.00]**. The documented "+0.00 null" had only
   ever been measured on a single-depth `GpuBlueprintAgent`. Any router duel
   labelled "CRN on" before this date was uncoupled — including the resolver
   on/off duels (+17.82 / +54.45 / +28.12, all spanning zero), which is one more
   reason those intervals were uninformative.

**Standing rules:** every new harness gets a NULL test before its numbers are
believed — and the null must be run through the agent that will actually be
measured, since bug 5 was a null that passed on a different class; runtimes must be consistent with claimed work (a search eval that
finishes at blueprint speed didn't search); decision-level verification (logs
prove the choice came from the claimed component). Also: `CUDA_VISIBLE_DEVICES=""`
does NOT hide GPUs on this Windows/torch — never rely on it.

## 5. Infrastructure that exists and is trusted

- **Duel gate**: `python -m backend.eval.duel --data-dir D --stack-bb N --pairs 3000 [--promote]`
  — NULL-tested head-to-head with auto-promotion; the only promotion signal. CRN
  coupling now reaches the serving router (§4 bug 5), so paired A/Bs of two
  stochastic agents read exactly 0.00 on a null instead of ±80 bb/100 of noise.
- **Per-decision logger**: every served move in `backend/data/server-debug.jsonl`.
  Now records `decided_by` (which engine chose), `resolver.acting_mix` (the acting
  distribution, not the blueprint's), `blueprint_actions`, and `all_in_rescaled`.
  Before 2026-07-29 it could not see the resolver at all — see §4.
- **Overbet audit**: `python tools/overbet_audit.py --hands N --stack-bb D
  --resolver on|off --opponent always-call|always-min-raise|self` — counts jams
  larger than a legitimate menu size, but only where a smaller raise was legal, so
  a forced short-stack shove never counts. JSONL trace per decision plus a summary
  JSON. This is what root-caused the 24x-pot shove (§3.6).
- **LBR exploitability probe**: `backend/eval/lbr.py`, multi-size probes, validated
  against an analytic anchor (LBR vs always-fold = +75.0000 bb/100, zero variance).
- **Slumbot harness**: `backend/eval/slumbot.py` — real external opponent, uses the
  AGENT's own action mapping rather than a reimplementation, anchored by
  always-fold-from-the-button reading exactly −50.0000 bb/100 with zero variance.
- **GTO Wizard AI harness** (NEW 2026-07-30): `backend/eval/gtowizard.py` +
  `tools/gtowizard_benchmark.py`. The first external opponent here that both
  **probes** (it resolves in real time, unlike Slumbot) and returns
  **AIVAT-adjusted** results, which the paper reports reaches equal significance
  with ten times fewer hands. 200bb, arbitrary bet sizes accepted, 100k hands per
  month. Mirrors each hand into a real `HeadsUpHoldem` and lets the agent's own
  `select()`/`execute()` choose, exactly as the Slumbot harness does.
  Null-validated against the published anchors: always-fold measured
  **−69.74 bb/100 [−80.82, −58.65]** over 493 hands versus the published
  **−64.6 ± 3.3** (overlaps), with raw **−64.30** and **zero board desyncs**, and
  the harness's own arithmetic agreed with the server's AIVAT tally to
  **0.01 bb/100**. Protocol facts that silently corrupt a run if guessed:
  `blinds` is **[big, small]**, the **SB is the button**, `action_history` is a
  list using `"_"` as an end-of-round marker, stacks reset every hand, and
  Cloudflare rejects the default urllib User-Agent with HTTP 403 code 1010.
  Hero identification must use the registered bot name, not visible hole cards —
  the villain's cards are revealed once a hand is decided, which dropped 5.6% of
  hands before it was fixed.
- **Monitors**: training runs get 10k gates + stop-on-plateau + VRAM/RAM guards.
- **Exact-card resolvers**: flop / turn / river, identity 1,326 buckets, one
  bucket per private combo — `backend/search/exact_{flop,turn,river}.py` plus
  `continual.py` for session-level range advancement.
- **Depth-limited solving**: HORIZON trees + evaluators (proven plumbing) —
  ready for any future value-function work.
- **Safe re-solve gadget (v1/v2) + AIVAT chance-variates**: built and tested.
- **River CFV pipeline** (single pipeline as of 2026-07-28; two dead ones deleted):
  `backend/cfv/river_dataset.py` + `river_net.py`, 76,411 rows generated,
  resumable. Net trained and **failed** its gate — see docs/PLAN_V2 §9.

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

- **Done:** solver; serving blueprints; histogram abstraction; the honest eval
  stack (Slumbot + LBR + AIVAT + duel, each with a null or analytic anchor);
  house-rules capability; the search post-mortem; Phase 4 projection repair
  (130/130 resolves, 0 fallbacks at three depths); exact-card flop/turn/river
  resolvers wired into serving; throughput diagnosis (latency-bound, fusion
  +1.26x, situation batching +1.68x, multi-process 0.33x — rejected).
- **Serving now:** depth-routed blueprints with exact-card resolving ON for
  flop/turn/river at capped own-bet menus (0.33/0.75/1.4, cap 2). See
  docs/SERVING.md; verify with `GET /api/health` before believing any claim.
- **Recently closed:** native 20bb blueprint. The final configuration reached
  50,000 iterations; histogram@5k won the bootstrap confirmation by +32.08
  [+20.16,+44.01] and was promoted. Every later milestone was a tie or
  regression (50k confirmation +0.32 [−10.28,+10.91]), so identical training
  is stopped. See `docs/20BB_BLUEPRINT_PLAN.md`.
- **Known open leaks, in priority order:**
  1. **Preflop translation** — half the observed overbets were preflop, and
     nothing covers it before P5 (§3.6).
  2. **River value net still OFF** (failed gate: 38% action agreement, policy L1
     1.15 vs ~0.21 solver noise). Representation work reached ratio 0.301 with
     strength-ordered inputs but is not integrated into `RiverCfvNet`.
- **Next measurements, in order:** LBR guard-on vs guard-off (settles §3.6);
  the four-depth smoke of the flop resolve path; the agreement-vs-ratio curve for
  the river net, since the 0.1 acceptance threshold was assumed and never measured.
- **Data-scaling verdict:** ~14% error reduction per 3.2x rows means usable river
  net accuracy needs ~12 **billion** rows. Representation is the limit, not data —
  do not simply generate more.
- **Parked with revival criteria:** CFV net (better targets/representation),
  bucketed turn search (retired), AIVAT decision-variates, Slumbot rematch (worth
  doing once any new model or the full resolving stack clears its gates).
