# Plan V3 — Literature-Aligned Route to the Strongest Player

**Date:** 2026-07-30
**Status:** proposed; supersedes PLAN_V2 §0 *ordering* only. PLAN_V2's diagnosis
stays valid, and `STATUS.md` remains the living experiment record.
**Hardware:** local RTX 3060 12GB (display-attached) + rented 1x RTX 4070 Ti Super
16GB (headless, `tools/cloud_setup.sh`)

---

## 0. The one-paragraph answer

A literature review (2020–2026) says PLAN_V2 picked the right *architecture* and
the wrong *order*. The CFV-network line it puts on the critical path is blocked by
a hard arithmetic wall — Supremus needed **50 million river targets**, this
pipeline generates **1.2 rows/s**, which is **482 GPU-days** — and the
multi-row-per-solve trick that was supposed to close that gap has already been
measured here as producing 12–171% biased targets. So the CFV net cannot be next.
What *is* next, in order: **(1) get an honest absolute number** from the new GTO
Wizard Benchmark API, which fixes PLAN_V2's own "single biggest problem" against a
*probing* opponent that Slumbot structurally cannot be; **(2) two cheap solver
upgrades from the literature** (DCFR+ averaging, simultaneous updates) that raise
quality per iteration at every depth for a few days of work; **(3) fix preflop**,
where half the observed overbet leaks live and nothing is planned; **(4) Parallel
CFR** for 3.3x more iterations inside the same latency budget. The CFV net is
demoted to a **bounded three-day representation experiment on existing data** with
a hard kill gate, because two of its three candidate fixes are free to test and
the answer decides whether months of datagen are justified at all.

---

## 1. What the literature says that this project did not already know

| finding | source | what it means here |
|---|---|---|
| **GTO Wizard Benchmark exists**: public API, 200bb, 50/100 blinds, **arbitrary bet sizes**, AIVAT built in, 100k hands/month | [arXiv 2603.23660](https://arxiv.org/abs/2603.23660) | Fixes the #1 documented problem. Stronger *and* probing, unlike Slumbot |
| GTO Wizard AI **beat Slumbot by 19.4 ± 4.1 bb/100** over 150k hands | same | A genuinely harder opponent than the current external check |
| AIVAT gives **equal significance with 10x fewer hands** | same | The 11.6-hour / 20,000-hand Slumbot run becomes ~2,000 hands |
| Published null anchors: Always-Fold **−64.6 ± 3.3**, Check-Call **−241.1**, All-In **−380.6** bb/100 | same | Ready-made harness validation, satisfying the standing NULL-test rule |
| **Leaderboard is empty** (0 entries, updated hourly) | [gtowizard.com/benchmark](https://gtowizard.com/benchmark) | First real agent posted is the first entry |
| **Parallel CFR**: 3.3–3.4x postflop speedup, **47–54 ms/iteration** on a >1B-history depth-limited tree, single desktop device | [arXiv 2605.19928](https://arxiv.org/abs/2605.19928) | This resolver runs ~119 ms/iteration (120 iters / 14.23 s). ~2.2x headroom |
| Parallel CFR's two orthogonal axes: **by information set** and **by tree node**, 7-stage pipeline, GPU-batched leaf eval | same | The infoset/node axes need **no** value net. Only leaf-NN batching does |
| **Supremus outputs expected values**, as a fraction of pot, over **1,000 buckets** | [arXiv 2007.10442](https://arxiv.org/abs/2007.10442) | This project's net outputs pot-normalised **CFVs** over **1,326 raw combos** |
| A whole paper argues the EV-vs-CFV target choice is the deciding one | [AAAI 25661](https://ojs.aaai.org/index.php/AAAI/article/view/25661/25433) | Cheap to test; not yet read (PDF would not parse) |
| Supremus net: 7 hidden layers x 500 — **already matched here** | [arXiv 2007.10442](https://arxiv.org/abs/2007.10442) | Architecture is not the gap. Representation and scale are |
| Supremus data: river **50M**, turn 20M, flop 5M, aux 10M; **4,000 DCFR+ iters/player** per target | same | River here: **76,411 rows**. 654x fewer |
| Supremus river loss 0.010 train / 0.015 val; turn 0.008/0.010; flop 0.0092/0.011 | same | Target accuracy bar for any rebuilt net |
| **DCFR+**: average-policy weight `max{0, t−d}`, **d=100** — linear, not quadratic | same | Cheap solver change, applies to blueprint *and* resolver |
| With value nets, **simultaneous** regret updates beat alternating — reversing the classical result | same | Only matters once a net is in the loop; note for later |
| Supremus **beats** LBR by **951 ± 96** mbb/g; DeepStack reimpl **+536 ± 68** | same | This agent is at **−2,910** (LBR wins). The gap, quantified |
| AlphaExploitem: opponent-adaptive exploitation, but **Leduc/Kuhn only**, PPO-based, and explicitly does **not** address bet-size translation | [arXiv 2605.09150](https://arxiv.org/abs/2605.09150) | Park. PPO was already retired here, and it does not solve the guard dilemma |

### Where the player actually sits

| agent | LBR result (mbb/g) | reading |
|---|---:|---|
| Supremus | **+951 ± 96** | beats a best responder |
| DeepStack (reimpl) | **+536 ± 68** | beats a best responder |
| **this project** | **−2,910** | LBR wins by ~2.9 bb/hand |

The distance to the resolving-agent class is roughly **3,860 mbb/g**. That is the
number the plan has to move, and only architecture moves it — not more blueprint
iterations, which `STATUS.md` §2.2 already shows plateau.

---

## 2. The wall that reorders everything

PLAN_V2 puts the CFV-net hierarchy on the critical path and counts on TurboReBeL's
~250x "single-solve/multi-iteration" speedup to make datagen affordable. That plan
is already falsified by this repo's own measurements, recorded in
`tools/generate_river_cfv.py`:

> `--emit 0` — MEASURED DEFAULT. Multi-row-per-solve emission was tried twice and
> both variants produce biased targets: randomly blended ranges priced against the
> original solve are wrong by 50–171%, and interior-node harvesting (the correct
> form of the TurboReBeL idea) is still 12–17% off because a node's value under
> the solved average is the value of FOLLOWING that strategy, not the equilibrium
> value for the ranges arising there.

That reasoning is correct, and depth/reach-mass filtering did not rescue it
(16.2% near the root vs 17.0% deeper). So targets cost **one solve each**.

**The arithmetic:**

| | rows | rate | single-GPU time |
|---|---:|---:|---:|
| current holdings | 76,411 | — | ~17.7 h spent |
| Supremus river parity | 50,000,000 | 1.2 rows/s | **482 days** |
| even a 10x-reduced target | 5,000,000 | 1.2 rows/s | **48 days** |

Renting does not fix this either: `tools/cloud_datagen.sh` is written for a
**6-GPU** box because the workload is latency-bound and *GPU count is everything*
— and the rented instance is **1 GPU**. Two workers on one device measured a **3x
regression**. So the rented 4070 Ti Super adds roughly one worker's throughput,
not a multiple.

**Conclusion:** the only affordable path to a working net is to need *far fewer
rows*, which is a **representation** question — and representation is testable on
the 76,411 rows already on disk, for free. That test therefore comes before any
new datagen, and it is the plan's main anti-time-waste gate.

---

## 3. The plan

Ordered by measured-value-per-day, not by architectural ambition. Every phase has
an explicit **kill criterion** — the thing that makes it stop instead of grind.

### P0 — Honest absolute number (GTO Wizard Benchmark) · ~4 days · **do first**

PLAN_V2 §1.1 calls unmeasured strength "the single biggest problem," and it still
is: the only external number is Slumbot **−18.33 ± 22.1 bb/100**, taken against an
opponent the docs correctly note is *also* a non-probing abstraction agent. A
probing opponent changes the measurement class.

1. Request the API key (form, approval by email) — **needs your action**, §5.
2. Implement `backend/eval/gtowizard.py` reusing the *agent's own* action mapping,
   exactly as `slumbot.py` does — never a reimplementation.
3. **NULL-test before believing anything**: reproduce Always-Fold (−64.6 ± 3.3),
   Check-Call (−241.1), All-In (−380.6). Standing rule, and bug 5 in `STATUS.md`
   §4 proves the null must run through `MultiStackBlueprintAgent`, the actual
   serving router, not a single-depth agent.
4. Run 5,000 hands at 200bb, resolver ON, then 5,000 with resolver OFF.

| milestone | gate | kill criterion |
|---|---|---|
| P0.1 harness | all three nulls inside published CI | mismatch ⇒ fix harness, do not report a number |
| P0.2 baseline | 5,000 hands, AIVAT-adjusted, CI reported | — |
| P0.3 resolver A/B | resolver ON vs OFF, CRN-coupled | if ON is *worse* beyond CI, the served default is wrong — that is a finding, not a failure |

**Why this is first:** it costs days, it retires the project's largest known
blind spot, it finally settles the resolver-on/off question that three
inconclusive duels could not (all three predate the CRN fix), and the leaderboard
is empty.

### P1 — Two cheap open measurements · ~2 days · run alongside P0

Straight off `STATUS.md` §7's own queue, both already tooled:

- **LBR guard-on vs guard-off** (`tools/lbr_search_gate.py`) — settles §3.6. The
  guard costs 124–269 bb/100 against a min-raiser but may reduce exploitability;
  right now it is shipped off on a coin-flip rather than a measurement.
- **Four-depth flop-resolver smoke** — 20/50/100/200bb, confirming admission
  falls back to the blueprint rather than failing, per `20BB_BLUEPRINT_PLAN.md`.

Kill criterion: none needed; these are measurements, not bets.

### P2 — DCFR+ and averaging schedule · ~3 days · best quality-per-effort

Supremus's concrete solver deltas, none of which need a value net:

- average-policy weight **`max{0, t−d}`, d=100** instead of `t²`;
- keep alternating updates for now (simultaneous only wins *with* nets);
- the hyperparameter-schedule line ([arXiv 2404.09097](https://arxiv.org/html/2404.09097)) is worth reading before tuning `d` by hand.

| milestone | gate | kill criterion |
|---|---|---|
| P2.1 | `tests/test_gpu_convergence.py` still 0.0 mbb on the converged control | any regression ⇒ revert |
| P2.2 | Kuhn/Leduc exploitability ≤ current at equal iterations | worse ⇒ revert |
| P2.3 | 3,000-pair duel vs current champion at 100bb and 200bb, CRN on | CI does not clear zero at **both** depths ⇒ do not promote, keep the code behind a flag |

**Why it is cheap:** it is a weighting change in the accumulator plus a guard run.
It improves the blueprint *and* every resolve, at all four depths.

### P3 — Preflop · ~1–2 weeks · the only *known live leak* on the list

`STATUS.md` §3.6: **four of eight observed overbets were preflop**, which no
postflop resolving can ever reach, and nothing covers it before P5 of the old
plan. Two candidate fixes, cheapest first:

1. **Richer preflop menu / higher raise cap.** Raise-cap exhaustion is what makes
   `_locate` land on a node whose pot is far larger than reality. Preflop depth is
   cheap (`STATUS.md` §6: "preflop depth is cheap; postflop explodes"), so this
   may be a config change plus a retrain.
2. **Preflop resolving.** Structural, expensive, and only justified if (1) fails.

| milestone | gate | kill criterion |
|---|---|---|
| P3.1 | `tools/overbet_audit.py` vs min-raiser: preflop overbets → 0 | — |
| P3.2 | LBR at 100/200bb improves vs frozen baseline | no improvement ⇒ the overbet was not costing what we assumed; stop and record it |
| P3.3 | GTO Wizard re-run (P0 harness) shows no regression | regression ⇒ revert |

### P4 — Parallel CFR latency · ~1–2 weeks · quality at fixed latency

The paper reports **47–54 ms/iteration**; this resolver is at **~119 ms**. The
gain converts directly into solve quality because iterations-in-budget rises at
unchanged wall-clock — which is exactly the stated priority ordering (quality
first, latency still wanted). Adopt the two axes that need no value net:
parallelism **by information set** and **by tree node**.

| milestone | gate | kill criterion |
|---|---|---|
| P4.1 | fixed-seed eager-vs-optimised control: **max root-policy difference 0.0** | any nonzero difference ⇒ it is a behaviour change, not an optimisation |
| P4.2 | ms/iteration measured by `tools/benchmark_resolver_latency.py` | <1.5x ⇒ stop; the remaining speedup needs the leaf-NN path, i.e. P5 |
| P4.3 | iterations raised 120 → budget-filling, then duel + LBR | no strength gain ⇒ record that iterations past 120 do not pay, and stop |

Note the honest caveat: the paper's 3.3–3.4x is **versus its own single-threaded
baseline** on a DGX Spark, not a claim about a 3060. Treat it as an architectural
ceiling to chase, not a number to expect.

### P5 — CFV net, demoted to a 3-day experiment · **gate before any datagen**

Three candidate causes of the failed gate (0.3766 agreement vs 0.90 required;
policy L1 1.1474 vs 0.30). **Two are free to test on the existing 76,411 rows**,
and that is the whole point:

| variant | change | cost |
|---|---|---|
| A (control) | current: pot-normalised **CFVs** over **1,326 raw combos** | already measured, fails |
| B | **1,000 Supremus-style buckets** instead of 1,326 raw combos | retrain only |
| C | **expected values** instead of CFVs, as pot fraction | retrain only |
| D | B + C together | retrain only |

The prior for B is strong and comes from this project's own data: raw-combo I/O
scored *below the zero-predictor*, while 2×169 buckets scored **+58%** on
identical data. Supremus's **1,000** sits between the two failures, and 1,000
river buckets is ~6x finer than 169 while being ~1.3x coarser than raw combos.
Reducing input dimensionality is also the only lever that reduces the required
row count, which is what §2 says the line lives or dies on.

| milestone | gate | kill criterion |
|---|---|---|
| P5.1 | B/C/D each trained on the same 76,411 rows, same budget | **if no variant beats A by ≥2x on masked MAE-vs-zero-baseline, the CFV line stays parked — do not generate one more row** |
| P5.2 | best variant's **agreement-vs-rows curve** at 76k / 250k / 1M | curve must extrapolate to ≥0.90 agreement inside 5M rows |
| P5.3 | measure the **iterations-vs-target-accuracy frontier** (50 / 200 / 1,000 / 4,000) | if 50 iters holds ≤2% target error, datagen gets 4x cheaper for free |
| P5.4 | only now: datagen to the curve's implied count, cloud + local | 0.90 agreement / 0.30 L1 or it does not ship |

Also worth doing in P5.2: the docs note the **0.1 acceptance threshold was assumed
and never measured**. Measure the agreement-vs-strength curve rather than
inheriting a guessed bar.

**This retires the "12 billion rows" verdict as an artifact.** That extrapolation
(~14% error reduction per 3.2x rows) was measured on the raw-combo representation
— the one that cannot generalise. A scaling law fitted on a representation that
does not learn is a measurement of the representation, not of the data
requirement. Supremus reached usable accuracy at 50M with 1,000 buckets.

---

## 4. The eval ladder — what "better" is allowed to mean

Nothing in P0–P5 promotes on a self-play or scripted-bot number. In strength
order, weakest evidence first:

| instrument | what it proves | anchor | cost |
|---|---|---|---|
| `styles.py` field | nothing about equilibrium | — | minutes |
| duel gate (3,000 pairs, CRN) | relative, same-abstraction | self-duel = **0.00** | ~1 h |
| `tools/overbet_audit.py` | structural leak counting | jams only where a smaller raise was legal | minutes |
| **LBR** | exploitability lower bound | always-fold = **+75.0000**, zero variance | hours |
| **Slumbot** | external, **non-probing** | always-fold-from-button = **−50.0000** | 11.6 h / 20k hands |
| **GTO Wizard AI** *(new)* | external, **probing**, AIVAT | published Always-Fold **−64.6 ± 3.3** | ~2k hands for significance |

Promotion needs: a positive **disjoint confirmatory** interval, no LBR regression,
mapping/fallback gates green, and — new in V3 — **no GTO Wizard regression**.

The two rules that already caught five instrument bugs stay: every harness gets a
NULL test *through the agent that will actually be measured*, and runtimes must be
consistent with the claimed work.

---

## 5. Hardware allocation, cost, and what needs your decision

The rented box is best used as a **second independent machine** — the
"never overlap GPU work" rule is per-machine, so local and remote double the
measurement queue width without the multi-process pathology.

| workload | where | why |
|---|---|---|
| P0 GTO Wizard hands | **cloud** | network-bound, unattended, 99% waiting (as Slumbot was) |
| P1 LBR runs | **cloud** | CPU-heavy; 12 cores vs 8 |
| P2 solver change + guards | **local** | fast iteration, GPU-light |
| P4 latency benchmarking | **local** | must be measured on the card that serves |
| P5 datagen (only if P5.1 passes) | **both** | resumable by design |
| serving | **local** | see the preemption risk below |

Free quality win, no code change: on the **headless** 16GB card the ceilings that
exist only because the 3060 drives a monitor can be lifted via
`HOLDEM_RESOLVER_MAX_VRAM_MB` / `..._HEADROOM_MB` / `HOLDEM_SHOWDOWN_WORKSPACE_MB`.
`tools/cloud_setup.sh` prints the profile. It deliberately does **not** raise
`HOLDEM_FLOP_NODE_BUDGET`, because that is a latency guard too — raising it is a
measurement (P4.2), not a preference.

**Open items needing your input:**

1. **GTO Wizard API key** — the form requires your identity and acceptance of
   their terms, so I am not submitting it for you. 100k hands/month cap is ample
   (P0 needs ~10k).
2. **Is the rented instance interruptible?** `$0.009/hr` with "Save 40%" reads
   like spot pricing. Fine for datagen and eval (all checkpointed); **bad for
   serving**, where preemption lands mid-hand.
3. **Serving location.** Network RTT is a non-issue (decisions are 1.7–14.2 s of
   compute), but exposing the backend on a public port is outward-facing and needs
   an auth story before a port is opened.

---

## 6. What this plan deliberately does *not* do

- **No new blueprint training at any depth.** Plateau is measured at 20bb, 100bb
  and 200bb. More identical iterations are the clearest known waste of time.
- **No revival of bucketed subgame re-solving.** Retired on a −86 [−150,−22]
  measurement.
- **No opponent-adaptive/PPO work.** AlphaExploitem is Leduc/Kuhn-only and does
  not address translation; PPO is already retired here.
- **No datagen before P5.1.** That is the single most expensive way to be wrong.
- **No raising the flop node budget on latency grounds alone.**

## 7. Sources

- [GTO Wizard Benchmark (arXiv 2603.23660)](https://arxiv.org/abs/2603.23660) · [leaderboard/API](https://gtowizard.com/benchmark)
- [Real-Time Parallel CFR (arXiv 2605.19928)](https://arxiv.org/abs/2605.19928)
- [Unlocking the Potential of Deep Counterfactual Value Networks — Supremus (arXiv 2007.10442)](https://arxiv.org/abs/2007.10442)
- [Don't Predict Counterfactual Values, Predict Expected Values Instead (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/25661/25433) — **not yet read**
- [Deep (Predictive) Discounted CFR (arXiv 2511.08174)](https://arxiv.org/abs/2511.08174)
- [Faster Game Solving via Hyperparameter Schedules (arXiv 2404.09097)](https://arxiv.org/html/2404.09097)
- [AlphaExploitem (arXiv 2605.09150)](https://arxiv.org/abs/2605.09150)
- [awesome-poker-ai (resource index)](https://github.com/PokerBotAI/awesome-poker-ai)
- [DeepStack (arXiv 1701.01724)](https://arxiv.org/pdf/1701.01724) · [AlphaHoldem (AAAI)](https://cdn.aaai.org/ojs/20394/20394-13-24407-1-2-20220628.pdf)
