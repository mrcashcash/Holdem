# Research-Backed Improvement Roadmap

> **FINAL VERDICTS (updated 2026-07-27)** — after the eval corrections (see
> `docs/STATUS.md` §4-5), each item's honest outcome:
> 1. **Safe re-solving:** built (v1+v2, tests green) but MOOT for now — honest
>    duels show bucketed re-solving itself is a wash-to-regression vs the
>    blueprint (500 iters: −86 [−150,−22]). Root cause analysis points at
>    self-range inconsistency. The exact-card river successor completed a
>    3,000-pair confirmation at +7.62 [−21.73,+36.97], but missed its
>    reliability gate because 25/1,889 attempts fell back in the blueprint
>    projection adapter. Repair that adapter before another large A/B.
> 2. **Histogram-EMD abstraction: VALIDATED** — parity with 3.3x-trained scalar
>    + fixes the draw-fold leak class. Default abstraction for all new models.
> 3. **CFV net / depth-limited search:** pipeline built & mechanically proven;
>    v0 net does not beat blueprint flop play (−65 [−157,+27]) — target noise
>    bound. Parked with a concrete revival path (more eval runouts).
> 4. **RL-CFR sizing:** the static Phase 3 precursor was implemented, but the
>    combined v3 + Phase 3 model significantly regressed at 5k
>    (−94.92 [−166.36,−23.48]) and was retired.
> 5. **AIVAT:** chance-variates half built (~16% variance cut), wired into the
>    Slumbot harness; decision-variates pend a usable value function.
> Current active line: repair exact-river blueprint projection, then re-screen.
> The clean no-limp line plateaued by 40k and was not promoted.

**Date:** 2026-07-22 · **Goal:** strongest possible HUNL agent, browser-playable.
Ranked by (impact x fit to our measured gaps) / effort. See `docs/STATUS.md` for the
current state and the measurements that motivated each item.

---

## 1. Safe subgame re-solving (Reach-Maxmargin gadget) — TOP PRIORITY

**Our measured gap:** unsafe re-solving measured as a net regression (200bb TAG -195 →
-331 as solve budget grew). *2026-07-22 caveat:* review found two implementation bugs
(stale cross-hand subgame cache; bucket-formula mismatch) that contaminated that A/B —
both fixed; the A/B must be re-run to size the remaining, genuinely conceptual gap.
Regardless of its magnitude, unsafe re-solving is theoretically unsound against
off-blueprint opponents, and the literature is unambiguous that safe re-solving is
the correct architecture — so this item stays top priority; the re-run just decides
how much of the win comes from the bug fixes alone.

**The fix (Brown & Sandholm, NeurIPS 2017 — Libratus' method):** solve an *augmented*
subgame in which the opponent may either enter the subgame or take their blueprint
counterfactual value as an opt-out. Maximizing the minimum margin over opponent entry
hands guarantees the re-solved strategy is never worse than the blueprint against ANY
opponent. **Reach-Maxmargin** additionally credits path "gifts" (opponent mistakes en
route) for a larger safe improvement.

**Build:** gadget root layer over the existing `gpu_subgame.py` tree; opponent CFVs come
from the blueprint (computable with our verified value machinery). Re-run the search A/B
after — expect the 200bb regressions to flip positive. This also unblocks item 3.

> **BUILT 2026-07-22** — `backend/search/safe_subgame.py`: `GadgetCFR` (per-combo
> enter/opt-out gadget scaling the opponent's root reach by regret-matched enter
> probability), `opponent_alt_values` (both-frozen evaluation passes, side-effect
> free), `solve_subgame_safe` (σ0 → alt CFVs → gadget re-solve). Default solver when
> search is on (`HOLDEM_SAFE_SEARCH=0` selects plain, for A/B). Mechanics validated
> (`tests/test_safe_subgame.py`): worthless opt-out → always enter (reduces to plain
> re-solve); infinite opt-out → never enter; eval passes leave solver state intact and
> are zero-sum consistent. Remaining: the safe-vs-plain-vs-off A/B once the GPU frees
> (200bb trainer running), CUDA-graph support for the gadget loop (currently eager),
> and the reach-gifts upgrade.

- Safe and Nested Subgame Solving: https://arxiv.org/pdf/1705.02955
- Earlier workshop version: https://www.cs.cmu.edu/~sandholm/safeAndNested.aaa17WS.pdf

## 2. Potential-aware histogram abstraction (EMD k-means)

**Our state:** distribution-aware buckets v1 (mean-equity x equity-std bins) built and
consistency-tested; experiment paused at 26k iters.

**The upgrade (Ganzfried & Sandholm, AAAI 2014 — used by every top agent since):**
cluster full *histograms over future strength* with earth-mover's-distance k-means,
considering the trajectory across future rounds (potential-aware), not just terminal
equity moments. Our mean+std is a 2-moment approximation of this; the histogram version
separates draw *types* (nut draws vs weak draws vs backdoors), which our decision-log
analysis shows is the remaining abstraction gap (e.g. draws inheriting bucket-mates'
fold instead of a semi-bluff mix).

**Concrete exhibit (decision log, hand #222, 2026-07-22):** agent held 7♠5♠ on
4♠5♣6♠K♣ — pair + OESD + flush draw, ~19 outs, ~43% vs top pair — and folded 83%
to a 0.7-pot turn bet needing 29% (exact_match=true, scalar 200bb model, turn
bucket 13). The scalar bucket merges this combo-draw with static ~45%-equity hands,
which correctly fold vs polarized aggression; the draw inherits their fold. A
distribution-aware bucket separates them; safe turn re-solving (item 1) would play
the actual cards.

**Build:** replace `_bucket_from_mean_std` with EMD k-means over runout-equity
histograms in `deals.py` (numba/GPU kernels exist for equity already); refit, then
restart the 100bb distributional retrain and let the 10k milestone gate judge it.

> **BUILT 2026-07-22** — `DealSampler(histogram=True, hist_bins=N)`:
> `_equity_histograms` (full runout-equity distribution per combo),
> `fit_hist_centroids` (EMD k-means — 1-D EMD == L1 of CDFs, clustered in CDF
> space), `_bucket_from_histogram` (nearest-centroid assignment), state
> persistence, serving/training consistency by construction (shared code path;
> `tests/test_histogram_buckets.py` green). Trainer: `--histogram` flag selects
> `HISTOGRAM_SAMPLER` (150 flop/turn centroids, 24/16 runout samples, 10 bins)
> and auto-fits centroids on fresh runs. Subgame search inherits it via
> `partial_board_buckets`. Remaining: launch the retrain when the GPU frees
> (fresh run, `--tag hist`), gate it every 10k vs the scalar champion.

- Potential-Aware Imperfect-Recall Abstraction: https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf
- 2025 higher-resolution refinement: https://arxiv.org/html/2510.15094

## 3. Counterfactual-value network → depth-limited search on all streets

**The ceiling-raiser** (DeepStack, ReBeL, Supremus): stop solving to the end of the
game — search a few actions deep and evaluate the horizon with a neural net that
predicts per-hand counterfactual values from (pot, board, both ranges). This is the
architecture of every agent that beat professionals, and it makes preflop/flop search
tractable.

**Why we're well-positioned:** training data = solved subgames, and our GPU solver is a
verified-correct, fast subgame solver — exactly the data generator required. DeepStack's
net was small (7x500 FC, single GPU); Supremus' training recipe improves on it.

**Build (multi-week):** (a) random-situation generator (pot/board/ranges), (b) solve
each with the GPU solver, (c) train CFV net with zero-sum-enforcing output layer,
(d) depth-limited safe re-solve (item 1's gadget + net at the horizon), (e) A/B.

- DeepStack: https://arxiv.org/pdf/1701.01724
- ReBeL (RL+search formalization): https://arxiv.org/pdf/2007.13544
- Supremus DCVN training improvements: https://arxiv.org/pdf/2007.10442
- Survey of search in imperfect-info games: https://arxiv.org/pdf/2111.05884

## 4. RL-CFR: learned dynamic bet-sizing abstraction

Fixed action menus (our 0.5x/1x pot postflop) leave EV on the table. RL-CFR (ICML 2024)
learns *which sizes to consider per public state* and beat Slumbot by 84±17 mbb/hand.
Natural follow-up once search is safe; also the cheaper alternative — adding a third
static size to the menu — should be A/B'd first via the milestone gate.

- RL-CFR: https://arxiv.org/abs/2403.04344

## 5. Evaluation upgrades (cheap, do alongside)

- **AIVAT** variance reduction for Slumbot sessions → significance in hundreds, not
  thousands of hands.

> **CHANCE-VARIATES HALF BUILT 2026-07-23** — `backend/eval/aivat.py`
> (`ChanceCorrector`: board-reveal control variates, pot x equity-vs-uniform value
> fn, unbiased for any value fn), wired into `slumbot.py` (`aivat_bb_per_100` +
> `aivat_variance_reduction` in the report). Measured ~16% variance cut on a local
> styles fixture (`tests/test_aivat.py`); the remaining variance needs the
> decision-variates half (per-action value estimates — pair naturally with the
> CFV net of item 3). Safe re-solving v2 also landed the same day:
> `opponent_alt_values_br` prices the gadget opt-out at best-response-to-σ0 CFVs
> (the theoretically correct constraint; fixes v1's deviator blindspot) — its A/B
> vs plain (nit matchup especially) runs when the GPU frees.
- **GTO Wizard benchmark** as an external reference (their agent beats Slumbot by
  194±41 mbb — the bar for "genuinely strong"): https://arxiv.org/pdf/2509.23747
- Keep the head-to-head duel gate (`backend/eval/duel.py`) as the internal promotion
  signal — it measures "actually better," which styles benchmarks do not.

## Sequence

1. **Now:** repair Phase 4's blueprint projection so exact and coarse river
   trees may diverge safely; add shallow-stack, all-in, raise-cap, and off-tree
   regression tests.
2. **Then:** run a small exact-river engineering screen requiring zero
   projection failures and acceptable latency.
3. **Only after that screen passes:** repeat the 3,000-pair Phase 4
   confirmation. Keep normal serving search-off until its confidence interval
   clears zero and reliability remains at least 99%.
4. **Blueprint work:** do not add iterations to the plateaued clean no-limp
   checkpoint or resume the retired combined v3 + Phase 3 run unchanged.
5. **Later:** revive CFV-net depth-limited search only with higher-quality
   targets; continue evaluation upgrades and external benchmarking.
