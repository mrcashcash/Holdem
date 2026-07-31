"""P2 gate: does DCFR+ linear averaging beat DCFR quadratic averaging?

Supremus (arXiv 2007.10442) reports that weighting iteration t by max{0, t - d}
in the average policy, rather than t**gamma, converges faster -- with d = 100 in
their experiments. It is a few lines in `VectorCFR._discount` and needs no extra
memory or tree, which makes it the cheapest available quality lever at a moment
when the 200bb tree is already at this card's VRAM ceiling.

Measured with the project's trusted instrument: `abstract_exploitability_mbb`, the
independent Kuhn/Leduc-validated best response used by
`tests/test_gpu_convergence.py`. Lower mbb is better. Both arms share tree,
sampler, seed and iteration count, so the averaging rule is the only difference.

An iteration sweep matters more than a single point: Supremus's claim is about
convergence SPEED, so the interesting question is whether DCFR+ reaches a given
exploitability in fewer iterations, not only where both land at the end.

Usage:
    python tools/dcfr_plus_gate.py --iterations 1000 2000 4000 --delays 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.search.gpu_subgame import FixedBoardSampler  # noqa: E402
from backend.solver.gpu.cfr import VectorCFR  # noqa: E402
from backend.solver.gpu.convergence import abstract_exploitability_mbb  # noqa: E402
from backend.solver.gpu.deals import DealSampler  # noqa: E402
from backend.solver.gpu.tree import BettingTree, GpuActionConfig  # noqa: E402

# Identical to tests/test_gpu_convergence.py so the numbers are comparable with
# the recorded regression guard.
BOARD = (2, 7, 24, 33, 50)


def build_solver(*, dcfr_plus_delay: int | None, averaging_delay: int, seed: int):
    cfg = GpuActionConfig(
        preflop_fractions=(1.0,),
        postflop_fractions=(0.75,),
        max_raises_per_street=2,
        stack_bb=20.0,
    )
    tree = BettingTree(cfg, start_street=3, start_pot=8.0, start_stacks=(16.0, 16.0))
    sampler = FixedBoardSampler(DealSampler(river_buckets=20), BOARD)
    return VectorCFR(
        tree, sampler, device="cpu", seed=seed,
        averaging_delay=averaging_delay,
        dcfr_plus_delay=dcfr_plus_delay,
    )


def measure(*, iterations: int, dcfr_plus_delay: int | None, averaging_delay: int,
            seed: int) -> tuple[float, float]:
    solver = build_solver(
        dcfr_plus_delay=dcfr_plus_delay, averaging_delay=averaging_delay, seed=seed)
    started = time.time()
    solver.run(iterations)
    elapsed = time.time() - started
    return abstract_exploitability_mbb(solver, seed=0), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="DCFR vs DCFR+ averaging gate")
    parser.add_argument("--iterations", type=int, nargs="*", default=[1000, 2000, 4000])
    parser.add_argument("--delays", type=int, nargs="*", default=[100],
                        help="DCFR+ d values to try; Supremus used 100")
    parser.add_argument("--seeds", type=int, nargs="*", default=[23],
                        help="repeat each cell over these solver seeds")
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/dcfr-plus-gate.json"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output.with_suffix(".log")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    log(f"=== DCFR vs DCFR+ : iterations {arguments.iterations}, "
        f"delays {arguments.delays}, seeds {arguments.seeds} ===")
    log("instrument: abstract_exploitability_mbb (independent BR); lower is better")
    log(f"durable log: {log_path}")

    records = []
    for iterations in arguments.iterations:
        for seed in arguments.seeds:
            # Baseline uses averaging_delay=100, matching the recorded guard, so
            # the comparison is against the configuration actually in use.
            base_mbb, base_s = measure(
                iterations=iterations, dcfr_plus_delay=None,
                averaging_delay=100, seed=seed)
            log(f"it={iterations:>5} seed={seed}  DCFR (t^gamma)      "
                f"{base_mbb:>9.2f} mbb  {base_s:>6.1f}s")
            records.append({"iterations": iterations, "seed": seed, "arm": "dcfr",
                            "mbb": base_mbb, "elapsed_s": round(base_s, 1)})
            for delay in arguments.delays:
                # DCFR+ discards iterations <= d itself, so averaging_delay is
                # left at 0 to avoid applying the same cut twice.
                plus_mbb, plus_s = measure(
                    iterations=iterations, dcfr_plus_delay=delay,
                    averaging_delay=0, seed=seed)
                better = "BETTER" if plus_mbb < base_mbb else "worse"
                log(f"it={iterations:>5} seed={seed}  DCFR+ max(0,t-{delay:<3d}) "
                    f"{plus_mbb:>9.2f} mbb  {plus_s:>6.1f}s  "
                    f"{better} by {abs(plus_mbb - base_mbb):.2f}")
                records.append({"iterations": iterations, "seed": seed,
                                "arm": f"dcfr_plus_d{delay}", "mbb": plus_mbb,
                                "elapsed_s": round(plus_s, 1),
                                "delta_vs_dcfr": round(plus_mbb - base_mbb, 3)})

    log("")
    log("--- summary (mean mbb across seeds; lower is better) ---")
    arms = sorted({record["arm"] for record in records})
    log(f"{'iterations':>10} " + " ".join(f"{arm:>20}" for arm in arms))
    verdict_rows = []
    for iterations in arguments.iterations:
        cells = []
        for arm in arms:
            values = [r["mbb"] for r in records
                      if r["iterations"] == iterations and r["arm"] == arm]
            cells.append(sum(values) / len(values) if values else float("nan"))
        log(f"{iterations:>10} " + " ".join(f"{cell:>20.2f}" for cell in cells))
        verdict_rows.append((iterations, dict(zip(arms, cells))))

    log("")
    # Ties at the convergence floor are NOT wins. The fixed-river control game
    # reaches 0.00 mbb under both rules by 1,000 iterations, so without a
    # tolerance every converged cell reads as a DCFR+ victory on float noise.
    TIE = 0.5  # mbb; below this the arms are indistinguishable on this instrument
    wins, ties = 0, 0
    for _, row in verdict_rows:
        base = row.get("dcfr", float("inf"))
        best_other = min((value for arm, value in row.items() if arm != "dcfr"),
                         default=float("inf"))
        if abs(best_other - base) <= TIE:
            ties += 1
        elif best_other < base:
            wins += 1
    if ties == len(verdict_rows) and verdict_rows:
        log(f"every cell tied within {TIE} mbb -- this game is already converged at")
        log("these iteration counts, so the comparison is UNINFORMATIVE. Re-run at")
        log("lower iteration counts, or on a game that is not yet solved.")
    elif wins == len(verdict_rows) and verdict_rows:
        log("DCFR+ won at EVERY iteration count -> adopt it (then re-run the")
        log("convergence guard and a CRN duel before changing any default).")
    elif wins == 0:
        log("DCFR+ never won -> do not adopt; the quadratic averaging stands.")
    else:
        log(f"DCFR+ won at {wins}/{len(verdict_rows)} iteration counts -> mixed;")
        log("more seeds are needed before this is a decision rather than noise.")

    arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    handle.close()


if __name__ == "__main__":
    main()
