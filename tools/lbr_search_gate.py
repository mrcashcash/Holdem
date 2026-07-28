"""P1 gate: does exact-card continual resolving reduce LBR exploitability?

LBR is the plan's north star (docs/PLAN_V2_STRONGEST_PLAYER.md P0.2: Slumbot and
LBR disagree, and LBR is the one measuring distance from equilibrium). The
measured baseline is +291.23 bb/100 against the 200bb champion -- the probe wins
~2,910 mbb/hand.

The comparison is **paired**: both arms probe the same deals with the same seed,
so per-pair differences cancel card variance. Unpaired LBR at 400 pairs is
+-140 bb/100, which could not resolve anything short of a total fix.

Two honest caveats, stated because they bound what this measures:

1. **LBR models the agent as its blueprint.** Its range beliefs come from the
   blueprint policy, so against a resolving agent those beliefs are
   mis-specified and LBR is a weaker exploiter than it could be. The measured
   win rate is still real -- LBR is a fixed strategy that achieves it -- and the
   probe is IDENTICAL in both arms, so the A/B is apples-to-apples. It is a
   comparison of "loses less to this probe", not of true exploitability.
2. A reduction here is necessary, not sufficient. The Slumbot anchor and a
   head-to-head duel remain part of the P1 gate.

Usage:
    python tools/lbr_search_gate.py --pairs 120 --iterations 60
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _summary(samples: list[float]) -> dict:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(len(samples)) if samples else 0.0
    return {
        "bb_per_100": round(mean * 100, 2),
        "ci_low": round((mean - margin) * 100, 2),
        "ci_high": round((mean + margin) * 100, 2),
        "stdev": round(deviation, 4),
        "pairs": len(samples),
    }


def run_arm(checkpoint: Path, stack_bb: float, pairs: int, seed: int, continual: bool,
            iterations: int, budget_ms: int) -> dict:
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
    from backend.eval.lbr import local_best_response_probe

    agent = GpuBlueprintAgent.try_load(checkpoint)
    if agent is None:
        raise SystemExit(f"could not load {checkpoint}")
    agent.subgame_search = False
    agent.flop_search = False
    agent.exact_river_search = False
    agent.continual_search = continual
    agent.continual_iterations = iterations
    agent.continual_budget_ms = budget_ms

    started = time.time()
    report = local_best_response_probe(agent, hands=pairs * 2, seed=seed, stack_bb=stack_bb)
    report["elapsed_s"] = round(time.time() - started, 1)
    report["continual"] = continual
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired LBR gate for continual resolving")
    parser.add_argument("--checkpoint", type=Path, default=Path("backend/data/gpu_blueprint_200bb/champion.npz"))
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--budget-ms", type=int, default=20000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    print(f"LBR gate: {arguments.pairs} duplicate pairs, seed {arguments.seed}, "
          f"{arguments.iterations} resolve iterations")
    print("arm 1/2: blueprint only (search off)")
    sys.stdout.flush()
    off = run_arm(arguments.checkpoint, arguments.stack_bb, arguments.pairs, arguments.seed,
                  False, arguments.iterations, arguments.budget_ms)
    print(f"  LBR {off['lbr_bb_per_100']:+.2f} bb/100 "
          f"[{off['ci_low_bb_per_100']:+.2f},{off['ci_high_bb_per_100']:+.2f}] in {off['elapsed_s']}s")
    print("arm 2/2: continual exact-card turn+river resolving")
    sys.stdout.flush()
    on = run_arm(arguments.checkpoint, arguments.stack_bb, arguments.pairs, arguments.seed,
                 True, arguments.iterations, arguments.budget_ms)
    print(f"  LBR {on['lbr_bb_per_100']:+.2f} bb/100 "
          f"[{on['ci_low_bb_per_100']:+.2f},{on['ci_high_bb_per_100']:+.2f}] in {on['elapsed_s']}s")

    left, right = off["pair_samples"], on["pair_samples"]
    if len(left) != len(right):
        raise SystemExit(f"arms disagree on pair count ({len(left)} vs {len(right)})")
    # Positive difference = the probe wins MORE against the resolving agent,
    # i.e. resolving made things worse. Negative = exploitability reduced.
    deltas = [b - a for a, b in zip(left, right)]
    paired = _summary(deltas)
    verdict = (
        "IMPROVED" if paired["ci_high"] < 0
        else "REGRESSED" if paired["ci_low"] > 0
        else "INCONCLUSIVE"
    )
    report = {
        "gate": "lbr-continual-on-vs-off",
        "checkpoint": str(arguments.checkpoint),
        "stack_bb": arguments.stack_bb,
        "resolve_iterations": arguments.iterations,
        "off": {k: v for k, v in off.items() if k != "pair_samples"},
        "on": {k: v for k, v in on.items() if k != "pair_samples"},
        "paired_delta": paired,
        "verdict": verdict,
    }
    print("\n--- paired difference (on minus off), negative is better ---")
    print(f"  {paired['bb_per_100']:+.2f} bb/100 "
          f"[{paired['ci_low']:+.2f}, {paired['ci_high']:+.2f}] over {paired['pairs']} pairs")
    print(f"  VERDICT: {verdict}")
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  written to {arguments.output}")


if __name__ == "__main__":
    main()
