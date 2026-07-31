"""P1 gate: continual exact-card resolving ON vs OFF, one frozen checkpoint.

Why a duel and not LBR as the primary signal: LBR's per-pair spread at 200bb is
~15 bb/hand, and the two arms decorrelate immediately (different actions ->
different trajectories), so even a paired LBR comparison needs ~850 pairs to
resolve a 1 bb/hand effect. The duel plays the two arms against EACH OTHER on
duplicate seat-swapped deals, which cancels card luck by construction and is the
project's NULL-tested promotion instrument (`tests/test_duel_null.py`).

This measures resolver value, not checkpoint strength: both arms are the same
checkpoint, differing only in the flag. Per `docs/TRAINING_QUALITY_OPTIMIZATION_PLAN.md`
Phase 4, a checkpoint must never be promoted from this gate.

Requirements to call it a pass:
  1. the 95% CI on (on minus off) clears zero;
  2. resolve reliability is ~100% (no silent blueprint fallback);
  3. the latency distribution fits the serving budget;
  4. a model-vs-itself null run stays centred at zero.

Usage:
    python tools/continual_search_gate.py --pairs 100 --iterations 60
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(checkpoint: Path, continual: bool, iterations: int, budget_ms: int):
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    agent = GpuBlueprintAgent.try_load(checkpoint)
    if agent is None:
        raise SystemExit(f"could not load {checkpoint}")
    agent.subgame_search = False
    agent.flop_search = False
    agent.exact_river_search = False
    agent.continual_search = continual
    agent.continual_iterations = iterations
    agent.continual_budget_ms = budget_ms
    return agent


class _Recorder:
    """Wraps an agent to capture its continual-resolve diagnostics."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.status = collections.Counter()
        self.errors = collections.Counter()
        self.by_street = collections.Counter()
        self.latencies: list[float] = []

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def select(self, game, player):
        # Streets 1, 2 and 3 -- flop, turn, river. This previously watched only
        # (2, 3), so flop resolves were INVISIBLE to the counter and a run that
        # never resolved a flop was indistinguishable from one that resolved
        # every flop. That is the same blind spot as STATUS.md 4.3 (silent
        # flop-solve discard) and 4.4 (the decision log could not see the
        # resolver), and it is why the 2026-07-31 gate output had to be
        # cross-checked against latency to learn that only rivers resolved.
        watch = self.agent.continual_search and game.street in (1, 2, 3)
        choice = self.agent.select(game, player)
        if watch:
            detail = self.agent.last_continual_search
            if detail is not None:
                status = str(detail.get("status"))
                self.status[status] += 1
                self.by_street[(detail.get("street"), status)] += 1
                if status == "resolved":
                    self.latencies.append(float(detail.get("decision_elapsed_ms", 0.0)))
                else:
                    self.errors[str(detail.get("error"))[:120]] += 1
                self.agent.last_continual_search = None
        return choice

    def execute(self, game, player, choice):
        return self.agent.execute(game, player, choice)

    def report(self) -> dict:
        attempts = sum(self.status.values())
        resolved = self.status.get("resolved", 0)
        latencies = sorted(self.latencies)
        return {
            "attempts": attempts,
            "resolved": resolved,
            "fallbacks": attempts - resolved,
            "resolve_rate": round(resolved / max(attempts, 1), 6),
            "by_street": {f"street{k[0]}:{k[1]}": v for k, v in self.by_street.items()},
            "errors": dict(self.errors),
            "latency_ms_mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "latency_ms_p90": round(latencies[int(0.9 * (len(latencies) - 1))], 1) if latencies else None,
            "latency_ms_max": round(max(latencies), 1) if latencies else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual-resolving on/off duel gate")
    parser.add_argument("--checkpoint", type=Path, default=Path("backend/data/gpu_blueprint_200bb/champion.npz"))
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=51009)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--budget-ms", type=int, default=20000)
    parser.add_argument("--null", action="store_true", help="off-vs-off null run; must read ~0")
    parser.add_argument("--crn", action="store_true",
                        help="common random numbers: couple both arms' action sampling (huge variance cut)")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    from backend.eval.duel import head_to_head

    challenger_continual = not arguments.null
    challenger = _Recorder(
        _load(arguments.checkpoint, challenger_continual, arguments.iterations, arguments.budget_ms)
    )
    incumbent = _load(arguments.checkpoint, False, arguments.iterations, arguments.budget_ms)

    label = "off-vs-off NULL" if arguments.null else "continual ON vs OFF"
    print(f"{label}: {arguments.pairs} duplicate pairs @ {arguments.stack_bb:.0f}bb, "
          f"{arguments.iterations} resolve iterations")
    sys.stdout.flush()

    started = time.time()
    result = head_to_head(
        challenger, incumbent, stack_bb=arguments.stack_bb,
        pairs=arguments.pairs, seed=arguments.seed,
        common_random_numbers=arguments.crn,
    )
    elapsed = round(time.time() - started, 1)
    diagnostics = challenger.report()

    report = {
        "gate": "continual-exact-resolving-on-vs-off",
        "null_run": arguments.null,
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_iteration": int(challenger.agent.iteration),
        "stack_bb": arguments.stack_bb,
        "resolve_iterations": arguments.iterations,
        "budget_ms": arguments.budget_ms,
        "common_random_numbers": arguments.crn,
        "elapsed_s": elapsed,
        "duel": result,
        "resolver": diagnostics,
        "eligible": (
            not arguments.null
            and result["verdict"] == "PROMOTE"
            and diagnostics["attempts"] > 0
            and diagnostics["fallbacks"] == 0
        ),
    }
    print(f"\n  on minus off: {result['mean_bb_per_100']:+.2f} bb/100 "
          f"[{result['ci_low_bb_per_100']:+.2f}, {result['ci_high_bb_per_100']:+.2f}] "
          f"n={result['hands']} ({elapsed}s)")
    print(f"  VERDICT: {result['verdict']}")
    print(f"  resolves: {diagnostics['resolved']}/{diagnostics['attempts']} "
          f"(fallbacks {diagnostics['fallbacks']}), by street {diagnostics['by_street']}")
    if diagnostics["errors"]:
        print(f"  errors: {diagnostics['errors']}")
    print(f"  latency ms mean/p90/max: {diagnostics['latency_ms_mean']}/"
          f"{diagnostics['latency_ms_p90']}/{diagnostics['latency_ms_max']}")
    print(f"  eligible: {report['eligible']}")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  written to {arguments.output}")


if __name__ == "__main__":
    main()
