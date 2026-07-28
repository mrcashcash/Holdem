"""Phase 4 engineering screen: projection reliability and latency.

The plan (docs/PLAN_V2_STRONGEST_PLAYER.md P1.1) requires a small screen that
passes with **zero projection failures** and acceptable latency before any
further large Phase 4 confirmation is run. This is that screen. It is not a
strength measurement -- it deliberately makes no claim about bb/100.

It reports what the repaired projection actually does in live play:

* resolve attempts, successes and fallbacks (grouped by cause);
* the detachment rate -- how much of each exact tree could NOT be projected from
  the coarse blueprint and fell back to the safe-default baseline. Zero
  fallbacks with near-total detachment would mean the resolver never aborts but
  its gadget opt-out prices are mostly safe defaults, which is a weaker safety
  guarantee than it looks. Both numbers have to be read together;
* full-decision latency distribution against the serving budget.

Usage:
    python tools/phase4_projection_screen.py --checkpoint backend/data/gpu_blueprint_200bb/champion.npz \
        --stack-bb 200 --hands 40
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

# tools/ is not a package, so make the repo root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.poker import HeadsUpHoldem  # noqa: E402


def run_screen(checkpoint: Path, stack_bb: float, hands: int, iterations: int, budget_ms: int, seed: int) -> dict:
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    searcher = GpuBlueprintAgent.try_load(checkpoint)
    opponent = GpuBlueprintAgent.try_load(checkpoint)
    if searcher is None or opponent is None:
        raise SystemExit(f"could not load {checkpoint}")
    for agent in (searcher, opponent):
        agent.subgame_search = False
        agent.flop_search = False
        agent.exact_river_search = False
    searcher.exact_river_search = True
    searcher.exact_river_iterations = max(12, iterations)
    searcher.exact_river_budget_ms = max(1, budget_ms)

    records: list[dict] = []
    big_blind = 20
    for hand_index in range(hands):
        engine = HeadsUpHoldem(
            initial_stack=int(round(stack_bb * big_blind)),
            small_blind=big_blind // 2,
            big_blind=big_blind,
            rng=random.Random(seed * 1_000_003 + hand_index),
        )
        # Alternate who searches so both positions are exercised.
        search_seat = hand_index % 2
        guard = 0
        while not engine.hand_complete and guard < 200:
            guard += 1
            player = engine.current_player
            actor = searcher if player == search_seat else opponent
            river_decision = engine.street == 3 and player == search_seat
            choice = actor.select(engine, player)
            if river_decision and actor.last_river_search is not None:
                records.append(dict(actor.last_river_search))
            actor.execute(engine, player, choice)

    attempts = len(records)
    resolved = [r for r in records if r.get("status") == "resolved"]
    fallbacks = [r for r in records if r.get("status") != "resolved"]
    causes: dict[str, int] = {}
    for record in fallbacks:
        causes[str(record.get("error", "unknown"))] = causes.get(str(record.get("error", "unknown")), 0) + 1

    latencies = [float(r.get("decision_elapsed_ms", 0.0)) for r in records if r.get("decision_elapsed_ms")]
    detach_fractions = [
        float(r["projection_detached_fraction"]) for r in resolved if "projection_detached_fraction" in r
    ]
    detach_reasons: dict[str, int] = {}
    for record in resolved:
        for reason, count in (record.get("projection_detach_reasons") or {}).items():
            detach_reasons[reason] = detach_reasons.get(reason, 0) + int(count)

    report = {
        "screen": "phase4-projection-reliability",
        "checkpoint": str(checkpoint),
        "stack_bb": stack_bb,
        "hands": hands,
        "iterations": searcher.exact_river_iterations,
        "budget_ms": searcher.exact_river_budget_ms,
        "attempts": attempts,
        "resolved": len(resolved),
        "fallbacks": len(fallbacks),
        "fallback_rate": round(len(fallbacks) / max(attempts, 1), 6),
        "fallback_causes": causes,
        "projection_detached_fraction_mean": (
            round(statistics.fmean(detach_fractions), 4) if detach_fractions else None
        ),
        "projection_detached_fraction_max": (round(max(detach_fractions), 4) if detach_fractions else None),
        "projection_fully_attached_solves": sum(1 for value in detach_fractions if value == 0.0),
        "projection_detach_reasons": detach_reasons,
        "latency_ms_mean": round(statistics.fmean(latencies), 1) if latencies else None,
        "latency_ms_max": round(max(latencies), 1) if latencies else None,
        "latency_ms_p90": (
            round(sorted(latencies)[int(0.9 * (len(latencies) - 1))], 1) if latencies else None
        ),
        "passes_zero_fallback_requirement": attempts > 0 and not fallbacks,
    }
    if resolved:
        sample = next((r for r in resolved if r.get("projection_detach_samples")), None)
        if sample is not None:
            report["example_detach_sample"] = sample["projection_detach_samples"][0]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 projection reliability screen")
    parser.add_argument("--checkpoint", type=Path, default=Path("backend/data/gpu_blueprint_200bb/champion.npz"))
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--hands", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--budget-ms", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=8131)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    started = time.time()
    report = run_screen(
        arguments.checkpoint,
        arguments.stack_bb,
        arguments.hands,
        arguments.iterations,
        arguments.budget_ms,
        arguments.seed,
    )
    report["elapsed_s"] = round(time.time() - started, 1)
    print(json.dumps(report, indent=2))
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"written to {arguments.output}")


if __name__ == "__main__":
    main()
