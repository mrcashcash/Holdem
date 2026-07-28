"""Paired Phase 4 on/off evaluation against one frozen blueprint.

This is intentionally separate from model promotion: it measures the value of
the resolver while holding every checkpoint parameter constant.  A checkpoint
must first pass the normal blueprint gate with search off; Phase 4 then needs
its own positive confirmation before the full stack is eligible to serve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.eval.duel import head_to_head


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Duel one checkpoint with exact river resolving on vs off"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stack-bb", type=float, required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=94041)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--budget-ms", type=int, default=6000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    search = GpuBlueprintAgent.try_load(arguments.checkpoint)
    blueprint = GpuBlueprintAgent.try_load(arguments.checkpoint)
    if search is None or blueprint is None:
        raise SystemExit(f"could not load checkpoint: {arguments.checkpoint}")

    for agent in (search, blueprint):
        agent.subgame_search = False
        agent.flop_search = False
    search.exact_river_search = True
    search.exact_river_iterations = max(12, int(arguments.iterations))
    search.exact_river_budget_ms = max(1, int(arguments.budget_ms))
    blueprint.exact_river_search = False

    started = time.time()
    report = head_to_head(
        search,
        blueprint,
        stack_bb=arguments.stack_bb,
        pairs=arguments.pairs,
        seed=arguments.seed,
        collect_diagnostics=True,
    )
    report.update(
        {
            "experiment": "phase4-exact-river-on-vs-off",
            "checkpoint": str(arguments.checkpoint.resolve()),
            "checkpoint_iteration": int(search.iteration),
            "resolver_iterations": int(search.exact_river_iterations),
            "resolver_budget_ms": int(search.exact_river_budget_ms),
            "elapsed_seconds": round(time.time() - started, 1),
            "eligible": (
                report["verdict"] == "PROMOTE"
                and report["diagnostics"]["river_search"]["challenger"][
                    "attempts"
                ]
                > 0
                and report["diagnostics"]["river_search"]["challenger"][
                    "fallback_rate"
                ]
                <= 0.01
            ),
        }
    )

    destination = arguments.output
    if destination is None:
        destination = (
            arguments.checkpoint.parent
            / "evaluations"
            / f"phase4-river-ab-{int(time.time())}.json"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {destination.resolve()}")


if __name__ == "__main__":
    main()
