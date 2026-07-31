"""Run the agent (or a null agent) against GTO Wizard AI.

The NULL test comes first and is not optional. `docs/STATUS.md` §4 records five
instrument bugs that each invalidated weeks of results, and bug 5 specifically was
a null that PASSED on a different agent class than the one being measured. So:

    # validate the harness against the published anchors first
    python tools/gtowizard_benchmark.py --agent always-fold  --hands 200
    python tools/gtowizard_benchmark.py --agent always-call  --hands 200
    python tools/gtowizard_benchmark.py --agent always-all-in --hands 200

    # only then measure the real thing
    python tools/gtowizard_benchmark.py --agent serving --hands 2000

Published anchors (arXiv 2603.23660 Table 2), AIVAT-adjusted bb/100:
    always-fold -64.6 +- 3.3 | check-call -241.1 +- 26.2 | all-in -380.6 +- 4.3

Resumable: progress is checkpointed every 25 hands next to the output, so an
interrupted run continues instead of re-spending API quota. The monthly cap is
100,000 hands, so a wasted restart is a real cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.eval.gtowizard import (  # noqa: E402
    PUBLISHED_ANCHORS,
    play_match,
    server_results,
)

NULL_POLICIES = ("always-fold", "always-call", "always-min-raise", "always-all-in")


def build_agent(name: str):
    if name in NULL_POLICIES:
        from backend.eval.null_agents import ScriptedAgent

        return ScriptedAgent(name), f"null:{name}"
    if name == "serving":
        from backend.agents.serving import load_serving_agent

        agent = load_serving_agent()
        return agent, f"serving:{type(agent).__name__}"
    raise SystemExit(f"unknown --agent {name!r}; use 'serving' or one of {NULL_POLICIES}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark against GTO Wizard AI")
    parser.add_argument("--agent", default="always-fold",
                        help="'serving', or a null policy for harness validation")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--resolver", choices=("on", "off"), default="on",
                        help="continual exact-card resolving, for the serving agent")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    output = arguments.output or Path(
        f"backend/data/evaluations/gtowizard-{arguments.agent}-{arguments.hands}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    agent, label = build_agent(arguments.agent)
    if arguments.agent == "serving" and hasattr(agent, "continual_search"):
        agent.continual_search = arguments.resolver == "on"
        log(f"resolver: {'ON' if agent.continual_search else 'OFF'}")

    log(f"=== GTO Wizard benchmark: {label}, {arguments.hands} hands ===")
    log(f"durable log: {log_path}")

    before = server_results()
    log(f"server tally before: {before['total_hands']} hands, "
        f"{before['bb_per_100']:+.2f} bb/100")

    report = play_match(
        agent, arguments.hands, seed=arguments.seed, log=log,
        checkpoint=output.with_suffix(".checkpoint.json"))
    report["agent"] = label
    report["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    adjusted, raw = report["aivat"], report["raw"]
    log("")
    log(f"AIVAT : {adjusted['bb_per_100']:+.2f} bb/100 "
        f"[{adjusted['ci_low']:+.2f}, {adjusted['ci_high']:+.2f}] over "
        f"{adjusted['hands']} hands")
    log(f"raw   : {raw['bb_per_100']:+.2f} bb/100 "
        f"[{raw['ci_low']:+.2f}, {raw['ci_high']:+.2f}]")
    log(f"excluded={report['excluded']} board_desyncs={report['board_desyncs']} "
        f"positions={report['positions']}")
    log(f"timing: {report['timing_s']}")

    # Cross-check our arithmetic against the server's own tally.
    after = server_results()
    report["server_results_after"] = after
    log(f"server tally after : {after['total_hands']} hands, "
        f"{after['bb_per_100']:+.2f} bb/100, "
        f"aivat {after['aivat_score_bb_per_100']:+.2f} bb/100")

    # NULL validation against the published anchor, when one exists.
    anchor = PUBLISHED_ANCHORS.get(arguments.agent)
    if anchor is not None:
        expected, margin = anchor
        low, high = adjusted["ci_low"], adjusted["ci_high"]
        overlaps = low <= expected + margin and high >= expected - margin
        report["null_anchor"] = {
            "expected_bb_per_100": expected,
            "published_margin": margin,
            "overlaps": overlaps,
        }
        log("")
        log(f"NULL ANCHOR: published {expected:+.1f} +- {margin:.1f} bb/100")
        log(f"  measured  {adjusted['bb_per_100']:+.2f} [{low:+.2f},{high:+.2f}]")
        log(f"  {'PASS - intervals overlap' if overlaps else 'FAIL - DO NOT TRUST THIS HARNESS'}")

    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"written to {output}")
    handle.close()


if __name__ == "__main__":
    main()
