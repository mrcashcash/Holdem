"""Does the FIXED all-in geometry guard still cost 124-269 bb/100 head-to-head?

This is the measurement that put the guard behind a flag in the first place.
`docs/STATUS.md` 3.6 recorded, against an always-min-raise opponent:

    200bb  guard ON +80.58  vs  guard OFF +349.40   delta -268.82
    100bb  guard ON +104.65 vs  guard OFF +228.65   delta -124.00

and interpreted it as a GTO-versus-exploitation tension. That interpretation was
wrong: 89.6% of the guard's firings were false positives caused by the absolute
cap being folded into the trigger, so the guard was trimming perfectly translated
jams -- including every preflop shove. That defect is fixed.

The fixed guard now measurably REDUCES exploitability (LBR paired, 20,000 pairs:
100bb -5.65 [-10.30, -1.00], significant; 20/50/200bb neutral to helpful). But
exploitability is not the whole picture: a guard that shrinks jams could still
give up money against a station that calls anything. So re-run the head-to-head
that condemned it.

Both arms are the SAME checkpoint differing only in the flag, blueprint-only, with
CRN coupling so the arms diverge only where their policies differ. Each arm plays
the same scripted opponent on the same deals.

Usage:
    python tools/guard_headtohead_gate.py --pairs 3000 --stack-bb 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OPPONENTS = ("always-min-raise", "always-call")


def build_agent(checkpoint: Path, *, guard: bool):
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    agent = GpuBlueprintAgent.try_load(checkpoint)
    if agent is None:
        raise SystemExit(f"could not load {checkpoint}")
    # Blueprint only: isolate the guard from every search mechanism, matching the
    # configuration the original -268.82 was measured in.
    agent.subgame_search = False
    agent.flop_search = False
    agent.exact_river_search = False
    agent.continual_search = False
    agent.all_in_geometry_guard = guard
    if bool(agent.all_in_geometry_guard) is not guard:
        raise SystemExit(f"guard flag did not take (wanted {guard})")
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head cost of the fixed all-in geometry guard")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("backend/data/gpu_blueprint_200bb/champion.npz"))
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    from backend.eval.duel import head_to_head
    from backend.eval.null_agents import ScriptedAgent

    output = arguments.output or Path(
        f"backend/data/evaluations/guard-h2h-{int(arguments.stack_bb)}bb.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    log(f"=== guard head-to-head: {arguments.pairs} pairs @ "
        f"{arguments.stack_bb:.0f}bb, seed {arguments.seed} ===")
    log(f"checkpoint: {arguments.checkpoint}")

    results: dict = {}
    for opponent_name in OPPONENTS:
        log(f"--- opponent: {opponent_name} ---")
        arms = {}
        for label, guard in (("off", False), ("on", True)):
            started = time.time()
            hero = build_agent(arguments.checkpoint, guard=guard)
            report = head_to_head(
                hero,
                ScriptedAgent(opponent_name),
                stack_bb=arguments.stack_bb,
                pairs=arguments.pairs,
                seed=arguments.seed,
                common_random_numbers=True,
            )
            elapsed = round(time.time() - started, 1)
            # duel.py names these mean_bb_per_100 / ci_*_bb_per_100, while
            # lbr.py uses lbr_bb_per_100. Mixing the two conventions up is what
            # crashed the first attempt at this measurement (STATUS.md 3.6), so
            # read them explicitly and fail loudly if they are absent.
            missing = [
                key for key in ("mean_bb_per_100", "ci_low_bb_per_100", "ci_high_bb_per_100")
                if report.get(key) is None
            ]
            if missing:
                raise SystemExit(f"head_to_head did not return {missing}; got {sorted(report)}")
            arms[label] = {
                "bb_per_100": report["mean_bb_per_100"],
                "ci_low": report["ci_low_bb_per_100"],
                "ci_high": report["ci_high_bb_per_100"],
                "elapsed_s": elapsed,
            }
            log(f"  guard {label.upper():3}: {arms[label]['bb_per_100']:+.2f} bb/100 "
                f"[{arms[label]['ci_low']:+.2f},{arms[label]['ci_high']:+.2f}] in {elapsed}s")
        delta = arms["on"]["bb_per_100"] - arms["off"]["bb_per_100"]
        arms["delta_on_minus_off"] = round(delta, 2)
        log(f"  DELTA (on - off): {delta:+.2f} bb/100")
        results[opponent_name] = arms

    report = {
        "gate": "all-in-geometry-guard-head-to-head",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(arguments.checkpoint),
        "stack_bb": arguments.stack_bb,
        "pairs": arguments.pairs,
        "seed": arguments.seed,
        "search": "blueprint only, CRN coupled",
        "opponents": results,
        "prior_measurement_before_fix": {
            "200bb_always_min_raise_delta": -268.82,
            "100bb_always_min_raise_delta": -124.00,
        },
    }
    log("")
    log("--- comparison with the pre-fix measurement ---")
    for name, arms in results.items():
        log(f"  {name}: delta {arms['delta_on_minus_off']:+.2f} bb/100")
    log("A delta near zero means the -268.82 was the defect, not the mechanism.")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"written to {output}")
    handle.close()


if __name__ == "__main__":
    main()
