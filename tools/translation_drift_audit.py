"""How much geometry drift does `_locate` accumulate, and where?

`_locate` walks the abstract tree along the hand's public actions, translating each
one with `_translate_event`. That translation is locally sensible -- pseudo-harmonic
between the two nearest abstract sizes -- but **nothing tracks accumulated drift**,
so per-action errors compound in one direction. `docs/STATUS.md` §3.6 states the
consequence: `_locate` "matches nodes by translated action sequence and never
compares pot/stack geometry."

The 2026-07-31 distortion audit measured drift only where the agent went ALL-IN
(7 of 11 distorted, worst 6.25x). That is where drift becomes visible, not where it
occurs. This measures it at EVERY decision, so a fix has a yardstick.

The quantity, unitless so bb and chips cannot be confused (the same form the all-in
guard uses):

    ratio_abstract = stack_bb / abstract matched pot at the located node
    ratio_real     = (own committed + own stack) / real matched pot
    drift          = ratio_real / ratio_abstract

drift == 1 means the located node has the same stack-to-pot geometry as reality.
drift > 1 means the abstract node is deeper into its pot than reality is -- the
trained action there means something more committal than the situation warrants.

Reports the distribution by street and by how many translated actions have
accumulated, which is the compounding claim's direct test.

Usage:
    python tools/translation_drift_audit.py --hands 300 --opponent always-min-raise
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STREETS = ("preflop", "flop", "turn", "river")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translation geometry-drift audit")
    parser.add_argument("--hands", type=int, default=300)
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--opponent", default="always-min-raise")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/translation-drift.json"))
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

    from backend.agents.serving import load_serving_agent
    from backend.eval.null_agents import ScriptedAgent
    from backend.poker import HeadsUpHoldem

    agent = load_serving_agent()
    if not hasattr(agent, "agents"):
        raise SystemExit("expected the multi-stack router")
    agent.continual_search = False        # measure the blueprint's own translation
    agent.all_in_geometry_guard = False   # serving default; behaviour unchanged

    records: list[dict] = []

    # Wrap each sub-agent's _locate read-only: run the real one, then measure the
    # geometry of whatever node it returned. A probe must never change play.
    for sub in agent.agents.values():
        if not hasattr(sub, "_locate"):
            continue
        original = sub._locate

        def probe(game, player, _sub=sub, _original=original):
            node = _original(game, player)
            if node is None:
                return node
            try:
                matched_abstract = float(_sub._abstract_matched_pot(node))
                matched_real = 2.0 * float(min(game.contributions))
                if matched_abstract > 0.0 and matched_real > 0.0:
                    ratio_abstract = float(_sub.tree.config.stack_bb) / matched_abstract
                    ratio_real = (
                        float(game.contributions[player]) + float(game.stacks[player])
                    ) / matched_real
                    translated = sum(
                        1 for event in game.public_actions if event["action"] != "blind"
                    )
                    records.append({
                        "street": STREETS[min(int(game.street), 3)],
                        "translated_actions": translated,
                        "ratio_abstract": round(ratio_abstract, 4),
                        "ratio_real": round(ratio_real, 4),
                        "drift": round(ratio_real / max(ratio_abstract, 1e-9), 4),
                    })
            except Exception:
                pass
            return node

        sub._locate = probe

    log(f"=== translation drift: {arguments.hands} hands @ {arguments.stack_bb:.0f}bb "
        f"vs {arguments.opponent} ===")
    log(f"durable log: {log_path}")

    big_blind = 20
    stack = int(arguments.stack_bb) * big_blind
    villain = ScriptedAgent(arguments.opponent)
    engine = HeadsUpHoldem(initial_stack=stack, small_blind=big_blind // 2,
                           big_blind=big_blind, rng=random.Random(arguments.seed))
    for _ in range(arguments.hands):
        engine.stacks = [stack, stack]
        engine.new_hand()
        while not engine.hand_complete:
            seat = engine.current_player
            if seat != 0:
                villain.execute(engine, seat, villain.select(engine, seat))
                continue
            agent.execute(engine, 0, agent.select(engine, 0))

    if not records:
        log("no located decision was observed; nothing to measure")
        raise SystemExit(0)

    drifts = [r["drift"] for r in records]

    def describe(values: list[float]) -> str:
        ordered = sorted(values)
        median = statistics.median(ordered)
        p90 = ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))]
        return (f"n={len(values):>5} median={median:>6.2f} p90={p90:>7.2f} "
                f"max={max(ordered):>8.2f} >1.5x={sum(1 for v in values if v > 1.5) / len(values):>6.1%}")

    log("")
    log(f"ALL located decisions   {describe(drifts)}")
    log("")
    log("by street:")
    by_street: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_street[record["street"]].append(record["drift"])
    for street in STREETS:
        if by_street[street]:
            log(f"  {street:>7}  {describe(by_street[street])}")

    log("")
    log("by number of translated actions accumulated (the compounding test):")
    by_depth: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_depth[min(record["translated_actions"], 10)].append(record["drift"])
    for depth in sorted(by_depth):
        label = f"{depth}" if depth < 10 else "10+"
        log(f"  {label:>4} actions  {describe(by_depth[depth])}")

    # Does drift grow with translated depth? Compare the shallow and deep halves.
    shallow = [r["drift"] for r in records if r["translated_actions"] <= 3]
    deep = [r["drift"] for r in records if r["translated_actions"] >= 6]
    log("")
    if shallow and deep:
        log(f"median drift, <=3 translated actions : {statistics.median(shallow):.3f} "
            f"(n={len(shallow)})")
        log(f"median drift, >=6 translated actions : {statistics.median(deep):.3f} "
            f"(n={len(deep)})")
        if statistics.median(deep) > statistics.median(shallow) * 1.2:
            log("-> drift GROWS with translated depth: compounding confirmed, and a")
            log("   geometry-aware _locate has something real to correct.")
        else:
            log("-> drift does NOT grow appreciably with translated depth. The")
            log("   compounding story in STATUS.md 3.6 is not supported here, so a")
            log("   geometry-aware _locate would be fixing something that is not the")
            log("   mechanism. Re-examine before building it.")
    else:
        log("not enough spread in translated depth to test compounding")

    arguments.output.write_text(json.dumps({
        "hands": arguments.hands,
        "stack_bb": arguments.stack_bb,
        "opponent": arguments.opponent,
        "decisions": len(records),
        "drift_median": round(statistics.median(drifts), 4),
        "drift_max": round(max(drifts), 4),
        "share_above_1_5": round(sum(1 for v in drifts if v > 1.5) / len(drifts), 4),
        "records": records,
    }, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    handle.close()


if __name__ == "__main__":
    main()
