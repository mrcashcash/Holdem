"""Were the reported "preflop overbets" real artifacts, or legitimate shoves?

`tools/overbet_audit.py` flags a bet when `amount / pot_before >= 3.0`. That is an
ABSOLUTE pot multiple, and it is the same assumption that made the all-in geometry
guard fire on 89.6% false positives: preflop the matched pot is one or two big
blinds, so a perfectly correct 200bb shove is inherently 100-200x pot and trips any
absolute bound. STATUS.md 3.6 records "four of the eight were preflop, which no
postflop resolving can reach", and P3 in PLAN_V3 budgets 1-2 weeks on that basis.

If those four were legitimate shoves, the budget is aimed at a leak that does not
exist. The distinguishing quantity is DISTORTION, not size:

    ratio_abstract = stack_bb / abstract matched pot   (what the trained jam meant)
    ratio_real     = committed-after-shove / real matched pot
    distortion     = ratio_real / ratio_abstract

A translation artifact has distortion >> 1 -- the real geometry is nothing like the
node the action was trained on. A legitimate shove has distortion ~1 however large
the absolute multiple is. The 24x-pot live hand had ratio_abstract 3.33 against
ratio_real 24, i.e. distortion ~7. A 200bb preflop jam has both ratios at ~200,
distortion 1.0.

Method: play the real serving path against `always-min-raise` -- the only opponent
that provoked overbets at all, because only it exhausts the raise cap -- and wrap
each sub-agent's `_all_in_size` with a READ-ONLY probe. The guard stays OFF, so
behaviour is byte-identical to serving; the wrapper computes the ratios itself
rather than relying on the guard's diagnostics, which are only produced when the
guard fires.

Usage:
    python tools/overbet_distortion_audit.py --hands 200 --stack-bb 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Same tolerance the guard uses: beyond 1.5x the trained meaning is distortion.
DISTORTION_TOLERANCE = 1.5
#: The absolute bound the old audit used, kept only to reproduce its verdict.
LEGACY_OVERBET_RATIO = 3.0
STREETS = ("preflop", "flop", "turn", "river")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distortion-based overbet audit")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=20260728,
                        help="matches the seed the original audit used")
    parser.add_argument("--opponent", default="always-min-raise")
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/overbet-distortion.json"))
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
    if not hasattr(agent, "all_in_geometry_guard"):
        raise SystemExit("no blueprint checkpoint available to serve")
    agent.continual_search = False       # isolate the blueprint, as the original did
    agent.all_in_geometry_guard = False  # serving default; behaviour unchanged

    records: list[dict] = []
    big_blind = 20
    stack = int(arguments.stack_bb) * big_blind

    # Wrap every sub-agent's _all_in_size read-only. The guard is off, so the
    # original returns None immediately without computing ratios -- the wrapper
    # therefore derives them itself, using the same formulas the guard uses.
    subs = getattr(agent, "agents", None) or {arguments.stack_bb: agent}
    for sub in subs.values():
        if not hasattr(sub, "_all_in_size"):
            continue
        original = sub._all_in_size

        def probe(game, player, node, _sub=sub, _original=original):
            try:
                matched_abstract = float(_sub._abstract_matched_pot(node))
                matched_real = 2.0 * float(min(game.contributions))
                if matched_abstract > 0.0 and matched_real > 0.0:
                    ratio_abstract = float(_sub.tree.config.stack_bb) / matched_abstract
                    ratio_real = (
                        float(game.contributions[player]) + float(game.stacks[player])
                    ) / matched_real
                    records.append({
                        "street": STREETS[min(int(game.street), 3)],
                        "ratio_abstract": round(ratio_abstract, 3),
                        "ratio_real": round(ratio_real, 3),
                        "distortion": round(ratio_real / max(ratio_abstract, 1e-9), 3),
                        "matched_real_chips": matched_real,
                    })
            except Exception:  # a probe must never change play
                pass
            return _original(game, player, node)

        sub._all_in_size = probe

    log(f"=== distortion audit: {arguments.hands} hands @ {arguments.stack_bb:.0f}bb "
        f"vs {arguments.opponent}, guard OFF ===")
    log(f"durable log: {log_path}")

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
        log("no translated ALL-IN was reached; nothing to classify")
        raise SystemExit(0)

    distorted = [r for r in records if r["distortion"] > DISTORTION_TOLERANCE]
    legacy = [r for r in records if r["ratio_real"] >= LEGACY_OVERBET_RATIO]
    by_street = Counter(r["street"] for r in records)
    distorted_by_street = Counter(r["street"] for r in distorted)
    legacy_by_street = Counter(r["street"] for r in legacy)

    log("")
    log(f"translated ALL-INs reached: {len(records)}")
    log(f"{'street':>8} {'all-ins':>8} {'legacy >=3x pot':>16} {'DISTORTED >1.5x':>17}")
    for street in STREETS:
        if by_street[street]:
            log(f"{street:>8} {by_street[street]:>8} {legacy_by_street[street]:>16} "
                f"{distorted_by_street[street]:>17}")
    log(f"{'TOTAL':>8} {len(records):>8} {len(legacy):>16} {len(distorted):>17}")

    log("")
    preflop_legacy = legacy_by_street["preflop"]
    preflop_distorted = distorted_by_street["preflop"]
    if preflop_legacy and not preflop_distorted:
        log(f"FINDING: all {preflop_legacy} preflop jams the absolute criterion flags")
        log("are UNDISTORTED -- the real geometry matches what the trained jam meant.")
        log("They are legitimate shoves, not translation artifacts. The preflop")
        log("translation leak in STATUS.md 3.6 is an artifact of the >=3x-pot rule,")
        log("and PLAN_V3 P3's 1-2 week preflop budget is aimed at nothing.")
    elif preflop_distorted:
        log(f"FINDING: {preflop_distorted} of {by_street['preflop']} preflop jams ARE")
        log("genuinely distorted, so a real preflop translation leak exists and P3")
        log("is justified. Worst cases:")
        for row in sorted(distorted, key=lambda r: -r["distortion"])[:5]:
            if row["street"] == "preflop":
                log(f"  abstract {row['ratio_abstract']}x -> real {row['ratio_real']}x "
                    f"(distortion {row['distortion']}x)")
    else:
        log("no preflop all-in was flagged by either criterion in this sample")

    if distorted:
        log("")
        log("worst distortions overall:")
        for row in sorted(distorted, key=lambda r: -r["distortion"])[:6]:
            log(f"  {row['street']:>7}: abstract {row['ratio_abstract']}x -> real "
                f"{row['ratio_real']}x (distortion {row['distortion']}x)")

    arguments.output.write_text(json.dumps({
        "hands": arguments.hands,
        "stack_bb": arguments.stack_bb,
        "opponent": arguments.opponent,
        "distortion_tolerance": DISTORTION_TOLERANCE,
        "legacy_overbet_ratio": LEGACY_OVERBET_RATIO,
        "all_ins": len(records),
        "legacy_flagged": len(legacy),
        "distorted": len(distorted),
        "by_street": dict(by_street),
        "legacy_by_street": dict(legacy_by_street),
        "distorted_by_street": dict(distorted_by_street),
        "records": records,
    }, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    handle.close()


if __name__ == "__main__":
    main()
