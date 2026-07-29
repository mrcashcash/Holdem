"""Overbet audit: how often does the serving agent bet absurdly large?

Motivated by a live hand reported 2026-07-28: on T(c)J(s)K(h)A(c) with K(d)5(s)
the agent moved in for 3,980 chips into a 166-chip pot -- **24x the pot**. No
legitimate strategy contains that action: the widest own-bet menu the solver has
is 1.4x pot, so any raise larger than about 2x pot (1.4x pot plus a call) is an
artifact rather than a choice.

Two candidate mechanisms, and this tool separates them:

1. the frozen blueprint carries junk ALL-IN mass on postflop nodes it rarely
   visited (measured: 14.8% mean all-in probability over 2,000 turn nodes, and
   18.6% of those nodes above 20%); or
2. the exact-card resolver produces it.

Direct solves say the resolver is innocent at the reported spot -- it puts 0.16%
on all-in over the whole range and checks 94.2% with that exact hand -- so this
harness measures the RATE end-to-end through the real serving path, with
resolving on versus off, instead of arguing from one hand.

Definition used here: `ratio = amount / pot_before`, where `amount` is the total
the agent raises TO and `pot_before` is the pot as the agent saw it. An overbet
is `ratio >= OVERBET_RATIO`. Because a legal all-in is often a perfectly good
action in a small pot when stacks are short, the ratio is only counted when the
agent still had a real choice (a non-all-in raise was legal and affordable).

Every decision is appended to a JSONL trace and a summary JSON is written next
to it, so a run is auditable after the fact rather than only summarised.

**Conclusion reached with this tool (2026-07-29).** The blueprint made 8 overbets
in 640 decisions vs a min-raiser (worst 15.4x pot) and ZERO vs a calling station or
in self-play, because only a min-raise war exhausts the tree's 3-raise cap. Four of
the eight were preflop. `_all_in_size` repairs the translation and cuts the worst
case to 4.6x -- and then measured -268.82 bb/100 (200bb) / -124.00 (100bb) against
that same opponent, because a station calls any jam and so the huge shove is
correct exploitation. The guard therefore ships OFF. Re-run with `--resolver on` to
see the resolver's own sizing, which is sane by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.eval.null_agents import ScriptedAgent
from backend.poker import HeadsUpHoldem

#: A raise to this multiple of the pot or more cannot come from a 1.4x-max menu.
OVERBET_RATIO = 3.0
#: Ratio above which the bet is merely large rather than impossible; reported
#: separately so the two are never conflated.
LARGE_RATIO = 2.0
STREET_NAMES = ("preflop", "flop", "turn", "river")


def _decided_by(agent) -> str:
    """Which policy actually chose the last action."""
    search = getattr(agent, "last_continual_search", None)
    if isinstance(search, dict):
        status = str(search.get("status", ""))
        if status == "resolved":
            return "exact-resolver"
        if status:
            return f"blueprint ({status})"
    return "blueprint"


def _resolver_allin_mass(agent) -> float | None:
    """All-in probability in the acting mix, when the resolver decided."""
    search = getattr(agent, "last_continual_search", None)
    if not isinstance(search, dict) or search.get("status") != "resolved":
        return None
    for row in search.get("acting_mix") or ():
        if str(row.get("a")) == "all-in":
            return float(row.get("p", 0.0))
    return 0.0


def audit(
    hands: int,
    stack_bb: float,
    resolver: bool,
    opponent: str,
    trace_path: Path,
    seed: int = 20260728,
) -> dict:
    from backend.agents.serving import load_serving_agent

    agent = load_serving_agent()
    for target in [agent, *getattr(agent, "agents", {}).values()]:
        if hasattr(target, "continual_search"):
            target.continual_search = resolver
    # `self` plays the serving agent against itself, which is the only way to
    # reach nodes that need BOTH seats to bet -- a calling station never creates
    # a raised-and-reraised postflop pot, so it cannot probe the blueprint's
    # rarely-visited nodes where the junk all-in mass lives.
    villain = agent if opponent == "self" else ScriptedAgent(opponent)

    big_blind = 20
    stack = int(round(stack_bb * big_blind))
    engine = HeadsUpHoldem(
        initial_stack=stack,
        small_blind=big_blind // 2,
        big_blind=big_blind,
        rng=random.Random(seed),
    )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    decisions = 0
    aggressive = 0
    overbets: list[dict] = []
    large = 0
    by_street = {name: {"decisions": 0, "overbets": 0} for name in STREET_NAMES}
    resolver_decisions = 0
    rescales = 0
    started = time.monotonic()

    with trace_path.open("w", encoding="utf-8") as trace:
        trace.write(json.dumps({
            "event": "config", "hands": hands, "stack_bb": stack_bb,
            "resolver": resolver, "opponent": opponent, "seed": seed,
            "agent": type(agent).__name__,
            "overbet_ratio": OVERBET_RATIO,
        }) + "\n")

        for hand in range(hands):
            engine.stacks = [stack, stack]
            engine.new_hand()
            while not engine.hand_complete:
                seat = engine.current_player
                if seat != 0:
                    choice = villain.select(engine, seat)
                    villain.execute(engine, seat, choice)
                    continue

                street = int(engine.street)
                pot_before = int(engine.pot)
                legal = engine.legal_actions(0)
                # Did the agent have a smaller raise available, or was all-in
                # its only aggressive option? Only the former can be a mistake.
                raise_min = int(legal.get("raise_min", 0))
                raise_max = int(legal.get("raise_max", 0))
                had_choice = bool(legal.get("raise")) and raise_max > raise_min

                for target in [agent, *getattr(agent, "agents", {}).values()]:
                    if hasattr(target, "last_all_in_rescale"):
                        target.last_all_in_rescale = None
                choice = agent.select(engine, 0)
                agent.execute(engine, 0, choice)
                rescale = next(
                    (
                        getattr(t, "last_all_in_rescale", None)
                        for t in [agent, *getattr(agent, "agents", {}).values()]
                        if getattr(t, "last_all_in_rescale", None)
                    ),
                    None,
                )
                event = dict(engine.public_actions[-1])
                action = str(event.get("action", ""))
                amount = int(event.get("amount", 0) or 0)
                ratio = (amount / pot_before) if (pot_before and amount) else 0.0
                decided_by = _decided_by(agent)

                decisions += 1
                by_street[STREET_NAMES[street]]["decisions"] += 1
                if decided_by == "exact-resolver":
                    resolver_decisions += 1
                if rescale:
                    rescales += 1
                row = {
                    "event": "decision", "hand": hand, "street": STREET_NAMES[street],
                    "pot_before": pot_before, "action": action, "amount": amount,
                    "ratio": round(ratio, 3), "had_smaller_raise": had_choice,
                    "decided_by": decided_by,
                    "resolver_allin_p": _resolver_allin_mass(agent),
                    "board": len(engine.community),
                    "all_in_rescaled": rescale,
                }
                if action in {"raise", "all_in"}:
                    aggressive += 1
                    if had_choice and ratio >= LARGE_RATIO:
                        large += 1
                    if had_choice and ratio >= OVERBET_RATIO:
                        row["OVERBET"] = True
                        by_street[STREET_NAMES[street]]["overbets"] += 1
                        overbets.append(row)
                trace.write(json.dumps(row) + "\n")
                trace.flush()

            if hasattr(agent, "observe_completed_hand"):
                agent.observe_completed_hand(engine, 0)
            if (hand + 1) % 5 == 0:
                print(
                    f"  hand {hand + 1}/{hands}  decisions {decisions}  "
                    f"overbets {len(overbets)}  ({time.monotonic() - started:.0f}s)",
                    flush=True,
                )

        summary = {
            "event": "summary",
            "hands": hands, "stack_bb": stack_bb, "resolver": resolver,
            "opponent": opponent,
            "decisions": decisions, "aggressive": aggressive,
            "resolver_decisions": resolver_decisions,
            "all_in_rescaled": rescales,
            "overbets": len(overbets),
            "overbet_rate_per_decision": round(len(overbets) / max(decisions, 1), 4),
            "overbet_rate_per_aggressive": round(len(overbets) / max(aggressive, 1), 4),
            "large_bets_over_2x": large,
            "worst_ratio": round(max((row["ratio"] for row in overbets), default=0.0), 2),
            "by_street": by_street,
            "elapsed_s": round(time.monotonic() - started, 1),
        }
        trace.write(json.dumps(summary) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--resolver", choices=("on", "off"), default="off")
    parser.add_argument("--opponent", default="always-call", help="always-call/always-min-raise/always-all-in/always-fold, or self")
    parser.add_argument("--out", default="artifacts/overbet_audit")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    out = Path(args.out)
    tag = f"{int(args.stack_bb)}bb_resolver-{args.resolver}_vs-{args.opponent}"
    trace_path = out / f"trace_{tag}.jsonl"
    summary = audit(
        hands=args.hands, stack_bb=args.stack_bb, resolver=args.resolver == "on",
        opponent=args.opponent, trace_path=trace_path, seed=args.seed,
    )
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"stack {args.stack_bb:g}bb, resolver {args.resolver}, vs {args.opponent}")
    print(f"  decisions            {summary['decisions']}  ({summary['resolver_decisions']} by resolver)")
    print(f"  aggressive actions   {summary['aggressive']}")
    print(f"  OVERBETS (>={OVERBET_RATIO:g}x pot, smaller raise was legal): {summary['overbets']}"
          f"  = {summary['overbet_rate_per_decision']:.2%} of decisions")
    print(f"  bets over {LARGE_RATIO:g}x pot     {summary['large_bets_over_2x']}")
    print(f"  all-in resized by guard {summary['all_in_rescaled']}")
    print(f"  worst ratio          {summary['worst_ratio']:g}x pot")
    for name, stats in summary["by_street"].items():
        if stats["decisions"]:
            print(f"    {name:<8} {stats['overbets']:>4}/{stats['decisions']:<5} overbets")
    print(f"  trace -> {trace_path}")


if __name__ == "__main__":
    main()
