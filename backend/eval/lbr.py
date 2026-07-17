"""Local best response (LBR) probe: a cheap lower bound on exploitability.

Following Lisý & Bowling (arXiv:1612.07547), the probe plays with full
knowledge of its own cards and a fixed action set {fold, call, pot bet,
all-in}. At each decision it estimates the EV of each option using (a) its
Monte-Carlo equity against a uniform opponent range and (b) the blueprint's
*actual* fold probability at the responding infoset, queried from the
strategy table through the same mirror translation the serving agent uses.

The probe's winrate against the blueprint is a LOWER bound on real
exploitability: a strong blueprint should hold it near or below zero.
This is a probe, not formal exploitability.

CLI:  python -m backend.eval.lbr --hands 500
"""

from __future__ import annotations

import argparse
import math
import random
import statistics

from backend.abstraction.actions import FOLD
from backend.agents.blueprint_agent import BlueprintAgent
from backend.abstraction.equity import equity_histogram, river_equity
from backend.poker import HeadsUpHoldem
from backend.vectorized_engine import card_id

NEURAL_FOLD, NEURAL_CHECK_CALL, NEURAL_RAISE, NEURAL_ALL_IN = 0, 1, 2, 3


def _win_probability(engine: HeadsUpHoldem, player: int, rng: random.Random) -> float:
    hole = tuple(card_id(card) for card in engine.hole_cards[player])
    board = tuple(card_id(card) for card in engine.community)
    if len(board) == 5:
        return river_equity(hole, board)
    if len(board) in (3, 4):
        histogram = equity_histogram(hole, board, bins=8, scenarios=24, opponents_per_scenario=16, seed=rng.getrandbits(31))
        centers = [(slot + 0.5) / 8 for slot in range(8)]
        return float(sum(weight * center for weight, center in zip(histogram, centers)))
    # Preflop: cheap rank heuristic is enough for a probe.
    ranks = sorted((card // 4 for card in hole), reverse=True)
    suited = hole[0] % 4 == hole[1] % 4
    pair = ranks[0] == ranks[1]
    return min(0.85, 0.30 + ranks[0] * 0.02 + ranks[1] * 0.012 + 0.12 * pair + 0.03 * suited)


def _blueprint_fold_probability(agent: BlueprintAgent, engine: HeadsUpHoldem, probe_seat: int, raise_to: int) -> float:
    """Ask the blueprint how often it folds to this raise at its next infoset."""
    try:
        hypothetical = engine  # translate on the CURRENT state plus a synthetic raise event
        state = agent._mirror_state(hypothetical, 1 - probe_seat)
        if state is None or state.is_terminal() or state.is_chance():
            return 0.35
        event = {
            "action": "raise",
            "amount": raise_to,
            "pot_before": engine.pot,
            "to_call_before": engine.to_call(probe_seat),
            "current_bet_before": max(engine.round_bets),
            "action_index": 2,
        }
        translated = agent._translate_event(state, engine, event, random.Random(1))
        responded = state.child(translated)
        if responded.is_terminal() or responded.is_chance():
            return 0.0
        actions = list(responded.legal_actions())
        if FOLD not in actions:
            return 0.0
        probabilities = agent.table.average_strategy(responded.infoset_key(), actions)
        return float(probabilities[actions.index(FOLD)])
    except Exception:
        return 0.35


def _probe_action(agent: BlueprintAgent, engine: HeadsUpHoldem, seat: int, rng: random.Random) -> tuple[int, int | None]:
    legal = engine.legal_actions(seat)
    win_probability = _win_probability(engine, seat, rng)
    pot = engine.pot
    to_call = int(legal.get("to_call", 0))

    evs: list[tuple[float, int, int | None]] = []
    evs.append((win_probability * (pot + to_call) - to_call, NEURAL_CHECK_CALL, None))
    if to_call > 0:
        evs.append((0.0, NEURAL_FOLD, None))
    if legal.get("raise"):
        minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
        target = min(maximum, max(minimum, max(engine.round_bets) + pot))  # ~pot-size raise
        fold_probability = _blueprint_fold_probability(agent, engine, seat, target)
        invested = target - int(legal["player_bet"])
        ev = fold_probability * pot + (1 - fold_probability) * (win_probability * (pot + 2 * invested) - invested)
        evs.append((ev, NEURAL_RAISE, target))
    if legal.get("all_in"):
        maximum = int(legal["raise_max"])
        fold_probability = _blueprint_fold_probability(agent, engine, seat, maximum)
        invested = engine.stacks[seat]
        ev = fold_probability * pot + (1 - fold_probability) * (win_probability * (pot + 2 * invested) - invested)
        evs.append((ev, NEURAL_ALL_IN, None))

    evs.sort(key=lambda item: item[0], reverse=True)
    return evs[0][1], evs[0][2]


def local_best_response_probe(agent: BlueprintAgent, hands: int = 500, seed: int = 0) -> dict:
    """Probe winrate in bb/100 (positive = blueprint is exploitable by LBR)."""
    samples: list[float] = []
    for hand in range(hands):
        rng = random.Random(seed * 99_991 + hand)
        engine = HeadsUpHoldem(rng=random.Random(seed * 77_003 + hand))
        probe_seat = hand % 2
        before = list(engine.stacks)
        safety = 0
        while not engine.hand_complete and safety < 200:
            player = engine.current_player
            if player == probe_seat:
                action, target = _probe_action(agent, engine, player, rng)
                if action == NEURAL_RAISE:
                    engine.act(player, "raise", target)
                elif action == NEURAL_ALL_IN:
                    engine.act(player, "all_in")
                elif action == NEURAL_FOLD:
                    engine.act(player, "fold")
                else:
                    engine.act(player, "check" if engine.legal_actions(player).get("check") else "call")
            else:
                choice = agent.select(engine, player)
                agent.execute(engine, player, choice)
            safety += 1
        samples.append((engine.stacks[probe_seat] - before[probe_seat]) / engine.big_blind)
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(len(samples))
    return {
        "lbr_bb_per_100": round(mean * 100, 2),
        "ci_low_bb_per_100": round((mean - margin) * 100, 2),
        "ci_high_bb_per_100": round((mean + margin) * 100, 2),
        "hands": hands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LBR exploitability probe against the blueprint")
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    agent = BlueprintAgent.try_load()
    if agent is None:
        raise SystemExit("no blueprint artifacts found — train with `python -m backend.solver.blueprint` first")
    report = local_best_response_probe(agent, hands=arguments.hands, seed=arguments.seed)
    print(
        f"LBR probe: {report['lbr_bb_per_100']:+.2f} bb/100 "
        f"[{report['ci_low_bb_per_100']:+.2f}, {report['ci_high_bb_per_100']:+.2f}] over {report['hands']} hands "
        f"(positive = exploitable)"
    )


if __name__ == "__main__":
    main()
