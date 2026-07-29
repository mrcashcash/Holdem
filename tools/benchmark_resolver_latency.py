"""One-shot exact-resolver latency/equivalence probe for the serving profile.

Runs one deterministic check/check line at a requested street and prints only
the resolver diagnostics. Use separate processes for graph on/off comparisons
so CUDA allocator state and agent sessions cannot leak between arms.
"""

from __future__ import annotations

import argparse
import json
import os
import random


def _configure(arguments) -> None:
    os.environ["HOLDEM_RESOLVE_STREETS"] = "flop,turn,river"
    os.environ["HOLDEM_FLOP_SIZES"] = "0.33,0.75,1.4"
    os.environ["HOLDEM_FLOP_CAP"] = "2"
    os.environ["HOLDEM_TURN_SIZES"] = "0.33,0.75,1.4"
    os.environ["HOLDEM_TURN_CAP"] = "2"
    os.environ["HOLDEM_CONTINUAL_ITERS"] = str(arguments.iterations)
    os.environ["HOLDEM_CONTINUAL_MIN_ITERS"] = str(arguments.iterations)
    os.environ["HOLDEM_CONTINUAL_BUDGET_MS"] = "120000"
    os.environ["HOLDEM_SAFETY_PRICE_GRAPH"] = (
        "1" if arguments.safety_graph == "on" else "0"
    )
    os.environ["HOLDEM_RESOLVER_PREFETCH"] = (
        "1" if arguments.prefetch == "on" else "0"
    )
    os.environ["HOLDEM_RESOLVER_WARMUP"] = "0"


def _card(text: str):
    ranks = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
    }
    suits = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
    return ranks[text[0].upper()], suits[text[1].lower()]


def _spot(street: str):
    from backend.poker import HeadsUpHoldem, new_deck

    boards = {
        "flop": ["Js", "Jc", "5h"],
        "turn": ["Js", "Jc", "5h", "Ah"],
        "river": ["Js", "Jc", "5h", "Ah", "5c"],
    }
    actions = [
        (0, "raise", 50),
        (1, "call", None),
        (1, "check", None),
    ]
    if street in {"turn", "river"}:
        actions.extend([(0, "check", None), (1, "check", None)])
    if street == "river":
        actions.extend([(0, "check", None), (1, "check", None)])

    hero = [_card("7s"), _card("9c")]
    board = [_card(value) for value in boards[street]]
    known = set(hero + board)
    game = HeadsUpHoldem(initial_stack=4_000, rng=random.Random(17))
    game.hand_number = 0
    game.button_offset = 0
    game.new_hand()
    available = [card for card in new_deck() if card not in known]
    game.hole_cards = [hero, available[:2]]
    future = [card for card in available[2:] if card not in known]
    game.deck = future + list(reversed(board))
    for player, action, amount in actions:
        game.act(player, action, amount)
    if game.current_player != 0 or game.active_street != street:
        raise RuntimeError(
            f"probe construction ended at {game.active_street}, "
            f"player {game.current_player}"
        )
    return game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--street", choices=("flop", "turn", "river"), required=True)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--safety-graph", choices=("on", "off"), default="on")
    parser.add_argument("--prefetch", choices=("on", "off"), default="on")
    arguments = parser.parse_args()
    _configure(arguments)

    from backend.agents.serving import load_serving_agent
    from backend.agents.gpu_blueprint_agent import _COMBO_INDEX
    from backend.search.continual import resolve_decision
    from backend.vectorized_engine import card_id

    agent = load_serving_agent()
    game = _spot(arguments.street)
    routed = agent._route(game, 0) if hasattr(agent, "_route") else agent
    uid = routed._search_uid_for(game)
    key = (uid, int(game.hand_number))
    # Isolate the requested street's fresh solve. Live continual sessions carry
    # ranges from earlier streets, but that changes belief seeding rather than
    # the solve kernel being timed here.
    entry_street = int(game.street)
    solution = resolve_decision(
        routed,
        game,
        0,
        key=key,
        sessions=routed._continual_sessions,
        iterations=arguments.iterations,
        budget_ms=120_000,
        entry_street=entry_street,
    )
    hole = tuple(sorted(card_id(card) for card in game.hole_cards[0]))
    combo_index = _COMBO_INDEX[hole]
    probabilities = solution.strategy[solution.node, combo_index]
    result = dict(solution.diagnostics)
    result["root_probabilities"] = [
        float(value) for value in probabilities.tolist()
    ]
    result["probe"] = {
        "street": arguments.street,
        "iterations": arguments.iterations,
        "safety_graph": arguments.safety_graph,
        "prefetch": arguments.prefetch,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
