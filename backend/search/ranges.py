"""Infer a player's hand range from the blueprint along the public history.

For every non-colliding hole combination, replay the hand's public actions in
the abstract game while holding that combination in the target player's seat;
the combination's weight is the product of the blueprint's probabilities for
the actions the player actually took. This is the standard "blueprint range"
used by unsafe re-solving: it assumes the player follows the blueprint.
"""

from __future__ import annotations

import random
from dataclasses import replace

from backend.poker import HeadsUpHoldem
from backend.vectorized_engine import card_id


def blueprint_range(
    agent,  # BlueprintAgent (untyped to avoid an import cycle)
    engine: HeadsUpHoldem,
    target_player: int,
    extra_blocked: tuple[int, ...] = (),
    until_street: int = 3,
) -> dict[tuple[int, int], float]:
    """Weight per hole combo of ``target_player`` given actions before ``until_street``.

    ``extra_blocked`` removes card ids known to be unavailable (e.g. the
    viewer's own hole when computing the opponent's range). A player's own
    range is computed with only the board blocked.
    """
    board_ids = [card_id(card) for card in engine.community]
    blocked = set(board_ids) | set(extra_blocked)
    target_seat = agent._abstract_seat(engine, target_player)

    weights: dict[tuple[int, int], float] = {}
    for first in range(52):
        if first in blocked:
            continue
        for second in range(first + 1, 52):
            if second in blocked:
                continue
            combo = (first, second)
            weight = _combo_weight(agent, engine, combo, target_seat, board_ids, blocked, until_street)
            if weight > 0.0:
                weights[combo] = weight

    total = sum(weights.values())
    if total <= 0.0:
        uniform = {combo: 1.0 for combo in weights} or _uniform_range(blocked)
        total = sum(uniform.values())
        return {combo: weight / total for combo, weight in uniform.items()}
    return {combo: weight / total for combo, weight in weights.items()}


def _uniform_range(blocked: set[int]) -> dict[tuple[int, int], float]:
    return {
        (first, second): 1.0
        for first in range(52)
        if first not in blocked
        for second in range(first + 1, 52)
        if second not in blocked
    }


def _combo_weight(
    agent,
    engine: HeadsUpHoldem,
    combo: tuple[int, int],
    target_seat: int,
    board_ids: list[int],
    blocked: set[int],
    until_street: int,
) -> float:
    # The non-target seat's cards are never consulted (infosets are only
    # queried at the target's decisions), so any non-colliding dummies work.
    dummy = tuple(card for card in range(52) if card not in blocked and card not in combo)[:2]
    holes = (combo, dummy) if target_seat == 0 else (dummy, combo)
    state = replace(agent.game.initial_state(), hole=holes)
    rng = random.Random(1729)
    weight = 1.0
    try:
        for event in engine.public_actions:
            if event["action"] == "blind":
                continue
            if int(event.get("street", 0)) >= until_street:
                break
            state = agent._inject_board(state, board_ids)
            if state.is_terminal() or state.is_chance():
                break
            abstract_action = agent._translate_event(state, engine, event, rng)
            if state.current_player() == target_seat:
                actions = list(state.legal_actions())
                probabilities = agent.table.average_strategy(state.infoset_key(), actions)
                weight *= float(probabilities[actions.index(abstract_action)])
                if weight <= 1e-9:
                    return 0.0
            state = state.child(abstract_action)
    except Exception:
        return 1.0  # translation hiccup: keep the combo at neutral weight
    return weight
