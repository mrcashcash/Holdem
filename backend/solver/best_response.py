"""Exact best response and exploitability for small games.

Used to validate the solver against ground truth (Kuhn, Leduc). Enumerates
the full tree, so it is only suitable for games with at most ~10^5 states —
full hold'em exploitability is measured by the LBR probe in ``backend/eval``.

``best_response_value`` computes the value of the exact best response to the
solver's average strategy. For a Nash strategy the sum of both players' best
response values equals zero; ``exploitability`` returns half that sum (the
average amount an optimal adversary wins per game).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Sequence

import numpy as np

from backend.solver.game import Game, State

PolicyFn = Callable[[Hashable, Sequence[int]], np.ndarray]


def best_response_value(game: Game, policy: PolicyFn, br_player: int) -> float:
    """Expected value for ``br_player`` when it best-responds to ``policy``."""
    # Pass 1: group the best responder's decision states by infoset, with the
    # reach probability contributed by chance and the opponent only (the best
    # responder plays every action, so its own reach is 1).
    infoset_states: dict[Hashable, list[tuple[State, float]]] = defaultdict(list)

    def collect(state: State, reach: float) -> None:
        if state.is_terminal() or reach <= 0.0:
            return
        if state.is_chance():
            for successor, probability in state.chance_outcomes():
                collect(successor, reach * probability)
            return
        actions = state.legal_actions()
        if state.current_player() == br_player:
            infoset_states[state.infoset_key()].append((state, reach))
            for action in actions:
                collect(state.child(action), reach)
        else:
            strategy = policy(state.infoset_key(), actions)
            for position, action in enumerate(actions):
                collect(state.child(action), reach * float(strategy[position]))

    collect(game.initial_state(), 1.0)

    # Pass 2: choose the best action per infoset by counterfactual value.
    # Perfect recall guarantees the recursion terminates; memoise per infoset.
    chosen_actions: dict[Hashable, int] = {}

    def state_value(state: State) -> float:
        if state.is_terminal():
            return state.utility(br_player)
        if state.is_chance():
            return sum(probability * state_value(successor) for successor, probability in state.chance_outcomes())
        actions = state.legal_actions()
        if state.current_player() == br_player:
            return state_value(state.child(infoset_action(state.infoset_key())))
        strategy = policy(state.infoset_key(), actions)
        return float(
            sum(strategy[position] * state_value(state.child(action)) for position, action in enumerate(actions))
        )

    def infoset_action(key: Hashable) -> int:
        cached = chosen_actions.get(key)
        if cached is not None:
            return cached
        states = infoset_states[key]
        actions = states[0][0].legal_actions()
        best_action, best_value = actions[0], -np.inf
        for action in actions:
            value = sum(reach * state_value(state.child(action)) for state, reach in states)
            if value > best_value:
                best_action, best_value = action, value
        chosen_actions[key] = best_action
        return best_action

    return state_value(game.initial_state())


def exploitability(game: Game, policy: PolicyFn) -> float:
    """Average per-game value an optimal adversary extracts from ``policy``."""
    return (best_response_value(game, policy, 0) + best_response_value(game, policy, 1)) / 2.0
