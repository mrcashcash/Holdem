"""Exact-cards river subgame re-solving.

The river subgame is small enough to solve without card abstraction: both
players hold explicit combos drawn from their blueprint ranges, the board is
fixed, and betting uses the same abstract menu as the blueprint. The solver
is the validated Linear MCCFR core; showdown scores are precomputed once per
board so terminal evaluation is a table lookup.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace as dc_replace
from typing import Hashable, Sequence

from backend.abstraction.actions import ALL_IN, CHECK_CALL, FOLD, ActionAbstraction
from backend.poker import best_score
from backend.solver.holdem import card_tuple
from backend.solver.mccfr import LinearMCCFR


@dataclass(frozen=True)
class RiverState:
    street = 3  # class constant: shared translation code reads state.street

    game: "RiverSubgame"
    combos: tuple[tuple[int, int], tuple[int, int]] | None = None
    street_commit: tuple[float, float] = (0.0, 0.0)
    stacks: tuple[float, float] = (0.0, 0.0)
    acted: tuple[bool, bool] = (False, False)
    raises: int = 0
    last_raise_increment: float = 1.0
    to_act: int = 1  # big blind acts first postflop
    folded: int | None = None
    showdown: bool = False
    history: tuple[int, ...] = ()

    def is_chance(self) -> bool:
        return self.combos is None

    def sample_chance(self, rng: random.Random) -> "RiverState":
        game = self.game
        while True:
            hero = game.sample_combo(0, rng)
            villain = game.sample_combo(1, rng)
            if not set(hero) & set(villain):
                return dc_replace(self, combos=(hero, villain))

    def chance_outcomes(self) -> Sequence[tuple["RiverState", float]]:
        raise NotImplementedError("river deals are sampled from the tracked ranges")

    def is_terminal(self) -> bool:
        return self.folded is not None or self.showdown

    def utility(self, player: int) -> float:
        game = self.game
        if self.folded is not None:
            # The folder loses the pre-river pot share plus their river commits.
            loser = self.folded
            amount = game.pot_start / 2.0 + self.street_commit[loser]
            return -amount if player == loser else amount
        matched = game.pot_start / 2.0 + min(self.street_commit)
        first = game.score(self.combos[0])
        second = game.score(self.combos[1])
        if first == second:
            return 0.0
        winner = 0 if first > second else 1
        return matched if player == winner else -matched

    def current_player(self) -> int:
        return self.to_act

    def legal_actions(self) -> Sequence[int]:
        to_call = max(self.street_commit) - self.street_commit[self.to_act]
        return tuple(
            self.game.actions.menu(
                street=3,
                pot=self.game.pot_start + self.street_commit[0] + self.street_commit[1],
                to_call=to_call,
                stack_behind=self.stacks[self.to_act],
                raises_this_street=self.raises,
            )
        )

    def infoset_key(self) -> Hashable:
        combo = self.combos[self.to_act]
        return bytes((combo[0], combo[1], *self.history))

    def child(self, action: int) -> "RiverState":
        actor = self.to_act
        to_call = max(self.street_commit) - self.street_commit[actor]
        history = self.history + (action,)
        if action == FOLD:
            return dc_replace(self, folded=actor, history=history)

        stacks = list(self.stacks)
        commit = list(self.street_commit)
        raises = self.raises
        last_increment = self.last_raise_increment
        if action == CHECK_CALL:
            payment = min(to_call, stacks[actor])
        elif action == ALL_IN:
            payment = stacks[actor]
        else:
            pot = self.game.pot_start + commit[0] + commit[1]
            raise_by = self.game.actions.raise_amount(action, 3, pot, to_call)
            raise_by = max(raise_by, last_increment, 1.0)
            payment = min(to_call + raise_by, stacks[actor])
        stacks[actor] -= payment
        commit[actor] += payment
        increment = commit[actor] - max(self.street_commit)
        if increment > 0:
            raises += 1
            last_increment = max(increment, 1.0)

        acted = list(self.acted)
        acted[actor] = True
        state = dc_replace(
            self,
            stacks=tuple(stacks),
            street_commit=tuple(commit),
            acted=tuple(acted),
            raises=raises,
            last_raise_increment=last_increment,
            history=history,
        )
        opponent = 1 - actor
        owes = state.street_commit[actor] - state.street_commit[opponent]
        if owes > 0 and state.stacks[opponent] > 0:
            return dc_replace(state, to_act=opponent)
        if owes <= 0 and not state.acted[opponent] and state.stacks[opponent] > 0:
            return dc_replace(state, to_act=opponent)
        return dc_replace(state, showdown=True)


class RiverSubgame:
    """Two-player river subgame over explicit combos and tracked ranges.

    ``pot_start`` is the matched pot entering the river (both halves), and
    ``stacks`` are the chips behind at the start of river betting, all in bb.
    Seat convention matches the abstract game: 0 = button, 1 = big blind.
    """

    def __init__(
        self,
        board: tuple[int, ...],
        pot_start: float,
        stacks: tuple[float, float],
        ranges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]],
        actions: ActionAbstraction | None = None,
    ) -> None:
        if len(board) != 5:
            raise ValueError("river subgames need a complete board")
        self.board = board
        self.pot_start = pot_start
        self.stacks = stacks
        self.actions = actions or ActionAbstraction()
        self._score_cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self._range_samplers = tuple(self._build_sampler(side_range) for side_range in ranges)

    @staticmethod
    def _build_sampler(weights: dict[tuple[int, int], float]):
        combos = list(weights)
        cumulative: list[float] = []
        total = 0.0
        for combo in combos:
            total += weights[combo]
            cumulative.append(total)
        return combos, cumulative, total

    def sample_combo(self, seat: int, rng: random.Random) -> tuple[int, int]:
        combos, cumulative, total = self._range_samplers[seat]
        pick = rng.random() * total
        low, high = 0, len(cumulative) - 1
        while low < high:
            middle = (low + high) // 2
            if cumulative[middle] < pick:
                low = middle + 1
            else:
                high = middle
        return combos[low]

    def score(self, combo: tuple[int, int]) -> tuple[int, ...]:
        cached = self._score_cache.get(combo)
        if cached is None:
            cached = best_score([card_tuple(card) for card in combo + self.board])
            self._score_cache[combo] = cached
        return cached

    def initial_state(self) -> RiverState:
        return RiverState(game=self, stacks=self.stacks)

    def num_actions(self) -> int:
        return self.actions.num_actions()


def solve_river(subgame: RiverSubgame, iterations: int = 400, seed: int = 0) -> LinearMCCFR:
    solver = LinearMCCFR(subgame, seed=seed)
    solver.run(iterations)
    return solver
