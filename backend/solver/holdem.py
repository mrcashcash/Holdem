"""Abstracted heads-up no-limit hold'em for the blueprint solver.

A lean immutable state machine over compact card ids (0..51), scored with the
engine's hand evaluator. Player 0 is the button/small blind (first to act
preflop, last to act postflop). All chip amounts are in big blinds. Card
buckets come from ``CardAbstraction``; the betting menu from
``ActionAbstraction``. Infoset keys are (street, own bucket, public betting
history), giving the solver perfect recall of public actions.

Chance nodes are sampled (``chance_outcomes`` is intentionally unsupported:
exact best response is intractable at this scale — use the LBR probe).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Hashable, Sequence

from backend.abstraction.actions import ALL_IN, CHECK_CALL, FOLD, ActionAbstraction
from backend.abstraction.buckets import CardAbstraction
from backend.poker import SUITS, best_score

_STREET_BOARD = (0, 3, 4, 5)


def card_tuple(card: int) -> tuple[int, str]:
    return card // 4 + 2, SUITS[card % 4]


def pack_infoset_key(street: int, bucket: int, history: tuple[int, ...]) -> bytes:
    """Compact byte encoding of (street, bucket, public history).

    Bytes hash fast and cost a fraction of nested tuples in the strategy
    table (history entries are street*16+action < 64; buckets fit 16 bits).
    """
    return bytes((street, bucket & 0xFF, bucket >> 8, *history))


@dataclass(frozen=True)
class HoldemState:
    game: "AbstractHoldem"
    hole: tuple[tuple[int, int], tuple[int, int]] | None = None
    board: tuple[int, ...] = ()
    street: int = 0
    stacks: tuple[float, float] = (0.0, 0.0)
    committed: tuple[float, float] = (0.0, 0.0)
    street_commit: tuple[float, float] = (0.0, 0.0)
    acted: tuple[bool, bool] = (False, False)
    raises_this_street: int = 0
    last_raise_increment: float = 1.0
    to_act: int = 0
    folded: int | None = None
    in_runout: bool = False
    showdown: bool = False
    history: tuple[int, ...] = ()

    # -- chance ------------------------------------------------------------

    def is_chance(self) -> bool:
        if self.folded is not None or self.showdown:
            return False
        return self.hole is None or len(self.board) < _STREET_BOARD[self.street] or self.in_runout

    def sample_chance(self, rng: random.Random) -> "HoldemState":
        used = set(self.board)
        if self.hole is not None:
            used.update(self.hole[0])
            used.update(self.hole[1])
        if self.hole is None:
            cards = rng.sample([card for card in range(52) if card not in used], 4)
            return replace(self, hole=((cards[0], cards[1]), (cards[2], cards[3])))
        remaining = [card for card in range(52) if card not in used]
        if self.in_runout:
            drawn = rng.sample(remaining, 5 - len(self.board))
            return replace(self, board=self.board + tuple(drawn), street=3, in_runout=False, showdown=True)
        needed = _STREET_BOARD[self.street] - len(self.board)
        drawn = rng.sample(remaining, needed)
        return replace(self, board=self.board + tuple(drawn))

    def chance_outcomes(self) -> Sequence[tuple["HoldemState", float]]:
        raise NotImplementedError("full hold'em chance nodes are sampled, never enumerated")

    # -- terminal ------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.folded is not None or self.showdown

    def utility(self, player: int) -> float:
        if self.folded is not None:
            winnings = self.committed[self.folded]
            return winnings if player != self.folded else -winnings
        matched = min(self.committed)
        scores = [
            best_score([card_tuple(card) for card in self.hole[side] + self.board])
            for side in (0, 1)
        ]
        if scores[0] == scores[1]:
            return 0.0
        winner = 0 if scores[0] > scores[1] else 1
        return matched if player == winner else -matched

    # -- decisions -----------------------------------------------------------

    def current_player(self) -> int:
        return self.to_act

    def legal_actions(self) -> Sequence[int]:
        to_call = self._to_call(self.to_act)
        return tuple(
            self.game.actions.menu(
                street=self.street,
                pot=self.committed[0] + self.committed[1],
                to_call=to_call,
                stack_behind=self.stacks[self.to_act],
                raises_this_street=self.raises_this_street,
            )
        )

    def infoset_key(self) -> Hashable:
        bucket = self.game.abstraction.bucket(self.hole[self.to_act], self.board)
        return pack_infoset_key(self.street, bucket, self.history)

    def child(self, action: int) -> "HoldemState":
        actor = self.to_act
        to_call = self._to_call(actor)
        stacks = list(self.stacks)
        committed = list(self.committed)
        street_commit = list(self.street_commit)
        raises = self.raises_this_street
        last_increment = self.last_raise_increment
        history = self.history + (self.street * 16 + action,)

        if action == FOLD:
            return replace(self, folded=actor, history=history)

        if action == CHECK_CALL:
            payment = min(to_call, stacks[actor])
        elif action == ALL_IN:
            payment = stacks[actor]
        else:
            raise_by = self.game.actions.raise_amount(
                action, self.street, committed[0] + committed[1], to_call
            )
            raise_by = max(raise_by, last_increment, 1.0)
            payment = min(to_call + raise_by, stacks[actor])

        stacks[actor] -= payment
        committed[actor] += payment
        street_commit[actor] += payment
        increment = street_commit[actor] - max(self.street_commit)
        if increment > 0:
            raises += 1
            last_increment = max(increment, 1.0)

        acted = list(self.acted)
        acted[actor] = True
        state = replace(
            self,
            stacks=tuple(stacks),
            committed=tuple(committed),
            street_commit=tuple(street_commit),
            acted=tuple(acted),
            raises_this_street=raises,
            last_raise_increment=last_increment,
            history=history,
        )

        opponent = 1 - actor
        opponent_owes = state.street_commit[actor] - state.street_commit[opponent]
        if opponent_owes > 0:
            if state.stacks[opponent] > 0:
                return replace(state, to_act=opponent)
            return state._close_street()  # opponent already all-in; excess is returned at showdown
        if not state.acted[opponent] and state.stacks[opponent] > 0:
            return replace(state, to_act=opponent)
        return state._close_street()

    # -- internals -----------------------------------------------------------

    def _to_call(self, player: int) -> float:
        return max(self.street_commit) - self.street_commit[player]

    def _close_street(self) -> "HoldemState":
        if self.street == 3:
            return replace(self, showdown=True)
        if min(self.stacks) <= 0:
            return replace(self, in_runout=True)
        return replace(
            self,
            street=self.street + 1,
            street_commit=(0.0, 0.0),
            acted=(False, False),
            raises_this_street=0,
            last_raise_increment=1.0,
            to_act=1,  # big blind acts first postflop heads-up
        )


class AbstractHoldem:
    """Two-player abstracted NLHE ``Game`` for the MCCFR solver."""

    def __init__(
        self,
        abstraction: CardAbstraction,
        actions: ActionAbstraction | None = None,
        stack_bb: float = 50.0,
    ) -> None:
        self.abstraction = abstraction
        self.actions = actions or ActionAbstraction()
        self.stack_bb = stack_bb

    def initial_state(self) -> HoldemState:
        return HoldemState(
            game=self,
            stacks=(self.stack_bb - 0.5, self.stack_bb - 1.0),
            committed=(0.5, 1.0),
            street_commit=(0.5, 1.0),
            to_act=0,
        )

    def num_actions(self) -> int:
        return self.actions.num_actions()
