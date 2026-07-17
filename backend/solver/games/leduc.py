"""Leduc hold'em: the standard 6-card two-round ground-truth game.

Deck: two suits x three ranks {J=0, Q=1, K=2}. Each player antes 1 and gets
one private card; after the first betting round one public card is revealed
and a second round follows. Fixed bet sizes: 2 in round one, 4 in round two,
at most two raises per round. Pairing the public card wins; otherwise higher
rank wins. Nash game value for player 0 is about -0.0856 antes; converged
CFR exploitability approaches zero.

Actions: 0 = fold, 1 = check/call, 2 = bet/raise.
"""

from __future__ import annotations

import random
from typing import Hashable, Sequence

FOLD, CALL, RAISE = 0, 1, 2
_DECK = [(rank, suit) for rank in range(3) for suit in range(2)]
_MAX_RAISES = 2
_BET_SIZES = (2, 4)


class LeducState:
    __slots__ = ("cards", "board", "round_histories", "pot", "committed", "folded")

    def __init__(
        self,
        cards: tuple[int, int] | None = None,
        board: int | None = None,
        round_histories: tuple[str, str] = ("", ""),
        committed: tuple[int, int] = (1, 1),
        folded: int | None = None,
    ) -> None:
        self.cards = cards  # deck indices of each player's private card
        self.board = board  # deck index of the public card, once dealt
        self.round_histories = round_histories
        self.committed = committed
        self.folded = folded

    # -- helpers -------------------------------------------------------------

    def _round(self) -> int:
        return 0 if self.board is None else 1

    def _round_over(self, history: str) -> bool:
        if history.endswith("f") or history == "cc":
            return True
        # A call closes the round once a bet/raise has occurred ("rc", "crc",
        # "rrc", "crrc"); a lone opening check does not.
        return history.endswith("c") and "r" in history

    def _needs_board(self) -> bool:
        return self.board is None and self._round_over(self.round_histories[0]) and self.folded is None

    # -- protocol ------------------------------------------------------------

    def is_chance(self) -> bool:
        return self.cards is None or self._needs_board()

    def chance_outcomes(self) -> Sequence[tuple["LeducState", float]]:
        if self.cards is None:
            outcomes = []
            for first in range(len(_DECK)):
                for second in range(len(_DECK)):
                    if first != second:
                        outcomes.append((self._with(cards=(first, second)), 1.0 / (len(_DECK) * (len(_DECK) - 1))))
            return outcomes
        remaining = [index for index in range(len(_DECK)) if index not in self.cards]
        return [(self._with(board=index), 1.0 / len(remaining)) for index in remaining]

    def sample_chance(self, rng: random.Random) -> "LeducState":
        if self.cards is None:
            first, second = rng.sample(range(len(_DECK)), 2)
            return self._with(cards=(first, second))
        remaining = [index for index in range(len(_DECK)) if index not in self.cards]
        return self._with(board=rng.choice(remaining))

    def is_terminal(self) -> bool:
        if self.folded is not None:
            return True
        return self.board is not None and self._round_over(self.round_histories[1])

    def current_player(self) -> int:
        return len(self.round_histories[self._round()]) % 2

    def legal_actions(self) -> Sequence[int]:
        history = self.round_histories[self._round()]
        facing_bet = history.endswith("r")
        raises = history.count("r")
        actions = []
        if facing_bet:
            actions.append(FOLD)
        actions.append(CALL)
        if raises < _MAX_RAISES:
            actions.append(RAISE)
        return tuple(actions)

    def infoset_key(self) -> Hashable:
        player = self.current_player()
        private_rank = _DECK[self.cards[player]][0]
        board_rank = None if self.board is None else _DECK[self.board][0]
        return (private_rank, board_rank, self.round_histories)

    def child(self, action: int) -> "LeducState":
        round_index = self._round()
        history = self.round_histories[round_index]
        player = self.current_player()
        committed = list(self.committed)
        if action == FOLD:
            return self._with(round_history=(round_index, history + "f"), folded=player)
        if action == CALL:
            committed[player] = max(committed)
            return self._with(round_history=(round_index, history + "c"), committed=tuple(committed))
        committed[player] = max(committed) + _BET_SIZES[round_index]
        return self._with(round_history=(round_index, history + "r"), committed=tuple(committed))

    def utility(self, player: int) -> float:
        pot_share = min(self.committed)
        if self.folded is not None:
            winner = 1 - self.folded
        else:
            ranks = [_DECK[card][0] for card in self.cards]
            board_rank = _DECK[self.board][0]
            if ranks[0] == board_rank:
                winner = 0
            elif ranks[1] == board_rank:
                winner = 1
            elif ranks[0] != ranks[1]:
                winner = 0 if ranks[0] > ranks[1] else 1
            else:
                return 0.0
        return float(pot_share) if player == winner else -float(pot_share)

    # -- construction --------------------------------------------------------

    def _with(
        self,
        cards: tuple[int, int] | None = None,
        board: int | None = None,
        round_history: tuple[int, str] | None = None,
        committed: tuple[int, int] | None = None,
        folded: int | None = None,
    ) -> "LeducState":
        histories = self.round_histories
        if round_history is not None:
            index, updated = round_history
            histories = (updated, histories[1]) if index == 0 else (histories[0], updated)
        return LeducState(
            cards=self.cards if cards is None else cards,
            board=self.board if board is None else board,
            round_histories=histories,
            committed=self.committed if committed is None else committed,
            folded=self.folded if folded is None else folded,
        )


class LeducPoker:
    def initial_state(self) -> LeducState:
        return LeducState()

    def num_actions(self) -> int:
        return 3
