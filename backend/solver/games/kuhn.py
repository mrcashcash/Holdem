"""Kuhn poker: the standard 3-card ground-truth game.

Each player antes 1 and receives one of {J=0, Q=1, K=2}. One betting round,
bet size 1. The game value for player 0 at Nash equilibrium is -1/18.
Actions: 0 = check/fold (passive), 1 = bet/call (aggressive).
"""

from __future__ import annotations

import itertools
import random
from typing import Hashable, Sequence

PASS, BET = 0, 1
_DEALS = list(itertools.permutations(range(3), 2))


class KuhnState:
    __slots__ = ("cards", "history")

    def __init__(self, cards: tuple[int, int] | None = None, history: str = "") -> None:
        self.cards = cards
        self.history = history

    def is_chance(self) -> bool:
        return self.cards is None

    def chance_outcomes(self) -> Sequence[tuple["KuhnState", float]]:
        probability = 1.0 / len(_DEALS)
        return [(KuhnState(deal, ""), probability) for deal in _DEALS]

    def sample_chance(self, rng: random.Random) -> "KuhnState":
        return KuhnState(rng.choice(_DEALS), "")

    def is_terminal(self) -> bool:
        history = self.history
        return history in ("pp", "bb", "bp", "pbb", "pbp")

    def current_player(self) -> int:
        return len(self.history) % 2

    def legal_actions(self) -> Sequence[int]:
        return (PASS, BET)

    def infoset_key(self) -> Hashable:
        return (self.cards[self.current_player()], self.history)

    def child(self, action: int) -> "KuhnState":
        return KuhnState(self.cards, self.history + ("b" if action == BET else "p"))

    def utility(self, player: int) -> float:
        history, cards = self.history, self.cards
        winner = 0 if cards[0] > cards[1] else 1
        if history == "bp":
            result = (1.0, 0)  # player 1 folded to a bet; player 0 wins the ante
        elif history == "pbp":
            result = (1.0, 1)  # player 0 folded to a bet
        elif history == "pp":
            result = (1.0, winner)
        else:  # "bb" or "pbb": showdown for ante + bet
            result = (2.0, winner)
        amount, taker = result
        return amount if player == taker else -amount


class KuhnPoker:
    def initial_state(self) -> KuhnState:
        return KuhnState()

    def num_actions(self) -> int:
        return 2
