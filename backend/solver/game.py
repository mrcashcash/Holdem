"""Minimal extensive-form game protocol for the MCCFR solver.

Two-player zero-sum games with chance nodes and perfect recall. States are
immutable: ``child`` returns a new state. Infoset keys must be hashable and
identical for states a player cannot distinguish.
"""

from __future__ import annotations

import random
from typing import Hashable, Protocol, Sequence


class State(Protocol):
    def is_terminal(self) -> bool: ...

    def is_chance(self) -> bool: ...

    def current_player(self) -> int:
        """Acting player index (0 or 1) at a decision node."""
        ...

    def legal_actions(self) -> Sequence[int]:
        """Legal action ids at a decision node (stable ordering)."""
        ...

    def infoset_key(self) -> Hashable:
        """Information-set key for the acting player at a decision node."""
        ...

    def child(self, action: int) -> "State":
        """Successor state after the acting player takes ``action``."""
        ...

    def chance_outcomes(self) -> Sequence[tuple["State", float]]:
        """(successor, probability) pairs at a chance node."""
        ...

    def sample_chance(self, rng: random.Random) -> "State":
        """Sample one chance successor (may be cheaper than enumerating)."""
        ...

    def utility(self, player: int) -> float:
        """Terminal payoff for ``player`` (zero-sum)."""
        ...


class Game(Protocol):
    def initial_state(self) -> State: ...

    def num_actions(self) -> int:
        """Upper bound on action ids, sizing the per-infoset tables."""
        ...
