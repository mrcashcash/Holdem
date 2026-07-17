"""Betting (action) abstraction and off-tree bet translation.

The solver's tree offers a small menu per decision: fold, check/call, a few
pot-fraction raises, and all-in. Live opponents bet arbitrary sizes; those
are mapped onto the two neighbouring menu sizes with the pseudo-harmonic
weights of Ganzfried & Sandholm (probabilistic action translation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

FOLD = 0
CHECK_CALL = 1
ALL_IN = 2
_FIRST_RAISE_ID = 3


@dataclass
class ActionAbstraction:
    """Produces the abstract action menu for a betting decision.

    Raise sizes are pot fractions applied after the caller matches: the
    raise-to amount is ``call_total + fraction * (pot + to_call)``. Raise
    counts per street are capped to bound the tree; past the cap only
    fold / call / all-in remain.
    """

    preflop_fractions: tuple[float, ...] = (0.75, 1.0, 1.5)
    flop_fractions: tuple[float, ...] = (0.33, 0.5, 0.75, 1.0, 2.0)
    turn_fractions: tuple[float, ...] = (0.5, 0.75, 1.0, 2.0)
    river_fractions: tuple[float, ...] = (0.5, 1.0, 2.0)
    max_raises_per_street: int = 4
    _by_street: tuple[tuple[float, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        self._by_street = (
            self.preflop_fractions,
            self.flop_fractions,
            self.turn_fractions,
            self.river_fractions,
        )

    def num_actions(self) -> int:
        widest = max(len(fractions) for fractions in self._by_street)
        return _FIRST_RAISE_ID + widest

    def fractions_for_street(self, street: int) -> tuple[float, ...]:
        return self._by_street[street]

    def raise_action_id(self, fraction_index: int) -> int:
        return _FIRST_RAISE_ID + fraction_index

    def menu(
        self,
        street: int,
        pot: float,
        to_call: float,
        stack_behind: float,
        raises_this_street: int,
    ) -> list[int]:
        """Legal abstract action ids, cheapest first."""
        actions: list[int] = []
        if to_call > 0:
            actions.append(FOLD)
        actions.append(CHECK_CALL)
        if stack_behind <= to_call:
            return actions  # calling already commits the stack
        if raises_this_street < self.max_raises_per_street:
            for index, fraction in enumerate(self.fractions_for_street(street)):
                raise_by = fraction * (pot + to_call)
                if to_call + raise_by < stack_behind:
                    actions.append(self.raise_action_id(index))
        actions.append(ALL_IN)
        return actions

    def raise_amount(self, action: int, street: int, pot: float, to_call: float) -> float:
        """Chips added beyond the call for a raise action id."""
        fraction = self.fractions_for_street(street)[action - _FIRST_RAISE_ID]
        return fraction * (pot + to_call)


def pseudo_harmonic_weights(observed: float, lower: float, upper: float) -> tuple[float, float]:
    """Probabilities of mapping an off-tree bet onto the neighbouring sizes.

    All sizes are pot fractions. Ganzfried & Sandholm's pseudo-harmonic map:
    f(x) = (upper - x)(1 + lower) / ((upper - lower)(1 + x)) is the weight of
    the smaller size; it is 1 at x = lower and 0 at x = upper.
    """
    if upper <= lower:
        return 1.0, 0.0
    clamped = min(max(observed, lower), upper)
    weight_lower = (upper - clamped) * (1.0 + lower) / ((upper - lower) * (1.0 + clamped))
    weight_lower = min(max(weight_lower, 0.0), 1.0)
    return weight_lower, 1.0 - weight_lower
