"""AIVAT-style variance reduction for live evaluation (Burch et al., AAAI 2018).

Implemented portion: the CHANCE control variates — the dominant variance source
in heads-up evals (runout luck: flops hitting ranges, all-in coinflips, hit-or-
miss rivers). At every board reveal we subtract

    correction = v(actual cards) - E_cards[v(alternative cards)]

whose expectation is exactly zero (cards are uniform given play reached the
reveal), so the corrected winnings stay UNBIASED for any value function v.
Variance reduction depends only on v's quality; we use hero pot equity times
the pot at the reveal — cheap and strongly correlated with realized luck.

Our-decision control variates (the other AIVAT half) need per-action value
estimates; they are a later upgrade — chance terms alone recover most of the
benefit in practice for heads-up bots.

Usage: create one ``ChanceCorrector`` per hand, call ``observe(engine)`` after
every state change (it detects new board cards itself), then
``corrected = winnings_bb - corrector.total_bb()`` at hand end.
"""

from __future__ import annotations

import random

from backend.solver.gpu.deals import DealSampler, combos
from backend.vectorized_engine import card_id

_COMBO_INDEX: dict[tuple[int, int], int] | None = None


def _combo_index(hero_cards) -> int:
    global _COMBO_INDEX
    if _COMBO_INDEX is None:
        _COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(combos())}
    a, b = sorted(card_id(card) for card in hero_cards)
    return _COMBO_INDEX[(a, b)]


def _equity(combo: int, board_ids: tuple[int, ...], rng: random.Random, samples: int) -> float:
    """Hero equity vs a uniform opponent on a (possibly partial) board."""
    value = DealSampler._mean_equity(board_ids, rng, samples)[combo]
    return float(value) if value >= 0 else 0.5


class ChanceCorrector:
    """Accumulates chance control variates for one hand (hero = ``seat``)."""

    def __init__(self, engine, seat: int = 0, samples: int = 12, alternatives: int = 24, seed: int = 0) -> None:
        self.seat = seat
        self.samples = samples
        self.alternatives = alternatives
        self.rng = random.Random(seed)
        self.combo = _combo_index(engine.hole_cards[seat])
        self.hero_ids = {card_id(card) for card in engine.hole_cards[seat]}
        self.seen_board = 0
        self.corrections: list[float] = []  # chips

    def observe(self, engine) -> None:
        board = [card_id(card) for card in engine.community]
        if len(board) <= self.seen_board:
            return
        # Scale by the pot AT THE REVEAL. An all-in run-out deals the remaining
        # streets inside _runout_and_showdown and then zeroes engine.pot when
        # awarding it, so reading engine.pot here would silently scale the
        # correction to zero on exactly the highest-variance hands (all-in
        # coinflips) — the ones AIVAT exists to correct. last_pot holds the
        # contested pot in that case.
        pot = float(engine.pot) or float(getattr(engine, "last_pot", 0.0))
        known_before = set(board[: self.seen_board]) | self.hero_ids
        actual = tuple(board)
        actual_equity = _equity(self.combo, actual, random.Random(17), self.samples)
        # E[v] over alternative reveals of the same number of cards.
        remaining = [c for c in range(52) if c not in known_before]
        reveal_count = len(board) - self.seen_board
        total = 0.0
        for _ in range(self.alternatives):
            alt = tuple(board[: self.seen_board]) + tuple(self.rng.sample(remaining, reveal_count))
            total += _equity(self.combo, alt, random.Random(17), self.samples)
        expected_equity = total / self.alternatives
        self.corrections.append(pot * (actual_equity - expected_equity))
        self.seen_board = len(board)

    def total_chips(self) -> float:
        return sum(self.corrections)

    def total_bb(self, big_blind: float) -> float:
        return self.total_chips() / big_blind
