"""Deterministic blueprint-audit helpers for the strategic training loop.

These metrics are deliberately conservative.  They evaluate reproducible
seat-swapped match results and a tiny tabular Kuhn-poker CFR sanity probe; they
are not claims of exact no-limit Hold'em exploitability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class BlueprintAudit:
    score: float
    lower_confidence: float
    floor: float
    hands: int
    paired_hands: int


@dataclass(frozen=True)
class KuhnCfrAudit:
    iterations: int
    average_value: float
    value_gap: float
    average_positive_regret: float


def _wilson_lower_bound(score: float, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    denominator = 1 + z * z / samples
    centre = score + z * z / (2 * samples)
    margin = z * math.sqrt((score * (1 - score) + z * z / (4 * samples)) / samples)
    return max(0.0, (centre - margin) / denominator)


def _bb_per_100_quality(value: float, scale: float = 80.0) -> float:
    """Map chip EV onto the existing bounded blueprint-gate scale."""
    return min(1.0, max(0.0, 0.5 + float(value) / max(1.0, 2.0 * scale)))


def score_blueprint(results: dict[str, tuple[float, float, int]]) -> BlueprintAudit:
    """Aggregate paired chip-EV audits instead of raw hand-win frequency.

    A profitable poker strategy can lose most individual hands by folding small
    pots and winning fewer, larger pots.  Treating that as a blueprint failure
    trains the wrong objective.  Each lane supplies its BB/100 point estimate,
    paired-deal lower confidence bound, and hand count.  The aggregate lower
    score remains conservative while the lane floor prevents one fixed matchup
    from being hidden by strong results elsewhere.
    """
    total_hands = sum(max(0, hands) for _, _, hands in results.values())
    weighted_score = sum(_bb_per_100_quality(value) * max(0, hands) for value, _, hands in results.values()) / max(1, total_hands)
    weighted_lower = sum(_bb_per_100_quality(lower) * max(0, hands) for _, lower, hands in results.values()) / max(1, total_hands)
    floor = min((_bb_per_100_quality(value) for value, _, hands in results.values() if hands > 0), default=0.0)
    return BlueprintAudit(weighted_score, weighted_lower, floor, total_hands, total_hands // 2 * 2)


def _terminal_utility(cards: tuple[int, int], history: str) -> float | None:
    high_card_wins = 1.0 if cards[0] > cards[1] else -1.0
    if history == "pp":
        return high_card_wins
    if history == "bp":
        return 1.0
    if history == "pbp":
        return -1.0
    if history in {"bb", "pbb"}:
        return 2.0 * high_card_wins
    return None


def kuhn_cfr_audit(iterations: int = 192) -> KuhnCfrAudit:
    """Run a small deterministic CFR probe; its known game value is -1/18."""
    regrets: dict[str, list[float]] = {}
    strategy_sums: dict[str, list[float]] = {}

    def strategy(key: str) -> list[float]:
        values = regrets.setdefault(key, [0.0, 0.0])
        positive = [max(0.0, value) for value in values]
        total = sum(positive)
        return [value / total for value in positive] if total > 1e-12 else [0.5, 0.5]

    def cfr(cards: tuple[int, int], history: str, reach_zero: float, reach_one: float) -> float:
        terminal = _terminal_utility(cards, history)
        if terminal is not None:
            return terminal
        player = len(history) % 2
        key = f"{cards[player]}:{history}"
        current_strategy = strategy(key)
        strategy_total = strategy_sums.setdefault(key, [0.0, 0.0])
        own_reach = reach_zero if player == 0 else reach_one
        for index, probability in enumerate(current_strategy):
            strategy_total[index] += own_reach * probability
        actions = ("p", "b")
        utilities = [
            cfr(cards, history + action, reach_zero * current_strategy[index], reach_one)
            if player == 0
            else cfr(cards, history + action, reach_zero, reach_one * current_strategy[index])
            for index, action in enumerate(actions)
        ]
        node_utility = sum(probability * utility for probability, utility in zip(current_strategy, utilities))
        opponent_reach = reach_one if player == 0 else reach_zero
        for index, utility in enumerate(utilities):
            regret = utility - node_utility if player == 0 else node_utility - utility
            regrets[key][index] += opponent_reach * regret
        return node_utility

    for _ in range(max(1, iterations)):
        for cards in permutations((0, 1, 2), 2):
            cfr(cards, "", 1.0, 1.0)

    def average_strategy(key: str) -> list[float]:
        values = strategy_sums.get(key, [0.0, 0.0])
        total = sum(values)
        return [value / total for value in values] if total > 1e-12 else [0.5, 0.5]

    def evaluate(cards: tuple[int, int], history: str) -> float:
        terminal = _terminal_utility(cards, history)
        if terminal is not None:
            return terminal
        player = len(history) % 2
        probabilities = average_strategy(f"{cards[player]}:{history}")
        return sum(probability * evaluate(cards, history + action) for probability, action in zip(probabilities, ("p", "b")))

    average_value = sum(evaluate(cards, "") for cards in permutations((0, 1, 2), 2)) / 6
    positive_regret = [max(0.0, value) for values in regrets.values() for value in values]
    return KuhnCfrAudit(max(1, iterations), average_value, abs(average_value + 1 / 18), sum(positive_regret) / max(1, len(positive_regret) * iterations))
