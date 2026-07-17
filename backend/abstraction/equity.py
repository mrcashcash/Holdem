"""Equity and equity-histogram computation over compact card ids.

Builds on the Numba seven-card evaluator from ``vectorized_engine`` when
available (a pure-Python fallback keeps tests runnable anywhere). The
potential-aware inputs to bucketing are histograms of *future* equity:
for a flop hand, the distribution of showdown equity over sampled runouts;
for a turn hand, over all possible rivers. River hands get exact equity
against every opponent combination.
"""

from __future__ import annotations

import numpy as np

from backend.poker import SUITS, best_score
from backend.vectorized_engine import vectorized_enabled

try:  # pragma: no cover - exercised implicitly by the numba path
    from numba import njit

    from backend.vectorized_engine import _evaluate_seven

    _NUMBA = vectorized_enabled()
except Exception:  # pragma: no cover
    _NUMBA = False


def _card_tuple(card: int) -> tuple[int, str]:
    return card // 4 + 2, SUITS[card % 4]


def _score_python(cards: list[int]) -> tuple[int, ...]:
    return best_score([_card_tuple(card) for card in cards])


if _NUMBA:

    @njit(cache=True)
    def _seven(hole_a: int, hole_b: int, board: np.ndarray) -> np.int64:
        cards = np.empty(7, dtype=np.int64)
        cards[0] = hole_a
        cards[1] = hole_b
        for index in range(5):
            cards[index + 2] = board[index]
        return _evaluate_seven(cards)

    @njit(cache=True)
    def _river_equity_kernel(hole: np.ndarray, board: np.ndarray) -> float:
        dead = np.zeros(52, dtype=np.bool_)
        dead[hole[0]] = dead[hole[1]] = True
        for index in range(5):
            dead[board[index]] = True
        hero = _seven(hole[0], hole[1], board)
        wins = 0.0
        total = 0.0
        for first in range(52):
            if dead[first]:
                continue
            for second in range(first + 1, 52):
                if dead[second]:
                    continue
                villain = _seven(first, second, board)
                if hero > villain:
                    wins += 1.0
                elif hero == villain:
                    wins += 0.5
                total += 1.0
        return wins / total

    @njit(cache=True)
    def _histogram_kernel(
        hole: np.ndarray,
        partial_board: np.ndarray,
        board_size: int,
        bins: int,
        scenarios: int,
        opponents_per_scenario: int,
        seed: int,
    ) -> np.ndarray:
        """Histogram of showdown equity over sampled/enumerated runouts.

        ``scenarios <= 0`` enumerates single-card completions (turn street).
        """
        np.random.seed(seed)
        dead = np.zeros(52, dtype=np.bool_)
        dead[hole[0]] = dead[hole[1]] = True
        for index in range(board_size):
            dead[partial_board[index]] = True
        remaining = np.empty(52 - 2 - board_size, dtype=np.int64)
        cursor = 0
        for card in range(52):
            if not dead[card]:
                remaining[cursor] = card
                cursor += 1

        histogram = np.zeros(bins, dtype=np.float64)
        fill = 5 - board_size
        board = np.empty(5, dtype=np.int64)
        for index in range(board_size):
            board[index] = partial_board[index]

        if scenarios <= 0:
            scenario_count = remaining.shape[0]
        else:
            scenario_count = scenarios

        for scenario in range(scenario_count):
            if scenarios <= 0:
                board[board_size] = remaining[scenario]
                used_first = remaining[scenario]
                used_second = -1
            else:
                first = np.random.randint(remaining.shape[0])
                second = np.random.randint(remaining.shape[0] - 1)
                if second >= first:
                    second += 1
                board[board_size] = remaining[first]
                if fill > 1:
                    board[board_size + 1] = remaining[second]
                    used_second = remaining[second]
                else:
                    used_second = -1
                used_first = remaining[first]

            hero = _seven(hole[0], hole[1], board)
            wins = 0.0
            total = 0.0
            for _ in range(opponents_per_scenario):
                while True:
                    a = remaining[np.random.randint(remaining.shape[0])]
                    b = remaining[np.random.randint(remaining.shape[0])]
                    if a != b and a != used_first and b != used_first and a != used_second and b != used_second:
                        break
                villain = _seven(a, b, board)
                if hero > villain:
                    wins += 1.0
                elif hero == villain:
                    wins += 0.5
                total += 1.0
            equity = wins / total
            slot = int(equity * bins)
            if slot >= bins:
                slot = bins - 1
            histogram[slot] += 1.0

        return histogram / histogram.sum()


def river_equity(hole: tuple[int, int], board: tuple[int, ...]) -> float:
    """Exact showdown equity on the river against a uniform opponent range."""
    if _NUMBA:
        return float(_river_equity_kernel(np.asarray(hole, dtype=np.int64), np.asarray(board, dtype=np.int64)))
    dead = set(hole) | set(board)
    hero = _score_python(list(hole) + list(board))
    wins = total = 0.0
    for first in range(52):
        if first in dead:
            continue
        for second in range(first + 1, 52):
            if second in dead:
                continue
            villain = _score_python([first, second] + list(board))
            if hero > villain:
                wins += 1.0
            elif hero == villain:
                wins += 0.5
            total += 1.0
    return wins / total


def equity_histogram(
    hole: tuple[int, int],
    board: tuple[int, ...],
    bins: int = 8,
    scenarios: int = 48,
    opponents_per_scenario: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Future-equity histogram for a flop (sampled runouts) or turn (all rivers) hand."""
    board_size = len(board)
    if board_size not in (3, 4):
        raise ValueError("equity histograms are defined for flop and turn boards")
    scenario_count = scenarios if board_size == 3 else 0
    if _NUMBA:
        return _histogram_kernel(
            np.asarray(hole, dtype=np.int64),
            np.asarray(board, dtype=np.int64),
            board_size,
            bins,
            scenario_count,
            opponents_per_scenario,
            seed,
        )
    return _histogram_python(hole, board, bins, scenario_count, opponents_per_scenario, seed)


def _histogram_python(
    hole: tuple[int, int],
    board: tuple[int, ...],
    bins: int,
    scenarios: int,
    opponents_per_scenario: int,
    seed: int,
) -> np.ndarray:  # pragma: no cover - numba path is the default
    import random

    rng = random.Random(seed)
    dead = set(hole) | set(board)
    remaining = [card for card in range(52) if card not in dead]
    fill = 5 - len(board)
    histogram = np.zeros(bins, dtype=np.float64)
    runouts = [[card] for card in remaining] if scenarios <= 0 else [rng.sample(remaining, fill) for _ in range(scenarios)]
    for runout in runouts:
        full_board = list(board) + runout
        hero = _score_python(list(hole) + full_board)
        used = dead | set(runout)
        wins = total = 0.0
        for _ in range(opponents_per_scenario):
            a, b = rng.sample([card for card in range(52) if card not in used], 2)
            villain = _score_python([a, b] + full_board)
            if hero > villain:
                wins += 1.0
            elif hero == villain:
                wins += 0.5
            total += 1.0
        slot = min(bins - 1, int(wins / total * bins))
        histogram[slot] += 1.0
    return histogram / histogram.sum()
