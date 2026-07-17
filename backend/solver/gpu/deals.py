"""Per-board bucket assignment for all 1,326 combos (docs/GPU_CFR_PLAN.md §3).

On a fixed board the equity of every combo against a uniform opponent is a
function of the sorted 7-card scores, so a whole street's bucket ids cost one
Numba scoring batch (~1 ms) plus a sort. Buckets:

  preflop — the 169 lossless classes
  flop    — quantized mean river-equity over sampled runouts from the flop
  turn    — quantized mean river-equity over sampled rivers
  river   — quantized exact-rank equity on the full board

This scalar-equity abstraction trades per-bucket quality for enormous
throughput; the CFR core keeps lossless combo ranges within each iteration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from backend.abstraction.cards import preflop_class

try:  # pragma: no cover - the numba path is the default in this project
    from numba import njit

    from backend.vectorized_engine import _evaluate_seven, vectorized_enabled

    _NUMBA = vectorized_enabled()
except Exception:  # pragma: no cover
    _NUMBA = False

NUM_COMBOS = 1326

_COMBOS = np.asarray(
    [(first, second) for first in range(52) for second in range(first + 1, 52)],
    dtype=np.int64,
)

_PREFLOP_CLASS = np.asarray(
    [preflop_class((int(first), int(second))) for first, second in _COMBOS],
    dtype=np.int32,
)

# card_in_combo[c, k] — does combo k contain card c (for collision math).
CARD_IN_COMBO = np.zeros((52, NUM_COMBOS), dtype=bool)
for _index, (_a, _b) in enumerate(_COMBOS):
    CARD_IN_COMBO[_a, _index] = True
    CARD_IN_COMBO[_b, _index] = True


def combos() -> np.ndarray:
    return _COMBOS


if _NUMBA:

    @njit(cache=True)
    def _score_all_combos(board: np.ndarray, combo_array: np.ndarray) -> np.ndarray:
        scores = np.full(combo_array.shape[0], -1, dtype=np.int64)
        cards = np.empty(7, dtype=np.int64)
        for position in range(5):
            cards[position + 2] = board[position]
        for index in range(combo_array.shape[0]):
            first, second = combo_array[index, 0], combo_array[index, 1]
            collides = False
            for position in range(5):
                if board[position] == first or board[position] == second:
                    collides = True
                    break
            if collides:
                continue
            cards[0] = first
            cards[1] = second
            scores[index] = _evaluate_seven(cards)
        return scores


def score_all_combos(board: tuple[int, ...] | np.ndarray) -> np.ndarray:
    """7-card score per combo on a 5-card board; -1 where the combo collides."""
    board_array = np.asarray(board, dtype=np.int64)
    if _NUMBA:
        return _score_all_combos(board_array, _COMBOS)
    from backend.poker import best_score

    from backend.solver.holdem import card_tuple

    scores = np.full(NUM_COMBOS, -1, dtype=np.int64)
    board_set = set(int(card) for card in board_array)
    board_cards = [card_tuple(int(card)) for card in board_array]
    ranking: dict[tuple, int] = {}
    for index, (first, second) in enumerate(_COMBOS):
        if int(first) in board_set or int(second) in board_set:
            continue
        key = best_score([card_tuple(int(first)), card_tuple(int(second))] + board_cards)
        ranking.setdefault(key, len(ranking))
        scores[index] = 0  # placeholder, replaced below
    ordered = sorted(ranking.keys())
    order_map = {key: position for position, key in enumerate(ordered)}
    for index, (first, second) in enumerate(_COMBOS):
        if scores[index] >= 0:
            key = best_score([card_tuple(int(first)), card_tuple(int(second))] + board_cards)
            scores[index] = order_map[key]
    return scores


def equity_from_scores(scores: np.ndarray) -> np.ndarray:
    """Exact uniform-range equity per combo (card removal included); -1 invalid.

    Global pass: wins vs all valid combos via one sort. Correction pass: for
    each of the hero's two cards, subtract the contribution of opponent combos
    sharing that card (52 small sorts) — the standard blocker correction that
    turns O(n^2) pairwise enumeration into O(n log n).
    """
    valid = scores >= 0
    equity = np.full(NUM_COMBOS, -1.0, dtype=np.float64)
    values = scores[valid]
    count = values.shape[0]
    if count <= 1:
        return equity

    order = np.sort(values)
    below_all = np.searchsorted(order, values, side="left").astype(np.float64)
    ties_all = (np.searchsorted(order, values, side="right") - below_all - 1).astype(np.float64)
    wins = below_all + 0.5 * ties_all  # vs all valid combos except self

    valid_indices = np.flatnonzero(valid)
    position_of = np.full(NUM_COMBOS, -1, dtype=np.int64)
    position_of[valid_indices] = np.arange(count)

    blocked_wins = np.zeros(count, dtype=np.float64)
    blocked_counts = np.zeros(count, dtype=np.int64)
    for card in range(52):
        members = valid & CARD_IN_COMBO[card]
        member_positions = position_of[np.flatnonzero(members)]
        if member_positions.size == 0:
            continue
        member_scores = values[member_positions]
        member_order = np.sort(member_scores)
        below = np.searchsorted(member_order, member_scores, side="left").astype(np.float64)
        ties = (np.searchsorted(member_order, member_scores, side="right") - below - 1).astype(np.float64)
        blocked_wins[member_positions] += below + 0.5 * ties
        blocked_counts[member_positions] += member_scores.shape[0] - 1

    opponents = (count - 1) - blocked_counts
    equity[valid_indices] = (wins - blocked_wins) / np.maximum(opponents, 1)
    return equity


@dataclass
class Deal:
    """Sampled runout with per-street bucket ids for every combo."""

    board: tuple[int, int, int, int, int]
    buckets: np.ndarray  # int32 [4, NUM_COMBOS]; -1 where the combo collides
    valid: np.ndarray  # bool [NUM_COMBOS] — combo does not collide with board
    river_scores: np.ndarray  # int64 [NUM_COMBOS]; -1 where invalid


class DealSampler:
    """Samples runouts and computes bucket tensors (docs/GPU_CFR_PLAN.md §3)."""

    PREFLOP_BUCKETS = 169

    def __init__(
        self,
        flop_buckets: int = 20,
        turn_buckets: int = 20,
        river_buckets: int = 20,
        flop_samples: int = 8,
        turn_samples: int = 8,
    ) -> None:
        self.flop_buckets = flop_buckets
        self.turn_buckets = turn_buckets
        self.river_buckets = river_buckets
        self.flop_samples = flop_samples
        self.turn_samples = turn_samples

    def bucket_counts(self) -> tuple[int, int, int, int]:
        return (self.PREFLOP_BUCKETS, self.flop_buckets, self.turn_buckets, self.river_buckets)

    def sample(self, rng: random.Random) -> Deal:
        board = tuple(rng.sample(range(52), 5))
        return self.for_board(board, rng)

    def for_board(self, board: tuple[int, ...], rng: random.Random) -> Deal:
        board = tuple(int(card) for card in board)
        river_scores = score_all_combos(board)
        river_equity = equity_from_scores(river_scores)
        valid = river_scores >= 0

        buckets = np.full((4, NUM_COMBOS), -1, dtype=np.int32)
        buckets[0][valid] = _PREFLOP_CLASS[valid]
        buckets[3][valid] = self._quantize(river_equity[valid], self.river_buckets)

        turn_equity = self._mean_equity(board[:4], rng, self.turn_samples)
        turn_valid = valid & (turn_equity >= 0)
        buckets[2][turn_valid] = self._quantize(turn_equity[turn_valid], self.turn_buckets)

        flop_equity = self._mean_equity(board[:3], rng, self.flop_samples)
        flop_valid = valid & (flop_equity >= 0)
        buckets[1][flop_valid] = self._quantize(flop_equity[flop_valid], self.flop_buckets)

        return Deal(board=board, buckets=buckets, valid=valid, river_scores=river_scores)

    @staticmethod
    def _quantize(equity: np.ndarray, count: int) -> np.ndarray:
        return np.minimum((equity * count).astype(np.int32), count - 1)

    @staticmethod
    def _mean_equity(partial_board: tuple[int, ...], rng: random.Random, samples: int) -> np.ndarray:
        """Mean river equity per combo over sampled completions of the board."""
        used = set(partial_board)
        remaining = [card for card in range(52) if card not in used]
        fill = 5 - len(partial_board)
        total = np.zeros(NUM_COMBOS, dtype=np.float64)
        counts = np.zeros(NUM_COMBOS, dtype=np.int64)
        for _ in range(samples):
            completion = rng.sample(remaining, fill)
            scores = score_all_combos(tuple(partial_board) + tuple(completion))
            equity = equity_from_scores(scores)
            seen = equity >= 0
            total[seen] += equity[seen]
            counts[seen] += 1
        result = np.full(NUM_COMBOS, -1.0, dtype=np.float64)
        seen_any = counts > 0
        result[seen_any] = total[seen_any] / counts[seen_any]
        return result
