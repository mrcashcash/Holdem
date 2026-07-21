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
    if board_array.shape != (5,) or board_array.min() < 0 or board_array.max() > 51:
        # The numba kernel would read out of bounds (native crash, not an
        # exception) — observed via a street-drift bug handing it 4 cards.
        raise ValueError(f"score_all_combos requires 5 valid card ids, got {board_array.tolist()}")
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
        distributional: bool = False,
        std_bins: int = 4,
    ) -> None:
        self.flop_buckets = flop_buckets
        self.turn_buckets = turn_buckets
        self.river_buckets = river_buckets
        self.flop_samples = flop_samples
        self.turn_samples = turn_samples
        # Distribution-aware flop/turn: bucket on (mean-equity bin, equity-std
        # bin) so draws separate from made hands and air. std_bins is the
        # number of drawiness bins; the mean axis keeps flop_buckets/turn_buckets
        # granularity, so total buckets = mean_bins * std_bins (kept <= 169).
        self.distributional = distributional
        self.std_bins = std_bins
        self._std_edges: dict[int, np.ndarray] = {}  # street -> std quantile edges

    def bucket_counts(self) -> tuple[int, int, int, int]:
        flop = self.flop_buckets * self.std_bins if self.distributional else self.flop_buckets
        turn = self.turn_buckets * self.std_bins if self.distributional else self.turn_buckets
        return (self.PREFLOP_BUCKETS, flop, turn, self.river_buckets)

    def state(self) -> dict:
        """Full serializable state incl. fitted std edges (for checkpoints)."""
        return {
            "flop_buckets": self.flop_buckets,
            "turn_buckets": self.turn_buckets,
            "river_buckets": self.river_buckets,
            "flop_samples": self.flop_samples,
            "turn_samples": self.turn_samples,
            "distributional": self.distributional,
            "std_bins": self.std_bins,
            "std_edges": {str(s): edges.tolist() for s, edges in self._std_edges.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "DealSampler":
        edges = state.pop("std_edges", {})
        sampler = cls(**{k: state[k] for k in state if k != "std_edges"})
        sampler._std_edges = {int(s): np.asarray(e, dtype=np.float64) for s, e in edges.items()}
        return sampler

    def fit_std_edges(self, samples: int = 400, seed: int = 0) -> None:
        """Learn equity-std quantile edges per street (one-time, cheap)."""
        rng = random.Random(seed)
        for street, size in ((1, 3), (2, 4)):
            stds = []
            for _ in range(samples):
                cards = rng.sample(range(52), size)
                _, std = self._mean_std_equity(tuple(cards), rng, self.flop_samples if street == 1 else self.turn_samples)
                stds.append(std[std >= 0])
            pooled = np.concatenate(stds)
            self._std_edges[street] = np.quantile(pooled, np.linspace(0, 1, self.std_bins + 1)[1:-1])

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
        # River has no future street, so its equity distribution is a point
        # mass — scalar equity quantile is exact there.
        buckets[3][valid] = self._quantize(river_equity[valid], self.river_buckets)

        self._assign_street(buckets, board[:4], rng, self.turn_samples, self.turn_buckets, 2, valid)
        self._assign_street(buckets, board[:3], rng, self.flop_samples, self.flop_buckets, 1, valid)
        return Deal(board=board, buckets=buckets, valid=valid, river_scores=river_scores)

    def _mean_bins_for(self, street: int) -> int:
        return self.turn_buckets if street == 2 else self.flop_buckets

    def _bucket_from_mean_std(self, mean: np.ndarray, std: np.ndarray, street: int) -> np.ndarray:
        """The one bucketing formula, shared by training (vectorized) and
        serving (single combo) so their buckets are identical by construction."""
        mean_bins = self._mean_bins_for(street)
        mean_bin = np.minimum((mean * mean_bins).astype(np.int32), mean_bins - 1)
        if not self.distributional:
            return mean_bin
        edges = self._std_edges.get(street)
        std_bin = (
            np.searchsorted(edges, std).astype(np.int32)
            if edges is not None and len(edges)
            else np.zeros_like(mean_bin)
        )
        return mean_bin * self.std_bins + std_bin

    def _assign_street(self, buckets, partial_board, rng, samples, mean_bins, street, valid):
        mean, std = self._mean_std_equity(partial_board, rng, samples)
        seen = valid & (mean >= 0)
        buckets[street][seen] = self._bucket_from_mean_std(mean[seen], std[seen], street)

    def street_bucket_for_combo(self, partial_board, street: int, combo_index: int, rng) -> int | None:
        """Bucket one combo on a partial board — serving uses this so it maps
        into the exact bucket the trainer learned (validated by test)."""
        samples = self.turn_samples if street == 2 else self.flop_samples
        mean, std = self._mean_std_equity(tuple(int(c) for c in partial_board), rng, samples)
        if mean[combo_index] < 0:
            return None
        bucket = self._bucket_from_mean_std(mean[combo_index : combo_index + 1], std[combo_index : combo_index + 1], street)
        return int(bucket[0])

    @staticmethod
    def _quantize(equity: np.ndarray, count: int) -> np.ndarray:
        return np.minimum((equity * count).astype(np.int32), count - 1)

    @staticmethod
    def _mean_equity(partial_board: tuple[int, ...], rng: random.Random, samples: int) -> np.ndarray:
        """Mean river equity per combo over sampled completions of the board."""
        mean, _ = DealSampler._mean_std_equity(partial_board, rng, samples)
        return mean

    @staticmethod
    def _mean_std_equity(
        partial_board: tuple[int, ...], rng: random.Random, samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) of river equity per combo over sampled runouts.

        The std is the distribution-aware signal: at equal mean equity a draw
        (equity swings between streets) has high std while a made hand has low
        std — the axis scalar bucketing collapses, causing draws to be folded
        as if they were air.
        """
        used = set(partial_board)
        remaining = [card for card in range(52) if card not in used]
        fill = 5 - len(partial_board)
        total = np.zeros(NUM_COMBOS, dtype=np.float64)
        total_sq = np.zeros(NUM_COMBOS, dtype=np.float64)
        counts = np.zeros(NUM_COMBOS, dtype=np.int64)
        for _ in range(samples):
            completion = rng.sample(remaining, fill)
            scores = score_all_combos(tuple(partial_board) + tuple(completion))
            equity = equity_from_scores(scores)
            seen = equity >= 0
            total[seen] += equity[seen]
            total_sq[seen] += equity[seen] ** 2
            counts[seen] += 1
        mean = np.full(NUM_COMBOS, -1.0, dtype=np.float64)
        std = np.full(NUM_COMBOS, -1.0, dtype=np.float64)
        seen_any = counts > 0
        m = total[seen_any] / counts[seen_any]
        variance = np.maximum(total_sq[seen_any] / counts[seen_any] - m**2, 0.0)
        mean[seen_any] = m
        std[seen_any] = np.sqrt(variance)
        return mean, std
