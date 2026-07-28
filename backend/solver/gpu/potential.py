"""Recursive potential-aware card abstraction for GPU blueprint v3.

The abstraction is fitted from the river backwards:

* river combos use OCHS-style equity against several static opponent ranges,
  plus blocker and public-board texture features;
* a turn combo is represented by its transition distribution over ordered
  river clusters;
* a flop combo is represented by its transition distribution over ordered
  turn clusters; and
* transition distributions are clustered by L1 distance between CDF
  landmarks, a compact approximation of one-dimensional EMD.

Every assignment canonicalizes the public board under all 24 suit
permutations. Future cards are a deterministic common stratified set selected
in canonical coordinates, so all private combos on a board see the same
runouts and suit-isomorphic situations map identically.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np

from backend.solver.gpu.deals import (
    CARD_IN_COMBO,
    NUM_COMBOS,
    combos,
    score_all_combos,
)

_COMBOS = combos()
_COMBO_INDEX = {
    (int(first), int(second)): index
    for index, (first, second) in enumerate(_COMBOS)
}
_SUIT_PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _card_permutation_map(permutation: tuple[int, ...]) -> np.ndarray:
    mapped = np.empty(NUM_COMBOS, dtype=np.int64)
    for index, (first, second) in enumerate(_COMBOS):
        a = int(first) // 4 * 4 + permutation[int(first) % 4]
        b = int(second) // 4 * 4 + permutation[int(second) % 4]
        mapped[index] = _COMBO_INDEX[tuple(sorted((a, b)))]
    return mapped


_COMBO_PERMUTATIONS = {
    permutation: _card_permutation_map(permutation)
    for permutation in _SUIT_PERMUTATIONS
}


def _canonical_public_board(board: tuple[int, ...]) -> tuple[tuple[int, ...], np.ndarray]:
    """Canonical board plus actual-combo -> canonical-combo index map."""
    best_board: tuple[int, ...] | None = None
    best_permutation: tuple[int, ...] | None = None
    for permutation in _SUIT_PERMUTATIONS:
        mapped_flop = sorted(
            int(card) // 4 * 4 + permutation[int(card) % 4]
            for card in board[:3]
        )
        mapped_later = [
            int(card) // 4 * 4 + permutation[int(card) % 4]
            for card in board[3:]
        ]
        candidate = tuple(mapped_flop + mapped_later)
        if best_board is None or candidate < best_board:
            best_board = candidate
            best_permutation = permutation
    assert best_board is not None and best_permutation is not None
    return best_board, _COMBO_PERMUTATIONS[best_permutation]


def _preflop_strengths() -> np.ndarray:
    first_rank = _COMBOS[:, 0] // 4
    second_rank = _COMBOS[:, 1] // 4
    high = np.maximum(first_rank, second_rank).astype(np.float64) / 12.0
    low = np.minimum(first_rank, second_rank).astype(np.float64) / 12.0
    pair = first_rank == second_rank
    suited = _COMBOS[:, 0] % 4 == _COMBOS[:, 1] % 4
    gap = np.abs(first_rank - second_rank)
    connected = np.maximum(0.0, 1.0 - gap.astype(np.float64) / 5.0)
    return (
        0.48 * high
        + 0.24 * low
        + 0.20 * pair.astype(np.float64)
        + 0.05 * suited.astype(np.float64)
        + 0.03 * connected
    )


_PREFLOP_STRENGTH = _preflop_strengths()
_TOP_10 = _PREFLOP_STRENGTH >= np.quantile(_PREFLOP_STRENGTH, 0.90)
_TOP_35 = _PREFLOP_STRENGTH >= np.quantile(_PREFLOP_STRENGTH, 0.65)
_BOTTOM_20 = _PREFLOP_STRENGTH <= np.quantile(_PREFLOP_STRENGTH, 0.20)
_RANGE_WEIGHTS = np.stack(
    [
        np.ones(NUM_COMBOS, dtype=np.float32),
        _TOP_10.astype(np.float32),
        _TOP_35.astype(np.float32),
        (_TOP_10 | _BOTTOM_20).astype(np.float32),
    ],
    axis=1,
)


def _board_connectivity(ranks: np.ndarray) -> float:
    unique = set(int(rank) for rank in ranks)
    if 12 in unique:
        unique.add(-1)  # ace also participates in the wheel
    best = 0
    for low in range(-1, 9):
        best = max(best, sum(rank in unique for rank in range(low, low + 5)))
    return best / 5.0


def _equity_against_ranges(scores: np.ndarray) -> np.ndarray:
    """Blocker-corrected equity against all static ranges in O(R*N log N)."""
    valid = scores >= 0
    valid_indices = np.flatnonzero(valid)
    values = scores[valid]
    weights = _RANGE_WEIGHTS[valid]
    count, ranges = weights.shape
    result = np.full((NUM_COMBOS, ranges), -1.0, dtype=np.float32)
    if count <= 1:
        return result

    order = np.argsort(values, kind="stable")
    sorted_scores = values[order]
    sorted_weights = weights[order]
    prefix = np.vstack(
        [
            np.zeros((1, ranges), dtype=np.float32),
            np.cumsum(sorted_weights, axis=0),
        ]
    )
    left = np.searchsorted(sorted_scores, values, side="left")
    right = np.searchsorted(sorted_scores, values, side="right")
    wins = prefix[left]
    ties = prefix[right] - prefix[left] - weights
    opponents = prefix[-1] - weights

    blocked_wins = np.zeros_like(wins)
    blocked_ties = np.zeros_like(ties)
    blocked_opponents = np.zeros_like(opponents)
    position_of = np.full(NUM_COMBOS, -1, dtype=np.int64)
    position_of[valid_indices] = np.arange(count)
    for card in range(52):
        member_indices = np.flatnonzero(valid & CARD_IN_COMBO[card])
        if member_indices.size <= 1:
            continue
        positions = position_of[member_indices]
        member_scores = values[positions]
        member_weights = weights[positions]
        member_order = np.argsort(member_scores, kind="stable")
        ordered_scores = member_scores[member_order]
        ordered_weights = member_weights[member_order]
        member_prefix = np.vstack(
            [
                np.zeros((1, ranges), dtype=np.float32),
                np.cumsum(ordered_weights, axis=0),
            ]
        )
        member_left = np.searchsorted(ordered_scores, member_scores, side="left")
        member_right = np.searchsorted(ordered_scores, member_scores, side="right")
        blocked_wins[positions] += member_prefix[member_left]
        blocked_ties[positions] += (
            member_prefix[member_right] - member_prefix[member_left] - member_weights
        )
        blocked_opponents[positions] += member_prefix[-1] - member_weights

    compatible = opponents - blocked_opponents
    equity = np.divide(
        wins - blocked_wins + 0.5 * (ties - blocked_ties),
        np.maximum(compatible, 1e-12),
        out=np.zeros_like(wins),
        where=compatible > 1e-12,
    )
    result[valid_indices] = equity
    return result


def river_ochs_features(board: tuple[int, ...]) -> np.ndarray:
    """Opponent-aware river features for every private combo."""
    scores = score_all_combos(board)
    valid = scores >= 0
    equities = _equity_against_ranges(scores)

    board_array = np.asarray(board, dtype=np.int64)
    board_ranks = board_array // 4
    board_suits = board_array % 4
    suit_counts = np.bincount(board_suits, minlength=4)
    dominant_suit = int(np.argmax(suit_counts))
    unique_ranks = len(np.unique(board_ranks))
    paired = 1.0 - unique_ranks / 5.0
    suit_density = float(suit_counts[dominant_suit]) / 5.0
    connectivity = _board_connectivity(board_ranks)

    combo_ranks = _COMBOS // 4
    combo_suits = _COMBOS % 4
    suited_rank = np.where(
        combo_suits == dominant_suit,
        (combo_ranks + 1) / 13.0,
        0.0,
    ).max(axis=1)
    pair_blocker = np.isin(combo_ranks, board_ranks).sum(axis=1) / 2.0
    rank_distance = np.abs(combo_ranks[:, :, None] - board_ranks[None, None, :])
    straight_blocker = (rank_distance <= 1).any(axis=2).sum(axis=1) / 2.0
    valid_scores = scores[valid]
    nut_score = valid_scores.max()
    near_nut_score = np.quantile(valid_scores, 0.95)
    nut_hand = (scores == nut_score).astype(np.float32)
    near_nut = (scores >= near_nut_score).astype(np.float32)

    features = np.column_stack(
        [
            equities,
            nut_hand,
            near_nut,
            suited_rank,
            pair_blocker,
            straight_blocker,
            np.full(NUM_COMBOS, paired),
            np.full(NUM_COMBOS, suit_density),
            np.full(NUM_COMBOS, connectivity),
        ]
    ).astype(np.float32)
    features[~valid] = -1.0
    return features


def _stable_seed(board: tuple[int, ...], stage: int, seed: int) -> int:
    value = (2166136261 ^ int(seed) ^ (stage * 16777619)) & 0xFFFFFFFF
    for card in board:
        value ^= int(card) + 1
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def _common_future_cards(
    board: tuple[int, ...],
    count: int,
    stage: int,
    seed: int,
) -> list[int]:
    available = [card for card in range(52) if card not in board]
    if count <= 0 or count >= len(available):
        return available
    base = _stable_seed(board, stage, seed)
    ranked = sorted(available, key=lambda card: (_mix32(base ^ (card * 0x9E3779B9)), card))
    return ranked[:count]


def _transition_cdf_landmarks(
    next_rows: Iterable[np.ndarray],
    cluster_count: int,
    valid: np.ndarray,
    landmarks: int,
) -> np.ndarray:
    counts = np.zeros((NUM_COMBOS, cluster_count), dtype=np.float32)
    totals = np.zeros(NUM_COMBOS, dtype=np.float32)
    indices = np.arange(NUM_COMBOS)
    for row in next_rows:
        usable = valid & (row >= 0)
        np.add.at(counts, (indices[usable], row[usable]), 1.0)
        totals[usable] += 1.0
    seen = totals > 0
    counts[seen] /= totals[seen, None]
    cdf = np.cumsum(counts, axis=1)
    positions = np.linspace(0, cluster_count - 1, min(landmarks, cluster_count))
    positions = np.unique(np.rint(positions).astype(np.int64))
    features = cdf[:, positions]
    features[~seen] = -1.0
    return features


def _predict_l1(features: np.ndarray, centroids: np.ndarray, chunk: int = 2048) -> np.ndarray:
    result = np.full(features.shape[0], -1, dtype=np.int32)
    valid = features[:, 0] >= 0
    valid_indices = np.flatnonzero(valid)
    for start in range(0, valid_indices.size, chunk):
        indices = valid_indices[start : start + chunk]
        distances = np.abs(
            features[indices, None, :] - centroids[None, :, :]
        ).sum(axis=2)
        result[indices] = distances.argmin(axis=1).astype(np.int32)
    return result


def _fit_l1_centroids(
    rows: np.ndarray,
    clusters: int,
    iterations: int,
    seed: int,
    max_rows: int = 60_000,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    rows = rows[rows[:, 0] >= 0]
    if rows.shape[0] < clusters:
        raise ValueError(f"need at least {clusters} fitting rows, got {rows.shape[0]}")
    rng = np.random.default_rng(seed)
    if rows.shape[0] > max_rows:
        rows = rows[rng.choice(rows.shape[0], size=max_rows, replace=False)]

    strength_order = np.argsort(rows.mean(axis=1), kind="stable")
    seeds = np.linspace(0, len(strength_order) - 1, clusters)
    centroids = rows[strength_order[np.rint(seeds).astype(np.int64)]].copy()
    for _ in range(max(1, iterations)):
        assignments = _predict_l1(rows, centroids)
        updated = centroids.copy()
        for cluster in range(clusters):
            members = rows[assignments == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centroids, rtol=0.0, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    return centroids


class PotentialAwareBuckets:
    """Fitted recursive abstraction used by ``DealSampler``."""

    VERSION = 1

    def __init__(
        self,
        flop_clusters: int,
        turn_clusters: int,
        river_clusters: int,
        flop_transition_samples: int = 8,
        turn_transition_samples: int = 16,
        flop_landmarks: int = 24,
        turn_landmarks: int = 16,
        seed: int = 0,
    ) -> None:
        self.flop_clusters = int(flop_clusters)
        self.turn_clusters = int(turn_clusters)
        self.river_clusters = int(river_clusters)
        self.flop_transition_samples = int(flop_transition_samples)
        self.turn_transition_samples = int(turn_transition_samples)
        self.flop_landmarks = int(flop_landmarks)
        self.turn_landmarks = int(turn_landmarks)
        self.seed = int(seed)
        self.river_centroids: np.ndarray | None = None
        self.turn_centroids: np.ndarray | None = None
        self.flop_centroids: np.ndarray | None = None
        self._cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    @property
    def fitted(self) -> bool:
        return (
            self.river_centroids is not None
            and self.turn_centroids is not None
            and self.flop_centroids is not None
        )

    def state(self) -> dict:
        return {
            "version": self.VERSION,
            "river_centroids": (
                self.river_centroids.tolist() if self.river_centroids is not None else []
            ),
            "turn_centroids": (
                self.turn_centroids.tolist() if self.turn_centroids is not None else []
            ),
            "flop_centroids": (
                self.flop_centroids.tolist() if self.flop_centroids is not None else []
            ),
        }

    def load_state(self, state: dict) -> None:
        if int(state.get("version", 0)) != self.VERSION:
            raise ValueError(f"unsupported potential abstraction version: {state.get('version')}")
        for name in ("river_centroids", "turn_centroids", "flop_centroids"):
            values = state.get(name, [])
            setattr(
                self,
                name,
                np.asarray(values, dtype=np.float32) if len(values) else None,
            )
        self._cache.clear()

    def fit(self, boards_per_street: int = 12, iterations: int = 12) -> None:
        rng = np.random.default_rng(self.seed)

        river_rows = []
        for _ in range(max(4, boards_per_street * 2)):
            board = tuple(int(card) for card in rng.choice(52, size=5, replace=False))
            canonical, _ = _canonical_public_board(board)
            features = river_ochs_features(canonical)
            river_rows.append(features[features[:, 0] >= 0])
        self.river_centroids = _fit_l1_centroids(
            np.concatenate(river_rows),
            self.river_clusters,
            iterations,
            self.seed + 101,
        )
        self.river_centroids = self.river_centroids[
            np.argsort(self.river_centroids[:, 0], kind="stable")
        ]
        self._cache.clear()

        turn_rows = []
        for _ in range(max(4, boards_per_street)):
            board = tuple(int(card) for card in rng.choice(52, size=4, replace=False))
            canonical, _ = _canonical_public_board(board)
            features = self._turn_features_canonical(canonical)
            turn_rows.append(features[features[:, 0] >= 0])
        self.turn_centroids = _fit_l1_centroids(
            np.concatenate(turn_rows),
            self.turn_clusters,
            iterations,
            self.seed + 211,
        )
        self.turn_centroids = self.turn_centroids[
            np.argsort(-self.turn_centroids.mean(axis=1), kind="stable")
        ]
        self._cache.clear()

        flop_rows = []
        for _ in range(max(4, boards_per_street)):
            board = tuple(int(card) for card in rng.choice(52, size=3, replace=False))
            canonical, _ = _canonical_public_board(board)
            features = self._flop_features_canonical(canonical)
            flop_rows.append(features[features[:, 0] >= 0])
        self.flop_centroids = _fit_l1_centroids(
            np.concatenate(flop_rows),
            self.flop_clusters,
            iterations,
            self.seed + 307,
        )
        self.flop_centroids = self.flop_centroids[
            np.argsort(-self.flop_centroids.mean(axis=1), kind="stable")
        ]
        self._cache.clear()

    def bucket_row(self, board: tuple[int, ...], street: int) -> np.ndarray:
        required = {
            1: self.flop_centroids,
            2: self.turn_centroids,
            3: self.river_centroids,
        }.get(street)
        if required is None:
            raise RuntimeError(f"potential-aware street {street} is not fitted")
        board = tuple(int(card) for card in board)
        expected = (0, 3, 4, 5)[street]
        if len(board) != expected or street == 0:
            raise ValueError(f"street {street} expects {expected} public cards, got {len(board)}")
        key = (street, board)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()

        canonical, combo_map = _canonical_public_board(board)
        if street == 3:
            canonical_row = _predict_l1(
                river_ochs_features(canonical),
                self.river_centroids,
            )
        elif street == 2:
            canonical_row = _predict_l1(
                self._turn_features_canonical(canonical),
                self.turn_centroids,
            )
        else:
            canonical_row = _predict_l1(
                self._flop_features_canonical(canonical),
                self.flop_centroids,
            )
        result = canonical_row[combo_map]
        if len(self._cache) >= 256:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result.copy()
        return result

    def _turn_features_canonical(self, board: tuple[int, ...]) -> np.ndarray:
        valid = np.ones(NUM_COMBOS, dtype=bool)
        for card in board:
            valid &= ~CARD_IN_COMBO[card]
        rivers = _common_future_cards(
            board,
            self.turn_transition_samples,
            stage=2,
            seed=self.seed,
        )
        rows = (self.bucket_row(board + (river,), 3) for river in rivers)
        return _transition_cdf_landmarks(
            rows,
            self.river_clusters,
            valid,
            self.turn_landmarks,
        )

    def _flop_features_canonical(self, board: tuple[int, ...]) -> np.ndarray:
        valid = np.ones(NUM_COMBOS, dtype=bool)
        for card in board:
            valid &= ~CARD_IN_COMBO[card]
        turns = _common_future_cards(
            board,
            self.flop_transition_samples,
            stage=1,
            seed=self.seed,
        )
        rows = (self.bucket_row(board + (turn,), 2) for turn in turns)
        return _transition_cdf_landmarks(
            rows,
            self.turn_clusters,
            valid,
            self.flop_landmarks,
        )
