"""Street bucket models: the card abstraction the blueprint solver sees.

Preflop is lossless (169 classes). Flop and turn hands are clustered with
k-means under the 1D Wasserstein metric on future-equity histograms (for 1D
distributions EMD equals the L1 distance between CDFs, so it stays cheap).
River hands are bucketed by exact-equity quantiles. Bucket assignment is
memoised per suit-isomorphic canonical key, so MCCFR pays the equity cost
once per distinct situation.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.abstraction.cards import canonical_key, preflop_class
from backend.abstraction.equity import equity_histogram, river_equity

PREFLOP_BUCKETS = 169


def wasserstein(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(np.cumsum(first - second)).sum())


class WassersteinKMeans:
    """Lloyd's k-means with the 1D Wasserstein (EMD) metric on histograms."""

    def __init__(self, clusters: int, iterations: int = 25, seed: int = 0) -> None:
        self.clusters = clusters
        self.iterations = iterations
        self.seed = seed
        self.centroid_cdfs: np.ndarray | None = None

    def fit(self, histograms: np.ndarray) -> "WassersteinKMeans":
        rng = np.random.default_rng(self.seed)
        cdfs = np.cumsum(histograms, axis=1)
        count = cdfs.shape[0]
        clusters = min(self.clusters, count)

        # k-means++ seeding under the Wasserstein metric.
        centroids = [cdfs[rng.integers(count)]]
        for _ in range(clusters - 1):
            distances = np.min(
                np.stack([np.abs(cdfs - centroid).sum(axis=1) for centroid in centroids]), axis=0
            )
            weights = distances**2
            total = weights.sum()
            probabilities = weights / total if total > 0 else np.full(count, 1.0 / count)
            centroids.append(cdfs[rng.choice(count, p=probabilities)])
        centroid_cdfs = np.stack(centroids)

        for _ in range(self.iterations):
            assignments = self._assign(cdfs, centroid_cdfs)
            updated = centroid_cdfs.copy()
            for cluster in range(clusters):
                members = cdfs[assignments == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
            if np.allclose(updated, centroid_cdfs):
                break
            centroid_cdfs = updated

        self.centroid_cdfs = centroid_cdfs
        return self

    @staticmethod
    def _assign(cdfs: np.ndarray, centroid_cdfs: np.ndarray, chunk: int = 8192) -> np.ndarray:
        assignments = np.empty(cdfs.shape[0], dtype=np.int64)
        for start in range(0, cdfs.shape[0], chunk):
            block = cdfs[start : start + chunk]
            distances = np.abs(block[:, None, :] - centroid_cdfs[None, :, :]).sum(axis=2)
            assignments[start : start + chunk] = distances.argmin(axis=1)
        return assignments

    def predict_one(self, histogram: np.ndarray) -> int:
        cdf = np.cumsum(histogram)
        return int(np.abs(self.centroid_cdfs - cdf).sum(axis=1).argmin())


@dataclass
class AbstractionConfig:
    flop_buckets: int = 200
    turn_buckets: int = 200
    river_buckets: int = 20
    histogram_bins: int = 8
    flop_scenarios: int = 48
    opponents_per_scenario: int = 32
    fit_samples_per_street: int = 20000
    seed: int = 0


@dataclass
class CardAbstraction:
    """Maps (hole, board) to a per-street bucket id, memoised per canonical key."""

    config: AbstractionConfig = field(default_factory=AbstractionConfig)
    flop_model: WassersteinKMeans | None = None
    turn_model: WassersteinKMeans | None = None
    river_quantiles: np.ndarray | None = None
    _cache: dict[tuple[int, ...], int] = field(default_factory=dict)

    # -- fitting ---------------------------------------------------------

    def fit(self, progress: bool = False) -> "CardAbstraction":
        config = self.config
        rng = random.Random(config.seed)

        flop_histograms = self._sample_histograms(rng, board_size=3)
        self.flop_model = WassersteinKMeans(config.flop_buckets, seed=config.seed).fit(flop_histograms)
        if progress:
            print(f"flop model fitted on {len(flop_histograms)} samples")

        turn_histograms = self._sample_histograms(rng, board_size=4)
        self.turn_model = WassersteinKMeans(config.turn_buckets, seed=config.seed + 1).fit(turn_histograms)
        if progress:
            print(f"turn model fitted on {len(turn_histograms)} samples")

        equities = []
        for _ in range(config.fit_samples_per_street):
            hole, board = self._random_deal(rng, 5)
            equities.append(river_equity(hole, board))
        quantiles = np.quantile(np.asarray(equities), np.linspace(0, 1, config.river_buckets + 1)[1:-1])
        self.river_quantiles = quantiles
        if progress:
            print(f"river quantiles fitted on {len(equities)} samples")
        return self

    def _sample_histograms(self, rng: random.Random, board_size: int) -> np.ndarray:
        config = self.config
        histograms = np.empty((config.fit_samples_per_street, config.histogram_bins))
        for index in range(config.fit_samples_per_street):
            hole, board = self._random_deal(rng, board_size)
            histograms[index] = equity_histogram(
                hole,
                board,
                bins=config.histogram_bins,
                scenarios=config.flop_scenarios,
                opponents_per_scenario=config.opponents_per_scenario,
                seed=rng.getrandbits(31),
            )
        return histograms

    @staticmethod
    def _random_deal(rng: random.Random, board_size: int) -> tuple[tuple[int, int], tuple[int, ...]]:
        cards = rng.sample(range(52), 2 + board_size)
        return (cards[0], cards[1]), tuple(cards[2:])

    # -- assignment --------------------------------------------------------

    def is_fitted(self) -> bool:
        return self.flop_model is not None and self.turn_model is not None and self.river_quantiles is not None

    def bucket_count(self, street: int) -> int:
        return (
            PREFLOP_BUCKETS,
            self.config.flop_buckets,
            self.config.turn_buckets,
            self.config.river_buckets,
        )[street]

    def bucket(self, hole: tuple[int, int], board: tuple[int, ...]) -> int:
        if not board:
            return preflop_class(tuple(hole))
        key = canonical_key(tuple(hole), tuple(board))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        config = self.config
        if len(board) == 5:
            equity = river_equity(hole, board)
            value = int(np.searchsorted(self.river_quantiles, equity))
        else:
            histogram = equity_histogram(
                hole,
                board,
                bins=config.histogram_bins,
                scenarios=config.flop_scenarios,
                opponents_per_scenario=config.opponents_per_scenario,
                seed=hash(key) & 0x7FFFFFFF,
            )
            model = self.flop_model if len(board) == 3 else self.turn_model
            value = model.predict_one(histogram)
        self._cache[key] = value
        return value

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            config=json.dumps(self.config.__dict__),
            flop_centroids=self.flop_model.centroid_cdfs,
            turn_centroids=self.turn_model.centroid_cdfs,
            river_quantiles=self.river_quantiles,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CardAbstraction":
        payload = np.load(path, allow_pickle=False)
        config = AbstractionConfig(**json.loads(str(payload["config"])))
        abstraction = cls(config=config)
        abstraction.flop_model = WassersteinKMeans(config.flop_buckets, seed=config.seed)
        abstraction.flop_model.centroid_cdfs = payload["flop_centroids"]
        abstraction.turn_model = WassersteinKMeans(config.turn_buckets, seed=config.seed + 1)
        abstraction.turn_model.centroid_cdfs = payload["turn_centroids"]
        abstraction.river_quantiles = payload["river_quantiles"]
        return abstraction
