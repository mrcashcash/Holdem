"""Histogram-EMD bucketing (roadmap #2): draws separate from air and made
hands, serving buckets match training buckets exactly, and centroids survive
the checkpoint state roundtrip."""

import random
import unittest

import numpy as np

from backend.solver.gpu.deals import NUM_COMBOS, DealSampler, combos

_COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(combos())}


def card(rank: int, suit: int) -> int:
    return (rank - 2) * 4 + suit


def make_sampler() -> DealSampler:
    s = DealSampler(
        flop_buckets=40, turn_buckets=40, river_buckets=20,
        flop_samples=10, turn_samples=8, histogram=True, hist_bins=8,
    )
    s.fit_hist_centroids(boards=12, seed=3, iterations=6)
    return s


class HistogramBucketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sampler = make_sampler()

    def test_draw_made_and_air_separate(self) -> None:
        s = self.sampler
        flop = (card(12, 0), card(7, 0), card(2, 1))  # Qs 7s 2h
        rng = random.Random(4)
        draw = _COMBO_INDEX[tuple(sorted((card(10, 0), card(9, 0))))]   # Ts9s: flush draw + gutter
        made = _COMBO_INDEX[tuple(sorted((card(12, 2), card(12, 3))))]  # QhQd: top set (point mass)
        air = _COMBO_INDEX[tuple(sorted((card(4, 2), card(3, 3))))]     # 4h3d: nothing
        buckets = {}
        for name, combo in (("draw", draw), ("made", made), ("air", air)):
            buckets[name] = s.street_bucket_for_combo(flop, 1, combo, random.Random(9))
            self.assertIsNotNone(buckets[name])
        self.assertNotEqual(buckets["draw"], buckets["air"], "draw collapsed with air")
        self.assertNotEqual(buckets["draw"], buckets["made"], "draw collapsed with made hand")
        del rng

    def test_serving_single_matches_training_vectorized(self) -> None:
        s = self.sampler
        flop = (card(12, 0), card(7, 0), card(2, 1))
        rng_train = random.Random(99)
        hist = s._equity_histograms(flop, rng_train, s.flop_samples, s.hist_bins)
        valid = hist[:, 0] >= 0
        train_buckets = np.full(NUM_COMBOS, -1, dtype=np.int64)
        train_buckets[valid] = s._bucket_from_histogram(hist[valid], 1)
        checked = 0
        for idx in range(0, NUM_COMBOS, 137):
            if not valid[idx]:
                continue
            served = s.street_bucket_for_combo(flop, 1, idx, random.Random(99))
            self.assertEqual(served, int(train_buckets[idx]))
            checked += 1
        self.assertGreater(checked, 3)

    def test_state_roundtrip_preserves_centroids(self) -> None:
        s = self.sampler
        restored = DealSampler.from_state(s.state())
        self.assertTrue(restored.histogram)
        self.assertEqual(restored.hist_bins, s.hist_bins)
        for street in (1, 2):
            np.testing.assert_allclose(restored._hist_centroids[street], s._hist_centroids[street])
        # And the restored sampler produces identical buckets.
        flop = (card(9, 3), card(8, 3), card(3, 0))
        for idx in (0, 400, 900):
            a = s.street_bucket_for_combo(flop, 1, idx, random.Random(5))
            b = restored.street_bucket_for_combo(flop, 1, idx, random.Random(5))
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
