"""Distribution-aware bucketing: draws separate from air, and serving buckets
match training buckets exactly (a mismatch would make the agent look up the
wrong strategy)."""

import random
import unittest

import numpy as np

from backend.solver.gpu.deals import NUM_COMBOS, DealSampler, combos

_COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(combos())}


def card(rank: int, suit: int) -> int:
    return (rank - 2) * 4 + suit


def make_sampler() -> DealSampler:
    s = DealSampler(flop_buckets=30, turn_buckets=30, river_buckets=30, flop_samples=12, turn_samples=12, distributional=True, std_bins=4)
    s.fit_std_edges(samples=120, seed=1)
    return s


class DistributionalBucketTests(unittest.TestCase):
    def test_draw_and_air_get_different_flop_buckets(self) -> None:
        # T-high flop with a flush+straight draw hand vs pure air.
        s = make_sampler()
        flop = (card(12, 0), card(7, 0), card(2, 1))  # Qs 7s 2h
        rng = random.Random(4)
        big_draw = _COMBO_INDEX[tuple(sorted((card(10, 0), card(9, 0))))]  # Ts9s: flush + gutter, high variance
        air = _COMBO_INDEX[tuple(sorted((card(4, 2), card(3, 3))))]        # 4h3d: nothing
        b_draw = s.street_bucket_for_combo(flop, 1, big_draw, rng)
        b_air = s.street_bucket_for_combo(flop, 1, air, rng)
        self.assertIsNotNone(b_draw)
        self.assertIsNotNone(b_air)
        self.assertNotEqual(b_draw, b_air, "draw and air collapsed into the same flop bucket")

    def test_state_roundtrip_preserves_edges(self) -> None:
        s = make_sampler()
        restored = DealSampler.from_state(s.state())
        for street in (1, 2):
            np.testing.assert_allclose(restored._std_edges[street], s._std_edges[street])
        self.assertTrue(restored.distributional)
        self.assertEqual(restored.std_bins, s.std_bins)

    def test_serving_single_matches_training_vectorized(self) -> None:
        # The core consistency guarantee: for a full board, the bucket a combo
        # gets in for_board (training, vectorized) must equal what
        # street_bucket_for_combo (serving, single) returns for the same
        # partial board + rng.
        s = make_sampler()
        board5 = (card(12, 0), card(7, 0), card(2, 1), card(11, 2), card(5, 3))
        # Training path buckets on the flop prefix with a fixed rng.
        flop = board5[:3]
        rng_train = random.Random(99)
        mean, std = s._mean_std_equity(flop, rng_train, s.flop_samples)
        valid = mean >= 0
        train_buckets = np.full(NUM_COMBOS, -1, dtype=np.int64)
        train_buckets[valid] = s._bucket_from_mean_std(mean[valid], std[valid], 1)
        # Serving path for a handful of valid combos with the SAME rng seed.
        combo_array = combos()
        checked = 0
        for idx in range(0, NUM_COMBOS, 137):
            if not valid[idx]:
                continue
            rng_serve = random.Random(99)
            served = s.street_bucket_for_combo(flop, 1, idx, rng_serve)
            self.assertEqual(served, int(train_buckets[idx]), f"combo {tuple(combo_array[idx])} mismatch")
            checked += 1
        self.assertGreater(checked, 3)


if __name__ == "__main__":
    unittest.main()
