"""Deal sampling and sort-based equity for the GPU CFR (Plan §3)."""

import random
import unittest

import numpy as np

from backend.abstraction.equity import river_equity
from backend.solver.gpu.deals import (
    NUM_COMBOS,
    DealSampler,
    combos,
    equity_from_scores,
    score_all_combos,
)


def card(rank: int, suit: int) -> int:
    return (rank - 2) * 4 + suit


class ScoreEquityTests(unittest.TestCase):
    BOARD = (card(12, 0), card(11, 0), card(10, 0), card(3, 1), card(7, 2))

    def test_colliding_combos_are_invalid(self) -> None:
        scores = score_all_combos(self.BOARD)
        combo_array = combos()
        for index in range(NUM_COMBOS):
            collides = bool(set(map(int, combo_array[index])) & set(self.BOARD))
            self.assertEqual(scores[index] < 0, collides)

    def test_rank_equity_matches_exact_enumeration(self) -> None:
        # equity_from_scores must agree with the exhaustive pairwise
        # river_equity used by the CPU abstraction (same uniform-range def).
        scores = score_all_combos(self.BOARD)
        equity = equity_from_scores(scores)
        combo_array = combos()
        rng = random.Random(3)
        for index in rng.sample(range(NUM_COMBOS), 12):
            if scores[index] < 0:
                continue
            exact = river_equity((int(combo_array[index][0]), int(combo_array[index][1])), self.BOARD)
            self.assertAlmostEqual(float(equity[index]), exact, places=9)

    def test_nuts_have_top_equity(self) -> None:
        scores = score_all_combos(self.BOARD)
        equity = equity_from_scores(scores)
        combo_array = combos()
        royal = next(
            index
            for index in range(NUM_COMBOS)
            if {int(combo_array[index][0]), int(combo_array[index][1])} == {card(14, 0), card(13, 0)}
        )
        self.assertEqual(float(equity[royal]), 1.0)


class DealSamplerTests(unittest.TestCase):
    def test_sampled_deal_shapes_and_ranges(self) -> None:
        sampler = DealSampler(flop_buckets=10, turn_buckets=10, river_buckets=10, flop_samples=4, turn_samples=4)
        deal = sampler.sample(random.Random(1))
        self.assertEqual(deal.buckets.shape, (4, NUM_COMBOS))
        counts = sampler.bucket_counts()
        for street in range(4):
            street_buckets = deal.buckets[street][deal.valid]
            self.assertTrue((street_buckets >= 0).all())
            self.assertTrue((street_buckets < counts[street]).all())
        self.assertEqual(int(deal.valid.sum()), NUM_COMBOS - (52 - 47) * 51 + 10)  # 1081 valid combos

    def test_preflop_buckets_are_lossless_classes(self) -> None:
        sampler = DealSampler()
        deal = sampler.sample(random.Random(2))
        from backend.abstraction.cards import preflop_class

        combo_array = combos()
        for index in random.Random(3).sample(range(NUM_COMBOS), 20):
            if deal.valid[index]:
                expected = preflop_class((int(combo_array[index][0]), int(combo_array[index][1])))
                self.assertEqual(int(deal.buckets[0][index]), expected)

    def test_stronger_hands_get_higher_river_buckets(self) -> None:
        board = (card(12, 0), card(11, 0), card(10, 0), card(3, 1), card(7, 2))
        sampler = DealSampler(river_buckets=10)
        deal = sampler.for_board(board, random.Random(4))
        combo_array = combos()
        royal = next(
            index
            for index in range(NUM_COMBOS)
            if {int(combo_array[index][0]), int(combo_array[index][1])} == {card(14, 0), card(13, 0)}
        )
        air = next(
            index
            for index in range(NUM_COMBOS)
            if {int(combo_array[index][0]), int(combo_array[index][1])} == {card(2, 1), card(4, 3)}
        )
        self.assertGreater(int(deal.buckets[3][royal]), int(deal.buckets[3][air]))
        self.assertEqual(int(deal.buckets[3][royal]), 9)


if __name__ == "__main__":
    unittest.main()
