"""Phase 2 acceptance tests for the card and action abstraction."""

import random
import unittest

import numpy as np

from backend.abstraction import canonical_key, preflop_class, pseudo_harmonic_weights
from backend.abstraction.actions import ALL_IN, CHECK_CALL, FOLD, ActionAbstraction
from backend.abstraction.buckets import AbstractionConfig, CardAbstraction, WassersteinKMeans, wasserstein
from backend.abstraction.equity import equity_histogram, river_equity


def card(rank: int, suit: int) -> int:
    """rank 2..14, suit 0..3 -> compact id."""
    return (rank - 2) * 4 + suit


class CanonicalizationTests(unittest.TestCase):
    def test_suit_permutations_share_a_key(self) -> None:
        rng = random.Random(1)
        for _ in range(200):
            deal = rng.sample(range(52), 5)
            hole, board = tuple(deal[:2]), tuple(deal[2:])
            permutation = list(range(4))
            rng.shuffle(permutation)
            mapped_hole = tuple((c // 4) * 4 + permutation[c % 4] for c in hole)
            mapped_board = tuple((c // 4) * 4 + permutation[c % 4] for c in board)
            self.assertEqual(canonical_key(hole, board), canonical_key(mapped_hole, mapped_board))

    def test_hole_and_flop_order_invariance(self) -> None:
        hole = (card(14, 0), card(13, 1))
        flop = (card(9, 2), card(5, 3), card(2, 0))
        base = canonical_key(hole, flop)
        self.assertEqual(base, canonical_key((hole[1], hole[0]), flop))
        self.assertEqual(base, canonical_key(hole, (flop[2], flop[0], flop[1])))

    def test_distinct_situations_differ(self) -> None:
        board = (card(9, 2), card(5, 3), card(2, 0))
        self.assertNotEqual(
            canonical_key((card(14, 0), card(14, 1)), board),
            canonical_key((card(14, 0), card(13, 1)), board),
        )

    def test_turn_and_river_keep_street_identity(self) -> None:
        hole = (card(14, 0), card(13, 0))
        first = canonical_key(hole, (card(9, 1), card(5, 2), card(2, 3), card(11, 1), card(3, 2)))
        second = canonical_key(hole, (card(9, 1), card(5, 2), card(2, 3), card(3, 2), card(11, 1)))
        self.assertNotEqual(first, second)


class PreflopClassTests(unittest.TestCase):
    def test_exactly_169_classes(self) -> None:
        classes = set()
        for first in range(52):
            for second in range(first + 1, 52):
                classes.add(preflop_class((first, second)))
        self.assertEqual(len(classes), 169)

    def test_suited_and_offsuit_separate(self) -> None:
        suited = preflop_class((card(14, 0), card(13, 0)))
        offsuit = preflop_class((card(14, 0), card(13, 1)))
        self.assertNotEqual(suited, offsuit)

    def test_pairs_map_to_diagonal(self) -> None:
        aces_one = preflop_class((card(14, 0), card(14, 1)))
        aces_two = preflop_class((card(14, 2), card(14, 3)))
        self.assertEqual(aces_one, aces_two)


class EquityTests(unittest.TestCase):
    def test_river_nuts_have_top_equity(self) -> None:
        # Royal flush holding on an unpaired board.
        hole = (card(14, 0), card(13, 0))
        board = (card(12, 0), card(11, 0), card(10, 0), card(3, 1), card(7, 2))
        self.assertGreater(river_equity(hole, board), 0.99)

    def test_river_air_has_low_equity(self) -> None:
        hole = (card(2, 1), card(3, 2))
        board = (card(12, 0), card(11, 0), card(10, 0), card(9, 3), card(7, 2))
        self.assertLess(river_equity(hole, board), 0.25)

    def test_histogram_separates_made_hands_from_draws(self) -> None:
        # Flush draw (drawy: mass at both ends) vs middle set (made: right mass).
        board = (card(12, 0), card(7, 0), card(2, 1))
        draw = equity_histogram((card(14, 0), card(5, 0)), board, seed=3)
        made = equity_histogram((card(7, 2), card(7, 3)), board, seed=3)
        self.assertGreater(wasserstein(draw, made), 0.4)

    def test_histogram_is_normalized(self) -> None:
        board = (card(12, 0), card(7, 0), card(2, 1))
        histogram = equity_histogram((card(14, 1), card(9, 2)), board, seed=5)
        self.assertAlmostEqual(float(histogram.sum()), 1.0, places=6)


class KMeansTests(unittest.TestCase):
    def test_recovers_separated_clusters(self) -> None:
        rng = np.random.default_rng(0)
        left = np.zeros((50, 8))
        left[:, 0] = 1.0
        right = np.zeros((50, 8))
        right[:, 7] = 1.0
        noise = rng.uniform(0, 0.02, size=(100, 8))
        data = np.vstack([left, right]) + noise
        data /= data.sum(axis=1, keepdims=True)

        model = WassersteinKMeans(2, seed=1).fit(data)

        first = {model.predict_one(row) for row in data[:50]}
        second = {model.predict_one(row) for row in data[50:]}
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first, second)


class CardAbstractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = AbstractionConfig(
            flop_buckets=20,
            turn_buckets=20,
            river_buckets=10,
            fit_samples_per_street=300,
            flop_scenarios=24,
            opponents_per_scenario=16,
            seed=2,
        )
        cls.abstraction = CardAbstraction(config=config).fit()

    def test_preflop_needs_no_fit(self) -> None:
        self.assertEqual(
            self.abstraction.bucket((card(14, 0), card(14, 1)), ()),
            preflop_class((card(14, 0), card(14, 1))),
        )

    def test_buckets_are_stable_and_within_range(self) -> None:
        rng = random.Random(9)
        for board_size, street in ((3, 1), (4, 2), (5, 3)):
            deal = rng.sample(range(52), 2 + board_size)
            hole, board = (deal[0], deal[1]), tuple(deal[2:])
            first = self.abstraction.bucket(hole, board)
            second = self.abstraction.bucket(hole, board)
            self.assertEqual(first, second)
            self.assertTrue(0 <= first < self.abstraction.bucket_count(street))

    def test_suit_isomorphic_hands_share_buckets(self) -> None:
        board = (card(9, 2), card(5, 3), card(2, 0))
        mapped_board = (card(9, 3), card(5, 2), card(2, 1))
        original = self.abstraction.bucket((card(14, 0), card(13, 1)), board)
        mapped = self.abstraction.bucket((card(14, 1), card(13, 0)), mapped_board)
        self.assertEqual(original, mapped)

    def test_river_nuts_and_air_separate(self) -> None:
        board = (card(12, 0), card(11, 0), card(10, 0), card(3, 1), card(7, 2))
        nuts = self.abstraction.bucket((card(14, 0), card(13, 0)), board)
        air = self.abstraction.bucket((card(2, 1), card(4, 3)), board)
        self.assertGreater(nuts, air)

    def test_save_load_round_trip(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "abstraction.npz"
            self.abstraction.save(path)
            loaded = CardAbstraction.load(path)
            board = (card(9, 2), card(5, 3), card(2, 0))
            self.assertEqual(
                loaded.bucket((card(14, 0), card(13, 1)), board),
                self.abstraction.bucket((card(14, 0), card(13, 1)), board),
            )


class ActionAbstractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = ActionAbstraction()

    def test_facing_bet_offers_fold(self) -> None:
        menu = self.actions.menu(street=1, pot=10, to_call=5, stack_behind=100, raises_this_street=1)
        self.assertIn(FOLD, menu)
        self.assertIn(CHECK_CALL, menu)
        self.assertIn(ALL_IN, menu)

    def test_unraised_node_has_no_fold(self) -> None:
        menu = self.actions.menu(street=1, pot=10, to_call=0, stack_behind=100, raises_this_street=0)
        self.assertNotIn(FOLD, menu)

    def test_raise_cap_strips_raises(self) -> None:
        menu = self.actions.menu(street=1, pot=10, to_call=5, stack_behind=100, raises_this_street=4)
        self.assertEqual([action for action in menu if action >= 3], [])
        self.assertIn(ALL_IN, menu)

    def test_short_stack_menu_collapses_to_fold_call(self) -> None:
        menu = self.actions.menu(street=1, pot=10, to_call=20, stack_behind=15, raises_this_street=0)
        self.assertEqual(set(menu), {FOLD, CHECK_CALL})

    def test_raise_amount_uses_pot_fraction(self) -> None:
        amount = self.actions.raise_amount(3, street=1, pot=10, to_call=5)
        self.assertAlmostEqual(amount, 0.33 * 15)

    def test_pseudo_harmonic_endpoints_and_midpoint(self) -> None:
        at_lower, _ = pseudo_harmonic_weights(0.5, 0.5, 1.0)
        self.assertAlmostEqual(at_lower, 1.0)
        _, at_upper = pseudo_harmonic_weights(1.0, 0.5, 1.0)
        self.assertAlmostEqual(at_upper, 1.0)
        lower_weight, upper_weight = pseudo_harmonic_weights(0.75, 0.5, 1.0)
        self.assertTrue(0.0 < lower_weight < 1.0)
        self.assertAlmostEqual(lower_weight + upper_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
