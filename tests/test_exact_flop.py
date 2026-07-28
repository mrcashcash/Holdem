"""Exact-card flop sampler for the shallow depths.

20bb is the worst-served depth: LBR beats the 100bb-trained blueprint there by
+130.31 bb/100 [+95.22, +165.40], an interval that clears zero decisively. But a
20bb exact flop-to-river tree is only 5,303 nodes (987 at a 12bb pot), so that
depth can be played with no value network at all — solve every postflop street
exactly to showdown.

Unlike the turn's 48 runouts, a flop has 1,081 turn+river completions, so runouts
are sampled rather than enumerated. These tests pin the properties the solver
relies on, and the affordability guard that keeps this path off deep stacks where
the tree would need 10-27 GiB.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from backend.search.exact_flop import (
    FLOP_FRACTIONS,
    FLOP_RAISE_CAP,
    ExactFlopSampler,
    exact_flop_is_affordable,
)
from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS

BOARD = (0, 17, 30)


class ExactFlopSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sampler = ExactFlopSampler(BOARD)

    def test_rejects_malformed_boards(self) -> None:
        for bad in ((0, 1), (0, 1, 2, 3), (0, 1, 1)):
            with self.assertRaises(ValueError):
                ExactFlopSampler(bad)

    def test_identity_buckets_on_every_postflop_street(self) -> None:
        self.assertEqual(self.sampler.bucket_counts(), (1, NUM_COMBOS, NUM_COMBOS, NUM_COMBOS))
        deal = self.sampler.deal_for_runout(self.sampler.remaining[0], self.sampler.remaining[5])
        valid = deal.valid
        expected = np.arange(NUM_COMBOS)[valid]
        for street in (1, 2, 3):
            np.testing.assert_array_equal(deal.buckets[street][valid], expected)
            self.assertTrue(np.all(deal.buckets[street][~valid] == -1))

    def test_runout_count_is_the_combinatorial_answer(self) -> None:
        # 49 unseen cards -> C(49,2) = 1176 turn+river completions.
        self.assertEqual(len(self.sampler.remaining), 49)
        self.assertEqual(self.sampler.runout_count(), 49 * 48 // 2)

    def test_flop_buckets_are_runout_independent(self) -> None:
        """What lets flop regrets accumulate per combo across runouts."""
        left = self.sampler.deal_for_runout(self.sampler.remaining[0], self.sampler.remaining[1])
        right = self.sampler.deal_for_runout(self.sampler.remaining[10], self.sampler.remaining[11])
        shared = left.valid & right.valid
        self.assertGreater(int(shared.sum()), 900)
        np.testing.assert_array_equal(left.buckets[1][shared], right.buckets[1][shared])

    def test_runout_order_does_not_create_two_deals(self) -> None:
        first, second = self.sampler.remaining[3], self.sampler.remaining[9]
        self.assertIs(
            self.sampler.deal_for_runout(first, second),
            self.sampler.deal_for_runout(second, first),
        )

    def test_board_and_runout_cards_block_the_right_combos(self) -> None:
        turn, river = self.sampler.remaining[2], self.sampler.remaining[7]
        deal = self.sampler.deal_for_runout(turn, river)
        blocked = np.zeros(NUM_COMBOS, dtype=bool)
        for card in (*BOARD, turn, river):
            blocked |= CARD_IN_COMBO[card]
        np.testing.assert_array_equal(deal.valid, ~blocked)
        self.assertEqual(int(deal.valid.sum()), 47 * 46 // 2)

    def test_sampling_covers_many_distinct_runouts(self) -> None:
        rng = random.Random(7)
        seen = {tuple(sorted(self.sampler.sample(rng).board[3:])) for _ in range(300)}
        self.assertGreater(len(seen), 200, "sampling is not exploring the runout space")


class AffordabilityGuardTests(unittest.TestCase):
    """Exact flop resolving must be refused where it would not fit."""

    def test_shallow_is_affordable(self) -> None:
        self.assertTrue(exact_flop_is_affordable(20.0, 6.0))
        self.assertTrue(exact_flop_is_affordable(20.0, 12.0))

    def test_deep_is_refused(self) -> None:
        # 100bb needs ~10.5 GiB of tables and 200bb ~27 GiB.
        self.assertFalse(exact_flop_is_affordable(100.0, 6.0))
        self.assertFalse(exact_flop_is_affordable(200.0, 6.0))


class ExactFlopSolveTests(unittest.TestCase):
    def test_a_20bb_flop_solve_runs_and_normalises(self) -> None:
        import torch

        from backend.solver.gpu.cfr import VectorCFR
        from backend.solver.gpu.tree import (
            DECISION,
            BettingRootState,
            BettingTree,
            GpuActionConfig,
        )

        stack, pot = 20.0, 12.0
        config = GpuActionConfig(
            preflop_fractions=(1.0,), postflop_fractions=FLOP_FRACTIONS,
            max_raises_per_street=FLOP_RAISE_CAP, stack_bb=stack,
        )
        root = BettingRootState(
            street=1, to_act=1, committed=(pot / 2, pot / 2), street_commit=(0.0, 0.0),
            stacks=(stack - pot / 2, stack - pot / 2), acted=(False, False),
            raises=0, last_increment=1.0,
        )
        tree = BettingTree(config, root_state=root)
        self.assertLess(len(tree), 5000, "20bb flop tree unexpectedly large")

        sampler = ExactFlopSampler(BOARD)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        solver = VectorCFR(tree, sampler, device=device, seed=5, averaging_delay=2)
        live = np.ones(NUM_COMBOS, dtype=bool)
        for card in BOARD:
            live &= ~CARD_IN_COMBO[card]
        ranges = np.stack([live / live.sum()] * 2).astype(np.float32)
        solver.root_reach = torch.as_tensor(ranges, device=solver.device)
        solver.run(5)

        strategy = solver.average_strategy_tables()
        self.assertTrue(np.all(np.isfinite(strategy)))
        legal = np.asarray(tree.legal[tree.root], dtype=bool)
        row = strategy[tree.root][live]
        self.assertTrue(np.all(row[:, ~legal] == 0.0))
        np.testing.assert_allclose(row.sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
