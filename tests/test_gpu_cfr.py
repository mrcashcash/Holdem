"""Correctness of the vectorized CFR core (docs/GPU_CFR_PLAN.md §4).

The showdown/fold tensor math is validated against O(n^2) brute force, and
the full loop is validated by solving a short-stack push/fold game whose
equilibrium properties are known (premium hands call jams, trash folds).
"""

import random
import unittest

import numpy as np
import torch

from backend.abstraction.cards import preflop_class
from backend.solver.gpu.cfr import MAX_BUCKETS, VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS, DealSampler, combos
from backend.solver.gpu.tree import ALL_IN, CHECK_CALL, FOLD, BettingTree, GpuActionConfig


def card(rank: int, suit: int) -> int:
    return (rank - 2) * 4 + suit


class TerminalMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = BettingTree(GpuActionConfig())
        cls.solver = VectorCFR(cls.tree, DealSampler(flop_samples=2, turn_samples=2), device="cpu", seed=1)
        cls.deal = cls.solver.sampler.sample(random.Random(7))

    def _brute_force_showdown(self, node: int, opponent_reach: np.ndarray) -> np.ndarray:
        scores = self.deal.river_scores
        combo_array = combos()
        pot = float(self.tree.matched_pot[node])
        expected = np.zeros(NUM_COMBOS)
        for hero in range(NUM_COMBOS):
            if scores[hero] < 0:
                continue
            hero_cards = set(map(int, combo_array[hero]))
            value = 0.0
            for villain in range(NUM_COMBOS):
                if villain == hero or scores[villain] < 0:
                    continue
                if hero_cards & set(map(int, combo_array[villain])):
                    continue
                if scores[hero] > scores[villain]:
                    value += opponent_reach[villain]
                elif scores[hero] < scores[villain]:
                    value -= opponent_reach[villain]
            expected[hero] = pot * value
        return expected

    def test_showdown_values_match_brute_force(self) -> None:
        solver, deal = self.solver, self.deal
        nodes = len(self.tree)
        rng = np.random.default_rng(5)
        reach = torch.zeros((2, nodes, NUM_COMBOS))
        opponent_reach = rng.uniform(0, 1, NUM_COMBOS) * deal.valid
        showdown = int(solver.showdown_nodes[0])
        reach[1, showdown, :] = torch.tensor(opponent_reach, dtype=torch.float32)

        values = torch.zeros((nodes, NUM_COMBOS))
        solver._showdown_values(
            values,
            reach,
            torch.tensor(deal.river_scores, dtype=torch.long),
            torch.tensor(deal.valid),
            player=0,
        )

        expected = self._brute_force_showdown(showdown, opponent_reach)
        np.testing.assert_allclose(values[showdown].numpy(), expected, rtol=1e-4, atol=1e-3)

    def test_fold_values_match_brute_force(self) -> None:
        solver, deal = self.solver, self.deal
        nodes = len(self.tree)
        rng = np.random.default_rng(6)
        opponent_reach = rng.uniform(0, 1, NUM_COMBOS) * deal.valid
        fold_node = int(solver.fold_nodes[0])
        loser = int(self.tree.fold_loser[fold_node])
        amount = float(self.tree.fold_loser_committed[fold_node])
        winner = 1 - loser

        reach = torch.zeros((2, nodes, NUM_COMBOS))
        reach[loser, fold_node, :] = torch.tensor(opponent_reach, dtype=torch.float32)
        values = torch.zeros((nodes, NUM_COMBOS))
        solver._fold_values(values, reach, player=winner)

        combo_array = combos()
        expected = np.zeros(NUM_COMBOS)
        for hero in range(NUM_COMBOS):
            hero_cards = set(map(int, combo_array[hero]))
            mass = 0.0
            for villain in range(NUM_COMBOS):
                if villain != hero and not hero_cards & set(map(int, combo_array[villain])):
                    mass += opponent_reach[villain]
            expected[hero] = amount * mass
        np.testing.assert_allclose(values[fold_node].numpy(), expected, rtol=1e-4, atol=1e-3)


class PushFoldConvergenceTests(unittest.TestCase):
    def test_short_stack_equilibrium_shape(self) -> None:
        config = GpuActionConfig(preflop_fractions=(), postflop_fractions=(), max_raises_per_street=0, stack_bb=4.0)
        tree = BettingTree(config)
        solver = VectorCFR(tree, DealSampler(flop_samples=2, turn_samples=2), device="cpu", seed=3)
        solver.run(300)

        strategy = solver.average_strategy_tables()
        root = tree.root
        jam_node = int(tree.children[root][ALL_IN])
        self.assertGreaterEqual(jam_node, 0)

        aces = preflop_class((card(14, 0), card(14, 1)))
        seven_deuce = preflop_class((card(7, 0), card(2, 1)))

        facing = strategy[jam_node]
        self.assertGreater(facing[aces][CHECK_CALL], 0.8, "AA must call a 4bb jam")
        self.assertGreater(
            facing[seven_deuce][FOLD],
            facing[seven_deuce][CHECK_CALL],
            "72o should fold to a jam more often than call at 4bb",
        )

        # The button must jam some hands and premium hands must act strongest.
        button = strategy[root]
        self.assertGreater(button[aces][ALL_IN] + button[aces][CHECK_CALL], 0.9)
        overall_jam = float(np.mean(strategy[root, :169, ALL_IN]))
        self.assertGreater(overall_jam, 0.1)


if __name__ == "__main__":
    unittest.main()
