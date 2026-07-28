"""Depth-limited solving correctness (backend/search/depth_limited.py).

Equivalence proof for the horizon plumbing: solving an ``end_street`` tree
whose HORIZON terminals are priced by ShowdownOracle must produce IDENTICAL
regrets and strategy sums to a plain VectorCFR solve of the same tree with
those HORIZON nodes relabeled as true SHOWDOWN terminals — the oracle and the
terminal kernel compute the same quantity, so any difference is a bug in the
reach extraction / value injection path that the CFV net will later flow
through.
"""

import random
import unittest

import numpy as np
import torch

from backend.search.depth_limited import DepthLimitedCFR, ShowdownOracle
from backend.search.gpu_subgame import FixedBoardSampler
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import HORIZON, SHOWDOWN, BettingTree, GpuActionConfig

FLOP = (2, 17, 33)


def build(seed: int):
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(1.0,), max_raises_per_street=2, stack_bb=20.0
    )
    tree = BettingTree(config, start_street=1, start_pot=4.0, start_stacks=(18.0, 18.0), end_street=1)
    sampler = FixedBoardSampler(
        DealSampler(flop_buckets=10, turn_buckets=10, river_buckets=10, flop_samples=4, turn_samples=4), FLOP
    )
    return tree, sampler


class DepthLimitedTests(unittest.TestCase):
    def test_tree_has_horizons_and_allins_still_showdown(self) -> None:
        tree, _ = build(0)
        kinds = tree.kind
        self.assertGreater(int((kinds == HORIZON).sum()), 0, "no horizon nodes built")
        self.assertGreater(int((kinds == SHOWDOWN).sum()), 0, "all-in runouts should stay showdowns")
        # Horizon pots recorded (needed by evaluators).
        horizon_pots = tree.matched_pot[kinds == HORIZON]
        self.assertTrue((horizon_pots > 0).all())

    def test_oracle_solve_equals_relabeled_showdown_solve(self) -> None:
        tree_a, sampler_a = build(0)
        solver_a = DepthLimitedCFR(
            tree_a, sampler_a, device="cpu", seed=5, averaging_delay=10, horizon_evaluator=ShowdownOracle()
        )
        solver_a.run(60)

        tree_b, sampler_b = build(0)
        tree_b.kind = np.where(tree_b.kind == HORIZON, np.int8(SHOWDOWN), tree_b.kind)
        solver_b = VectorCFR(tree_b, sampler_b, device="cpu", seed=5, averaging_delay=10)
        solver_b.run(60)

        torch.testing.assert_close(solver_a.regrets, solver_b.regrets, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(solver_a.strategy_sums, solver_b.strategy_sums, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
