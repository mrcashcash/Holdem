"""Board batching must be mathematically exact (docs/GPU_CFR_PLAN.md).

The sharpest possible check: one _iterate over a batch of two IDENTICAL
deals must produce exactly twice the regret and strategy-sum increments of
one _iterate over the single deal (the scatter is linear in the batch).
"""

import random
import unittest

import torch

from backend.search.gpu_subgame import FixedBoardSampler
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

BOARD = (2, 7, 24, 33, 50)


def make_solver(seed: int) -> VectorCFR:
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(0.75,), max_raises_per_street=2, stack_bb=20.0
    )
    tree = BettingTree(config, start_street=3, start_pot=8.0, start_stacks=(16.0, 16.0))
    return VectorCFR(tree, FixedBoardSampler(DealSampler(river_buckets=20), BOARD), device="cpu", seed=seed)


class BatchingExactnessTests(unittest.TestCase):
    def test_duplicate_batch_doubles_increments_exactly(self) -> None:
        deal = FixedBoardSampler(DealSampler(river_buckets=20), BOARD).sample(random.Random(3))

        single = make_solver(1)
        single.iteration = 1
        single._iterate(deal, traverser=0)
        single._iterate(deal, traverser=1)

        double = make_solver(1)
        double.iteration = 1
        double._iterate([deal, deal], traverser=0)
        double._iterate([deal, deal], traverser=1)

        torch.testing.assert_close(double.regrets, single.regrets * 2.0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(double.strategy_sums, single.strategy_sums * 2.0, rtol=1e-4, atol=1e-4)

    def test_mixed_batch_equals_sum_of_independent_singles(self) -> None:
        # A mini-batch applies ONE strategy to all boards, so the reference is
        # the sum of increments from independent fresh solvers (not sequential
        # iterations, which would update the strategy between boards).
        sampler = FixedBoardSampler(DealSampler(river_buckets=20), BOARD)
        deal_a = sampler.sample(random.Random(5))
        deal_b = sampler.sample(random.Random(6))  # same board, different bucket sampling rng

        first = make_solver(2)
        first.iteration = 1
        first._iterate(deal_a, traverser=0)
        second = make_solver(2)
        second.iteration = 1
        second._iterate(deal_b, traverser=0)

        batched = make_solver(2)
        batched.iteration = 1
        batched._iterate([deal_a, deal_b], traverser=0)

        torch.testing.assert_close(batched.regrets, first.regrets + second.regrets, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            batched.strategy_sums, first.strategy_sums + second.strategy_sums, rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
