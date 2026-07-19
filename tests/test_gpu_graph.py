"""CUDA-graph replay must be numerically equivalent to eager iteration."""

import random
import unittest

import torch

from backend.search.gpu_subgame import FixedBoardSampler
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

BOARD = (2, 7, 24, 33, 50)


def make_solver(device: str) -> VectorCFR:
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(0.75,), max_raises_per_street=2, stack_bb=20.0
    )
    tree = BettingTree(config, start_street=3, start_pot=8.0, start_stacks=(16.0, 16.0))
    return VectorCFR(
        tree, FixedBoardSampler(DealSampler(river_buckets=20), BOARD), device=device, seed=7, averaging_delay=2
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GraphEquivalenceTests(unittest.TestCase):
    def test_graph_replay_matches_eager(self) -> None:
        from backend.solver.gpu.graph import GraphRunner

        eager = make_solver("cuda")
        eager.run(12)  # eager path with its own rng-consumed deal sequence

        graphed = make_solver("cuda")
        runner = GraphRunner(graphed, warmup=2)
        # Reset state polluted by warmup/capture, then replay 12 iterations
        # with an identical rng so both consume the same deal sequence.
        graphed.regrets.zero_()
        graphed.strategy_sums.zero_()
        graphed.iteration = 0
        graphed.rng = random.Random(7)
        eager_reference = make_solver("cuda")  # fresh rng state identical to `eager`'s start
        del eager_reference
        runner.run(12, random.Random(7))

        torch.cuda.synchronize()
        # Same deal sequence + same math => regrets must match closely
        # (float32 op-ordering differences only).
        eager_solver = make_solver("cuda")
        eager_solver.rng = random.Random(7)
        eager_solver.run(12)
        torch.cuda.synchronize()
        difference = (graphed.regrets - eager_solver.regrets).abs().max().item()
        scale = eager_solver.regrets.abs().max().item()
        self.assertLess(difference, max(scale * 1e-3, 1e-4), f"graph/eager diverged: {difference} vs scale {scale}")


if __name__ == "__main__":
    unittest.main()
