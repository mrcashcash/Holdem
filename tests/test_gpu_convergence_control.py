"""Permanent regression control: the fixed-river subgame must converge.

This is the test that would have caught both 2026-07-18 solver bugs (opponent
-node sigma double-application; invalid-combo leakage): on a tiny fixed-board
game, CFR must drive bucket-bound exploitability to ~0, and a fixed profile
evaluated against itself must be exactly zero-sum. Slow (~3-5 min) but
non-negotiable — run before touching solver code.
"""

import random
import unittest

import numpy as np
import torch

from backend.search.gpu_subgame import FixedBoardSampler
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.exploit import _fixed_policy_value, average_strategy_tensor, cfr_br_exploitability
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

BOARD = (2, 7, 24, 33, 50)


def make_solver(seed: int = 23) -> VectorCFR:
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(0.75,), max_raises_per_street=2, stack_bb=20.0
    )
    tree = BettingTree(config, start_street=3, start_pot=8.0, start_stacks=(16.0, 16.0))
    sampler = FixedBoardSampler(DealSampler(river_buckets=20), BOARD)
    return VectorCFR(tree, sampler, device="cpu", seed=seed, averaging_delay=100)


class ConvergenceControlTests(unittest.TestCase):
    def test_fixed_river_control_converges_to_equilibrium(self) -> None:
        solver = make_solver()
        solver.run(2500)
        exploitability = cfr_br_exploitability(solver, br_iterations=150, eval_boards=3, seed=888)
        self.assertLess(exploitability, 25.0, "solver no longer converges — check cfr.py value recursion")

    def test_self_play_is_exactly_zero_sum(self) -> None:
        solver = make_solver(seed=31)
        solver.run(300)
        policy = average_strategy_tensor(solver)
        deal = solver.sampler.sample(random.Random(0))
        valid = torch.tensor(deal.valid).float()
        pairs = float((valid * solver._opponent_mass(valid.unsqueeze(0)).squeeze(0)).sum())
        v0 = _fixed_policy_value(solver, deal, 0, policy, policy)
        v1 = _fixed_policy_value(solver, deal, 1, policy, policy)
        self.assertLess(abs(v0 + v1) / pairs * 1000, 0.5, "zero-sum violated — value evaluator is biased")


if __name__ == "__main__":
    unittest.main()
