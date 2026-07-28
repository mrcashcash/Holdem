"""Safe re-solving gadget mechanics (backend/search/safe_subgame.py).

Limiting-case validation on a small fixed-river subgame:
- opt-out worth -inf  => the opponent always enters (gadget reduces to the
  plain re-solve; enter probability ~1 everywhere);
- opt-out worth +inf  => the opponent always opts out (enter ~0);
- a both-frozen evaluation pass is side-effect free and zero-sum consistent.
"""

import random
import unittest

import numpy as np
import torch

from backend.search.gpu_subgame import FixedBoardSampler
from backend.search.safe_subgame import GadgetCFR, opponent_alt_values, opponent_alt_values_br
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS, DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

BOARD = (2, 7, 24, 33, 50)


def build_solver(seed: int = 5) -> tuple[VectorCFR, BettingTree, np.ndarray]:
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(0.75,), max_raises_per_street=2, stack_bb=20.0
    )
    tree = BettingTree(config, start_street=3, start_pot=8.0, start_stacks=(16.0, 16.0))
    sampler = FixedBoardSampler(DealSampler(river_buckets=20), BOARD)
    solver = VectorCFR(tree, sampler, device="cpu", seed=seed, averaging_delay=20)
    deal = sampler.sample(random.Random(0))
    ranges = np.tile(np.where(deal.valid, 1.0, 0.0) / max(1, deal.valid.sum()), (2, 1))
    return solver, tree, ranges


class GadgetMechanicsTests(unittest.TestCase):
    def test_worthless_opt_out_means_always_enter(self) -> None:
        solver, tree, ranges = build_solver()
        alt = torch.full((NUM_COMBOS,), -1e9)
        gadget = GadgetCFR(solver, constrained=1, base_ranges=ranges, alt=alt)
        gadget.run(150)
        enter = gadget.enter_probability().numpy()
        valid = ranges[1] > 0
        self.assertGreater(enter[valid].mean(), 0.99, "worthless opt-out should never be taken")
        # And the solve still produced a usable normalized strategy.
        strategy = solver.average_strategy_tables()
        row_sums = strategy.sum(axis=2)
        self.assertTrue(np.all((row_sums < 1.0 + 1e-4) & (row_sums > -1e-6)))

    def test_infinite_opt_out_means_never_enter(self) -> None:
        solver, _, ranges = build_solver(seed=6)
        alt = torch.full((NUM_COMBOS,), 1e9)
        gadget = GadgetCFR(solver, constrained=1, base_ranges=ranges, alt=alt)
        gadget.run(150)
        enter = gadget.enter_probability().numpy()
        valid = ranges[1] > 0
        self.assertLess(enter[valid].mean(), 0.01, "an infinitely good opt-out should always be taken")

    def test_alt_value_pass_is_side_effect_free(self) -> None:
        solver, tree, ranges = build_solver(seed=7)
        solver.root_reach = torch.tensor(ranges, dtype=torch.float32)
        solver.run(120)
        average = solver.average_strategy_tensor()
        regrets_before = solver.regrets.clone()
        sums_before = solver.strategy_sums.clone()
        solver.root_reach = torch.tensor(ranges, dtype=torch.float32)
        alt0 = opponent_alt_values(solver, average, constrained=0, boards=3)
        alt1 = opponent_alt_values(solver, average, constrained=1, boards=3)
        torch.testing.assert_close(solver.regrets, regrets_before)
        torch.testing.assert_close(solver.strategy_sums, sums_before)
        # Zero-sum consistency of the evaluation: both players' total root
        # values (weighted by their own range mass) must cancel.
        total0 = float((alt0 * torch.tensor(ranges[0], dtype=torch.float32)).sum())
        total1 = float((alt1 * torch.tensor(ranges[1], dtype=torch.float32)).sum())
        self.assertLess(abs(total0 + total1), 1e-2 * max(1.0, abs(total0)))

    def test_br_alt_values_dominate_following_values(self) -> None:
        # v2 safety pricing: a best response to sigma0 must be worth at least
        # as much as obediently following sigma0 (up to solver noise) — and
        # strictly more in aggregate on an unconverged sigma0.
        solver, tree, ranges = build_solver(seed=8)
        solver.root_reach = torch.tensor(ranges, dtype=torch.float32)
        solver.run(60)  # deliberately under-converged: BR should find real gaps
        average = solver.average_strategy_tensor()
        solver.root_reach = torch.tensor(ranges, dtype=torch.float32)
        follow = opponent_alt_values(solver, average, constrained=1, boards=4)
        solver.root_reach = torch.tensor(ranges, dtype=torch.float32)
        br = opponent_alt_values_br(solver, average, constrained=1, br_iterations=120, eval_boards=4)
        valid = torch.tensor(ranges[1]) > 0
        gap = (br - follow)[valid]
        # Aggregate: BR strictly better on an under-converged strategy.
        self.assertGreater(float(gap.mean()), 0.0)
        # Per-combo, allow sampling noise but no systematic domination failure.
        self.assertGreater(float((gap > -0.05 * follow[valid].abs().clamp_min(1.0)).float().mean()), 0.9)


if __name__ == "__main__":
    unittest.main()
