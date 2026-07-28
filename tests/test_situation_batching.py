"""Independent-situation batching: B equilibria solved in one pass (P2).

These solves are latency-bound — ~530 dependent GPU ops per iteration, 123x above
the bandwidth floor, ~90% of the card idle. Widening each kernel is therefore
close to free, but adding PROCESSES is not: four datagen workers measured
0.4 rows/s against 1.20 for a single worker, because separate CUDA contexts
time-slice rather than interleave. So the idle GPU can only be used in-process.

`batch_boards` already folds B boards into the combo axis, but those boards share
one regret table (B chance samples of ONE game). `independent_situations` makes
the same axis carry B separate games by giving the tables a batch dimension and
every index a per-situation offset.

The dangerous failure mode is silent LEAKAGE — situations bleeding into each
other through a shared row — which would look like a slightly-off strategy rather
than an error. That gets a dedicated test.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from backend.search.exact_river import (
    RIVER_FRACTIONS,
    RIVER_RAISE_CAP,
    ExactRiverSampler,
)
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS
from backend.solver.gpu.tree import BettingRootState, BettingTree, GpuActionConfig

BOARDS = [(0, 17, 30, 43, 8), (1, 18, 31, 44, 9), (2, 19, 32, 45, 10)]


class RoundRobinSampler:
    """Hands deal i to situation i: `run` calls sample() batch_boards times."""

    def __init__(self, boards) -> None:
        self.subs = [ExactRiverSampler(board) for board in boards]
        self.index = 0

    def bucket_counts(self):
        return (1, 1, 1, NUM_COMBOS)

    def sample(self, rng):
        deal = self.subs[self.index % len(self.subs)].sample(rng)
        self.index += 1
        return deal


def _tree():
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=RIVER_FRACTIONS,
        max_raises_per_street=RIVER_RAISE_CAP, stack_bb=100.0,
    )
    root = BettingRootState(
        street=3, to_act=1, committed=(10.0, 10.0), street_commit=(0.0, 0.0),
        stacks=(90.0, 90.0), acted=(False, False), raises=0, last_increment=1.0,
    )
    return BettingTree(config, root_state=root)


def _ranges(seed: int = 0):
    generator = np.random.default_rng(seed)
    out = []
    for board in BOARDS:
        live = np.ones(NUM_COMBOS, dtype=bool)
        for card in board:
            live &= ~CARD_IN_COMBO[card]
        first = generator.random(NUM_COMBOS) * live
        second = generator.random(NUM_COMBOS) * live
        out.append(np.stack([first / first.sum(), second / second.sum()]).astype(np.float32))
    return out


def _batched(ranges, iterations=6, device="cpu"):
    tree = _tree()
    solver = VectorCFR(
        tree, RoundRobinSampler(BOARDS), device=device, seed=5, averaging_delay=2,
        batch_boards=len(BOARDS), independent_situations=True,
    )
    solver.root_reach = torch.as_tensor(np.concatenate(ranges, axis=1), dtype=torch.float32)
    solver.run(iterations)
    return solver


class SituationBatchingTests(unittest.TestCase):
    def test_matches_independent_solves(self) -> None:
        """Tolerance, not equality: the showdown kernel switches between a
        channel-trick and a per-card loop on a byte threshold that depends on
        batch size. The two are mathematically equal but reduce in a different
        float32 order, so exact equality is not achievable across batch widths."""
        ranges = _ranges()
        batched = _batched(ranges)
        for index, board in enumerate(BOARDS):
            solo = VectorCFR(
                _tree(), ExactRiverSampler(board), device="cpu", seed=5, averaging_delay=2
            )
            solo.root_reach = torch.as_tensor(ranges[index], dtype=torch.float32)
            solo.run(6)
            delta = float(
                (solo.average_strategy_tensor() - batched.average_strategy_tensor(index))
                .abs().max()
            )
            self.assertLess(delta, 1e-4, f"situation {index} diverged by {delta:.2e}")

    def test_situations_do_not_leak_into_each_other(self) -> None:
        """The failure mode that would look like a mildly wrong strategy.

        Perturbing ONE situation's range must leave the others untouched. If any
        table row were shared, this would move them.
        """
        base = _ranges()
        perturbed = [r.copy() for r in base]
        generator = np.random.default_rng(99)
        live = perturbed[1][0] > 0
        noise = generator.random(NUM_COMBOS) * live
        perturbed[1] = np.stack([noise / noise.sum(), perturbed[1][1]]).astype(np.float32)

        first = _batched(base)
        second = _batched(perturbed)
        for index in (0, 2):
            torch.testing.assert_close(
                first.average_strategy_tensor(index),
                second.average_strategy_tensor(index),
                rtol=0, atol=0,
                msg=f"situation {index} changed when situation 1's range moved",
            )
        moved = float(
            (first.average_strategy_tensor(1) - second.average_strategy_tensor(1)).abs().max()
        )
        self.assertGreater(moved, 1e-6, "the perturbation had no effect; test is vacuous")

    def test_tables_are_sized_per_situation(self) -> None:
        solver = _batched(_ranges(), iterations=1)
        self.assertEqual(solver.situations, len(BOARDS))
        self.assertEqual(
            solver.regrets.shape[0], solver.layout.total_rows * len(BOARDS)
        )

    def test_shared_table_mode_is_untouched(self) -> None:
        """Default behaviour must be byte-identical to before the feature."""
        tree = _tree()
        solver = VectorCFR(
            tree, ExactRiverSampler(BOARDS[0]), device="cpu", seed=5, averaging_delay=2
        )
        self.assertEqual(solver.situations, 1)
        self.assertIsNone(solver.t_situation_offset)
        self.assertEqual(solver.regrets.shape[0], solver.layout.total_rows)


if __name__ == "__main__":
    unittest.main()
