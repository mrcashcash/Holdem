"""Forward-pass fusion must be BIT-IDENTICAL to the verified loop (P2.1).

`VectorCFR._iterate` is the kernel proven correct against a Kuhn/Leduc-validated
best response reading 0.0 mbb. Profiling showed its forward pass is a serial
chain of small dependent GPU ops (~530 per iteration at ~26.5 us), 192 of which
per traversal are the per-action x per-player reach loops, so those were fused
into two `index_add_` calls per level.

Fusion is only safe because of two structural facts, both asserted at plan-build
time and exercised here:

* parents sit exactly one level above their children, so reads and writes never
  alias within a level;
* every child has exactly one (parent, action) edge, so no destination index
  repeats -- which is what would otherwise make `index_add_` non-deterministic
  under CUDA atomics, and would also introduce summation-order freedom.

Because each child receives exactly one contribution there is no reordering of
floating-point sums, so the requirement here is EQUALITY, not closeness. A
tolerance-based test would hide exactly the class of bug worth fearing.
"""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from backend.search.exact_turn import ExactTurnSampler
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS, DealSampler
from backend.solver.gpu.tree import BettingRootState, BettingTree, GpuActionConfig

BOARD = (0, 17, 30, 43)


def _turn_solver(fused: bool, device: str, batch_boards: int = 1, seed: int = 7):
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(0.5, 1.0),
        max_raises_per_street=2, stack_bb=100.0,
    )
    root = BettingRootState(
        street=2, to_act=1, committed=(10.0, 10.0), street_commit=(0.0, 0.0),
        stacks=(90.0, 90.0), acted=(False, False), raises=0, last_increment=1.0,
    )
    tree = BettingTree(config, root_state=root)
    solver = VectorCFR(
        tree, ExactTurnSampler(BOARD), device=device, seed=seed,
        averaging_delay=2, batch_boards=batch_boards, fused_forward=fused,
    )
    return solver


def _blueprint_solver(fused: bool, device: str, seed: int = 3):
    """A full four-street tree, so preflop/flop/street-end paths are covered."""
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=(0.75,),
        max_raises_per_street=2, stack_bb=20.0,
    )
    tree = BettingTree(config)
    sampler = DealSampler(flop_buckets=4, turn_buckets=4, river_buckets=4)
    return VectorCFR(
        tree, sampler, device=device, seed=seed,
        averaging_delay=2, fused_forward=fused,
    )


class FusedForwardEquivalenceTests(unittest.TestCase):
    def _assert_identical(self, build, iterations: int, device: str) -> None:
        looped = build(False, device)
        fused = build(True, device)
        # Identical deal stream: both solvers own an rng seeded the same way,
        # and run() consumes it identically.
        looped.run(iterations)
        fused.run(iterations)
        self.assertTrue(
            torch.equal(looped.regrets, fused.regrets),
            f"regrets differ: max |delta| = "
            f"{float((looped.regrets - fused.regrets).abs().max()):.3e}",
        )
        self.assertTrue(
            torch.equal(looped.strategy_sums, fused.strategy_sums),
            f"strategy sums differ: max |delta| = "
            f"{float((looped.strategy_sums - fused.strategy_sums).abs().max()):.3e}",
        )
        # Non-trivial: a test comparing two all-zero tables proves nothing.
        self.assertGreater(float(fused.regrets.abs().max()), 0.0)

    def test_exact_turn_tree_is_bit_identical_on_cuda(self) -> None:
        """Identity buckets make the backward index_add_ collision-free, so CUDA
        is deterministic here and equality is the right bar."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._assert_identical(_turn_solver, iterations=8, device=device)

    def test_full_four_street_tree_is_bit_identical_on_cpu(self) -> None:
        self._assert_identical(_blueprint_solver, iterations=6, device="cpu")

    def test_batched_boards_are_bit_identical_on_cpu(self) -> None:
        self._assert_identical(
            lambda fused, dev: _turn_solver(fused, dev, batch_boards=4),
            iterations=5,
            device="cpu",
        )

    def test_cpu_path_is_bit_identical(self) -> None:
        self._assert_identical(_turn_solver, iterations=4, device="cpu")

    def test_cuda_bucket_sharing_is_nondeterministic_and_fusion_stays_within_it(self) -> None:
        """Where bit-identity is impossible, bound fusion by the intrinsic noise.

        When several combos share a bucket row, the BACKWARD pass's
        `regrets.index_add_` has repeated destination indices, so CUDA
        accumulates them with atomics in arbitrary order. That makes the solver
        non-deterministic run-to-run — a pre-existing property of the verified
        kernel, not of this fusion, and the reason bit-identical reproducibility
        is unavailable for any bucketed configuration on GPU.

        The honest test is therefore: fused-vs-looped must be no worse than
        looped-vs-itself.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        for label, build in (
            ("four-street, 4 buckets", _blueprint_solver),
            ("exact turn, batch 4", lambda fused, dev: _turn_solver(fused, dev, batch_boards=4)),
        ):
            first, second, fused = build(False, "cuda"), build(False, "cuda"), build(True, "cuda")
            for solver in (first, second, fused):
                solver.run(6)
            floor = float((first.regrets - second.regrets).abs().max())
            observed = float((first.regrets - fused.regrets).abs().max())
            self.assertGreater(floor, 0.0, f"{label}: expected nondeterminism, saw none")
            self.assertLessEqual(
                observed, floor * 3.0,
                f"{label}: fusion delta {observed:.3e} exceeds the "
                f"nondeterminism floor {floor:.3e}",
            )

    def test_average_strategies_match(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        looped, fused = _turn_solver(False, device), _turn_solver(True, device)
        looped.run(10)
        fused.run(10)
        np.testing.assert_array_equal(
            looped.average_strategy_tables(), fused.average_strategy_tables()
        )


class FusedPlanStructureTests(unittest.TestCase):
    """The invariants that make fusion safe are asserted, not assumed."""

    def test_every_child_has_exactly_one_parent_edge(self) -> None:
        solver = _turn_solver(True, "cpu")
        seen: set[int] = set()
        for plan in solver.level_plans:
            fused = plan.get("fused")
            if fused is None:
                continue
            destinations = fused["actor_dst"].tolist()
            self.assertEqual(len(destinations), len(set(destinations)),
                             "duplicate destination within a level would race")
            overlap = seen & set(destinations)
            self.assertFalse(overlap, f"node written from two levels: {sorted(overlap)[:5]}")
            seen |= set(destinations)

    def test_parents_and_children_never_share_a_level(self) -> None:
        solver = _turn_solver(True, "cpu")
        nodes = len(solver.tree)
        for plan in solver.level_plans:
            fused = plan.get("fused")
            if fused is None:
                continue
            sources = {index % nodes for index in fused["actor_src"].tolist()}
            destinations = {index % nodes for index in fused["actor_dst"].tolist()}
            self.assertFalse(sources & destinations,
                             "read/write aliasing within a level breaks fusion")

    def test_fusion_is_enabled_by_default(self) -> None:
        self.assertTrue(_turn_solver.__defaults__ is not None or True)
        solver = VectorCFR(
            BettingTree(GpuActionConfig(stack_bb=20.0)),
            DealSampler(flop_buckets=2, turn_buckets=2, river_buckets=2),
            device="cpu",
        )
        self.assertTrue(solver.fused_forward)


if __name__ == "__main__":
    unittest.main()
