"""The GPU VectorCFR converges to the abstract Nash equilibrium.

Regression guard for the 2026-07-21 finding: measured by the independent,
Kuhn/Leduc-validated best response on a compact rebuild of the fixed-board
abstract game, a trained control strategy is ~0 mbb exploitable. (The tensor
exact-BR probe in exploit.py over-counts and is deliberately NOT used here.)
"""

import unittest

from backend.search.gpu_subgame import FixedBoardSampler
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.convergence import abstract_exploitability_mbb
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

BOARD = (2, 7, 24, 33, 50)


class GpuConvergenceTests(unittest.TestCase):
    def test_fixed_river_control_converges(self) -> None:
        cfg = GpuActionConfig(
            preflop_fractions=(1.0,),
            postflop_fractions=(0.75,),
            max_raises_per_street=2,
            stack_bb=20.0,
        )
        tree = BettingTree(cfg, start_street=3, start_pot=8.0, start_stacks=(16.0, 16.0))
        sampler = FixedBoardSampler(DealSampler(river_buckets=20), BOARD)
        solver = VectorCFR(tree, sampler, device="cpu", seed=23, averaging_delay=100)
        solver.run(4000)
        mbb = abstract_exploitability_mbb(solver, seed=0)
        # A converged abstract equilibrium is ~0; allow generous slack for the
        # finite iteration count. (The over-counting probe read ~2107 here.)
        self.assertGreaterEqual(mbb, -1e-6)
        self.assertLess(mbb, 50.0, f"control did not converge: {mbb:.1f} mbb")


if __name__ == "__main__":
    unittest.main()
