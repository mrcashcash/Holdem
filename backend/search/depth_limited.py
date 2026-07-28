"""Depth-limited solving: HORIZON terminals priced by an external evaluator.

The flop re-solve builds its tree with ``end_street=1`` — betting enumerates
only the flop; where a full tree would deal the turn, a HORIZON terminal
stands in. Each CFR iteration the evaluator receives the players' reach at
every horizon node (plus the iteration's sampled board) and writes the
traverser's per-combo values, exactly as _fold/_showdown do for real
terminals. Chance stays unbiased because the evaluator sees the PCS-sampled
turn card, matching how the rest of the tree is sampled.

Two evaluators:
- ``ShowdownOracle`` — prices horizons as an immediate showdown for the
  matched pot on the sampled board. Used by the equivalence test: a
  depth-limited solve with this oracle must match a full solve of the game
  variant whose streets simply END there (tree built with the same rules), so
  every piece of horizon plumbing (reach extraction, value injection,
  scaling) is proven against the trusted terminal machinery.
- ``RiverNetEvaluator`` (backend/search/river_horizon.py) — prices a turn
  tree's river horizon from the P3a river net.

The v0 ``NetEvaluator`` and ``solve_flop_subgame`` were deleted on 2026-07-28
along with the rest of the 169-bucket CFV pipeline.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS
from backend.solver.gpu.tree import HORIZON



class DepthLimitedCFR(VectorCFR):
    """VectorCFR over an ``end_street`` tree with a horizon evaluator."""

    def __init__(self, *args, horizon_evaluator=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.horizon_nodes = torch.tensor(
            np.flatnonzero(self.tree.kind == HORIZON), dtype=torch.long, device=self.device
        )
        self.horizon_pots = torch.tensor(
            self.tree.matched_pot[self.tree.kind == HORIZON], dtype=torch.float32, device=self.device
        )
        self.evaluator = horizon_evaluator
        if self.horizon_nodes.numel():
            self._horizon_hook = self._price_horizons

    def _price_horizons(self, values, reach, traverser, deal, valid) -> None:
        self.evaluator(self, values, reach, traverser, deal, valid)


class ShowdownOracle:
    """Immediate showdown at the horizon (equivalence-test evaluator)."""

    def __call__(self, solver: DepthLimitedCFR, values, reach, traverser, deal, valid) -> None:
        nodes = solver.horizon_nodes
        stacked = torch.zeros_like(reach)
        stacked[1 - traverser, nodes, :] = reach[1 - traverser, nodes, :]
        # Reuse the trusted showdown kernel on JUST the horizon nodes by
        # masquerading them as showdowns of their matched pot.
        scores = self._scores(solver, deal)
        horizon_values = torch.zeros_like(values)
        solver._showdown_values(
            horizon_values, stacked, scores, valid, traverser, nodes=nodes, pots=solver.horizon_pots
        )
        values[nodes, :] = horizon_values[nodes, :]

    @staticmethod
    def _scores(solver, deal):
        import numpy as _np

        if isinstance(deal, list):
            return torch.tensor(_np.stack([d.river_scores for d in deal]), dtype=torch.long, device=solver.device)
        return torch.tensor(deal.river_scores, dtype=torch.long, device=solver.device).unsqueeze(0)
