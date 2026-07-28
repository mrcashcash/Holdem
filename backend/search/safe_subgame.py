"""Safe subgame re-solving via the max-margin resolve gadget.

Why: the plain re-solve in ``gpu_subgame.py`` trusts the blueprint-tracked
ranges for BOTH players. Against an opponent who deviates from the blueprint
(any real opponent), the subgame is solved against a phantom range and can be
worse than not searching at all. The fix, following CFR-D re-solving (Burch,
Johanson & Bowling 2014) and the max-margin gadget (Moravcik et al.; Brown &
Sandholm 2017, arXiv 1705.02955): give the opponent a per-hand OPT-OUT worth
their estimated counterfactual value of the original strategy. Solving the
augmented game means our re-solved strategy cannot give any opponent hand more
than that estimate — the re-solve is safe relative to the estimate quality,
whatever range the opponent actually holds.

Pipeline (``solve_subgame_safe``):
  1. preliminary solve of the subgame (both ranges as tracked) -> sigma0;
  2. price the opt-out: alt[c] = opponent combo c's root CFV when both play
     sigma0 (both-frozen evaluation passes, averaged over sampled runouts);
  3. gadget re-solve with ``GadgetCFR``: each iteration the opponent's root
     reach is scaled by a regret-matched per-combo enter probability whose
     alternative action pays alt[c]; our side re-solves normally.

The gadget runs eager (no CUDA-graph capture yet): the per-iteration gadget
update sits between traversals, which capture can't express without moving it
into the graph — a future optimization.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.search.gpu_subgame import (
    SUBGAME_ITERATIONS,
    SubgameSolution,
    build_subgame,
)
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS


class GadgetCFR:
    """Max-margin resolve gadget wrapped around a ``VectorCFR`` subgame solver.

    ``constrained`` is the abstract seat of the OPPONENT (the player whose
    range we do not trust). Their root reach each iteration is
    ``base_range * enter_prob`` where ``enter_prob`` regret-matches against the
    fixed opt-out values ``alt`` (their CFVs under the preliminary solution).
    """

    def __init__(self, solver: VectorCFR, constrained: int, base_ranges: np.ndarray, alt: torch.Tensor) -> None:
        self.solver = solver
        self.constrained = constrained
        self.base = torch.tensor(base_ranges, dtype=torch.float32, device=solver.device)  # [2, C]
        self.alt = alt.to(solver.device)  # [C]
        # Gadget regrets per opponent combo: [:, 0]=enter, [:, 1]=opt out.
        self.gadget_regrets = torch.zeros((NUM_COMBOS, 2), dtype=torch.float32, device=solver.device)

    def enter_probability(self) -> torch.Tensor:
        positive = self.gadget_regrets.clamp_min(0.0)
        total = positive.sum(dim=1)
        # No positive regret yet -> enter (matches the pre-gadget behaviour and
        # keeps early iterations informative for both actions).
        enter = torch.where(total > 0, positive[:, 0] / total.clamp_min(1e-30), torch.ones_like(total))
        return enter

    def run(self, iterations: int) -> None:
        # Mirrors VectorCFR.run's update protocol (alternating traversals +
        # DCFR discount), inserting the gadget reach/regret steps around it.
        solver = self.solver
        for _ in range(iterations):
            solver.iteration += 1
            deals = [solver.sampler.sample(solver.rng) for _ in range(solver.batch_boards)]
            enter = self.enter_probability()  # [C]
            reach = self.base.clone()
            reach[self.constrained] = reach[self.constrained] * enter
            solver.root_reach = reach

            for traverser in (0, 1):
                solver._iterate(deals, traverser=traverser)
                if traverser == self.constrained:
                    # Root CFVs of the opponent entering, per combo (mean over
                    # the board batch when batching).
                    root = solver._last_root_values.view(-1, NUM_COMBOS).mean(dim=0)
                    node_value = enter * root + (1.0 - enter) * self.alt
                    self.gadget_regrets[:, 0] += root - node_value
                    self.gadget_regrets[:, 1] += self.alt - node_value
            solver._discount()


def opponent_alt_values(
    solver: VectorCFR, average: torch.Tensor, constrained: int, boards: int = 8
) -> torch.Tensor:
    """[v1] Opponent per-combo root CFVs when both players follow ``average``.

    Pure evaluation: regrets/strategy sums are snapshotted and restored, so the
    passes leave the solver untouched. NOTE: following-values UNDERSTATE what a
    deviating opponent can get, which under-prices the gadget's opt-out — the
    measured failure vs extreme deviators (nit). Prefer
    ``opponent_alt_values_br`` (v2), the theoretically correct choice.
    """
    regrets_backup = solver.regrets.clone()
    sums_backup = solver.strategy_sums.clone()
    values = torch.zeros(NUM_COMBOS, dtype=torch.float32, device=solver.device)
    for _ in range(max(1, boards)):
        deal = solver.sampler.sample(solver.rng)
        solver._iterate(deal, traverser=constrained, frozen_average=average, frozen_player=None)
        values += solver._last_root_values.view(-1, NUM_COMBOS).mean(dim=0)
    solver.regrets.copy_(regrets_backup)
    solver.strategy_sums.copy_(sums_backup)
    return values / max(1, boards)


def opponent_alt_values_br(
    solver: VectorCFR,
    average: torch.Tensor,
    constrained: int,
    br_iterations: int = 80,
    eval_boards: int = 8,
) -> torch.Tensor:
    """[v2] Opponent per-combo BEST-RESPONSE-to-sigma0 root CFVs.

    CFR-D re-solving and the max-margin gadget both define safety against the
    opponent's best response to the original strategy, not their obedient
    continuation of it (Burch et al. 2014; Brown & Sandholm 2017). We train a
    bucket-bound responder with the existing CFR-BR machinery (we stay frozen
    to sigma0, the opponent regret-matches) and read its root CFVs. Higher alt
    values for strong hands force the gadget re-solve to actually defend
    against the top of a deviator's range — the v1 gap.

    Solver state is snapshotted/restored; the responder trains on scratch
    regrets so sigma0's tensors are untouched.
    """
    regrets_backup = solver.regrets.clone()
    sums_backup = solver.strategy_sums.clone()
    iteration_backup = solver.iteration
    solver.regrets.zero_()
    solver.strategy_sums.zero_()
    us = 1 - constrained
    # Train the responder: opponent explores via regret matching against our
    # frozen average. (Alternating traversals are unnecessary — only the
    # opponent learns.)
    for _ in range(max(1, br_iterations)):
        deal = solver.sampler.sample(solver.rng)
        solver._iterate(deal, traverser=constrained, frozen_average=average, frozen_player=us)
    # Read the trained responder's root CFVs over fresh boards.
    values = torch.zeros(NUM_COMBOS, dtype=torch.float32, device=solver.device)
    for _ in range(max(1, eval_boards)):
        deal = solver.sampler.sample(solver.rng)
        solver._iterate(deal, traverser=constrained, frozen_average=average, frozen_player=us)
        values += solver._last_root_values.view(-1, NUM_COMBOS).mean(dim=0)
    solver.regrets.copy_(regrets_backup)
    solver.strategy_sums.copy_(sums_backup)
    solver.iteration = iteration_backup
    return values / max(1, eval_boards)


def solve_subgame_safe(agent, game, player: int, iterations: int = SUBGAME_ITERATIONS) -> SubgameSolution:
    """Drop-in safe replacement for ``gpu_subgame.solve_subgame``."""
    # Stage 1: preliminary (unsafe) solve — sigma0 and the value estimate.
    solver, tree, street_buckets, ranges = build_subgame(agent, game, iterations)
    solver.run(iterations)
    sigma0 = solver.average_strategy_tables()
    average0 = torch.tensor(sigma0, dtype=torch.float32, device=solver.device)

    # Stage 2: price the opponent's opt-out under sigma0. Default is the v2
    # best-response pricing (theoretically correct; defends vs deviators);
    # HOLDEM_SAFE_ALT=follow selects the cheaper v1 following-value estimate.
    import os as _os

    our_seat = agent._abstract_seat(game, player)
    constrained = 1 - our_seat
    solver.root_reach = torch.tensor(ranges, dtype=torch.float32, device=solver.device)
    if _os.environ.get("HOLDEM_SAFE_ALT", "br") == "follow":
        alt = opponent_alt_values(solver, average0, constrained, boards=8)
    else:
        alt = opponent_alt_values_br(
            solver, average0, constrained, br_iterations=max(40, iterations // 2), eval_boards=8
        )

    # Stage 3: gadget re-solve from scratch on the same tree.
    solver.regrets.zero_()
    solver.strategy_sums.zero_()
    solver.iteration = 0
    gadget = GadgetCFR(solver, constrained, ranges, alt)
    gadget.run(iterations)

    strategy = solver.average_strategy_tables()
    return SubgameSolution(tree, strategy.astype(np.float64), agent.sampler, street_buckets)
