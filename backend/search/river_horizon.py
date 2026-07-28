"""Price a turn tree's river horizon with the P3a river CFV net.

This is the piece that cashes in the 9x: a turn tree solved to showdown has 726
nodes, the same tree truncated at the river has 81. The net stands in for the
whole river subtree.

Deliberately NOT reusing `depth_limited.NetEvaluator`, which is the v0
implementation: it aggregates reach into **169 buckets**, where DeepStack and
Supremus both use ~1,000 and P3a uses exact per-combo values. Feeding a
169-bucket horizon into an exact-card solve would reintroduce, at the horizon,
precisely the abstraction the rest of P1 removes.

Value convention, which is the fiddly part. `VectorCFR._iterate` expects
`values[node, combo]` to be the traverser's counterfactual value *already
weighted by the opponent's reach at that node*. The net instead returns
pot-normalised CFVs for unit-mass ranges. So each horizon node needs:

    value = net_output(normalised ranges) * pot * opponent_reach_mass

with the ranges renormalised per node, because reach at a horizon node is
generally not a probability distribution.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.solver.gpu.deals import NUM_COMBOS


class RiverNetEvaluator:
    """Horizon evaluator backed by `backend.cfv.river_net.RiverCfvNet`.

    One batched net call per iteration covers every horizon node, so the cost is
    a single MLP forward over [horizon_nodes, ...] rather than anything
    proportional to the river subtree that was removed.
    """

    def __init__(self, net, device, board_turn: tuple[int, ...], stack_bb: float) -> None:
        if len(board_turn) != 4:
            raise ValueError("a river horizon sits on a four-card turn board")
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.stack_bb = float(stack_bb)
        # The river card is unknown at the horizon, so the board fed to the net
        # is the turn prefix. The net was trained on five-card boards, so the
        # missing card is represented by its absence from the multi-hot; the
        # sampled deal supplies it when available (see __call__).
        self.board_turn = tuple(int(card) for card in board_turn)

    def _board_hot(self, river: int | None) -> torch.Tensor:
        hot = torch.zeros(52, device=self.device)
        for card in self.board_turn:
            hot[card] = 1.0
        if river is not None:
            hot[int(river)] = 1.0
        return hot

    def __call__(self, solver, values, reach, traverser, deal, valid) -> None:
        nodes = solver.horizon_nodes
        count = int(nodes.numel())
        if not count:
            return
        deals = deal if isinstance(deal, list) else [deal]
        batch = len(deals)
        width = batch * NUM_COMBOS

        with torch.no_grad():
            reach_here = reach[:, nodes, :].reshape(2, count, batch, NUM_COMBOS)
            # `matched_pot` stores min(committed) — the WINNER'S GAIN, i.e. half
            # the pot when commitments are level. `river_dataset` records the
            # FULL pot, so feeding matched_pot straight to the net would be 2x
            # off in both the input features and the value rescaling. Silent:
            # the solve would run and simply price a differently-sized game.
            pots = 2.0 * solver.horizon_pots  # [H], full pot in bb
            for index, single in enumerate(deals):
                combo_valid = torch.as_tensor(single.valid, device=self.device)
                river = single.board[4] if len(single.board) > 4 else None
                board_hot = self._board_hot(river).expand(count, -1)

                ranges = reach_here[:, :, index, :] * combo_valid.float()   # [2, H, C]
                mass = ranges.sum(dim=2).clamp_min(1e-12)                   # [2, H]
                normalised = (ranges / mass.unsqueeze(2)).permute(1, 0, 2)  # [H, 2, C]

                # Per-node, not pots.max(): horizon nodes sit at different pots,
                # so a single scalar would give every node but the largest the
                # wrong SPR feature.
                behind = (self.stack_bb - pots / 2.0).clamp_min(1e-3)       # [H]
                scalars = torch.stack(
                    [pots / self.stack_bb, pots / behind], dim=1
                ).to(self.device)                                           # [H, 2]

                predicted = self.net(scalars, board_hot, normalised)        # [H, 2, C]
                # Undo the training normalisations: pot-normalised, unit-mass ->
                # chip-valued and opponent-reach weighted, which is the
                # convention _iterate's terminal kernels use.
                per_combo = (
                    predicted[:, traverser, :]
                    * pots.unsqueeze(1)
                    * mass[1 - traverser].unsqueeze(1)
                ) * combo_valid.float()

                flat = values[nodes, :].reshape(count, batch, NUM_COMBOS)
                flat[:, index, :] = per_combo
                values[nodes, :] = flat.reshape(count, width)


def build_turn_tree_with_river_horizon(config, root_state):
    """Turn tree truncated at the river (81 nodes vs 726 solved to showdown)."""
    from backend.solver.gpu.tree import BettingTree

    return BettingTree(config, root_state=root_state, end_street=2)


def solve_turn_with_river_net(
    net,
    board_turn: tuple[int, ...],
    root_state,
    config,
    ranges: np.ndarray,
    iterations: int = 240,
    device: str | None = None,
):
    """Depth-limited exact-card turn solve, river priced by the net."""
    from backend.search.depth_limited import DepthLimitedCFR
    from backend.search.exact_turn import ExactTurnSampler

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tree = build_turn_tree_with_river_horizon(config, root_state)
    solver = DepthLimitedCFR(
        tree,
        ExactTurnSampler(board_turn),
        device=device,
        seed=11,
        averaging_delay=max(2, iterations // 6),
        horizon_evaluator=RiverNetEvaluator(net, device, board_turn, config.stack_bb),
    )
    solver.root_reach = torch.as_tensor(
        np.asarray(ranges, dtype=np.float32), device=solver.device
    )
    solver.run(iterations)
    return tree, solver.average_strategy_tables()
