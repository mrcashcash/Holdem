"""Vectorized public-chance-sampling CFR over the flattened betting tree.

One iteration samples a board runout, then walks the betting tree ONCE while
carrying reach-probability vectors over all 1,326 private combos for both
players (Johanson et al.'s public chance sampling, the formulation behind
DeepStack-style solvers). All per-node work is batched by tree depth into
tensor ops, so the device (CPU or CUDA) sees a few hundred large kernels per
iteration instead of millions of tiny Python steps.

Terminal values use the sort-based showdown evaluation with per-card blocker
corrections (one batched sort per deal, prefix sums per showdown node), and
fold values use the same inclusion-exclusion card correction.

Regrets/strategy sums are dense float32 tensors [nodes, 169, actions]
(169 = the largest street bucket count), linear-CFR weighted.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS, Deal, DealSampler, combos
from backend.solver.gpu.tree import DECISION, FOLD_NODE, SHOWDOWN, STREET_END, BettingTree

MAX_BUCKETS = 169


class VectorCFR:
    def __init__(
        self,
        tree: BettingTree,
        sampler: DealSampler | None = None,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.tree = tree
        self.sampler = sampler or DealSampler()
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.iteration = 0

        nodes = len(tree)
        actions = tree.config.num_actions
        self.num_actions = actions
        self.regrets = torch.zeros((nodes, MAX_BUCKETS, actions), dtype=torch.float32, device=self.device)
        self.strategy_sums = torch.zeros_like(self.regrets)

        self._prepare_static_tensors()

    # -- static structure ------------------------------------------------------

    def _prepare_static_tensors(self) -> None:
        tree, device = self.tree, self.device
        self.t_children = torch.tensor(tree.children, dtype=torch.long, device=device)
        self.t_legal = torch.tensor(tree.legal, device=device)
        self.t_actor = torch.tensor(tree.actor, dtype=torch.long, device=device)
        self.t_street = torch.tensor(tree.street, dtype=torch.long, device=device)
        self.t_matched_pot = torch.tensor(tree.matched_pot, dtype=torch.float32, device=device)
        self.t_fold_loser = torch.tensor(tree.fold_loser, dtype=torch.long, device=device)
        self.t_fold_committed = torch.tensor(tree.fold_loser_committed, dtype=torch.float32, device=device)
        self.t_card_in_combo = torch.tensor(CARD_IN_COMBO, dtype=torch.float32, device=device)
        self.t_combos = torch.tensor(combos(), dtype=torch.long, device=device)

        # Group nodes by depth for level-batched processing (parents always
        # precede children in index order, but depth grouping lets one gather/
        # scatter handle a whole level).
        depth = np.zeros(len(tree), dtype=np.int64)
        for node in range(len(tree)):
            for child in tree.children[node]:
                if child >= 0:
                    depth[child] = depth[node] + 1
        max_depth = int(depth.max())
        self.levels: list[torch.Tensor] = []
        for level in range(max_depth + 1):
            members = np.flatnonzero(depth == level)
            self.levels.append(torch.tensor(members, dtype=torch.long, device=self.device))

        kind = tree.kind
        self.decision_mask = torch.tensor(kind == DECISION, device=device)
        self.showdown_nodes = torch.tensor(np.flatnonzero(kind == SHOWDOWN), dtype=torch.long, device=device)
        self.fold_nodes = torch.tensor(np.flatnonzero(kind == FOLD_NODE), dtype=torch.long, device=device)
        self.street_end_nodes = torch.tensor(np.flatnonzero(kind == STREET_END), dtype=torch.long, device=device)

    # -- strategy --------------------------------------------------------------

    def _node_strategies(self, node_ids: torch.Tensor, node_buckets: torch.Tensor) -> torch.Tensor:
        """Regret-matched strategy [L, C, A] for decision nodes ``node_ids``.

        ``node_buckets`` [L, C]: bucket of the acting player's combo (already
        street-resolved); invalid combos may carry bucket 0 — their reach is
        zero so their contribution vanishes.
        """
        gathered = self.regrets[node_ids.unsqueeze(1), node_buckets]  # [L, C, A]
        positive = gathered.clamp_min(0.0)
        legal = self.t_legal[node_ids].unsqueeze(1)  # [L, 1, A]
        positive = positive * legal
        totals = positive.sum(dim=2, keepdim=True)
        legal_counts = legal.sum(dim=2, keepdim=True).clamp_min(1).to(positive.dtype)
        uniform = legal.to(positive.dtype) / legal_counts
        strategy = torch.where(totals > 0, positive / totals.clamp_min(1e-30), uniform)
        return strategy

    # -- one iteration -----------------------------------------------------------

    def run(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iteration += 1
            deal = self.sampler.sample(self.rng)
            self._iterate(deal)

    def _iterate(self, deal: Deal) -> None:
        device = self.device
        nodes = len(self.tree)
        valid = torch.tensor(deal.valid, device=device)
        buckets = torch.tensor(deal.buckets, dtype=torch.long, device=device).clamp_min(0)  # [4, C]
        scores = torch.tensor(deal.river_scores, dtype=torch.long, device=device)

        reach = torch.zeros((2, nodes, NUM_COMBOS), dtype=torch.float32, device=device)
        reach[:, self.tree.root, :] = valid.float()

        strategies: dict[int, torch.Tensor] = {}
        level_decisions: dict[int, torch.Tensor] = {}

        # ---- forward: push reach through levels --------------------------------
        for level_index, level in enumerate(self.levels):
            decisions = level[self.decision_mask[level]]
            passthrough = level[~self.decision_mask[level]]
            if passthrough.numel():
                street_ends = passthrough[self.tree_kind(passthrough) == STREET_END]
                if street_ends.numel():
                    children = self.t_children[street_ends, 0]
                    reach[:, children, :] += reach[:, street_ends, :]
            if not decisions.numel():
                continue
            node_buckets = buckets[self.t_street[decisions]]  # [L, C]
            strategy = self._node_strategies(decisions, node_buckets)  # [L, C, A]
            level_decisions[level_index] = decisions
            strategies[level_index] = strategy
            actors = self.t_actor[decisions]  # [L]
            for action in range(self.num_actions):
                legal_here = self.t_legal[decisions, action]
                if not bool(legal_here.any()):
                    continue
                acting = decisions[legal_here]
                children = self.t_children[acting, action]
                acting_actor = actors[legal_here]
                probability = strategies[level_index][legal_here, :, action]  # [K, C]
                for player in (0, 1):
                    is_actor = acting_actor == player
                    if bool(is_actor.any()):
                        source = acting[is_actor]
                        reach[player, self.t_children[source, action], :] += (
                            reach[player, source, :] * probability[is_actor]
                        )
                    is_opponent = ~is_actor
                    if bool(is_opponent.any()):
                        source = acting[is_opponent]
                        reach[player, self.t_children[source, action], :] += reach[player, source, :]

        # ---- terminal values ----------------------------------------------------
        values = torch.zeros((2, nodes, NUM_COMBOS), dtype=torch.float32, device=device)
        self._fold_values(values, reach)
        self._showdown_values(values, reach, scores, valid)

        # ---- backward: roll values up, accumulate regrets and strategy sums ------
        weight = float(self.iteration)
        for level_index in range(len(self.levels) - 1, -1, -1):
            level = self.levels[level_index]
            street_ends = level[self.tree_kind(level) == STREET_END]
            if street_ends.numel():
                children = self.t_children[street_ends, 0]
                values[:, street_ends, :] = values[:, children, :]
            decisions = level_decisions.get(level_index)
            if decisions is None or not decisions.numel():
                continue
            strategy = strategies[level_index]  # [L, C, A]
            child_values = torch.zeros(
                (2, decisions.shape[0], NUM_COMBOS, self.num_actions), dtype=torch.float32, device=device
            )
            for action in range(self.num_actions):
                legal_here = self.t_legal[decisions, action]
                if not bool(legal_here.any()):
                    continue
                children = self.t_children[decisions[legal_here], action]
                child_values[:, legal_here, :, action] = values[:, children, :]

            actors = self.t_actor[decisions]
            node_buckets = buckets[self.t_street[decisions]]  # [L, C]
            legal = self.t_legal[decisions].unsqueeze(1).float()  # [L, 1, A]
            for player in (0, 1):
                acted = actors == player
                if not bool(acted.any()):
                    continue
                own = decisions[acted]
                own_strategy = strategy[acted]  # [K, C, A]
                own_children = child_values[player, acted]  # [K, C, A]
                node_value = (own_strategy * own_children).sum(dim=2)  # [K, C]
                values[player, own, :] = node_value
                # Terminal values are opponent-reach weighted already, so the
                # counterfactual regret is simply the child/node value gap.
                regret_increment = (own_children - node_value.unsqueeze(2)) * legal[acted]
                sum_increment = own_strategy * reach[player, own, :].unsqueeze(2)
                flat_index = (own.unsqueeze(1) * MAX_BUCKETS + node_buckets[acted]).reshape(-1)
                self.regrets.view(-1, self.num_actions).index_add_(
                    0, flat_index, (weight * regret_increment).reshape(-1, self.num_actions)
                )
                self.strategy_sums.view(-1, self.num_actions).index_add_(
                    0, flat_index, (weight * sum_increment).reshape(-1, self.num_actions)
                )
                # Opponent value at this node: their reach is unchanged, value
                # is the strategy-weighted average of child values.
                other_children = child_values[1 - player, acted]
                values[1 - player, own, :] = (own_strategy * other_children).sum(dim=2)

    def tree_kind(self, node_ids: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.tree.kind, device=self.device)[node_ids]

    # -- terminal math -----------------------------------------------------------

    def _opponent_mass(self, opponent_reach: torch.Tensor) -> torch.Tensor:
        """Reach mass of compatible opponent combos, per hero combo. [.., C]"""
        total = opponent_reach.sum(dim=-1, keepdim=True)  # [.., 1]
        per_card = opponent_reach @ self.t_card_in_combo.T  # [.., 52]
        card_a = self.t_combos[:, 0]
        card_b = self.t_combos[:, 1]
        blocked = per_card[..., card_a] + per_card[..., card_b] - opponent_reach
        return total - blocked

    def _fold_values(self, values: torch.Tensor, reach: torch.Tensor) -> None:
        nodes = self.fold_nodes
        if not nodes.numel():
            return
        amount = self.t_fold_committed[nodes].unsqueeze(1)  # [F, 1]
        loser = self.t_fold_loser[nodes]  # [F]
        for player in (0, 1):
            sign = torch.where(loser == player, -1.0, 1.0).unsqueeze(1)
            opponent_mass = self._opponent_mass(reach[1 - player, nodes, :])  # [F, C]
            values[player, nodes, :] = sign * amount * opponent_mass

    def _showdown_values(
        self,
        values: torch.Tensor,
        reach: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        nodes = self.showdown_nodes
        if not nodes.numel():
            return
        order = torch.argsort(scores)  # invalid (-1) scores sort first
        sorted_scores = scores[order]
        boundaries_left = torch.searchsorted(sorted_scores, scores, side="left")
        boundaries_right = torch.searchsorted(sorted_scores, scores, side="right")
        pots = self.t_matched_pot[nodes].unsqueeze(1)  # [S, 1]
        card_in_combo = self.t_card_in_combo > 0  # [52, C] bool
        combo_cards = self.t_combos  # [C, 2]

        for player in (0, 1):
            opponent = reach[1 - player, nodes, :] * valid.float()  # [S, C]
            ordered = opponent[:, order]  # [S, C] in score order
            prefix = torch.cumsum(ordered, dim=1)
            total = prefix[:, -1:]
            zeros = torch.zeros((nodes.shape[0], 1), device=self.device, dtype=prefix.dtype)
            padded = torch.cat([zeros, prefix], dim=1)
            worse = padded[:, boundaries_left]  # [S, C] opponents with lower score
            better = total - padded[:, boundaries_right]

            # Blocker correction, one card at a time: subtract the worse/better
            # mass contributed by opponent combos sharing a card with the hero.
            correction = torch.zeros_like(worse)  # (worse_blocked - better_blocked)
            for card in range(52):
                members = card_in_combo[card]  # [C] combos containing this card
                masked = ordered * members[order].unsqueeze(0)  # [S, C]
                card_prefix = torch.cumsum(masked, dim=1)
                card_total = card_prefix[:, -1:]
                card_padded = torch.cat([zeros, card_prefix], dim=1)
                holders = torch.nonzero(
                    (combo_cards[:, 0] == card) | (combo_cards[:, 1] == card)
                ).squeeze(1)
                worse_blocked = card_padded[:, boundaries_left[holders]]
                better_blocked = card_total - card_padded[:, boundaries_right[holders]]
                correction[:, holders] += worse_blocked - better_blocked

            values[player, nodes, :] = pots * (worse - better - correction) * valid.float()

    # -- outputs -----------------------------------------------------------------

    def average_strategy_tables(self) -> np.ndarray:
        """Normalized average strategy [nodes, MAX_BUCKETS, A] (uniform where unseen)."""
        sums = self.strategy_sums.cpu().numpy()
        legal = self.tree.legal[:, None, :]
        totals = sums.sum(axis=2, keepdims=True)
        legal_counts = legal.sum(axis=2, keepdims=True).clip(min=1)
        uniform = legal / legal_counts
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = np.where(totals > 0, sums / np.maximum(totals, 1e-30), uniform)
        return normalized * legal
