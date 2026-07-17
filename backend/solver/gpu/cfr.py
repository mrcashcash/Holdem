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
        discount_alpha: float = 1.5,
        discount_beta: float = 0.0,
        discount_gamma: float = 2.0,
        averaging_delay: int = 0,
    ) -> None:
        self.tree = tree
        self.sampler = sampler or DealSampler()
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.iteration = 0
        # DCFR(alpha, beta, gamma) (Brown & Sandholm AAAI 2019): the setting
        # (1.5, 0, 2) is the paper's best on NLHE and what desktop-postflop/
        # opensolver ship. averaging_delay skips strategy-sum accumulation for
        # the earliest (worst) iterations (Supremus' DCFR+ trick).
        self.discount_alpha = discount_alpha
        self.discount_beta = discount_beta
        self.discount_gamma = discount_gamma
        self.averaging_delay = averaging_delay
        # Optional [2, NUM_COMBOS] root reach (re-solving subgames start from
        # tracked ranges instead of uniform deals).
        self.root_reach: torch.Tensor | None = None

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

        # Static per-level plans: everything about legality, children, and
        # actors is fixed by the tree, so precompute the index tensors once.
        # This removes ~1000 CPU<->GPU sync points per iteration (the .any()/
        # boolean-mask pattern) that made iteration time size-independent.
        self.level_plans: list[dict] = []
        for level_members in self.levels:
            members_np = level_members.cpu().numpy()
            decisions_np = members_np[kind[members_np] == DECISION]
            street_ends_np = members_np[kind[members_np] == STREET_END]
            plan = {
                "decisions": torch.tensor(decisions_np, dtype=torch.long, device=device),
                "street_ends": torch.tensor(street_ends_np, dtype=torch.long, device=device),
                "street_end_children": torch.tensor(
                    tree.children[street_ends_np, 0], dtype=torch.long, device=device
                ),
                "actions": [],
                "actor_rows": {
                    player: torch.tensor(
                        np.flatnonzero(tree.actor[decisions_np] == player), dtype=torch.long, device=device
                    )
                    for player in (0, 1)
                },
            }
            for action in range(tree.config.num_actions):
                legal_rows = np.flatnonzero(tree.legal[decisions_np, action])
                if legal_rows.size == 0:
                    continue
                acting_np = decisions_np[legal_rows]
                actors_np = tree.actor[acting_np]
                plan["actions"].append(
                    {
                        "action": action,
                        "rows": torch.tensor(legal_rows, dtype=torch.long, device=device),
                        "children": torch.tensor(
                            tree.children[acting_np, action], dtype=torch.long, device=device
                        ),
                        "actor_split": {
                            player: (
                                torch.tensor(
                                    acting_np[actors_np == player], dtype=torch.long, device=device
                                ),
                                torch.tensor(
                                    tree.children[acting_np[actors_np == player], action],
                                    dtype=torch.long,
                                    device=device,
                                ),
                                torch.tensor(
                                    legal_rows[actors_np == player], dtype=torch.long, device=device
                                ),
                            )
                            for player in (0, 1)
                        },
                    }
                )
            self.level_plans.append(plan)

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
            # Alternating updates (Burch et al. JAIR 2019): each player's
            # regrets are updated in their own pass against the opponent's
            # freshly regret-matched strategy.
            self._iterate(deal, traverser=0)
            self._iterate(deal, traverser=1)
            self._discount()

    def _discount(self) -> None:
        t = float(self.iteration)
        positive_factor = t**self.discount_alpha / (t**self.discount_alpha + 1.0)
        negative_factor = t**self.discount_beta / (t**self.discount_beta + 1.0)
        self.regrets = torch.where(
            self.regrets > 0, self.regrets * positive_factor, self.regrets * negative_factor
        )
        if self.iteration > self.averaging_delay:
            self.strategy_sums *= (t / (t + 1.0)) ** self.discount_gamma
        else:
            self.strategy_sums.zero_()

    def _iterate(
        self,
        deal: Deal,
        traverser: int = 0,
        frozen_average: torch.Tensor | None = None,
        frozen_player: int | None = None,
    ) -> None:
        """One traversal. With ``frozen_average``/``frozen_player`` set, that
        player follows the given average-strategy tensor instead of
        regret matching (CFR-BR: the traverser best-responds to it)."""
        device = self.device
        nodes = len(self.tree)
        valid = torch.tensor(deal.valid, device=device)
        buckets = torch.tensor(deal.buckets, dtype=torch.long, device=device).clamp_min(0)  # [4, C]
        scores = torch.tensor(deal.river_scores, dtype=torch.long, device=device)

        reach = torch.zeros((2, nodes, NUM_COMBOS), dtype=torch.float32, device=device)
        if self.root_reach is not None:
            reach[:, self.tree.root, :] = self.root_reach * valid.float()
        else:
            reach[:, self.tree.root, :] = valid.float()

        strategies: dict[int, torch.Tensor] = {}
        level_decisions: dict[int, torch.Tensor] = {}

        # ---- forward: push reach through levels (static plans, no syncs) --------
        for level_index, plan in enumerate(self.level_plans):
            street_ends = plan["street_ends"]
            if street_ends.numel():
                reach[:, plan["street_end_children"], :] += reach[:, street_ends, :]
            decisions = plan["decisions"]
            if not decisions.numel():
                continue
            node_buckets = buckets[self.t_street[decisions]]  # [L, C]
            strategy = self._node_strategies(decisions, node_buckets)  # [L, C, A]
            if frozen_average is not None and frozen_player is not None:
                rows = plan["actor_rows"][frozen_player]
                if rows.numel():
                    strategy[rows] = frozen_average[decisions[rows].unsqueeze(1), node_buckets[rows]]
            level_decisions[level_index] = decisions
            strategies[level_index] = strategy
            for action_plan in plan["actions"]:
                action = action_plan["action"]
                for player in (0, 1):
                    actor_nodes, actor_children, actor_rows = action_plan["actor_split"][player]
                    if actor_nodes.numel():
                        reach[player, actor_children, :] += (
                            reach[player, actor_nodes, :] * strategy[actor_rows, :, action]
                        )
                    other_nodes, other_children, _ = action_plan["actor_split"][1 - player]
                    if other_nodes.numel():
                        reach[player, other_children, :] += reach[player, other_nodes, :]

        # ---- terminal values (traverser's perspective only) ----------------------
        values = torch.zeros((nodes, NUM_COMBOS), dtype=torch.float32, device=device)
        self._fold_values(values, reach, traverser)
        self._showdown_values(values, reach, scores, valid, traverser)

        # ---- backward: roll values up, accumulate traverser regrets/sums ---------
        for level_index in range(len(self.level_plans) - 1, -1, -1):
            plan = self.level_plans[level_index]
            street_ends = plan["street_ends"]
            if street_ends.numel():
                values[street_ends, :] = values[plan["street_end_children"], :]
            decisions = level_decisions.get(level_index)
            if decisions is None or not decisions.numel():
                continue
            strategy = strategies[level_index]  # [L, C, A]
            child_values = torch.zeros(
                (decisions.shape[0], NUM_COMBOS, self.num_actions), dtype=torch.float32, device=device
            )
            for action_plan in plan["actions"]:
                child_values[action_plan["rows"], :, action_plan["action"]] = values[action_plan["children"], :]

            node_buckets = buckets[self.t_street[decisions]]  # [L, C]
            legal = self.t_legal[decisions].unsqueeze(1).float()  # [L, 1, A]
            node_value = (strategy * child_values).sum(dim=2)  # [L, C]
            values[decisions, :] = node_value

            acted_rows = plan["actor_rows"][traverser]
            if acted_rows.numel():
                own = decisions[acted_rows]
                # Terminal values are opponent-reach weighted already, so the
                # counterfactual regret is simply the child/node value gap.
                regret_increment = (
                    child_values[acted_rows] - node_value[acted_rows].unsqueeze(2)
                ) * legal[acted_rows]
                sum_increment = strategy[acted_rows] * reach[traverser, own, :].unsqueeze(2)
                flat_index = (own.unsqueeze(1) * MAX_BUCKETS + node_buckets[acted_rows]).reshape(-1)
                self.regrets.view(-1, self.num_actions).index_add_(
                    0, flat_index, regret_increment.reshape(-1, self.num_actions)
                )
                self.strategy_sums.view(-1, self.num_actions).index_add_(
                    0, flat_index, sum_increment.reshape(-1, self.num_actions)
                )

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

    def _fold_values(self, values: torch.Tensor, reach: torch.Tensor, player: int) -> None:
        nodes = self.fold_nodes
        if not nodes.numel():
            return
        amount = self.t_fold_committed[nodes].unsqueeze(1)  # [F, 1]
        loser = self.t_fold_loser[nodes]  # [F]
        sign = torch.where(loser == player, -1.0, 1.0).unsqueeze(1)
        opponent_mass = self._opponent_mass(reach[1 - player, nodes, :])  # [F, C]
        values[nodes, :] = sign * amount * opponent_mass

    def _showdown_values(
        self,
        values: torch.Tensor,
        reach: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        player: int,
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

        values[nodes, :] = pots * (worse - better - correction) * valid.float()

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
