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

Regrets/strategy sums use compact decision-node, per-street storage. The four
street shards are concatenated into one float32 ``[stored rows, actions]``
tensor so CUDA-graph in-place operations remain available while terminal
nodes and unused per-street bucket columns consume no VRAM.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS, Deal, DealSampler, combos
from backend.solver.gpu.storage import CompactTableLayout
from backend.solver.gpu.tree import DECISION, FOLD_NODE, SHOWDOWN, STREET_END, BettingTree

# Legacy compatibility for modules/checkpoints that used the old global cap.
# New storage is dynamic and does not use this value for indexing.
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
        batch_boards: int = 1,
        fused_forward: bool = True,
        independent_situations: bool = False,
    ) -> None:
        # SITUATION BATCHING (P2). `batch_boards` folds B boards into the combo
        # axis but they share one regret table — B chance samples of ONE game.
        # With `independent_situations` the same axis instead carries B separate
        # games: regrets/strategy sums get a batch dimension and every index
        # gains a per-situation offset, so B equilibria are solved at once.
        #
        # This is the only way to use the idle GPU here. These solves are
        # latency-bound (~530 dependent ops/iteration, 123x above the bandwidth
        # floor), so widening each kernel is free while adding processes is not:
        # 4 worker processes measured 0.4 rows/s against 1.20 for one, because
        # separate CUDA contexts time-slice instead of interleaving.
        self.independent_situations = bool(independent_situations)
        # Forward-pass fusion (P2.1). Profiling showed these solves are
        # latency-bound on a serial chain of ~530 dependent GPU ops per
        # iteration at ~26.5us each — not bandwidth-bound (123x above the
        # floor) and not occupancy-bound (batching gave only 1.47x). 192 of the
        # 265 forward ops per traversal were the per-action x per-player reach
        # loops; fusion collapses them into two index_add_ calls per level.
        # Keep the flag: it is the control arm for the bit-identical
        # equivalence test in tests/test_gpu_fused_forward.py.
        self.fused_forward = fused_forward
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
        self.batch_boards = max(1, batch_boards)
        # Optional [2, NUM_COMBOS] root reach (re-solving subgames start from
        # tracked ranges instead of uniform deals).
        self.root_reach: torch.Tensor | None = None

        actions = tree.config.num_actions
        self.num_actions = actions
        self.layout = CompactTableLayout(tree, self.sampler.bucket_counts())
        self.bucket_counts = self.layout.bucket_counts
        # One table per independent situation, stacked along the row axis.
        self.situations = self.batch_boards if self.independent_situations else 1
        self.regrets = torch.zeros(
            (self.layout.total_rows * self.situations, actions),
            dtype=torch.float32,
            device=self.device,
        )
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
        self.t_node_base = torch.tensor(self.layout.node_base, dtype=torch.long, device=device)
        # Per-column offset into the stacked tables: column c of the combo axis
        # belongs to situation c // NUM_COMBOS, whose table starts at
        # situation * total_rows. Zero when situations are not independent, so
        # the shared-table path is untouched.
        if self.independent_situations and self.situations > 1:
            self.t_situation_offset = torch.repeat_interleave(
                torch.arange(self.situations, dtype=torch.long, device=device)
                * self.layout.total_rows,
                NUM_COMBOS,
            )
        else:
            self.t_situation_offset = None

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

        # Combos holding each card — static; precomputed so the showdown loop
        # is free of torch.nonzero (a sync op, illegal under graph capture).
        self.card_holders = [
            torch.tensor(
                np.flatnonzero((CARD_IN_COMBO[card])), dtype=torch.long, device=device
            )
            for card in range(52)
        ]

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

            # Fused forward plan: one flat (parent -> child) edge list for the
            # whole level instead of a Python loop over actions x players.
            #
            # Per edge the semantics are unchanged: the ACTOR's reach is scaled
            # by their probability of the action, the opponent's passes through.
            # Encoding both players in one flat index over a [2*nodes, width]
            # view lets a single index_add_ do each half.
            #
            # Safe and deterministic because parents sit exactly one level above
            # their children (so reads and writes never alias within a level)
            # and every child has exactly one (parent, action) edge (so no
            # destination index repeats, which is what would otherwise make
            # index_add_ non-deterministic under atomics).
            edge_src, edge_dst, edge_row, edge_action = [], [], [], []
            for action in range(tree.config.num_actions):
                legal_rows = np.flatnonzero(tree.legal[decisions_np, action])
                if legal_rows.size == 0:
                    continue
                acting = decisions_np[legal_rows]
                children = tree.children[acting, action]
                if np.any(children < 0):
                    raise ValueError("legal action with no child in the betting tree")
                edge_src.append(acting)
                edge_dst.append(children)
                edge_row.append(legal_rows)
                edge_action.append(np.full(legal_rows.size, action, dtype=np.int64))
            if edge_src:
                src = np.concatenate(edge_src)
                dst = np.concatenate(edge_dst).astype(np.int64)
                row = np.concatenate(edge_row).astype(np.int64)
                act = np.concatenate(edge_action)
                actor = tree.actor[src].astype(np.int64)
                node_count = len(tree)
                if np.unique(dst).size != dst.size:
                    raise ValueError("a node has more than one parent edge; fusion would race")
                plan["fused"] = {
                    "actor_src": torch.tensor(actor * node_count + src, dtype=torch.long, device=device),
                    "actor_dst": torch.tensor(actor * node_count + dst, dtype=torch.long, device=device),
                    "opponent_src": torch.tensor((1 - actor) * node_count + src, dtype=torch.long, device=device),
                    "opponent_dst": torch.tensor((1 - actor) * node_count + dst, dtype=torch.long, device=device),
                    # Index into strategy viewed as [L * A, width].
                    "strategy_index": torch.tensor(
                        row * tree.config.num_actions + act, dtype=torch.long, device=device
                    ),
                }
            else:
                plan["fused"] = None
            self.level_plans.append(plan)

    # -- strategy --------------------------------------------------------------

    def _node_strategies(self, node_ids: torch.Tensor, node_buckets: torch.Tensor) -> torch.Tensor:
        """Regret-matched strategy [L, C, A] for decision nodes ``node_ids``.

        ``node_buckets`` [L, C]: bucket of the acting player's combo (already
        street-resolved); invalid combos may carry bucket 0 — their reach is
        zero so their contribution vanishes.
        """
        rows = self.t_node_base[node_ids].unsqueeze(1) + node_buckets
        if self.t_situation_offset is not None:
            rows = rows + self.t_situation_offset
        gathered = self.regrets[rows]  # [L, C, A]
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
        # Deal preparation (board sampling + bucket computation) is CPU work
        # worth ~25-40ms per iteration; prefetch it on a thread so the GPU
        # never waits. Identical update sequence — zero quality difference.
        import queue
        import threading

        deals: queue.Queue = queue.Queue(maxsize=4)
        stop = threading.Event()

        def producer() -> None:
            try:
                for _ in range(iterations):
                    if stop.is_set():
                        return
                    deals.put([self.sampler.sample(self.rng) for _ in range(self.batch_boards)])
            except BaseException as error:  # surface producer death to the consumer
                deals.put(error)

        worker = threading.Thread(target=producer, daemon=True)
        worker.start()
        try:
            for _ in range(iterations):
                self.iteration += 1
                # Timeout guards against a silently dead producer: blocking
                # forever here is how the 2026-07-19 4-hour wedge presented.
                deal = deals.get(timeout=600)
                if isinstance(deal, BaseException):
                    raise deal
                # Alternating updates (Burch et al. JAIR 2019): each player's
                # regrets are updated in their own pass against the opponent's
                # freshly regret-matched strategy.
                self._iterate(deal, traverser=0)
                self._iterate(deal, traverser=1)
                self._discount()
        finally:
            stop.set()
            while not deals.empty():
                deals.get_nowait()
            worker.join(timeout=5)

    def _discount(self) -> None:
        t = float(self.iteration)
        positive_factor = t**self.discount_alpha / (t**self.discount_alpha + 1.0)
        negative_factor = t**self.discount_beta / (t**self.discount_beta + 1.0)
        self.regrets.mul_(
            torch.where(
                self.regrets > 0,
                torch.tensor(positive_factor, dtype=torch.float32, device=self.device),
                torch.tensor(negative_factor, dtype=torch.float32, device=self.device),
            )
        )
        if self.iteration > self.averaging_delay:
            self.strategy_sums *= (t / (t + 1.0)) ** self.discount_gamma
        else:
            self.strategy_sums.zero_()
        # float32 headroom guard: regret matching and average extraction are
        # invariant to uniform positive scaling, so shrink before precision
        # is lost (increments must stay above the ulp of the running sums).
        if self.iteration % 500 == 0:
            peak = float(self.regrets.abs().max().item())
            if peak > 1e7:
                self.regrets *= 1e6 / peak
            sums_peak = float(self.strategy_sums.abs().max().item())
            if sums_peak > 1e7:
                self.strategy_sums *= 1e6 / sums_peak

    def _iterate(
        self,
        deal: Deal | list[Deal],
        traverser: int = 0,
        frozen_average: torch.Tensor | None = None,
        frozen_player: int | None = None,
    ) -> None:
        """One traversal over one deal or a batch of deals.

        Batching folds the boards into the combo axis (width B*NUM_COMBOS):
        the tree walk is board-agnostic, so forward/backward code is shared,
        and the regret/strategy scatter naturally sums the batch — an exact
        mini-batch of unbiased chance samples. Only root seeding and the
        terminal functions are board-aware. With ``frozen_average``/
        ``frozen_player`` set, that player follows the given average-strategy
        tensor instead of regret matching (CFR-BR)."""
        device = self.device
        nodes = len(self.tree)
        if isinstance(deal, list) or isinstance(deal, Deal):
            deals = deal if isinstance(deal, list) else [deal]
            batch = len(deals)
            valid = torch.tensor(np.concatenate([d.valid for d in deals]), device=device)  # [B*C]
            buckets = torch.tensor(
                np.concatenate([d.buckets for d in deals], axis=1), dtype=torch.long, device=device
            ).clamp_min(0)  # [4, B*C]
            scores = torch.tensor(
                np.stack([d.river_scores for d in deals]), dtype=torch.long, device=device
            )  # [B, C]
        else:
            # Graph-capture sentinel: read the runner's static input buffers
            # (backend/solver/gpu/graph.py) so replays see fresh deal data.
            batch = deal.batch
            valid, buckets, scores = self._graph_inputs(None)
        self._batch = batch

        width = batch * NUM_COMBOS
        reach = torch.zeros((2, nodes, width), dtype=torch.float32, device=device)
        if self.root_reach is not None:
            reach[:, self.tree.root, :] = self.root_reach.repeat(1, batch)[:, :width] * valid.float()
        else:
            # Probability-normalized reach keeps terminal values (and thus
            # cumulative regrets) ~3 orders of magnitude smaller — unnormalized
            # masses saturated float32 (ulp 16 at 2e8) and froze learning.
            per_board = valid.float().view(batch, NUM_COMBOS)
            per_board = per_board / per_board.sum(dim=1, keepdim=True).clamp_min(1.0)
            reach[:, self.tree.root, :] = per_board.reshape(-1)

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
            if frozen_average is not None:
                if frozen_player is not None:
                    rows = plan["actor_rows"][frozen_player]
                    if rows.numel():
                        strategy[rows] = frozen_average[decisions[rows].unsqueeze(1), node_buckets[rows]]
                else:
                    # frozen_player=None freezes BOTH players: a pure evaluation
                    # pass of the given average strategy (used by safe
                    # re-solving to price the opponent's opt-out alternative).
                    strategy = frozen_average[decisions.unsqueeze(1), node_buckets]
            level_decisions[level_index] = decisions
            strategies[level_index] = strategy
            fused = plan.get("fused") if self.fused_forward else None
            if fused is not None:
                # Two ops per level instead of up to 4 x |actions|. Bit-identical
                # to the loop below: every child receives exactly one edge, so
                # there is no summation-order freedom to change the result.
                reach_flat = reach.view(2 * nodes, width)
                strategy_flat = strategy.permute(0, 2, 1).reshape(-1, width)
                reach_flat.index_add_(
                    0,
                    fused["actor_dst"],
                    reach_flat[fused["actor_src"]] * strategy_flat[fused["strategy_index"]],
                )
                reach_flat.index_add_(
                    0, fused["opponent_dst"], reach_flat[fused["opponent_src"]]
                )
            else:
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
        values = torch.zeros((nodes, width), dtype=torch.float32, device=device)
        self._fold_values(values, reach, traverser)
        self._showdown_values(values, reach, scores, valid, traverser)
        # Depth-limited trees: an external evaluator prices HORIZON terminals
        # from the players' reach there (backend/search/depth_limited.py).
        if getattr(self, "_horizon_hook", None) is not None:
            self._horizon_hook(values, reach, traverser, deal, valid)

        # ---- backward: roll values up, accumulate traverser regrets/sums ---------
        for level_index in range(len(self.level_plans) - 1, -1, -1):
            plan = self.level_plans[level_index]
            street_ends = plan["street_ends"]
            if street_ends.numel():
                values[street_ends, :] = values[plan["street_end_children"], :]
            decisions = level_decisions.get(level_index)
            if decisions is None or not decisions.numel():
                continue
            strategy = strategies[level_index]  # [L, width, A]
            child_values = torch.zeros(
                (decisions.shape[0], width, self.num_actions), dtype=torch.float32, device=device
            )
            for action_plan in plan["actions"]:
                child_values[action_plan["rows"], :, action_plan["action"]] = values[action_plan["children"], :]

            node_buckets = buckets[self.t_street[decisions]]  # [L, C]
            legal = self.t_legal[decisions].unsqueeze(1).float()  # [L, 1, A]
            # Vector-CFR value aggregation (b-inary/OpenSpiel convention):
            # at the TRAVERSER's nodes, v = sum_a sigma(a) v(child_a);
            # at the OPPONENT's nodes, v = plain sum over children — the
            # opponent's sigma was already applied via the reach folded into
            # the terminal values. Weighting again applies it twice (and with
            # the traverser's bucket indexing, which isn't even their hand).
            node_value = torch.where(
                (self.t_actor[decisions] == traverser).unsqueeze(1),
                (strategy * child_values).sum(dim=2),
                child_values.sum(dim=2),
            )
            values[decisions, :] = node_value

            acted_rows = plan["actor_rows"][traverser]
            if acted_rows.numel():
                own = decisions[acted_rows]
                # Terminal values are opponent-reach weighted already, so the
                # counterfactual regret is simply the child/node value gap.
                # Board-colliding combos (bucket -1, clamped to 0) carry
                # nonzero fold values — mask them or they pollute bucket 0.
                regret_increment = (
                    child_values[acted_rows] - node_value[acted_rows].unsqueeze(2)
                ) * legal[acted_rows] * valid.float().unsqueeze(0).unsqueeze(2)
                sum_increment = strategy[acted_rows] * reach[traverser, own, :].unsqueeze(2)
                flat_index = self.t_node_base[own].unsqueeze(1) + node_buckets[acted_rows]
                if self.t_situation_offset is not None:
                    flat_index = flat_index + self.t_situation_offset
                flat_index = flat_index.reshape(-1)
                self.regrets.index_add_(
                    0,
                    flat_index,
                    regret_increment.reshape(-1, self.num_actions),
                )
                self.strategy_sums.index_add_(
                    0,
                    flat_index,
                    sum_increment.reshape(-1, self.num_actions),
                )

        # Root counterfactual values of this traversal (traverser's, per combo,
        # opponent-reach weighted). The safe re-solving gadget reads these.
        self._last_root_values = values[self.tree.root, :]
        # Opt-in capture of the full per-node reach and value tensors. CFV
        # datagen harvests one (belief, value) sample per INTERIOR node from a
        # single evaluation pass: the reach at a node IS the belief there, and
        # its value under the frozen average strategy is the corresponding CFV.
        # Off by default because holding these alive would add ~12 MB per
        # traversal to every training iteration for no benefit.
        if getattr(self, "capture_internals", False):
            self._last_reach = reach
            self._last_values = values

    def tree_kind(self, node_ids: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.tree.kind, device=self.device)[node_ids]

    # -- terminal math -----------------------------------------------------------

    def _opponent_mass(self, opponent_reach: torch.Tensor) -> torch.Tensor:
        """Reach mass of compatible opponent combos, per hero combo.

        Accepts [.., C] or the batched flat layout [.., B*C]; totals and
        per-card sums are always taken within each board's segment.
        """
        width = opponent_reach.shape[-1]
        batch = width // NUM_COMBOS
        shaped = opponent_reach.reshape(*opponent_reach.shape[:-1], batch, NUM_COMBOS)
        total = shaped.sum(dim=-1, keepdim=True)  # [.., B, 1]
        per_card = shaped @ self.t_card_in_combo.T  # [.., B, 52]
        card_a = self.t_combos[:, 0]
        card_b = self.t_combos[:, 1]
        blocked = per_card[..., card_a] + per_card[..., card_b] - shaped
        return (total - blocked).reshape(*opponent_reach.shape)

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
        nodes: torch.Tensor | None = None,
        pots: torch.Tensor | None = None,
    ) -> None:
        # nodes/pots overrides let depth-limited horizon evaluators reuse this
        # trusted kernel on their own node set (backend/search/depth_limited).
        if nodes is None:
            nodes = self.showdown_nodes
        if not nodes.numel():
            return
        # scores: [B, C]; reach/values use the flat [.., B*C] layout.
        if scores.dim() == 1:  # single-deal callers (exploit paths, tests)
            scores = scores.unsqueeze(0)
        batch = scores.shape[0]
        showdowns = nodes.shape[0]
        order = torch.argsort(scores, dim=1)  # [B, C]; invalid (-1) sort first
        sorted_scores = torch.gather(scores, 1, order)
        boundaries_left = torch.searchsorted(sorted_scores, scores, side="left")  # [B, C]
        boundaries_right = torch.searchsorted(sorted_scores, scores, side="right")
        pots = (self.t_matched_pot[nodes] if pots is None else pots).view(showdowns, 1, 1)
        card_in_combo = self.t_card_in_combo > 0  # [52, C] bool
        combo_cards = self.t_combos  # [C, 2]

        opponent = (reach[1 - player, nodes, :] * valid.float()).view(showdowns, batch, NUM_COMBOS)
        order_e = order.unsqueeze(0).expand(showdowns, -1, -1)
        ordered = torch.gather(opponent, 2, order_e)  # [S, B, C] in per-board score order
        prefix = torch.cumsum(ordered, dim=2)
        total = prefix[..., -1:]
        zeros = torch.zeros((showdowns, batch, 1), device=self.device, dtype=prefix.dtype)
        padded = torch.cat([zeros, prefix], dim=2)
        left_e = boundaries_left.unsqueeze(0).expand(showdowns, -1, -1)
        right_e = boundaries_right.unsqueeze(0).expand(showdowns, -1, -1)
        worse = torch.gather(padded, 2, left_e)  # opponents with lower score
        better = total - torch.gather(padded, 2, right_e)

        # Blocker correction: subtract the worse/better mass of opponent
        # combos sharing a card with the hero. Small trees use one batched
        # pass with cards as a channel dim (few large kernels — what makes
        # graph-replayed subgame solves fast); the big blueprint tree keeps
        # the per-card loop to bound memory.
        channel_bytes = showdowns * batch * (NUM_COMBOS + 1) * 52 * 4 * 3
        if channel_bytes < 2_000_000_000:
            members_all = (self.t_card_in_combo > 0).T.float()  # [C, 52]
            members_ordered = members_all[order]  # [B, C, 52]
            masked = ordered.unsqueeze(3) * members_ordered.unsqueeze(0)  # [S, B, C, 52]
            channel_prefix = torch.cumsum(masked, dim=2)
            channel_total = channel_prefix[:, :, -1:, :]
            channel_padded = torch.cat(
                [torch.zeros_like(channel_prefix[:, :, :1, :]), channel_prefix], dim=2
            )
            left_e4 = left_e.unsqueeze(3).expand(-1, -1, -1, 52)
            right_e4 = right_e.unsqueeze(3).expand(-1, -1, -1, 52)
            worse_by_card = torch.gather(channel_padded, 2, left_e4)
            better_by_card = channel_total - torch.gather(channel_padded, 2, right_e4)
            per_card_gap = worse_by_card - better_by_card  # [S, B, C, 52]
            combo_channels = self.t_combos.view(1, 1, NUM_COMBOS, 2).expand(
                showdowns, batch, -1, -1
            )
            correction = torch.gather(per_card_gap, 3, combo_channels).sum(dim=3)
        else:
            correction = torch.zeros_like(worse)  # (worse_blocked - better_blocked)
            for card in range(52):
                members = card_in_combo[card]  # [C] combos containing this card
                members_ordered = torch.gather(members.expand(batch, -1), 1, order)  # [B, C]
                masked = ordered * members_ordered.unsqueeze(0)
                card_prefix = torch.cumsum(masked, dim=2)
                card_total = card_prefix[..., -1:]
                card_padded = torch.cat([zeros, card_prefix], dim=2)
                holders = self.card_holders[card]
                left_h = boundaries_left[:, holders].unsqueeze(0).expand(showdowns, -1, -1)
                right_h = boundaries_right[:, holders].unsqueeze(0).expand(showdowns, -1, -1)
                worse_blocked = torch.gather(card_padded, 2, left_h)
                better_blocked = card_total - torch.gather(card_padded, 2, right_h)
                correction[:, :, holders] += worse_blocked - better_blocked

        values[nodes, :] = (pots * (worse - better - correction)).reshape(
            showdowns, batch * NUM_COMBOS
        ) * valid.float()

    # -- outputs -----------------------------------------------------------------

    def average_strategy_compact(self, situation: int = 0) -> torch.Tensor:
        """Normalized average strategy in compact ``[stored rows, A]`` form.

        With independent situations the tables are stacked along the row axis,
        so ``situation`` selects which block to extract.
        """
        rows = self.layout.total_rows
        sums = self.strategy_sums[situation * rows : (situation + 1) * rows]
        # Only serving/diagnostic extraction needs a legal mask expanded per
        # stored row. Build it transiently so training does not spend VRAM on
        # a third table-sized persistent tensor.
        legal = torch.tensor(self.layout.legal_rows(), device=self.device)
        totals = sums.sum(dim=1, keepdim=True)
        legal_counts = legal.sum(dim=1, keepdim=True).clamp_min(1).to(sums.dtype)
        uniform = legal.to(sums.dtype) / legal_counts
        return torch.where(totals > 0, sums / totals.clamp_min(1e-30), uniform) * legal

    def average_strategy_tensor(self, situation: int = 0) -> torch.Tensor:
        """Dense ``[nodes, max street buckets, A]`` view for small consumers.

        Blueprint training and checkpoints never materialize this view. It is
        retained for subgame solving and diagnostic code whose trees are small.
        """
        compact = self.average_strategy_compact(situation)
        dense = torch.zeros(
            (len(self.tree), self.layout.max_buckets, self.num_actions),
            dtype=compact.dtype,
            device=self.device,
        )
        for shard in self.layout.shards:
            if not shard.decision_count:
                continue
            nodes = torch.tensor(shard.decision_nodes, dtype=torch.long, device=self.device)
            dense[nodes, : shard.bucket_count] = compact[
                shard.start : shard.stop
            ].reshape(shard.decision_count, shard.bucket_count, self.num_actions)
        return dense

    def average_strategy_tables(self, situation: int = 0) -> np.ndarray:
        """Dense CPU average-strategy view for serving small subgames."""
        return self.average_strategy_tensor(situation).cpu().numpy()

    def load_tables(self, regrets: np.ndarray, strategy_sums: np.ndarray) -> str:
        """Load compact tables or migrate a legacy dense checkpoint in-place."""
        if regrets.ndim == 3:
            regrets = self.layout.compact_from_dense(regrets)
            strategy_sums = self.layout.compact_from_dense(strategy_sums)
            source = "legacy-dense"
        else:
            self.layout.validate_compact(regrets)
            self.layout.validate_compact(strategy_sums)
            source = "compact-v2"
        self.regrets.copy_(
            torch.as_tensor(regrets, dtype=torch.float32, device=self.device)
        )
        self.strategy_sums.copy_(
            torch.as_tensor(strategy_sums, dtype=torch.float32, device=self.device)
        )
        return source

    def storage_report(self) -> dict:
        dense_rows = len(self.tree) * max(self.bucket_counts)
        compact_rows = self.layout.total_rows
        table_bytes = compact_rows * self.num_actions * 4
        return {
            **self.layout.state(),
            "actions": self.num_actions,
            "regret_bytes": table_bytes,
            "strategy_sum_bytes": table_bytes,
            "table_bytes_total": table_bytes * 2,
            "legacy_dense_rows": dense_rows,
            "row_reduction_fraction": round(
                1.0 - compact_rows / max(dense_rows, 1),
                6,
            ),
        }
