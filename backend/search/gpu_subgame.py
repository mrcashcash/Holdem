"""Turn/river subgame re-solving with the GPU vector-CFR engine.

At the first decision on the turn (or river), the agent:
1. tracks both players' ranges by replaying the public history through the
   blueprint (vectorized over all 1,326 combos),
2. builds a subgame betting tree rooted at the current street with a RICHER
   bet menu than the blueprint (adds ~1/3-pot and overbet sizes — the
   documented leak of coarse abstractions),
3. solves it with VectorCFR (root reach = the tracked ranges; chance =
   sampled river completions), and
4. plays the remainder of the hand from the solved average strategy.

This is unsafe re-solving (ranges trust the blueprint) solved to the END of
the game — no depth limit is needed because a turn-rooted subgame is tiny
for the dense kernels. Supremus demonstrated this exact pattern (GPU DCFR
re-solving) at +176 mbb/g vs Slumbot.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import (
    _PREFLOP_CLASS,
    NUM_COMBOS,
    Deal,
    DealSampler,
    combos,
    equity_from_scores,
    score_all_combos,
)
from backend.solver.gpu.tree import DECISION, STREET_END, BettingTree, GpuActionConfig

SUBGAME_FRACTIONS = (0.33, 0.5, 0.75, 1.0, 1.4)
SUBGAME_RAISE_CAP = 3
SUBGAME_ITERATIONS = 120

_STREET_BOARD = (0, 3, 4, 5)


class FixedBoardSampler:
    """DealSampler view that completes a fixed partial board.

    Streets already on the board have identical buckets in every sample, so
    they are computed once and reused; per sample only the river-dependent
    parts (scores, exact-equity buckets, validity) are recomputed. For a full
    5-card board the deal is fully deterministic and cached outright.
    """

    def __init__(self, base: DealSampler, partial_board: tuple[int, ...]) -> None:
        self.base = base
        self.partial_board = tuple(int(card) for card in partial_board)
        for name in ("flop_buckets", "turn_buckets", "river_buckets", "flop_samples", "turn_samples"):
            setattr(self, name, getattr(base, name))
        self._full_deal = None
        self._prefix_buckets = None

    def bucket_counts(self):
        return self.base.bucket_counts()

    def sample(self, rng: random.Random):
        if len(self.partial_board) == 5:
            if self._full_deal is None:
                self._full_deal = self.base.for_board(self.partial_board, rng)
            return self._full_deal

        if len(self.partial_board) == 4:
            if getattr(self.base, "potential_aware", False):
                river = self._any_river(rng)
                return self.base.for_board(self.partial_board + (river,), rng)
            if self._prefix_buckets is None:
                template = self.base.for_board(
                    self.partial_board + (self._any_river(rng),), rng
                )
                self._prefix_buckets = template.buckets[:3].copy()  # pre/flop/turn: river-independent
            river = self._any_river(rng)
            board = self.partial_board + (river,)
            scores = score_all_combos(board)
            equity = equity_from_scores(scores)
            valid = scores >= 0
            buckets = np.full((4, NUM_COMBOS), -1, dtype=np.int32)
            buckets[:3] = self._prefix_buckets
            buckets[:3, ~valid] = -1  # combos blocked by the river card
            counts = self.bucket_counts()
            seen = valid & (equity >= 0)
            buckets[3][seen] = np.minimum((equity[seen] * counts[3]).astype(np.int32), counts[3] - 1)
            return Deal(board=board, buckets=buckets, valid=valid, river_scores=scores)

        used = set(self.partial_board)
        remaining = [card for card in range(52) if card not in used]
        completion = rng.sample(remaining, 5 - len(self.partial_board))
        return self.base.for_board(self.partial_board + tuple(completion), rng)

    def _any_river(self, rng: random.Random) -> int:
        remaining = [card for card in range(52) if card not in self.partial_board]
        return rng.choice(remaining)


def partial_board_buckets(board_ids: tuple[int, ...], sampler: DealSampler, seed: int = 11) -> np.ndarray:
    """Bucket per combo per dealt street on the actual board. [-1 = collision]"""
    rng = random.Random(seed)
    if getattr(sampler, "potential_aware", False):
        return sampler.bucket_rows(tuple(board_ids), rng).astype(np.int64)
    streets = sum(1 for size in _STREET_BOARD if len(board_ids) >= size)
    buckets = np.full((4, NUM_COMBOS), -1, dtype=np.int64)
    blocked = np.zeros(NUM_COMBOS, dtype=bool)
    for card in board_ids:
        from backend.solver.gpu.deals import CARD_IN_COMBO

        blocked |= CARD_IN_COMBO[card]
    valid = ~blocked
    buckets[0][valid] = _PREFLOP_CLASS[valid]
    # Delegate to the sampler's shared bucketing (_assign_street /_quantize) so
    # these buckets match the trained strategy tensor exactly — including the
    # distribution-aware (mean x equity-std) scheme. A reimplemented scalar
    # quantile here silently indexed the wrong strategy rows for distributional
    # checkpoints (found in review 2026-07-22).
    for street in (1, 2):
        size = _STREET_BOARD[street]
        if len(board_ids) < size:
            break
        samples = sampler.flop_samples if street == 1 else sampler.turn_samples
        mean_bins = sampler.flop_buckets if street == 1 else sampler.turn_buckets
        sampler._assign_street(buckets, tuple(board_ids[:size]), rng, samples, mean_bins, street, valid)
    if len(board_ids) == 5:
        equity = equity_from_scores(score_all_combos(board_ids))
        seen = valid & (equity >= 0)
        buckets[3][seen] = sampler._quantize(equity[seen], sampler.river_buckets)
    return buckets


def gpu_blueprint_range(agent, game, target_player: int, street_buckets: np.ndarray) -> np.ndarray:
    """Per-combo weight for ``target_player`` from the blueprint along history."""
    tree = agent.tree
    weights = np.ones(NUM_COMBOS, dtype=np.float64)
    node = tree.root
    rng = random.Random(game.hand_number * 131 + 7)
    try:
        for event in game.public_actions:
            if event["action"] == "blind":
                continue
            while tree.kind[node] == STREET_END:
                node = int(tree.children[node][0])
            if tree.kind[node] != DECISION:
                break
            action = agent._translate_event(node, game, event, rng)
            child = int(tree.children[node][action])
            if child < 0:
                break
            if int(tree.actor[node]) == target_player:
                street = int(tree.street[node])
                bucket_row = street_buckets[street]
                usable = bucket_row >= 0
                probabilities = agent.strategy[node, np.clip(bucket_row, 0, None), action]
                weights[usable] *= probabilities[usable]
                weights[~usable] = 0.0
            node = child
    except Exception:
        pass
    total = weights.sum()
    if total <= 1e-12:
        weights = np.where(street_buckets[0] >= 0, 1.0, 0.0)
        total = weights.sum()
    return weights / max(total, 1e-12)


class SubgameSolution:
    """Solved subgame plus everything needed to play from it."""

    def __init__(self, tree: BettingTree, strategy: np.ndarray, sampler: DealSampler, street_buckets: np.ndarray) -> None:
        self.tree = tree
        self.strategy = strategy
        self.sampler = sampler
        self.street_buckets = street_buckets  # buckets on the actual board so far


def build_subgame(agent, game, iterations: int) -> tuple:
    """Shared subgame construction: (solver, tree, street_buckets, ranges).

    ``ranges`` are the blueprint-tracked per-combo reach weights by abstract
    seat; ``solver.root_reach`` is already seeded with them."""
    from backend.vectorized_engine import card_id

    board_ids = tuple(card_id(card) for card in game.community)
    street = {3: 1, 4: 2, 5: 3}[len(board_ids)]
    street_buckets = partial_board_buckets(board_ids, agent.sampler, seed=game.hand_number)

    ranges = np.zeros((2, NUM_COMBOS), dtype=np.float64)
    for engine_player in (0, 1):
        seat = agent._abstract_seat(game, engine_player)
        ranges[seat] = gpu_blueprint_range(agent, game, seat, street_buckets)

    bb = float(game.big_blind)
    matched_entering = 2.0 * min(
        game.contributions[side] - game.round_bets[side] for side in (0, 1)
    ) / bb
    stacks_by_seat = [0.0, 0.0]
    for engine_player in (0, 1):
        seat = agent._abstract_seat(game, engine_player)
        stacks_by_seat[seat] = (game.stacks[engine_player] + game.round_bets[engine_player]) / bb

    config = GpuActionConfig(
        preflop_fractions=SUBGAME_FRACTIONS,
        postflop_fractions=SUBGAME_FRACTIONS,
        max_raises_per_street=SUBGAME_RAISE_CAP,
        stack_bb=100.0,
    )
    tree = BettingTree(
        config,
        start_street=street,
        start_pot=matched_entering,
        start_stacks=(stacks_by_seat[0], stacks_by_seat[1]),
    )
    sampler = FixedBoardSampler(agent.sampler, board_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = VectorCFR(tree, sampler, device=device, seed=game.hand_number, averaging_delay=iterations // 6)
    solver.root_reach = torch.tensor(ranges, dtype=torch.float32, device=solver.device)
    return solver, tree, street_buckets, ranges


# `average_from_sums` was removed on 2026-07-27. It assumed the pre-Phase-2
# dense [nodes, buckets, actions] layout; `VectorCFR.strategy_sums` is now a 2-D
# compact table, so the helper raised on every call and its only callers (the
# safe-gadget tests) had been silently failing since compact storage landed.
# Use `solver.average_strategy_tables()` / `average_strategy_tensor()`, which
# understand the compact layout.


def solve_subgame(agent, game, player: int, iterations: int = SUBGAME_ITERATIONS) -> SubgameSolution:
    solver, tree, street_buckets, _ = build_subgame(agent, game, iterations)
    if str(solver.device) != "cpu":
        # Graph capture + replay: an order of magnitude fewer kernel launches
        # (small trees are launch-bound); numerically identical to eager.
        from backend.solver.gpu.graph import GraphRunner

        runner = GraphRunner(solver, warmup=2)
        solver.regrets.zero_()
        solver.strategy_sums.zero_()
        solver.iteration = 0
        runner.run(iterations, random.Random(game.hand_number * 31 + 5))
    else:
        solver.run(iterations)

    strategy = solver.average_strategy_tables()
    if str(solver.device) != "cpu":
        # Each solve's graph pool otherwise stays RESERVED by the caching
        # allocator; long live-play sessions ratcheted server VRAM into
        # datagen/trainer headroom (2026-07-23 alert). Strategy is already on
        # the CPU — release everything.
        del solver
        torch.cuda.empty_cache()
    return SubgameSolution(tree, strategy.astype(np.float64), agent.sampler, street_buckets)
