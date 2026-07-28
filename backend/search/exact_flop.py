"""Exact-card FLOP resolving: identity 1,326 buckets on flop, turn and river.

Measured 2026-07-28 (`tools/size_blueprint_menus.py` and the tree sizing in
docs/PLAN_V2_STRONGEST_PLAYER.md): with the standard 0.33/0.5/0.75/1/1.4 menu an
exact flop-to-river tree is

    20bb   5,303 nodes (pot 6bb) / 987 (pot 12bb)   -> 0.4 GiB tables, FEASIBLE
    50bb  41,135 / 9,771                            -> feasible for medium pots
   100bb 132,107 / 41,135                           -> 10.5 GiB, too big
   200bb 339,433 / 132,107                          -> impossible

So the shallow depths can be played with **no value network at all**: solve flop,
turn and river exactly to showdown. That matters because 20bb is the worst-served
depth today — LBR beats the 100bb-trained blueprint there by +130.31 bb/100
[+95.22, +165.40], a margin whose confidence interval clears zero decisively.

Deep stacks still need the river/turn nets (P3), which is why this module is a
shallow-depth path rather than a replacement for them.

Unlike the turn (48 runouts, all cacheable) a flop has 1,081 turn+river
completions, so runouts are SAMPLED per iteration in the usual public-chance
style. Deals are cached on first use, so a long solve converges to reusing a
warm set rather than re-scoring boards.
"""

from __future__ import annotations

import random

import numpy as np

from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS, Deal, score_all_combos

# Same standard postflop menu as the turn/river resolvers.
FLOP_FRACTIONS = (0.33, 0.5, 0.75, 1.0, 1.4)
FLOP_RAISE_CAP = 2

#: Depth ceiling for exact flop resolving. Above this the flop tree exceeds the
#: card's memory (100bb needs 10.5 GiB) and the turn/river nets take over.
MAX_EXACT_FLOP_STACK_BB = 60.0


class ExactFlopSampler:
    """Identity-bucket sampler for a fixed 3-card board.

    Buckets are the combo index on flop, turn AND river, so nothing in the
    resulting strategy is card-abstracted. Each sample draws one (turn, river)
    completion; the deal is cached so repeated runouts cost a dict lookup.
    """

    def __init__(self, board: tuple[int, ...], cache_limit: int = 4096) -> None:
        if len(board) != 3:
            raise ValueError("exact flop resolving requires three board cards")
        self.board = tuple(int(card) for card in board)
        if len(set(self.board)) != 3:
            raise ValueError("board cards must be distinct")

        blocked = np.zeros(NUM_COMBOS, dtype=bool)
        for card in self.board:
            blocked |= CARD_IN_COMBO[card]
        self._board_valid = ~blocked
        self.remaining = tuple(card for card in range(52) if card not in self.board)
        self._identity = np.arange(NUM_COMBOS, dtype=np.int32)
        self._deals: dict[tuple[int, int], Deal] = {}
        self._cache_limit = cache_limit

    def bucket_counts(self) -> tuple[int, int, int, int]:
        # Preflop rows are never indexed by a flop-rooted tree.
        return (1, NUM_COMBOS, NUM_COMBOS, NUM_COMBOS)

    def runout_count(self) -> int:
        count = len(self.remaining)
        return count * (count - 1) // 2

    def deal_for_runout(self, turn: int, river: int) -> Deal:
        key = (int(turn), int(river)) if turn < river else (int(river), int(turn))
        cached = self._deals.get(key)
        if cached is not None:
            return cached
        board = self.board + key
        scores = score_all_combos(board)
        valid = scores >= 0
        buckets = np.full((4, NUM_COMBOS), -1, dtype=np.int32)
        # Identity on every postflop street: no card abstraction anywhere.
        for street in (1, 2, 3):
            buckets[street, valid] = self._identity[valid]
        deal = Deal(board=board, buckets=buckets, valid=valid, river_scores=scores)
        if len(self._deals) < self._cache_limit:
            self._deals[key] = deal
        return deal

    def sample(self, rng: random.Random) -> Deal:
        turn, river = rng.sample(self.remaining, 2)
        return self.deal_for_runout(turn, river)


def exact_flop_is_affordable(stack_bb: float, pot_bb: float, node_budget: int = 60_000) -> bool:
    """Would an exact flop-to-river tree fit for this geometry?

    Builds the tree to answer honestly rather than extrapolating; callers use it
    to choose between exact flop resolving and the (net-based) deep path.
    """
    if stack_bb > MAX_EXACT_FLOP_STACK_BB:
        return False
    from backend.solver.gpu.tree import BettingRootState, BettingTree, GpuActionConfig

    behind = max(stack_bb - pot_bb / 2.0, 1.0)
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=FLOP_FRACTIONS,
        max_raises_per_street=FLOP_RAISE_CAP, stack_bb=stack_bb,
    )
    root = BettingRootState(
        street=1, to_act=1, committed=(pot_bb / 2.0, pot_bb / 2.0),
        street_commit=(0.0, 0.0), stacks=(behind, behind),
        acted=(False, False), raises=0, last_increment=1.0,
    )
    return len(BettingTree(config, root_state=root)) <= node_budget
