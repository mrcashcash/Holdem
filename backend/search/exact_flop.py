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

Deep stacks first try a smaller exact action menu under explicit node and VRAM
limits. They fall back to the promoted blueprint when no tier is safe; a failed
value network is never required for admission.

Unlike the turn (48 runouts, all cacheable) a flop has 1,081 turn+river
completions, so runouts are SAMPLED per iteration in the usual public-chance
style. Deals are cached on first use, so a long solve converges to reusing a
warm set rather than re-scoring boards.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS, Deal, score_all_combos

# Same standard postflop menu as the turn/river resolvers.
# Flop resolve menu. Capped by default for serving: an exact flop-to-river tree
# is 3,203 nodes at 3 sizes/cap2 but 132,107 at 5 sizes/cap2, and the leak this
# fixes (a 19-out draw folded 99.1% by the 150-bucket blueprint vs 17.9% exact) is
# far larger than the cost of two fewer own-bet sizes. Observed opponent sizes are
# inserted into the tree regardless, so responses stay exact.
# Env-overridable so serving can cap the RESOLVE menu while the blueprint keeps
# its own trained sizes, and study profiles can widen it back.
FLOP_FRACTIONS = tuple(
    float(x) for x in os.environ.get(
        "HOLDEM_FLOP_SIZES", "0.33,0.5,0.75,1.0,1.4"
    ).split(",") if x.strip()
)
FLOP_RAISE_CAP = int(os.environ.get("HOLDEM_FLOP_CAP", "2"))

#: There is deliberately no stack-only cutoff. Affordability depends on live
#: pot/stack geometry, selected menu, node count, process allocation, and free
#: VRAM headroom.


def flop_fraction_tiers(
    fractions: tuple[float, ...] = FLOP_FRACTIONS,
) -> tuple[tuple[float, ...], ...]:
    """Rich-to-compact exact-card menus used by the serving admission ladder."""

    clean = tuple(dict.fromkeys(float(value) for value in fractions))
    tiers = [clean]
    if len(clean) >= 3:
        tiers.append((clean[0], clean[-1]))
    if len(clean) >= 2:
        middle = min(clean, key=lambda value: abs(value - 0.75))
        tiers.append((middle,))
    return tuple(dict.fromkeys(tiers))


def flop_fraction_tiers_for_spr(
    fractions: tuple[float, ...],
    spr: float,
) -> tuple[tuple[float, ...], ...]:
    """Skip a predictably explosive rich tier at deep stack-to-pot ratios."""

    tiers = flop_fraction_tiers(fractions)
    rich_max_spr = max(
        0.5,
        float(os.environ.get("HOLDEM_FLOP_RICH_MAX_SPR", "4.0")),
    )
    if spr > rich_max_spr and len(tiers) > 1:
        return tiers[1:]
    return tiers


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


def exact_flop_resource_decision(
    stack_bb: float,
    pot_bb: float,
    node_budget: int | None = None,
):
    """Estimate the worst street-entry flop tree before opening a session."""
    from backend.search.resources import (
        ResolverResourceLimits,
        decide_exact_solver,
    )
    from backend.solver.gpu.tree import BettingRootState, BettingTree, GpuActionConfig

    limits = ResolverResourceLimits.from_env()
    if node_budget is not None:
        limits = ResolverResourceLimits(
            physical_budget_bytes=limits.physical_budget_bytes,
            required_free_headroom_bytes=limits.required_free_headroom_bytes,
            flop_node_budget=max(1, int(node_budget)),
        )

    behind = max(stack_bb - pot_bb / 2.0, 1.0)
    root = BettingRootState(
        street=1, to_act=1, committed=(pot_bb / 2.0, pot_bb / 2.0),
        street_commit=(0.0, 0.0), stacks=(behind, behind),
        acted=(False, False), raises=0, last_increment=1.0,
    )
    last = None
    spr = behind / max(pot_bb, 1e-6)
    for fractions in flop_fraction_tiers_for_spr(FLOP_FRACTIONS, spr):
        config = GpuActionConfig(
            preflop_fractions=(1.0,), postflop_fractions=fractions,
            max_raises_per_street=FLOP_RAISE_CAP, stack_bb=stack_bb,
        )
        tree = BettingTree(config, root_state=root)
        last = decide_exact_solver(
            tree,
            (1, NUM_COMBOS, NUM_COMBOS, NUM_COMBOS),
            street=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            limits=limits,
        )
        if last.allowed:
            return last
    return last


def exact_flop_is_affordable(
    stack_bb: float,
    pot_bb: float,
    node_budget: int | None = None,
) -> bool:
    """Would an exact flop-to-river tree fit for this geometry?

    Builds the tree to answer honestly rather than extrapolating; callers use it
    to choose between exact flop resolving and the (net-based) deep path.
    """
    return exact_flop_resource_decision(stack_bb, pot_bb, node_budget).allowed
