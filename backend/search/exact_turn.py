"""Exact-card turn resolving: identity 1,326 buckets on turn AND river.

P1.2 of `docs/PLAN_V2_STRONGEST_PLAYER.md`. The blueprint stores turn strategy in
150 buckets and river strategy in 30, which is the measured ceiling
(LBR beats the serving champion by ~291 bb/100). Re-solving the turn at the
blueprint's own resolution cannot help by construction — that is exactly why the
2026-07-23 bucketed search measured a regression. This module removes the card
abstraction from the played turn+river strategy entirely: every private combo is
its own bucket on both streets, so the solver reasons about actual cards.

`VectorCFR` already carries exact per-combo reach vectors at every node; the
abstraction only ever appeared in how regret/strategy rows were indexed. So
"exact cards" costs memory, not a new kernel — the same trick `ExactRiverSampler`
uses one street later.

Two properties worth stating:

* **A turn board has only 48 possible rivers.** Every deal the solver can ever
  see is enumerable and cached at construction, so sampling is a dict lookup and
  there is no per-iteration bucketing work at all.
* **Turn buckets are river-independent.** A combo's turn bucket is its own index
  in every sample, so turn regrets accumulate per combo across river runouts,
  which is what makes the turn strategy learnable rather than noise.
"""

from __future__ import annotations

import random

import numpy as np

from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS, Deal, score_all_combos

# Standard postflop menu, specified 2026-07-28: 0.33 / 0.5 / 0.75 / 1.0 / 1.4
# pot, on every postflop street. The turn tree multiplies by its river subtree so
# sizes cost steeply here (tools/exact_turn_probe.py measures it), but quality
# takes priority over latency, and the river value net removes the river subtree
# entirely (726 -> 81 nodes), which is what pays for the wider menu.
TURN_FRACTIONS = (0.33, 0.5, 0.75, 1.0, 1.4)
TURN_RAISE_CAP = 2


class ExactTurnSampler:
    """Identity-bucket sampler for a fixed 4-card board.

    Enumerates all 48 river completions once. ``sample`` returns one of them
    uniformly, which is the same public-chance-sampling contract `VectorCFR`
    expects, with zero per-sample cost.
    """

    def __init__(self, board: tuple[int, ...]) -> None:
        if len(board) != 4:
            raise ValueError("exact turn resolving requires four board cards")
        self.board = tuple(int(card) for card in board)
        if len(set(self.board)) != 4:
            raise ValueError("board cards must be distinct")

        blocked = np.zeros(NUM_COMBOS, dtype=bool)
        for card in self.board:
            blocked |= CARD_IN_COMBO[card]
        # Combos that survive the 4-card board; the river card removes more.
        self._board_valid = ~blocked
        self.rivers = tuple(card for card in range(52) if card not in self.board)

        identity = np.arange(NUM_COMBOS, dtype=np.int32)
        self._deals: dict[int, Deal] = {}
        for river in self.rivers:
            scores = score_all_combos(self.board + (river,))
            valid = scores >= 0
            buckets = np.full((4, NUM_COMBOS), -1, dtype=np.int32)
            # Identity on turn and river alike: no card abstraction anywhere in
            # the strategy this solver produces.
            buckets[2, valid] = identity[valid]
            buckets[3, valid] = identity[valid]
            self._deals[river] = Deal(
                board=self.board + (river,),
                buckets=buckets,
                valid=valid,
                river_scores=scores,
            )

    def bucket_counts(self) -> tuple[int, int, int, int]:
        # Preflop/flop rows are never indexed by a turn-rooted tree, so they
        # cost one row each rather than 169/1,326.
        return (1, 1, NUM_COMBOS, NUM_COMBOS)

    def sample(self, rng: random.Random) -> Deal:
        return self._deals[rng.choice(self.rivers)]

    def deal_for_river(self, river: int) -> Deal:
        """The deal for one specific river card (exact-chance enumeration)."""
        return self._deals[int(river)]

    def enumerate_deals(self) -> list[Deal]:
        """All 48 river deals, for an exact rather than sampled chance pass."""
        return [self._deals[river] for river in self.rivers]
