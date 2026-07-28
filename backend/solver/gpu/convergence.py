"""Trustworthy abstract-game convergence measurement for the GPU VectorCFR.

Background (2026-07-21): the tensor exact-BR probe in ``exploit.py``
(``exact_abstract_exploitability``) over-counts and must not be used. The
convergence question was settled instead by rebuilding the abstract game the
solver actually solves in a COMPACT, exactly-enumerable form and measuring it
with the independent, Kuhn/Leduc-validated best-response in
``backend.solver.best_response`` — which read 0.0 mbb on a converged control
strategy (both for a trusted-MCCFR solution and for the GPU solver's own
average strategy). This module turns that recipe into a reusable check.

Scope: a SINGLE fixed board (a postflop subgame rooted at that board), which is
what the fixed-board control uses. The abstract game is compact because, on a
fixed board, a player's only private information is its equity bucket: chance
picks an ordered bucket pair ``(b0, b1)`` with the empirical probability of
that pairing over non-colliding combos, and a showdown pays the pot times the
exact per-pair win-rate ``E[sign(score0 - score1)]``. The betting structure is
the solver's own ``BettingTree`` (so node/bucket indexing lines up with the
solver's average-strategy tensor), with ``STREET_END`` nodes resolved through.
"""

from __future__ import annotations

import random
from typing import Hashable, Sequence

import numpy as np
import torch

from backend.solver.best_response import exploitability
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS, combos
from backend.solver.gpu.exploit import average_strategy_tensor
from backend.solver.gpu.tree import FOLD_NODE, SHOWDOWN, STREET_END


def _compact_game(solver: VectorCFR, deal, num_actions: int):
    """Build the compact abstract game for a single dealt board.

    Returns ``(game, total_pairs, root_node, win_rate, pair_list)`` where
    ``total_pairs`` is the number of ordered non-colliding combo pairs (the
    normalisation constant that turns summed edges into per-game value).
    """
    tree = solver.tree
    street = int(tree.street[tree.root])  # fixed-board subgame: one street of buckets
    buckets = np.asarray(deal.buckets[street], dtype=np.int64)
    valid = np.asarray(deal.valid, dtype=bool)
    scores = np.asarray(deal.river_scores)

    cards = combos()
    masks = np.zeros(NUM_COMBOS, dtype=np.int64)
    for i, (a, b) in enumerate(cards):
        masks[i] = (1 << int(a)) | (1 << int(b))

    vi = np.where(valid)[0]
    mb, bb, sc = masks[vi], buckets[vi], scores[vi]
    num_buckets = int(buckets[valid].max()) + 1

    overlap = (mb[:, None] & mb[None, :]) != 0
    np.fill_diagonal(overlap, True)
    ok = ~overlap
    sign = np.sign(sc[:, None] - sc[None, :]).astype(np.float64)

    counts = np.zeros((num_buckets, num_buckets))
    wsum = np.zeros((num_buckets, num_buckets))
    bi = np.broadcast_to(bb[:, None], ok.shape)[ok]
    bj = np.broadcast_to(bb[None, :], ok.shape)[ok]
    np.add.at(counts, (bi, bj), 1.0)
    np.add.at(wsum, (bi, bj), sign[ok])
    total = float(counts.sum())
    prob = counts / total
    win_rate = np.where(counts > 0, wsum / np.maximum(counts, 1), 0.0)
    pairs = [
        (b0, b1, float(prob[b0, b1]))
        for b0 in range(num_buckets)
        for b1 in range(num_buckets)
        if counts[b0, b1] > 0
    ]

    def resolve(node: int) -> int:
        while tree.kind[node] == STREET_END:
            node = int(tree.children[node][0])
        return node

    root = resolve(int(tree.root))

    class PlayState:
        __slots__ = ("node", "b0", "b1")

        def __init__(self, node: int, b0: int, b1: int) -> None:
            self.node = resolve(int(node))
            self.b0, self.b1 = b0, b1

        def is_terminal(self) -> bool:
            return tree.kind[self.node] in (FOLD_NODE, SHOWDOWN)

        def is_chance(self) -> bool:
            return False

        def current_player(self) -> int:
            return int(tree.actor[self.node])

        def legal_actions(self) -> Sequence[int]:
            return [a for a in range(num_actions) if tree.children[self.node][a] >= 0]

        def infoset_key(self) -> Hashable:
            return (int(self.node), self.b0 if self.current_player() == 0 else self.b1)

        def child(self, action: int) -> "PlayState":
            return PlayState(int(tree.children[self.node][action]), self.b0, self.b1)

        def utility(self, player: int) -> float:
            if tree.kind[self.node] == FOLD_NODE:
                loser = int(tree.fold_loser[self.node])
                amount = float(tree.fold_loser_committed[self.node])
                v0 = -amount if loser == 0 else amount
            else:
                v0 = float(tree.matched_pot[self.node]) * win_rate[self.b0, self.b1]
            return v0 if player == 0 else -v0

    class ChanceState:
        def is_terminal(self) -> bool:
            return False

        def is_chance(self) -> bool:
            return True

        def current_player(self) -> int:
            return -1

        def legal_actions(self) -> Sequence[int]:
            return []

        def infoset_key(self) -> Hashable:
            return None

        def utility(self, player: int) -> float:
            return 0.0

        def chance_outcomes(self):
            return [(PlayState(root, b0, b1), p) for (b0, b1, p) in pairs]

        def sample_chance(self, rng: random.Random) -> "PlayState":
            r, acc = rng.random(), 0.0
            for (b0, b1, p) in pairs:
                acc += p
                if r <= acc:
                    return PlayState(root, b0, b1)
            return PlayState(root, pairs[-1][0], pairs[-1][1])

    class CompactGame:
        def initial_state(self) -> ChanceState:
            return ChanceState()

        def num_actions(self) -> int:
            return num_actions

    return CompactGame(), total


def abstract_exploitability_mbb(solver: VectorCFR, seed: int = 0) -> float:
    """Exact abstract-game exploitability (mbb/game) of the solver's average
    strategy on its fixed board, via the trusted best response.

    Only valid for a fixed-board (single-street-of-buckets) solver — the
    control used to verify convergence. Returns >= 0; ~0 means the average
    strategy has reached the abstract Nash equilibrium.
    """
    deal = solver.sampler.sample(random.Random(seed))
    average = average_strategy_tensor(solver).numpy()
    game, total = _compact_game(solver, deal, solver.num_actions)

    def policy(key: Hashable, actions: Sequence[int]) -> np.ndarray:
        node, bucket = key
        row = average[node, bucket, list(actions)].astype(np.float64)
        s = row.sum()
        return row / s if s > 0 else np.full(len(actions), 1.0 / len(actions))

    return exploitability(game, policy) / total * 1000.0
