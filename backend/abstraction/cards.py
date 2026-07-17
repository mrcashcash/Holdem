"""Suit-isomorphic canonicalization of (hole, board) card sets.

Card id: 0..51 with rank = id // 4 (0 => deuce .. 12 => ace), suit = id % 4.
Two situations that differ only by a permutation of suits (and by ordering
within the hole pair or within the flop) are strategically identical; the
canonical key is the lexicographic minimum over all 24 suit permutations,
which is small enough to enumerate and provably exact.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations

_SUIT_PERMUTATIONS = tuple(permutations(range(4)))


def rank_of(card: int) -> int:
    return card // 4


def suit_of(card: int) -> int:
    return card % 4


@lru_cache(maxsize=1 << 20)
def canonical_key(hole: tuple[int, ...], board: tuple[int, ...]) -> tuple[int, ...]:
    """Canonical tuple for (hole, board) under suit isomorphism.

    The flop is order-invariant; the turn and river cards keep their street
    identity. The hole pair is order-invariant.
    """
    flop, later = board[:3], board[3:]
    best: tuple[int, ...] | None = None
    for permutation in _SUIT_PERMUTATIONS:
        mapped_hole = sorted(rank_of(card) * 4 + permutation[suit_of(card)] for card in hole)
        mapped_flop = sorted(rank_of(card) * 4 + permutation[suit_of(card)] for card in flop)
        mapped_later = [rank_of(card) * 4 + permutation[suit_of(card)] for card in later]
        candidate = tuple(mapped_hole) + (len(hole),) + tuple(mapped_flop) + tuple(mapped_later)
        if best is None or candidate < best:
            best = candidate
    return best


@lru_cache(maxsize=4096)
def preflop_class(hole: tuple[int, ...]) -> int:
    """Lossless preflop bucket: 169 strategically distinct starting hands.

    Index layout: pairs occupy (r, r); suited hands the upper triangle;
    offsuit hands the lower triangle of a 13x13 grid.
    """
    first, second = sorted(hole, key=rank_of, reverse=True)
    high, low = rank_of(first), rank_of(second)
    if high == low:
        return high * 13 + high
    if suit_of(first) == suit_of(second):
        return low * 13 + high  # upper triangle (row < column): suited
    return high * 13 + low  # lower triangle: offsuit
