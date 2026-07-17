"""Card and action abstraction for the hold'em blueprint solver.

Cards are compact ids 0..51 (``(rank-2)*4 + suit``, matching
``vectorized_engine.card_id``). The card abstraction maps any (hole, board)
to a small bucket id per street: 169 lossless preflop classes, Wasserstein
k-means over equity histograms on flop/turn (potential-aware, Ganzfried &
Sandholm AAAI-14), and equity quantiles on the river. The action abstraction
restricts betting to a small menu of pot-fraction raises with pseudo-harmonic
translation for off-tree opponent sizes.
"""

from backend.abstraction.actions import ActionAbstraction, pseudo_harmonic_weights
from backend.abstraction.buckets import CardAbstraction
from backend.abstraction.cards import canonical_key, preflop_class

__all__ = [
    "ActionAbstraction",
    "CardAbstraction",
    "canonical_key",
    "preflop_class",
    "pseudo_harmonic_weights",
]
