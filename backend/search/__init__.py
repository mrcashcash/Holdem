"""Real-time subgame search on top of the blueprint.

Phase 5 of docs/REDESIGN_PLAN.md. The river re-solver replaces blueprint
play on the final street with an exact-cards CFR solve of the actual river
subgame, using ranges inferred from the blueprint along the public history
(unsafe re-solving in the Libratus taxonomy — the range estimate trusts the
blueprint). Depth-limited turn/flop solving is the planned next step and
should implement the safe (multi-continuation) variant of Brown & Sandholm.
"""

from backend.search.ranges import blueprint_range
from backend.search.river import RiverSubgame, solve_river

__all__ = ["blueprint_range", "RiverSubgame", "solve_river"]
