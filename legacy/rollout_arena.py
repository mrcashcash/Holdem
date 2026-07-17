"""Native-ready interleaving primitives for independent self-play hands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .vectorized_engine import resolve_vectorized_runtime


@dataclass(frozen=True)
class RolloutBackendCapabilities:
    """Stable contract for swapping in a validated native simulator later."""

    mode: str
    vectorized_inference: bool
    native_ready: bool
    validated_native: bool
    reason: str = ""


PYTHON_BATCHED_CAPABILITIES = RolloutBackendCapabilities(
    mode="python-batched",
    vectorized_inference=True,
    native_ready=True,
    validated_native=False,
    reason="Reference Python rules engine with batched neural inference",
)


def active_rollout_capabilities() -> RolloutBackendCapabilities:
    """Expose the runtime-selected acceleration mode without changing game rules."""
    runtime = resolve_vectorized_runtime()
    if not runtime.enabled:
        return RolloutBackendCapabilities("python-batched", True, True, False, runtime.reason)
    return RolloutBackendCapabilities(runtime.mode, True, True, False, runtime.reason)


@dataclass
class ArenaHand:
    """Per-hand state deliberately kept separate while inference is batched."""

    game: Any
    learner_player: int
    opponent: Any
    path: Any
    profile: str
    learner_state: Any = None
    opponent_state: Any = None
    recorded_counterfactual: bool = False
    recorded_solver_teacher: bool = False
    safety: int = 0
    cache: dict[tuple[int, tuple[float, ...]], Any] = field(default_factory=dict)


class BatchedRolloutArena:
    """Round-robin scheduler with no shared game, RNG, or recurrent state."""

    capabilities = PYTHON_BATCHED_CAPABILITIES

    def __init__(self, hands: list[ArenaHand]) -> None:
        self.hands = hands
        self._active_hands: list[ArenaHand] = list(hands)
        self._active_dirty = False

    def active(self) -> list[ArenaHand]:
        # A training tick queries the same active set for learner, heuristic,
        # and neural-opponent groups. Rebuild only after an action mutates a
        # game instead of scanning the full arena for every query.
        if self._active_dirty:
            self._active_hands = [hand for hand in self.hands if not hand.game.hand_complete]
            self._active_dirty = False
        return self._active_hands

    def select(self, predicate: Callable[[ArenaHand], bool]) -> list[ArenaHand]:
        return [hand for hand in self.active() if predicate(hand)]

    def note_action(self, hand: ArenaHand) -> None:
        hand.safety += 1
        self._active_dirty = True
        if hand.safety > 100:
            raise RuntimeError("Strategic self-play hand exceeded the 100-action safety limit")

    @property
    def complete(self) -> bool:
        return not self.active()
