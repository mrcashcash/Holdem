"""Continual re-solving across turn and river with exact cards (P1.3).

The 2026-07-23 bucketed search regressed (-86 bb/100 at 500 iterations) for two
reasons, both addressed here:

1. **No information edge.** Re-solving at the blueprint's own 150/30-bucket
   resolution can only re-derive the blueprint's answer from fewer samples.
   Every solve here is exact-card (`ExactTurnSampler` / `ExactRiverSampler`),
   which is a genuine edge — see `tests/test_exact_turn.py::DrawFoldLeakTests`.
2. **Self-range inconsistency.** The old range tracker assumed the agent's own
   past actions followed the blueprint, so a searching agent solved subgames
   conditioned on a falsified own-history. Here the agent's range advances by the
   per-combo likelihood of the action *the solution that actually chose it*
   assigned. The blueprint is consulted exactly once, to seed the session at its
   entry street, and never again.

What this adds over Phase 4 (river-only): the session enters at the **turn**, so
river ranges are carried forward from the turn solves actually played instead of
being re-derived from the blueprint at the street boundary. That re-derivation is
the same falsified-history error one street later.

Street responsibilities, and why they differ:

* **Flop decisions** use a flop-rooted exact solve chosen by a richest-safe
  action-menu ladder. Node and VRAM admission happens before solver allocation;
  if no tier fits, that flop uses the promoted blueprint.
* **Turn decisions** use a turn-rooted exact solve (turn + river betting).
* **River decisions re-solve river-rooted with the actual river card.** A
  turn-rooted tree indexes river rows by combo only -- there is no river-card
  axis -- so its river strategy is averaged over all 48 runouts and is *card
  blind*. It is a value estimator for the turn decision, not a playable river
  strategy. (A river value network is the principled replacement for that
  horizon; see P3a.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.search.exact_river import (
    MIN_RESOLVE_ITERATIONS,
    PendingLikelihood,
    NodeStrategy,
    RiverResolveError,
    _blueprint_ranges,
    _event_action,
    _normalize_range,
    _resolve_at,
    _seat,
)
from backend.search.gpu_subgame import partial_board_buckets
from backend.solver.gpu.deals import NUM_COMBOS, combos
from backend.solver.gpu.tree import DECISION
from backend.vectorized_engine import card_id

FLOP_STREET, TURN_STREET, RIVER_STREET = 1, 2, 3
_BOARD_CARDS = {FLOP_STREET: 3, TURN_STREET: 4, RIVER_STREET: 5}
POSTFLOP_STREETS = (FLOP_STREET, TURN_STREET, RIVER_STREET)


class ContinualResolveError(RiverResolveError):
    """The session cannot continue; the caller must fall back to the blueprint."""


@dataclass
class ContinualSession:
    """Beliefs and played-policy history for one hand, from the entry street on.

    ``ranges`` is [2, NUM_COMBOS] indexed by ABSTRACT SEAT (button = 0), holding
    each player's exact per-combo range given the public history processed so
    far. ``pending`` maps a public-action index to the per-combo likelihood of
    the action we ourselves chose there, recorded by
    :func:`register_selected_action` from the solution that chose it.
    """

    key: tuple[int, int]
    board: tuple[int, ...]
    controlled_player: int
    entry_street: int
    ranges: np.ndarray
    next_event: int
    pending: dict[int, PendingLikelihood] = field(default_factory=dict)
    frontiers: dict[int, "PendingFrontier"] = field(default_factory=dict)
    failed: bool = False
    failure: str | None = None
    resolves: int = 0
    own_updates: int = 0
    opponent_updates: int = 0
    sampler_cache: dict = field(default_factory=dict)
    blueprint_bucket_cache: dict = field(default_factory=dict)


@dataclass
class ContinualSolution:
    tree: object
    strategy: NodeStrategy
    session: ContinualSession
    diagnostics: dict
    node: int


@dataclass
class PendingFrontier:
    """Opponent policy at the child reached by our previously selected action."""

    actor_seat: int
    street: int
    tree: object
    node: int
    probability: np.ndarray


def _live_mask(board: tuple[int, ...]) -> np.ndarray:
    holdings = combos()
    live = np.ones(NUM_COMBOS, dtype=bool)
    for card in board:
        live &= (holdings[:, 0] != card) & (holdings[:, 1] != card)
    return live


def _first_event_on_street(game, street: int) -> int:
    for index, event in enumerate(game.public_actions):
        if int(event.get("street", -1)) >= street:
            return index
    return len(game.public_actions)


def open_session(
    agent,
    game,
    controlled_player: int,
    key: tuple[int, int],
    sessions: dict[tuple[int, int], ContinualSession],
    entry_street: int = TURN_STREET,
) -> ContinualSession:
    """Fetch or seed the session for this hand.

    A session is reused while the board it was seeded on remains a prefix of the
    live board, so turning the river card does NOT reseed from the blueprint —
    that continuity is the whole point.
    """
    board = tuple(card_id(card) for card in game.community)
    current = sessions.get(key)
    if (
        current is not None
        and current.controlled_player == controlled_player
        and board[: len(current.board)] == current.board
    ):
        return current

    entry_cards = _BOARD_CARDS[entry_street]
    if len(board) < entry_cards:
        raise ContinualResolveError(
            f"cannot open a street-{entry_street} session with {len(board)} board cards"
        )
    seed_board = board[:entry_cards]
    first_event = _first_event_on_street(game, entry_street)
    street_buckets = partial_board_buckets(seed_board, agent.sampler, seed=game.hand_number)
    ranges = _blueprint_ranges(agent, game, street_buckets, stop=first_event, street=entry_street)

    current = ContinualSession(
        key=key,
        board=seed_board,
        controlled_player=controlled_player,
        entry_street=entry_street,
        ranges=ranges,
        next_event=first_event,
    )
    sessions[key] = current
    if len(sessions) > 8:
        oldest = next(iter(sessions))
        if oldest != key:
            sessions.pop(oldest, None)
    return current


def _resolve_street(
    agent,
    game,
    controlled_player: int,
    stop: int,
    ranges: np.ndarray,
    iterations: int,
    deadline: float,
    street: int,
    observed_event: dict | None = None,
    sampler_cache: dict | None = None,
    blueprint_bucket_cache: dict | None = None,
):
    """Dispatch to the exact resolver for `street`."""
    if street == RIVER_STREET:
        return _resolve_at(
            agent, game, controlled_player, stop=stop, ranges=ranges,
            iterations=iterations, deadline=deadline, observed_event=observed_event,
        )
    from backend.search.exact_turn_resolve import resolve_postflop_at

    return resolve_postflop_at(
        agent, game, controlled_player, stop, ranges,
        iterations, deadline, street, observed_event,
        sampler_cache=sampler_cache,
        blueprint_bucket_cache=blueprint_bucket_cache,
    )


def resolve_decision(
    agent,
    game,
    controlled_player: int,
    *,
    key: tuple[int, int],
    sessions: dict[tuple[int, int], ContinualSession],
    iterations: int,
    budget_ms: int,
    entry_street: int = TURN_STREET,
) -> ContinualSolution:
    """Advance beliefs over unprocessed actions, then resolve the live state."""
    street = int(game.street)
    if street not in POSTFLOP_STREETS:
        raise ContinualResolveError(f"continual resolving covers postflop streets only (street {street})")
    if len(game.community) != _BOARD_CARDS[street]:
        raise ContinualResolveError(
            f"street {street} needs {_BOARD_CARDS[street]} board cards, saw {len(game.community)}"
        )

    session = open_session(agent, game, controlled_player, key, sessions, entry_street)
    if session.failed:
        raise ContinualResolveError(session.failure or "continual resolving was disabled for this hand")

    deadline = time.monotonic() + max(int(budget_ms), 1) / 1000.0
    catch_up: list[dict] = []
    continuation: tuple[object, NodeStrategy, int, dict] | None = None
    try:
        while session.next_event < len(game.public_actions):
            index = session.next_event
            event = game.public_actions[index]
            event_street = int(event.get("street", -1))
            if event_street < session.entry_street or event["action"] == "blind":
                session.next_event += 1
                continue
            actor_seat = _seat(game, int(event["player"]))

            pending = session.pending.pop(index, None)
            if pending is not None:
                # Our own action: use the likelihood from the solution that
                # actually chose it. Re-deriving it from the blueprint here is
                # exactly the falsified own-history that sank v1 search.
                if pending.actor_seat != actor_seat:
                    raise ContinualResolveError("recorded action has the wrong actor")
                likelihood = pending.probability
                session.own_updates += 1
            else:
                frontier = session.frontiers.pop(index, None)
                if (
                    frontier is not None
                    and frontier.actor_seat == actor_seat
                    and frontier.street == event_street
                ):
                    observed_action = _event_action(
                        frontier.tree, event, node=frontier.node
                    )
                    likelihood = frontier.probability[:, observed_action]
                    catch_up.append(
                        {
                            "purpose": "stored-frontier-belief-update",
                            "street": event_street,
                            "tree_nodes": int(len(frontier.tree)),
                            "frontier_node": int(frontier.node),
                            "gpu_solve_reused": True,
                        }
                    )
                else:
                    tree, strategy, diagnostics = _resolve_street(
                        agent, game, controlled_player, index, session.ranges,
                        max(MIN_RESOLVE_ITERATIONS, iterations // 2), deadline,
                        street=event_street,
                        observed_event=event,
                        sampler_cache=session.sampler_cache,
                        blueprint_bucket_cache=session.blueprint_bucket_cache,
                    )
                    observed_action = _event_action(tree, event)
                    likelihood = strategy[tree.root, :, observed_action]
                    diagnostics["purpose"] = "observed-action-belief-update"
                    diagnostics["street"] = event_street
                    catch_up.append(diagnostics)
                    session.resolves += 1

                    # If this is the last unprocessed action, the retrospective
                    # nested solve already contains our current child policy.
                    # Use it instead of discarding the solve and immediately
                    # solving the same public subtree again.
                    child = int(tree.children[tree.root][observed_action])
                    if (
                        index + 1 == len(game.public_actions)
                        and child in strategy
                        and tree.kind[child] == DECISION
                        and int(tree.actor[child]) == _seat(game, controlled_player)
                    ):
                        continuation = (tree, strategy, child, diagnostics)
                session.opponent_updates += 1

            live = _live_mask(tuple(card_id(card) for card in game.community))
            session.ranges[actor_seat] = _normalize_range(
                session.ranges[actor_seat] * likelihood, live
            )
            session.next_event += 1

        if continuation is not None:
            tree, strategy, decision_node, diagnostics = continuation
            diagnostics = {
                **diagnostics,
                "continuation_reused": True,
                "fresh_current_solve": False,
            }
        else:
            tree, strategy, diagnostics = _resolve_street(
                agent, game, controlled_player, len(game.public_actions), session.ranges,
                max(MIN_RESOLVE_ITERATIONS, iterations),
                deadline,
                street=street,
                sampler_cache=session.sampler_cache,
                blueprint_bucket_cache=session.blueprint_bucket_cache,
            )
            decision_node = int(tree.root)
            session.resolves += 1
            diagnostics["continuation_reused"] = False
            diagnostics["fresh_current_solve"] = True
        diagnostics.update(
            {
                "mode": f"continual-exact-v1-street{street}",
                "street": street,
                "entry_street": session.entry_street,
                "catch_up_resolves": catch_up,
                "belief_events_processed": int(session.next_event - _first_event_on_street(game, session.entry_street)),
                "own_policy_updates": session.own_updates,
                "opponent_policy_updates": session.opponent_updates,
                "session_resolves": session.resolves,
                "budget_ms": int(budget_ms),
            }
        )
        return ContinualSolution(
            tree=tree,
            strategy=strategy,
            session=session,
            diagnostics=diagnostics,
            node=decision_node,
        )
    except Exception as error:
        # Fail closed for the rest of the hand: resuming with a half-updated
        # belief would silently condition every later solve on a false range.
        session.failed = True
        session.failure = str(error)
        raise


def register_selected_action(
    solution: ContinualSolution,
    event_index: int,
    actor_seat: int,
    action: int,
) -> None:
    """Record the exact per-combo likelihood of the action we just chose."""
    probability = np.asarray(
        solution.strategy[solution.node, :, action], dtype=np.float64
    ).copy()
    solution.session.pending[int(event_index)] = PendingLikelihood(
        actor_seat=int(actor_seat), probability=probability, action=int(action)
    )
    child = int(solution.tree.children[solution.node][action])
    if (
        child >= 0
        and child in solution.strategy
        and solution.tree.kind[child] == DECISION
    ):
        solution.session.frontiers[int(event_index) + 1] = PendingFrontier(
            actor_seat=int(solution.tree.actor[child]),
            street=int(solution.tree.street[child]),
            tree=solution.tree,
            node=child,
            probability=np.asarray(
                solution.strategy[child], dtype=np.float64
            ).copy(),
        )
