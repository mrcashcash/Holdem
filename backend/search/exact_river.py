"""Exact-card continual resolving for live river decisions.

The blueprint remains the default policy.  When Phase 4 is enabled, this
module builds a fresh river tree at the exact public betting state, assigns
one information bucket to every private-card combination, and resolves under
a max-margin safety gadget.  The opponent opt-out values come from a
best-response evaluation of the loaded blueprint projected into that exact
tree.

Beliefs are advanced after every real river action.  Our own actions use the
probability from the solution that actually selected them.  An observed
opponent bet size is inserted directly into a retrospective root solve before
its likelihood is applied; it is never translated merely for range tracking.
Any deadline, mapping, or numerical failure disables Phase 4 for the rest of
the hand and lets the caller use the frozen blueprint.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from backend.search.gpu_subgame import partial_board_buckets
from backend.search.safe_subgame import GadgetCFR
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS, Deal, combos, score_all_combos
from backend.solver.gpu.tree import (
    ALL_IN,
    CHECK_CALL,
    DECISION,
    FOLD,
    FOLD_NODE,
    SHOWDOWN,
    STREET_END,
    BettingRootState,
    BettingTree,
    GpuActionConfig,
)
from backend.vectorized_engine import card_id

RIVER_FRACTIONS = (0.33, 0.5, 0.75, 1.0, 1.4)
RIVER_RAISE_CAP = 2
MIN_RESOLVE_ITERATIONS = 12


class RiverResolveError(RuntimeError):
    """Phase 4 could not safely produce a decision."""


class RiverResolveTimeout(RiverResolveError):
    """The configured river latency budget expired."""


class ExactRiverSampler:
    """A deterministic full-board sampler with identity river buckets."""

    def __init__(self, board: tuple[int, ...]) -> None:
        if len(board) != 5:
            raise ValueError("exact river resolving requires five board cards")
        self.board = tuple(int(card) for card in board)
        scores = score_all_combos(self.board)
        valid = scores >= 0
        buckets = np.zeros((4, NUM_COMBOS), dtype=np.int32)
        buckets[:, ~valid] = -1
        buckets[3, valid] = np.flatnonzero(valid).astype(np.int32)
        self._deal = Deal(
            board=self.board,
            buckets=buckets,
            valid=valid,
            river_scores=scores,
        )

    def bucket_counts(self) -> tuple[int, int, int, int]:
        return (1, 1, 1, NUM_COMBOS)

    def sample(self, rng: random.Random) -> Deal:
        return self._deal


@dataclass
class _Ledger:
    street: int = 0
    contributions: list[float] = field(default_factory=lambda: [0.0, 0.0])
    round_bets: list[float] = field(default_factory=lambda: [0.0, 0.0])
    acted: list[bool] = field(default_factory=lambda: [False, False])
    raises: int = 0
    last_increment: float = 0.0


@dataclass
class PendingLikelihood:
    actor_seat: int
    probability: np.ndarray
    action: int


@dataclass
class RiverBeliefSession:
    key: tuple[int, int]
    board: tuple[int, ...]
    controlled_player: int
    ranges: np.ndarray
    next_event: int
    pending: dict[int, PendingLikelihood] = field(default_factory=dict)
    failed: bool = False
    failure: str | None = None


@dataclass
class ExactRiverSolution:
    tree: BettingTree
    strategy: np.ndarray
    session: RiverBeliefSession
    diagnostics: dict


def _seat(game, engine_player: int) -> int:
    return 0 if int(engine_player) == int(game.button) else 1


def _normalize_range(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.where(valid, np.maximum(np.asarray(weights, dtype=np.float64), 0.0), 0.0)
    total = float(result.sum())
    if not np.isfinite(total) or total <= 1e-30:
        raise RiverResolveError("an exact-card belief lost all probability mass")
    return result / total


def _advance_street(ledger: _Ledger, street: int, big_blind: float) -> None:
    if street < ledger.street:
        raise RiverResolveError("public action history moved backwards")
    if street == ledger.street:
        return
    ledger.street = street
    ledger.round_bets = [0.0, 0.0]
    ledger.acted = [False, False]
    ledger.raises = 0
    ledger.last_increment = big_blind


def _ledger_before(game, stop: int) -> _Ledger:
    """Replay public chip movement through, but excluding, ``stop``."""

    ledger = _Ledger(last_increment=float(game.big_blind))
    events = game.public_actions
    for index, event in enumerate(events[:stop]):
        street = int(event.get("street", ledger.street))
        _advance_street(ledger, street, float(game.big_blind))
        player = int(event["player"])
        opponent = 1 - player
        action = str(event["action"])
        amount = float(event.get("amount", 0.0))

        if action == "blind":
            payment = max(amount - ledger.round_bets[player], 0.0)
            ledger.round_bets[player] += payment
            ledger.contributions[player] += payment
            continue
        if action == "check":
            ledger.acted[player] = True
            continue
        if action == "call":
            ledger.round_bets[player] += amount
            ledger.contributions[player] += amount
            ledger.acted[player] = True
            continue
        if action == "raise":
            old_high = max(ledger.round_bets)
            payment = max(amount - ledger.round_bets[player], 0.0)
            increment = max(amount - old_high, 0.0)
            ledger.round_bets[player] = amount
            ledger.contributions[player] += payment
            ledger.acted[player] = True
            ledger.acted[opponent] = False
            ledger.raises += 1
            if increment >= ledger.last_increment - 1e-9:
                ledger.last_increment = max(increment, float(game.big_blind))
            continue
        if action == "fold":
            ledger.acted[player] = True

    target_street = (
        int(events[stop].get("street", game.street))
        if stop < len(events)
        else int(game.street)
    )
    _advance_street(ledger, target_street, float(game.big_blind))
    return ledger


def _root_state(game, stop: int, expect_street: int = 3) -> BettingRootState:
    ledger = _ledger_before(game, stop)
    if ledger.street != expect_street:
        raise RiverResolveError(
            f"exact resolver root is on street {ledger.street}, expected {expect_street}"
        )
    if stop < len(game.public_actions):
        to_act_engine = int(game.public_actions[stop]["player"])
    elif game.current_player is not None:
        to_act_engine = int(game.current_player)
    else:
        raise RiverResolveError("cannot resolve a completed hand")

    starting = [
        float(game.stacks[player]) + float(game.contributions[player])
        for player in (0, 1)
    ]
    remaining = [
        max(starting[player] - ledger.contributions[player], 0.0)
        for player in (0, 1)
    ]
    scale = max(float(game.big_blind), 1.0)

    def by_seat(values):
        result = [None, None]
        for engine_player in (0, 1):
            result[_seat(game, engine_player)] = values[engine_player]
        return tuple(result)

    return BettingRootState(
        street=expect_street,
        to_act=_seat(game, to_act_engine),
        committed=by_seat([value / scale for value in ledger.contributions]),
        street_commit=by_seat([value / scale for value in ledger.round_bets]),
        stacks=by_seat([value / scale for value in remaining]),
        acted=by_seat(ledger.acted),
        raises=int(ledger.raises),
        last_increment=max(float(ledger.last_increment) / scale, 1e-6),
    )


_ORPHAN_REASON = "orphaned by a detached ancestor"


def _event_fraction(event: dict) -> float | None:
    if event.get("action") != "raise" or int(event.get("action_index", -1)) == 3:
        return None
    pot = float(event.get("pot_before", 0.0))
    to_call = float(event.get("to_call_before", 0.0))
    current_bet = float(event.get("current_bet_before", 0.0))
    raise_by = max(float(event.get("amount", 0.0)) - current_bet, 0.0)
    denominator = max(pot + to_call, 1.0)
    fraction = raise_by / denominator
    return fraction if fraction > 0 else None


def _config(
    observed_event: dict | None,
    stack_bb: float,
    base_fractions: tuple[float, ...] = RIVER_FRACTIONS,
    raise_cap: int = RIVER_RAISE_CAP,
) -> GpuActionConfig:
    """Action config for an exact resolver tree.

    An observed off-tree size is inserted into the menu rather than translated
    away (nested subgame solving), so the opponent's real action is priced
    exactly. Turn resolving passes a coarser base menu because the turn tree
    multiplies by the river subtree — see tools/exact_turn_probe.py for the
    measured cost of each extra size.
    """
    fractions = set(base_fractions)
    if observed_event is not None:
        observed = _event_fraction(observed_event)
        if observed is not None:
            fractions.add(round(float(observed), 8))
    return GpuActionConfig(
        preflop_fractions=(1.0,),
        postflop_fractions=tuple(sorted(fractions)),
        max_raises_per_street=raise_cap,
        stack_bb=max(float(stack_bb), 1.0),
    )


def _blueprint_node(agent, game, stop: int) -> int | None:
    node = int(agent.tree.root)
    rng = random.Random(game.hand_number * 8191 + stop)
    try:
        for event in game.public_actions[:stop]:
            if event["action"] == "blind":
                continue
            while agent.tree.kind[node] == STREET_END:
                node = int(agent.tree.children[node][0])
            if agent.tree.kind[node] != DECISION:
                return None
            action = agent._translate_event(node, game, event, rng)
            child = int(agent.tree.children[node][action])
            if child < 0:
                return None
            node = child
        while agent.tree.kind[node] == STREET_END:
            node = int(agent.tree.children[node][0])
        return node if agent.tree.kind[node] == DECISION else None
    except Exception:
        return None


def _blueprint_ranges(
    agent,
    game,
    street_buckets: np.ndarray,
    stop: int,
    street: int = 3,
) -> np.ndarray:
    """Blueprint posterior per combo for both seats after the history up to ``stop``.

    Used only to SEED a continual-resolving session at its entry street. Once
    seeded, ranges advance from the policies actually played (see
    `backend.search.continual`), never by re-deriving from the blueprint — that
    re-derivation is the self-range inconsistency that made v1 search regress.
    """
    valid = street_buckets[street] >= 0
    result = np.zeros((2, NUM_COMBOS), dtype=np.float64)
    for target_seat in (0, 1):
        weights = valid.astype(np.float64)
        node = int(agent.tree.root)
        rng = random.Random(game.hand_number * 131 + target_seat)
        for event in game.public_actions[:stop]:
            if event["action"] == "blind":
                continue
            while agent.tree.kind[node] == STREET_END:
                node = int(agent.tree.children[node][0])
            if agent.tree.kind[node] != DECISION:
                raise RiverResolveError("blueprint history ended before the river")
            action = agent._translate_event(node, game, event, rng)
            child = int(agent.tree.children[node][action])
            if child < 0:
                raise RiverResolveError("blueprint history did not map to a legal child")
            if int(agent.tree.actor[node]) == target_seat:
                street = int(agent.tree.street[node])
                bucket_row = street_buckets[street]
                usable = bucket_row >= 0
                probabilities = agent.strategy[
                    node,
                    np.clip(bucket_row, 0, None),
                    action,
                ]
                weights[usable] *= probabilities[usable]
                weights[~usable] = 0.0
            node = child
        result[target_seat] = _normalize_range(weights, valid)
    return result


def _map_action(
    source_tree: BettingTree,
    source_node: int,
    source_action: int,
    target_tree: BettingTree,
    target_node: int,
) -> int | None:
    """Translate a tree action into another tree's menu.

    The fallback precedence deliberately mirrors
    ``GpuBlueprintAgent._translate_event`` so Phase 4 and normal serving agree on
    what an unavailable action becomes. They previously disagreed in three ways,
    which is one half of the projection-failure defect: check/call fell back to
    fold instead of the smallest legal raise, an unavailable raise always became
    an all-in (serving requires the size to exceed 1.5 pot first), and an all-in
    was remapped onto the largest sized raise.
    """
    legal = target_tree.legal[target_node]
    if source_action in (FOLD, CHECK_CALL, ALL_IN) and legal[source_action]:
        return int(source_action)
    if source_action == FOLD:
        # Serving: a fold that is not legal means checking is free.
        return CHECK_CALL if legal[CHECK_CALL] else None
    if source_action == CHECK_CALL:
        # Serving maps an unavailable check/call (a no_limp tree has no
        # open-limp branch) to the SMALLEST legal raise before anything else.
        sized = np.flatnonzero(legal[3:])
        if sized.size:
            fractions = target_tree.config.fractions(int(target_tree.street[target_node]))
            return int(min((int(a) + 3 for a in sized), key=lambda action: fractions[action - 3]))
        return ALL_IN if legal[ALL_IN] else (FOLD if legal[FOLD] else None)
    if source_action == ALL_IN:
        return CHECK_CALL if legal[CHECK_CALL] else (FOLD if legal[FOLD] else None)

    source_fraction = source_tree.config.fractions(int(source_tree.street[source_node]))[
        source_action - 3
    ]
    sized = [
        action
        for action in range(3, target_tree.config.num_actions)
        if legal[action]
    ]
    if sized:
        return min(
            sized,
            key=lambda action: abs(
                target_tree.config.fractions(int(target_tree.street[target_node]))[
                    action - 3
                ]
                - source_fraction
            ),
        )
    # Serving only promotes an untranslatable raise to an all-in when the size
    # was genuinely large; otherwise it check/calls.
    if legal[ALL_IN] and source_fraction > 1.5:
        return ALL_IN
    return CHECK_CALL if legal[CHECK_CALL] else (FOLD if legal[FOLD] else None)


def _node_matched_pot(tree: BettingTree, node: int, depth: int = 0) -> float:
    """Matched pot in bb at a decision node.

    The tree fills ``matched_pot`` only on SHOWDOWN/HORIZON nodes, and stores a
    fold's amount in ``fold_loser_committed`` instead, so reading ``matched_pot``
    at a decision node yields a misleading 0.0. Both stored amounts are one
    player's committed total (the winner's gain), so the pot is twice that.

    Folding adds no chips, so the FOLD child's ``fold_loser_committed`` is the
    acting player's committed total, which is exactly ``min(committed)`` at the
    decision — whether or not the actor faces a bet. When folding is illegal the
    commitments are already level, so a SHOWDOWN reached by checking carries the
    same matched level.
    """
    fold_child = int(tree.children[node][FOLD])
    if fold_child >= 0 and tree.kind[fold_child] == FOLD_NODE:
        return round(2.0 * float(tree.fold_loser_committed[fold_child]), 3)
    call_child = int(tree.children[node][CHECK_CALL])
    if call_child >= 0:
        if tree.kind[call_child] in (SHOWDOWN, FOLD_NODE):
            return round(2.0 * float(tree.matched_pot[call_child]), 3)
        if tree.kind[call_child] == DECISION and depth < 4:
            # Fold is illegal, so this is a free check that leaves the matched
            # level untouched; the answer is the same one node down. The chain
            # is at most check->check->terminal, and depth bounds it anyway.
            return _node_matched_pot(tree, call_child, depth + 1)
    return 0.0


def _safe_default_policy(tree: BettingTree, node: int) -> np.ndarray:
    """The serving agent's off-tree default as a distribution over actions.

    Mirrors ``GpuBlueprintAgent._safe_default``: take a free check or a call,
    else move all-in, else fold. Used as the projected baseline wherever the
    coarse blueprint has no usable counterpart for an exact node, so a local
    divergence costs a locally-approximate opt-out price instead of the whole
    resolve.
    """
    legal = np.asarray(tree.legal[node], dtype=bool)
    policy = np.zeros(tree.config.num_actions, dtype=np.float32)
    if legal[CHECK_CALL]:
        policy[CHECK_CALL] = 1.0
    elif legal[ALL_IN]:
        policy[ALL_IN] = 1.0
    elif legal[FOLD]:
        policy[FOLD] = 1.0
    elif legal.any():  # pragma: no cover - a decision node always has one of the above
        policy[legal] = 1.0 / float(legal.sum())
    return policy


def _project_blueprint(
    agent,
    exact_tree: BettingTree,
    blueprint_root: int,
    blueprint_bucket_row: np.ndarray,
    valid: np.ndarray,
    expected_street: int = 3,
) -> tuple[torch.Tensor, dict]:
    """Project the loaded river blueprint onto an exact-card resolver tree.

    The exact tree uses real stacks, a richer size menu and a live mid-street
    root, so it legitimately contains decisions where the coarse blueprint tree
    has already terminated, has the other player acting, or has no matching
    child. That is not an error: in shallow-stack, all-in, raise-cap and
    off-tree branches the two topologies simply diverge.

    Previously any such divergence aborted the whole resolve, which is what
    produced every one of the 25 fallbacks (1.32%) in the 3,000-pair Phase 4
    confirmation — all reported as "blueprint projection reached an
    incompatible public state". The projection is only a *baseline* used to
    price the safe gadget's opponent opt-out, so a locally-approximate baseline
    in a rare subtree is enormously better than losing exact-card resolving for
    the entire hand. Divergent nodes are therefore **detached**: they and their
    descendants take the serving agent's safe-default policy, the mismatch is
    recorded, and the resolve proceeds.
    """

    actions = exact_tree.config.num_actions
    projected = np.zeros((len(exact_tree), NUM_COMBOS, actions), dtype=np.float32)
    blueprint_nodes: dict[int, int] = {int(exact_tree.root): int(blueprint_root)}
    detach_reasons: dict[str, int] = {}
    detach_samples: list[dict] = []
    decision_nodes = 0
    detached_nodes = 0
    detached_roots = 0

    def detach(node: int, reason: str, blueprint_node: int | None) -> None:
        """Give `node` the safe default and record why it left the blueprint."""
        nonlocal detached_nodes, detached_roots
        detached_nodes += 1
        # "orphaned" means an ancestor already detached; anything else is a
        # genuine topology divergence at this node.
        root_cause = reason != _ORPHAN_REASON
        if root_cause:
            detached_roots += 1
        detach_reasons[reason] = detach_reasons.get(reason, 0) + 1
        policy = _safe_default_policy(exact_tree, node)
        projected[node, valid] = policy
        projected[node, ~valid] = 0.0
        if root_cause and len(detach_samples) < 5:
            pot_bb = _node_matched_pot(exact_tree, node)
            sample = {
                "reason": reason,
                "exact_node": int(node),
                "exact_actor": int(exact_tree.actor[node]),
                "exact_street": int(exact_tree.street[node]),
                "exact_pot_bb": pot_bb,
                "exact_spr": (
                    round(max(float(exact_tree.config.stack_bb) - pot_bb / 2.0, 0.0) / pot_bb, 3)
                    if pot_bb
                    else None
                ),
                "exact_legal": [int(flag) for flag in exact_tree.legal[node]],
                "stack_bb": float(exact_tree.config.stack_bb),
            }
            if blueprint_node is not None:
                sample.update(
                    {
                        "blueprint_node": int(blueprint_node),
                        "blueprint_kind": int(agent.tree.kind[blueprint_node]),
                        "blueprint_actor": int(agent.tree.actor[blueprint_node]),
                        "blueprint_street": int(agent.tree.street[blueprint_node]),
                        "blueprint_pot_bb": round(float(agent.tree.matched_pot[blueprint_node]), 3),
                    }
                )
            detach_samples.append(sample)

    for node in range(len(exact_tree)):
        if exact_tree.kind[node] != DECISION:
            continue
        decision_nodes += 1
        blueprint_node = blueprint_nodes.get(node)
        if blueprint_node is None:
            detach(node, _ORPHAN_REASON, None)
            continue
        while agent.tree.kind[blueprint_node] == STREET_END:
            blueprint_node = int(agent.tree.children[blueprint_node][0])
        if agent.tree.kind[blueprint_node] != DECISION:
            detach(node, "blueprint terminated while the exact tree still acts", blueprint_node)
            continue
        if int(agent.tree.actor[blueprint_node]) != int(exact_tree.actor[node]):
            detach(node, "blueprint actor differs from the exact actor", blueprint_node)
            continue
        if int(agent.tree.street[blueprint_node]) != expected_street:
            detach(
                node,
                f"blueprint node is on street {int(agent.tree.street[blueprint_node])}, expected {expected_street}",
                blueprint_node,
            )
            continue

        bucket_ids = np.clip(blueprint_bucket_row, 0, None)
        blueprint_probabilities = np.asarray(
            agent.strategy[blueprint_node, bucket_ids],
            dtype=np.float32,
        )
        for blueprint_action in range(agent.tree.config.num_actions):
            if not agent.tree.legal[blueprint_node][blueprint_action]:
                continue
            exact_action = _map_action(
                agent.tree,
                blueprint_node,
                blueprint_action,
                exact_tree,
                node,
            )
            if exact_action is not None:
                projected[node, :, exact_action] += blueprint_probabilities[
                    :, blueprint_action
                ]

        legal = exact_tree.legal[node]
        projected[node, ~valid] = 0.0
        totals = projected[node].sum(axis=1)
        missing = valid & (totals <= 1e-30)
        if np.any(missing):
            projected[node, missing] = legal.astype(np.float32) / max(int(legal.sum()), 1)
            totals = projected[node].sum(axis=1)
        projected[node, valid] /= totals[valid, None]

        for exact_action in range(actions):
            child = int(exact_tree.children[node][exact_action])
            if child < 0 or exact_tree.kind[child] != DECISION:
                continue
            blueprint_action = _map_action(
                exact_tree,
                node,
                exact_action,
                agent.tree,
                blueprint_node,
            )
            # An unmappable action or an absent blueprint child leaves the child
            # orphaned; it takes the safe default when its turn comes rather
            # than aborting the resolve.
            if blueprint_action is None:
                continue
            blueprint_child = int(agent.tree.children[blueprint_node][blueprint_action])
            if blueprint_child < 0:
                continue
            blueprint_nodes[child] = blueprint_child

    diagnostics = {
        "projection_decision_nodes": decision_nodes,
        "projection_detached_nodes": detached_nodes,
        "projection_detached_roots": detached_roots,
        "projection_detached_fraction": round(detached_nodes / max(decision_nodes, 1), 6),
        "projection_detach_reasons": detach_reasons,
        "projection_detach_samples": detach_samples,
    }
    return torch.as_tensor(projected, dtype=torch.float32), diagnostics


def _check_deadline(deadline: float, device: torch.device, synchronize: bool) -> None:
    if synchronize and device.type == "cuda":
        torch.cuda.synchronize(device)
    if time.monotonic() >= deadline:
        raise RiverResolveTimeout("exact river resolver exceeded its latency budget")


def _blueprint_alt_values(
    solver: VectorCFR,
    baseline: torch.Tensor,
    controlled_seat: int,
    iterations: int,
    deadline: float,
    resample: bool = False,
) -> torch.Tensor:
    """Opt-out prices: opponent CFVs when best-responding to the frozen baseline.

    ``resample`` must be True whenever the sampler has more than one possible
    deal. A river sampler has exactly one, so holding it fixed is exact; a TURN
    sampler has 48 river runouts, and reusing a single one would price the
    opt-out as if the river card were already known.
    """
    opponent = 1 - controlled_seat

    held = solver.sampler.sample(solver.rng)
    for iteration in range(max(MIN_RESOLVE_ITERATIONS, iterations)):
        deal = solver.sampler.sample(solver.rng) if resample else held
        solver._iterate(
            deal,
            traverser=opponent,
            frozen_average=baseline,
            frozen_player=controlled_seat,
        )
        if iteration % 4 == 3:
            _check_deadline(deadline, solver.device, synchronize=True)
    # A final pass whose root values are the ones returned. With resampling the
    # single-deal estimate would be noisy, so average the last few runouts.
    if resample:
        total = None
        passes = 8
        for _ in range(passes):
            solver._iterate(
                solver.sampler.sample(solver.rng),
                traverser=opponent,
                frozen_average=baseline,
                frozen_player=controlled_seat,
            )
            values = solver._last_root_values.view(-1, NUM_COMBOS).mean(dim=0)
            total = values.clone() if total is None else total + values
        _check_deadline(deadline, solver.device, synchronize=True)
        return (total / float(passes)).detach().clone()
    solver._iterate(
        held,
        traverser=opponent,
        frozen_average=baseline,
        frozen_player=controlled_seat,
    )
    _check_deadline(deadline, solver.device, synchronize=True)
    return solver._last_root_values.view(-1, NUM_COMBOS).mean(dim=0).detach().clone()


def _run_gadget(
    gadget: GadgetCFR,
    iterations: int,
    deadline: float,
    resample: bool = False,
) -> int:
    """Drive the safe gadget. ``resample`` draws a fresh chance outcome per
    iteration, which is required for any sampler with more than one deal."""
    solver = gadget.solver
    if solver.device.type == "cuda":
        runner = _GadgetGraphRunner(gadget, resample=resample)
        _check_deadline(deadline, solver.device, synchronize=True)
        return runner.run(iterations, deadline)

    completed = 0
    for iteration in range(max(MIN_RESOLVE_ITERATIONS, iterations)):
        solver.iteration += 1
        deal = solver.sampler.sample(solver.rng)
        enter = gadget.enter_probability()
        reach = gadget.base.clone()
        reach[gadget.constrained] *= enter
        solver.root_reach = reach
        for traverser in (0, 1):
            solver._iterate(deal, traverser=traverser)
            if traverser == gadget.constrained:
                root = solver._last_root_values.view(-1, NUM_COMBOS).mean(dim=0)
                node_value = enter * root + (1.0 - enter) * gadget.alt
                gadget.gadget_regrets[:, 0] += root - node_value
                gadget.gadget_regrets[:, 1] += gadget.alt - node_value
        solver._discount()
        completed += 1
        if iteration % 4 == 3:
            _check_deadline(deadline, solver.device, synchronize=True)
    _check_deadline(deadline, solver.device, synchronize=True)
    return completed


class _GadgetGraphRunner:
    """Capture one complete safe-gadget iteration as a CUDA graph.

    Exact river trees are small enough to be dominated by Windows kernel
    launch overhead. The board, tree, ranges, and opt-out values are fixed for
    one resolve, while enter probabilities and regrets live in persistent GPU
    tensors. Capturing the entire gadget update preserves the math and turns
    thousands of eager launches per iteration into one replay.
    """

    def __init__(self, gadget: GadgetCFR, resample: bool = False) -> None:
        solver = gadget.solver
        self.gadget = gadget
        self.solver = solver
        self.resample = resample
        device = solver.device
        deal = solver.sampler.sample(solver.rng)

        self.valid = torch.as_tensor(deal.valid, dtype=torch.bool, device=device)
        self.buckets = torch.as_tensor(
            deal.buckets,
            dtype=torch.long,
            device=device,
        ).clamp_min(0)
        self.scores = torch.as_tensor(
            deal.river_scores,
            dtype=torch.long,
            device=device,
        ).view(1, NUM_COMBOS)
        self.positive_factor = torch.ones((), dtype=torch.float32, device=device)
        self.negative_factor = torch.ones((), dtype=torch.float32, device=device)
        self.strategy_factor = torch.ones((), dtype=torch.float32, device=device)
        self.reach = gadget.base.clone()
        solver.root_reach = self.reach

        def buffered_inputs(_deals):
            return self.valid, self.buckets, self.scores

        solver._graph_inputs = buffered_inputs
        self.sentinel = _ExactGraphDeal()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._body()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

        solver.regrets.zero_()
        solver.strategy_sums.zero_()
        gadget.gadget_regrets.zero_()
        solver.iteration = 0
        torch.cuda.synchronize(device)

    def _body(self) -> None:
        solver = self.solver
        gadget = self.gadget
        enter = gadget.enter_probability()
        self.reach.copy_(gadget.base)
        self.reach[gadget.constrained].mul_(enter)
        for traverser in (0, 1):
            solver._iterate(self.sentinel, traverser=traverser)
            if traverser == gadget.constrained:
                root = solver._last_root_values.view(-1, NUM_COMBOS).mean(dim=0)
                node_value = enter * root + (1.0 - enter) * gadget.alt
                gadget.gadget_regrets[:, 0].add_(root - node_value)
                gadget.gadget_regrets[:, 1].add_(gadget.alt - node_value)
        solver.regrets.mul_(
            torch.where(
                solver.regrets > 0,
                self.positive_factor,
                self.negative_factor,
            )
        )
        solver.strategy_sums.mul_(self.strategy_factor)

    def _fill(self, deal) -> None:
        """Copy a fresh deal into the buffers the captured graph reads.

        The graph holds pointers to these tensors, so new chance outcomes must be
        copied IN rather than rebound. Without this a turn solve would replay one
        frozen river card every iteration — i.e. play as if the river were known.
        """
        self.valid.copy_(torch.as_tensor(deal.valid, dtype=torch.bool))
        self.buckets.copy_(
            torch.as_tensor(deal.buckets, dtype=torch.long).clamp_min(0)
        )
        self.scores.copy_(
            torch.as_tensor(deal.river_scores, dtype=torch.long).view(1, NUM_COMBOS)
        )

    def run(self, iterations: int, deadline: float) -> int:
        solver = self.solver
        completed = 0
        for iteration in range(max(MIN_RESOLVE_ITERATIONS, iterations)):
            solver.iteration += 1
            if self.resample:
                self._fill(solver.sampler.sample(solver.rng))
            t = float(solver.iteration)
            self.positive_factor.fill_(
                t**solver.discount_alpha / (t**solver.discount_alpha + 1.0)
            )
            self.negative_factor.fill_(
                t**solver.discount_beta / (t**solver.discount_beta + 1.0)
            )
            self.strategy_factor.fill_(
                (t / (t + 1.0)) ** solver.discount_gamma
                if solver.iteration > solver.averaging_delay
                else 0.0
            )
            self.graph.replay()
            completed += 1
            if iteration % 16 == 15:
                _check_deadline(deadline, solver.device, synchronize=True)
        _check_deadline(deadline, solver.device, synchronize=True)
        return completed


class _ExactGraphDeal:
    __slots__ = ("batch",)

    def __init__(self) -> None:
        self.batch = 1


def _resolve_at(
    agent,
    game,
    controlled_player: int,
    stop: int,
    ranges: np.ndarray,
    iterations: int,
    deadline: float,
    observed_event: dict | None = None,
) -> tuple[BettingTree, np.ndarray, dict]:
    started = time.monotonic()
    root_state = _root_state(game, stop)
    stack_bb = max(
        root_state.committed[seat] + root_state.stacks[seat]
        for seat in (0, 1)
    )
    tree = BettingTree(
        _config(observed_event, stack_bb),
        root_state=root_state,
    )
    board = tuple(card_id(card) for card in game.community)
    sampler = ExactRiverSampler(board)
    blueprint_root = _blueprint_node(agent, game, stop)
    if blueprint_root is None:
        raise RiverResolveError("the loaded blueprint has no matching river root")

    street_buckets = partial_board_buckets(
        board,
        agent.sampler,
        seed=game.hand_number * 17 + stop,
    )
    valid = sampler._deal.valid
    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = VectorCFR(
        tree,
        sampler,
        device=device,
        seed=game.hand_number * 1009 + stop,
        averaging_delay=max(2, iterations // 6),
    )
    try:
        _check_deadline(deadline, solver.device, synchronize=False)
        solver.root_reach = torch.as_tensor(
            ranges,
            dtype=torch.float32,
            device=solver.device,
        )
        baseline_cpu, projection_diagnostics = _project_blueprint(
            agent,
            tree,
            blueprint_root,
            street_buckets[3],
            valid,
        )
        baseline = baseline_cpu.to(solver.device)
        _check_deadline(deadline, solver.device, synchronize=True)
        controlled_seat = _seat(game, controlled_player)
        alt = _blueprint_alt_values(
            solver,
            baseline,
            controlled_seat,
            iterations=max(MIN_RESOLVE_ITERATIONS, iterations // 3),
            deadline=deadline,
        )

        solver.regrets.zero_()
        solver.strategy_sums.zero_()
        solver.iteration = 0
        solver.root_reach = torch.as_tensor(
            ranges,
            dtype=torch.float32,
            device=solver.device,
        )
        gadget = GadgetCFR(
            solver,
            constrained=1 - controlled_seat,
            base_ranges=ranges,
            alt=alt,
        )
        completed = _run_gadget(gadget, iterations=iterations, deadline=deadline)
        strategy = solver.average_strategy_tables().astype(np.float64)
        diagnostics = {
            "mode": "exact-card-safe-river-v1",
            "tree_nodes": int(len(tree)),
            "exact_private_combos": NUM_COMBOS,
            "iterations": int(completed),
            "blueprint_alt_source": "projected-blueprint-best-response",
            "observed_size_inserted": _event_fraction(observed_event)
            if observed_event is not None
            else None,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            **projection_diagnostics,
        }
        return tree, strategy, diagnostics
    finally:
        del solver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _event_action(tree: BettingTree, event: dict) -> int:
    legal = tree.legal[tree.root]
    action = str(event["action"])
    if action == "fold":
        return FOLD
    if action in ("check", "call"):
        return CHECK_CALL
    if action != "raise":
        raise RiverResolveError(f"unsupported river action: {action}")
    if int(event.get("action_index", -1)) == 3:
        return ALL_IN
    observed = _event_fraction(event)
    if observed is None:
        raise RiverResolveError("observed river raise has no usable size")
    sized = [
        candidate
        for candidate in range(3, tree.config.num_actions)
        if legal[candidate]
    ]
    if not sized:
        return ALL_IN
    return min(
        sized,
        key=lambda candidate: abs(
            tree.config.fractions(3)[candidate - 3] - observed
        ),
    )


def _first_river_event(game) -> int:
    for index, event in enumerate(game.public_actions):
        if int(event.get("street", -1)) == 3:
            return index
    return len(game.public_actions)


def _session(
    agent,
    game,
    controlled_player: int,
    key: tuple[int, int],
    sessions: dict[tuple[int, int], RiverBeliefSession],
) -> RiverBeliefSession:
    board = tuple(card_id(card) for card in game.community)
    current = sessions.get(key)
    if (
        current is not None
        and current.board == board
        and current.controlled_player == controlled_player
    ):
        return current

    first_river = _first_river_event(game)
    street_buckets = partial_board_buckets(
        board,
        agent.sampler,
        seed=game.hand_number,
    )
    ranges = _blueprint_ranges(
        agent,
        game,
        street_buckets,
        stop=first_river,
    )
    current = RiverBeliefSession(
        key=key,
        board=board,
        controlled_player=controlled_player,
        ranges=ranges,
        next_event=first_river,
    )
    sessions[key] = current
    if len(sessions) > 8:
        oldest = next(iter(sessions))
        if oldest != key:
            sessions.pop(oldest, None)
    return current


def solve_exact_river(
    agent,
    game,
    controlled_player: int,
    *,
    key: tuple[int, int],
    sessions: dict[tuple[int, int], RiverBeliefSession],
    iterations: int,
    budget_ms: int,
) -> ExactRiverSolution:
    """Advance exact beliefs and freshly resolve the current river state."""

    if game.street != 3 or len(game.community) != 5:
        raise RiverResolveError("Phase 4 is river-only")
    session = _session(agent, game, controlled_player, key, sessions)
    if session.failed:
        raise RiverResolveError(session.failure or "Phase 4 was disabled for this hand")

    deadline = time.monotonic() + max(int(budget_ms), 1) / 1000.0
    historical_resolves: list[dict] = []
    try:
        while session.next_event < len(game.public_actions):
            index = session.next_event
            event = game.public_actions[index]
            if int(event.get("street", -1)) != 3:
                session.next_event += 1
                continue
            actor_seat = _seat(game, int(event["player"]))
            pending = session.pending.pop(index, None)
            if pending is not None:
                if pending.actor_seat != actor_seat:
                    raise RiverResolveError("recorded Phase 4 action has the wrong actor")
                likelihood = pending.probability
            else:
                tree, strategy, diagnostics = _resolve_at(
                    agent,
                    game,
                    controlled_player,
                    stop=index,
                    ranges=session.ranges,
                    iterations=max(MIN_RESOLVE_ITERATIONS, iterations // 2),
                    deadline=deadline,
                    observed_event=event,
                )
                observed_action = _event_action(tree, event)
                likelihood = strategy[tree.root, :, observed_action]
                diagnostics["purpose"] = "observed-action-belief-update"
                historical_resolves.append(diagnostics)
            valid = np.ones(NUM_COMBOS, dtype=bool)
            holdings = combos()
            for card in session.board:
                valid &= (holdings[:, 0] != card) & (holdings[:, 1] != card)
            session.ranges[actor_seat] = _normalize_range(
                session.ranges[actor_seat] * likelihood,
                valid,
            )
            session.next_event += 1

        tree, strategy, diagnostics = _resolve_at(
            agent,
            game,
            controlled_player,
            stop=len(game.public_actions),
            ranges=session.ranges,
            iterations=max(MIN_RESOLVE_ITERATIONS, iterations),
            deadline=deadline,
        )
        diagnostics["history_resolves"] = historical_resolves
        diagnostics["belief_events_processed"] = int(
            session.next_event - _first_river_event(game)
        )
        diagnostics["budget_ms"] = int(budget_ms)
        return ExactRiverSolution(
            tree=tree,
            strategy=strategy,
            session=session,
            diagnostics=diagnostics,
        )
    except Exception as error:
        session.failed = True
        session.failure = str(error)
        raise


def register_selected_action(
    solution: ExactRiverSolution,
    event_index: int,
    actor_seat: int,
    action: int,
) -> None:
    """Remember the exact policy likelihood used by the action we just chose."""

    probability = np.asarray(
        solution.strategy[solution.tree.root, :, action],
        dtype=np.float64,
    ).copy()
    solution.session.pending[int(event_index)] = PendingLikelihood(
        actor_seat=int(actor_seat),
        probability=probability,
        action=int(action),
    )
