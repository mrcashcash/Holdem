"""Exact-card turn re-solve: safe gadget over a turn+river tree (P1.3).

The turn analogue of `exact_river._resolve_at`. Same structure and the same
trusted pieces — the blueprint is projected onto the exact tree to price the
opponent's opt-out, then a max-margin gadget CFR runs on exact per-combo ranges —
with three turn-specific differences:

* `ExactTurnSampler` supplies identity buckets on turn AND river, and enumerates
  all 48 river runouts once (so there is no per-iteration bucketing cost);
* the tree roots at street 2 and carries a coarser base menu, because a turn tree
  multiplies by its river subtree (measured cost per extra size:
  `tools/exact_turn_probe.py`);
* the blueprint projection expects street-2 nodes.

The solve is CUDA-graph captured. That is not an optimization here: eager mode
manages 9-18 iterations inside a 2 s budget, capture manages 116-162, and these
trees are launch-bound rather than compute-bound (2.6-12.5x measured).
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import torch

from backend.search.exact_river import (
    MIN_RESOLVE_ITERATIONS,
    NodeStrategy,
    _root_and_frontier_strategy,
    RiverResolveError,
    RunoutBlueprintPolicy,
    _blueprint_alt_values,
    _check_deadline,
    _config,
    _event_fraction,
    _root_state,
    _run_gadget,
    _seat,
)
from backend.search.exact_flop import (
    FLOP_FRACTIONS,
    FLOP_RAISE_CAP,
    ExactFlopSampler,
    flop_fraction_tiers,
    flop_fraction_tiers_for_spr,
)
from backend.search.exact_turn import TURN_FRACTIONS, TURN_RAISE_CAP, ExactTurnSampler
from backend.search.resources import ResolverResourceError, decide_exact_solver
from backend.search.safe_subgame import GadgetCFR
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS
from backend.solver.gpu.tree import BettingTree
from backend.vectorized_engine import card_id

TURN_STREET = 2
FLOP_STREET = 1

#: Per-street resolver configuration: board prefix length, sampler, menu, cap.
_STREET_SETUP = {
    FLOP_STREET: (3, ExactFlopSampler, FLOP_FRACTIONS, FLOP_RAISE_CAP),
    TURN_STREET: (4, ExactTurnSampler, TURN_FRACTIONS, TURN_RAISE_CAP),
}


def _blueprint_node_at(agent, game, stop: int) -> int | None:
    from backend.search.exact_river import _blueprint_node

    return _blueprint_node(agent, game, stop)


def resolve_turn_at(agent, game, controlled_player, stop, ranges, iterations,
                    deadline, observed_event=None):
    """Backwards-compatible alias for the turn street."""
    return resolve_postflop_at(agent, game, controlled_player, stop, ranges,
                               iterations, deadline, TURN_STREET, observed_event)


def resolve_postflop_at(
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
) -> tuple[BettingTree, NodeStrategy, dict]:
    """Exact-card subgame solved to SHOWDOWN, rooted at public action ``stop``.

    Street 1 (flop) uses a richest-safe action-menu ladder. Every actual
    mid-street candidate is checked against node and VRAM limits before solver
    allocation; if all tiers fail, the caller uses the promoted blueprint.
    """
    if street not in _STREET_SETUP:
        raise RiverResolveError(f"no exact resolver for street {street}")
    board_cards, sampler_class, base_fractions, raise_cap = _STREET_SETUP[street]
    started = time.monotonic()
    root_state = _root_state(game, stop, expect_street=street)
    stack_bb = max(
        root_state.committed[seat] + root_state.stacks[seat] for seat in (0, 1)
    )
    # Use the 4-card PREFIX, not the live board. Belief catch-up replays turn
    # events while the hand may already be on the river, and a retrospective
    # turn resolve must see the board as it was then — the river card was not
    # known when that action was taken.
    full_board = tuple(card_id(card) for card in game.community)
    if len(full_board) < board_cards:
        raise RiverResolveError(
            f"street-{street} resolving needs {board_cards} board cards, saw {len(full_board)}"
        )
    board = full_board[:board_cards]
    sampler_key = (int(street), tuple(board))
    use_sampler_cache = (
        sampler_cache is not None
        and os.environ.get("HOLDEM_SESSION_RUNOUT_CACHE", "1") != "0"
    )
    sampler = (
        sampler_cache.get(sampler_key)
        if use_sampler_cache
        else None
    )
    sampler_reused = sampler is not None
    if sampler is None:
        sampler = sampler_class(board)
        if use_sampler_cache:
            sampler_cache[sampler_key] = sampler
    river_net = (
        getattr(agent, "resolver_river_net", None)
        if street == TURN_STREET
        else None
    )
    river_horizon = river_net is not None

    blueprint_root = _blueprint_node_at(agent, game, stop)
    if blueprint_root is None:
        raise RiverResolveError(f"the loaded blueprint has no matching street-{street} root")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if street == FLOP_STREET:
        current_pot = max(sum(root_state.committed), 1e-6)
        current_spr = max(root_state.stacks) / current_pot
        all_fraction_tiers = flop_fraction_tiers(base_fractions)
        fraction_tiers = flop_fraction_tiers_for_spr(
            base_fractions,
            current_spr,
        )
    else:
        current_spr = None
        all_fraction_tiers = (tuple(base_fractions),)
        fraction_tiers = (tuple(base_fractions),)
    tree = None
    resource_decision = None
    selected_fractions = None
    for fractions in fraction_tiers:
        candidate = BettingTree(
            _config(
                observed_event,
                stack_bb,
                base_fractions=fractions,
                raise_cap=raise_cap,
            ),
            root_state=root_state,
            end_street=TURN_STREET if river_horizon else None,
        )
        decision = decide_exact_solver(
            candidate,
            sampler.bucket_counts(),
            street=street,
            device=device,
            graph_capture=not river_horizon,
        )
        tree = candidate
        resource_decision = decision
        selected_fractions = tuple(fractions)
        if decision.allowed:
            break
    assert tree is not None and resource_decision is not None
    if not resource_decision.allowed:
        raise ResolverResourceError(resource_decision)
    solver_kwargs = {
        "device": device,
        "seed": game.hand_number * 1009 + stop,
        "averaging_delay": max(2, iterations // 6),
    }
    if river_horizon:
        from backend.search.depth_limited import DepthLimitedCFR
        from backend.search.river_horizon import RiverNetEvaluator

        solver = DepthLimitedCFR(
            tree,
            sampler,
            horizon_evaluator=RiverNetEvaluator(
                river_net,
                device,
                board,
                stack_bb,
            ),
            **solver_kwargs,
        )
        # The graph runner uses a lightweight static-deal sentinel. A neural
        # horizon consumes the actual sampled river card, so keep this path
        # eager unless/until the evaluator owns static graph input buffers.
        solver.disable_graph_capture = True
    else:
        solver = VectorCFR(tree, sampler, **solver_kwargs)
    baseline = None
    alt = None
    gadget = None
    diagnostics = None
    stage_started = started
    safety_timings: dict = {}
    gadget_timings: dict = {}
    try:
        _check_deadline(deadline, solver.device, synchronize=False)
        solver.root_reach = torch.as_tensor(ranges, dtype=torch.float32, device=solver.device)
        baseline = RunoutBlueprintPolicy(
            agent,
            tree,
            blueprint_root,
            solver.device,
            bucket_cache=blueprint_bucket_cache,
        )
        projection_diagnostics = baseline.projection_diagnostics
        _check_deadline(deadline, solver.device, synchronize=True)
        setup_done = time.monotonic()

        controlled_seat = _seat(game, controlled_player)
        # resample=True is mandatory here: the turn sampler has 48 river
        # runouts, and holding one fixed would price the opponent's opt-out as
        # if the river card were already known.
        alt = _blueprint_alt_values(
            solver,
            baseline,
            controlled_seat,
            iterations=max(MIN_RESOLVE_ITERATIONS, iterations // 3),
            deadline=deadline,
            resample=True,
            timings=safety_timings,
        )
        safety_done = time.monotonic()

        solver.regrets.zero_()
        solver.strategy_sums.zero_()
        solver.iteration = 0
        solver.root_reach = torch.as_tensor(ranges, dtype=torch.float32, device=solver.device)
        gadget = GadgetCFR(
            solver,
            constrained=1 - controlled_seat,
            base_ranges=ranges,
            alt=alt,
        )
        # Shared graph-accelerated driver; resampling the river each iteration.
        completed = _run_gadget(
            gadget,
            iterations,
            deadline,
            resample=True,
            timings=gadget_timings,
        )
        gadget_done = time.monotonic()
        strategy = _root_and_frontier_strategy(solver, tree)
        export_done = time.monotonic()
        runouts = (
            getattr(
                sampler,
                "runout_count",
                lambda: len(getattr(sampler, "rivers", ())),
            )()
            if not hasattr(sampler, "rivers")
            else len(sampler.rivers)
        )
        diagnostics = {
            "mode": f"exact-card-safe-street{street}-v1",
            "tree_nodes": int(len(tree)),
            "exact_private_combos": NUM_COMBOS,
            "iterations": int(completed),
            "runouts": int(runouts),
            "river_runouts": int(runouts) if street == TURN_STREET else None,
            "blueprint_alt_source": "projected-blueprint-best-response",
            "river_horizon": (
                "gated-river-cfv-net" if river_horizon else "exact-to-showdown"
            ),
            "observed_size_inserted": (
                _event_fraction(observed_event) if observed_event is not None else None
            ),
            "selected_fractions": list(selected_fractions or ()),
            "menu_tier": int(all_fraction_tiers.index(selected_fractions)),
            "root_spr": round(float(current_spr), 3)
            if current_spr is not None
            else None,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            "stage_ms": {
                "setup_projection": round((setup_done - stage_started) * 1000.0, 1),
                "safety_price": round((safety_done - setup_done) * 1000.0, 1),
                "gadget_solve": round((gadget_done - safety_done) * 1000.0, 1),
                "strategy_export": round((export_done - gadget_done) * 1000.0, 1),
                "cleanup": None,
            },
            "safety_price_execution": safety_timings,
            "gadget_execution": gadget_timings,
            "sampler_reused": bool(sampler_reused),
            "blueprint_bucket_cache_hits": int(baseline.bucket_cache_hits),
            "blueprint_bucket_cache_misses": int(baseline.bucket_cache_misses),
            "resource_admission": resource_decision.diagnostics(),
            **projection_diagnostics,
        }
        return tree, strategy, diagnostics
    finally:
        cleanup_started = time.monotonic()
        gadget = None
        alt = None
        baseline = None
        solver = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if diagnostics is not None:
            diagnostics["stage_ms"]["cleanup"] = round(
                (time.monotonic() - cleanup_started) * 1000.0,
                1,
            )
            diagnostics["elapsed_ms"] = round(
                (time.monotonic() - started) * 1000.0,
                1,
            )


