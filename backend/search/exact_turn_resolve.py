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

import time

import numpy as np
import torch

from backend.search.exact_river import (
    MIN_RESOLVE_ITERATIONS,
    RiverResolveError,
    _blueprint_alt_values,
    _check_deadline,
    _config,
    _event_fraction,
    _project_blueprint,
    _root_state,
    _run_gadget,
    _seat,
)
from backend.search.exact_flop import FLOP_FRACTIONS, FLOP_RAISE_CAP, ExactFlopSampler
from backend.search.exact_turn import TURN_FRACTIONS, TURN_RAISE_CAP, ExactTurnSampler
from backend.search.gpu_subgame import partial_board_buckets
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
) -> tuple[BettingTree, np.ndarray, dict]:
    """Exact-card subgame solved to SHOWDOWN, rooted at public action ``stop``.

    Street 1 (flop) is only affordable on shallow stacks — a 20bb flop-to-river
    tree is 5,303 nodes but a 100bb one is 132,107 (~10.5 GiB of exact-combo
    tables). `backend.search.exact_flop.exact_flop_is_affordable` is the guard;
    deep stacks use the turn entry plus the value nets instead.
    """
    if street not in _STREET_SETUP:
        raise RiverResolveError(f"no exact resolver for street {street}")
    board_cards, sampler_class, base_fractions, raise_cap = _STREET_SETUP[street]
    started = time.monotonic()
    root_state = _root_state(game, stop, expect_street=street)
    stack_bb = max(
        root_state.committed[seat] + root_state.stacks[seat] for seat in (0, 1)
    )
    tree = BettingTree(
        _config(observed_event, stack_bb, base_fractions=base_fractions, raise_cap=raise_cap),
        root_state=root_state,
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
    sampler = sampler_class(board)

    blueprint_root = _blueprint_node_at(agent, game, stop)
    if blueprint_root is None:
        raise RiverResolveError(f"the loaded blueprint has no matching street-{street} root")

    street_buckets = partial_board_buckets(board, agent.sampler, seed=game.hand_number * 17 + stop)
    # Validity is river-dependent, so use the board-only mask for projection:
    # a combo live on the turn is live in at least one runout.
    valid = street_buckets[street] >= 0

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
        solver.root_reach = torch.as_tensor(ranges, dtype=torch.float32, device=solver.device)
        baseline_cpu, projection_diagnostics = _project_blueprint(
            agent, tree, blueprint_root, street_buckets[street], valid,
            expected_street=street,
        )
        baseline = baseline_cpu.to(solver.device)
        _check_deadline(deadline, solver.device, synchronize=True)

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
        )

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
        completed = _run_gadget(gadget, iterations, deadline, resample=True)
        strategy = solver.average_strategy_tables().astype(np.float64)
        diagnostics = {
            "mode": f"exact-card-safe-street{street}-v1",
            "tree_nodes": int(len(tree)),
            "exact_private_combos": NUM_COMBOS,
            "iterations": int(completed),
            "runouts": getattr(sampler, "runout_count", lambda: len(getattr(sampler, "rivers", ())))()
            if not hasattr(sampler, "rivers") else len(sampler.rivers),
            "blueprint_alt_source": "projected-blueprint-best-response",
            "observed_size_inserted": (
                _event_fraction(observed_event) if observed_event is not None else None
            ),
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
            **projection_diagnostics,
        }
        return tree, strategy, diagnostics
    finally:
        del solver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


