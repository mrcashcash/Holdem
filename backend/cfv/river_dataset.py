"""River counterfactual-value dataset generation (P3a).

The first net in the bottom-up stack, and deliberately so:

* **Its targets are exact.** A river subgame ends at showdown, so there are no
  runouts to enumerate and the counterfactual values carry zero Monte-Carlo
  noise. CFV v0 was bounded by target noise from 4 sampled runouts
  (`docs/CFV_NET_PLAN.md`); that failure mode cannot occur here.
* **It is the cheapest solve in the stack**, so the whole pipeline gets debugged
  where iteration is fast.
* **It buys the most throughput.** Truncating a turn tree at the river with this
  net at the horizon takes it from 726 nodes to 81 — a measured 9.0x — which is
  the real fix for both serving latency and turn-net datagen cost.

Two corrections to the v0 pipeline are baked in:

1. **I/O resolution.** v0 used 169 buckets per player; DeepStack and Supremus both
   use ~1,000. On a river board only ~1,081 combos are live, so this module emits
   **exact per-combo** ranges and values — at that width clustering would be
   nearly lossless anyway, and exactness removes a whole class of doubt. (v0's
   "raw-combo I/O cannot generalize" finding was measured at 7,750 samples, where
   nothing could.)
2. **Multi-iterate emission (TurboReBeL).** One CFR solve yields one sample in
   the naive scheme. Fixing the subgame policy to the CFR average and pricing
   each intermediate belief against it yields ~T samples per solve at a few
   percent extra cost. `VectorCFR._iterate(frozen_average=..., frozen_player=None)`
   already provides exactly that primitive.

Solver and CUDA graph are cached per (pot, stack) grid cell, because capture
costs 0.4-0.7 s and would otherwise dominate.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from backend.search.exact_river import RIVER_FRACTIONS, RIVER_RAISE_CAP, ExactRiverSampler
from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import NUM_COMBOS
from backend.solver.gpu.tree import BettingRootState, BettingTree, GpuActionConfig

# Target depths are 20/50/100/200bb, so the pot grid is expressed as a FRACTION
# of the effective stack rather than in absolute bb: a 160bb pot is meaningless
# at 20bb, and an absolute grid would give the shallow depths almost no coverage.
# Quantizing still lets one captured CUDA graph serve many situations.
STACK_GRID_BB = (20.0, 50.0, 100.0, 200.0)
POT_FRACTION_GRID = (0.05, 0.10, 0.18, 0.30, 0.45, 0.65, 0.90, 1.20, 1.60)

_SOLVER_CACHE: dict[tuple, tuple] = {}


def snap(value: float, grid: tuple[float, ...]) -> float:
    return min(grid, key=lambda point: abs(point - value))


def random_board(rng: random.Random) -> tuple[int, ...]:
    return tuple(rng.sample(range(52), 5))


def random_range(rng: random.Random, valid: np.ndarray) -> np.ndarray:
    """Recursive pseudo-random range (DeepStack supplement).

    Uniform Dirichlet-ish weights produce ranges that are far too flat to look
    like anything re-solving actually meets, so mass is split recursively over
    random halves — this yields the lumpy, polarized shapes real play generates.
    """
    weights = np.zeros(NUM_COMBOS, dtype=np.float64)
    live = np.flatnonzero(valid)
    if live.size == 0:
        return weights

    def assign(indices: np.ndarray, mass: float) -> None:
        if indices.size == 0 or mass <= 0:
            return
        if indices.size == 1:
            weights[indices[0]] = mass
            return
        cut = indices.size // 2
        share = rng.random()
        # Occasionally hand a subtree nothing at all: real ranges have holes.
        if rng.random() < 0.15:
            share = 0.0 if rng.random() < 0.5 else 1.0
        assign(indices[:cut], mass * share)
        assign(indices[cut:], mass * (1.0 - share))

    shuffled = live.copy()
    rng.shuffle(shuffled)
    assign(shuffled, 1.0)
    total = weights.sum()
    return weights / total if total > 0 else weights


def grid_cells() -> list[tuple[float, float]]:
    """Every (stack, pot) cell, for cell-major generation."""
    return [
        (stack, round(fraction * stack, 4))
        for stack in STACK_GRID_BB
        for fraction in POT_FRACTION_GRID
    ]


def sample_situation(rng: random.Random, cell: tuple[float, float] | None = None) -> dict:
    """Random board and ranges. ``cell`` pins (stack, pot) so a caller can hold
    one captured graph across many situations instead of thrashing VRAM."""
    board = random_board(rng)
    sampler = ExactRiverSampler(board)
    valid = sampler._deal.valid
    if cell is not None:
        stack_bb, pot_bb = float(cell[0]), float(cell[1])
    else:
        stack_bb = rng.choice(STACK_GRID_BB)
        # pot < 2 * stack keeps at least a sliver behind, which the tree requires.
        pot_bb = round(rng.choice(POT_FRACTION_GRID) * stack_bb, 4)
    return {
        "board": board,
        "pot_bb": pot_bb,
        "stack_bb": stack_bb,
        "ranges": np.stack([random_range(rng, valid), random_range(rng, valid)]),
        "valid": valid,
    }


#: Bounded on purpose. Each cached entry pins a captured CUDA graph's memory
#: pool, and on this box only ~3 GB of the 12 GB card is actually free (the
#: desktop holds the rest). Caching all 36 grid cells filled VRAM and collapsed
#: throughput to 0.5 rows/s through allocator thrashing. Generate CELL-MAJOR
#: (see tools/generate_river_cfv.py) so one live solver serves many situations.
MAX_CACHED_SOLVERS = 2


def _solver_for(pot_bb: float, stack_bb: float, board: tuple[int, ...], device: str, iterations: int):
    """Cached solver + captured graph for one (pot, stack) cell.

    The tree depends only on the grid cell, so a single capture serves every
    board in that cell; only the sampler and root reach change per situation.
    """
    key = (pot_bb, stack_bb, device)
    cached = _SOLVER_CACHE.get(key)
    if cached is not None:
        tree, solver, runner = cached
        solver.sampler = ExactRiverSampler(board)
        return tree, solver, runner

    while len(_SOLVER_CACHE) >= MAX_CACHED_SOLVERS:
        oldest = next(iter(_SOLVER_CACHE))
        _SOLVER_CACHE.pop(oldest, None)
        if device == "cuda":
            torch.cuda.empty_cache()

    config = GpuActionConfig(
        preflop_fractions=(1.0,),
        postflop_fractions=RIVER_FRACTIONS,
        max_raises_per_street=RIVER_RAISE_CAP,
        stack_bb=stack_bb,
    )
    behind = max(stack_bb - pot_bb / 2.0, 1.0)
    root = BettingRootState(
        street=3, to_act=1, committed=(pot_bb / 2.0, pot_bb / 2.0),
        street_commit=(0.0, 0.0), stacks=(behind, behind),
        acted=(False, False), raises=0, last_increment=1.0,
    )
    tree = BettingTree(config, root_state=root)
    solver = VectorCFR(
        tree, ExactRiverSampler(board), device=device, seed=17,
        averaging_delay=max(2, iterations // 6),
    )
    solver.root_reach = torch.zeros((2, NUM_COMBOS), dtype=torch.float32, device=solver.device)
    runner = None
    if solver.device.type == "cuda":
        from backend.solver.gpu.graph import GraphRunner

        # resample is pointless on the river: the sampler has exactly one deal.
        runner = GraphRunner(solver, warmup=2)
    _SOLVER_CACHE[key] = (tree, solver, runner)
    return tree, solver, runner


def solve_situation(situation: dict, device: str = "cuda", iterations: int = 500,
                    emit_iterates: int = 0) -> list[dict]:
    """Solve one river situation; return one or more (belief, value) samples.

    With ``emit_iterates > 0`` the TurboReBeL trick applies: the solved average
    strategy is held fixed and additional intermediate beliefs are priced against
    it, so one CFR solve yields several training rows.
    """
    tree, solver, runner = _solver_for(
        situation["pot_bb"], situation["stack_bb"], situation["board"], device, iterations
    )
    ranges = np.asarray(situation["ranges"], dtype=np.float32)
    solver.root_reach.copy_(torch.as_tensor(ranges, device=solver.device))
    solver.regrets.zero_()
    solver.strategy_sums.zero_()
    solver.iteration = 0
    if runner is not None:
        runner.run(iterations, random.Random(situation["board"][0] * 131 + 7))
    else:
        solver.run(iterations)

    average = solver.average_strategy_tensor()
    deal = solver.sampler.sample(solver.rng)
    samples = [_price(solver, deal, average, ranges, situation)]
    if emit_iterates:
        samples.extend(_harvest_interior(solver, deal, average, situation, emit_iterates))
    return samples


def _harvest_interior(solver: VectorCFR, deal, average: torch.Tensor, situation: dict,
                      limit: int, min_mass: float = 1e-3) -> list[dict]:
    """Extra (belief, value) rows from INTERIOR nodes of the same solve.

    This is TurboReBeL's multi-sample-per-solve idea done correctly. An earlier
    version priced randomly *blended* ranges against the strategy solved for the
    original range; measurement showed those targets are wrong by 50% at
    blend 0.8, 107% at 0.5 and 171% at 0.2 — because the value of playing a
    mismatched strategy is not the equilibrium value of the blended range.

    Interior nodes have no such problem: the reach at a node IS the belief that
    genuinely arises there under the solved strategy, and its value under that
    same strategy is the matching CFV. Each node also sits at a different pot,
    which the net takes as an input, so the extra rows add pot coverage for free.
    """
    from backend.search.exact_river import _node_matched_pot
    from backend.solver.gpu.tree import DECISION

    tree = solver.tree
    valid_np = situation["valid"]
    valid = torch.as_tensor(valid_np, device=solver.device)

    captured = {}
    solver.capture_internals = True
    try:
        for player in (0, 1):
            solver.root_reach.copy_(
                torch.as_tensor(np.asarray(situation["ranges"], dtype=np.float32), device=solver.device)
            )
            solver._iterate(deal, traverser=player, frozen_average=average, frozen_player=None)
            captured[player] = (solver._last_reach.clone(), solver._last_values.clone())
    finally:
        solver.capture_internals = False

    reach = captured[0][0]  # reach is traverser-independent under a frozen policy
    rows: list[dict] = []
    for node in range(len(tree)):
        if len(rows) >= limit:
            break
        if tree.kind[node] != DECISION or node == tree.root:
            continue
        node_ranges = reach[:, node, :] * valid.float()
        masses = node_ranges.sum(dim=1)
        # Unreachable or near-unreachable nodes give wild normalised values.
        if float(masses.min()) < min_mass:
            continue
        pot = _node_matched_pot(tree, node)
        if pot <= 0:
            continue
        normalised = (node_ranges / masses.unsqueeze(1)).cpu().numpy()
        values = np.zeros((2, NUM_COMBOS), dtype=np.float32)
        for player in (0, 1):
            # Undo the opponent-reach weighting that _iterate's convention
            # carries, giving a unit-mass CFV in chips.
            raw = captured[player][1][node, :] / masses[1 - player].clamp_min(1e-9)
            values[player] = raw.cpu().numpy()
        values[:, ~valid_np] = 0.0
        rows.append(
            {
                "board": np.asarray(situation["board"], dtype=np.int8),
                "pot_bb": np.float32(pot),
                "stack_bb": np.float32(situation["stack_bb"]),
                "ranges": normalised.astype(np.float16),
                "values": values.astype(np.float16),
                "valid": valid_np,
            }
        )
    return rows


def _normalize(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, weights, 0.0)
    total = masked.sum()
    return masked / total if total > 0 else masked


def _price(solver: VectorCFR, deal, average: torch.Tensor, ranges: np.ndarray,
           situation: dict) -> dict:
    """Per-combo CFVs for both players under the frozen average strategy.

    Both players frozen, so this is a pure evaluation pass. On the river it is
    EXACT: the subgame ends at showdown, so there is no chance left to sample.
    """
    solver.root_reach.copy_(torch.as_tensor(ranges, dtype=torch.float32, device=solver.device))
    values = np.zeros((2, NUM_COMBOS), dtype=np.float32)
    for player in (0, 1):
        solver._iterate(deal, traverser=player, frozen_average=average, frozen_player=None)
        values[player] = solver._last_root_values.view(-1, NUM_COMBOS)[0].cpu().numpy()
    # Combos that collide with the board carry nonzero FOLD values — the fold
    # kernel weights by opponent mass without masking the hero's own combo (see
    # the note in cfr.py). Their reach is zero so play is unaffected, but stored
    # as targets they would be pure noise for the net to chase. Zero them.
    values[:, ~situation["valid"]] = 0.0
    return {
        "board": np.asarray(situation["board"], dtype=np.int8),
        "pot_bb": np.float32(situation["pot_bb"]),
        "stack_bb": np.float32(situation["stack_bb"]),
        "ranges": ranges.astype(np.float16),
        "values": values.astype(np.float16),
        "valid": situation["valid"],
    }
