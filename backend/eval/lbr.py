"""GPU-blueprint Local Best Response (LBR) exploitability probe.

This is a restricted response, not an exact best response.  It evaluates
{fold, check/call, pot raise, all-in} using the probe's exact cards and a
Bayesian range for the GPU blueprint reconstructed from its public strategy.

Quality properties:

* duplicate, seat-swapped deals remove most card/position variance;
* winnings are measured from the full starting stack (not post-blind stacks);
* opponent ranges are updated from the GPU blueprint after every public action;
* showdown equity is weighted by that range rather than a uniform opponent;
* the serving abstraction and off-tree translation path are reused directly.

CLI:
    python -m backend.eval.lbr --data-dir backend/data/gpu_blueprint \
        --pairs 500 --stack-bb 100
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import statistics
from pathlib import Path

import numpy as np

from backend.poker import HeadsUpHoldem
from backend.solver.gpu.deals import CARD_IN_COMBO, NUM_COMBOS, combos, score_all_combos
from backend.solver.gpu.tree import FOLD
from backend.vectorized_engine import card_id

NEURAL_FOLD, NEURAL_CHECK_CALL, NEURAL_RAISE, NEURAL_ALL_IN = 0, 1, 2, 3

# LBR's power comes from the breadth of sizes it may probe: a probe restricted to
# one raise size understates exploitability, because a real exploiter picks the
# size the victim's abstraction handles worst (Lisy & Bowling 2016 sweep a size
# set). Fractions are of the pot after calling, matching the serving agent's
# own sizing convention in `_raise_target_for_choice`.
PROBE_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

_COMBOS = combos()
_COMBO_INDEX = {(int(a), int(b)): index for index, (a, b) in enumerate(_COMBOS)}
_STREET_NAMES = ("preflop", "flop", "turn", "river")


def _mean_ci(samples: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(samples) if samples else 0.0
    margin = 1.96 * statistics.stdev(samples) / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    return mean, mean - margin, mean + margin


def _hero_combo(engine: HeadsUpHoldem, player: int) -> tuple[int, tuple[int, int]]:
    hole = tuple(sorted(card_id(card) for card in engine.hole_cards[player]))
    return _COMBO_INDEX[hole], hole


def _mask_range(weights: np.ndarray, hero_hole: tuple[int, int], board: tuple[int, ...]) -> np.ndarray:
    masked = np.asarray(weights, dtype=np.float64).copy()
    blocked = (CARD_IN_COMBO[hero_hole[0]] | CARD_IN_COMBO[hero_hole[1]]).copy()
    for card in board:
        blocked |= CARD_IN_COMBO[card]
    masked[blocked] = 0.0
    total = float(masked.sum())
    if total <= 1e-15:
        masked = (~blocked).astype(np.float64)
        total = float(masked.sum())
    return masked / max(total, 1e-15)


def _board_buckets(agent, engine: HeadsUpHoldem) -> np.ndarray:
    """`partial_board_buckets` memoized per (board, history length).

    One LBR decision now issues up to seven fold-response queries, each of which
    would otherwise recompute the identical bucket assignment for the identical
    board. Callers must treat the result as read-only (both consumers do).
    """
    from backend.search.gpu_subgame import partial_board_buckets

    board = tuple(card_id(card) for card in engine.community)
    seed = engine.hand_number * 1009 + len(engine.public_actions) * 17 + 11
    cache = getattr(agent, "_lbr_bucket_cache", None)
    if cache is None:
        cache = {}
        agent._lbr_bucket_cache = cache
    key = (board, seed)
    cached = cache.get(key)
    if cached is None:
        if len(cache) > 512:
            cache.clear()
        cached = partial_board_buckets(board, agent.sampler, seed=seed)
        cache[key] = cached
    return cached


def _blueprint_range(agent, engine: HeadsUpHoldem, blueprint_player: int, probe_hole: tuple[int, int]) -> np.ndarray:
    """Blueprint posterior over private combos after the observed public history."""
    from backend.search.gpu_subgame import gpu_blueprint_range

    board = tuple(card_id(card) for card in engine.community)
    buckets = _board_buckets(agent, engine)
    abstract_seat = agent._abstract_seat(engine, blueprint_player)
    weights = gpu_blueprint_range(agent, engine, abstract_seat, buckets)
    return _mask_range(weights, probe_hole, board)


class _RangeEquity:
    """Cached showdown outcomes for one probe hand over sampled runouts."""

    def __init__(self, hero_combo: int, hero_hole: tuple[int, int], seed: int, samples: int = 24) -> None:
        self.hero_combo = hero_combo
        self.hero_hole = hero_hole
        self.seed = seed
        self.samples = samples
        self._cache: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]] = {}

    def _outcomes(self, board: tuple[int, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
        cached = self._cache.get(board)
        if cached is not None:
            return cached

        used = set(board) | set(self.hero_hole)
        remaining = [card for card in range(52) if card not in used]
        fill = 5 - len(board)
        if fill == 0:
            runouts = [()]
        else:
            street_samples = self.samples if len(board) <= 3 else max(self.samples, len(remaining))
            rng_seed = self.seed * 1_000_003 + sum((index + 1) * card for index, card in enumerate(board)) + 97
            rng = random.Random(rng_seed)
            if fill == 1 and street_samples >= len(remaining):
                runouts = [(card,) for card in remaining]
            else:
                runouts = [tuple(rng.sample(remaining, fill)) for _ in range(street_samples)]

        rows: list[tuple[np.ndarray, np.ndarray]] = []
        for completion in runouts:
            scores = score_all_combos(board + completion)
            hero_score = scores[self.hero_combo]
            valid = scores >= 0
            outcome = np.zeros(NUM_COMBOS, dtype=np.float64)
            outcome[valid & (scores < hero_score)] = 1.0
            outcome[valid & (scores == hero_score)] = 0.5
            rows.append((valid, outcome))
        self._cache[board] = rows
        return rows

    def estimate(self, opponent_range: np.ndarray, board: tuple[int, ...]) -> float:
        numerator = 0.0
        denominator = 0.0
        for valid, outcome in self._outcomes(board):
            compatible = opponent_range * valid
            numerator += float(np.dot(compatible, outcome))
            denominator += float(compatible.sum())
        return numerator / denominator if denominator > 1e-15 else 0.5


def _fold_response(
    agent,
    engine: HeadsUpHoldem,
    probe_player: int,
    probe_hole: tuple[int, int],
    raise_to: int,
) -> tuple[float, np.ndarray, int]:
    """Expected blueprint fold probability and conditional continuing range."""
    blueprint_player = 1 - probe_player
    hypothetical = copy.deepcopy(engine)
    hypothetical.act(probe_player, "raise", raise_to)
    call_amount = min(hypothetical.to_call(blueprint_player), hypothetical.stacks[blueprint_player])
    prior = _blueprint_range(agent, hypothetical, blueprint_player, probe_hole)

    if hypothetical.hand_complete or hypothetical.current_player != blueprint_player:
        return 0.0, prior, call_amount
    node = agent._locate(hypothetical, blueprint_player)
    if node is None or not agent.tree.legal[node][FOLD]:
        return 0.0, prior, call_amount

    buckets = _board_buckets(agent, hypothetical)
    street = int(agent.tree.street[node])
    bucket_row = buckets[street]
    usable = bucket_row >= 0
    per_combo = np.zeros(NUM_COMBOS, dtype=np.float64)
    per_combo[usable] = agent.strategy[node, bucket_row[usable], FOLD]
    fold_probability = float(np.dot(prior, per_combo))
    continuing = prior * np.clip(1.0 - per_combo, 0.0, 1.0)
    total = float(continuing.sum())
    continuing = continuing / total if total > 1e-15 else prior
    return min(1.0, max(0.0, fold_probability)), continuing, call_amount


def _probe_action(
    agent,
    engine: HeadsUpHoldem,
    seat: int,
    equity: _RangeEquity,
    diagnostics: dict,
) -> tuple[int, int | None]:
    legal = engine.legal_actions(seat)
    hero_combo, hero_hole = _hero_combo(engine, seat)
    del hero_combo  # encoded in ``equity``
    board = tuple(card_id(card) for card in engine.community)
    opponent = 1 - seat
    opponent_range = _blueprint_range(agent, engine, opponent, hero_hole)
    win_probability = equity.estimate(opponent_range, board)
    pot = float(engine.pot)
    to_call = int(legal.get("to_call", 0))

    diagnostics["range_updates"] += 1
    diagnostics["equity_queries"] += 1
    evs: list[tuple[float, int, int | None]] = []
    evs.append((win_probability * (pot + to_call) - to_call, NEURAL_CHECK_CALL, None))
    if to_call > 0:
        evs.append((0.0, NEURAL_FOLD, None))

    if legal.get("raise"):
        minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
        player_bet = int(legal["player_bet"])
        # Probe every size in the menu, plus all-in. Clamping collapses some
        # fractions onto the same chip target (and onto all-in); dedupe so the
        # expensive fold-response query runs once per distinct amount.
        candidates: list[int] = []
        for fraction in PROBE_FRACTIONS:
            raise_by = fraction * (pot + to_call)
            target = int(round(player_bet + to_call + raise_by))
            candidates.append(min(maximum, max(minimum, target)))
        candidates.append(maximum)
        for amount in sorted(set(candidates)):
            action = NEURAL_ALL_IN if amount >= maximum else NEURAL_RAISE
            fold_probability, continuing, opponent_call = _fold_response(
                agent, engine, seat, hero_hole, amount
            )
            equity_if_called = equity.estimate(continuing, board)
            invested = float(amount - player_bet)
            showdown_pot = pot + invested + float(opponent_call)
            called_ev = equity_if_called * showdown_pot - invested
            ev = fold_probability * pot + (1.0 - fold_probability) * called_ev
            evs.append((ev, action, amount if action == NEURAL_RAISE else None))
            diagnostics["fold_queries"] += 1
            diagnostics["probe_sizes"] = max(diagnostics.get("probe_sizes", 0), len(set(candidates)))
    elif legal.get("all_in"):
        evs.append((win_probability * (pot + to_call) - min(to_call, engine.stacks[seat]), NEURAL_ALL_IN, None))

    evs.sort(key=lambda item: item[0], reverse=True)
    return evs[0][1], evs[0][2]


def _play_lbr_hand(
    agent,
    probe_seat: int,
    seed: int,
    stack_bb: float,
    diagnostics: dict,
) -> tuple[float, int]:
    engine = HeadsUpHoldem(
        initial_stack=int(round(stack_bb * 20)),
        small_blind=10,
        big_blind=20,
        rng=random.Random(seed),
    )
    before = float(engine.initial_stack)
    hero_combo, hero_hole = _hero_combo(engine, probe_seat)
    equity = _RangeEquity(hero_combo, hero_hole, seed)
    if hasattr(agent, "_rng"):
        agent._rng = random.Random(seed * 31 + 5)

    safety = 0
    while not engine.hand_complete and safety < 200:
        player = engine.current_player
        if player == probe_seat:
            action, target = _probe_action(agent, engine, player, equity, diagnostics)
            if action == NEURAL_RAISE:
                engine.act(player, "raise", target)
            elif action == NEURAL_ALL_IN:
                engine.act(player, "all_in")
            elif action == NEURAL_FOLD:
                engine.act(player, "fold")
            else:
                engine.act(player, "check" if engine.legal_actions(player).get("check") else "call")
        else:
            query = agent.strategy_for_state(engine, player)
            diagnostics["blueprint_decisions"] += 1
            diagnostics["exact_nodes"] += int(bool(query.get("exact_match")))
            diagnostics["fallbacks"] += int(not query.get("exact_match"))
            choice = agent.select(engine, player)
            agent.execute(engine, player, choice)
        safety += 1
    if not engine.hand_complete:
        raise RuntimeError("LBR hand did not terminate")
    return (engine.stacks[probe_seat] - before) / engine.big_blind, int(engine.street)


def local_best_response_probe(agent, hands: int = 500, seed: int = 0, stack_bb: float | None = None) -> dict:
    """Restricted-response win rate in bb/100; positive means exploitable."""
    if not hasattr(agent, "tree") or not hasattr(agent, "strategy"):
        raise TypeError("GPU LBR requires a GpuBlueprintAgent-compatible strategy")
    if hands < 2:
        raise ValueError("LBR requires at least two hands for one duplicate pair")
    agent.subgame_search = False
    stack_bb = float(stack_bb or agent.tree.config.stack_bb)
    if stack_bb <= 0:
        raise ValueError("stack_bb must be positive")
    pairs = max(1, hands // 2)
    pair_samples: list[float] = []
    street_samples: dict[int, list[float]] = {street: [] for street in range(4)}
    diagnostics = {
        "range_updates": 0,
        "equity_queries": 0,
        "fold_queries": 0,
        "blueprint_decisions": 0,
        "exact_nodes": 0,
        "fallbacks": 0,
    }

    for pair in range(pairs):
        deal_seed = seed * 1_000_003 + pair
        first, first_street = _play_lbr_hand(agent, 0, deal_seed, stack_bb, diagnostics)
        second, second_street = _play_lbr_hand(agent, 1, deal_seed, stack_bb, diagnostics)
        pair_samples.append((first + second) / 2.0)
        street_samples[first_street].append(first)
        street_samples[second_street].append(second)

    mean, low, high = _mean_ci(pair_samples)
    by_terminal_street = {}
    for street, samples in street_samples.items():
        if not samples:
            continue
        street_mean, street_low, street_high = _mean_ci(samples)
        by_terminal_street[_STREET_NAMES[street]] = {
            "bb_per_100": round(street_mean * 100.0, 2),
            "ci_low_bb_per_100": round(street_low * 100.0, 2),
            "ci_high_bb_per_100": round(street_high * 100.0, 2),
            "hands": len(samples),
        }

    decisions = max(1, diagnostics["blueprint_decisions"])
    return {
        # Per-pair values, so an on/off comparison can be PAIRED. Unpaired LBR at
        # 400 pairs is +-140 bb/100, too wide to resolve anything but a huge
        # change; the same deals on both arms cancel most card variance.
        "pair_samples": [round(float(value), 6) for value in pair_samples],
        "lbr_bb_per_100": round(mean * 100.0, 2),
        "ci_low_bb_per_100": round(low * 100.0, 2),
        "ci_high_bb_per_100": round(high * 100.0, 2),
        "hands": pairs * 2,
        "pairs": pairs,
        "stack_bb": stack_bb,
        "by_terminal_street": by_terminal_street,
        "diagnostics": {
            **diagnostics,
            "exact_node_rate": round(diagnostics["exact_nodes"] / decisions, 6),
            "fallback_rate": round(diagnostics["fallbacks"] / decisions, 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-blueprint Local Best Response probe")
    parser.add_argument("--pairs", type=int, default=250, help="duplicate deal pairs")
    parser.add_argument("--hands", type=int, default=None, help="deprecated alias; rounded down to duplicate pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stack-bb", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default="backend/data/gpu_blueprint")
    parser.add_argument("--checkpoint", type=str, default=None)
    arguments = parser.parse_args()

    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    data_dir = Path(arguments.data_dir)
    checkpoint = Path(arguments.checkpoint) if arguments.checkpoint else data_dir / "champion.npz"
    agent = GpuBlueprintAgent.try_load(checkpoint)
    if agent is None:
        raise SystemExit(f"no GPU blueprint found at {checkpoint}")
    hands = arguments.hands if arguments.hands is not None else arguments.pairs * 2
    report = local_best_response_probe(agent, hands=hands, seed=arguments.seed, stack_bb=arguments.stack_bb)
    print(
        f"LBR: {report['lbr_bb_per_100']:+.2f} bb/100 "
        f"[{report['ci_low_bb_per_100']:+.2f}, {report['ci_high_bb_per_100']:+.2f}] "
        f"over {report['hands']} hands; fallback={report['diagnostics']['fallback_rate']:.2%} "
        "(positive = exploitable)"
    )


if __name__ == "__main__":
    main()
