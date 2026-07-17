"""Optional Numba-compiled primitives for the self-play rollout hot path.

The browser table and complete public hand history remain owned by
``HeadsUpHoldem``.  This module concentrates the expensive, pure computations
that are safe to run over arrays: seven-card scoring, sampled-showdown equity,
legal semantic masks, and continuous raise interpolation.  It is deliberately
optional so a missing native/JIT dependency always falls back to the reference
Python implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

try:
    import numpy as np
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    np = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


_BASE = 15
_BASE_2 = _BASE * _BASE
_BASE_3 = _BASE_2 * _BASE
_BASE_4 = _BASE_3 * _BASE
_BASE_5 = _BASE_4 * _BASE
_SUIT_INDEX = {"♠": 0, "♥": 1, "♦": 2, "♣": 3}


@dataclass(frozen=True)
class VectorizedRuntime:
    requested: str
    enabled: bool
    mode: str
    reason: str


def resolve_vectorized_runtime() -> VectorizedRuntime:
    requested = os.environ.get("HOLDEM_ROLLOUT_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "python", "vectorized", "compiled"}:
        requested = "auto"
    effective = "compiled" if requested == "auto" and NUMBA_AVAILABLE else "python" if requested == "auto" else requested
    if effective == "compiled" and NUMBA_AVAILABLE:
        return VectorizedRuntime(requested, True, "compiled-numba", "Numba-compiled action-transition planning, equity, scoring, masks, and sizing enabled")
    if effective == "vectorized" and NUMBA_AVAILABLE:
        return VectorizedRuntime(requested, True, "vectorized-numba", "Numba-compiled equity, scoring, masks, and sizing enabled")
    if effective in {"vectorized", "compiled"}:
        return VectorizedRuntime(requested, False, "python-batched", "Numba is unavailable; using the reference Python backend")
    return VectorizedRuntime(requested, False, "python-batched", "Reference Python backend explicitly requested")


def vectorized_enabled() -> bool:
    return resolve_vectorized_runtime().enabled


def compiled_transition_enabled() -> bool:
    """Whether the opt-in compiled action-transition planner is selected."""
    runtime = resolve_vectorized_runtime()
    return runtime.enabled and runtime.mode == "compiled-numba"


def card_id(card: tuple[int, str]) -> int:
    """Map the reference engine's rank/suit tuple to a compact 0..51 integer."""
    return (card[0] - 2) * 4 + _SUIT_INDEX[card[1]]


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _straight_high(rank_counts):
        for high in range(14, 4, -1):
            complete = True
            for rank in range(high - 4, high + 1):
                if rank_counts[rank] == 0:
                    complete = False
                    break
            if complete:
                return high
        if rank_counts[14] > 0:
            wheel = True
            for rank in range(2, 6):
                if rank_counts[rank] == 0:
                    wheel = False
                    break
            if wheel:
                return 5
        return 0


    @njit(cache=True)
    def _pack(category, first, second, third, fourth, fifth):
        return category * _BASE_5 + first * _BASE_4 + second * _BASE_3 + third * _BASE_2 + fourth * _BASE + fifth


    @njit(cache=True)
    def _evaluate_seven(cards):
        rank_counts = np.zeros(15, dtype=np.int64)
        suit_counts = np.zeros(4, dtype=np.int64)
        suit_rank_counts = np.zeros((4, 15), dtype=np.int64)
        for index in range(7):
            card = cards[index]
            rank = card // 4 + 2
            suit = card % 4
            rank_counts[rank] += 1
            suit_counts[suit] += 1
            suit_rank_counts[suit, rank] += 1

        for suit in range(4):
            if suit_counts[suit] >= 5:
                straight_flush = _straight_high(suit_rank_counts[suit])
                if straight_flush:
                    return _pack(8, straight_flush, 0, 0, 0, 0)

        quad = 0
        for rank in range(14, 1, -1):
            if rank_counts[rank] == 4:
                quad = rank
                break
        if quad:
            kicker = 0
            for rank in range(14, 1, -1):
                if rank != quad and rank_counts[rank] > 0:
                    kicker = rank
                    break
            return _pack(7, quad, kicker, 0, 0, 0)

        triple = 0
        pair = 0
        for rank in range(14, 1, -1):
            if rank_counts[rank] >= 3 and triple == 0:
                triple = rank
            elif rank_counts[rank] >= 2 and pair == 0:
                pair = rank
        if triple and pair:
            return _pack(6, triple, pair, 0, 0, 0)

        for suit in range(4):
            if suit_counts[suit] >= 5:
                values = np.zeros(5, dtype=np.int64)
                position = 0
                for rank in range(14, 1, -1):
                    if suit_rank_counts[suit, rank] > 0:
                        values[position] = rank
                        position += 1
                        if position == 5:
                            break
                return _pack(5, values[0], values[1], values[2], values[3], values[4])

        straight = _straight_high(rank_counts)
        if straight:
            return _pack(4, straight, 0, 0, 0, 0)

        if triple:
            kickers = np.zeros(2, dtype=np.int64)
            position = 0
            for rank in range(14, 1, -1):
                if rank != triple and rank_counts[rank] > 0:
                    kickers[position] = rank
                    position += 1
                    if position == 2:
                        break
            return _pack(3, triple, kickers[0], kickers[1], 0, 0)

        first_pair = 0
        second_pair = 0
        for rank in range(14, 1, -1):
            if rank_counts[rank] >= 2:
                if first_pair == 0:
                    first_pair = rank
                elif second_pair == 0:
                    second_pair = rank
                    break
        if second_pair:
            kicker = 0
            for rank in range(14, 1, -1):
                if rank != first_pair and rank != second_pair and rank_counts[rank] > 0:
                    kicker = rank
                    break
            return _pack(2, first_pair, second_pair, kicker, 0, 0)

        if first_pair:
            kickers = np.zeros(3, dtype=np.int64)
            position = 0
            for rank in range(14, 1, -1):
                if rank != first_pair and rank_counts[rank] > 0:
                    kickers[position] = rank
                    position += 1
                    if position == 3:
                        break
            return _pack(1, first_pair, kickers[0], kickers[1], kickers[2], 0)

        values = np.zeros(5, dtype=np.int64)
        position = 0
        for rank in range(14, 1, -1):
            if rank_counts[rank] > 0:
                values[position] = rank
                position += 1
                if position == 5:
                    break
        return _pack(0, values[0], values[1], values[2], values[3], values[4])


    @njit(cache=True)
    def _equity_from_hands(hero_hands, opponent_hands):
        wins = 0.0
        count = hero_hands.shape[0]
        for index in range(count):
            hero_score = _evaluate_seven(hero_hands[index])
            opponent_score = _evaluate_seven(opponent_hands[index])
            if hero_score > opponent_score:
                wins += 1.0
            elif hero_score == opponent_score:
                wins += 0.5
        return wins / max(1, count)


    @njit(cache=True)
    def _legal_masks(round_bets, stacks, last_raises, raise_open, players):
        count = players.shape[0]
        masks = np.zeros((count, 4), dtype=np.bool_)
        for row in range(count):
            player = players[row]
            current_high = round_bets[row, 0]
            if round_bets[row, 1] > current_high:
                current_high = round_bets[row, 1]
            call_amount = current_high - round_bets[row, player]
            maximum = round_bets[row, player] + stacks[row, player]
            can_raise = raise_open[row, player] and stacks[row, player] > call_amount and maximum > current_high
            masks[row, 0] = call_amount > 0
            masks[row, 1] = True
            masks[row, 2] = can_raise
            masks[row, 3] = stacks[row, player] > 0 and (stacks[row, player] <= call_amount or can_raise)
        return masks


    @njit(cache=True)
    def _raise_targets(minimums, maximums, fractions):
        count = minimums.shape[0]
        targets = np.empty(count, dtype=np.int64)
        for index in range(count):
            fraction = min(0.995, max(0.005, fractions[index]))
            raw_target = minimums[index] + (maximums[index] - minimums[index]) * fraction
            targets[index] = min(maximums[index], max(minimums[index], int(round(raw_target))))
        return targets


    @njit(cache=True)
    def _semantic_transition_plans(round_bets, stacks, pots, last_raises, raise_open, acted, streets, big_blinds, preflop_raise_counts, players, actions, fractions, preflop_open_raise_cap_bb, preflop_three_bet_pot_cap_multiplier):
        """Plan immediate semantic-action transitions over independent heads-up states.

        The authoritative Python game still applies the returned plan, including
        street changes, cards, history, and showdown.  Keeping this compact
        transition kernel pure lets rollout collection batch the repeated
        stack/bet/raise legality calculations without changing live rules.
        """
        count = players.shape[0]
        labels = np.empty(count, dtype=np.int8)
        amounts = np.zeros(count, dtype=np.int64)
        next_bets = round_bets.copy()
        next_stacks = stacks.copy()
        next_pots = pots.copy()
        next_last_raises = last_raises.copy()
        next_raise_open = raise_open.copy()
        next_acted = acted.copy()
        terminal = np.zeros(count, dtype=np.bool_)
        for row in range(count):
            player = players[row]
            opponent = 1 - player
            high = next_bets[row, 0]
            if next_bets[row, 1] > high:
                high = next_bets[row, 1]
            call_amount = high - next_bets[row, player]
            maximum = next_bets[row, player] + next_stacks[row, player]
            minimum = high + next_last_raises[row]
            action = actions[row]
            labels[row] = action
            if action == 0:
                if call_amount > 0:
                    terminal[row] = True
                    continue
                # The compiled planner is defensive because callers can bypass
                # the policy mask. A zero-cost fold has check semantics.
                action = 1
                labels[row] = 1
            if action == 3 and next_stacks[row, player] <= call_amount:
                action = 1
                labels[row] = 1
            if action == 1:
                paid = call_amount
                if paid > next_stacks[row, player]:
                    paid = next_stacks[row, player]
                next_stacks[row, player] -= paid
                next_bets[row, player] += paid
                next_pots[row] += paid
                next_acted[row, player] = True
                next_raise_open[row, player] = False
                if next_stacks[row, player] == 0:
                    terminal[row] = True
                continue
            if action == 3:
                target = maximum
            else:
                fraction = min(0.995, max(0.005, fractions[row]))
                normal_maximum = maximum
                if streets[row] == 0:
                    if high <= big_blinds[row]:
                        open_cap = int(round(big_blinds[row] * preflop_open_raise_cap_bb))
                        normal_maximum = min(maximum, max(minimum, open_cap))
                    elif preflop_raise_counts[row] == 1:
                        three_bet_cap = int(round(pots[row] * preflop_three_bet_pot_cap_multiplier))
                        normal_maximum = min(maximum, max(minimum, three_bet_cap))
                target = int(round(minimum + (normal_maximum - minimum) * fraction))
                target = min(normal_maximum, max(minimum, target))
            can_raise = next_raise_open[row, player] and next_stacks[row, player] > call_amount and maximum > high
            if not can_raise or target <= high:
                labels[row] = 1
                paid = call_amount
                if paid > next_stacks[row, player]:
                    paid = next_stacks[row, player]
                next_stacks[row, player] -= paid
                next_bets[row, player] += paid
                next_pots[row] += paid
                next_acted[row, player] = True
                next_raise_open[row, player] = False
                continue
            amounts[row] = target
            paid = target - next_bets[row, player]
            next_stacks[row, player] -= paid
            next_bets[row, player] = target
            next_pots[row] += paid
            raise_size = target - high
            if raise_size >= next_last_raises[row]:
                next_last_raises[row] = raise_size
                next_raise_open[row, 0] = True
                next_raise_open[row, 1] = True
                next_raise_open[row, player] = False
            elif high == 0:
                next_raise_open[row, opponent] = True
                next_raise_open[row, player] = False
            else:
                next_raise_open[row, player] = False
            next_acted[row, 0] = False
            next_acted[row, 1] = False
            next_acted[row, player] = True
        return labels, amounts, next_bets, next_stacks, next_pots, next_last_raises, next_raise_open, next_acted, terminal


def score_seven(cards: Sequence[tuple[int, str]]) -> tuple[int, ...] | None:
    """Return a lexicographically compatible score for exactly seven cards."""
    if not vectorized_enabled() or len(cards) != 7:
        return None
    assert np is not None
    encoded = np.asarray([card_id(card) for card in cards], dtype=np.int64)
    packed = int(_evaluate_seven(encoded))
    category = packed // _BASE_5
    packed %= _BASE_5
    first = packed // _BASE_4
    packed %= _BASE_4
    second = packed // _BASE_3
    packed %= _BASE_3
    third = packed // _BASE_2
    packed %= _BASE_2
    fourth = packed // _BASE
    fifth = packed % _BASE
    score = (category, first, second, third, fourth, fifth)
    score_lengths = (6, 5, 4, 4, 2, 6, 3, 3, 2)
    return score[:score_lengths[category]]


def sampled_equity(hero_hands: Sequence[Sequence[tuple[int, str]]], opponent_hands: Sequence[Sequence[tuple[int, str]]]) -> float | None:
    """Evaluate equally sized sampled seven-card hands in one compiled loop."""
    if not vectorized_enabled() or not hero_hands:
        return None
    assert np is not None
    hero = np.asarray([[card_id(card) for card in hand] for hand in hero_hands], dtype=np.int64)
    opponent = np.asarray([[card_id(card) for card in hand] for hand in opponent_hands], dtype=np.int64)
    if hero.shape != opponent.shape or hero.ndim != 2 or hero.shape[1] != 7:
        return None
    return float(_equity_from_hands(hero, opponent))


def legal_masks_batch(round_bets: Sequence[Sequence[int]], stacks: Sequence[Sequence[int]], last_raises: Sequence[int], raise_open: Sequence[Sequence[bool]], players: Sequence[int]) -> list[list[bool]] | None:
    """Return production semantic masks for many active states at once."""
    if not vectorized_enabled():
        return None
    assert np is not None
    return _legal_masks(
        np.asarray(round_bets, dtype=np.int64),
        np.asarray(stacks, dtype=np.int64),
        np.asarray(last_raises, dtype=np.int64),
        np.asarray(raise_open, dtype=np.bool_),
        np.asarray(players, dtype=np.int64),
    ).tolist()


def raise_targets_batch(minimums: Sequence[int], maximums: Sequence[int], fractions: Sequence[float]) -> list[int] | None:
    """Interpolate legal raise-to amounts in compiled array code."""
    if not vectorized_enabled():
        return None
    assert np is not None
    return _raise_targets(
        np.asarray(minimums, dtype=np.int64),
        np.asarray(maximums, dtype=np.int64),
        np.asarray(fractions, dtype=np.float64),
    ).tolist()


def semantic_transition_plans_batch(round_bets: Sequence[Sequence[int]], stacks: Sequence[Sequence[int]], pots: Sequence[int], last_raises: Sequence[int], raise_open: Sequence[Sequence[bool]], acted: Sequence[Sequence[bool]], streets: Sequence[int], big_blinds: Sequence[int], preflop_raise_counts: Sequence[int], players: Sequence[int], actions: Sequence[int], fractions: Sequence[float], preflop_open_raise_cap_bb: float, preflop_three_bet_pot_cap_multiplier: float) -> tuple[list[int], list[int]] | None:
    """Return compiled semantic-action labels and raise-to amounts for a batch.

    The post-transition arrays are deliberately retained inside the Numba
    kernel for rule planning, while the reference game applies the resulting
    action and remains the sole source of public state.
    """
    if not compiled_transition_enabled() or not players:
        return None
    assert np is not None
    labels, amounts, _, _, _, _, _, _, _ = _semantic_transition_plans(
        np.asarray(round_bets, dtype=np.int64),
        np.asarray(stacks, dtype=np.int64),
        np.asarray(pots, dtype=np.int64),
        np.asarray(last_raises, dtype=np.int64),
        np.asarray(raise_open, dtype=np.bool_),
        np.asarray(acted, dtype=np.bool_),
        np.asarray(streets, dtype=np.int64),
        np.asarray(big_blinds, dtype=np.int64),
        np.asarray(preflop_raise_counts, dtype=np.int64),
        np.asarray(players, dtype=np.int64),
        np.asarray(actions, dtype=np.int64),
        np.asarray(fractions, dtype=np.float64),
        float(preflop_open_raise_cap_bb),
        float(preflop_three_bet_pot_cap_multiplier),
    )
    return labels.tolist(), amounts.tolist()
