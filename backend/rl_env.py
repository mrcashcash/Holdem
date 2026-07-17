"""Numerical observations, public betting memory, and legal neural poker actions."""

from __future__ import annotations

import math
import os
import random
from collections import OrderedDict

from .poker import SUITS, HeadsUpHoldem, best_score, new_deck
from .vectorized_engine import legal_masks_batch as compiled_legal_masks_batch, sampled_equity, semantic_transition_plans_batch

ACTION_NAMES = ("fold", "check_call", "raise", "all_in")
ACTION_COUNT = len(ACTION_NAMES)
# A single raise decision owns the entire legal no-limit interval.  The old
# representation used six categorical labels which all sampled from the same
# interval, making raise policy probabilities non-identifiable.
RAISE_ACTIONS = (2,)
RAISE_ACTION_COUNT = len(RAISE_ACTIONS)
try:
    PREFLOP_OPEN_RAISE_CAP_BB = min(20.0, max(2.0, float(os.environ.get("HOLDEM_PREFLOP_OPEN_RAISE_CAP_BB", "3.5"))))
except ValueError:
    PREFLOP_OPEN_RAISE_CAP_BB = 3.5
try:
    PREFLOP_THREE_BET_POT_CAP_MULTIPLIER = min(4.0, max(1.0, float(os.environ.get("HOLDEM_PREFLOP_THREE_BET_POT_CAP_MULTIPLIER", "2"))))
except ValueError:
    PREFLOP_THREE_BET_POT_CAP_MULTIPLIER = 2.0
PUBLIC_FEATURE_SIZE = 76
PRIVATE_CARD_FEATURE_SIZE = 52
BOARD_CARD_FEATURE_SIZE = 52
OBSERVATION_SIZE = PUBLIC_FEATURE_SIZE + PRIVATE_CARD_FEATURE_SIZE + BOARD_CARD_FEATURE_SIZE
RANGE_BUCKETS = 1_326
ACTION_CONTEXT_SIZE = 10
_EQUITY_CACHE: OrderedDict[tuple, tuple[float, float]] = OrderedDict()
_EQUITY_CANDIDATE_CACHE: OrderedDict[tuple[tuple[int, str], ...], tuple[list[tuple[int, str]], list[tuple[tuple[int, str], tuple[int, str]]]]] = OrderedDict()
_ACTION_CODES = {"blind": 0, "fold": 1, "check": 2, "call": 3, "raise": 4}
# The rules engine owns card identity. Deriving every local suit mapping from
# its canonical tuple prevents source-encoding differences from corrupting
# observations during background self-play.
_SUIT_ORDER = {suit: index for index, suit in enumerate(SUITS)}
_SUIT_VALUE = {suit: index + 1 for index, suit in enumerate(SUITS)}


def _straight_draw(ranks: list[int]) -> float:
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    best = 0
    for high in range(14, 4, -1):
        best = max(best, sum(rank in unique for rank in range(high - 4, high + 1)))
    return 1.0 if best >= 4 else 0.5 if best >= 3 else 0.0


def _preflop_proxy(hole: list[tuple[int, str]]) -> float:
    ranks = sorted((card[0] for card in hole), reverse=True)
    pair_bonus = 0.22 if ranks[0] == ranks[1] else 0.0
    suited_bonus = 0.035 if hole[0][1] == hole[1][1] else 0.0
    connected_bonus = max(0.0, 0.04 - abs(ranks[0] - ranks[1]) * 0.006)
    return min(0.86, max(0.14, 0.18 + (ranks[0] + ranks[1]) / 42 + pair_bonus + suited_bonus + connected_bonus))


def card_index(card: tuple[int, str]) -> int:
    """Stable 0..51 identity that preserves rank and suit blockers."""
    return (card[0] - 2) * 4 + _SUIT_ORDER[card[1]]


def hand_bucket(cards: list[tuple[int, str]] | tuple[tuple[int, str], tuple[int, str]]) -> int:
    """Map a two-card holding to its unique exact-combination range index (0..1325)."""
    first, second = sorted(card_index(card) for card in cards)
    return first * (103 - first) // 2 + (second - first - 1)


def _range_weight(cards: tuple[tuple[int, str], tuple[int, str]], events: list[dict[str, int | str]], viewer: int, range_bias: list[float] | None) -> float:
    ranks = sorted((card[0] for card in cards), reverse=True)
    pair = ranks[0] == ranks[1]
    suited = cards[0][1] == cards[1][1]
    strength = (ranks[0] + ranks[1]) / 28 + (0.48 if pair else 0) + (0.09 if suited else 0)
    weight = 0.25 + strength
    for event in events:
        if event["player"] == viewer:
            continue
        action = event["action"]
        if action == "raise":
            weight *= 0.25 + strength * strength
        elif action == "call":
            weight *= 0.55 + strength * 0.7
        elif action == "check":
            weight *= 1.25 - min(0.55, strength * 0.28)
    if range_bias is not None:
        weight *= 0.25 + range_bias[hand_bucket(cards)] * RANGE_BUCKETS
    return max(1e-5, weight)


def estimate_range_equity(game: HeadsUpHoldem, player: int, samples: int | None = None, range_bias: list[float] | None = None) -> tuple[float, float]:
    """Estimate equity against a public-action-weighted legal opponent range."""
    hole, board = game.hole_cards[player], game.community
    if not board:
        return _preflop_proxy(hole), 0.85
    sample_count = samples or (4 if len(board) == 3 else 6 if len(board) == 4 else 8)
    action_key = tuple((event["player"], event["action"], event["amount"], event["street"]) for event in game.public_actions[-8:])
    bias_key = tuple((index, round(value, 3)) for index, value in sorted(enumerate(range_bias or []), key=lambda item: item[1], reverse=True)[:16])
    key = (tuple(sorted(hole + board)), action_key, bias_key)
    cached = _EQUITY_CACHE.get(key)
    if cached is not None:
        _EQUITY_CACHE.move_to_end(key)
        return cached
    known = set(hole + board)
    known_key = tuple(sorted(known))
    candidate_state = _EQUITY_CANDIDATE_CACHE.get(known_key)
    if candidate_state is None:
        deck = [card for card in new_deck() if card not in known]
        candidates = [(deck[first], deck[second]) for first in range(len(deck)) for second in range(first + 1, len(deck))]
        _EQUITY_CANDIDATE_CACHE[known_key] = (deck, candidates)
        if len(_EQUITY_CANDIDATE_CACHE) > 2_048:
            _EQUITY_CANDIDATE_CACHE.popitem(last=False)
    else:
        deck, candidates = candidate_state
        _EQUITY_CANDIDATE_CACHE.move_to_end(known_key)
    weights = [_range_weight(cards, game.public_actions, player, range_bias) for cards in candidates]
    seed = sum(card[0] * 11 + _SUIT_VALUE[card[1]] for card in known) + len(board) * 97
    rng = random.Random(seed)
    equity = 0.0
    hero_hands: list[list[tuple[int, str]]] = []
    opponent_hands: list[list[tuple[int, str]]] = []
    missing = 5 - len(board)
    for _ in range(sample_count):
        opponent = rng.choices(candidates, weights=weights, k=1)[0]
        runout = rng.sample([card for card in deck if card not in opponent], missing)
        hero_hands.append(hole + board + runout)
        opponent_hands.append(list(opponent) + board + runout)
    compiled_equity = sampled_equity(hero_hands, opponent_hands)
    if compiled_equity is None:
        for own_cards, opponent_cards in zip(hero_hands, opponent_hands):
            own_score = best_score(own_cards)
            opponent_score = best_score(opponent_cards)
            equity += 1.0 if own_score > opponent_score else 0.5 if own_score == opponent_score else 0.0
    else:
        equity = compiled_equity * sample_count
    total = sum(weights)
    entropy = -sum((weight / total) * math.log(weight / total + 1e-12) for weight in weights) / math.log(len(weights))
    result = (equity / sample_count, entropy)
    _EQUITY_CACHE[key] = result
    if len(_EQUITY_CACHE) > 4_096:
        _EQUITY_CACHE.popitem(last=False)
    return result


def _history_features(game: HeadsUpHoldem, player: int) -> list[float]:
    events = game.public_actions[-4:]
    features: list[float] = []
    for event in events:
        actor = 1.0 if event["player"] == player else -1.0
        action = _ACTION_CODES.get(str(event["action"]), 0) / 4
        amount = min(2.0, int(event["amount"]) / game.initial_stack)
        street = int(event["street"]) / 3
        features.extend((actor, action, amount, street))
    features.extend([0.0] * (16 - len(features)))
    features.extend(min(1.0, count / 8) for count in game.street_action_counts)
    last_aggressor = next((event["player"] for event in reversed(game.public_actions) if event["action"] == "raise"), None)
    features.append(1.0 if last_aggressor == player else -1.0 if last_aggressor is not None else 0.0)
    return features


def public_belief_features(game: HeadsUpHoldem, player: int) -> list[float]:
    """Compact public state used by the range model and depth-limited resolver."""
    opponent = 1 - player
    raises = [event for event in game.public_actions if event["action"] == "raise"]
    player_raises = sum(event["player"] == player for event in raises)
    opponent_raises = sum(event["player"] == opponent for event in raises)
    last_aggressor = raises[-1]["player"] if raises else None
    board_ranks = [card[0] for card in game.community]
    return [
        game.street / 3,
        min(2.0, game.pot / game.initial_stack),
        min(2.0, min(game.stacks) / game.initial_stack),
        min(1.0, game.to_call(player) / max(1, game.pot + game.to_call(player))),
        min(1.0, player_raises / 4),
        min(1.0, opponent_raises / 4),
        min(1.0, len(game.public_actions) / 10),
        1.0 if last_aggressor == player else -1.0 if last_aggressor is not None else 0.0,
        max(board_ranks, default=0) / 14,
        float(len(board_ranks) != len(set(board_ranks))),
    ]


def action_context_features(game: HeadsUpHoldem, player: int) -> list[float]:
    """Public pre-action context used by the learned range-action likelihood head."""
    return [
        game.street / 3,
        min(2.0, game.pot / game.initial_stack),
        min(1.0, game.to_call(player) / max(1, game.pot + game.to_call(player))),
        min(2.0, game.stacks[player] / game.initial_stack),
        min(2.0, min(game.stacks) / game.initial_stack),
        min(2.0, max(game.round_bets) / game.initial_stack),
        float(game.button == player),
        float(game.raise_open[player]),
        min(2.0, game.round_bets[player] / game.initial_stack),
        min(1.0, len(game.public_actions) / 10),
    ]


def event_context_features(event: dict[str, int | str], initial_stack: int) -> list[float]:
    """Rebuild an action's public pre-state from the event record without hidden cards."""
    pot = int(event.get("pot_before", 0))
    to_call = int(event.get("to_call_before", 0))
    stack = int(event.get("stack_before", 0))
    effective_stack = int(event.get("effective_stack_before", 0))
    current_bet = int(event.get("current_bet_before", 0))
    player_bet = int(event.get("player_bet_before", 0))
    return [
        int(event.get("street", 0)) / 3,
        min(2.0, pot / initial_stack),
        min(1.0, to_call / max(1, pot + to_call)),
        min(2.0, stack / initial_stack),
        min(2.0, effective_stack / initial_stack),
        min(2.0, current_bet / initial_stack),
        float(int(event.get("button_before", 0))),
        float(int(event.get("raise_open_before", 0))),
        min(2.0, player_bet / initial_stack),
        min(1.0, int(event.get("action_count_before", 0)) / 10),
    ]


def observation(game: HeadsUpHoldem, player: int) -> list[float]:
    """Return a normalized observation with equity and public action-sequence context."""
    hole = game.hole_cards[player]
    opponent = 1 - player
    ranks = sorted((card[0] for card in hole), reverse=True)
    board = game.community
    board_ranks = [card[0] for card in board]
    board_suits = [card[1] for card in board]
    all_cards = hole + board
    made_category = best_score(all_cards)[0] if len(all_cards) >= 5 else 0
    board_histogram = [board_ranks.count(rank) / 4 for rank in range(2, 15)]
    suit_histogram = [board_suits.count(suit) / 5 for suit in SUITS]
    suit_peak = max((sum(card[1] == suit for card in all_cards) for suit in {card[1] for card in all_cards}), default=0)
    current_bet = max(game.round_bets)
    self_stack = game.stacks[player]
    private_cards = {card_index(card) for card in hole}
    board_cards = {card_index(card) for card in board}
    features = [
        ranks[0] / 14, ranks[1] / 14, float(ranks[0] == ranks[1]), float(hole[0][1] == hole[1][1]), min(1.0, abs(ranks[0] - ranks[1]) / 12),
        *[float(game.street == index) for index in range(4)], *board_histogram, *suit_histogram,
        max(board_ranks, default=0) / 14, float(len(board_ranks) != len(set(board_ranks))), made_category / 8,
        min(1.0, sum(rank in board_ranks for rank in ranks) / 2), min(1.0, suit_peak / 5), _straight_draw([card[0] for card in all_cards]),
        min(2.0, game.pot / game.initial_stack), min(2.0, self_stack / game.initial_stack), min(2.0, game.stacks[opponent] / game.initial_stack),
        min(1.0, game.to_call(player) / max(1, self_stack)), min(2.0, current_bet / game.initial_stack), min(2.0, game.round_bets[player] / game.initial_stack),
        float(game.button == player), float(game.raise_open[player]), float(game.acted[player]), float(game.acted[opponent]), len(board) / 5,
        *_history_features(game, player), *public_belief_features(game, player), *estimate_range_equity(game, player),
        # Keep public board cards and private blocker cards separate.  The policy
        # always sees its own cards, while belief/search inputs remain public-only.
        *[float(index in private_cards) for index in range(52)],
        *[float(index in board_cards) for index in range(52)],
    ]
    if len(features) != OBSERVATION_SIZE:
        raise RuntimeError(f"Observation has {len(features)} features; expected {OBSERVATION_SIZE}")
    return features


def legal_action_mask(game: HeadsUpHoldem, player: int) -> list[bool]:
    legal = game.legal_actions(player)
    can_raise = bool(legal.get("raise"))
    return [bool(legal.get("fold")), bool(legal.get("check") or legal.get("call")), can_raise, bool(legal.get("all_in"))]


def legal_action_masks_batch(games: list[HeadsUpHoldem], players: list[int]) -> list[list[bool]]:
    """Use the compiled semantic-mask primitive when the vectorized backend is active."""
    compiled = compiled_legal_masks_batch(
        [game.round_bets for game in games],
        [game.stacks for game in games],
        [game.last_raise for game in games],
        [game.raise_open for game in games],
        players,
    ) if games else []
    if compiled is not None:
        return compiled
    return [legal_action_mask(game, player) for game, player in zip(games, players)]


def preflop_voluntary_raise_count(game: HeadsUpHoldem) -> int:
    """Count voluntary raises in the current preflop round, excluding posted blinds."""
    return sum(event.get("street") == 0 and event.get("action") == "raise" for event in game.public_actions)


def normal_raise_bounds(game: HeadsUpHoldem, player: int) -> tuple[int, int]:
    """Return the normal-raise interval with conservative opening and 3-bet caps."""
    legal = game.legal_actions(player)
    minimum = int(legal["raise_min"])
    maximum = int(legal["raise_max"])
    blind_only_preflop = game.street == 0 and max(game.round_bets) <= game.big_blind
    if blind_only_preflop:
        open_cap = round(game.big_blind * PREFLOP_OPEN_RAISE_CAP_BB)
        maximum = min(maximum, max(minimum, open_cap))
    elif game.street == 0 and preflop_voluntary_raise_count(game) == 1:
        three_bet_cap = round(game.pot * PREFLOP_THREE_BET_POT_CAP_MULTIPLIER)
        maximum = min(maximum, max(minimum, three_bet_cap))
    return minimum, maximum


def continuous_raise_target(game: HeadsUpHoldem, player: int, fraction: float) -> int:
    """Map a learned fraction into the normal-raise interval; all-in remains separate."""
    minimum, maximum = normal_raise_bounds(game, player)
    unit_fraction = min(0.995, max(0.005, float(fraction)))
    return min(maximum, max(minimum, round(minimum + (maximum - minimum) * unit_fraction)))


def execute_action(game: HeadsUpHoldem, player: int, action: int, raise_fraction: float | None = None) -> None:
    """Translate a neural action into the concrete engine, using a learned legal raise size when supplied."""
    legal = game.legal_actions(player)
    if action == 0:
        game.act(player, "fold")
    elif action == 1:
        game.act(player, "check" if legal.get("check") else "call")
    elif action == 2:
        game.act(player, "raise", continuous_raise_target(game, player, 0.5 if raise_fraction is None else raise_fraction))
    elif action == 3:
        game.act(player, "all_in")
    else:
        raise ValueError(f"Unknown neural action {action}")


def execute_actions_batch(actions: list[tuple[HeadsUpHoldem, int, int, float | None]]) -> int:
    """Apply independent rollout actions through an opt-in compiled rule planner.

    The planner batches repeated stack/bet/raise-bound calculations.  Each
    result is still applied by ``HeadsUpHoldem`` so card dealing, public
    history, street changes, and the browser protocol retain one canonical
    implementation.  Returns the number of actions planned by Numba.
    """
    if not actions:
        return 0
    plans = semantic_transition_plans_batch(
        [game.round_bets for game, _, _, _ in actions],
        [game.stacks for game, _, _, _ in actions],
        [game.pot for game, _, _, _ in actions],
        [game.last_raise for game, _, _, _ in actions],
        [game.raise_open for game, _, _, _ in actions],
        [game.acted for game, _, _, _ in actions],
        [game.street for game, _, _, _ in actions],
        [game.big_blind for game, _, _, _ in actions],
        [preflop_voluntary_raise_count(game) for game, _, _, _ in actions],
        [player for _, player, _, _ in actions],
        [action for _, _, action, _ in actions],
        [0.5 if fraction is None else fraction for _, _, _, fraction in actions],
        PREFLOP_OPEN_RAISE_CAP_BB,
        PREFLOP_THREE_BET_POT_CAP_MULTIPLIER,
    )
    if plans is None:
        for game, player, action, fraction in actions:
            execute_action(game, player, action, fraction)
        return 0
    labels, amounts = plans
    for (game, player, _, _), label, amount in zip(actions, labels, amounts):
        if label == 0:
            game.act(player, "fold")
        elif label == 1:
            legal = game.legal_actions(player)
            game.act(player, "check" if legal.get("check") else "call")
        elif label == 2:
            game.act(player, "raise", int(amount))
        elif label == 3:
            game.act(player, "all_in")
        else:
            raise ValueError(f"Unknown compiled semantic action {label}")
    return len(actions)
