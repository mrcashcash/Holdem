"""A dependency-free, heads-up no-limit Texas Hold'em rules engine."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Literal

from .vectorized_engine import score_seven

Rank = int
Card = tuple[Rank, str]
Action = Literal["fold", "check", "call", "raise", "all_in"]

RANK_LABELS = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T", 9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}
SUITS = ("♠", "♥", "♦", "♣")
STREETS = ("preflop", "flop", "turn", "river")


class InvalidAction(ValueError):
    """Raised when an action is not legal for the current game state."""


def card_text(card: Card) -> str:
    return f"{RANK_LABELS[card[0]]}{card[1]}"


def new_deck() -> list[Card]:
    return [(rank, suit) for rank in range(2, 15) for suit in SUITS]


def _straight_high(ranks: list[int]) -> int | None:
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    for high in range(14, 4, -1):
        if all(rank in unique for rank in range(high - 4, high + 1)):
            return high
    return None


def score_five(cards: tuple[Card, ...]) -> tuple[int, ...]:
    """Return a lexicographically comparable score for exactly five cards."""
    ranks = sorted((card[0] for card in cards), reverse=True)
    counts = {rank: ranks.count(rank) for rank in set(ranks)}
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({card[1] for card in cards}) == 1
    straight = _straight_high(ranks)

    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        quad = groups[0][1]
        return (7, quad, next(rank for rank in ranks if rank != quad))
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        triple = groups[0][1]
        kickers = [rank for rank in ranks if rank != triple]
        return (3, triple, *kickers)
    if groups[0][0] == 2 and groups[1][0] == 2:
        high_pair, low_pair = sorted((groups[0][1], groups[1][1]), reverse=True)
        kicker = next(rank for rank in ranks if rank not in (high_pair, low_pair))
        return (2, high_pair, low_pair, kicker)
    if groups[0][0] == 2:
        pair = groups[0][1]
        return (1, pair, *(rank for rank in ranks if rank != pair))
    return (0, *ranks)


def best_score(cards: list[Card]) -> tuple[int, ...]:
    if len(cards) < 5:
        raise ValueError("At least five cards are required to score a hand")
    compiled = score_seven(cards)
    if compiled is not None:
        return compiled
    return max(score_five(combo) for combo in itertools.combinations(cards, 5))


HAND_NAMES = ("high card", "pair", "two pair", "three of a kind", "straight", "flush", "full house", "four of a kind", "straight flush")


@dataclass
class SessionStats:
    """Match-wide, viewer-safe statistics retained until a new match begins."""

    hands_completed: int = 0
    hand_wins: list[int] = field(default_factory=lambda: [0, 0])
    match_wins: list[int] = field(default_factory=lambda: [0, 0])
    total_buy_in: list[int] = field(default_factory=lambda: [0, 0])
    split_pots: int = 0
    showdown_hands: int = 0
    showdown_wins: list[int] = field(default_factory=lambda: [0, 0])
    fold_wins: list[int] = field(default_factory=lambda: [0, 0])
    total_pot: int = 0
    biggest_pot: int = 0
    folds: list[int] = field(default_factory=lambda: [0, 0])
    calls: list[int] = field(default_factory=lambda: [0, 0])
    raises: list[int] = field(default_factory=lambda: [0, 0])
    vpip_hands: list[int] = field(default_factory=lambda: [0, 0])
    pfr_hands: list[int] = field(default_factory=lambda: [0, 0])

    def snapshot(self) -> dict:
        def player_snapshot(player: int) -> dict:
            hands = self.hands_completed
            voluntary_actions = self.calls[player] + self.raises[player]
            return {
                "hand_wins": self.hand_wins[player],
                "match_wins": self.match_wins[player],
                "total_buy_in": self.total_buy_in[player],
                "showdown_wins": self.showdown_wins[player],
                "fold_wins": self.fold_wins[player],
                "folds": self.folds[player],
                "calls": self.calls[player],
                "raises": self.raises[player],
                "vpip": round(self.vpip_hands[player] / hands * 100) if hands else 0,
                "pfr": round(self.pfr_hands[player] / hands * 100) if hands else 0,
                "aggression": round(self.raises[player] / voluntary_actions * 100) if voluntary_actions else None,
            }

        return {
            "hands_completed": self.hands_completed,
            "split_pots": self.split_pots,
            "showdown_hands": self.showdown_hands,
            "total_pot": self.total_pot,
            "biggest_pot": self.biggest_pot,
            "players": [player_snapshot(0), player_snapshot(1)],
        }


@dataclass
class HeadsUpHoldem:
    """A single hand-at-a-time heads-up Hold'em match state."""

    initial_stack: int = 2_000  # 100 bb at 10/20 — standard heads-up depth
    small_blind: int = 10
    big_blind: int = 20
    rng: random.Random = field(default_factory=random.Random)
    stacks: list[int] = field(default_factory=lambda: [2_000, 2_000])
    hand_number: int = 0
    deck: list[Card] = field(default_factory=list)
    hole_cards: list[list[Card]] = field(default_factory=lambda: [[], []])
    community: list[Card] = field(default_factory=list)
    pot: int = 0
    street: int = 0
    current_player: int | None = None
    button: int = 0
    button_offset: int = 0
    round_bets: list[int] = field(default_factory=lambda: [0, 0])
    contributions: list[int] = field(default_factory=lambda: [0, 0])
    last_raise: int = 20
    raise_open: list[bool] = field(default_factory=lambda: [True, True])
    acted: list[bool] = field(default_factory=lambda: [False, False])
    public_actions: list[dict[str, int | str]] = field(default_factory=list)
    street_action_counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    history: list[str] = field(default_factory=list)
    result: str | None = None
    winner: int | None = None
    showdown_scores: list[tuple[int, ...]] = field(default_factory=list)
    last_pot: int = 0
    session_stats: SessionStats = field(default_factory=SessionStats)

    def __post_init__(self) -> None:
        self.new_match()

    @property
    def hand_complete(self) -> bool:
        return self.current_player is None

    @property
    def active_street(self) -> str:
        return STREETS[self.street]

    def new_match(self) -> None:
        self.stacks = [self.initial_stack, self.initial_stack]
        self.hand_number = 0
        self.session_stats = SessionStats(
            total_buy_in=[self.initial_stack, self.initial_stack]
        )
        self.new_hand()

    def new_hand(self) -> None:
        self.history = []
        for player, stack in enumerate(self.stacks):
            if stack != 0:
                continue
            self.stacks[player] = self.initial_stack
            self.session_stats.total_buy_in[player] += self.initial_stack
            label = "Hero" if player == 0 else "Agent"
            self.history.append(
                f"{label} auto-reloads {self.initial_stack:,} chips after busting."
            )

        self.hand_number += 1
        self.button = (self.hand_number - 1 + self.button_offset) % 2
        self.deck = new_deck()
        self.rng.shuffle(self.deck)
        self.hole_cards = [[self.deck.pop(), self.deck.pop()], [self.deck.pop(), self.deck.pop()]]
        self.community = []
        self.pot = 0
        self.street = 0
        self.round_bets = [0, 0]
        self.contributions = [0, 0]
        self.last_raise = self.big_blind
        self.raise_open = [True, True]
        self.acted = [False, False]
        self.public_actions = []
        self.street_action_counts = [0, 0, 0, 0]
        self.result = None
        self.winner = None
        self.showdown_scores = []
        self.last_pot = 0

        small_blind_player = self.button
        big_blind_player = 1 - self.button
        self._put_chips(small_blind_player, min(self.small_blind, self.stacks[small_blind_player]))
        self._put_chips(big_blind_player, min(self.big_blind, self.stacks[big_blind_player]))
        self._record_public_action(small_blind_player, "blind", self.round_bets[small_blind_player])
        self._record_public_action(big_blind_player, "blind", self.round_bets[big_blind_player])
        self.history.extend(
            [
                f"Hand {self.hand_number}: Player {small_blind_player + 1} posts small blind {self.round_bets[small_blind_player]}.",
                f"Player {big_blind_player + 1} posts big blind {self.round_bets[big_blind_player]}.",
            ]
        )
        self.current_player = self.button

    def reload_cash(self, players: list[int], amount: int) -> None:
        """Add chips immediately without changing the table configuration."""
        if amount <= 0:
            raise ValueError("Reload amount must be greater than zero.")
        if not players or any(player not in (0, 1) for player in players):
            raise ValueError("Choose Hero, Agent, or both players.")

        unique_players = sorted(set(players))
        for player in unique_players:
            self.stacks[player] += amount
            self.session_stats.total_buy_in[player] += amount
        labels = " and ".join("Hero" if player == 0 else "Agent" for player in unique_players)
        verb = "reload" if len(unique_players) > 1 else "reloads"
        self.history.append(f"{labels} {verb} {amount:,} chips.")

    def _put_chips(self, player: int, amount: int) -> None:
        amount = max(0, min(amount, self.stacks[player]))
        self.stacks[player] -= amount
        self.round_bets[player] += amount
        self.contributions[player] += amount
        self.pot += amount

    def _public_action_context(self, player: int) -> dict[str, int]:
        """Public pre-action state used for learned Bayesian range updates."""
        return {
            "pot_before": self.pot,
            "to_call_before": self.to_call(player),
            "stack_before": self.stacks[player],
            "effective_stack_before": min(self.stacks),
            "current_bet_before": max(self.round_bets),
            "player_bet_before": self.round_bets[player],
            "button_before": int(self.button == player),
            "raise_open_before": int(self.raise_open[player]),
            "action_count_before": len(self.public_actions),
        }

    @staticmethod
    def _abstract_action_index(action: str, amount: int, context: dict[str, int]) -> int:
        if action == "fold":
            return 0
        if action in {"check", "call"}:
            return 1
        if action != "raise":
            return -1
        maximum = context["player_bet_before"] + context["stack_before"]
        if amount >= maximum:
            return 3
        return 2

    def _record_public_action(self, player: int, action: str, amount: int = 0, context: dict[str, int] | None = None) -> None:
        """Keep machine-readable public betting history without exposing hole cards."""
        public_context = context or {}
        self.public_actions.append({"player": player, "action": action, "amount": amount, "street": self.street, "action_index": self._abstract_action_index(action, amount, public_context), **public_context})
        self.street_action_counts[self.street] += 1

    def to_call(self, player: int) -> int:
        return max(self.round_bets) - self.round_bets[player]

    def legal_actions(self, player: int) -> dict[str, int | bool]:
        if self.hand_complete or player != self.current_player:
            return {}
        call_amount = self.to_call(player)
        maximum = self.round_bets[player] + self.stacks[player]
        minimum = max(self.round_bets) + self.last_raise
        can_raise = self.raise_open[player] and self.stacks[player] > call_amount and maximum > max(self.round_bets)
        can_all_in = self.stacks[player] > 0 and (self.stacks[player] <= call_amount or can_raise)
        return {
            "fold": call_amount > 0,
            "check": call_amount == 0,
            "call": call_amount > 0,
            "raise": can_raise,
            "all_in": can_all_in,
            "to_call": call_amount,
            "current_bet": max(self.round_bets),
            "player_bet": self.round_bets[player],
            "raise_min": min(minimum, maximum),
            "raise_max": maximum,
        }

    def act(self, player: int, action: Action, amount: int | None = None) -> None:
        if self.hand_complete:
            raise InvalidAction("This hand has already finished. Deal the next hand.")
        if player != self.current_player:
            raise InvalidAction("It is not this player's turn.")

        call_amount = self.to_call(player)
        action_context = self._public_action_context(player)
        label = f"Player {player + 1}"
        if action == "all_in":
            if self.stacks[player] == 0:
                raise InvalidAction("You have no chips left to wager.")
            if self.stacks[player] <= call_amount:
                action = "call"
            else:
                if not self.raise_open[player]:
                    raise InvalidAction("Betting is not reopened after the short all-in raise.")
                action = "raise"
                amount = self.round_bets[player] + self.stacks[player]
        if action == "fold":
            if not call_amount:
                raise InvalidAction("There is no bet to fold to; check instead.")
            self.history.append(f"{label} folds.")
            self._record_public_action(player, "fold", context=action_context)
            self._finish_fold(1 - player)
            return
        if action == "check":
            if call_amount:
                raise InvalidAction("You cannot check while facing a bet.")
            self.history.append(f"{label} checks.")
            self._record_public_action(player, "check", context=action_context)
            self.acted[player] = True
            self.raise_open[player] = False
        elif action == "call":
            if not call_amount:
                raise InvalidAction("There is no bet to call; check instead.")
            paid = min(call_amount, self.stacks[player])
            self._put_chips(player, paid)
            suffix = " all-in" if paid < call_amount or self.stacks[player] == 0 else ""
            self.history.append(f"{label} calls {paid}{suffix}.")
            self._record_public_action(player, "call", paid, action_context)
            self.acted[player] = True
            self.raise_open[player] = False
            if self.stacks[player] == 0:
                self._runout_and_showdown()
                return
        elif action == "raise":
            if amount is None:
                raise InvalidAction("A raise-to amount is required.")
            maximum = self.round_bets[player] + self.stacks[player]
            current_high = max(self.round_bets)
            minimum = current_high + self.last_raise
            if not self.raise_open[player]:
                raise InvalidAction("Betting is not reopened after the short all-in raise.")
            if amount <= current_high or amount > maximum:
                raise InvalidAction("Raise amount must be above the current bet and within your stack.")
            is_all_in = amount == maximum
            if amount < minimum and not is_all_in:
                raise InvalidAction(f"Minimum raise-to is {minimum}.")
            paid = amount - self.round_bets[player]
            self._put_chips(player, paid)
            raise_size = amount - current_high
            if raise_size >= self.last_raise:
                self.last_raise = raise_size
                self.raise_open = [True, True]
                self.raise_open[player] = False
            elif current_high == 0:
                # An opening short all-in still gives the opponent a chance to raise.
                self.raise_open[1 - player] = True
                self.raise_open[player] = False
            else:
                self.raise_open[player] = False
            self.acted = [False, False]
            self.acted[player] = True
            suffix = " all-in" if self.stacks[player] == 0 else ""
            self.history.append(f"{label} raises to {amount}{suffix}.")
            self._record_public_action(player, "raise", amount, action_context)
        else:
            raise InvalidAction("Unknown action.")

        self._advance_after_action(player)

    def _advance_after_action(self, player: int) -> None:
        opponent = 1 - player
        if self.stacks[0] == 0 or self.stacks[1] == 0:
            if self.round_bets[0] == self.round_bets[1] or self.stacks[opponent] == 0:
                self._runout_and_showdown()
                return
        if self.acted[0] and self.acted[1] and self.round_bets[0] == self.round_bets[1]:
            if self.street == 3:
                self._showdown()
            else:
                self._next_street()
            return
        self.current_player = opponent

    def _next_street(self) -> None:
        self.street += 1
        cards_to_deal = 3 if self.street == 1 else 1
        self.community.extend(self.deck.pop() for _ in range(cards_to_deal))
        self.round_bets = [0, 0]
        self.last_raise = self.big_blind
        self.raise_open = [True, True]
        self.acted = [False, False]
        self.current_player = 1 - self.button
        self.history.append(f"{self.active_street.title()}: {' '.join(card_text(card) for card in self.community)}")

    def _runout_and_showdown(self) -> None:
        while self.street < 3:
            self._next_street()
        self._showdown()

    def _finish_fold(self, winner: int) -> None:
        self.last_pot = self.pot
        self.stacks[winner] += self.pot
        self.result = f"Player {winner + 1} wins {self.pot} chips after a fold."
        self.winner = winner
        self._record_session_hand(winner, ended_at_showdown=False)
        self.pot = 0
        self.current_player = None

    def _showdown(self) -> None:
        scores = [best_score(self.hole_cards[player] + self.community) for player in range(2)]
        self.showdown_scores = scores
        # In heads-up play, any unmatched all-in excess is returned before awarding the contested pot.
        matched = min(self.contributions)
        excess = [self.contributions[player] - matched for player in range(2)]
        for player, returned in enumerate(excess):
            if returned:
                self.stacks[player] += returned
                self.pot -= returned
                self.history.append(f"Player {player + 1} receives {returned} unmatched chips back.")
        contested = self.pot
        self.last_pot = contested
        if scores[0] > scores[1]:
            self.stacks[0] += contested
            self.winner = 0
            self.result = f"Player 1 wins {contested} chips with {HAND_NAMES[scores[0][0]]}."
        elif scores[1] > scores[0]:
            self.stacks[1] += contested
            self.winner = 1
            self.result = f"Player 2 wins {contested} chips with {HAND_NAMES[scores[1][0]]}."
        else:
            first_share = contested // 2 + contested % 2
            self.stacks[0] += first_share
            self.stacks[1] += contested // 2
            self.winner = None
            self.result = f"Showdown tie: the {contested}-chip pot is split."
        self.history.append(self.result)
        self._record_session_hand(self.winner, ended_at_showdown=True)
        self.pot = 0
        self.current_player = None

    def _record_session_hand(self, winner: int | None, ended_at_showdown: bool) -> None:
        stats = self.session_stats
        stats.hands_completed += 1
        stats.total_pot += self.last_pot
        stats.biggest_pot = max(stats.biggest_pot, self.last_pot)

        if ended_at_showdown:
            stats.showdown_hands += 1

        if winner is None:
            stats.split_pots += 1
        else:
            stats.hand_wins[winner] += 1
            if ended_at_showdown:
                stats.showdown_wins[winner] += 1
            else:
                stats.fold_wins[winner] += 1
            if self.stacks[1 - winner] == 0:
                stats.match_wins[winner] += 1

        for player in range(2):
            player_actions = [action for action in self.public_actions if action["player"] == player]
            stats.folds[player] += sum(action["action"] == "fold" for action in player_actions)
            stats.calls[player] += sum(action["action"] == "call" for action in player_actions)
            stats.raises[player] += sum(action["action"] == "raise" for action in player_actions)
            preflop_actions = [action for action in player_actions if action["street"] == 0]
            stats.vpip_hands[player] += int(any(action["action"] in {"call", "raise"} for action in preflop_actions))
            stats.pfr_hands[player] += int(any(action["action"] == "raise" for action in preflop_actions))

    def snapshot(self, viewer: int = 0) -> dict:
        legal = self.legal_actions(viewer)
        complete = self.hand_complete
        hero_hand_strength = HAND_NAMES[best_score(self.hole_cards[viewer] + self.community)[0]].title() if self.community else None
        return {
            "hand_number": self.hand_number,
            "street": self.active_street,
            "button": self.button,
            "current_player": self.current_player,
            "stacks": self.stacks,
            "pot": self.pot,
            "last_pot": self.last_pot,
            "round_bets": self.round_bets,
            "to_call": self.to_call(viewer) if not complete else 0,
            "hero_cards": [card_text(card) for card in self.hole_cards[viewer]],
            "hero_hand_strength": hero_hand_strength,
            "opponent_cards": [card_text(card) for card in self.hole_cards[1 - viewer]] if complete else [],
            "community": [card_text(card) for card in self.community],
            "history": self.history[-20:],
            "legal_actions": legal,
            "complete": complete,
            "result": self.result,
            "winner": self.winner,
            "session_stats": self.session_stats.snapshot(),
            "settings": {
                "initial_stack": self.initial_stack,
                "small_blind": self.small_blind,
                "big_blind": self.big_blind,
            },
        }
