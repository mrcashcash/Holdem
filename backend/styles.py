"""Scripted opponents: benchmark archetypes and a heuristic fallback agent.

Extracted from the retired PPO trainer (legacy/learning.py) — these are the
only pieces the new architecture still uses. The styles are intentionally
leaky, reproducible rule players for regression benchmarking; the heuristic
agent serves hands if no blueprint artifacts exist yet.
"""

from __future__ import annotations

from backend.poker import HeadsUpHoldem
from backend.rl_env import execute_action

BENCHMARK_STYLES = ("tight_aggressive", "loose_aggressive", "calling_station", "trapper", "pressure")
AUDIT_STYLES = ("nit", "maniac", "river_hunter")


def style_action(game: HeadsUpHoldem, player: int, style: str) -> int:
    """Fixed benchmark archetypes with intentionally different, reproducible leaks."""
    legal = game.legal_actions(player)
    ranks = sorted((card[0] for card in game.hole_cards[player]), reverse=True)
    pair = ranks[0] == ranks[1]
    strong = pair or ranks[0] >= 12
    facing = game.to_call(player) > 0
    if style == "calling_station":
        if legal.get("check") or legal.get("call"):
            return 1
        return 2
    if style == "loose_aggressive":
        if legal.get("raise") and (strong or not facing):
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "trapper":
        if strong and legal.get("check"):
            return 1
        if strong and legal.get("raise"):
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "pressure":
        if legal.get("raise") and (not facing or strong):
            return 2
        if facing and not strong:
            return 0
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "nit":
        if facing and not (pair or ranks[0] >= 13):
            return 0
        if legal.get("raise") and pair and ranks[0] >= 11:
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "maniac":
        if legal.get("raise"):
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "river_hunter":
        if game.street == 3 and legal.get("raise") and (strong or not facing):
            return 2
        if facing and game.street == 3 and not strong:
            return 0
        return 1 if legal.get("check") or legal.get("call") else 0
    # Tight-aggressive default.
    if facing and not strong:
        return 0
    if strong and legal.get("raise"):
        return 2
    return 1 if legal.get("check") or legal.get("call") else 0


def heuristic_action(game: HeadsUpHoldem, player: int) -> int:
    """A strength-aware baseline: varied, exploitable, always legal."""
    legal = game.legal_actions(player)
    ranks = sorted((card[0] for card in game.hole_cards[player]), reverse=True)
    pair = ranks[0] == ranks[1]
    suited = game.hole_cards[player][0][1] == game.hole_cards[player][1][1]
    strength = 4 if pair and ranks[0] >= 10 else 3 if pair or ranks[0] >= 12 else 2 if ranks[0] >= 10 or suited else 1
    if legal.get("raise") and strength >= 4:
        return 2
    if legal.get("raise") and strength == 3 and game.to_call(player) == 0:
        return 2
    if legal.get("check"):
        return 1
    if legal.get("call"):
        pressure = game.to_call(player) / max(1, game.pot + game.to_call(player))
        return 1 if strength >= 2 or pressure < 0.16 else 0
    return 0


class HeuristicAgent:
    """Minimal serving agent used only when no blueprint artifacts exist."""

    ready = True

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        return heuristic_action(game, player)

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        execute_action(game, player, choice, None)

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        return None

    def parameter_count(self) -> int:
        return 0
