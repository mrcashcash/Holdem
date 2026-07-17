"""Equilibrium-finding core: Linear MCCFR blueprint solver and validation games.

This package replaces the PPO self-play trainer as the strategy learner
(docs/REDESIGN_PLAN.md). The solver is generic over the ``Game``/``State``
protocol in ``game.py`` so it can be validated against Kuhn and Leduc ground
truth before being pointed at abstracted heads-up no-limit hold'em.
"""

from backend.solver.best_response import best_response_value, exploitability
from backend.solver.game import Game, State
from backend.solver.mccfr import LinearMCCFR, StrategyTable

__all__ = [
    "Game",
    "State",
    "LinearMCCFR",
    "StrategyTable",
    "best_response_value",
    "exploitability",
]
