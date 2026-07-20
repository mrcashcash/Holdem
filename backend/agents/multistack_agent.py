"""Depth-routing serving agent: one table, many stack-depth blueprints.

Each stack depth is a different game (stack-to-pot ratios change which lines
are correct — the 100bb-at-200bb Slumbot swing of ~107 bb/100 proved it), so
the serving layer keeps a library of per-depth blueprints and, at the start
of every hand, routes to the one nearest the current effective stack. The
player never switches anything; residual mismatch (e.g. an odd 137bb stack)
is absorbed by the re-solver, which always solves at the exact stack.

Discovers the canonical 100bb blueprint (backend/data/gpu_blueprint/) plus
any backend/data/gpu_blueprint_<N>bb/ depth directories.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.poker import HeadsUpHoldem


class MultiStackBlueprintAgent:
    """Routes each hand to the nearest-depth blueprint by effective stack."""

    def __init__(self, agents: dict[float, GpuBlueprintAgent]) -> None:
        if not agents:
            raise ValueError("at least one depth blueprint is required")
        self.agents = agents  # {stack_bb: agent}
        self.depths = sorted(agents)
        self.ready = True
        self._active: GpuBlueprintAgent | None = None
        self._active_hand: tuple[int, int] | None = None

    @classmethod
    def try_load(cls, data_dir: Path | None = None) -> "MultiStackBlueprintAgent | None":
        from backend.solver.gpu import train as gpu_train

        base = data_dir or gpu_train.DATA_DIR.parent
        agents: dict[float, GpuBlueprintAgent] = {}

        # Canonical 100bb: prefer champion.npz, else checkpoint.npz.
        champion = base / "gpu_blueprint" / "champion.npz"
        checkpoint = base / "gpu_blueprint" / "checkpoint.npz"
        primary = champion if champion.exists() else checkpoint
        if primary.exists():
            agent = GpuBlueprintAgent.try_load(primary)
            if agent is not None:
                agents[float(agent.tree.config.stack_bb)] = agent

        # Additional depths: gpu_blueprint_<N>bb/ (champion or checkpoint).
        for directory in sorted(base.glob("gpu_blueprint_*bb")):
            match = re.search(r"gpu_blueprint_(\d+)bb", directory.name)
            if not match:
                continue
            depth_champ = directory / "champion.npz"
            depth_ckpt = directory / "checkpoint.npz"
            source = depth_champ if depth_champ.exists() else depth_ckpt
            if not source.exists():
                continue
            agent = GpuBlueprintAgent.try_load(source)
            if agent is not None:
                agents[float(agent.tree.config.stack_bb)] = agent

        if not agents:
            return None
        return cls(agents)

    # -- routing ---------------------------------------------------------------

    @staticmethod
    def _effective_stack_bb(game: HeadsUpHoldem, player: int) -> float:
        # Heads-up effective stack = smaller remaining stack, measured before
        # this street's wagers, i.e. current stack + already-committed chips.
        effective = min(
            game.stacks[seat] + game.round_bets[seat] for seat in (0, 1)
        )
        return effective / game.big_blind

    def _route(self, game: HeadsUpHoldem, player: int) -> GpuBlueprintAgent:
        # Lock the choice for the whole hand so select()/execute() agree even
        # as stacks shrink through it. The key includes id(game) so a new
        # match (hand_number reset to 1) or a different game object re-routes.
        hand_key = (id(game), game.hand_number)
        if self._active is None or self._active_hand != hand_key:
            target = self._effective_stack_bb(game, player)
            nearest = min(self.depths, key=lambda depth: abs(depth - target))
            self._active = self.agents[nearest]
            self._active_hand = hand_key
        return self._active

    # -- serving contract ------------------------------------------------------

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        return self._route(game, player).select(game, player)

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        self._route(game, player).execute(game, player, choice)

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        if self._active is not None:
            self._active.observe_completed_hand(game, player)

    def parameter_count(self) -> int:
        return sum(agent.parameter_count() for agent in self.agents.values())

    # -- status surface (main.py reads these) ---------------------------------

    @property
    def subgame_search(self) -> bool:
        return any(agent.subgame_search for agent in self.agents.values())

    @subgame_search.setter
    def subgame_search(self, value: bool) -> None:
        for agent in self.agents.values():
            agent.subgame_search = value

    @property
    def subgame_iterations(self) -> int:
        return next(iter(self.agents.values())).subgame_iterations

    @subgame_iterations.setter
    def subgame_iterations(self, value: int) -> None:
        for agent in self.agents.values():
            agent.subgame_iterations = value

    @property
    def iteration(self) -> int:
        return max(agent.iteration for agent in self.agents.values())

    def depth_summary(self) -> dict:
        return {f"{int(d)}bb": self.agents[d].iteration for d in self.depths}
