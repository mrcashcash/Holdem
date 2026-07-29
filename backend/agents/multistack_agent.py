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

import random
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

    def selected_depth(self, game: HeadsUpHoldem, player: int) -> float:
        """Return the depth serving THIS hand: the locked routing choice when one
        exists (so status/logs match the agent actually acting — a fresh
        recomputation drifts once streets consume round_bets), else the
        nearest depth for the current effective stack."""
        if self._active is not None and self._active_hand == (id(game), game.hand_number):
            return float(self._active.tree.config.stack_bb)
        target = self._effective_stack_bb(game, player)
        return min(self.depths, key=lambda depth: abs(depth - target))

    def _route(self, game: HeadsUpHoldem, player: int) -> GpuBlueprintAgent:
        # Lock the choice for the whole hand so select()/execute() agree even
        # as stacks shrink through it. The key includes id(game) so a new
        # match (hand_number reset to 1) or a different game object re-routes.
        hand_key = (id(game), game.hand_number)
        if self._active is None or self._active_hand != hand_key:
            nearest = self.selected_depth(game, player)
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

    # Continual exact-card resolving. Without these pass-throughs, setting
    # `router.continual_search = True` silently creates a dead attribute on the
    # router while the sub-agents that actually decide hands stay off — and
    # `last_continual_search` lives on the sub-agent, so diagnostics read empty
    # too. That combination reads as "the resolver is on and never fires".
    @property
    def continual_search(self) -> bool:
        return any(agent.continual_search for agent in self.agents.values())

    @continual_search.setter
    def continual_search(self, value: bool) -> None:
        for agent in self.agents.values():
            agent.continual_search = value

    @property
    def continual_streets(self) -> tuple[int, ...]:
        return next(iter(self.agents.values())).continual_streets

    @continual_streets.setter
    def continual_streets(self, value) -> None:
        for agent in self.agents.values():
            agent.continual_streets = tuple(value)

    # Action-sampling RNG. `backend.eval.duel.head_to_head`'s common-random-numbers
    # coupling reseeds `agent._rng` before every hand so two arms draw identical
    # variates at identical infosets and diverge ONLY where their policies differ.
    # The router had no `_rng`, so `hasattr(target, "_rng")` was False and CRN
    # silently did nothing for the SERVING agent — the one every real comparison
    # uses. The measured "off-vs-off null reads exactly +0.00 bb/100" came from a
    # single-depth GpuBlueprintAgent, so it did not cover this.
    @property
    def _rng(self) -> random.Random:
        return next(iter(self.agents.values()))._rng

    @_rng.setter
    def _rng(self, value: random.Random) -> None:
        # Each depth needs its OWN generator: sharing one object would let a hand
        # routed to 100bb advance the 200bb stream, which reintroduces exactly the
        # desync CRN exists to remove.
        state = value.getstate()
        for agent in self.agents.values():
            generator = random.Random()
            generator.setstate(state)
            agent._rng = generator

    @property
    def all_in_geometry_guard(self) -> bool:
        return all(agent.all_in_geometry_guard for agent in self.agents.values())

    @all_in_geometry_guard.setter
    def all_in_geometry_guard(self, value: bool) -> None:
        for agent in self.agents.values():
            agent.all_in_geometry_guard = bool(value)

    @property
    def all_in_max_pot_multiple(self) -> float:
        return next(iter(self.agents.values())).all_in_max_pot_multiple

    @all_in_max_pot_multiple.setter
    def all_in_max_pot_multiple(self, value: float) -> None:
        for agent in self.agents.values():
            agent.all_in_max_pot_multiple = float(value)

    @property
    def all_in_geometry_tolerance(self) -> float:
        return next(iter(self.agents.values())).all_in_geometry_tolerance

    @all_in_geometry_tolerance.setter
    def all_in_geometry_tolerance(self, value: float) -> None:
        for agent in self.agents.values():
            agent.all_in_geometry_tolerance = float(value)

    @property
    def last_all_in_rescale(self) -> dict | None:
        """The most recent ALL-IN resize from whichever depth just acted."""
        active = getattr(self, "_active", None)
        if active is not None and getattr(active, "last_all_in_rescale", None):
            return active.last_all_in_rescale
        for agent in self.agents.values():
            if getattr(agent, "last_all_in_rescale", None):
                return agent.last_all_in_rescale
        return None

    @last_all_in_rescale.setter
    def last_all_in_rescale(self, value) -> None:
        for agent in self.agents.values():
            agent.last_all_in_rescale = value

    @property
    def continual_iterations(self) -> int:
        return next(iter(self.agents.values())).continual_iterations

    @continual_iterations.setter
    def continual_iterations(self, value: int) -> None:
        for agent in self.agents.values():
            agent.continual_iterations = value

    @property
    def continual_budget_ms(self) -> int:
        return next(iter(self.agents.values())).continual_budget_ms

    @continual_budget_ms.setter
    def continual_budget_ms(self, value: int) -> None:
        for agent in self.agents.values():
            agent.continual_budget_ms = value

    @property
    def last_continual_search(self) -> dict | None:
        """The most recent resolve diagnostics from whichever depth just acted."""
        active = getattr(self, "_active", None)
        if active is not None and getattr(active, "last_continual_search", None):
            return active.last_continual_search
        for agent in self.agents.values():
            if getattr(agent, "last_continual_search", None):
                return agent.last_continual_search
        return None

    @last_continual_search.setter
    def last_continual_search(self, value) -> None:
        for agent in self.agents.values():
            agent.last_continual_search = value

    @property
    def exact_river_search(self) -> bool:
        return any(agent.exact_river_search for agent in self.agents.values())

    @exact_river_search.setter
    def exact_river_search(self, value: bool) -> None:
        for agent in self.agents.values():
            agent.exact_river_search = value

    @property
    def exact_river_iterations(self) -> int:
        return next(iter(self.agents.values())).exact_river_iterations

    @exact_river_iterations.setter
    def exact_river_iterations(self, value: int) -> None:
        for agent in self.agents.values():
            agent.exact_river_iterations = value

    @property
    def exact_river_budget_ms(self) -> int:
        return next(iter(self.agents.values())).exact_river_budget_ms

    @exact_river_budget_ms.setter
    def exact_river_budget_ms(self, value: int) -> None:
        for agent in self.agents.values():
            agent.exact_river_budget_ms = value

    @property
    def iteration(self) -> int:
        # min, not max: the serving gate (GPU_SERVE_MIN_ITERATIONS) must hold
        # for EVERY depth — otherwise one trained depth smuggles an
        # under-trained sibling into production.
        return min(agent.iteration for agent in self.agents.values())

    def depth_summary(self) -> dict[float, int]:
        return {depth: self.agents[depth].iteration for depth in self.depths}
