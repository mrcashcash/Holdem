"""External-sampling Monte Carlo CFR with linear weighting (Linear MCCFR).

The traverser explores every legal action at its own decision nodes; chance
and the opponent are sampled. Regret and average-strategy increments are
weighted by the iteration index (Linear CFR, Brown & Sandholm AAAI 2019),
which converges 2-10x faster than vanilla CFR and is far more robust to the
large payoff ranges of no-limit games. Optional regret-based pruning skips
actions whose cumulative regret is hopelessly negative (Pluribus).

The average strategy — not the current one — converges to Nash;
``StrategyTable`` therefore accumulates strategy sums and exposes
``average_strategy``.

Memory layout: one dict entry per infoset holding a single (2, num_actions)
float64 array — row 0 cumulative regrets, row 1 strategy sums. Combined with
compact (e.g. bytes) infoset keys this is ~2x smaller than the original
two-dict layout. Checkpoints in the legacy two-dict format load transparently
(see ``__setstate__`` / ``load``).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np

import random

from backend.solver.game import Game, State

REGRETS, SUMS = 0, 1


class StrategyTable:
    """Per-infoset cumulative regrets and average-strategy weights."""

    __slots__ = ("num_actions", "rows", "_baseline", "_touched")

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.rows: dict[Hashable, np.ndarray] = {}
        self._baseline: dict[Hashable, np.ndarray] = {}
        self._touched: set[Hashable] | None = None

    def _row(self, key: Hashable) -> np.ndarray:
        row = self.rows.get(key)
        if row is None:
            row = np.zeros((2, self.num_actions), dtype=np.float64)
            self.rows[key] = row
        touched = getattr(self, "_touched", None)
        if touched is not None and key not in touched:
            touched.add(key)
            self._baseline[key] = row.copy()
        return row

    def current_strategy(self, key: Hashable, actions: Sequence[int]) -> np.ndarray:
        """Regret-matching policy over ``actions`` (indexed by position)."""
        regrets = self._row(key)[REGRETS]
        positive = np.maximum(regrets[list(actions)], 0.0)
        total = positive.sum()
        if total <= 0.0:
            return np.full(len(actions), 1.0 / len(actions))
        return positive / total

    def average_strategy(self, key: Hashable, actions: Sequence[int]) -> np.ndarray:
        row = self.rows.get(key)
        if row is None:
            return np.full(len(actions), 1.0 / len(actions))
        weights = row[SUMS][list(actions)]
        total = weights.sum()
        if total <= 0.0:
            return np.full(len(actions), 1.0 / len(actions))
        return weights / total

    def __len__(self) -> int:
        return len(self.rows)

    # -- delta tracking (parallel training) ---------------------------------

    def begin_delta(self) -> None:
        """Start recording changes so they can be shipped to other processes."""
        self._baseline = {}
        self._touched = set()

    def collect_delta(self) -> dict[Hashable, np.ndarray]:
        """Return per-key (2, num_actions) increments since ``begin_delta``."""
        delta: dict[Hashable, np.ndarray] = {}
        zeros = np.zeros((2, self.num_actions), dtype=np.float64)
        for key in self._touched or ():
            change = self.rows.get(key, zeros) - self._baseline.get(key, zeros)
            if change.any():
                delta[key] = change
        self._touched = set()
        self._baseline = {}
        return delta

    def apply_delta(self, delta: dict[Hashable, np.ndarray]) -> None:
        """Additively merge increments produced by another process."""
        for key, change in delta.items():
            self._row(key)
            self.rows[key] += change

    # -- persistence ----------------------------------------------------------

    # Checkpoints use pickle because infoset keys are arbitrary hashables.
    # Trust boundary: these files are produced and consumed only by this local
    # trainer under backend/data/; never load a table from an untrusted source.
    def save(self, path: str | Path) -> None:
        payload = {"format": 2, "num_actions": self.num_actions, "rows": self.rows}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "StrategyTable":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        table = cls(payload["num_actions"])
        if payload.get("format", 1) >= 2:
            table.rows = payload["rows"]
        else:
            table._adopt_legacy(payload["regrets"], payload["strategy_sums"])
        return table

    def _adopt_legacy(
        self,
        regrets: dict[Hashable, np.ndarray],
        strategy_sums: dict[Hashable, np.ndarray],
    ) -> None:
        """Convert the legacy two-dict layout into packed rows."""
        for key in regrets.keys() | strategy_sums.keys():
            row = np.zeros((2, self.num_actions), dtype=np.float64)
            regret_row = regrets.get(key)
            sum_row = strategy_sums.get(key)
            if regret_row is not None:
                row[REGRETS] = regret_row
            if sum_row is not None:
                row[SUMS] = sum_row
            self.rows[key] = row

    def __getstate__(self) -> dict:
        return {"num_actions": self.num_actions, "rows": self.rows}

    def __setstate__(self, state: dict) -> None:
        # Whole-object pickles: slots classes pickle as (None, slot_dict).
        if isinstance(state, tuple):
            state = state[1] or {}
        self.num_actions = state["num_actions"]
        self._baseline = {}
        self._touched = None
        if "rows" in state:
            self.rows = state["rows"]
        else:  # legacy layout pickled before the packed-row format
            self.rows = {}
            self._adopt_legacy(state.get("regrets", {}), state.get("strategy_sums", {}))

    def remap_keys(self, mapper) -> int:
        """Re-encode every infoset key with ``mapper`` (checkpoint migration)."""
        self.rows = {mapper(key): row for key, row in self.rows.items()}
        return len(self.rows)


class LinearMCCFR:
    """External-sampling Linear MCCFR over a two-player zero-sum ``Game``."""

    def __init__(
        self,
        game: Game,
        seed: int = 0,
        pruning_threshold: float | None = None,
        pruning_probability: float = 0.95,
        pruning_warmup_iterations: int = 0,
    ) -> None:
        self.game = game
        self.table = StrategyTable(game.num_actions())
        self.rng = random.Random(seed)
        self.iteration = 0
        self.pruning_threshold = pruning_threshold
        self.pruning_probability = pruning_probability
        self.pruning_warmup_iterations = pruning_warmup_iterations

    def run(self, iterations: int) -> None:
        """Run ``iterations`` full iterations (one traversal per player each)."""
        for _ in range(iterations):
            self.iteration += 1
            prune = (
                self.pruning_threshold is not None
                and self.iteration > self.pruning_warmup_iterations
                and self.rng.random() < self.pruning_probability
            )
            for traverser in (0, 1):
                self._traverse(self.game.initial_state(), traverser, prune)

    def _traverse(self, state: State, traverser: int, prune: bool) -> float:
        if state.is_terminal():
            return state.utility(traverser)
        if state.is_chance():
            return self._traverse(state.sample_chance(self.rng), traverser, prune)

        actions = state.legal_actions()
        key = state.infoset_key()
        strategy = self.table.current_strategy(key, actions)

        if state.current_player() != traverser:
            # Opponent node: sample the action, accumulate the linear-weighted
            # average-strategy contribution for the opponent.
            sums = self.table._row(key)[SUMS]
            for position, action in enumerate(actions):
                sums[action] += self.iteration * strategy[position]
            choice = self.rng.choices(range(len(actions)), weights=strategy)[0]
            return self._traverse(state.child(actions[choice]), traverser, prune)

        # Traverser node: explore every action (optionally pruning hopeless ones).
        regrets = self.table._row(key)[REGRETS]
        action_values = np.zeros(len(actions))
        explored = np.ones(len(actions), dtype=bool)
        # Under linear weighting, sampling noise in cumulative regret grows
        # ~t^1.5 (std of a sum with weights 1..t) while a truly dominated
        # action's regret drifts ~-t^2. Scaling the cutoff by t^1.5 keeps it
        # above the noise floor so mixed-strategy actions are never pruned,
        # while dominated actions still fall through.
        cutoff = (self.pruning_threshold or 0.0) * self.iteration**1.5
        for position, action in enumerate(actions):
            if prune and len(actions) > 1 and regrets[action] < cutoff:
                explored[position] = False
                continue
            action_values[position] = self._traverse(state.child(action), traverser, prune)
        node_value = float(np.dot(strategy[explored], action_values[explored])) if explored.any() else 0.0
        for position, action in enumerate(actions):
            if explored[position]:
                regrets[action] += self.iteration * (action_values[position] - node_value)
        return node_value

    def average_policy(self, key: Hashable, actions: Sequence[int]) -> np.ndarray:
        return self.table.average_strategy(key, actions)
