"""External-sampling Monte Carlo CFR with linear weighting (Linear MCCFR).

The traverser explores every legal action at its own decision nodes; chance
and the opponent are sampled. Regret and average-strategy increments are
weighted by the iteration index (Linear CFR, Brown & Sandholm AAAI 2019),
which converges 2-10x faster than vanilla CFR and is far more robust to the
large payoff ranges of no-limit games. Optional regret-based pruning skips
actions whose cumulative regret is hopelessly negative (Pluribus).

The average strategy — not the current one — converges to Nash; ``StrategyTable``
therefore accumulates strategy sums and exposes ``average_strategy``.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np

from backend.solver.game import Game, State


class StrategyTable:
    """Per-infoset cumulative regrets and average-strategy weights."""

    __slots__ = ("num_actions", "regrets", "strategy_sums", "_baseline_regrets", "_baseline_sums", "_touched")

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self.regrets: dict[Hashable, np.ndarray] = {}
        self.strategy_sums: dict[Hashable, np.ndarray] = {}
        self._baseline_regrets: dict[Hashable, np.ndarray] = {}
        self._baseline_sums: dict[Hashable, np.ndarray] = {}
        self._touched: set[Hashable] | None = None

    def _regret_row(self, key: Hashable) -> np.ndarray:
        self._touch(key)
        row = self.regrets.get(key)
        if row is None:
            row = np.zeros(self.num_actions, dtype=np.float64)
            self.regrets[key] = row
        return row

    def _strategy_row(self, key: Hashable) -> np.ndarray:
        self._touch(key)
        row = self.strategy_sums.get(key)
        if row is None:
            row = np.zeros(self.num_actions, dtype=np.float64)
            self.strategy_sums[key] = row
        return row

    def current_strategy(self, key: Hashable, actions: Sequence[int]) -> np.ndarray:
        """Regret-matching policy over ``actions`` (indexed by position)."""
        regrets = self._regret_row(key)
        positive = np.maximum(regrets[list(actions)], 0.0)
        total = positive.sum()
        if total <= 0.0:
            return np.full(len(actions), 1.0 / len(actions))
        return positive / total

    def average_strategy(self, key: Hashable, actions: Sequence[int]) -> np.ndarray:
        sums = self.strategy_sums.get(key)
        if sums is None:
            return np.full(len(actions), 1.0 / len(actions))
        weights = sums[list(actions)]
        total = weights.sum()
        if total <= 0.0:
            return np.full(len(actions), 1.0 / len(actions))
        return weights / total

    def __len__(self) -> int:
        return len(self.regrets)

    # -- delta tracking (parallel training) ---------------------------------

    def begin_delta(self) -> None:
        """Start recording changes so they can be shipped to other processes."""
        self._baseline_regrets = {}
        self._baseline_sums = {}
        self._touched = set()

    def _touch(self, key: Hashable) -> None:
        # getattr: tables unpickled from checkpoints written before delta
        # tracking existed have no _touched slot at all.
        touched = getattr(self, "_touched", None)
        if touched is None or key in touched:
            return
        touched.add(key)
        existing_regrets = self.regrets.get(key)
        existing_sums = self.strategy_sums.get(key)
        if existing_regrets is not None:
            self._baseline_regrets[key] = existing_regrets.copy()
        if existing_sums is not None:
            self._baseline_sums[key] = existing_sums.copy()

    def collect_delta(self) -> dict[Hashable, tuple[np.ndarray, np.ndarray]]:
        """Return per-key (regret, strategy-sum) increments since begin_delta."""
        delta: dict[Hashable, tuple[np.ndarray, np.ndarray]] = {}
        zeros = np.zeros(self.num_actions, dtype=np.float64)
        for key in self._touched:
            regret_change = self.regrets.get(key, zeros) - self._baseline_regrets.get(key, zeros)
            sum_change = self.strategy_sums.get(key, zeros) - self._baseline_sums.get(key, zeros)
            if regret_change.any() or sum_change.any():
                delta[key] = (regret_change, sum_change)
        self._touched = set()
        self._baseline_regrets = {}
        self._baseline_sums = {}
        return delta

    def apply_delta(self, delta: dict[Hashable, tuple[np.ndarray, np.ndarray]]) -> None:
        """Additively merge increments produced by another process."""
        for key, (regret_change, sum_change) in delta.items():
            self._regret_row(key)
            self._strategy_row(key)
            self.regrets[key] += regret_change
            self.strategy_sums[key] += sum_change

    # Checkpoints use pickle because infoset keys are arbitrary hashable tuples.
    # Trust boundary: these files are produced and consumed only by this local
    # trainer under backend/data/; never load a table from an untrusted source.
    def save(self, path: str | Path) -> None:
        payload = {
            "num_actions": self.num_actions,
            "regrets": self.regrets,
            "strategy_sums": self.strategy_sums,
        }
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
        table.regrets = payload["regrets"]
        table.strategy_sums = payload["strategy_sums"]
        return table


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
            sums = self.table._strategy_row(key)
            for position, action in enumerate(actions):
                sums[action] += self.iteration * strategy[position]
            choice = self.rng.choices(range(len(actions)), weights=strategy)[0]
            return self._traverse(state.child(actions[choice]), traverser, prune)

        # Traverser node: explore every action (optionally pruning hopeless ones).
        regrets = self.table._regret_row(key)
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
