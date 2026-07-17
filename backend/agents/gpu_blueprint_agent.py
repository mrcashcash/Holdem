"""Serve the dense GPU blueprint through the live game's agent contract.

Mirrors the real hand onto the flattened betting tree: public actions are
translated onto the coarse GPU menu (pseudo-harmonic for raises), and at our
decisions the average strategy at (node, street bucket) is sampled. Street
buckets are computed exactly as in training: blocker-corrected sort equity on
the current board (cached per board+street).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from backend.abstraction.actions import pseudo_harmonic_weights
from backend.abstraction.cards import preflop_class
from backend.poker import HeadsUpHoldem
from backend.rl_env import execute_action
from backend.solver.gpu.deals import DealSampler, NUM_COMBOS, combos, equity_from_scores, score_all_combos
from backend.solver.gpu.tree import ALL_IN, CHECK_CALL, DECISION, FOLD, STREET_END, BettingTree, GpuActionConfig
from backend.vectorized_engine import card_id

NEURAL_FOLD, NEURAL_CHECK_CALL, NEURAL_RAISE, NEURAL_ALL_IN = 0, 1, 2, 3

_COMBO_INDEX = {(int(a), int(b)): index for index, (a, b) in enumerate(combos())}


class GpuBlueprintAgent:
    """Drop-in serving agent backed by the dense GPU blueprint tables."""

    def __init__(
        self, tree: BettingTree, strategy: np.ndarray, sampler: DealSampler, iteration: int = 0
    ) -> None:
        self.tree = tree
        self.strategy = strategy  # [nodes, MAX_BUCKETS, actions], normalized
        self.sampler = sampler
        self.iteration = iteration
        self.ready = True
        self._raise_fraction: float | None = None
        self._rng = random.Random(97)
        self._equity_cache: dict[tuple, float] = {}

    @classmethod
    def try_load(cls, checkpoint_path: Path | None = None) -> "GpuBlueprintAgent | None":
        from backend.solver.gpu import train as gpu_train

        path = checkpoint_path or gpu_train.CHECKPOINT_PATH
        if not path.exists():
            return None
        payload = np.load(path, allow_pickle=False)
        config = GpuActionConfig(**json.loads(str(payload["config"])))
        sampler = DealSampler(**json.loads(str(payload["sampler"])))
        tree = BettingTree(config)
        sums = payload["strategy_sums"]
        legal = tree.legal[:, None, :]
        totals = sums.sum(axis=2, keepdims=True)
        uniform = legal / legal.sum(axis=2, keepdims=True).clip(min=1)
        strategy = np.where(totals > 0, sums / np.maximum(totals, 1e-30), uniform) * legal
        return cls(tree, strategy.astype(np.float64), sampler, iteration=int(payload["iteration"]))

    # -- serving contract ------------------------------------------------------

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        self._raise_fraction = None
        located = self._locate(game, player)
        if located is None:
            return self._safe_default(game, player)
        node = located
        bucket = self._bucket(game, player, int(self.tree.street[node]))
        if bucket is None:
            return self._safe_default(game, player)
        probabilities = self.strategy[node, bucket]
        actions = [action for action in range(self.tree.config.num_actions) if self.tree.legal[node][action]]
        weights = [max(float(probabilities[action]), 0.0) for action in actions]
        if sum(weights) <= 0:
            weights = [1.0] * len(actions)
        choice = self._rng.choices(actions, weights=weights)[0]
        if choice == FOLD:
            return NEURAL_FOLD
        if choice == CHECK_CALL:
            return NEURAL_CHECK_CALL
        if choice == ALL_IN:
            return NEURAL_ALL_IN
        return self._to_neural_raise(game, player, int(self.tree.street[node]), choice)

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        execute_action(game, player, choice, self._raise_fraction)

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        return None

    def parameter_count(self) -> int:
        return int(self.strategy.size)

    # -- translation ------------------------------------------------------------

    def _locate(self, game: HeadsUpHoldem, player: int) -> int | None:
        """Walk the flattened tree along the hand's public actions."""
        try:
            abstract_seat = 0 if player == game.button else 1
            node = self.tree.root
            rng = random.Random(game.hand_number * 8191 + len(game.public_actions))
            for event in game.public_actions:
                if event["action"] == "blind":
                    continue
                while self.tree.kind[node] == STREET_END:
                    node = int(self.tree.children[node][0])
                if self.tree.kind[node] != DECISION:
                    return None
                action = self._translate_event(node, game, event, rng)
                child = int(self.tree.children[node][action])
                if child < 0:
                    return None
                node = child
            while self.tree.kind[node] == STREET_END:
                node = int(self.tree.children[node][0])
            if self.tree.kind[node] != DECISION or int(self.tree.actor[node]) != abstract_seat:
                return None
            return node
        except Exception:
            return None

    def _translate_event(self, node: int, game: HeadsUpHoldem, event: dict, rng: random.Random) -> int:
        legal = self.tree.legal[node]
        kind = event["action"]
        if kind == "fold":
            return FOLD if legal[FOLD] else CHECK_CALL
        if kind in ("check", "call"):
            return CHECK_CALL
        if kind == "all_in" or event.get("action_index") == 3:
            return ALL_IN if legal[ALL_IN] else CHECK_CALL

        pot_before = float(event.get("pot_before", game.pot))
        to_call_before = float(event.get("to_call_before", 0))
        current_bet_before = float(event.get("current_bet_before", 0))
        pot_after_call = max(pot_before + to_call_before, 1.0)
        observed = max(float(event["amount"]) - current_bet_before, 0.0) / pot_after_call

        street = int(self.tree.street[node])
        fractions = self.tree.config.fractions(street)
        raise_ids = [3 + index for index in range(len(fractions)) if legal[3 + index]]
        if not raise_ids:
            return ALL_IN if legal[ALL_IN] and observed > 1.5 else CHECK_CALL
        sized = sorted(raise_ids, key=lambda action: fractions[action - 3])
        below = [action for action in sized if fractions[action - 3] <= observed]
        above = [action for action in sized if fractions[action - 3] >= observed]
        if not below:
            return above[0]
        if not above:
            return below[-1]
        lower, upper = below[-1], above[0]
        if lower == upper:
            return lower
        weight_lower, weight_upper = pseudo_harmonic_weights(observed, fractions[lower - 3], fractions[upper - 3])
        return rng.choices([lower, upper], weights=[weight_lower, weight_upper])[0]

    def _bucket(self, game: HeadsUpHoldem, player: int, street: int) -> int | None:
        hole = tuple(sorted(card_id(card) for card in game.hole_cards[player]))
        if street == 0:
            return preflop_class(hole)
        board = tuple(card_id(card) for card in game.community)[: (0, 3, 4, 5)[street]]
        cache_key = (hole, board, street)
        cached = self._equity_cache.get(cache_key)
        if cached is None:
            combo_index = _COMBO_INDEX[hole]
            if street == 3:
                equity = equity_from_scores(score_all_combos(board))[combo_index]
            else:
                rng = random.Random(hash((hole, board)) & 0x7FFFFFFF)
                samples = self.sampler.turn_samples if street == 2 else self.sampler.flop_samples
                equity = DealSampler._mean_equity(board, rng, samples)[combo_index]
            if equity < 0:
                return None
            cached = float(equity)
            self._equity_cache[cache_key] = cached
        counts = self.sampler.bucket_counts()
        return min(int(cached * counts[street]), counts[street] - 1)

    def _to_neural_raise(self, game: HeadsUpHoldem, player: int, street: int, choice: int) -> int:
        legal = game.legal_actions(player)
        if not legal.get("raise"):
            return NEURAL_ALL_IN if legal.get("all_in") else NEURAL_CHECK_CALL
        to_call = float(legal["to_call"])
        fraction = self.tree.config.fractions(street)[choice - 3]
        raise_by = fraction * (game.pot + to_call)
        target = float(legal["player_bet"]) + to_call + raise_by
        minimum, maximum = float(legal["raise_min"]), float(legal["raise_max"])
        if maximum <= minimum:
            self._raise_fraction = 0.5
        else:
            self._raise_fraction = min(0.995, max(0.005, (target - minimum) / (maximum - minimum)))
        return NEURAL_RAISE

    @staticmethod
    def _safe_default(game: HeadsUpHoldem, player: int) -> int:
        legal = game.legal_actions(player)
        if legal.get("check") or legal.get("call"):
            return NEURAL_CHECK_CALL
        if legal.get("all_in"):
            return NEURAL_ALL_IN
        return NEURAL_FOLD
