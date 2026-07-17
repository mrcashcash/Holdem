"""Bounded tabular CFR+ teacher for local heads-up Hold'em abstractions.

This is deliberately an abstraction oracle, not a claim to solve full no-limit
Hold'em. It supplies reproducible public-state policy/value targets for neural
distillation while the rules engine remains the source of legal play.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .poker import HeadsUpHoldem
from .rl_env import ACTION_COUNT


# Kept only as an opt-in diagnostic teacher.  These map to the semantic action
# space: fold, check/call, raise, all-in.
ABSTRACT_ACTIONS = (0, 1, 2, 3)


@dataclass
class AbstractTeacherRecord:
    observation: list[float]
    mask: list[bool]
    strategy: list[float]
    value: float
    confidence: float
    street: int

    def payload(self) -> dict:
        return {
            "observation": self.observation,
            "mask": self.mask,
            "strategy": self.strategy,
            "value": self.value,
            "confidence": self.confidence,
            "street": self.street,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> AbstractTeacherRecord:
        return cls(list(payload["observation"]), list(payload["mask"]), list(payload["strategy"]), float(payload.get("value", 0.0)), float(payload.get("confidence", 0.0)), int(payload.get("street", 0)))


class AbstractTeacherMemory:
    """Prioritized bounded replay of offline abstract-solver targets."""

    def __init__(self, capacity: int = 24_000) -> None:
        self.capacity = capacity
        self.records: list[AbstractTeacherRecord] = []
        self.seen = 0

    def extend(self, records: list[AbstractTeacherRecord], rng: random.Random) -> None:
        for record in records:
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                continue
            priority = record.confidence * (1.0 + record.street / 3)
            weakest = min(range(len(self.records)), key=lambda index: self.records[index].confidence * (1.0 + self.records[index].street / 3))
            weakest_priority = self.records[weakest].confidence * (1.0 + self.records[weakest].street / 3)
            if priority >= weakest_priority or rng.random() < 0.03:
                self.records[weakest] = record

    def sample(self, count: int, rng: random.Random) -> list[AbstractTeacherRecord]:
        if not self.records:
            return []
        requested = min(count, len(self.records))
        ranked = sorted(self.records, key=lambda record: record.confidence * (1.0 + record.street / 3), reverse=True)
        pool = ranked[:max(1, len(ranked) * 3 // 4)]
        return rng.sample(pool, min(requested, len(pool)))

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.records = [AbstractTeacherRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]


class AbstractCfrOracle:
    """Diagnostic abstraction teacher, not a Hold'em CFR implementation.

    It remains available for controlled experiments, but the production trainer
    leaves it disabled because its hand-authored utilities do not model an
    opponent strategy or an extensive-form game tree.
    """

    def __init__(self) -> None:
        self.regrets: dict[str, list[float]] = {}
        self.strategy_sums: dict[str, list[float]] = {}
        self.iterations = 0

    @staticmethod
    def _key(street: int, spr: int, equity: int, texture: int, pressure: int) -> str:
        return f"{street}:{spr}:{equity}:{texture}:{pressure}"

    @staticmethod
    def _regret_strategy(regrets: list[float]) -> list[float]:
        positive = [max(0.0, value) for value in regrets]
        total = sum(positive)
        return [value / total for value in positive] if total > 1e-9 else [0.25] * len(regrets)

    @staticmethod
    def _action_values(street: int, spr: int, equity: int, texture: int, pressure: int) -> list[float]:
        equity_value = equity / 4
        pressure_cost = (pressure + 1) * 0.14
        depth = 0.16 + spr * 0.09
        texture_risk = (texture - 1) * 0.035
        fold = -pressure_cost
        call = (equity_value - 0.50) * (0.75 + depth) - texture_risk * (1 - equity_value)
        half_pot = call + (equity_value - 0.48) * (0.20 + depth) - pressure * 0.035
        pot = call + (equity_value - 0.54) * (0.42 + depth) + (street / 3) * 0.025 - pressure * 0.055
        return [fold, call, half_pot, pot]

    def solve(self, iterations: int = 24) -> None:
        """Advance a compact local normal-form CFR+ approximation deterministically."""
        for _ in range(max(1, iterations)):
            for street in range(4):
                for spr in range(4):
                    for equity in range(5):
                        for texture in range(3):
                            for pressure in range(3):
                                key = self._key(street, spr, equity, texture, pressure)
                                regrets = self.regrets.setdefault(key, [0.0] * len(ABSTRACT_ACTIONS))
                                sums = self.strategy_sums.setdefault(key, [0.0] * len(ABSTRACT_ACTIONS))
                                strategy = self._regret_strategy(regrets)
                                values = self._action_values(street, spr, equity, texture, pressure)
                                baseline = sum(weight * value for weight, value in zip(strategy, values))
                                for index, (weight, value) in enumerate(zip(strategy, values)):
                                    regrets[index] = max(0.0, regrets[index] + value - baseline)
                                    sums[index] += weight
            self.iterations += 1

    @staticmethod
    def _bucket(game: HeadsUpHoldem, player: int) -> tuple[int, int, int, int, int]:
        ranks = sorted((card[0] for card in game.hole_cards[player]), reverse=True)
        pair = ranks[0] == ranks[1]
        suited = game.hole_cards[player][0][1] == game.hole_cards[player][1][1]
        connected = abs(ranks[0] - ranks[1]) <= 2
        raw_strength = (ranks[0] + ranks[1]) / 28 + (0.24 if pair else 0.0) + (0.06 if suited else 0.0) + (0.04 if connected else 0.0)
        if game.community:
            matched = sum(rank in [card[0] for card in game.community] for rank in ranks)
            raw_strength += matched * 0.08
        equity = min(4, max(0, int(raw_strength * 5)))
        spr = min(3, int(min(game.stacks) / max(1, game.pot + game.big_blind)))
        board_suits = [card[1] for card in game.community]
        texture = 2 if board_suits and max(board_suits.count(suit) for suit in set(board_suits)) >= 3 else 1 if len({card[0] for card in game.community}) < len(game.community) else 0
        pressure = min(2, int(game.to_call(player) / max(1, game.big_blind * 2)))
        return game.street, spr, equity, texture, pressure

    def target(self, game: HeadsUpHoldem, player: int, mask: list[bool], observation: list[float]) -> AbstractTeacherRecord:
        street, spr, equity, texture, pressure = self._bucket(game, player)
        key = self._key(street, spr, equity, texture, pressure)
        sums = self.strategy_sums.get(key, [1.0] * len(ABSTRACT_ACTIONS))
        total = sum(sums)
        abstract_strategy = [value / total for value in sums] if total > 1e-9 else [0.25] * len(sums)
        projected = [0.0] * ACTION_COUNT
        for action, weight in zip(ABSTRACT_ACTIONS, abstract_strategy):
            if mask[action]:
                projected[action] += weight
        legal_total = sum(projected)
        if legal_total <= 1e-9:
            legal = [index for index, allowed in enumerate(mask) if allowed]
            for action in legal:
                projected[action] = 1.0 / max(1, len(legal))
        else:
            projected = [value / legal_total for value in projected]
        values = self._action_values(street, spr, equity, texture, pressure)
        value = sum(weight * action_value for weight, action_value in zip(abstract_strategy, values))
        confidence = min(0.96, self.iterations / (self.iterations + 24)) * (0.65 + 0.35 * (street / 3))
        return AbstractTeacherRecord(observation, mask, projected, value, confidence, street)

    def snapshot(self) -> dict:
        return {"iterations": self.iterations, "regrets": self.regrets, "strategy_sums": self.strategy_sums}

    def restore(self, payload: dict) -> None:
        self.iterations = max(0, int(payload.get("iterations", 0)))
        self.regrets = {str(key): [float(value) for value in values] for key, values in payload.get("regrets", {}).items() if isinstance(values, list) and len(values) == len(ABSTRACT_ACTIONS)}
        self.strategy_sums = {str(key): [float(value) for value in values] for key, values in payload.get("strategy_sums", {}).items() if isinstance(values, list) and len(values) == len(ABSTRACT_ACTIONS)}
