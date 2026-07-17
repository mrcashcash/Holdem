"""Compact public-belief targets for range-conditioned counterfactual values.

The representation intentionally aggregates exact two-card combinations into a
rank grid plus suit-category mass.  It is a bounded approximation suitable for
the local trainer, not a claim of exact full-game safe solving.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


BELIEF_VALUE_CLASSES = 13 * 13
BELIEF_FEATURE_SIZE = BELIEF_VALUE_CLASSES + 6
TWO_SIDED_BELIEF_FEATURE_SIZE = BELIEF_FEATURE_SIZE * 2


def belief_class(cards: tuple[tuple[int, str], tuple[int, str]] | list[tuple[int, str]]) -> int:
    """Map an exact holding to an order-invariant rank-grid cell."""
    high, low = sorted((card[0] - 2 for card in cards), reverse=True)
    return high * 13 + low


def belief_features(belief: object) -> list[float]:
    """Compress exact posterior mass while retaining suitedness information."""
    grid = [0.0] * BELIEF_VALUE_CLASSES
    suit_mass = [0.0, 0.0, 0.0]  # pair, suited non-pair, offsuit
    for cards, probability in zip(getattr(belief, "candidates", []), getattr(belief, "combination_reach", [])):
        grid[belief_class(cards)] += float(probability)
        if cards[0][0] == cards[1][0]:
            suit_mass[0] += float(probability)
        elif cards[0][1] == cards[1][1]:
            suit_mass[1] += float(probability)
        else:
            suit_mass[2] += float(probability)
    return [*grid, *suit_mass, float(getattr(belief, "entropy", 1.0)), float(getattr(belief, "effective_support", 1.0)), float(getattr(belief, "top_mass", 0.0))]


def private_belief_features(cards: tuple[tuple[int, str], tuple[int, str]] | list[tuple[int, str]]) -> list[float]:
    """Encode the acting player's exact private holding as a degenerate range."""
    grid = [0.0] * BELIEF_VALUE_CLASSES
    grid[belief_class(cards)] = 1.0
    pair = float(cards[0][0] == cards[1][0])
    suited = float(not pair and cards[0][1] == cards[1][1])
    return [*grid, pair, suited, float(not pair and not suited), 0.0, 1.0, 1.0]


@dataclass
class CounterfactualValueRecord:
    observation: list[float]
    own_belief: list[float]
    belief: list[float]
    classes: list[int]
    values: list[float]
    weights: list[float]
    confidence: float
    depth: int

    @classmethod
    def from_cfr(cls, record: object) -> CounterfactualValueRecord | None:
        own_belief = list(getattr(record, "own_belief_features", []) or [])
        belief = list(getattr(record, "belief_features", []) or [])
        classes = [int(value) for value in (getattr(record, "counterfactual_classes", []) or [])]
        values = [float(value) for value in (getattr(record, "counterfactual_values", []) or [])]
        weights = [float(value) for value in (getattr(record, "counterfactual_weights", []) or [])]
        if len(own_belief) != BELIEF_FEATURE_SIZE or len(belief) != BELIEF_FEATURE_SIZE or not classes or len(classes) != len(values) or len(values) != len(weights):
            return None
        valid = [(kind, value, weight) for kind, value, weight in zip(classes, values, weights) if 0 <= kind < BELIEF_VALUE_CLASSES and weight > 0]
        if not valid:
            return None
        compact_classes, compact_values, compact_weights = zip(*valid)
        return cls(list(getattr(record, "observation")), own_belief, belief, list(compact_classes), list(compact_values), list(compact_weights), float(getattr(record, "resolver_confidence", 0.0)), int(getattr(record, "search_depth", 0)))

    def payload(self) -> dict:
        return {"observation": self.observation, "own_belief": self.own_belief, "belief": self.belief, "classes": self.classes, "values": self.values, "weights": self.weights, "confidence": self.confidence, "depth": self.depth}

    @classmethod
    def from_payload(cls, payload: dict) -> CounterfactualValueRecord:
        return cls(list(payload["observation"]), list(payload.get("own_belief", [])), list(payload["belief"]), [int(value) for value in payload["classes"]], [float(value) for value in payload["values"]], [float(value) for value in payload["weights"]], float(payload.get("confidence", 0.0)), int(payload.get("depth", 0)))


class CounterfactualValueMemory:
    """Bounded prioritized replay of sparse, solver-created value surfaces."""

    def __init__(self, capacity: int = 12_000) -> None:
        self.capacity = capacity
        self.records: list[CounterfactualValueRecord] = []
        self.seen = 0

    def extend(self, records: list[CounterfactualValueRecord], rng: random.Random) -> None:
        for record in records:
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                continue
            priority = record.confidence * (1.0 + record.depth / 16) * max(record.weights)
            weakest = min(range(len(self.records)), key=lambda index: self.records[index].confidence * (1.0 + self.records[index].depth / 16) * max(self.records[index].weights))
            weakest_priority = self.records[weakest].confidence * (1.0 + self.records[weakest].depth / 16) * max(self.records[weakest].weights)
            if priority >= weakest_priority or rng.random() < 0.04:
                self.records[weakest] = record

    def sample(self, count: int, rng: random.Random) -> list[CounterfactualValueRecord]:
        if not self.records:
            return []
        ranked = sorted(self.records, key=lambda record: record.confidence * (1.0 + record.depth / 16) * max(record.weights), reverse=True)
        pool = ranked[:max(1, len(ranked) * 3 // 4)]
        return rng.sample(pool, min(count, len(pool)))

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.records = [CounterfactualValueRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]
