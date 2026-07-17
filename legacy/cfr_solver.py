"""Counterfactual samples, reservoir memory, and public-belief re-solving helpers."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Callable

from .poker import HeadsUpHoldem, new_deck
from .rl_env import ACTION_COUNT, RAISE_ACTIONS, RANGE_BUCKETS, action_context_features, continuous_raise_target, estimate_range_equity, event_context_features, execute_action, hand_bucket, normal_raise_bounds
from .counterfactual_values import belief_class, belief_features, private_belief_features


# Numerical bound for approximate value leaves. The environment payoff remains
# the actual terminal chip delta measured in big blinds.
MAX_APPROXIMATE_VALUE_BB = 200.0


@dataclass
class CFRRecord:
    """External-sampling target that never exposes opponent cards to policy inputs."""

    observation: list[float]
    mask: list[bool]
    sampled: list[bool]
    advantages: list[float]
    strategy: list[float]
    rare: bool
    iteration: int
    reach_weight: float
    leaf_evaluations: int
    priority: float = 1.0
    sizing_targets: list[float] | None = None
    sizing_weights: list[float] | None = None
    search_value: float = 0.0
    search_uncertainty: float = 0.0
    search_depth: int = 0
    belief_entropy: float = 1.0
    belief_support: float = 1.0
    resolver_confidence: float = 0.0
    own_belief_features: list[float] | None = None
    belief_features: list[float] | None = None
    counterfactual_classes: list[int] | None = None
    counterfactual_values: list[float] | None = None
    counterfactual_weights: list[float] | None = None

    def payload(self) -> dict:
        return {"observation": self.observation, "mask": self.mask, "sampled": self.sampled, "advantages": self.advantages, "strategy": self.strategy, "rare": self.rare, "iteration": self.iteration, "reach_weight": self.reach_weight, "leaf_evaluations": self.leaf_evaluations, "priority": self.priority, "sizing_targets": self.sizing_targets or [0.5] * ACTION_COUNT, "sizing_weights": self.sizing_weights or [0.0] * ACTION_COUNT, "search_value": self.search_value, "search_uncertainty": self.search_uncertainty, "search_depth": self.search_depth, "belief_entropy": self.belief_entropy, "belief_support": self.belief_support, "resolver_confidence": self.resolver_confidence, "own_belief_features": self.own_belief_features or [], "belief_features": self.belief_features or [], "counterfactual_classes": self.counterfactual_classes or [], "counterfactual_values": self.counterfactual_values or [], "counterfactual_weights": self.counterfactual_weights or []}

    @classmethod
    def from_payload(cls, payload: dict) -> CFRRecord:
        sampled = list(payload.get("sampled", payload["mask"]))
        return cls(list(payload["observation"]), list(payload["mask"]), sampled, list(payload["advantages"]), list(payload["strategy"]), bool(payload.get("rare", False)), int(payload.get("iteration", 1)), float(payload.get("reach_weight", 1.0)), int(payload.get("leaf_evaluations", 0)), float(payload.get("priority", 1.0)), list(payload.get("sizing_targets", [0.5] * ACTION_COUNT)), list(payload.get("sizing_weights", [0.0] * ACTION_COUNT)), float(payload.get("search_value", 0.0)), float(payload.get("search_uncertainty", 0.0)), int(payload.get("search_depth", 0)), float(payload.get("belief_entropy", 1.0)), float(payload.get("belief_support", 1.0)), float(payload.get("resolver_confidence", 0.0)), list(payload.get("own_belief_features", [])), list(payload.get("belief_features", [])), [int(value) for value in payload.get("counterfactual_classes", [])], [float(value) for value in payload.get("counterfactual_values", [])], [float(value) for value in payload.get("counterfactual_weights", [])])


@dataclass
class ActionLikelihoodRecord:
    context: list[float]
    history: list[list[float]]
    range_class: int
    action: int

    def payload(self) -> dict:
        return {"context": self.context, "history": self.history, "range_class": self.range_class, "action": self.action}

    @classmethod
    def from_payload(cls, payload: dict) -> ActionLikelihoodRecord:
        context = list(payload["context"])
        history = [list(item) for item in payload.get("history", [context])]
        return cls(context, history, int(payload["range_class"]), int(payload["action"]))


class ReservoirMemory:
    """Stratified replay for rare, high-regret, and recently discovered decisions."""

    def __init__(self, capacity: int = 16_000) -> None:
        self.capacity = capacity
        self.records: list[CFRRecord] = []
        self.seen = 0

    def extend(self, records: list[CFRRecord], rng: random.Random) -> None:
        for record in records:
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                continue
            rare_records = [index for index, current in enumerate(self.records) if current.rare]
            non_rare_records = [index for index, current in enumerate(self.records) if not current.rare]
            if record.rare and len(rare_records) / max(1, len(self.records)) < 0.38 and non_rare_records:
                replacement = min(non_rare_records, key=lambda index: self.records[index].priority)
            elif rng.random() < 0.18:
                replacement = min(range(len(self.records)), key=lambda index: self.records[index].iteration)
            else:
                ranked = sorted(range(len(self.records)), key=lambda index: self.records[index].priority)
                pool = ranked[:max(1, len(ranked) // 5)]
                replacement = rng.choice(pool)
                if record.priority < self.records[replacement].priority and rng.random() > 0.08:
                    continue
            self.records[replacement] = record

    def sample(self, count: int, rng: random.Random) -> list[CFRRecord]:
        if not self.records:
            return []
        requested = min(count, len(self.records))
        selected: list[CFRRecord] = []
        selected_ids: set[int] = set()

        def take(pool: list[CFRRecord], amount: int) -> None:
            for record in rng.sample(pool, min(amount, len(pool))):
                if id(record) not in selected_ids:
                    selected.append(record)
                    selected_ids.add(id(record))

        rare = [record for record in self.records if record.rare]
        high_priority = sorted(self.records, key=lambda record: record.priority, reverse=True)[:max(1, len(self.records) // 3)]
        recent = sorted(self.records, key=lambda record: record.iteration, reverse=True)[:max(1, len(self.records) // 3)]
        take(rare, round(requested * 0.42))
        take(high_priority, round(requested * 0.36))
        take(recent, requested - len(selected))
        if len(selected) < requested:
            take([record for record in self.records if id(record) not in selected_ids], requested - len(selected))
        return selected[:requested]

    def composition(self) -> dict[str, float]:
        if not self.records:
            return {"rare": 0.0, "priority": 0.0, "recent": 0.0}
        newest = max(record.iteration for record in self.records)
        return {
            "rare": sum(record.rare for record in self.records) / len(self.records),
            "priority": sum(record.priority for record in self.records) / len(self.records),
            "recent": sum(record.iteration >= newest - 2 for record in self.records) / len(self.records),
        }

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.records = [CFRRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]


class StrategyMemory:
    """Independent reservoir for the SD-CFR-style average-strategy target."""

    def __init__(self, capacity: int = 24_000) -> None:
        self.capacity = capacity
        self.records: list[CFRRecord] = []
        self.seen = 0

    def extend(self, records: list[CFRRecord], rng: random.Random) -> None:
        for record in records:
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                continue
            index = rng.randrange(self.seen)
            if index < self.capacity:
                self.records[index] = record

    def sample(self, count: int, rng: random.Random) -> list[CFRRecord]:
        return rng.sample(self.records, min(count, len(self.records))) if self.records else []

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.records = [CFRRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]


@dataclass
class SearchValueRecord:
    """Belief-search value target used to distil local resolving into the value ensemble."""

    observation: list[float]
    mask: list[bool]
    value: float
    uncertainty: float
    depth: int
    priority: float

    @classmethod
    def from_cfr(cls, record: CFRRecord) -> SearchValueRecord:
        return cls(record.observation, record.mask, record.search_value, record.search_uncertainty, record.search_depth, record.priority)

    def payload(self) -> dict:
        return {"observation": self.observation, "mask": self.mask, "value": self.value, "uncertainty": self.uncertainty, "depth": self.depth, "priority": self.priority}

    @classmethod
    def from_payload(cls, payload: dict) -> SearchValueRecord:
        return cls(list(payload["observation"]), list(payload["mask"]), float(payload.get("value", 0.0)), float(payload.get("uncertainty", 0.0)), int(payload.get("depth", 0)), float(payload.get("priority", 1.0)))


class SearchValueMemory:
    """Prioritized bounded replay of search-generated value targets."""

    def __init__(self, capacity: int = 18_000) -> None:
        self.capacity = capacity
        self.records: list[SearchValueRecord] = []
        self.seen = 0

    def extend(self, records: list[SearchValueRecord], rng: random.Random) -> None:
        for record in records:
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                continue
            replacement = min(range(len(self.records)), key=lambda index: self.records[index].priority) if rng.random() < 0.72 else rng.randrange(len(self.records))
            if record.priority >= self.records[replacement].priority or rng.random() < 0.08:
                self.records[replacement] = record

    def sample(self, count: int, rng: random.Random) -> list[SearchValueRecord]:
        if not self.records:
            return []
        requested = min(count, len(self.records))
        selected: list[SearchValueRecord] = []
        selected_ids: set[int] = set()

        def take(pool: list[SearchValueRecord], amount: int) -> None:
            for record in rng.sample(pool, min(amount, len(pool))):
                if id(record) not in selected_ids:
                    selected.append(record)
                    selected_ids.add(id(record))

        by_priority = sorted(self.records, key=lambda record: record.priority, reverse=True)[:max(1, len(self.records) * 3 // 4)]
        by_uncertainty = sorted(self.records, key=lambda record: record.uncertainty * max(1, record.depth), reverse=True)[:max(1, len(self.records) // 2)]
        by_depth = sorted(self.records, key=lambda record: record.depth, reverse=True)[:max(1, len(self.records) // 2)]
        take(by_priority, round(requested * 0.50))
        take(by_uncertainty, round(requested * 0.30))
        take(by_depth, requested - len(selected))
        if len(selected) < requested:
            take([record for record in self.records if id(record) not in selected_ids], requested - len(selected))
        return selected[:requested]

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.records = [SearchValueRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]


class ActionLikelihoodMemory:
    """Reservoir of revealed self-play actions for supervised Bayesian likelihood training."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.records: list[ActionLikelihoodRecord] = []
        self.seen = 0

    def extend(self, records: list[ActionLikelihoodRecord], rng: random.Random) -> None:
        for record in records:
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                continue
            index = rng.randrange(self.seen)
            if index < self.capacity:
                self.records[index] = record

    def sample(self, count: int, rng: random.Random) -> list[ActionLikelihoodRecord]:
        return rng.sample(self.records, min(count, len(self.records))) if self.records else []

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.records = [ActionLikelihoodRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]


@dataclass(frozen=True)
class ActionChoice:
    """One legal public action plus an exact continuous raise fraction when needed."""

    action: int
    raise_fraction: float | None = None


ContinuationPolicy = Callable[[HeadsUpHoldem, int], ActionChoice]
LeafEvaluator = Callable[[HeadsUpHoldem, int], float]
ActionLikelihoodModel = Callable[[list[list[float]]], list[list[list[float]]]]


@dataclass
class PublicBeliefState:
    """Reach-weighted posterior over all 1,326 exact opponent card combinations."""

    class_reach: list[float]
    entropy: float
    candidates: list[tuple[tuple[int, str], tuple[int, str]]]
    combination_reach: list[float]
    effective_support: float
    top_mass: float


@dataclass
class SearchResult:
    """Bounded public-belief root search summary for a single real decision."""

    action: int
    raise_fraction: float | None
    depth: int
    width: int
    leaf_evaluations: int
    value_spread: float
    confidence: float
    root_value: float = 0.0
    adaptive_raises: int = 0
    endgame_worlds: int = 0
    safety_rejections: int = 0
    safety_margin: float = 0.0
    safety_confidence: float = 0.0
    confident_actions: int = 0
    iterations: int = 0
    average_strategy_peak: float = 0.0


def _candidate_likelihood(cards: tuple[tuple[int, str], tuple[int, str]], game: HeadsUpHoldem, viewer: int, range_bias: list[float], action_likelihoods: dict[int, list[list[float]]] | None) -> float:
    ranks = sorted((card[0] for card in cards), reverse=True)
    pair = ranks[0] == ranks[1]
    suited = cards[0][1] == cards[1][1]
    strength = (ranks[0] + ranks[1]) / 28 + (0.48 if pair else 0) + (0.09 if suited else 0)
    range_class = hand_bucket(cards)
    likelihood = 0.2 + range_bias[range_class] * RANGE_BUCKETS
    for index, event in enumerate(game.public_actions):
        if event["player"] == viewer:
            continue
        action_index = int(event.get("action_index", -1))
        if action_likelihoods is not None and action_index >= 0:
            likelihood *= max(1e-6, action_likelihoods[index][range_class][action_index])
        elif event["action"] == "raise":
            likelihood *= 0.20 + strength * strength
        elif event["action"] == "call":
            likelihood *= 0.55 + strength * 0.70
        elif event["action"] == "check":
            likelihood *= 1.25 - min(0.55, strength * 0.28)
    return max(1e-8, likelihood)


def build_public_belief(game: HeadsUpHoldem, player: int, range_bias: list[float], action_likelihood_model: ActionLikelihoodModel | None = None) -> PublicBeliefState:
    """Apply card removal and public-action reach likelihoods without inspecting hole cards."""
    known = set(game.hole_cards[player] + game.community)
    deck = [card for card in new_deck() if card not in known]
    sequence_events = [(index, event) for index, event in enumerate(game.public_actions) if int(event.get("action_index", -1)) >= 0]
    sequence_outputs = action_likelihood_model([event_context_features(event, game.initial_stack) for _, event in sequence_events]) if action_likelihood_model is not None and sequence_events else []
    action_likelihoods = {
        index: sequence_outputs[position]
        for position, (index, event) in enumerate(sequence_events)
        if event["player"] != player
    } if sequence_outputs else None
    candidates = [(deck[first], deck[second]) for first in range(len(deck)) for second in range(first + 1, len(deck))]
    raw_weights = [_candidate_likelihood(cards, game, player, range_bias, action_likelihoods) for cards in candidates]
    total_weight = sum(raw_weights) or 1.0
    combination_reach = [weight / total_weight for weight in raw_weights]
    masses = [0.0] * RANGE_BUCKETS
    for cards, probability in zip(candidates, combination_reach):
        masses[hand_bucket(cards)] += probability
    total = sum(masses) or 1.0
    reach = [mass / total for mass in masses]
    entropy = -sum(value * math.log(value + 1e-12) for value in reach if value) / math.log(RANGE_BUCKETS)
    combination_entropy = -sum(value * math.log(value + 1e-12) for value in combination_reach if value)
    effective_support = math.exp(combination_entropy) / max(1, len(candidates))
    return PublicBeliefState(reach, entropy, candidates, combination_reach, effective_support, max(combination_reach, default=0.0))


def _sample_belief_world(game: HeadsUpHoldem, player: int, belief: PublicBeliefState, rng: random.Random) -> HeadsUpHoldem:
    """Sample a legal opponent hand and remaining deck from a public posterior."""
    known = set(game.hole_cards[player] + game.community)
    candidates = belief.candidates
    weights = belief.combination_reach
    if not candidates or len(candidates) != len(weights):
        deck = [card for card in new_deck() if card not in known]
        candidates = [(deck[first], deck[second]) for first in range(len(deck)) for second in range(first + 1, len(deck))]
        weights = [1.0] * len(candidates)
    chosen = rng.choices(candidates, weights=weights, k=1)[0]
    world = copy.deepcopy(game)
    opponent = 1 - player
    world.hole_cards[opponent] = list(chosen)
    world.deck = [card for card in new_deck() if card not in known and card not in chosen]
    rng.shuffle(world.deck)
    return world


def _finish_counterfactual(game: HeadsUpHoldem, player: int, continuation: ContinuationPolicy, value_leaf: LeafEvaluator, depth_limit: int = 8) -> tuple[float, int]:
    """Approximate rollout with a learned value leaf once the lookahead ends."""
    depth = 0
    while not game.hand_complete and depth < depth_limit:
        current = game.current_player
        assert current is not None
        decision = continuation(game, current)
        execute_action(game, current, decision.action, decision.raise_fraction)
        depth += 1
    if not game.hand_complete:
        return max(-MAX_APPROXIMATE_VALUE_BB, min(MAX_APPROXIMATE_VALUE_BB, value_leaf(game, player))), 1
    return (game.stacks[player] - game.initial_stack) / game.big_blind, 0


def _adaptive_raise_fractions(game: HeadsUpHoldem, player: int, base_fraction: float, proposed_fractions: list[float] | None = None) -> list[float]:
    """Translate strategic pot/commitment landmarks into legal continuous raise fractions."""
    legal = game.legal_actions(player)
    if not legal.get("raise"):
        return []
    minimum, maximum = normal_raise_bounds(game, player)
    current = max(game.round_bets)
    if maximum <= minimum:
        return [0.5]
    landmarks = [minimum, maximum]
    landmarks.extend(current + round(game.pot * fraction) for fraction in (0.25, 1 / 3, 0.5, 2 / 3, 1.0, 1.5))
    observed_raises = [int(event.get("amount", 0)) for event in game.public_actions if event.get("player") != player and event.get("action") == "raise"]
    if observed_raises:
        latest = observed_raises[-1]
        landmarks.extend((latest, current + max(game.big_blind, latest - current), current + max(game.big_blind, round((latest - current) * 1.5))))
    base = min(0.995, max(0.005, base_fraction))
    landmarks.extend(round(minimum + (maximum - minimum) * fraction) for fraction in (base, max(0.01, base - 0.14), min(0.99, base + 0.14)))
    if proposed_fractions:
        landmarks.extend(round(minimum + (maximum - minimum) * min(0.995, max(0.005, fraction))) for fraction in proposed_fractions)
    unique_targets = {min(maximum, max(minimum, target)) for target in landmarks}
    fractions = [(target - minimum) / max(1, maximum - minimum) for target in unique_targets]
    return sorted(fractions, key=lambda fraction: (abs(fraction - base), fraction))


def _candidate_decisions(game: HeadsUpHoldem, player: int, mask: list[bool], chosen_action: int, chosen_fraction: float | None, raise_fractions: list[float] | None = None, limit: int = 5, advantage_scores: list[float] | None = None, policy_scores: list[float] | None = None, raise_proposals: dict[int, list[float]] | None = None) -> list[ActionChoice]:
    """Keep a compact root set with learned, pot-based, and commitment-aware raise sizes."""
    legal = [index for index, available in enumerate(mask) if available]
    preferred = [chosen_action, 1, 2, 3, 0]
    selected: list[ActionChoice] = []
    seen_raise_targets: set[int] = set()
    score_floor = max(policy_scores) - 2.4 if policy_scores else float("-inf")

    def fraction_for(action: int) -> float | None:
        if action not in RAISE_ACTIONS:
            return None
        if action == chosen_action and chosen_fraction is not None:
            return min(0.995, max(0.005, chosen_fraction))
        if raise_fractions is not None and len(raise_fractions) > action:
            return min(0.995, max(0.005, raise_fractions[action]))
        return 0.5

    def append(action: int, fraction: float | None) -> None:
        if len(selected) >= limit:
            return
        if action in RAISE_ACTIONS and fraction is not None:
            target = continuous_raise_target(game, player, fraction)
            if target in seen_raise_targets:
                return
            seen_raise_targets.add(target)
        decision = ActionChoice(action, fraction)
        if decision not in selected:
            selected.append(decision)

    for action in preferred:
        prunable = action in RAISE_ACTIONS and action != chosen_action and advantage_scores is not None and policy_scores is not None and advantage_scores[action] < -0.65 and policy_scores[action] < score_floor
        if action in legal and not prunable:
            base = fraction_for(action)
            if action in RAISE_ACTIONS and base is not None:
                fraction_limit = 3 if action == chosen_action else 1
                for fraction in _adaptive_raise_fractions(game, player, base, (raise_proposals or {}).get(action))[:fraction_limit]:
                    append(action, fraction)
                    if len(selected) >= limit:
                        break
            else:
                append(action, base)
        if len(selected) == limit:
            return selected
    for action in legal:
        base = fraction_for(action)
        if action in RAISE_ACTIONS and base is not None:
            for fraction in _adaptive_raise_fractions(game, player, base, (raise_proposals or {}).get(action))[:1]:
                append(action, fraction)
                if len(selected) >= limit:
                    break
        else:
            append(action, base)
        if len(selected) == limit:
            break
    return selected


def _average_regret_strategy(values: dict[ActionChoice, float], iterations: int) -> dict[ActionChoice, float]:
    """Run bounded regret matching over fixed sampled root values."""
    regrets = {decision: 0.0 for decision in values}
    average = {decision: 0.0 for decision in values}
    for _ in range(max(1, iterations)):
        positive = {decision: max(0.0, regret) for decision, regret in regrets.items()}
        total = sum(positive.values())
        strategy = {decision: positive[decision] / total for decision in values} if total > 1e-8 else {decision: 1 / len(values) for decision in values}
        expected = sum(strategy[decision] * values[decision] for decision in values)
        for decision in values:
            regrets[decision] += values[decision] - expected
            average[decision] += strategy[decision]
    total_average = sum(average.values())
    return {decision: average[decision] / max(1e-8, total_average) for decision in values}


def external_sample_record(game: HeadsUpHoldem, player: int, features: list[float], mask: list[bool], chosen_action: int, chosen_fraction: float | None, range_bias: list[float], action_likelihood_model: ActionLikelihoodModel, continuation: ContinuationPolicy, value_leaf: LeafEvaluator, iteration: int, rng: random.Random, world_samples: int = 2, action_limit: int = 5, depth_limit: int = 8, resolver_iterations: int = 4) -> CFRRecord | None:
    """Collect an opt-in approximate resolver-distillation record.

    This is not MCCFR or Deep CFR: it evaluates a bounded root set with learned
    continuations. Production training leaves this experimental lane disabled
    until a full information-set implementation has passed validation.
    """
    decisions = _candidate_decisions(game, player, mask, chosen_action, chosen_fraction, limit=action_limit)
    if len(decisions) < 2:
        return None
    belief = build_public_belief(game, player, range_bias, action_likelihood_model)
    values = {decision: 0.0 for decision in decisions}
    sampled_values = {decision: [] for decision in decisions}
    world_values: dict[int, dict[ActionChoice, list[float]]] = {}
    leaf_evaluations = 0
    for _ in range(world_samples):
        world = _sample_belief_world(game, player, belief, rng)
        world_class = belief_class(world.hole_cards[1 - player])
        world_values.setdefault(world_class, {})
        for decision in decisions:
            branch = copy.deepcopy(world)
            execute_action(branch, player, decision.action, decision.raise_fraction)
            value, leaves = _finish_counterfactual(branch, player, continuation, value_leaf, depth_limit)
            sampled_values[decision].append(value)
            world_values[world_class].setdefault(decision, []).append(value)
            leaf_evaluations += leaves
    values = {decision: sum(outcomes) / len(outcomes) for decision, outcomes in sampled_values.items()}
    sampling_error = sum(
        math.sqrt(sum((value - values[decision]) ** 2 for value in outcomes) / max(1, len(outcomes) - 1)) / math.sqrt(len(outcomes))
        for decision, outcomes in sampled_values.items()
    ) / len(sampled_values)
    baseline = sum(values.values()) / len(values)
    advantages = [0.0] * ACTION_COUNT
    sampled = [False] * ACTION_COUNT
    sizing_targets = [0.5] * ACTION_COUNT
    sizing_weights = [0.0] * ACTION_COUNT
    action_values: dict[int, list[tuple[float, float | None]]] = {}
    for decision, value in values.items():
        action_values.setdefault(decision.action, []).append((value, decision.raise_fraction))
    decision_strategy = _average_regret_strategy(values, resolver_iterations)
    positive = 0.0
    for action, pairs in action_values.items():
        value = sum(item[0] for item in pairs) / len(pairs)
        sampled[action] = True
        advantages[action] = value - baseline
        positive += max(0.0, advantages[action])
        if action in RAISE_ACTIONS:
            best_value, best_fraction = max(pairs, key=lambda item: item[0])
            sizing_targets[action] = 0.5 if best_fraction is None else best_fraction
            sizing_weights[action] = max(0.02, best_value - baseline)
    strategy = [0.0] * ACTION_COUNT
    for decision, probability in decision_strategy.items():
        strategy[decision.action] += probability
    opponent_raises = sum(event["player"] == (1 - player) and event["action"] == "raise" for event in game.public_actions)
    rare = game.street >= 2 or game.to_call(player) >= max(game.big_blind * 3, game.pot * 0.4) or opponent_raises >= 2
    posterior_confidence = 0.35 + 0.40 * (1 - belief.entropy) + 0.25 * (1 - belief.effective_support)
    sampling_confidence = 1 / (1 + sampling_error)
    resolver_confidence = max(0.05, min(0.98, posterior_confidence * sampling_confidence))
    reach_weight = 0.35 + 0.65 * resolver_confidence
    regret_spread = max(values.values()) - min(values.values())
    resolved_value = sum(strategy[action] * (baseline + advantages[action]) for action in action_values)
    resolved_uncertainty = math.sqrt(sum((value - resolved_value) ** 2 for value in values.values()) / len(values))
    priority = max(0.05, regret_spread) * (1.75 if rare else 1.0) * (1.0 + 0.55 * resolver_confidence) * (1.0 + min(1.5, resolved_uncertainty) * 0.35) * (1.0 + min(0.5, depth_limit / 24))
    class_mass: dict[int, float] = {}
    for cards, probability in zip(belief.candidates, belief.combination_reach):
        kind = belief_class(cards)
        class_mass[kind] = class_mass.get(kind, 0.0) + probability
    target_classes: list[int] = []
    target_values: list[float] = []
    target_weights: list[float] = []
    for kind, decision_values in world_values.items():
        grouped_samples: dict[int, list[float]] = {}
        for decision, samples in decision_values.items():
            grouped_samples.setdefault(decision.action, []).extend(samples)
        grouped = {action: sum(samples) / len(samples) for action, samples in grouped_samples.items()}
        target = sum(strategy[action] * value for action, value in grouped.items())
        target_classes.append(kind)
        target_values.append(max(-MAX_APPROXIMATE_VALUE_BB, min(MAX_APPROXIMATE_VALUE_BB, target)))
        target_weights.append(max(1e-4, class_mass.get(kind, 0.0)))
    return CFRRecord(features, mask, sampled, advantages, strategy, rare, iteration, reach_weight, leaf_evaluations, priority, sizing_targets, sizing_weights, resolved_value, resolved_uncertainty, depth_limit, belief.entropy, belief.effective_support, resolver_confidence, private_belief_features(game.hole_cards[player]), belief_features(belief), target_classes, target_values, target_weights)


def robust_belief_search(
    game: HeadsUpHoldem,
    player: int,
    policy_scores: list[float],
    advantage_scores: list[float],
    legal: list[bool],
    raise_fractions: list[float],
    range_bias: list[float],
    action_likelihood_model: ActionLikelihoodModel,
    continuation: ContinuationPolicy,
    value_leaf: LeafEvaluator,
    value_uncertainty: float,
    rng: random.Random,
    world_samples: int = 2,
    depth_limit: int = 4,
    counterfactual_leaf: Callable[[HeadsUpHoldem, int, PublicBeliefState], tuple[float, float]] | None = None,
    resolver_iterations: int = 4,
    raise_proposals: dict[int, list[float]] | None = None,
) -> SearchResult:
    """Choose from posterior-world roots using lower-confidence values as a bounded safety guard."""
    policy_choice = max((index for index, available in enumerate(legal) if available), key=lambda index: policy_scores[index])
    policy_fraction = raise_fractions[policy_choice] if policy_choice in RAISE_ACTIONS else None
    decisions = _candidate_decisions(game, player, legal, policy_choice, policy_fraction, raise_fractions, max(4, min(9, 3 + world_samples)), advantage_scores, policy_scores, raise_proposals)
    if len(decisions) < 2:
        return SearchResult(policy_choice, policy_fraction, 0, len(decisions), 0, 0.0, 0.0)
    belief = build_public_belief(game, player, range_bias, action_likelihood_model)

    def resolver_leaf(branch: HeadsUpHoldem, focal_player: int) -> float:
        fallback = value_leaf(branch, focal_player)
        if counterfactual_leaf is None:
            return fallback
        branch_belief = build_public_belief(branch, focal_player, range_bias, action_likelihood_model)
        counterfactual_value, counterfactual_uncertainty = counterfactual_leaf(branch, focal_player, branch_belief)
        blend = max(0.10, min(0.55, 0.12 + 0.43 * (1 - branch_belief.entropy) * (1 - branch_belief.effective_support)))
        confidence = 1 / (1 + max(0.0, counterfactual_uncertainty))
        blend *= confidence
        return (1 - blend) * fallback + blend * max(-MAX_APPROXIMATE_VALUE_BB, min(MAX_APPROXIMATE_VALUE_BB, counterfactual_value))

    samples = {decision: [] for decision in decisions}
    leaf_evaluations = 0
    for _ in range(world_samples):
        world = _sample_belief_world(game, player, belief, rng)
        for decision in decisions:
            branch = copy.deepcopy(world)
            execute_action(branch, player, decision.action, decision.raise_fraction)
            value, leaves = _finish_counterfactual(branch, player, continuation, resolver_leaf, depth_limit)
            samples[decision].append(value)
            leaf_evaluations += leaves
    robust_values: dict[ActionChoice, float] = {}
    raw_means: dict[ActionChoice, float] = {}
    lower_bounds: dict[ActionChoice, float] = {}
    standard_errors: dict[ActionChoice, float] = {}
    risk_scale = 0.14 + 0.18 * belief.entropy + 0.13 * belief.effective_support + 0.10 * value_uncertainty
    confidence_scale = 0.90 + 0.45 * belief.entropy + 0.35 * belief.effective_support
    for action, outcomes in samples.items():
        mean = sum(outcomes) / len(outcomes)
        deviation = math.sqrt(sum((outcome - mean) ** 2 for outcome in outcomes) / max(1, len(outcomes) - 1))
        standard_error = deviation / math.sqrt(len(outcomes))
        uncertainty_floor = 0.035 + 0.035 * belief.entropy + 0.025 * value_uncertainty
        raw_means[action] = mean
        standard_errors[action] = standard_error
        lower_bounds[action] = mean - confidence_scale * standard_error - uncertainty_floor
        robust_values[action] = lower_bounds[action] - risk_scale * deviation
    resolver_strategy = _average_regret_strategy(robust_values, resolver_iterations)
    centre = sum(robust_values.values()) / len(robust_values)
    spread = math.sqrt(sum((value - centre) ** 2 for value in robust_values.values()) / len(robust_values)) + 1e-6
    max_policy = max(policy_scores[decision.action] for decision in decisions)
    positive_advantages = {decision: max(0.0, advantage_scores[decision.action]) for decision in decisions}
    advantage_total = sum(positive_advantages.values())
    combined: dict[ActionChoice, float] = {}
    for decision in decisions:
        robust_z = (robust_values[decision] - centre) / spread
        policy_prior = policy_scores[decision.action] - max_policy
        regret_prior = positive_advantages[decision] / advantage_total if advantage_total > 1e-8 else 1 / len(decisions)
        combined[decision] = 0.92 * robust_z + 0.38 * policy_prior + 0.38 * (regret_prior - 1 / len(decisions)) + 0.60 * math.log(resolver_strategy[decision] + 1e-8)
    blueprint_candidates = [decision for decision in decisions if decision.action == policy_choice]
    blueprint_decision = min(blueprint_candidates, key=lambda decision: abs((decision.raise_fraction or 0.5) - (policy_fraction or 0.5))) if blueprint_candidates else max(decisions, key=lambda decision: raw_means[decision])
    blueprint_lower_bound = lower_bounds[blueprint_decision]
    safety_margin = 0.06 + 0.07 * belief.entropy + 0.05 * value_uncertainty + 0.03 / math.sqrt(max(1, world_samples))
    safe_decisions = [decision for decision in decisions if lower_bounds[decision] >= blueprint_lower_bound - safety_margin]
    if blueprint_decision not in safe_decisions:
        safe_decisions.append(blueprint_decision)
    safety_rejections = len(decisions) - len(safe_decisions)
    choice = max(safe_decisions or decisions, key=lambda decision: combined[decision])
    value_spread = max(raw_means.values()) - min(raw_means.values())
    average_error = sum(standard_errors.values()) / max(1, len(standard_errors))
    confidence = max(0.0, min(1.0, (0.55 * (1 - belief.entropy) + 0.45 * (1 - belief.effective_support)) * (1 / (1 + value_uncertainty + average_error))))
    safety_confidence = max(0.0, min(1.0, confidence * (1 / (1 + average_error))))
    adaptive_raises = sum(decision.action in RAISE_ACTIONS for decision in decisions)
    endgame_worlds = world_samples if game.street >= 2 else 0
    return SearchResult(choice.action, choice.raise_fraction, depth_limit, len(decisions), leaf_evaluations, value_spread, confidence, robust_values[choice], adaptive_raises, endgame_worlds, safety_rejections, safety_margin, safety_confidence, len(safe_decisions), resolver_iterations, max(resolver_strategy.values()))


class PublicBeliefResolver:
    """Small public-range re-solver; it deliberately never looks at hidden opponent cards."""

    @staticmethod
    def resolve(game: HeadsUpHoldem, player: int, policy_scores: list[float], advantage_scores: list[float], legal: list[bool], range_bias: list[float], action_likelihood_model: ActionLikelihoodModel, value_uncertainty: float) -> tuple[int, int]:
        equity, uncertainty = estimate_range_equity(game, player, samples=12, range_bias=range_bias)
        belief = build_public_belief(game, player, range_bias, action_likelihood_model)
        raises = sum(event["player"] == (1 - player) and event["action"] == "raise" for event in game.public_actions)
        depth = min(3, 1 + game.street + int(raises > 0))
        scores = [policy + 0.22 * advantage for policy, advantage in zip(policy_scores, advantage_scores)]
        call_amount = game.to_call(player)
        scores[1] += (equity * (game.pot + call_amount) - call_amount) * 0.06
        for action, fraction in ((2, 0.5), (3, 2.0)):
            if legal[action]:
                future_value = (equity - 0.5 - (0.65 * uncertainty + 0.35 * belief.entropy + 0.20 * value_uncertainty) * 0.10 - raises * 0.015) * game.pot * fraction
                scores[action] += future_value * (0.035 * depth)
        for action, available in enumerate(legal):
            if not available:
                scores[action] = -1e9
        return max(range(len(scores)), key=lambda action: scores[action]), depth
