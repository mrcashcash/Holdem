"""Pure training objectives and population-safety rules.

Keeping these rules outside the trainer makes them independently testable and
prevents the 10k-line orchestration module from becoming the only place where
policy safety can be reasoned about.
"""

from __future__ import annotations

import math

from torch import Tensor


POPULATION_BEHAVIOR_MAX_DEGENERACY = 0.08


def hierarchical_range_objective(
    exact_loss: Tensor,
    coarse_loss: Tensor,
    *,
    exact_buckets: int,
    coarse_buckets: int,
    exact_weight: float = 0.15,
) -> Tensor:
    """Favor learnable rank/suitedness classes while retaining exact detail.

    Exact 1,326-combination classification is extremely sparse in self-play.
    Normalizing both cross-entropies and down-weighting the exact term prevents
    its near-random gradient from overwhelming the useful coarse posterior.
    """
    normalized_exact = exact_loss / max(1.0, math.log(float(exact_buckets)))
    normalized_coarse = coarse_loss / max(1.0, math.log(float(coarse_buckets)))
    return float(exact_weight) * normalized_exact + normalized_coarse


def tail_all_in_risk_loss(
    all_in_probabilities: Tensor,
    targets: Tensor,
    tail_weights: Tensor,
    *,
    baseline_weight: float,
) -> Tensor:
    """CVaR-style penalty for excess shove mass on costly trajectories."""
    excess_tail_weight = (tail_weights - float(baseline_weight)).clamp_min(0.0)
    if not bool((excess_tail_weight > 0).any()):
        return all_in_probabilities.new_zeros(())
    excess_probability = (all_in_probabilities - targets).clamp_min(0.0)
    return (excess_probability.square() * excess_tail_weight).sum() / excess_tail_weight.sum().clamp_min(1e-6)


def adversarial_tail_credit_weights(
    *,
    base_weight: float,
    tail_weight: float,
    reward_bb: float,
    large_loss_bb: float,
    advantages: list[float],
    streets: list[int],
    masks: list[list[bool]],
    decisions: int = 2,
) -> list[float]:
    """Assign CVaR emphasis to the last avoidable bad decisions in a loss.

    Weighting every action in a losing hand gives incorrect credit assignment:
    a river mistake can suppress a sound preflop action.  PPO already learns
    from the whole trajectory, so the extra tail emphasis is reserved for at
    most the last two negative-advantage decisions with a genuine alternative.
    """
    count = len(advantages)
    weights = [float(base_weight)] * count
    if reward_bb > -float(large_loss_bb) or tail_weight <= 1.0 or decisions <= 0:
        return weights
    eligible = [
        index
        for index in range(count - 1, -1, -1)
        if advantages[index] < 0.0 and index < len(masks) and sum(bool(value) for value in masks[index]) > 1
    ][:decisions]
    for rank, index in enumerate(eligible):
        street = streets[index] if index < len(streets) else 0
        recency = 1.0 if rank == 0 else 0.65
        street_credit = 0.82 + 0.06 * min(3, max(0, int(street)))
        focused_tail_weight = 1.0 + (float(tail_weight) - 1.0) * recency * street_credit
        weights[index] = float(base_weight) * focused_tail_weight
    return weights


def population_safety_score(member: dict) -> float:
    """Conservative EV score used only for population scheduling."""
    overall = float(member.get("bb_per_100", 0.0))
    adversarial = float(member.get("adversarial_bb_per_100", overall))
    preflop = float(member.get("preflop_worst_lcb_bb_per_100", overall))
    return 0.25 * overall + 0.45 * adversarial + 0.30 * preflop


def population_member_is_catastrophic(member: dict) -> bool:
    """Detect collapse even when one audit lane has not populated a metric.

    A severe overall loss is sufficient evidence by itself. For less extreme
    losses, require the zero-score signal plus either adversarial or preflop
    confirmation. This catches the observed -402 bb/100 member whose missing
    adversarial result was serialized as zero.
    """
    overall = float(member.get("bb_per_100", 0.0))
    adversarial = float(member.get("adversarial_bb_per_100", overall))
    preflop = float(member.get("preflop_worst_lcb_bb_per_100", overall))
    score = float(member.get("score", 0.5))
    severe_overall_collapse = overall <= -150.0
    confirmed_collapse = score <= 0.02 and overall <= -45.0 and (adversarial <= -45.0 or preflop <= -64.0)
    return severe_overall_collapse or confirmed_collapse


def population_member_is_trainable(member: dict, update: int) -> bool:
    """Exclude known collapses and members still in recovery cooldown."""
    return (
        not population_member_is_catastrophic(member)
        and float(member.get("behavior_degeneracy", 0.0)) <= POPULATION_BEHAVIOR_MAX_DEGENERACY
        and int(member.get("recovery_cooldown_until", 0)) <= int(update)
    )


def population_behavior_degeneracy(audit: dict) -> float:
    """Score current greedy fold/shove collapse independently of stale EV."""
    fold_rate = float(audit.get("fold_rate", 0.0))
    all_in_rate = float(audit.get("all_in_rate", 0.0))
    return max(0.0, fold_rate - 0.72) + max(0.0, all_in_rate - 0.35)


def population_behavior_is_safe(audit: dict) -> bool:
    """Require a policy to retain meaningful non-fold, non-shove action mass."""
    return population_behavior_degeneracy(audit) <= POPULATION_BEHAVIOR_MAX_DEGENERACY


def population_behavior_selection_index(members: list[dict], audits: list[dict], update: int) -> int:
    """Select a current, non-degenerate member for the expensive final audit."""
    if not members or len(members) != len(audits):
        raise ValueError("Population members and behavior audits must be non-empty and aligned.")
    trainable = [index for index, member in enumerate(members) if population_member_is_trainable(member, update)]
    behavior_safe = [index for index, audit in enumerate(audits) if population_behavior_is_safe(audit)]
    candidates = [index for index in trainable if index in behavior_safe] or behavior_safe or trainable or list(range(len(members)))
    return max(
        candidates,
        key=lambda index: (
            -population_behavior_degeneracy(audits[index]),
            population_safety_score(members[index]),
            float(members[index].get("score", 0.0)),
            -index,
        ),
    )
