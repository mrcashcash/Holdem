"""Flat enumeration of the action-abstracted heads-up betting tree.

Chip amounts are deterministic functions of the action sequence, so the
public betting structure is a finite tree independent of cards. It is
enumerated once into numpy arrays that the vectorized CFR core walks in
topological order.

The GPU blueprint intentionally uses a coarse menu (docs/GPU_CFR_PLAN.md):
fewer raise sizes and a lower raise cap than the CPU blueprint keep the tree
in the 10^4-10^5 node range; play-time re-solving restores sizing richness.

Node kinds:
  DECISION   — ``actor`` acts; ``children[n, a]`` >= 0 for legal actions
  STREET_END — betting closed on a non-river street; single child continues
               on the next street (the board deal happens implicitly:
               bucket tensors switch to the next street's)
  FOLD       — terminal; ``fold_loser`` folded, winner takes ``matched_pot``
               ... the loser's committed total, to be precise
  SHOWDOWN   — terminal; winner takes ``matched_pot`` (min of commitments)

Action ids reuse the abstraction convention: 0 fold, 1 check/call, 2 all-in,
3+ raise sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FOLD, CHECK_CALL, ALL_IN = 0, 1, 2

DECISION, STREET_END, FOLD_NODE, SHOWDOWN = 0, 1, 2, 3


@dataclass(frozen=True)
class GpuActionConfig:
    """Coarse betting menu for the dense blueprint."""

    preflop_fractions: tuple[float, ...] = (1.0,)
    postflop_fractions: tuple[float, ...] = (0.75,)
    max_raises_per_street: int = 3
    stack_bb: float = 50.0

    def fractions(self, street: int) -> tuple[float, ...]:
        return self.preflop_fractions if street == 0 else self.postflop_fractions

    @property
    def num_actions(self) -> int:
        return 3 + max(len(self.preflop_fractions), len(self.postflop_fractions))


@dataclass
class _Builder:
    config: GpuActionConfig
    kind: list[int] = field(default_factory=list)
    street: list[int] = field(default_factory=list)
    actor: list[int] = field(default_factory=list)
    matched_pot: list[float] = field(default_factory=list)
    fold_loser: list[int] = field(default_factory=list)
    fold_loser_committed: list[float] = field(default_factory=list)
    children: list[list[int]] = field(default_factory=list)
    legal: list[list[bool]] = field(default_factory=list)

    def add(self, kind: int, street: int, actor: int = -1) -> int:
        self.kind.append(kind)
        self.street.append(street)
        self.actor.append(actor)
        self.matched_pot.append(0.0)
        self.fold_loser.append(-1)
        self.fold_loser_committed.append(0.0)
        self.children.append([-1] * self.config.num_actions)
        self.legal.append([False] * self.config.num_actions)
        return len(self.kind) - 1


class BettingTree:
    """Flattened betting tree; all chip amounts in big blinds."""

    def __init__(self, config: GpuActionConfig | None = None) -> None:
        self.config = config or GpuActionConfig()
        builder = _Builder(self.config)
        # Preflop: player 0 = button/SB (acts first), commits 0.5/1.0.
        self.root = _enumerate(
            builder,
            self.config,
            street=0,
            to_act=0,
            committed=(0.5, 1.0),
            street_commit=(0.5, 1.0),
            stacks=(self.config.stack_bb - 0.5, self.config.stack_bb - 1.0),
            acted=(False, False),
            raises=0,
            last_increment=1.0,
        )
        self.kind = np.asarray(builder.kind, dtype=np.int8)
        self.street = np.asarray(builder.street, dtype=np.int8)
        self.actor = np.asarray(builder.actor, dtype=np.int8)
        self.matched_pot = np.asarray(builder.matched_pot, dtype=np.float32)
        self.fold_loser = np.asarray(builder.fold_loser, dtype=np.int8)
        self.fold_loser_committed = np.asarray(builder.fold_loser_committed, dtype=np.float32)
        self.children = np.asarray(builder.children, dtype=np.int32)
        self.legal = np.asarray(builder.legal, dtype=bool)

    def __len__(self) -> int:
        return len(self.kind)

    def decision_nodes(self) -> np.ndarray:
        return np.flatnonzero(self.kind == DECISION)

    def describe(self) -> dict:
        return {
            "nodes": int(len(self.kind)),
            "decisions": int((self.kind == DECISION).sum()),
            "showdowns": int((self.kind == SHOWDOWN).sum()),
            "folds": int((self.kind == FOLD_NODE).sum()),
            "street_ends": int((self.kind == STREET_END).sum()),
            "num_actions": self.config.num_actions,
        }


def _menu(
    config: GpuActionConfig,
    street: int,
    pot: float,
    to_call: float,
    stack_behind: float,
    raises: int,
) -> list[int]:
    actions: list[int] = []
    if to_call > 0:
        actions.append(FOLD)
    actions.append(CHECK_CALL)
    if stack_behind <= to_call:
        return actions
    if raises < config.max_raises_per_street:
        for index, fraction in enumerate(config.fractions(street)):
            raise_by = fraction * (pot + to_call)
            if to_call + raise_by < stack_behind:
                actions.append(3 + index)
    actions.append(ALL_IN)
    return actions


def _enumerate(
    builder: _Builder,
    config: GpuActionConfig,
    street: int,
    to_act: int,
    committed: tuple[float, float],
    street_commit: tuple[float, float],
    stacks: tuple[float, float],
    acted: tuple[bool, bool],
    raises: int,
    last_increment: float,
) -> int:
    node = builder.add(DECISION, street, to_act)
    pot = committed[0] + committed[1]
    to_call = max(street_commit) - street_commit[to_act]
    for action in _menu(config, street, pot, to_call, stacks[to_act], raises):
        builder.legal[node][action] = True
        builder.children[node][action] = _apply(
            builder, config, action, street, to_act, committed, street_commit, stacks, acted, raises, last_increment
        )
    return node


def _apply(
    builder: _Builder,
    config: GpuActionConfig,
    action: int,
    street: int,
    actor: int,
    committed: tuple[float, float],
    street_commit: tuple[float, float],
    stacks: tuple[float, float],
    acted: tuple[bool, bool],
    raises: int,
    last_increment: float,
) -> int:
    if action == FOLD:
        node = builder.add(FOLD_NODE, street)
        builder.fold_loser[node] = actor
        builder.fold_loser_committed[node] = committed[actor]
        return node

    to_call = max(street_commit) - street_commit[actor]
    if action == CHECK_CALL:
        payment = min(to_call, stacks[actor])
    elif action == ALL_IN:
        payment = stacks[actor]
    else:
        fraction = config.fractions(street)[action - 3]
        raise_by = max(fraction * (committed[0] + committed[1] + to_call), last_increment, 1.0)
        payment = min(to_call + raise_by, stacks[actor])

    new_stacks = list(stacks)
    new_committed = list(committed)
    new_street_commit = list(street_commit)
    new_stacks[actor] -= payment
    new_committed[actor] += payment
    new_street_commit[actor] += payment
    increment = new_street_commit[actor] - max(street_commit)
    new_raises = raises
    new_increment = last_increment
    if increment > 0:
        new_raises += 1
        new_increment = max(increment, 1.0)
    new_acted = list(acted)
    new_acted[actor] = True

    opponent = 1 - actor
    opponent_owes = new_street_commit[actor] - new_street_commit[opponent]
    if opponent_owes > 0 and new_stacks[opponent] > 0:
        return _enumerate(
            builder, config, street, opponent,
            tuple(new_committed), tuple(new_street_commit), tuple(new_stacks),
            tuple(new_acted), new_raises, new_increment,
        )
    if opponent_owes <= 0 and not new_acted[opponent] and new_stacks[opponent] > 0:
        return _enumerate(
            builder, config, street, opponent,
            tuple(new_committed), tuple(new_street_commit), tuple(new_stacks),
            tuple(new_acted), new_raises, new_increment,
        )
    return _close_street(
        builder, config, street, tuple(new_committed), tuple(new_stacks)
    )


def _close_street(
    builder: _Builder,
    config: GpuActionConfig,
    street: int,
    committed: tuple[float, float],
    stacks: tuple[float, float],
) -> int:
    if street == 3 or min(stacks) <= 0:
        # River done, or someone is all-in: remaining streets have no
        # decisions, so the runout collapses straight to showdown.
        node = builder.add(SHOWDOWN, street)
        builder.matched_pot[node] = min(committed)
        return node
    node = builder.add(STREET_END, street)
    child = _enumerate(
        builder, config,
        street=street + 1,
        to_act=1,  # big blind acts first postflop
        committed=committed,
        street_commit=(0.0, 0.0),
        stacks=stacks,
        acted=(False, False),
        raises=0,
        last_increment=1.0,
    )
    builder.children[node][0] = child
    builder.legal[node][0] = True
    return node
