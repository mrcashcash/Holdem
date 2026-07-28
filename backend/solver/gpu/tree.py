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
  HORIZON    — terminal for DEPTH-LIMITED trees (``end_street`` set): betting
               closed on ``end_street`` with neither player all-in; values are
               injected per iteration by an external evaluator (a CFV net or
               an oracle) instead of enumerating the remaining streets.

Action ids reuse the abstraction convention: 0 fold, 1 check/call, 2 all-in,
3+ raise sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.solver.gpu.action_profile import (
    SUPPORTED_PROFILES,
    UNIFORM_PROFILE,
    StructuralActionState,
    select_raise_fractions,
)

FOLD, CHECK_CALL, ALL_IN = 0, 1, 2

DECISION, STREET_END, FOLD_NODE, SHOWDOWN, HORIZON = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class GpuActionConfig:
    """Coarse betting menu for the compact blueprint."""

    preflop_fractions: tuple[float, ...] = (1.0,)
    postflop_fractions: tuple[float, ...] = (0.75,)
    max_raises_per_street: int = 3
    stack_bb: float = 50.0
    # Raise-or-fold preflop: the small blind's open-limp branch is not
    # generated at all (facing a raise, calling stays legal). House rule.
    no_limp: bool = False
    # Per-street override: preflop raise depth is cheap (the multiplicative
    # blowup lives postflop), so e.g. preflop cap 3 (open/3bet/sized-4bet,
    # 5bet = jam) can pair with postflop cap 2 on a small card. None = use
    # max_raises_per_street everywhere.
    preflop_raise_cap: int | None = None
    # Phase 3 keeps a global candidate pool for stable action ids, then exposes
    # only two or three sizes at each structural public state. Offline
    # local-solve overrides are embedded in the checkpoint config so serving
    # reconstructs the identical tree without an external profile file.
    action_profile: str = UNIFORM_PROFILE
    max_sized_raises_per_node: int = 3
    phase3_overrides: tuple[tuple[str, tuple[float, ...]], ...] = ()
    phase3_profile_sha256: str | None = None

    def __post_init__(self) -> None:
        preflop = tuple(float(value) for value in self.preflop_fractions)
        postflop = tuple(float(value) for value in self.postflop_fractions)
        overrides = tuple(
            (str(key), tuple(float(value) for value in fractions))
            for key, fractions in self.phase3_overrides
        )
        object.__setattr__(self, "preflop_fractions", preflop)
        object.__setattr__(self, "postflop_fractions", postflop)
        object.__setattr__(self, "phase3_overrides", overrides)
        if self.action_profile not in SUPPORTED_PROFILES:
            raise ValueError(f"unsupported action profile: {self.action_profile}")
        if self.max_sized_raises_per_node not in (2, 3):
            raise ValueError("max_sized_raises_per_node must be 2 or 3")
        for label, fractions in (("preflop", preflop), ("postflop", postflop)):
            # An EMPTY menu is legal: a push-fold tree has no sized raises at
            # all, only fold / check-call / all-in, and `num_actions` already
            # accounts for it (3 + 0). Rejecting empty broke
            # test_gpu_cfr.PushFoldConvergenceTests and the whole
            # test_gpu_exploit suite from the moment Phase 3 added this
            # validation. Non-positive sizes remain an error.
            if any(value <= 0 for value in fractions):
                raise ValueError(f"{label} fractions must be positive")
            if tuple(sorted(set(fractions))) != fractions:
                raise ValueError(f"{label} fractions must be sorted and unique")
        override_map = dict(overrides)
        if len(override_map) != len(overrides):
            raise ValueError("Phase 3 override keys must be unique")
        for key, fractions in overrides:
            if not 2 <= len(fractions) <= self.max_sized_raises_per_node:
                raise ValueError(f"Phase 3 override {key} must contain two or three sizes")
            allowed = set(preflop if key.startswith("s0|") else postflop)
            if any(value not in allowed for value in fractions):
                raise ValueError(f"Phase 3 override {key} contains a non-candidate fraction")

    def raise_cap(self, street: int) -> int:
        if street == 0 and self.preflop_raise_cap is not None:
            return self.preflop_raise_cap
        return self.max_raises_per_street

    def fractions(self, street: int) -> tuple[float, ...]:
        return self.preflop_fractions if street == 0 else self.postflop_fractions

    def raise_indices(
        self,
        street: int,
        actor: int,
        pot: float,
        to_call: float,
        stack_behind: float,
        raises: int,
    ) -> tuple[int, ...]:
        candidates = self.fractions(street)
        state = StructuralActionState(
            street=street,
            actor=actor,
            pot=pot,
            to_call=to_call,
            stack_behind=stack_behind,
            raises=raises,
        )
        selected = select_raise_fractions(
            self.action_profile,
            state,
            candidates,
            overrides=dict(self.phase3_overrides),
            max_sizes=self.max_sized_raises_per_node,
        )
        return tuple(candidates.index(fraction) for fraction in selected)

    @property
    def num_actions(self) -> int:
        return 3 + max(len(self.preflop_fractions), len(self.postflop_fractions))


@dataclass(frozen=True)
class BettingRootState:
    """Exact public betting state for a mid-street re-solving root.

    All chip values are expressed in big blinds and indexed by abstract seat
    (button/SB = 0, big blind = 1). Blueprint training never supplies this
    object; it exists for continual resolving after real, possibly off-tree,
    actions.
    """

    street: int
    to_act: int
    committed: tuple[float, float]
    street_commit: tuple[float, float]
    stacks: tuple[float, float]
    acted: tuple[bool, bool]
    raises: int
    last_increment: float

    def __post_init__(self) -> None:
        if self.street not in (0, 1, 2, 3):
            raise ValueError(f"invalid root street: {self.street}")
        if self.to_act not in (0, 1):
            raise ValueError(f"invalid root actor: {self.to_act}")
        for label, values in (
            ("committed", self.committed),
            ("street_commit", self.street_commit),
            ("stacks", self.stacks),
        ):
            if len(values) != 2 or any(float(value) < 0 for value in values):
                raise ValueError(f"{label} must contain two non-negative values")
        if len(self.acted) != 2:
            raise ValueError("acted must contain two flags")
        if self.raises < 0:
            raise ValueError("raises cannot be negative")
        if self.last_increment <= 0:
            raise ValueError("last_increment must be positive")


@dataclass
class _Builder:
    config: GpuActionConfig
    end_street: int | None = None  # depth limit: HORIZON instead of continuing
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
    """Flattened betting tree; all chip amounts in big blinds.

    With ``start_street``/``start_pot``/``start_stacks`` the tree roots at the
    beginning of a later street's betting (a re-solving subgame): both players
    have already matched ``start_pot / 2`` and the big blind (seat 1) acts
    first.
    """

    def __init__(
        self,
        config: GpuActionConfig | None = None,
        start_street: int = 0,
        start_pot: float | None = None,
        start_stacks: tuple[float, float] | None = None,
        end_street: int | None = None,
        root_state: BettingRootState | None = None,
    ) -> None:
        self.config = config or GpuActionConfig()
        self.start_street = root_state.street if root_state is not None else start_street
        self.end_street = end_street
        builder = _Builder(self.config, end_street=end_street)
        if root_state is not None:
            self.root = _enumerate(
                builder,
                self.config,
                street=root_state.street,
                to_act=root_state.to_act,
                committed=tuple(float(value) for value in root_state.committed),
                street_commit=tuple(float(value) for value in root_state.street_commit),
                stacks=tuple(float(value) for value in root_state.stacks),
                acted=tuple(bool(value) for value in root_state.acted),
                raises=int(root_state.raises),
                last_increment=float(root_state.last_increment),
            )
        elif start_street == 0:
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
        else:
            half_pot = (start_pot or 0.0) / 2.0
            stacks = start_stacks or (self.config.stack_bb - half_pot, self.config.stack_bb - half_pot)
            self.root = _enumerate(
                builder,
                self.config,
                street=start_street,
                to_act=1,  # big blind acts first postflop
                committed=(half_pot, half_pot),
                street_commit=(0.0, 0.0),
                stacks=stacks,
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
        decision_nodes = self.decision_nodes()
        sized_counts = self.legal[decision_nodes, 3:].sum(axis=1)
        return {
            "nodes": int(len(self.kind)),
            "decisions": int((self.kind == DECISION).sum()),
            "showdowns": int((self.kind == SHOWDOWN).sum()),
            "folds": int((self.kind == FOLD_NODE).sum()),
            "street_ends": int((self.kind == STREET_END).sum()),
            "num_actions": self.config.num_actions,
            "action_profile": self.config.action_profile,
            "sized_raise_nodes": {
                str(count): int((sized_counts == count).sum())
                for count in sorted(set(int(value) for value in sized_counts))
            },
        }


def _menu(
    config: GpuActionConfig,
    street: int,
    actor: int,
    pot: float,
    to_call: float,
    stack_behind: float,
    raises: int,
) -> list[int]:
    actions: list[int] = []
    if to_call > 0:
        actions.append(FOLD)
    # no_limp: the preflop open (street 0, no raise yet, SB owes the blind
    # difference) offers raise-or-fold only — the limp branch never exists.
    if not (config.no_limp and street == 0 and raises == 0 and to_call > 0):
        actions.append(CHECK_CALL)
    if stack_behind <= to_call:
        if CHECK_CALL not in actions:
            actions.append(CHECK_CALL)  # calling an all-in is never a limp
        return actions
    if raises < config.raise_cap(street):
        fractions = config.fractions(street)
        for index in config.raise_indices(
            street, actor, pot, to_call, stack_behind, raises
        ):
            fraction = fractions[index]
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
    for action in _menu(config, street, to_act, pot, to_call, stacks[to_act], raises):
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
    if builder.end_street is not None and street >= builder.end_street:
        # Depth limit: neither player all-in, betting closed on the limit
        # street. An external evaluator (CFV net / oracle) prices the node.
        node = builder.add(HORIZON, street)
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
