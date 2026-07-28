"""Static state-dependent action abstraction for blueprint-v3.

The public betting tree is card-independent, so board-specific menus cannot be
swapped in during CFR without enumerating the full action superset. Phase 3
instead compiles richer local-solve measurements into a fixed menu for each
structural public-state class. The classifier uses street, position, SPR, pot,
raise number, and facing-bet pressure at runtime; board texture, range
advantage, and nut advantage are retained in the offline samples that choose
each class's two or three sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROFILE_VERSION = 1
UNIFORM_PROFILE = "uniform-v1"
PHASE3_STATIC_PROFILE = "phase3-static-v1"
SUPPORTED_PROFILES = (UNIFORM_PROFILE, PHASE3_STATIC_PROFILE)


@dataclass(frozen=True)
class StructuralActionState:
    """Card-independent features available while the betting tree is built."""

    street: int
    actor: int
    pot: float
    to_call: float
    stack_behind: float
    raises: int

    @property
    def spr(self) -> float:
        return max(self.stack_behind - self.to_call, 0.0) / max(
            self.pot + self.to_call, 1.0
        )

    @property
    def position(self) -> str:
        # Abstract seat 0 is the button: first to act preflop, in position
        # postflop. Seat 1 is the big blind/out-of-position player.
        return "btn" if self.actor == 0 else "oop"

    @property
    def spr_band(self) -> str:
        value = self.spr
        if value < 1.5:
            return "spr0"
        if value < 4.0:
            return "spr1"
        if value < 10.0:
            return "spr2"
        return "spr3"

    @property
    def pot_band(self) -> str:
        if self.pot < 6.0:
            return "pot0"
        if self.pot < 20.0:
            return "pot1"
        if self.pot < 60.0:
            return "pot2"
        return "pot3"

    @property
    def pressure_band(self) -> str:
        if self.to_call <= 0:
            return "free"
        pressure = self.to_call / max(self.pot + self.to_call, 1.0)
        if pressure < 0.25:
            return "small"
        if pressure < 0.60:
            return "medium"
        return "large"

    @property
    def raise_band(self) -> str:
        return f"r{min(max(int(self.raises), 0), 2)}"

    def key(self) -> str:
        return "|".join(
            (
                f"s{int(self.street)}",
                self.position,
                self.spr_band,
                self.pot_band,
                self.raise_band,
                self.pressure_band,
            )
        )


def _nearest_available(
    requested: Sequence[float],
    candidates: Sequence[float],
    limit: int,
) -> tuple[float, ...]:
    """Map a policy menu onto the configured candidate pool without duplicates."""
    available = tuple(float(value) for value in candidates)
    selected: list[float] = []
    for target in requested:
        if not available:
            break
        value = min(available, key=lambda candidate: (abs(candidate - target), candidate))
        if value not in selected:
            selected.append(value)
        if len(selected) >= limit:
            break
    if len(selected) < min(2, len(available)):
        for value in available:
            if value not in selected:
                selected.append(value)
            if len(selected) >= min(2, len(available)):
                break
    return tuple(sorted(selected))


def default_phase3_menu(
    state: StructuralActionState,
    candidates: Sequence[float],
    max_sizes: int = 3,
) -> tuple[float, ...]:
    """Deterministic seed profile used before local-EV overrides are compiled."""
    if state.street == 0:
        if state.raises == 0:
            # Keep two preflop sizes: with a four-raise cap, a third branch at
            # every level multiplies the number of postflop tree copies.
            requested = (0.75, 1.5)
        elif state.raises == 1:
            requested = (0.5, 2.25)
        elif state.spr < 4.0 or state.pressure_band == "large":
            requested = (0.5, 1.0)
        else:
            requested = (0.5, 1.5)
        return _nearest_available(requested, candidates, max_sizes)

    if state.spr < 1.5:
        requested = (0.33, 0.75)
    elif state.raises >= 1:
        requested = (
            (0.5, 1.0)
            if state.pressure_band in ("medium", "large")
            else (0.33, 0.75, 1.5)
        )
    elif state.to_call > 0:
        requested = (
            (0.5, 1.0)
            if state.pressure_band == "large"
            else (0.33, 0.75, 1.5)
        )
    elif state.position == "oop":
        requested = (
            (0.33, 0.75, 1.5)
            if state.spr_band in ("spr2", "spr3")
            else (0.33, 0.75, 1.0)
        )
    else:
        requested = (
            (0.25, 0.75, 1.5)
            if state.spr_band in ("spr2", "spr3")
            else (0.33, 0.75, 1.0)
        )
    return _nearest_available(requested, candidates, max_sizes)


def select_raise_fractions(
    profile: str,
    state: StructuralActionState,
    candidates: Sequence[float],
    overrides: Mapping[str, Sequence[float]] | None = None,
    max_sizes: int = 3,
) -> tuple[float, ...]:
    if profile == UNIFORM_PROFILE:
        return tuple(float(value) for value in candidates)
    if profile != PHASE3_STATIC_PROFILE:
        raise ValueError(f"unsupported action profile: {profile}")
    if overrides and state.key() in overrides:
        return _nearest_available(overrides[state.key()], candidates, max_sizes)
    return default_phase3_menu(state, candidates, max_sizes=max_sizes)


def _sample_state(sample: Mapping[str, object]) -> StructuralActionState:
    return StructuralActionState(
        street=int(sample["street"]),
        actor=int(sample["actor"]),
        pot=float(sample["pot"]),
        to_call=float(sample["to_call"]),
        stack_behind=float(sample["stack_behind"]),
        raises=int(sample["raises"]),
    )


def _candidate_evs(sample: Mapping[str, object]) -> dict[float, float]:
    raw = sample.get("candidate_evs")
    if not isinstance(raw, Mapping):
        raise ValueError("every action sample needs a candidate_evs object")
    result = {
        float(fraction): float(value)
        for fraction, value in raw.items()
        if math.isfinite(float(value))
    }
    if len(result) < 2:
        raise ValueError("every action sample needs at least two finite candidate EVs")
    return result


def _subset_loss(records: Sequence[tuple[float, dict[float, float]]], subset: set[float]) -> float:
    loss = 0.0
    for weight, values in records:
        usable = [values[fraction] for fraction in subset if fraction in values]
        if not usable:
            continue
        loss += weight * (max(values.values()) - max(usable))
    return loss


def compile_action_profile(
    samples: Iterable[Mapping[str, object]],
    *,
    max_sizes: int = 3,
    min_third_gain_bb: float = 0.01,
    source_sha256: str | None = None,
) -> dict:
    """Greedily retain the sizes with the largest reach-weighted marginal EV.

    A local-solve record may additionally contain ``board_texture``,
    ``range_advantage``, and ``nut_advantage``. They are not runtime tree keys;
    keeping each observation separate here makes strategically different
    boards contribute independently to the structural class's marginal-EV
    objective.
    """
    if max_sizes not in (2, 3):
        raise ValueError("Phase 3 supports exactly two or three sized raises per node")
    grouped: dict[str, list[tuple[float, dict[float, float]]]] = {}
    strategic_coverage: dict[str, set[tuple[str, str, str]]] = {}
    count = 0
    for sample in samples:
        state = _sample_state(sample)
        values = _candidate_evs(sample)
        weight = max(float(sample.get("reach", 1.0)), 0.0)
        if weight <= 0:
            continue
        key = state.key()
        grouped.setdefault(key, []).append((weight, values))
        strategic_coverage.setdefault(key, set()).add(
            (
                str(sample.get("board_texture", "unknown")),
                str(sample.get("range_advantage", "unknown")),
                str(sample.get("nut_advantage", "unknown")),
            )
        )
        count += 1

    rules: list[dict] = []
    for key in sorted(grouped):
        records = grouped[key]
        candidates = sorted(set.intersection(*(set(values) for _, values in records)))
        if len(candidates) < 2:
            continue
        selected: set[float] = set()
        current_loss = _subset_loss(records, set(candidates))
        total_weight = sum(weight for weight, _ in records)
        while len(selected) < min(max_sizes, len(candidates)):
            options = []
            for candidate in candidates:
                if candidate in selected:
                    continue
                subset = selected | {candidate}
                options.append((_subset_loss(records, subset), candidate))
            next_loss, candidate = min(options, key=lambda item: (item[0], item[1]))
            gain = current_loss - next_loss if selected else float("inf")
            if (
                len(selected) >= 2
                and gain / max(total_weight, 1e-30) < min_third_gain_bb
            ):
                break
            selected.add(candidate)
            current_loss = next_loss
        rules.append(
            {
                "key": key,
                "fractions": sorted(selected),
                "samples": len(records),
                "weighted_reach": round(total_weight, 9),
                "strategic_contexts": len(strategic_coverage[key]),
                "residual_ev_loss_bb": round(current_loss / max(total_weight, 1e-30), 9),
            }
        )

    return {
        "version": PROFILE_VERSION,
        "kind": PHASE3_STATIC_PROFILE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": source_sha256,
        "sample_count": count,
        "class_count": len(rules),
        "max_sizes": max_sizes,
        "min_third_gain_bb": min_third_gain_bb,
        "strategic_dimensions": [
            "board_texture",
            "range_advantage",
            "nut_advantage",
        ],
        "rules": rules,
    }


def load_compiled_profile(path: Path) -> tuple[tuple[tuple[str, tuple[float, ...]], ...], str]:
    encoded = path.read_bytes()
    profile = json.loads(encoded)
    if int(profile.get("version", -1)) != PROFILE_VERSION:
        raise ValueError(f"unsupported Phase 3 profile version: {profile.get('version')}")
    if profile.get("kind") != PHASE3_STATIC_PROFILE:
        raise ValueError(f"not a Phase 3 static profile: {profile.get('kind')}")
    rules = tuple(
        (str(rule["key"]), tuple(float(value) for value in rule["fractions"]))
        for rule in profile.get("rules", [])
    )
    return rules, hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile local-solve EV samples into a Phase 3 action profile")
    parser.add_argument("--samples", type=Path, required=True, help="JSONL local-solve action samples")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sizes", type=int, choices=(2, 3), default=3)
    parser.add_argument("--min-third-gain-bb", type=float, default=0.01)
    arguments = parser.parse_args()

    encoded = arguments.samples.read_bytes()
    samples = [
        json.loads(line)
        for line in encoded.decode("utf-8").splitlines()
        if line.strip()
    ]
    profile = compile_action_profile(
        samples,
        max_sizes=arguments.max_sizes,
        min_third_gain_bb=arguments.min_third_gain_bb,
        source_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)
    print(
        f"phase3 profile classes={profile['class_count']} samples={profile['sample_count']} "
        f"output={arguments.output.resolve()}"
    )


if __name__ == "__main__":
    main()
