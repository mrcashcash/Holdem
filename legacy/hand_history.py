"""Opt-in, local-only teacher-data ingestion for Text Hold'em training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .rl_env import ACTION_CONTEXT_SIZE, ACTION_COUNT, OBSERVATION_SIZE, RANGE_BUCKETS


HAND_HISTORY_DIRECTORY = Path(__file__).parent / "data" / "hand_history"


@dataclass(frozen=True)
class TeacherAction:
    """One validated action target produced by an offline hand-history exporter."""

    observation: list[float]
    mask: list[bool]
    action: int
    return_value: float
    context: list[float] | None = None
    history: list[list[float]] | None = None
    range_class: int | None = None


@dataclass(frozen=True)
class HandHistoryReport:
    filename: str
    accepted: int
    rejected: int
    message: str

    def payload(self) -> dict[str, int | str]:
        return {
            "filename": self.filename,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "message": self.message,
        }


def _safe_path(filename: str) -> Path:
    requested = Path(filename)
    if requested.name != filename or requested.suffix.lower() != ".jsonl":
        raise ValueError("Use a .jsonl filename placed in backend/data/hand_history.")
    HAND_HISTORY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return HAND_HISTORY_DIRECTORY / requested.name


def _numeric_vector(value: object, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def load_teacher_actions(filename: str, maximum_records: int = 20_000) -> tuple[list[TeacherAction], HandHistoryReport]:
    """Load validated, training-ready JSONL records without reading arbitrary paths.

    Each line needs ``observation``, ``mask``, ``action``, and ``return_value``.
    Optional ``context``, ``history``, and ``range_class`` seed the public
    action-likelihood model as well. The schema keeps imported data explicit:
    raw third-party hand histories are not silently guessed or reinterpreted.
    """

    path = _safe_path(filename)
    if not path.is_file():
        raise ValueError(f"No local teacher-data file named {path.name}.")
    accepted: list[TeacherAction] = []
    rejected = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if len(accepted) >= maximum_records:
                break
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record must be an object")
                observation = _numeric_vector(payload.get("observation"), OBSERVATION_SIZE)
                raw_mask = payload.get("mask")
                action = int(payload.get("action"))
                return_value = float(payload.get("return_value"))
                if observation is None or not isinstance(raw_mask, list) or len(raw_mask) != ACTION_COUNT:
                    raise ValueError("invalid policy target")
                mask = [bool(item) for item in raw_mask]
                if not 0 <= action < ACTION_COUNT or not mask[action] or not -200.0 <= return_value <= 200.0:
                    raise ValueError("illegal action or value")
                context = _numeric_vector(payload.get("context"), ACTION_CONTEXT_SIZE)
                raw_history = payload.get("history")
                history = None
                if isinstance(raw_history, list) and context is not None:
                    history = [_numeric_vector(item, ACTION_CONTEXT_SIZE) for item in raw_history[-12:]]
                    if any(item is None for item in history):
                        history = None
                range_class_value = payload.get("range_class")
                range_class = int(range_class_value) if range_class_value is not None else None
                if range_class is not None and not 0 <= range_class < RANGE_BUCKETS:
                    range_class = None
                accepted.append(TeacherAction(observation, mask, action, return_value, context, history, range_class))
            except (TypeError, ValueError, json.JSONDecodeError):
                rejected += 1
    report = HandHistoryReport(path.name, len(accepted), rejected, "ready" if accepted else "no valid teacher targets")
    return accepted, report
