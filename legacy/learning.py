"""Strategic recurrent PPO self-play with curriculum and an Elo-rated champion league."""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import random
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, current_thread
from time import perf_counter
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.distributions import Beta, Categorical

from .cfr_solver import ActionChoice, ActionLikelihoodMemory, ActionLikelihoodRecord, CFRRecord, ReservoirMemory, SearchValueMemory, SearchValueRecord, StrategyMemory, external_sample_record, robust_belief_search
from .abstract_solver import AbstractCfrOracle, AbstractTeacherMemory, AbstractTeacherRecord
from .abstract_cfr import HoldemAbstractionCfr, SolverTeacherRecord
from .benchmark_suite import kuhn_cfr_audit, score_blueprint
from .counterfactual_values import BELIEF_FEATURE_SIZE, BELIEF_VALUE_CLASSES, TWO_SIDED_BELIEF_FEATURE_SIZE, CounterfactualValueMemory, CounterfactualValueRecord, belief_features, private_belief_features
from .hand_history import HandHistoryReport, load_teacher_actions
from .poker import HeadsUpHoldem, new_deck
from .rl_env import ACTION_CONTEXT_SIZE, ACTION_COUNT, BOARD_CARD_FEATURE_SIZE, OBSERVATION_SIZE, PREFLOP_OPEN_RAISE_CAP_BB, PREFLOP_THREE_BET_POT_CAP_MULTIPLIER, PRIVATE_CARD_FEATURE_SIZE, PUBLIC_FEATURE_SIZE, RAISE_ACTIONS, RAISE_ACTION_COUNT, RANGE_BUCKETS, action_context_features, continuous_raise_target, event_context_features, execute_action, execute_actions_batch, hand_bucket, legal_action_mask, legal_action_masks_batch, normal_raise_bounds, observation
from .rollout_arena import ArenaHand, BatchedRolloutArena, active_rollout_capabilities
from .training_objectives import adversarial_tail_credit_weights, hierarchical_range_objective, population_behavior_degeneracy, population_behavior_is_safe, population_behavior_selection_index, population_member_is_catastrophic, population_member_is_trainable, population_safety_score, tail_all_in_risk_loss

MODEL_PATH = Path(__file__).parent / "data" / "strategic_champion.pt"
MODEL_BACKUP_PATH = MODEL_PATH.with_name(f"{MODEL_PATH.stem}.backup{MODEL_PATH.suffix}")
TRAINING_REPORT_DIRECTORY = MODEL_PATH.parent / "training_reports"
TRAINING_DEBUG_LOG_PATH = TRAINING_REPORT_DIRECTORY / "training-debug.jsonl"
_TRAINING_DEBUG_LOG_LOCK = RLock()
TRAINING_REPORT_SCHEMA_VERSION = 7
MODEL_VERSION = 26
COMPATIBLE_MODEL_VERSIONS = {25, MODEL_VERSION}
POLICY_EXECUTION_VERSION = 2
EMBEDDING_SIZE = 160
HIDDEN_SIZE = 256

# Process-local model reuse for the single CUDA rollout collector and the
# optional CUDA audit lane. Learner/target states load every update; frozen
# opponents may omit a state only after the worker acknowledges its exact
# revision. Missing or stale revisions fail before gameplay.
_INFERENCE_MODEL_CACHE: dict[str, PolicyValueNetwork] = {}
_INFERENCE_MODEL_REVISIONS: dict[str, str] = {}


class RolloutCacheMiss(RuntimeError):
    """Raised before gameplay when a worker lacks an omitted frozen model."""


class AuxiliaryUpdateRejected(RuntimeError):
    """Raised after an unsafe auxiliary step has been fully rolled back."""


def rollout_opponent_cache_key(entry: dict, fallback_index: int) -> str:
    """Scope a worker cache slot to both opponent identity and exact revision."""
    opponent_id = str(entry.get("id", fallback_index))
    revision = str(entry.get("state_revision", ""))
    return f"rollout-opponent-{opponent_id}:{revision}" if revision else f"rollout-opponent-{opponent_id}"


def retain_unique_entries_by_id(entries: list[dict], limit: int) -> list[dict]:
    """Keep the newest bounded entry for each logical policy identifier."""
    retained_reversed: list[dict] = []
    seen: set[str] = set()
    for entry in reversed(entries):
        entry_id = str(entry.get("id", ""))
        if not entry_id or entry_id in seen:
            continue
        seen.add(entry_id)
        retained_reversed.append(entry)
        if len(retained_reversed) >= limit:
            break
    return list(reversed(retained_reversed))


PARETO_SPECIALIST_METRICS = (
    "direct_lcb",
    "adversarial_lcb",
    "preflop_lcb",
    "holdout_lcb",
    "restricted_br",
    "audit_proxy",
)


def pareto_dominates(left: dict[str, float], right: dict[str, float]) -> bool:
    """Return whether left is no worse everywhere and better somewhere."""
    comparable = [metric for metric in PARETO_SPECIALIST_METRICS if metric in left and metric in right]
    return bool(comparable) and all(left[metric] >= right[metric] for metric in comparable) and any(left[metric] > right[metric] for metric in comparable)


def retain_pareto_specialists(entries: list[dict], limit: int = 8) -> list[dict]:
    """Keep a bounded non-dominated archive of rejected but useful policies."""
    unique = retain_unique_entries_by_id(entries, max(limit * 3, limit))
    retained = [
        entry
        for entry in unique
        if isinstance(entry.get("metrics"), dict)
        and not any(
            other is not entry and pareto_dominates(other["metrics"], entry["metrics"])
            for other in unique
            if isinstance(other.get("metrics"), dict)
        )
    ]
    if len(retained) <= limit:
        return retained
    def rank(entry: dict) -> tuple[float, int]:
        metrics = entry["metrics"]
        normalized_floor = min(float(metrics.get(metric, -1_000.0)) for metric in PARETO_SPECIALIST_METRICS)
        return normalized_floor, int(entry.get("updates", 0))
    return sorted(retained, key=rank, reverse=True)[:limit]


def log_training_debug(event: str, **details: Any) -> None:
    """Append one durable lifecycle event without affecting training on log failure."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        "pid": os.getpid(),
        "thread": current_thread().name,
        **details,
    }
    try:
        with _TRAINING_DEBUG_LOG_LOCK:
            TRAINING_REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            with TRAINING_DEBUG_LOG_PATH.open("a", encoding="utf-8", newline="\n") as log_file:
                json.dump(record, log_file, default=str, sort_keys=True)
                log_file.write("\n")
                log_file.flush()
                os.fsync(log_file.fileno())
    except Exception:
        pass
# The critic predicts a normalized BB return while PPO advantages continue to
# use the true terminal BB payoff. This preserves the objective without
# allowing deep-stack all-ins to destabilize the value head.
VALUE_RETURN_SCALE_BB = 20.0
VALUE_SUPPORT_MIN = -10.0
VALUE_SUPPORT_MAX = 10.0
VALUE_BINS = 41
VALUE_ENSEMBLE_SIZE = 3
PUBLIC_STATE_EXPERTS = 8
ROOT_CONDITIONED_OUTPUT_SIZE = ACTION_COUNT * 3 + VALUE_ENSEMBLE_SIZE + VALUE_ENSEMBLE_SIZE * VALUE_BINS + RAISE_ACTION_COUNT * 2
POLICY_HISTORY = 16
CURRICULUM_PHASES = (
    (2_500, 0, "Foundation 20 BB", 0.55),
    (12_500, 1, "Expansion 40 BB", 0.68),
    (50_000, 2, "Standard 100 BB", 0.80),
    (float("inf"), 3, "Deep-stack 200 BB", 0.90),
)
STAGE_STACKS = (400, 800, 2_000, 4_000)
BENCHMARK_STYLES = ("tight_aggressive", "loose_aggressive", "calling_station", "trapper", "pressure")
AUDIT_STYLES = ("nit", "maniac", "river_hunter")
HOLDOUT_SCENARIOS = (("short-stack", 0, 830_101, "short_pressure"), ("medium-stack", 1, 830_151, "balanced"), ("standard-stack", 2, 830_211, "draw_heavy"), ("deep-stack", 3, 830_251, "balanced"), ("paired-board", 2, 830_307, "paired"), ("monotone-board", 2, 830_401, "monotone"))
BLUEPRINT_AUDITS = (("nit", 930_101), ("maniac", 930_211), ("river_hunter", 930_307))
SCENARIO_PROFILES = ("balanced", "short_pressure", "draw_heavy", "paired", "monotone")
ENABLE_APPROXIMATE_RESOLVER = os.environ.get("HOLDEM_ENABLE_APPROXIMATE_RESOLVER", "1").strip().lower() in {"1", "true", "yes"}
ENABLE_HEURISTIC_ORACLE = os.environ.get("HOLDEM_ENABLE_HEURISTIC_ORACLE", "0").strip().lower() in {"1", "true", "yes"}
# Keep every independently measured threat eligible for bounded adversarial
# training.  The fixed audit exposed calling stations and river hunters as
# material leaks, so leaving them audit-only would make the curriculum chase
# stale proxy opponents instead of the observed losses.
ADVERSARIAL_TRAINING_STYLES = ("pressure", "loose_aggressive", "calling_station", "trapper", "maniac", "nit", "tight_aggressive", "river_hunter")
try:
    ADVERSARIAL_ROLLOUT_FRACTION = min(0.80, max(0.10, float(os.environ.get("HOLDEM_ADVERSARIAL_ROLLOUT_FRACTION", "0.40"))))
except ValueError:
    ADVERSARIAL_ROLLOUT_FRACTION = 0.40
try:
    ADVERSARIAL_FOCUS_SHARE = min(0.90, max(0.50, float(os.environ.get("HOLDEM_ADVERSARIAL_FOCUS_SHARE", "0.70"))))
except ValueError:
    ADVERSARIAL_FOCUS_SHARE = 0.70
try:
    ADVERSARIAL_ROTATION_SHARE = min(0.45, max(0.10, float(os.environ.get("HOLDEM_ADVERSARIAL_ROTATION_SHARE", "0.25"))))
except ValueError:
    ADVERSARIAL_ROTATION_SHARE = 0.25
ADVERSARIAL_FOCUS_COUNT = max(1, min(len(ADVERSARIAL_TRAINING_STYLES), int(os.environ.get("HOLDEM_ADVERSARIAL_FOCUS_COUNT", "2"))))
try:
    ADVERSARIAL_EVALUATION_HANDS = max(64, int(os.environ.get("HOLDEM_ADVERSARIAL_EVALUATION_HANDS", "256")))
except ValueError:
    ADVERSARIAL_EVALUATION_HANDS = 256
try:
    ADVERSARIAL_SCREENING_HANDS = max(128, int(os.environ.get("HOLDEM_ADVERSARIAL_SCREENING_HANDS", "512")))
except ValueError:
    ADVERSARIAL_SCREENING_HANDS = 512
try:
    ADVERSARIAL_CONFIRMATION_HANDS = max(ADVERSARIAL_SCREENING_HANDS, int(os.environ.get("HOLDEM_ADVERSARIAL_CONFIRMATION_HANDS", "2048")))
except ValueError:
    ADVERSARIAL_CONFIRMATION_HANDS = 2_048
try:
    LARGE_LOSS_BB = min(40.0, max(2.0, float(os.environ.get("HOLDEM_LARGE_LOSS_BB", "6.0"))))
except ValueError:
    LARGE_LOSS_BB = 6.0
try:
    ADVERSARIAL_TAIL_WEIGHT = min(2.5, max(1.0, float(os.environ.get("HOLDEM_ADVERSARIAL_TAIL_WEIGHT", "1.6"))))
except ValueError:
    ADVERSARIAL_TAIL_WEIGHT = 1.6
try:
    BEHAVIORAL_AUDIT_STATES = max(64, int(os.environ.get("HOLDEM_BEHAVIORAL_AUDIT_STATES", "192")))
except ValueError:
    BEHAVIORAL_AUDIT_STATES = 192
try:
    PREFLOP_SIZING_AUDIT_HANDS = max(64, int(os.environ.get("HOLDEM_PREFLOP_SIZING_AUDIT_HANDS", "128")))
except ValueError:
    PREFLOP_SIZING_AUDIT_HANDS = 128
try:
    PREFLOP_FORCED_ROOT_FRACTION = min(0.45, max(0.08, float(os.environ.get("HOLDEM_PREFLOP_FORCED_ROOT_FRACTION", "0.30"))))
except ValueError:
    PREFLOP_FORCED_ROOT_FRACTION = 0.30
try:
    PREFLOP_SCENARIO_AUDIT_HANDS = max(64, int(os.environ.get("HOLDEM_PREFLOP_SCENARIO_AUDIT_HANDS", "128")))
except ValueError:
    PREFLOP_SCENARIO_AUDIT_HANDS = 128
try:
    PREFLOP_FINAL_AUDIT_MULTIPLIER = min(8, max(1, int(os.environ.get("HOLDEM_PREFLOP_FINAL_AUDIT_MULTIPLIER", "4"))))
except ValueError:
    PREFLOP_FINAL_AUDIT_MULTIPLIER = 4
try:
    PREFLOP_ALLIN_CALIBRATION_WEIGHT = min(0.25, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_ALLIN_CALIBRATION_WEIGHT", "0.18"))))
except ValueError:
    PREFLOP_ALLIN_CALIBRATION_WEIGHT = 0.18
try:
    PREFLOP_ALLIN_STABILITY_WEIGHT = min(0.20, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_ALLIN_STABILITY_WEIGHT", "0.12"))))
except ValueError:
    PREFLOP_ALLIN_STABILITY_WEIGHT = 0.12
try:
    PREFLOP_ALLIN_RANKING_WEIGHT = min(0.30, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_ALLIN_RANKING_WEIGHT", "0.14"))))
except ValueError:
    PREFLOP_ALLIN_RANKING_WEIGHT = 0.14
try:
    PREFLOP_ALLIN_RANKING_MARGIN = min(0.50, max(0.02, float(os.environ.get("HOLDEM_PREFLOP_ALLIN_RANKING_MARGIN", "0.12"))))
except ValueError:
    PREFLOP_ALLIN_RANKING_MARGIN = 0.12
try:
    PREFLOP_ALLIN_COMMITTED_FRACTION = min(0.85, max(0.35, float(os.environ.get("HOLDEM_PREFLOP_ALLIN_COMMITTED_FRACTION", "0.50"))))
except ValueError:
    PREFLOP_ALLIN_COMMITTED_FRACTION = 0.50
try:
    PREFLOP_ALLIN_TARGET_MAX = min(0.40, max(0.05, float(os.environ.get("HOLDEM_PREFLOP_ALLIN_TARGET_MAX", "0.18"))))
except ValueError:
    PREFLOP_ALLIN_TARGET_MAX = 0.18
try:
    PREFLOP_ROOT_PROMOTION_LCB_FLOOR = min(-5.0, max(-150.0, float(os.environ.get("HOLDEM_PREFLOP_ROOT_PROMOTION_LCB_FLOOR", "-64"))))
except ValueError:
    PREFLOP_ROOT_PROMOTION_LCB_FLOOR = -64.0
try:
    PREFLOP_3BET_STYLE_FOCUS_SHARE = min(0.90, max(0.25, float(os.environ.get("HOLDEM_PREFLOP_3BET_STYLE_FOCUS_SHARE", "0.72"))))
except ValueError:
    PREFLOP_3BET_STYLE_FOCUS_SHARE = 0.72
try:
    PREFLOP_3BET_TEACHER_SAMPLE_PROBABILITY = min(0.95, max(0.05, float(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_SAMPLE_PROBABILITY", "0.90"))))
except ValueError:
    PREFLOP_3BET_TEACHER_SAMPLE_PROBABILITY = 0.90
try:
    PREFLOP_3BET_TEACHER_MAX_ROOTS = min(128, max(1, int(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_MAX_ROOTS", "128"))))
except ValueError:
    PREFLOP_3BET_TEACHER_MAX_ROOTS = 128
try:
    PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS = min(
        PREFLOP_3BET_TEACHER_MAX_ROOTS,
        max(0, int(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS", "32"))),
    )
except ValueError:
    PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS = min(PREFLOP_3BET_TEACHER_MAX_ROOTS, 32)
try:
    PREFLOP_3BET_TEACHER_FACING_4BET_MIN_ROOTS = min(
        PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS,
        max(0, int(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_FACING_4BET_MIN_ROOTS", "16"))),
    )
except ValueError:
    PREFLOP_3BET_TEACHER_FACING_4BET_MIN_ROOTS = min(PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS, 16)
PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS = ("facing_open_2bb", "facing_open_3bb", "facing_open_4bb", "facing_open_5bb")
try:
    PREFLOP_TEACHER_SHALLOW_OPEN_MIN_ROOTS = min(
        PREFLOP_3BET_TEACHER_MAX_ROOTS,
        max(0, int(os.environ.get("HOLDEM_PREFLOP_TEACHER_SHALLOW_OPEN_MIN_ROOTS", "24"))),
    )
except ValueError:
    PREFLOP_TEACHER_SHALLOW_OPEN_MIN_ROOTS = min(PREFLOP_3BET_TEACHER_MAX_ROOTS, 24)
try:
    PREFLOP_TEACHER_FOCUS_MIN_ROOTS = min(
        PREFLOP_3BET_TEACHER_MAX_ROOTS,
        max(0, int(os.environ.get("HOLDEM_PREFLOP_TEACHER_FOCUS_MIN_ROOTS", "48"))),
    )
except ValueError:
    PREFLOP_TEACHER_FOCUS_MIN_ROOTS = min(PREFLOP_3BET_TEACHER_MAX_ROOTS, 48)
try:
    PREFLOP_FOCUS_ROOT_WEIGHT_MULTIPLIER = min(
        4.0,
        max(1.0, float(os.environ.get("HOLDEM_PREFLOP_FOCUS_ROOT_WEIGHT_MULTIPLIER", "2.8"))),
    )
except ValueError:
    PREFLOP_FOCUS_ROOT_WEIGHT_MULTIPLIER = 2.8
try:
    PREFLOP_3BET_TEACHER_DEPTH = min(32, max(4, int(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_DEPTH", "14"))))
except ValueError:
    PREFLOP_3BET_TEACHER_DEPTH = 14
try:
    PREFLOP_3BET_TEACHER_WEIGHT = min(0.35, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_WEIGHT", "0.26"))))
except ValueError:
    PREFLOP_3BET_TEACHER_WEIGHT = 0.26
try:
    PREFLOP_TEACHER_FACING_4BET_WEIGHT_MULTIPLIER = min(3.0, max(1.0, float(os.environ.get("HOLDEM_PREFLOP_TEACHER_FACING_4BET_WEIGHT_MULTIPLIER", "1.8"))))
except ValueError:
    PREFLOP_TEACHER_FACING_4BET_WEIGHT_MULTIPLIER = 1.8
try:
    PREFLOP_3BET_TEACHER_TEMPERATURE_BB = min(8.0, max(0.25, float(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_TEMPERATURE_BB", "1.50"))))
except ValueError:
    PREFLOP_3BET_TEACHER_TEMPERATURE_BB = 1.50
try:
    PREFLOP_3BET_TEACHER_CONFIDENCE_BB = min(8.0, max(0.10, float(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_CONFIDENCE_BB", "1.25"))))
except ValueError:
    PREFLOP_3BET_TEACHER_CONFIDENCE_BB = 1.25
try:
    PREFLOP_3BET_TEACHER_WORLDS = min(8, max(2, int(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_WORLDS", "3"))))
except ValueError:
    PREFLOP_3BET_TEACHER_WORLDS = 3
PREFLOP_3BET_TEACHER_CONFIDENCE_Z = 1.2815515655446004
try:
    PREFLOP_3BET_TEACHER_MIN_CONFIDENCE = min(0.75, max(0.05, float(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_MIN_CONFIDENCE", "0.25"))))
except ValueError:
    PREFLOP_3BET_TEACHER_MIN_CONFIDENCE = 0.25
try:
    PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE = min(0.95, max(PREFLOP_3BET_TEACHER_MIN_CONFIDENCE, float(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE", "0.55"))))
except ValueError:
    PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE = 0.55
try:
    PREFLOP_3BET_TEACHER_ALLIN_DISADVANTAGE_WEIGHT = min(1.50, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_3BET_TEACHER_ALLIN_DISADVANTAGE_WEIGHT", "0.75"))))
except ValueError:
    PREFLOP_3BET_TEACHER_ALLIN_DISADVANTAGE_WEIGHT = 0.75
try:
    PREFLOP_TEACHER_ALLIN_CONTRASTIVE_WEIGHT = min(0.40, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_TEACHER_ALLIN_CONTRASTIVE_WEIGHT", "0.18"))))
except ValueError:
    PREFLOP_TEACHER_ALLIN_CONTRASTIVE_WEIGHT = 0.18
try:
    PREFLOP_TEACHER_FACING_4BET_CALL_CONTRASTIVE_WEIGHT = min(0.40, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_TEACHER_FACING_4BET_CALL_CONTRASTIVE_WEIGHT", "0.22"))))
except ValueError:
    PREFLOP_TEACHER_FACING_4BET_CALL_CONTRASTIVE_WEIGHT = 0.22
try:
    PREFLOP_TEACHER_SHALLOW_ALLIN_MARGIN_WEIGHT = min(0.40, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_TEACHER_SHALLOW_ALLIN_MARGIN_WEIGHT", "0.24"))))
except ValueError:
    PREFLOP_TEACHER_SHALLOW_ALLIN_MARGIN_WEIGHT = 0.24
try:
    PREFLOP_TEACHER_ACTION_MARGIN = min(1.00, max(0.02, float(os.environ.get("HOLDEM_PREFLOP_TEACHER_ACTION_MARGIN", "0.16"))))
except ValueError:
    PREFLOP_TEACHER_ACTION_MARGIN = 0.16
try:
    PPO_RANGE_LOSS_WEIGHT = min(0.20, max(0.0, float(os.environ.get("HOLDEM_PPO_RANGE_LOSS_WEIGHT", "0.08"))))
except ValueError:
    PPO_RANGE_LOSS_WEIGHT = 0.08
try:
    PREFLOP_TAIL_ALLIN_WEIGHT = min(0.50, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_TAIL_ALLIN_WEIGHT", "0.24"))))
except ValueError:
    PREFLOP_TAIL_ALLIN_WEIGHT = 0.24
RECOVERY_ANCHOR_REGRESSION_MARGIN = max(10.0, float(os.environ.get("HOLDEM_RECOVERY_ANCHOR_REGRESSION_MARGIN", "40")))
RECOVERY_ANCHOR_COOLDOWN_UPDATES = max(3, int(os.environ.get("HOLDEM_RECOVERY_ANCHOR_COOLDOWN_UPDATES", "18")))
RECOVERY_FULL_EVALUATION_INTERVAL = max(3, int(os.environ.get("HOLDEM_RECOVERY_FULL_EVALUATION_INTERVAL", "3")))
RECOVERY_ANCHOR_BOOTSTRAP_SCORE = float(os.environ.get("HOLDEM_RECOVERY_ANCHOR_BOOTSTRAP_SCORE", "-200"))
PREFLOP_ROOT_PROBE_HANDS = max(4, int(os.environ.get("HOLDEM_PREFLOP_ROOT_PROBE_HANDS", "12")))
PREFLOP_ROOT_UPDATE_KL_LIMIT = max(0.01, float(os.environ.get("HOLDEM_PREFLOP_ROOT_UPDATE_KL_LIMIT", "0.04")))
PREFLOP_ROOT_ANCHOR_KL_LIMIT = max(PREFLOP_ROOT_UPDATE_KL_LIMIT, float(os.environ.get("HOLDEM_PREFLOP_ROOT_ANCHOR_KL_LIMIT", "0.14")))
PREFLOP_ROOT_UPDATE_ACTION_DELTA_LIMIT = min(0.30, max(0.03, float(os.environ.get("HOLDEM_PREFLOP_ROOT_UPDATE_ACTION_DELTA_LIMIT", "0.08"))))
PREFLOP_ROOT_ANCHOR_ACTION_DELTA_LIMIT = min(0.45, max(PREFLOP_ROOT_UPDATE_ACTION_DELTA_LIMIT, float(os.environ.get("HOLDEM_PREFLOP_ROOT_ANCHOR_ACTION_DELTA_LIMIT", "0.18"))))
try:
    PREFLOP_ROOT_FOLD_COLLAPSE_RATE = min(0.999, max(0.80, float(os.environ.get("HOLDEM_PREFLOP_ROOT_FOLD_COLLAPSE_RATE", "0.98"))))
except ValueError:
    PREFLOP_ROOT_FOLD_COLLAPSE_RATE = 0.98
try:
    PREFLOP_TEACHER_FOLD_CONTRASTIVE_WEIGHT = min(0.35, max(0.0, float(os.environ.get("HOLDEM_PREFLOP_TEACHER_FOLD_CONTRASTIVE_WEIGHT", "0.22"))))
except ValueError:
    PREFLOP_TEACHER_FOLD_CONTRASTIVE_WEIGHT = 0.22
try:
    ROBUST_STYLE_POLICY_WEIGHT = min(0.45, max(0.0, float(os.environ.get("HOLDEM_ROBUST_STYLE_POLICY_WEIGHT", "0.18"))))
except ValueError:
    ROBUST_STYLE_POLICY_WEIGHT = 0.18
try:
    POPULATION_SPECIALIST_MIN_UPDATES = max(12, int(os.environ.get("HOLDEM_POPULATION_SPECIALIST_MIN_UPDATES", "36")))
except ValueError:
    POPULATION_SPECIALIST_MIN_UPDATES = 36
try:
    EXPLOITER_REFRESH_UPDATES = max(18, int(os.environ.get("HOLDEM_EXPLOITER_REFRESH_UPDATES", "54")))
except ValueError:
    EXPLOITER_REFRESH_UPDATES = 54
ROLLOUT_INFERENCE_DEVICE = os.environ.get("HOLDEM_ROLLOUT_INFERENCE_DEVICE", "auto").strip().lower()
if ROLLOUT_INFERENCE_DEVICE not in {"auto", "cpu", "cuda"}:
    ROLLOUT_INFERENCE_DEVICE = "auto"
try:
    ROLLING_DIAGNOSTIC_UPDATES = min(128, max(4, int(os.environ.get("HOLDEM_ROLLING_DIAGNOSTIC_UPDATES", "32"))))
except ValueError:
    ROLLING_DIAGNOSTIC_UPDATES = 32
try:
    ADVERSARIAL_PROMOTION_LCB_FLOOR = min(0.0, max(-100.0, float(os.environ.get("HOLDEM_ADVERSARIAL_PROMOTION_LCB_FLOOR", "-32"))))
except ValueError:
    ADVERSARIAL_PROMOTION_LCB_FLOOR = -32.0
try:
    SNAPSHOT_MIN_DISTANCE = min(0.10, max(0.00001, float(os.environ.get("HOLDEM_SNAPSHOT_MIN_DISTANCE", "0.001"))))
except ValueError:
    SNAPSHOT_MIN_DISTANCE = 0.001
try:
    BLUEPRINT_PROMOTION_CONFIDENCE = min(0.75, max(0.45, float(os.environ.get("HOLDEM_BLUEPRINT_PROMOTION_CONFIDENCE", "0.50"))))
    BLUEPRINT_PROMOTION_FLOOR = min(0.75, max(0.40, float(os.environ.get("HOLDEM_BLUEPRINT_PROMOTION_FLOOR", "0.48"))))
except ValueError:
    BLUEPRINT_PROMOTION_CONFIDENCE = 0.50
    BLUEPRINT_PROMOTION_FLOOR = 0.48
PPO_MAX_EPOCHS = max(1, int(os.environ.get("HOLDEM_PPO_MAX_EPOCHS", "4")))
try:
    # A rollout is normally 384 hands on the CUDA collector. Keeping that
    # rollout together makes every KL decision representative of the policy
    # update that will actually be retained, instead of rolling back several
    # safe 128-hand AdamW steps when a later shard crosses the guard.
    PPO_MINIBATCH_HANDS = min(2_048, max(64, int(os.environ.get("HOLDEM_PPO_MINIBATCH_HANDS", "512"))))
except ValueError:
    PPO_MINIBATCH_HANDS = 512
try:
    PPO_MIN_FINAL_ROLLOUT_HANDS = max(32, int(os.environ.get("HOLDEM_PPO_MIN_FINAL_ROLLOUT_HANDS", "64")))
except ValueError:
    PPO_MIN_FINAL_ROLLOUT_HANDS = 64
PPO_RECOVERY_EPOCHS = 1
try:
    PPO_HARD_KL_MULTIPLIER = min(2.0, max(1.05, float(os.environ.get("HOLDEM_PPO_HARD_KL_MULTIPLIER", "1.35"))))
except ValueError:
    PPO_HARD_KL_MULTIPLIER = 1.35
PPO_RECOVERY_UPDATES = max(1, int(os.environ.get("HOLDEM_PPO_RECOVERY_UPDATES", "2")))
try:
    PPO_POST_STEP_RETRY_SCALE = min(0.80, max(0.10, float(os.environ.get("HOLDEM_PPO_POST_STEP_RETRY_SCALE", "0.50"))))
except ValueError:
    PPO_POST_STEP_RETRY_SCALE = 0.50
# Fused AdamW is faster on some CUDA builds, but candidate validation recorded
# repeated non-finite post-step policies in the default environment. Keep the
# stable implementation as the default and require an explicit operator opt-in.
PPO_USE_FUSED_ADAMW = os.environ.get("HOLDEM_PPO_USE_FUSED_ADAMW", "0").strip().lower() in {"1", "true", "yes"}
ABSTRACT_CFR_TEACHER_MODE = os.environ.get("HOLDEM_ABSTRACT_CFR_TEACHER_MODE", "on").strip().lower()
if ABSTRACT_CFR_TEACHER_MODE not in {"on", "ablate"}:
    ABSTRACT_CFR_TEACHER_MODE = "ablate"
ENABLE_ABSTRACT_CFR_TEACHER = (
    ABSTRACT_CFR_TEACHER_MODE == "on"
    and os.environ.get("HOLDEM_ENABLE_ABSTRACT_CFR_TEACHER", "1").strip().lower() in {"1", "true", "yes"}
)
try:
    ABSTRACT_CFR_TEACHER_WEIGHT = min(0.10, max(0.0, float(os.environ.get("HOLDEM_ABSTRACT_CFR_TEACHER_WEIGHT", "0.025"))))
except ValueError:
    ABSTRACT_CFR_TEACHER_WEIGHT = 0.025
PROMOTION_MIN_HANDS = max(256, int(os.environ.get("HOLDEM_PROMOTION_MIN_HANDS", "2048")))
PROMOTION_MAX_HANDS = max(PROMOTION_MIN_HANDS, int(os.environ.get("HOLDEM_PROMOTION_MAX_HANDS", "4096")))
HOLDOUT_HANDS = max(128, int(os.environ.get("HOLDEM_HOLDOUT_HANDS", "512")))
HOLDOUT_CONFIRMATION_HANDS = max(HOLDOUT_HANDS, int(os.environ.get("HOLDEM_HOLDOUT_CONFIRMATION_HANDS", "4096")))
PREFLOP_ROOT_CONFIRMATION_HANDS = max(PREFLOP_SCENARIO_AUDIT_HANDS, int(os.environ.get("HOLDEM_PREFLOP_ROOT_CONFIRMATION_HANDS", "2048")))
SEQUENTIAL_CONFIRMATION_SCENARIOS = 2
# Full audits are deliberately less frequent than cheap screening.  The old
# variable remains a supported fallback for existing deployments.
EVALUATION_INTERVAL = max(3, int(os.environ.get("HOLDEM_FULL_EVALUATION_INTERVAL", os.environ.get("HOLDEM_EVALUATION_INTERVAL", "27"))))
try:
    EVALUATION_WORKERS = max(1, min(4, os.cpu_count() or 1, int(os.environ.get("HOLDEM_EVALUATION_WORKERS", "4"))))
except ValueError:
    EVALUATION_WORKERS = min(4, max(1, os.cpu_count() or 1))
CUDA_EVALUATION_ENABLED = os.environ.get("HOLDEM_CUDA_EVALUATION", "1").strip().lower() in {"1", "true", "yes"}
try:
    CHECKPOINT_INTERVAL_HANDS = max(0, int(os.environ.get("HOLDEM_CHECKPOINT_INTERVAL_HANDS", "20000")))
except ValueError:
    CHECKPOINT_INTERVAL_HANDS = 20_000


def bb_per_100_score(value: float, scale: float = 80.0) -> float:
    """Map chip EV to a bounded league value without reverting to hand-win rate."""
    return min(1.0, max(0.0, 0.5 + float(value) / max(1.0, 2.0 * scale)))


def bb_per_100_quality(value: float, floor: float, target: float) -> float:
    """Map a safety floor and target BB/100 result to a curriculum-quality score."""
    return min(1.0, max(0.0, (float(value) - floor) / max(1e-6, target - floor)))


def adversarial_tail_policy_weight(reward_bb: float) -> float:
    """Bound the extra on-policy emphasis for one adverse terminal return."""
    severity = min(1.0, max(0.0, (-float(reward_bb) - LARGE_LOSS_BB) / LARGE_LOSS_BB))
    return 1.0 + (ADVERSARIAL_TAIL_WEIGHT - 1.0) * severity


RANGE_COARSE_BUCKETS = 13 * 13 * 3


def _range_coarse_bucket_index(first: int, second: int) -> int:
    """Group exact card combinations by rank grid and pair/suited/offsuit class."""
    first_rank, second_rank = first // 4, second // 4
    high, low = max(first_rank, second_rank), min(first_rank, second_rank)
    suitedness = 0 if first_rank == second_rank else 1 if first % 4 == second % 4 else 2
    return (high * 13 + low) * 3 + suitedness


def _range_coarse_index_table() -> Tensor:
    table = [0] * RANGE_BUCKETS
    for first in range(52):
        for second in range(first + 1, 52):
            exact_bucket = first * (103 - first) // 2 + (second - first - 1)
            table[exact_bucket] = _range_coarse_bucket_index(first, second)
    return torch.tensor(table, dtype=torch.long)


RANGE_COARSE_INDEX = _range_coarse_index_table()


def range_coarse_probabilities(probabilities: Tensor, index: Tensor | None = None) -> Tensor:
    """Aggregate the exact 1,326-combination posterior into public hand classes."""
    index = RANGE_COARSE_INDEX if index is None else index
    if index.device != probabilities.device:
        index = index.to(probabilities.device)
    expanded = index.unsqueeze(0).expand(probabilities.size(0), -1)
    coarse = torch.zeros((probabilities.size(0), RANGE_COARSE_BUCKETS), dtype=probabilities.dtype, device=probabilities.device)
    return coarse.scatter_add(1, expanded, probabilities)


@dataclass(frozen=True)
class TrainingRuntime:
    """Main-process training configuration with an optional single CUDA rollout collector."""

    device: torch.device
    requested: str
    cuda_enabled: bool
    name: str
    total_memory_mb: int
    reason: str


def resolve_training_runtime() -> TrainingRuntime:
    requested = os.environ.get("HOLDEM_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        requested = "auto"
    if requested != "cpu":
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                properties = torch.cuda.get_device_properties(device)
                torch.cuda.set_device(device)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
                return TrainingRuntime(device, requested, True, properties.name, round(properties.total_memory / 1024**2), "CUDA PPO enabled; rollout collection can use one CUDA inference worker")
            reason = "CUDA was not detected by this PyTorch installation"
        except (AssertionError, RuntimeError) as exc:
            reason = f"CUDA setup unavailable: {exc}"
        if requested == "cuda":
            reason = f"CUDA requested, but unavailable. {reason}"
    else:
        reason = "CPU selected by HOLDEM_DEVICE"
    return TrainingRuntime(torch.device("cpu"), requested, False, "CPU", 0, reason)


class SuitEquivariantCardEncoder(nn.Module):
    """DeepSets-style suit encoder: suit permutations share policy/value features."""

    def __init__(self) -> None:
        super().__init__()
        self.suit_encoder = nn.Sequential(nn.Linear(26, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, 48), nn.GELU())
        self.projection = nn.Sequential(nn.Linear(74, 72), nn.LayerNorm(72), nn.Tanh())

    def forward(self, cards: Tensor) -> Tensor:
        grid = cards.reshape(cards.size(0), cards.size(1), 2, 13, 4)
        suit_tokens = grid.permute(0, 1, 4, 2, 3).reshape(cards.size(0), cards.size(1), 4, 26)
        pooled_suits = self.suit_encoder(suit_tokens).mean(dim=-2)
        rank_counts = grid.sum(dim=-1).reshape(cards.size(0), cards.size(1), 26)
        return self.projection(torch.cat((pooled_suits, rank_counts), dim=-1))


class PolicyValueNetwork(nn.Module):
    """Public/private trunk with public-state-conditioned residual experts.

    The shared trunk learns cards, ranges, and action history.  A small router
    then assigns each state to a soft mixture of residual heads.  This gives
    strategically different roots independent policy/value capacity without
    duplicating the expensive recurrent trunk.  Residual heads are initialized
    to zero, so migrating an older checkpoint preserves its exact outputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.public_encoder = nn.Sequential(nn.Linear(PUBLIC_FEATURE_SIZE, 120), nn.LayerNorm(120), nn.Tanh())
        self.private_encoder = SuitEquivariantCardEncoder()
        self.encoder = nn.Sequential(nn.Linear(192, EMBEDDING_SIZE), nn.LayerNorm(EMBEDDING_SIZE), nn.Tanh())
        self.gru = nn.GRU(EMBEDDING_SIZE, HIDDEN_SIZE, num_layers=2, dropout=0.08, batch_first=True)
        attention_layer = nn.TransformerEncoderLayer(HIDDEN_SIZE, nhead=4, dim_feedforward=HIDDEN_SIZE * 2, dropout=0.06, activation="gelu", batch_first=True, norm_first=True)
        self.sequence_attention = nn.TransformerEncoder(attention_layer, num_layers=1)
        self.street_adapters = nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.LayerNorm(HIDDEN_SIZE), nn.Tanh()) for _ in range(4)])
        self.policy = nn.Linear(HIDDEN_SIZE, ACTION_COUNT)
        self.average_strategy = nn.Linear(HIDDEN_SIZE, ACTION_COUNT)
        self.value = nn.Linear(HIDDEN_SIZE, VALUE_ENSEMBLE_SIZE)
        self.value_distribution = nn.Linear(HIDDEN_SIZE, VALUE_ENSEMBLE_SIZE * VALUE_BINS)
        self.public_state_router = nn.Sequential(
            nn.Linear(PUBLIC_FEATURE_SIZE, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, PUBLIC_STATE_EXPERTS),
        )
        self.public_state_residuals = nn.ModuleList(
            nn.Linear(HIDDEN_SIZE, ROOT_CONDITIONED_OUTPUT_SIZE)
            for _ in range(PUBLIC_STATE_EXPERTS)
        )
        for residual in self.public_state_residuals:
            nn.init.zeros_(residual.weight)
            nn.init.zeros_(residual.bias)
        # The policy/value path is suit-equivariant.  This raw blocker path is
        # retained solely for exact 1,326-combination range supervision.
        self.range_card_encoder = nn.Sequential(nn.Linear(PRIVATE_CARD_FEATURE_SIZE + BOARD_CARD_FEATURE_SIZE, 48), nn.LayerNorm(48), nn.Tanh())
        self.range_context = nn.Sequential(nn.Linear(HIDDEN_SIZE + 48, HIDDEN_SIZE), nn.LayerNorm(HIDDEN_SIZE), nn.Tanh())
        self.range_head = nn.Linear(HIDDEN_SIZE, RANGE_BUCKETS)
        self.advantage_head = nn.Linear(HIDDEN_SIZE, ACTION_COUNT)
        self.raise_shapes = nn.Linear(HIDDEN_SIZE, RAISE_ACTION_COUNT * 2)
        self.counterfactual_belief_encoder = nn.Sequential(nn.Linear(BELIEF_FEATURE_SIZE, 112), nn.LayerNorm(112), nn.Tanh())
        self.counterfactual_value_head = nn.Sequential(nn.Linear(HIDDEN_SIZE + 112, HIDDEN_SIZE), nn.LayerNorm(HIDDEN_SIZE), nn.Tanh(), nn.Linear(HIDDEN_SIZE, BELIEF_VALUE_CLASSES * 2))
        self.public_belief_encoder = nn.Sequential(nn.Linear(TWO_SIDED_BELIEF_FEATURE_SIZE, 160), nn.LayerNorm(160), nn.Tanh())
        self.public_belief_value_head = nn.Sequential(nn.Linear(HIDDEN_SIZE + 160, HIDDEN_SIZE), nn.LayerNorm(HIDDEN_SIZE), nn.Tanh(), nn.Linear(HIDDEN_SIZE, BELIEF_VALUE_CLASSES * 4))
        self.likelihood_encoder = nn.Sequential(nn.Linear(ACTION_CONTEXT_SIZE, 128), nn.LayerNorm(128), nn.Tanh())
        self.likelihood_gru = nn.GRU(128, 128, batch_first=True)
        self.action_likelihood_head = nn.Linear(128, RANGE_BUCKETS * ACTION_COUNT)

    def _conditioned_heads(self, inputs: Tensor, output: Tensor, *, detach_router: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply a soft public-state expert mixture to the deployable heads."""
        router_logits = self.public_state_router(inputs[..., :PUBLIC_FEATURE_SIZE])
        router_weights = router_logits.softmax(dim=-1)
        if detach_router:
            # Off-policy solver/value replay may specialize its own residual
            # rows, but cannot reroute the deployed PPO policy.
            router_weights = router_weights.detach()
        expert_outputs = torch.stack([expert(output) for expert in self.public_state_residuals], dim=-2)
        residual = (expert_outputs * router_weights.unsqueeze(-1)).sum(dim=-2)
        policy_residual, average_residual, value_residual, distribution_residual, advantage_residual, raise_residual = residual.split(
            (ACTION_COUNT, ACTION_COUNT, VALUE_ENSEMBLE_SIZE, VALUE_ENSEMBLE_SIZE * VALUE_BINS, ACTION_COUNT, RAISE_ACTION_COUNT * 2),
            dim=-1,
        )
        return (
            self.policy(output) + policy_residual,
            self.average_strategy(output) + average_residual,
            self.value(output) + value_residual,
            self.value_distribution(output) + distribution_residual,
            self.advantage_head(output) + advantage_residual,
            self.raise_shapes(output) + raise_residual,
        )

    def _encode_sequence(self, inputs: Tensor, hidden: Tensor | None = None, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        public = self.public_encoder(inputs[..., :PUBLIC_FEATURE_SIZE])
        raw_cards = inputs[..., PUBLIC_FEATURE_SIZE:PUBLIC_FEATURE_SIZE + PRIVATE_CARD_FEATURE_SIZE + BOARD_CARD_FEATURE_SIZE]
        private = self.private_encoder(raw_cards)
        encoded = self.encoder(torch.cat((public, private), dim=-1))
        output, next_hidden = self.gru(encoded, hidden)
        sequence_length = output.size(1)
        causal_mask = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=output.device).triu(1)
        output = self.sequence_attention(output, mask=causal_mask, src_key_padding_mask=padding_mask)
        street_ids = inputs[..., 5:9].argmax(dim=-1)
        street_outputs = torch.stack([adapter(output) for adapter in self.street_adapters], dim=2)
        street_index = street_ids.unsqueeze(-1).unsqueeze(-1).expand(*street_ids.shape, 1, HIDDEN_SIZE)
        output = output + street_outputs.gather(2, street_index).squeeze(2)
        return output, next_hidden

    def forward(self, inputs: Tensor, hidden: Tensor | None = None, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        output, next_hidden = self._encode_sequence(inputs, hidden, padding_mask)
        raw_cards = inputs[..., PUBLIC_FEATURE_SIZE:PUBLIC_FEATURE_SIZE + PRIVATE_CARD_FEATURE_SIZE + BOARD_CARD_FEATURE_SIZE]
        range_context = self.range_context(torch.cat((output, self.range_card_encoder(raw_cards)), dim=-1))
        policy, average, value, distribution_logits, advantage, raise_shapes = self._conditioned_heads(inputs, output)
        distribution = distribution_logits.view(*output.shape[:-1], VALUE_ENSEMBLE_SIZE, VALUE_BINS)
        return policy, average, value, self.range_head(range_context), advantage, distribution, raise_shapes, next_hidden

    def detached_critic(self, inputs: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Run value heads on frozen policy features for off-policy critic replay."""
        output = self.detached_features(inputs, padding_mask)
        _, _, value, distribution_logits, _, _ = self._conditioned_heads(inputs, output, detach_router=True)
        distribution = distribution_logits.view(*output.shape[:-1], VALUE_ENSEMBLE_SIZE, VALUE_BINS)
        return value, distribution

    def detached_features(self, inputs: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Encode replay states without allowing off-policy gradients into PPO."""
        with torch.no_grad():
            output, _ = self._encode_sequence(inputs, None, padding_mask)
        return output.detach()

    def detached_cfr_heads(self, inputs: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Train solver-specific heads while preserving the deployed policy trunk."""
        output = self.detached_features(inputs, padding_mask)
        _, average, _, _, advantage, raise_shapes = self._conditioned_heads(inputs, output, detach_router=True)
        return average, advantage, raise_shapes

    def detached_public_belief_values(self, inputs: Tensor, own_belief: Tensor, opponent_belief: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Fit solver value surfaces without moving the on-policy representation."""
        output = self.detached_features(inputs, padding_mask)
        if own_belief.dim() == 2:
            own_belief = own_belief.unsqueeze(1).expand(-1, inputs.size(1), -1)
        if opponent_belief.dim() == 2:
            opponent_belief = opponent_belief.unsqueeze(1).expand(-1, inputs.size(1), -1)
        encoded = self.public_belief_encoder(torch.cat((own_belief, opponent_belief), dim=-1))
        values, opponent_values, raw_uncertainty, raw_opponent_uncertainty = self.public_belief_value_head(torch.cat((output, encoded), dim=-1)).chunk(4, dim=-1)
        return values * VALUE_RETURN_SCALE_BB, (nn.functional.softplus(raw_uncertainty) + 0.04) * VALUE_RETURN_SCALE_BB, opponent_values * VALUE_RETURN_SCALE_BB, (nn.functional.softplus(raw_opponent_uncertainty) + 0.04) * VALUE_RETURN_SCALE_BB

    def counterfactual_values(self, inputs: Tensor, belief: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Predict compact range-class values and learned aleatoric uncertainty."""
        output, _ = self._encode_sequence(inputs, None, padding_mask)
        if belief.dim() == 2:
            belief = belief.unsqueeze(1).expand(-1, inputs.size(1), -1)
        encoded_belief = self.counterfactual_belief_encoder(belief)
        values, raw_uncertainty = self.counterfactual_value_head(torch.cat((output, encoded_belief), dim=-1)).chunk(2, dim=-1)
        return values * VALUE_RETURN_SCALE_BB, (nn.functional.softplus(raw_uncertainty) + 0.04) * VALUE_RETURN_SCALE_BB

    def public_belief_values(self, inputs: Tensor, own_belief: Tensor, opponent_belief: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict paired counterfactual surfaces from both players' range summaries."""
        output, _ = self._encode_sequence(inputs, None, padding_mask)
        if own_belief.dim() == 2:
            own_belief = own_belief.unsqueeze(1).expand(-1, inputs.size(1), -1)
        if opponent_belief.dim() == 2:
            opponent_belief = opponent_belief.unsqueeze(1).expand(-1, inputs.size(1), -1)
        encoded = self.public_belief_encoder(torch.cat((own_belief, opponent_belief), dim=-1))
        values, opponent_values, raw_uncertainty, raw_opponent_uncertainty = self.public_belief_value_head(torch.cat((output, encoded), dim=-1)).chunk(4, dim=-1)
        return values * VALUE_RETURN_SCALE_BB, (nn.functional.softplus(raw_uncertainty) + 0.04) * VALUE_RETURN_SCALE_BB, opponent_values * VALUE_RETURN_SCALE_BB, (nn.functional.softplus(raw_opponent_uncertainty) + 0.04) * VALUE_RETURN_SCALE_BB

    def action_likelihood_sequence_logits(self, contexts: Tensor) -> Tensor:
        encoded = self.likelihood_encoder(contexts)
        output, _ = self.likelihood_gru(encoded)
        batch, sequence, _ = output.shape
        return self.action_likelihood_head(output).view(batch, sequence, RANGE_BUCKETS, ACTION_COUNT)

    def action_likelihood_logits(self, contexts: Tensor) -> Tensor:
        return self.action_likelihood_sequence_logits(contexts.unsqueeze(1))[:, 0]


def cached_inference_model(cache_key: str, state: dict[str, Tensor] | None, device: torch.device, revision: str = "") -> PolicyValueNetwork:
    """Reuse an exact process-local model, failing closed on an absent revision."""
    key = f"{device.type}:{device.index}:{cache_key}"
    model = _INFERENCE_MODEL_CACHE.get(key)
    if model is None:
        if state is None:
            raise RolloutCacheMiss(f"Missing rollout model cache entry {cache_key!r} for revision {revision!r}")
        fork_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            model = PolicyValueNetwork().to(device)
        _INFERENCE_MODEL_CACHE[key] = model
    loaded_revision = _INFERENCE_MODEL_REVISIONS.get(key)
    if not revision or loaded_revision != revision:
        if state is None:
            raise RolloutCacheMiss(f"Stale rollout model cache entry {cache_key!r}: expected revision {revision!r}, found {loaded_revision!r}")
        model.load_state_dict(state)
        if revision:
            _INFERENCE_MODEL_REVISIONS[key] = revision
    model.eval()
    return model


def clone_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def migrate_policy_state_dict(
    state: dict[str, Tensor],
    template: dict[str, Tensor],
) -> tuple[dict[str, Tensor], bool]:
    """Return a shape-safe policy state, initializing newly added parameters.

    Existing tensors are retained by reference to avoid duplicating very large
    checkpoints in memory.  New tensors come from one common initialized
    template, which keeps every migrated policy's router identical while the
    zero residual heads preserve its exact pre-migration outputs.
    """
    migrated: dict[str, Tensor] = {}
    changed = set(state) != set(template)
    for name, default in template.items():
        value = state.get(name)
        if isinstance(value, Tensor) and value.shape == default.shape and value.dtype == default.dtype:
            migrated[name] = value
            continue
        if value is not None:
            changed = True
        migrated[name] = default.detach().cpu().clone()
        changed = True
    return migrated, changed


def migrate_checkpoint_policy_states(payload: dict[str, Any], template: dict[str, Tensor]) -> bool:
    """Migrate every frozen/trainable policy stored in a checkpoint payload."""
    changed = False

    def migrate_mapping(mapping: dict[str, Any], key: str) -> None:
        nonlocal changed
        state = mapping.get(key)
        if isinstance(state, dict):
            mapping[key], migrated = migrate_policy_state_dict(state, template)
            changed = changed or migrated

    for key in ("model", "target_model", "champion", "recovery_anchor_state", "recovery_anchor_target_state"):
        migrate_mapping(payload, key)
    for collection_key in ("league", "exploiters", "population_members", "strategy_snapshots", "specialist_archive"):
        collection = payload.get(collection_key, [])
        if not isinstance(collection, list):
            continue
        for entry in collection:
            if not isinstance(entry, dict):
                continue
            migrate_mapping(entry, "state")
            migrate_mapping(entry, "target_state")
            if changed and collection_key == "population_members":
                entry["optimizer_state"] = None
                entry["grad_scaler_state"] = None
    return changed


def tensors_are_finite(*tensors: Tensor) -> bool:
    """Return whether every candidate tensor is finite without hiding corruption."""
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def ppo_candidate_is_finite(policy_logits: Tensor, raise_shapes: Tensor) -> bool:
    """Validate both policy heads before a PPO candidate is allowed to continue."""
    return tensors_are_finite(policy_logits, raise_shapes)


def ppo_candidate_precision_state(amp_logits: Tensor, amp_raise_shapes: Tensor, full_precision_logits: Tensor, full_precision_raise_shapes: Tensor) -> str:
    """Classify an invalid AMP candidate without treating FP16 overflow as corrupt FP32 weights."""
    if ppo_candidate_is_finite(amp_logits, amp_raise_shapes):
        return "finite"
    return "amp_overflow" if ppo_candidate_is_finite(full_precision_logits, full_precision_raise_shapes) else "nonfinite"


def ppo_retry_is_acceptable(precision_state: str, retry_kl: float, hard_kl_limit: float) -> bool:
    """Whether a restored-state PPO backoff candidate can be retained."""
    return precision_state == "finite" and math.isfinite(retry_kl) and retry_kl <= hard_kl_limit


def ppo_root_drift_guard_reasons(
    update_drift: dict[str, Any],
    anchor_before: dict[str, Any],
    anchor_after: dict[str, Any],
    protected_roots: set[str] | None = None,
) -> list[str]:
    """Return root-drift violations without deadlocking recovery outside an anchor.

    Per-update limits are always absolute.  An anchor limit is also absolute while
    the starting policy is inside it.  If a restored checkpoint already starts
    outside an anchor limit, retain only steps that do not worsen that distance;
    otherwise every safe recovery step would be rolled back forever.
    """
    reasons: list[str] = []
    update_kl = float(update_drift["max_kl"])
    update_action_delta = float(update_drift["max_action_delta"])
    if update_kl > PREFLOP_ROOT_UPDATE_KL_LIMIT:
        reasons.append(f"update KL {update_kl:.4f} at {update_drift['max_kl_root']}")
    if update_action_delta > PREFLOP_ROOT_UPDATE_ACTION_DELTA_LIMIT:
        reasons.append(f"update action delta {update_action_delta:.3f} at {update_drift['max_action_delta_root']}")

    anchor_checks = (
        ("max_kl", "max_kl_root", PREFLOP_ROOT_ANCHOR_KL_LIMIT, "anchor KL", 4),
        ("max_action_delta", "max_action_delta_root", PREFLOP_ROOT_ANCHOR_ACTION_DELTA_LIMIT, "anchor action delta", 3),
    )
    for metric, root_metric, limit, label, precision in anchor_checks:
        before = float(anchor_before.get(metric, 0.0))
        after = float(anchor_after.get(metric, 0.0))
        root = str(anchor_after.get(root_metric, "pending"))
        if protected_roots is not None:
            before_by_root = anchor_before.get("per_root", {})
            after_by_root = anchor_after.get("per_root", {})
            if not isinstance(before_by_root, dict) or not isinstance(after_by_root, dict):
                continue
            metric_name = "kl" if metric == "max_kl" else "action_delta"
            candidates = [
                (
                    float(dict(before_by_root.get(protected_root, {})).get(metric_name, 0.0)),
                    float(dict(after_by_root.get(protected_root, {})).get(metric_name, 0.0)),
                    protected_root,
                )
                for protected_root in protected_roots
                if protected_root in after_by_root
            ]
            if not candidates:
                continue
            before, after, root = max(candidates, key=lambda item: item[1])
        crossed_from_safe = before <= limit < after
        worsened_while_unsafe = before > limit and after > before + 1e-6
        if crossed_from_safe or worsened_while_unsafe:
            reasons.append(f"{label} {after:.{precision}f} at {root}")
    return reasons


def interpolate_model_state(
    before: dict[str, Tensor],
    after: dict[str, Tensor],
    scale: float,
) -> dict[str, Tensor]:
    """Interpolate floating model state while preserving discrete buffers."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("Model-state interpolation scale must be between zero and one.")
    if before.keys() != after.keys():
        raise ValueError("Model states must contain the same keys.")
    blended: dict[str, Tensor] = {}
    for name, start in before.items():
        end = after[name]
        if start.shape != end.shape or start.dtype != end.dtype:
            raise ValueError(f"Model-state tensor mismatch at {name}.")
        blended[name] = start + (end - start) * scale if start.is_floating_point() else start.clone()
    return blended


def empty_ppo_safety_counters() -> dict[str, float | int]:
    """Create backward-compatible per-run PPO safety counters."""
    return {
        "updates": 0,
        "pre_step_guards": 0,
        "post_step_guards": 0,
        "root_guards": 0,
        "retry_attempts": 0,
        "retry_accepted": 0,
        "root_backoff_attempts": 0,
        "root_backoff_accepted": 0,
        "reverted_updates": 0,
        "retry_kl_sum": 0.0,
    }


def approximate_policy_kl(old_log_probs: Tensor, new_log_probs: Tensor) -> Tensor:
    """Return Schulman's non-negative sampled KL approximation."""
    log_ratio = (new_log_probs - old_log_probs).float().clamp(-20.0, 20.0)
    return (torch.expm1(log_ratio) - log_ratio).mean()


def ppo_minibatch_size(batch_size: int, cuda_enabled: bool) -> int:
    """Keep a normal CUDA rollout intact while retaining an OOM escape hatch."""
    if batch_size <= 0:
        raise ValueError("PPO batch size must be positive.")
    return min(batch_size, PPO_MINIBATCH_HANDS) if cuda_enabled else batch_size


def prepare_deterministic_ppo_policy(model: nn.Module) -> None:
    """Keep cuDNN RNNs trainable while removing likelihood-ratio dropout noise."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.GRU):
            module.dropout = 0.0


def build_ppo_optimizer(parameters, *, learning_rate: float, cuda_enabled: bool, use_fused: bool | None = None) -> tuple[torch.optim.Optimizer, str]:
    """Create the PPO optimizer, using fused AdamW only after explicit opt-in."""
    options = {"lr": learning_rate, "weight_decay": 1e-5}
    parameter_list = list(parameters)
    fused_requested = PPO_USE_FUSED_ADAMW if use_fused is None else use_fused
    if fused_requested:
        if cuda_enabled:
            try:
                return torch.optim.AdamW(parameter_list, fused=True, **options), "fused-adamw"
            except (RuntimeError, TypeError, ValueError):
                pass
    return torch.optim.AdamW(parameter_list, **options), "adamw"


NONFINITE_LOGIT_REPAIRS = 0
LOGIT_SANITIZE_BOUND = 60.0


def masked_distribution(logits: Tensor, masks: Tensor, strict: bool = False) -> Categorical:
    """Build the legal-action policy, repairing non-finite logits instead of dying.

    A non-finite forward must not kill rollout workers, the PPO step, or the
    live serving path; the PPO candidate finiteness checks and rollback
    machinery remain the arbiters of whether an update is kept. ``strict=True``
    preserves the hard failure for callers that want the diagnostic.
    """
    if not tensors_are_finite(logits):
        if strict:
            raise FloatingPointError("Policy logits contain non-finite values.")
        global NONFINITE_LOGIT_REPAIRS
        NONFINITE_LOGIT_REPAIRS += 1
        logits = torch.nan_to_num(logits, nan=0.0, posinf=LOGIT_SANITIZE_BOUND, neginf=-LOGIT_SANITIZE_BOUND)
        logits = logits.clamp(-LOGIT_SANITIZE_BOUND, LOGIT_SANITIZE_BOUND)
    return Categorical(logits=logits.masked_fill(~masks.bool(), torch.finfo(logits.dtype).min))


def value_distribution_moments(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Return critic moments in true big-blind units."""
    support = torch.linspace(VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX, VALUE_BINS, dtype=logits.dtype, device=logits.device)
    probabilities = torch.softmax(logits, dim=-1)
    member_means = (probabilities * support).sum(dim=-1)
    member_variances = (probabilities * (support - member_means.unsqueeze(-1)).square()).sum(dim=-1)
    mean = member_means.mean(dim=-1) * VALUE_RETURN_SCALE_BB
    uncertainty = (member_variances.mean(dim=-1) + member_means.var(dim=-1, unbiased=False)).clamp_min(1e-8).sqrt() * VALUE_RETURN_SCALE_BB
    return mean, uncertainty


def value_support_bins(returns_bb: Tensor) -> Tensor:
    """Project true BB returns onto the normalized distributional support."""
    normalized = (returns_bb / VALUE_RETURN_SCALE_BB).clamp(VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX)
    return (((normalized - VALUE_SUPPORT_MIN) / (VALUE_SUPPORT_MAX - VALUE_SUPPORT_MIN)) * (VALUE_BINS - 1)).round().long()


def raise_distribution(raw_shapes: Tensor, actions: Tensor | int) -> Beta:
    """Action-conditioned, stable Beta policies over each legal raise interval."""
    shapes = raw_shapes.reshape(*raw_shapes.shape[:-1], RAISE_ACTION_COUNT, 2)
    if isinstance(actions, int):
        selected = shapes[..., max(0, min(RAISE_ACTION_COUNT - 1, actions - RAISE_ACTIONS[0])), :]
    else:
        indices = (actions.long() - RAISE_ACTIONS[0]).clamp(0, RAISE_ACTION_COUNT - 1)
        gathered = indices.unsqueeze(-1).unsqueeze(-1).expand(*indices.shape, 1, 2)
        selected = torch.gather(shapes, -2, gathered).squeeze(-2)
    selected = nn.functional.softplus(selected) + 1.05
    return Beta(selected[..., 0], selected[..., 1])


def raise_size_proposals(raw_shapes: Tensor, action: int) -> list[float]:
    """Deterministic learned sizing proposals around the Beta policy's mass."""
    distribution = raise_distribution(raw_shapes, action)
    mean = float(distribution.mean.clamp(0.005, 0.995).item())
    deviation = float(distribution.variance.clamp_min(1e-6).sqrt().item())
    alpha, beta = float(distribution.concentration1.item()), float(distribution.concentration0.item())
    mode = mean if alpha <= 1 or beta <= 1 else (alpha - 1) / (alpha + beta - 2)
    proposals = [mean, mode, mean - 0.85 * deviation, mean + 0.85 * deviation]
    return sorted({round(min(0.995, max(0.005, proposal)), 4) for proposal in proposals}, key=lambda proposal: abs(proposal - mean))


@dataclass
class PolicyState:
    """Bounded decision history so live play uses the same causal path as PPO."""

    observations: list[list[float]] = field(default_factory=list)


@dataclass
class RolloutPhaseProfile:
    """Non-synchronizing worker timings that cannot affect policy decisions."""

    tensor_preparation_seconds: float = 0.0
    inference_dispatch_seconds: float = 0.0
    action_postprocess_seconds: float = 0.0
    rule_execution_seconds: float = 0.0


def build_inference_batch(histories: list[list[list[float]]], features: list[list[float]], masks: list[list[bool]], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build one contiguous host batch and transfer each tensor to CUDA once."""
    lengths = [len(history) + 1 for history in histories]
    longest = max(lengths)
    pin_memory = device.type == "cuda"
    inputs = torch.zeros((len(features), longest, OBSERVATION_SIZE), dtype=torch.float32, pin_memory=pin_memory)
    padding = torch.ones((len(features), longest), dtype=torch.bool, pin_memory=pin_memory)
    for row, (history, feature) in enumerate(zip(histories, features)):
        sequence = [*history, feature]
        inputs[row, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.float32)
        padding[row, :len(sequence)] = False
    masks_tensor = torch.tensor(masks, dtype=torch.bool, pin_memory=pin_memory)
    positions = torch.tensor([length - 1 for length in lengths], dtype=torch.long, pin_memory=pin_memory)
    if pin_memory:
        inputs = inputs.to(device, non_blocking=True)
        padding = padding.to(device, non_blocking=True)
        masks_tensor = masks_tensor.to(device, non_blocking=True)
        positions = positions.to(device, non_blocking=True)
    return inputs, padding, masks_tensor, positions


def heuristic_action(game: HeadsUpHoldem, player: int) -> int:
    """A strength-aware opponent gives the league an exploitable but varied baseline."""
    legal = game.legal_actions(player)
    ranks = sorted((card[0] for card in game.hole_cards[player]), reverse=True)
    pair = ranks[0] == ranks[1]
    suited = game.hole_cards[player][0][1] == game.hole_cards[player][1][1]
    strength = 4 if pair and ranks[0] >= 10 else 3 if pair or ranks[0] >= 12 else 2 if ranks[0] >= 10 or suited else 1
    if legal.get("raise") and strength >= 4:
        return 2
    if legal.get("raise") and strength == 3 and game.to_call(player) == 0:
        return 2
    if legal.get("check"):
        return 1
    if legal.get("call"):
        pressure = game.to_call(player) / max(1, game.pot + game.to_call(player))
        return 1 if strength >= 2 or pressure < 0.16 else 0
    return 0


def network_action(model: PolicyValueNetwork, game: HeadsUpHoldem, player: int, state: PolicyState | None, greedy: bool = False, sample_raise: bool = True) -> tuple[list[float], list[bool], int, float, float, list[float], float, PolicyState]:
    features = observation(game, player)
    mask = legal_action_mask(game, player)
    history = [] if state is None else state.observations[-(POLICY_HISTORY - 1):]
    device = next(model.parameters()).device
    inputs = torch.tensor([[*history, features]], dtype=torch.float32, device=device)
    masks = torch.tensor([[mask]], dtype=torch.bool, device=device)
    with torch.inference_mode():
        logits, _, values, range_logits, _, _, raise_shapes, _ = model(inputs)
        policy = masked_distribution(logits[:, -1], masks[:, 0])
        action = int(policy.probs.argmax(dim=-1).item()) if greedy else int(policy.sample().item())
        log_prob = policy.log_prob(torch.tensor(action, device=device))
        raise_fraction = 0.5
        if action in RAISE_ACTIONS:
            sizing = raise_distribution(raise_shapes[0, -1], action)
            fraction = sizing.mean if greedy or not sample_raise else sizing.sample()
            fraction = fraction.clamp(0.005, 0.995)
            raise_fraction = float(fraction.item())
            log_prob = log_prob + sizing.log_prob(fraction)
    next_state = PolicyState([*history, features][-POLICY_HISTORY:])
    return features, mask, action, float(log_prob.item()), float((values[0, -1].mean() * VALUE_RETURN_SCALE_BB).item()), torch.softmax(range_logits[0, -1], dim=-1).detach().cpu().tolist(), raise_fraction, next_state


def network_actions_batch(model: PolicyValueNetwork, decisions: list[tuple[HeadsUpHoldem, int, PolicyState | None]], greedy: bool = False, sample_raise: bool = True, profile: RolloutPhaseProfile | None = None) -> list[tuple[list[float], list[bool], int, float, float, list[float], float, PolicyState]]:
    """Evaluate compatible independent poker states in one policy forward pass."""
    if not decisions:
        return []
    if len(decisions) == 1:
        game, player, state = decisions[0]
        started = perf_counter()
        result = [network_action(model, game, player, state, greedy, sample_raise)]
        if profile is not None:
            profile.inference_dispatch_seconds += perf_counter() - started
        return result
    preparation_started = perf_counter()
    features = [observation(game, player) for game, player, _ in decisions]
    masks = legal_action_masks_batch([game for game, _, _ in decisions], [player for _, player, _ in decisions])
    histories = [[] if state is None else state.observations[-(POLICY_HISTORY - 1):] for _, _, state in decisions]
    device = next(model.parameters()).device
    inputs, padding, masks_tensor, positions = build_inference_batch(histories, features, masks, device)
    if profile is not None:
        profile.tensor_preparation_seconds += perf_counter() - preparation_started
    inference_started = perf_counter()
    with torch.inference_mode():
        logits, _, values, range_logits, _, _, raise_shapes, _ = model(inputs, padding_mask=padding)
        rows = torch.arange(len(decisions), device=device)
        final_logits = logits[rows, positions]
        final_values = values[rows, positions]
        final_ranges = range_logits[rows, positions]
        final_shapes = raise_shapes[rows, positions]
        policy = masked_distribution(final_logits, masks_tensor)
        actions = policy.probs.argmax(dim=-1) if greedy else policy.sample()
        log_probs = policy.log_prob(actions)
    if profile is not None:
        profile.inference_dispatch_seconds += perf_counter() - inference_started
    postprocess_started = perf_counter()
    results: list[tuple[list[float], list[bool], int, float, float, list[float], float, PolicyState]] = []
    for row, (history, feature, mask, action) in enumerate(zip(histories, features, masks, actions.detach().cpu().tolist())):
        log_prob = log_probs[row]
        raise_fraction = 0.5
        if action in RAISE_ACTIONS:
            sizing = raise_distribution(final_shapes[row], action)
            fraction = sizing.mean if greedy or not sample_raise else sizing.sample()
            fraction = fraction.clamp(0.005, 0.995)
            raise_fraction = float(fraction.item())
            log_prob = log_prob + sizing.log_prob(fraction)
        next_state = PolicyState([*history, feature][-POLICY_HISTORY:])
        results.append((feature, mask, action, float(log_prob.item()), float((final_values[row].mean() * VALUE_RETURN_SCALE_BB).item()), torch.softmax(final_ranges[row], dim=-1).detach().cpu().tolist(), raise_fraction, next_state))
    if profile is not None:
        profile.action_postprocess_seconds += perf_counter() - postprocess_started
    return results


def network_policy_probabilities_batch(model: PolicyValueNetwork, decisions: list[tuple[HeadsUpHoldem, int, PolicyState | None]]) -> list[list[float]]:
    """Return masked action probabilities for audit telemetry without sampling."""
    if not decisions:
        return []
    features = [observation(game, player) for game, player, _ in decisions]
    masks = legal_action_masks_batch([game for game, _, _ in decisions], [player for _, player, _ in decisions])
    histories = [[] if state is None else state.observations[-(POLICY_HISTORY - 1):] for _, _, state in decisions]
    device = next(model.parameters()).device
    inputs, padding, masks_tensor, positions = build_inference_batch(histories, features, masks, device)
    with torch.inference_mode():
        logits, *_ = model(inputs, padding_mask=padding)
        rows = torch.arange(len(decisions), device=device)
        probabilities = masked_distribution(logits[rows, positions], masks_tensor).probs
    return probabilities.detach().cpu().tolist()


@dataclass
class HandTrajectory:
    observations: list[list[float]]
    masks: list[list[bool]]
    actions: list[int]
    log_probs: list[float]
    values: list[float]
    raise_fractions: list[float]
    reward: float
    range_label: int
    opponent_style: str = "league"
    adversarial: bool = False
    profile: str = "balanced"
    preflop_root: str = "blind"
    robust_weight: float = 1.0
    streets: list[int] = field(default_factory=list)
    all_in_probability_targets: list[float] = field(default_factory=list)
    all_in_calibration_active: list[bool] = field(default_factory=list)
    preflop_3bet_teacher_targets: list[list[float]] = field(default_factory=list)
    preflop_3bet_teacher_confidences: list[float] = field(default_factory=list)
    preflop_3bet_teacher_eligible: list[bool] = field(default_factory=list)
    preflop_3bet_teacher_raise_advantages: list[float] = field(default_factory=list)
    preflop_teacher_root_codes: list[int] = field(default_factory=list)


def trajectory_advantages(path: HandTrajectory) -> list[float]:
    """Return unnormalized GAE values for tail credit and PPO preparation."""
    gae = 0.0
    advantages = [0.0] * len(path.actions)
    for index in range(len(path.actions) - 1, -1, -1):
        next_value = path.values[index + 1] if index + 1 < len(path.actions) else 0.0
        reward = path.reward if index == len(path.actions) - 1 else 0.0
        delta = reward + 0.99 * next_value - path.values[index]
        gae = delta + 0.99 * 0.95 * gae
        advantages[index] = gae
    return advantages


@dataclass
class ImitationRecord:
    """A bounded decision retained outside the on-policy PPO batch."""

    observation: list[float]
    mask: list[bool]
    action: int
    return_value: float
    priority: float
    source: str = "external"

    def payload(self) -> dict:
        return {"observation": self.observation, "mask": self.mask, "action": self.action, "return_value": self.return_value, "priority": self.priority, "source": self.source}

    @classmethod
    def from_payload(cls, payload: dict) -> ImitationRecord:
        # Checkpoints created before this recovery migration contain only
        # survivor-biased self-play records, so treat their missing source as
        # self_play and purge them once at the next run boundary.
        return cls(list(payload["observation"]), list(payload["mask"]), int(payload["action"]), float(payload["return_value"]), float(payload.get("priority", 1.0)), str(payload.get("source", "self_play")))


class SelfImitationMemory:
    """Stores externally validated targets; ordinary winning self-play is not a teacher."""

    def __init__(self, capacity: int = 18_000) -> None:
        self.capacity = capacity
        self.records: list[ImitationRecord] = []
        self.seen = 0
        self.self_play_migration_complete = False

    def extend_paths(self, paths: list[HandTrajectory], rng: random.Random) -> None:
        # PPO already learns from every on-policy action and tail loss.  Adding
        # only winners here rewarded short-term all-in outcomes and amplified
        # the exact collapse the fixed-root audits found.
        return None

    def extend_records(self, records: list[ImitationRecord], rng: random.Random) -> None:
        """Add explicitly validated offline teacher targets outside the PPO batch."""
        for record in records:
            if record.return_value < 0.35:
                continue
            record.source = "external"
            self.seen += 1
            if len(self.records) < self.capacity:
                self.records.append(record)
                continue
            weakest = min(range(len(self.records)), key=lambda index: self.records[index].priority)
            if record.priority >= self.records[weakest].priority or rng.random() < 0.04:
                self.records[weakest] = record

    def sample(self, count: int, rng: random.Random) -> list[ImitationRecord]:
        if not self.records:
            return []
        pool = sorted(self.records, key=lambda record: record.priority, reverse=True)[:max(1, len(self.records) * 2 // 3)]
        return rng.sample(pool, min(count, len(pool)))

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "self_play_migration_complete": self.self_play_migration_complete, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.self_play_migration_complete = bool(payload.get("self_play_migration_complete", False))
        self.records = [ImitationRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]

    def discard_legacy_self_play(self) -> int:
        """Retain only explicit offline teacher data after a one-time migration."""
        before = len(self.records)
        self.records = [record for record in self.records if record.source == "external"]
        self.self_play_migration_complete = True
        return before - len(self.records)


@dataclass
class HardSpotRecord:
    """Loss-state critic target; it deliberately contains no policy imitation target."""

    observation: list[float]
    return_value: float
    priority: float
    style: str
    street: int

    def payload(self) -> dict:
        return {"observation": self.observation, "return_value": self.return_value, "priority": self.priority, "style": self.style, "street": self.street}

    @classmethod
    def from_payload(cls, payload: dict) -> HardSpotRecord:
        return cls(list(payload["observation"]), float(payload["return_value"]), float(payload.get("priority", 1.0)), str(payload.get("style", "unknown")), int(payload.get("street", 0)))


class HardSpotValueMemory:
    """Bounded replay for critic calibration on adversarial, high-cost losses only."""

    def __init__(self, capacity: int = 6_000) -> None:
        self.capacity = capacity
        self.records: list[HardSpotRecord] = []
        self.seen = 0

    def extend_paths(self, paths: list[HandTrajectory], rng: random.Random) -> None:
        for path in paths:
            if not path.adversarial or path.reward > -LARGE_LOSS_BB:
                continue
            loss_scale = min(3.0, max(1.0, -path.reward / LARGE_LOSS_BB))
            for observation_item, street in list(zip(path.observations, path.streets))[-2:]:
                priority = min(6.0, loss_scale * (1.0 + 0.20 * street))
                record = HardSpotRecord(observation_item, path.reward, priority, path.opponent_style, street)
                self.seen += 1
                if len(self.records) < self.capacity:
                    self.records.append(record)
                    continue
                weakest = min(range(len(self.records)), key=lambda index: self.records[index].priority)
                if priority >= self.records[weakest].priority or rng.random() < 0.025:
                    self.records[weakest] = record

    def sample(self, count: int, rng: random.Random, focus_styles: tuple[str, ...] = ()) -> list[HardSpotRecord]:
        if not self.records:
            return []
        pool = sorted(self.records, key=lambda record: record.priority, reverse=True)[:max(1, len(self.records) * 2 // 3)]
        sample_size = min(count, len(pool))
        focused = [record for record in pool if record.style in focus_styles]
        focused_count = min(len(focused), max(0, int(math.ceil(sample_size * 0.60))))
        selected = rng.sample(focused, focused_count) if focused_count else []
        remaining = [record for record in pool if record not in selected]
        selected.extend(rng.sample(remaining, min(sample_size - len(selected), len(remaining))))
        return selected

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "seen": self.seen, "records": [record.payload() for record in self.records]}

    def restore(self, payload: dict) -> None:
        self.capacity = max(1, int(payload.get("capacity", self.capacity)))
        self.seen = max(0, int(payload.get("seen", 0)))
        self.records = [HardSpotRecord.from_payload(record) for record in payload.get("records", []) if isinstance(record, dict)][-self.capacity:]


@dataclass
class RolloutResult:
    paths: list[HandTrajectory]
    hands: int
    actions: int
    cfr_records: list[CFRRecord]
    likelihood_records: list[ActionLikelihoodRecord]
    scenario_counts: dict[str, int]
    preflop_root_counts: dict[str, int] = field(default_factory=dict)
    oracle_records: list[AbstractTeacherRecord] = field(default_factory=list)
    solver_records: list[SolverTeacherRecord] = field(default_factory=list)
    paired_hands: int = 0
    adversarial_hands: int = 0
    compiled_transition_actions: int = 0
    model_sync_seconds: float = 0.0
    arena_setup_seconds: float = 0.0
    tensor_preparation_seconds: float = 0.0
    inference_dispatch_seconds: float = 0.0
    action_postprocess_seconds: float = 0.0
    rule_execution_seconds: float = 0.0
    play_seconds: float = 0.0
    worker_seconds: float = 0.0
    cached_opponent_models: int = 0
    opponent_revisions: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    ties: int = 0
    reward: float = 0.0
    returns_bb: list[float] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = self.wins + self.losses + self.ties
        return (self.wins + 0.5 * self.ties) / total if total else 0.5

    @property
    def hands(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def bb_per_100(self) -> float:
        return self.reward / max(1, self.hands) * 100

    @property
    def paired_returns_bb(self) -> list[float]:
        """Average consecutive same-deal, seat-swapped returns before inference."""
        pairs = [0.5 * (self.returns_bb[index] + self.returns_bb[index + 1]) for index in range(0, len(self.returns_bb) - 1, 2)]
        if len(self.returns_bb) % 2:
            pairs.append(self.returns_bb[-1])
        return pairs


@dataclass(frozen=True)
class EvaluationJob:
    """One independently seeded evaluation task."""

    key: str
    function: Callable[..., Any]
    args: tuple[Any, ...]


def run_evaluation_jobs(executor: ProcessPoolExecutor | None, jobs: list[EvaluationJob], gpu_jobs: list[EvaluationJob] | None = None) -> tuple[dict[str, Any], bool]:
    """Run independent evaluator jobs concurrently without changing their inputs.

    Each evaluator owns its random seed and uses one CPU thread, so scheduling it
    beside another evaluator cannot alter the policy, deal sequence, sample
    counts, or promotion metrics. A broken worker pool falls back to the same
    serial calls instead of weakening or skipping any quality gate.
    """
    gpu_jobs = gpu_jobs or []
    if executor is None or len(jobs) < 2:
        results = {job.key: job.function(*job.args) for job in jobs}
        results.update({job.key: job.function(*job.args) for job in gpu_jobs})
        return results, False
    futures: dict[str, Any] = {}
    try:
        futures = {job.key: executor.submit(job.function, *job.args) for job in jobs}
        # CUDA work runs in the coordinator while CPU evaluator processes keep
        # simulating hands. One lane prevents CUDA-context contention.
        results = {job.key: job.function(*job.args) for job in gpu_jobs}
        results.update({job.key: futures[job.key].result() for job in jobs})
        return results, True
    except BrokenProcessPool:
        for future in futures.values():
            future.cancel()
        return {job.key: job.function(*job.args) for job in jobs}, False


def wilson_lower_bound(score: float, samples: int, z: float = 1.96) -> float:
    """Conservative lower confidence bound for a fractional win score."""
    if samples <= 0:
        return 0.0
    denominator = 1 + z * z / samples
    centre = score + z * z / (2 * samples)
    margin = z * math.sqrt((score * (1 - score) + z * z / (4 * samples)) / samples)
    return max(0.0, (centre - margin) / denominator)


def wilson_upper_bound(score: float, samples: int, z: float = 1.96) -> float:
    """Upper confidence companion used to end clearly weak promotion matches early."""
    if samples <= 0:
        return 1.0
    denominator = 1 + z * z / samples
    centre = score + z * z / (2 * samples)
    margin = z * math.sqrt((score * (1 - score) + z * z / (4 * samples)) / samples)
    return min(1.0, (centre + margin) / denominator)


def bootstrap_bb_per_100_bounds(result: MatchResult, seed: int, samples: int = 800) -> tuple[float, float]:
    """Paired-deal bootstrap interval for chip EV, the promotion metric."""
    paired = result.paired_returns_bb
    if not paired:
        return 0.0, 0.0
    if len(paired) == 1:
        value = paired[0] * 100
        return value, value
    rng = random.Random(seed)
    count = len(paired)
    estimates = [sum(paired[rng.randrange(count)] for _ in range(count)) / count * 100 for _ in range(samples)]
    estimates.sort()
    lower_index = max(0, int(len(estimates) * 0.025) - 1)
    upper_index = min(len(estimates) - 1, int(len(estimates) * 0.975))
    return estimates[lower_index], estimates[upper_index]


def paired_bb_per_100_metrics(result: MatchResult, seed: int) -> tuple[float, float, float, float]:
    """Mean, paired bootstrap interval, and paired-return variance in BB/100."""
    paired = result.paired_returns_bb
    mean = sum(paired) / max(1, len(paired)) * 100
    lower, upper = bootstrap_bb_per_100_bounds(result, seed)
    variance = sum((value * 100 - mean) ** 2 for value in paired) / max(1, len(paired) - 1)
    return mean, lower, upper, variance


def _planned_board(game: HeadsUpHoldem) -> list[tuple[int, str]]:
    """Cards the engine will expose if the hand reaches a showdown."""
    return list(reversed(game.deck[-5:]))


def _scenario_matches(game: HeadsUpHoldem, profile: str) -> bool:
    if profile in {"balanced", "short_pressure"}:
        return True
    flop = _planned_board(game)[:3]
    ranks = [card[0] for card in flop]
    suits = [card[1] for card in flop]
    if profile == "paired":
        return len(set(ranks)) < len(ranks)
    if profile == "monotone":
        return max(suits.count(suit) for suit in set(suits)) == 3
    if profile == "draw_heavy":
        ordered = sorted(set(ranks))
        connected = any(right - left <= 2 for left, right in zip(ordered, ordered[1:]))
        return connected or max(suits.count(suit) for suit in set(suits)) >= 2
    return True


def scenario_game(initial_stack: int, rng: random.Random, button_offset: int, profile: str) -> HeadsUpHoldem:
    """Rejection-sample a bounded number of deterministic deck profiles for coverage."""
    fallback: HeadsUpHoldem | None = None
    for _ in range(18):
        candidate = HeadsUpHoldem(initial_stack=initial_stack, rng=random.Random(rng.randrange(1 << 30)), button_offset=button_offset)
        fallback = candidate
        if _scenario_matches(candidate, profile):
            return candidate
    assert fallback is not None
    return fallback


def scenario_profile(rng: random.Random, curriculum_stage: int, adaptive_weights: dict[str, float] | None = None) -> str:
    profiles = ("balanced", "short_pressure") if curriculum_stage == 0 else SCENARIO_PROFILES
    base_weights = (0.46, 0.54) if curriculum_stage == 0 else (0.28, 0.17, 0.22, 0.17, 0.16)
    weights = [base * (0.60 + 1.40 * min(1.0, max(0.0, float((adaptive_weights or {}).get(profile, 0.50))))) for profile, base in zip(profiles, base_weights)]
    return rng.choices(profiles, weights=weights, k=1)[0]


PREFLOP_SCENARIO_ROOTS = ("blind", "facing_open_2bb", "facing_open_3bb", "facing_open_4bb", "facing_open_5bb", "facing_3bet", "facing_4bet")
PREFLOP_FORCED_ROOTS = PREFLOP_SCENARIO_ROOTS[1:]


def preflop_root_button_offset(root: str, learner_player: int) -> int:
    """Choose a button so the learner is the next player after a scripted root."""
    return 1 - learner_player if root.startswith("facing_open_") or root == "facing_4bet" else learner_player


def _scripted_raise_to(game: HeadsUpHoldem, target: int) -> bool:
    player = game.current_player
    if player is None:
        return False
    legal = game.legal_actions(player)
    if not legal.get("raise"):
        return False
    minimum, maximum = int(legal["raise_min"]), int(legal["raise_max"])
    game.act(player, "raise", min(maximum, max(minimum, int(target))))
    return not game.hand_complete


def prepare_preflop_root(game: HeadsUpHoldem, root: str) -> bool:
    """Apply legal public prefixes without recording off-policy learner decisions."""
    if root == "blind":
        return True
    if root.startswith("facing_open_"):
        open_bb = int(root.rsplit("_", 1)[-1].removesuffix("bb"))
        return _scripted_raise_to(game, game.big_blind * open_bb)
    if root == "facing_3bet":
        return _scripted_raise_to(game, game.big_blind * 3) and _scripted_raise_to(game, round(game.pot * PREFLOP_THREE_BET_POT_CAP_MULTIPLIER))
    if root == "facing_4bet":
        if not _scripted_raise_to(game, game.big_blind * 3):
            return False
        if not _scripted_raise_to(game, round(game.pot * PREFLOP_THREE_BET_POT_CAP_MULTIPLIER)):
            return False
        player = game.current_player
        if player is None:
            return False
        legal = game.legal_actions(player)
        return bool(legal.get("raise")) and _scripted_raise_to(game, int(legal["raise_min"]))
    raise ValueError(f"Unknown preflop root: {root}")


def preflop_root_policy_signature(
    model: PolicyValueNetwork,
    curriculum_stage: int,
    *,
    hands_per_root: int = PREFLOP_ROOT_PROBE_HANDS,
    seed: int = 1_880_031,
) -> dict[str, list[list[float]]]:
    """Return card-matched first-action probabilities for every forced root."""
    device = next(model.parameters()).device
    was_training = model.training
    observations: list[list[float]] = []
    masks: list[list[bool]] = []
    roots: list[str] = []
    stack = stack_for_stage(curriculum_stage)
    for root_index, root in enumerate(PREFLOP_FORCED_ROOTS):
        for hand in range(max(4, hands_per_root)):
            player = hand % 2
            game = scenario_game(
                stack,
                random.Random(seed + root_index * 10_007 + hand * 97),
                preflop_root_button_offset(root, player),
                "balanced",
            )
            if prepare_preflop_root(game, root) and game.current_player == player:
                observations.append(observation(game, player))
                masks.append(legal_action_mask(game, player))
                roots.append(root)
    if not observations:
        return {}
    model.eval()
    with torch.inference_mode():
        inputs = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(1)
        legal_masks = torch.tensor(masks, dtype=torch.bool, device=device)
        logits, _, _, _, _, _, _, _ = model(inputs)
        probabilities = masked_distribution(logits[:, 0], legal_masks).probs.float().cpu().tolist()
    if was_training:
        model.train()
    signature = {root: [] for root in PREFLOP_FORCED_ROOTS}
    for root, probability in zip(roots, probabilities):
        signature[root].append([float(value) for value in probability])
    return signature


def preflop_root_policy_drift(
    reference: dict[str, list[list[float]]],
    candidate: dict[str, list[list[float]]],
) -> dict[str, Any]:
    """Measure the worst card-matched root KL and mean action-mass movement."""
    worst_kl = 0.0
    worst_kl_root = "pending"
    worst_action_delta = 0.0
    worst_action_root = "pending"
    per_root: dict[str, dict[str, float]] = {}
    for root in PREFLOP_FORCED_ROOTS:
        reference_rows = reference.get(root, [])
        candidate_rows = candidate.get(root, [])
        count = min(len(reference_rows), len(candidate_rows))
        if count <= 0:
            continue
        reference_tensor = torch.tensor(reference_rows[:count], dtype=torch.float64).clamp_min(1e-9)
        candidate_tensor = torch.tensor(candidate_rows[:count], dtype=torch.float64).clamp_min(1e-9)
        root_kl = float((reference_tensor * (reference_tensor.log() - candidate_tensor.log())).sum(dim=-1).mean().item())
        root_action_delta = float((reference_tensor.mean(dim=0) - candidate_tensor.mean(dim=0)).abs().max().item())
        per_root[root] = {"kl": root_kl, "action_delta": root_action_delta}
        if root_kl > worst_kl:
            worst_kl, worst_kl_root = root_kl, root
        if root_action_delta > worst_action_delta:
            worst_action_delta, worst_action_root = root_action_delta, root
    return {
        "max_kl": worst_kl,
        "max_kl_root": worst_kl_root,
        "max_action_delta": worst_action_delta,
        "max_action_delta_root": worst_action_root,
        "per_root": per_root,
    }


def focused_preflop_root(root_weights: dict[str, float] | None = None) -> str | None:
    """Return a clearly worst audited root, avoiding arbitrary fresh-model focus."""
    if not root_weights:
        return None
    ranked = sorted(
        ((root, float(root_weights.get(root, 0.50))) for root in PREFLOP_FORCED_ROOTS),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked) < 2 or ranked[0][1] - ranked[1][1] < 0.035:
        return None
    return ranked[0][0]


def sample_preflop_root(rng: random.Random, root_weights: dict[str, float] | None = None) -> str:
    if rng.random() >= PREFLOP_FORCED_ROOT_FRACTION:
        return "blind"
    focus_root = focused_preflop_root(root_weights)
    weights = [
        max(0.05, float((root_weights or {}).get(root, 0.50)))
        * (PREFLOP_FOCUS_ROOT_WEIGHT_MULTIPLIER if root == focus_root else 1.0)
        for root in PREFLOP_FORCED_ROOTS
    ]
    return rng.choices(PREFLOP_FORCED_ROOTS, weights=weights, k=1)[0]


def preflop_all_in_calibration(game: HeadsUpHoldem, player: int) -> tuple[bool, float, float]:
    """Return a soft all-in probability ceiling for non-committed preflop states.

    This is deliberately not an action mask. A player who has already committed
    half their stack (the normal 4-/5-bet region at the foundation stack) keeps
    the unrestricted all-in action. Earlier opens and low-commitment re-jams
    receive only a bounded PPO calibration loss.
    """
    legal = game.legal_actions(player)
    to_call = int(legal.get("to_call", 0))
    commitment = min(1.0, (game.round_bets[player] + to_call) / max(1, game.initial_stack))
    can_raise_all_in = bool(legal.get("all_in")) and game.stacks[player] > to_call
    if game.street != 0 or not can_raise_all_in or commitment >= PREFLOP_ALLIN_COMMITTED_FRACTION:
        return False, 1.0, commitment
    voluntary_raises = sum(
        event.get("street") == 0 and event.get("action") == "raise"
        for event in game.public_actions
    )
    # Re-raised pots remain guarded until stack commitment makes a shove
    # structurally normal. A high-confidence matched branch teacher can still
    # exempt an individual decision inside PPO, so profitable 4-/5-bet shoves
    # remain learnable without leaving every re-raise state unprotected.
    stack_depth_bb = game.initial_stack / max(1, game.big_blind)
    # Open-jamming is strategically useful at 20 BB but should be rare at
    # standard and deep stacks.  Preserve high-confidence branch-teacher
    # exemptions while making the generic ceiling stack-depth aware.
    depth_scale = min(1.0, max(0.20, 20.0 / max(20.0, stack_depth_bb)))
    target = min(
        PREFLOP_ALLIN_TARGET_MAX,
        (0.04 + 0.20 * commitment + 0.025 * min(4, voluntary_raises)) * depth_scale,
    )
    return True, target, commitment


def preflop_all_in_ranking_loss(logits: Tensor, masks: Tensor, margin: float) -> Tensor:
    """Prefer a legal non-all-in action when an early shove is not supported.

    This soft margin keeps unsupported early shoves from dominating the mixed
    policy without making the legal all-in action unavailable.
    """
    masked_logits = logits.masked_fill(~masks.bool(), torch.finfo(logits.dtype).min)
    best_non_all_in = masked_logits[..., :3].max(dim=-1).values
    return nn.functional.softplus(masked_logits[..., 3] - best_non_all_in + margin).mean()


PREFLOP_3BET_TEACHER_MULTI_RAISE_ROOTS = ("facing_3bet", "facing_4bet")
PREFLOP_3BET_TEACHER_ACTION_NAMES = ("fold", "check_call", "raise", "all_in")
PREFLOP_3BET_TEACHER_ROOTS = (
    "blind",
    "facing_open_2bb",
    "facing_open_3bb",
    "facing_open_4bb",
    "facing_open_5bb",
    *PREFLOP_3BET_TEACHER_MULTI_RAISE_ROOTS,
)
PREFLOP_TEACHER_ROOT_CODES = {root: index for index, root in enumerate(PREFLOP_3BET_TEACHER_ROOTS)}
PREFLOP_TEACHER_UNKNOWN_ROOT_CODE = -1


def empty_preflop_teacher_root_totals() -> dict[str, float | int | dict[str, float]]:
    return {
        "eligible_roots": 0,
        "sampled_roots": 0,
        "confidence_sum": 0.0,
        "target_action_sums": {name: 0.0 for name in PREFLOP_3BET_TEACHER_ACTION_NAMES},
        "actual_action_counts": {name: 0.0 for name in PREFLOP_3BET_TEACHER_ACTION_NAMES},
    }


def preflop_teacher_root_metrics(totals: dict[str, float | int | dict[str, float]]) -> dict[str, float | int | dict[str, float]]:
    samples = max(1, int(totals["sampled_roots"]))
    target_sums = totals["target_action_sums"]
    actual_counts = totals["actual_action_counts"]
    assert isinstance(target_sums, dict) and isinstance(actual_counts, dict)
    target_mix = {name: float(target_sums[name]) / samples for name in PREFLOP_3BET_TEACHER_ACTION_NAMES}
    actual_mix = {name: float(actual_counts[name]) / samples for name in PREFLOP_3BET_TEACHER_ACTION_NAMES}
    return {
        "eligible_roots": int(totals["eligible_roots"]),
        "sampled_roots": int(totals["sampled_roots"]),
        "coverage": int(totals["sampled_roots"]) / max(1, int(totals["eligible_roots"])),
        "mean_confidence": float(totals["confidence_sum"]) / samples,
        "target_action_mix": target_mix,
        "actual_action_mix": actual_mix,
        "call_overuse": actual_mix["check_call"] - target_mix["check_call"],
        "all_in_overuse": actual_mix["all_in"] - target_mix["all_in"],
    }


def preflop_3bet_teacher_eligible(game: HeadsUpHoldem, player: int, root: str, first_learner_decision: bool) -> bool:
    """Identify preflop roots for a matched legal-action comparison.

    The name is retained for checkpoint compatibility.  In addition to blind and
    open responses, the comparison now covers facing 3-/4-bet roots, where the
    generic early-shove calibration deliberately does not apply.
    """
    legal = game.legal_actions(player)
    return (
        first_learner_decision
        and root in PREFLOP_3BET_TEACHER_ROOTS
        and game.street == 0
        and bool(legal.get("call") or legal.get("check"))
        and bool(legal.get("raise"))
    )


def preflop_teacher_sampling_allowed(
    root: str,
    focus_root: str | None,
    root_teacher_remaining: int,
    multi_raise_teacher_remaining: int,
    facing_4bet_teacher_remaining: int,
    shallow_open_teacher_remaining: int,
    shallow_open_teacher_samples: dict[str, int],
    focus_teacher_remaining: int,
) -> bool:
    """Keep branch-teacher quotas while honoring an audited root's focus quota.

    The focus quota must remain reserved even when its root also belongs to the
    shallow-open or multi-raise groups.  Those group quotas provide coverage;
    they are not a substitute for repeatedly supervising the reported weak
    root.  A focused shallow root may therefore exceed the fairness cap only
    while it still has reserved focus samples to collect.
    """
    is_multi_raise_root = root in PREFLOP_3BET_TEACHER_MULTI_RAISE_ROOTS
    is_facing_4bet_root = root == "facing_4bet"
    is_shallow_open_root = root in PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS
    is_focus_root = root == focus_root
    focus_reserve_after = max(0, focus_teacher_remaining - int(is_focus_root))
    multi_after = max(0, multi_raise_teacher_remaining - int(is_multi_raise_root))
    shallow_after = max(0, shallow_open_teacher_remaining - int(is_shallow_open_root))
    reserved_after = multi_after + shallow_after + focus_reserve_after
    if root_teacher_remaining <= reserved_after:
        return False
    if is_shallow_open_root:
        shallow_cap = max(1, math.ceil(PREFLOP_TEACHER_SHALLOW_OPEN_MIN_ROOTS / len(PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS)))
        at_cap = shallow_open_teacher_samples.get(root, 0) >= shallow_cap
        if at_cap and not (is_focus_root and focus_teacher_remaining > 0):
            return False
    # A non-4-bet multi-raise sample cannot spend capacity reserved for the
    # specifically vulnerable facing-4-bet branch.
    return not (is_multi_raise_root and not is_facing_4bet_root and multi_after < facing_4bet_teacher_remaining)


def preflop_3bet_focus_style(root: str, rng: random.Random) -> str | None:
    """Couple the unsafe blind/open roots to the styles that exposed the leak."""
    if root not in PREFLOP_3BET_TEACHER_ROOTS or rng.random() >= PREFLOP_3BET_STYLE_FOCUS_SHARE:
        return None
    return "tight_aggressive" if rng.random() < 0.78 else "nit"


def preflop_3bet_teacher_leaf(target: PolicyValueNetwork, game: HeadsUpHoldem, learner_player: int) -> float:
    device = next(target.parameters()).device
    inputs = torch.tensor([[observation(game, learner_player)]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        _, _, _, _, _, distribution_logits, _, _ = target(inputs)
        value, uncertainty = value_distribution_moments(distribution_logits[0, 0])
    return float((value - 0.10 * uncertainty).item())


def preflop_3bet_branch_value(game: HeadsUpHoldem, learner: PolicyValueNetwork, target: PolicyValueNetwork, opponent: PolicyValueNetwork | str | None, learner_player: int, action: int) -> float:
    """Score one legal root action on an identical deal with a bounded greedy continuation.

    The simulated opponent cards are never copied into the policy input. They only
    provide one sampled environment outcome, just like an ordinary self-play hand.
    """
    branch = copy.deepcopy(game)
    execute_action(branch, learner_player, action, 0.50 if action == 2 else None)
    learner_state: PolicyState | None = PolicyState([observation(game, learner_player)])
    opponent_state: PolicyState | None = None
    steps = 1
    while not branch.hand_complete and steps < PREFLOP_3BET_TEACHER_DEPTH:
        current = branch.current_player
        assert current is not None
        if current == learner_player:
            _, _, chosen, _, _, _, fraction, learner_state = network_action(learner, branch, current, learner_state, greedy=True)
        elif opponent is None:
            chosen, fraction = heuristic_action(branch, current), None
        elif isinstance(opponent, str):
            chosen, fraction = style_action(branch, current, opponent), None
        else:
            _, _, chosen, _, _, _, fraction, opponent_state = network_action(opponent, branch, current, opponent_state, greedy=True)
        execute_action(branch, current, chosen, fraction if chosen in RAISE_ACTIONS else None)
        steps += 1
    if branch.hand_complete:
        return (branch.stacks[learner_player] - branch.initial_stack) / branch.big_blind
    return preflop_3bet_teacher_leaf(target, branch, learner_player)


def preflop_teacher_confidence_weight(confidence: Tensor) -> Tensor:
    """Give uncertain self-play targets zero weight, then scale smoothly."""
    return ((confidence - PREFLOP_3BET_TEACHER_MIN_CONFIDENCE) / max(1e-6, 1.0 - PREFLOP_3BET_TEACHER_MIN_CONFIDENCE)).clamp(0.0, 1.0)


def _preflop_teacher_worlds(game: HeadsUpHoldem, learner_player: int) -> list[HeadsUpHoldem]:
    """Build paired hidden-card worlds with identical learner/public state."""
    worlds = [copy.deepcopy(game)]
    learner_cards = set(game.hole_cards[learner_player])
    public_cards = set(game.community)
    public_seed = game.hand_number * 104_729 + learner_player * 7_919
    public_seed += sum((index + 1) * int(event.get("amount", 0)) for index, event in enumerate(game.public_actions))
    public_seed += sum(card[0] * (new_deck().index(card) + 1) for card in learner_cards)
    for world_index in range(1, PREFLOP_3BET_TEACHER_WORLDS):
        world = copy.deepcopy(game)
        available = [card for card in new_deck() if card not in learner_cards and card not in public_cards]
        world.rng = random.Random(public_seed + world_index * 1_000_003)
        world.rng.shuffle(available)
        opponent = 1 - learner_player
        world.hole_cards[opponent] = [available.pop(), available.pop()]
        world.deck = available
        worlds.append(world)
    return worlds


def _preflop_teacher_branch_values_batched(
    worlds: list[HeadsUpHoldem],
    candidate_actions: list[int],
    learner: PolicyValueNetwork,
    target: PolicyValueNetwork,
    opponent: PolicyValueNetwork | str | None,
    learner_player: int,
) -> dict[int, list[float]]:
    """Evaluate all paired worlds/actions with batched neural inference.

    This preserves the exact branch semantics of ``preflop_3bet_branch_value``
    while replacing dozens of tiny GPU forwards with one forward per simulated
    decision layer.
    """
    branches: list[tuple[int, HeadsUpHoldem]] = []
    learner_states: list[PolicyState | None] = []
    opponent_states: list[PolicyState | None] = []
    steps: list[int] = []
    for world in worlds:
        for action in candidate_actions:
            branch = copy.deepcopy(world)
            execute_action(branch, learner_player, action, 0.50 if action == 2 else None)
            branches.append((action, branch))
            learner_states.append(PolicyState([observation(world, learner_player)]))
            opponent_states.append(None)
            steps.append(1)

    while True:
        active = [index for index, (_, branch) in enumerate(branches) if not branch.hand_complete and steps[index] < PREFLOP_3BET_TEACHER_DEPTH]
        if not active:
            break
        learner_rows = [index for index in active if branches[index][1].current_player == learner_player]
        if learner_rows:
            decisions = [(branches[index][1], learner_player, learner_states[index]) for index in learner_rows]
            for index, result in zip(learner_rows, network_actions_batch(learner, decisions, greedy=True)):
                chosen, fraction, learner_states[index] = result[2], result[6], result[7]
                execute_action(branches[index][1], learner_player, chosen, fraction if chosen in RAISE_ACTIONS else None)
                steps[index] += 1
        opponent_rows = [index for index in active if not branches[index][1].hand_complete and steps[index] < PREFLOP_3BET_TEACHER_DEPTH and branches[index][1].current_player != learner_player]
        if isinstance(opponent, PolicyValueNetwork) and opponent_rows:
            decisions = [(branches[index][1], 1 - learner_player, opponent_states[index]) for index in opponent_rows]
            for index, result in zip(opponent_rows, network_actions_batch(opponent, decisions, greedy=True)):
                chosen, fraction, opponent_states[index] = result[2], result[6], result[7]
                execute_action(branches[index][1], 1 - learner_player, chosen, fraction if chosen in RAISE_ACTIONS else None)
                steps[index] += 1
        else:
            for index in opponent_rows:
                branch = branches[index][1]
                current = branch.current_player
                assert current is not None
                chosen = heuristic_action(branch, current) if opponent is None else style_action(branch, current, opponent)
                execute_action(branch, current, chosen, None)
                steps[index] += 1

    results = {action: [] for action in candidate_actions}
    leaf_rows = [index for index, (_, branch) in enumerate(branches) if not branch.hand_complete]
    leaf_values: dict[int, float] = {}
    if leaf_rows:
        device = next(target.parameters()).device
        inputs = torch.tensor([[observation(branches[index][1], learner_player)] for index in leaf_rows], dtype=torch.float32, device=device)
        with torch.inference_mode():
            _, _, _, _, _, distributions, _, _ = target(inputs)
            means, uncertainties = value_distribution_moments(distributions[:, 0])
        leaf_values = {
            index: float((mean - 0.10 * uncertainty).item())
            for index, mean, uncertainty in zip(leaf_rows, means, uncertainties)
        }
    for index, (action, branch) in enumerate(branches):
        value = (branch.stacks[learner_player] - branch.initial_stack) / branch.big_blind if branch.hand_complete else leaf_values[index]
        results[action].append(float(value))
    return results


def preflop_3bet_teacher_target(game: HeadsUpHoldem, learner: PolicyValueNetwork, target: PolicyValueNetwork, opponent: PolicyValueNetwork | str | None, learner_player: int) -> tuple[list[float], float, float]:
    """Create a confidence-gated target from paired hidden-card worlds.

    Every legal action sees the same sampled worlds.  The target is useful only
    when the paired lower confidence bound of the best action over the runner-up
    is positive; ambiguous self-generated labels therefore cannot train PPO.
    """
    mask = legal_action_mask(game, learner_player)
    candidate_actions = [action for action in range(ACTION_COUNT) if mask[action]]
    if len(candidate_actions) < 2 or 1 not in candidate_actions:
        return [0.0] * ACTION_COUNT, 0.0, 0.0
    worlds = _preflop_teacher_worlds(game, learner_player)
    samples = _preflop_teacher_branch_values_batched(worlds, candidate_actions, learner, target, opponent, learner_player)
    values = {action: sum(action_samples) / len(action_samples) for action, action_samples in samples.items()}
    ranked_actions = sorted(candidate_actions, key=lambda action: values[action], reverse=True)
    best_action, runner_up = ranked_actions[:2]
    paired_margins = [left - right for left, right in zip(samples[best_action], samples[runner_up])]
    mean_margin = sum(paired_margins) / len(paired_margins)
    if len(paired_margins) > 1:
        margin_variance = sum((margin - mean_margin) ** 2 for margin in paired_margins) / (len(paired_margins) - 1)
        margin_standard_error = math.sqrt(margin_variance / len(paired_margins))
    else:
        margin_standard_error = 0.0
    margin_lcb = mean_margin - PREFLOP_3BET_TEACHER_CONFIDENCE_Z * margin_standard_error
    confidence = min(1.0, max(0.0, margin_lcb / PREFLOP_3BET_TEACHER_CONFIDENCE_BB))
    conservative_values = {
        action: value - 0.5 * (
            math.sqrt(sum((sample - value) ** 2 for sample in samples[action]) / (len(samples[action]) - 1)) / math.sqrt(len(samples[action]))
            if len(samples[action]) > 1 else 0.0
        )
        for action, value in values.items()
    }
    largest = max(conservative_values.values())
    scaled = {action: math.exp(max(-20.0, min(0.0, (value - largest) / PREFLOP_3BET_TEACHER_TEMPERATURE_BB))) for action, value in conservative_values.items()}
    normalizer = sum(scaled.values())
    target_distribution = [scaled.get(action, 0.0) / max(1e-8, normalizer) for action in range(ACTION_COUNT)]
    raise_advantage = conservative_values.get(2, float("-inf")) - max(conservative_values.get(0, float("-inf")), conservative_values.get(1, float("-inf")))
    return target_distribution, confidence, raise_advantage


def stack_for_stage(curriculum_stage: int) -> int:
    return STAGE_STACKS[min(len(STAGE_STACKS) - 1, max(0, curriculum_stage))]


def collect_rollouts(model_state: dict[str, Tensor], target_state: dict[str, Tensor], opponent_entries: list[dict], hands: int, seed: int, curriculum_stage: int, cfr_iteration: int, best_response_lane: bool = False, adaptive_scenarios: dict[str, float] | None = None, preflop_root_weights: dict[str, float] | None = None) -> RolloutResult:
    """Process-safe recurrent self-play collection against the champion league."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    learner = PolicyValueNetwork()
    learner.load_state_dict(model_state)
    learner.eval()
    target = PolicyValueNetwork()
    target.load_state_dict(target_state)
    target.eval()
    opponents: list[PolicyValueNetwork | str] = []
    weights: list[float] = []
    robust_weights: list[float] = []
    adversarial_flags: list[bool] = []
    for entry in opponent_entries:
        if entry.get("kind") == "style":
            opponents.append(str(entry["style"]))
            weights.append(float(entry.get("weight", 1.0)))
            robust_weights.append(min(1.0 + ROBUST_STYLE_POLICY_WEIGHT, max(1.0, float(entry.get("robust_weight", 1.0)))))
            adversarial_flags.append(bool(entry.get("adversarial", False)))
            continue
        opponent = PolicyValueNetwork()
        opponent.load_state_dict(entry["state"])
        opponent.eval()
        opponents.append(opponent)
        weights.append(float(entry.get("weight", 1.0)))
        robust_weights.append(min(1.0 + ROBUST_STYLE_POLICY_WEIGHT, max(1.0, float(entry.get("robust_weight", 1.0)))))
        adversarial_flags.append(bool(entry.get("adversarial", False)) or entry.get("kind") == "exploiter")
    stack_size = stack_for_stage(curriculum_stage)
    league_probability = 0.94 if best_response_lane else (0.55, 0.70, 0.84, 0.92)[min(curriculum_stage, 3)]
    paths: list[HandTrajectory] = []
    cfr_records: list[CFRRecord] = []
    likelihood_records: list[ActionLikelihoodRecord] = []
    total_actions = 0
    scenario_counts = {profile: 0 for profile in SCENARIO_PROFILES}
    preflop_root_counts = {root: 0 for root in PREFLOP_SCENARIO_ROOTS}
    root_teacher_remaining = PREFLOP_3BET_TEACHER_MAX_ROOTS
    multi_raise_teacher_remaining = PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS
    facing_4bet_teacher_remaining = PREFLOP_3BET_TEACHER_FACING_4BET_MIN_ROOTS
    shallow_open_teacher_remaining = PREFLOP_TEACHER_SHALLOW_OPEN_MIN_ROOTS
    shallow_open_teacher_samples = {root: 0 for root in PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS}
    focus_root = focused_preflop_root(preflop_root_weights)
    focus_teacher_remaining = PREFLOP_TEACHER_FOCUS_MIN_ROOTS if focus_root else 0

    for _ in range(hands):
        profile = scenario_profile(rng, curriculum_stage, adaptive_scenarios)
        hand_stack = 250 if profile == "short_pressure" else stack_size
        learner_player = rng.randrange(2)
        root = sample_preflop_root(rng, preflop_root_weights)
        game = scenario_game(hand_stack, rng, preflop_root_button_offset(root, learner_player), profile)
        if not prepare_preflop_root(game, root):
            root = "blind"
            game = scenario_game(hand_stack, rng, learner_player, profile)
        scenario_counts[profile] += 1
        preflop_root_counts[root] += 1
        opponent_index = rng.choices(range(len(opponents)), weights=weights, k=1)[0] if opponents and rng.random() < league_probability else None
        opponent = opponents[opponent_index] if opponent_index is not None else None
        adversarial = bool(opponent_index is not None and adversarial_flags[opponent_index])
        robust_weight = robust_weights[opponent_index] if adversarial and opponent_index is not None else 1.0
        counterfactual_cache: dict[tuple[int, tuple[float, ...]], ActionChoice] = {}

        def cached_continuation(model: PolicyValueNetwork, branch: HeadsUpHoldem, current: int) -> ActionChoice:
            key = (current, tuple(round(value, 4) for value in observation(branch, current)))
            cached = counterfactual_cache.get(key)
            if cached is not None:
                return cached
            _, _, action, _, _, _, fraction, _ = network_action(model, branch, current, None, greedy=True)
            choice = ActionChoice(action, fraction if action in RAISE_ACTIONS else None)
            if len(counterfactual_cache) >= 384:
                counterfactual_cache.pop(next(iter(counterfactual_cache)))
            counterfactual_cache[key] = choice
            return choice

        def counterfactual_continuation(branch: HeadsUpHoldem, current: int) -> ActionChoice:
            if current == learner_player:
                return cached_continuation(learner, branch, current)
            if opponent is None:
                return ActionChoice(heuristic_action(branch, current))
            if isinstance(opponent, str):
                return ActionChoice(style_action(branch, current, opponent))
            return cached_continuation(opponent, branch, current)

        def counterfactual_value_leaf(branch: HeadsUpHoldem, focal_player: int) -> float:
            inputs = torch.tensor([[observation(branch, focal_player)]], dtype=torch.float32)
            with torch.inference_mode():
                _, _, _, _, _, distribution_logits, _, _ = target(inputs)
                value, uncertainty = value_distribution_moments(distribution_logits[0, 0])
            return float((value - 0.12 * uncertainty).item())

        def action_likelihood(history: list[list[float]]) -> list[list[list[float]]]:
            inputs = torch.tensor([history], dtype=torch.float32)
            with torch.inference_mode():
                logits = learner.action_likelihood_sequence_logits(inputs)[0]
            return torch.softmax(logits, dim=-1).tolist()

        learner_state: PolicyState | None = None
        opponent_state: PolicyState | None = None
        opponent_style = opponent if isinstance(opponent, str) else "exploiter" if adversarial else "league"
        path = HandTrajectory([], [], [], [], [], [], 0.0, 0, opponent_style=str(opponent_style), adversarial=adversarial, preflop_root=root, robust_weight=robust_weight)
        recorded_counterfactual = False
        safety = 0
        while not game.hand_complete and safety < 100:
            player = game.current_player
            assert player is not None
            context = action_context_features(game, player)
            history = [event_context_features(event, game.initial_stack) for event in game.public_actions if int(event.get("action_index", -1)) >= 0]
            action_fraction: float | None = None
            if player == learner_player:
                calibration_active, all_in_target, _ = preflop_all_in_calibration(game, player)
                teacher_eligible = preflop_3bet_teacher_eligible(game, player, root, not path.actions)
                teacher_target, teacher_confidence, teacher_raise_advantage = [0.0] * ACTION_COUNT, 0.0, 0.0
                is_multi_raise_root = root in PREFLOP_3BET_TEACHER_MULTI_RAISE_ROOTS
                is_facing_4bet_root = root == "facing_4bet"
                is_shallow_open_root = root in PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS
                is_focus_root = root == focus_root
                teacher_budget_available = preflop_teacher_sampling_allowed(
                    root,
                    focus_root,
                    root_teacher_remaining,
                    multi_raise_teacher_remaining,
                    facing_4bet_teacher_remaining,
                    shallow_open_teacher_remaining,
                    shallow_open_teacher_samples,
                    focus_teacher_remaining,
                )
                if teacher_eligible and teacher_budget_available and rng.random() < PREFLOP_3BET_TEACHER_SAMPLE_PROBABILITY:
                    teacher_target, teacher_confidence, teacher_raise_advantage = preflop_3bet_teacher_target(game, learner, target, opponent, learner_player)
                    root_teacher_remaining -= 1
                    if is_multi_raise_root:
                        multi_raise_teacher_remaining = max(0, multi_raise_teacher_remaining - 1)
                    if is_facing_4bet_root:
                        facing_4bet_teacher_remaining = max(0, facing_4bet_teacher_remaining - 1)
                    if is_shallow_open_root:
                        shallow_open_teacher_remaining = max(0, shallow_open_teacher_remaining - 1)
                        shallow_open_teacher_samples[root] += 1
                    if is_focus_root:
                        focus_teacher_remaining -= 1
                features, mask, action, log_prob, value, range_bias, action_fraction, learner_state = network_action(learner, game, player, learner_state)
                path.observations.append(features)
                path.masks.append(mask)
                path.actions.append(action)
                path.log_probs.append(log_prob)
                path.values.append(value)
                path.raise_fractions.append(action_fraction)
                path.streets.append(game.street)
                path.all_in_probability_targets.append(all_in_target)
                path.all_in_calibration_active.append(calibration_active)
                path.preflop_3bet_teacher_targets.append(teacher_target)
                path.preflop_3bet_teacher_confidences.append(teacher_confidence)
                path.preflop_3bet_teacher_eligible.append(teacher_eligible)
                path.preflop_3bet_teacher_raise_advantages.append(teacher_raise_advantage)
                path.preflop_teacher_root_codes.append(PREFLOP_TEACHER_ROOT_CODES.get(root, PREFLOP_TEACHER_UNKNOWN_ROOT_CODE))
                if ENABLE_APPROXIMATE_RESOLVER and not recorded_counterfactual and rng.random() < 0.10:
                    record = external_sample_record(game, player, features, mask, action, action_fraction if action in RAISE_ACTIONS else None, range_bias, action_likelihood, counterfactual_continuation, counterfactual_value_leaf, cfr_iteration, rng, world_samples=2 + curriculum_stage, action_limit=5 + curriculum_stage, depth_limit=8 + curriculum_stage * 2, resolver_iterations=3 + curriculum_stage)
                    if record is not None:
                        cfr_records.append(record)
                        recorded_counterfactual = True
            elif opponent is None:
                action = heuristic_action(game, player)
            elif isinstance(opponent, str):
                action = style_action(game, player, opponent)
            else:
                _, _, action, _, _, _, action_fraction, opponent_state = network_action(opponent, game, player, opponent_state)
            execute_action(game, player, action, action_fraction)
            observed_action = int(game.public_actions[-1].get("action_index", action)) if game.public_actions else action
            likelihood_records.append(ActionLikelihoodRecord(context, [*history, context], hand_bucket(game.hole_cards[player]), observed_action))
            safety += 1
        if not game.hand_complete:
            raise RuntimeError("Strategic self-play hand exceeded the 100-action safety limit")
        path.reward = (game.stacks[learner_player] - game.initial_stack) / game.big_blind
        path.range_label = hand_bucket(game.hole_cards[1 - learner_player])
        if path.actions:
            paths.append(path)
        total_actions += safety
    return RolloutResult(paths, hands, total_actions, cfr_records, likelihood_records, scenario_counts, preflop_root_counts=preflop_root_counts, adversarial_hands=sum(1 for path in paths if path.adversarial))


def collect_rollouts_batched(model_state: dict[str, Tensor], target_state: dict[str, Tensor], opponent_entries: list[dict], hands: int, seed: int, curriculum_stage: int, cfr_iteration: int, best_response_lane: bool = False, adaptive_scenarios: dict[str, float] | None = None, oracle_snapshot: dict | None = None, solver_snapshot: dict | None = None, preflop_root_weights: dict[str, float] | None = None, inference_device: str = "cpu") -> RolloutResult:
    """Interleave independent hands and batch compatible policy decisions per worker."""
    worker_started = perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    use_cuda_inference = inference_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda_inference else "cpu")
    if use_cuda_inference:
        torch.cuda.set_device(device)
    model_sync_started = perf_counter()
    phase_profile = RolloutPhaseProfile()
    learner = cached_inference_model("rollout-learner", model_state, device)
    target = cached_inference_model("rollout-target", target_state, device)
    oracle = AbstractCfrOracle()
    oracle.restore(oracle_snapshot or {})
    abstraction_solver = None
    if ENABLE_ABSTRACT_CFR_TEACHER:
        abstraction_solver = HoldemAbstractionCfr()
        abstraction_solver.restore(solver_snapshot or {})
    opponents: list[PolicyValueNetwork | str] = []
    weights: list[float] = []
    robust_weights: list[float] = []
    adversarial_indexes: list[int] = []
    focused_adversarial_indexes: list[int] = []
    cached_opponent_models = 0
    opponent_revisions: list[str] = []
    for entry in opponent_entries:
        if entry.get("kind") == "style":
            opponents.append(str(entry["style"]))
        else:
            revision = str(entry.get("state_revision", ""))
            state = entry.get("state")
            opponent = cached_inference_model(rollout_opponent_cache_key(entry, len(opponents)), state, device, revision)
            opponents.append(opponent)
            if revision:
                opponent_revisions.append(revision)
            if state is None:
                cached_opponent_models += 1
        weights.append(float(entry.get("weight", 1.0)))
        robust_weights.append(min(1.0 + ROBUST_STYLE_POLICY_WEIGHT, max(1.0, float(entry.get("robust_weight", 1.0)))))
        if bool(entry.get("adversarial")) or entry.get("kind") == "exploiter":
            adversarial_indexes.append(len(opponents) - 1)
        if bool(entry.get("focus")):
            focused_adversarial_indexes.append(len(opponents) - 1)
    model_sync_seconds = perf_counter() - model_sync_started
    arena_setup_started = perf_counter()
    rotating_adversarial_indexes = [index for index in adversarial_indexes if index not in focused_adversarial_indexes]
    stack_size = stack_for_stage(curriculum_stage)
    league_probability = 0.94 if best_response_lane else (0.55, 0.70, 0.84, 0.92)[min(curriculum_stage, 3)]
    paths: list[HandTrajectory] = []
    cfr_records: list[CFRRecord] = []
    likelihood_records: list[ActionLikelihoodRecord] = []
    oracle_records: list[AbstractTeacherRecord] = []
    solver_records: list[SolverTeacherRecord] = []
    scenario_counts = {profile: 0 for profile in SCENARIO_PROFILES}
    preflop_root_counts = {root: 0 for root in PREFLOP_SCENARIO_ROOTS}
    arena_hands: list[ArenaHand] = []
    paired_profiles: dict[int, str] = {}
    paired_opponents: dict[int, PolicyValueNetwork | str | None] = {}
    paired_adversarial: dict[int, bool] = {}
    paired_roots: dict[int, str] = {}
    paired_robust_weights: dict[int, float] = {}
    compiled_transition_actions = 0
    root_teacher_remaining = PREFLOP_3BET_TEACHER_MAX_ROOTS
    multi_raise_teacher_remaining = PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS
    facing_4bet_teacher_remaining = PREFLOP_3BET_TEACHER_FACING_4BET_MIN_ROOTS
    shallow_open_teacher_remaining = PREFLOP_TEACHER_SHALLOW_OPEN_MIN_ROOTS
    shallow_open_teacher_samples = {root: 0 for root in PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS}
    focus_root = focused_preflop_root(preflop_root_weights)
    focus_teacher_remaining = PREFLOP_TEACHER_FOCUS_MIN_ROOTS if focus_root else 0
    pair_count = (hands + 1) // 2
    forced_pair_count = min(pair_count, round(pair_count * ADVERSARIAL_ROLLOUT_FRACTION)) if adversarial_indexes else 0
    forced_adversarial_pairs = set(rng.sample(range(pair_count), forced_pair_count)) if forced_pair_count else set()
    rotating_pair_count = min(len(forced_adversarial_pairs), round(len(forced_adversarial_pairs) * ADVERSARIAL_ROTATION_SHARE))
    rotating_adversarial_pairs = set(rng.sample(list(forced_adversarial_pairs), rotating_pair_count)) if rotating_pair_count and rotating_adversarial_indexes else set()
    for hand_index in range(hands):
        pair_index = hand_index // 2
        if hand_index % 2 == 0:
            profile = scenario_profile(rng, curriculum_stage, adaptive_scenarios)
            force_adversarial = pair_index in forced_adversarial_pairs
            root = sample_preflop_root(rng, preflop_root_weights)
            if force_adversarial:
                candidate_indexes = (
                    rotating_adversarial_indexes
                    if pair_index in rotating_adversarial_pairs
                    else focused_adversarial_indexes
                    if focused_adversarial_indexes and rng.random() < ADVERSARIAL_FOCUS_SHARE
                    else adversarial_indexes
                )
                adversarial_weights = [weights[index] for index in candidate_indexes]
                opponent_index = rng.choices(candidate_indexes, weights=adversarial_weights, k=1)[0]
                opponent = opponents[opponent_index]
                robust_weight = robust_weights[opponent_index]
            else:
                if opponents and rng.random() < league_probability:
                    opponent_index = rng.choices(range(len(opponents)), weights=weights, k=1)[0]
                    opponent = opponents[opponent_index]
                    robust_weight = robust_weights[opponent_index]
                else:
                    opponent = None
                    robust_weight = 1.0
            paired_profiles[pair_index] = profile
            paired_opponents[pair_index] = opponent
            paired_adversarial[pair_index] = force_adversarial
            paired_roots[pair_index] = root
            paired_robust_weights[pair_index] = robust_weight if force_adversarial else 1.0
        else:
            profile = paired_profiles[pair_index]
            opponent = paired_opponents[pair_index]
        root = paired_roots[pair_index]
        hand_stack = 250 if profile == "short_pressure" else stack_size
        learner_player = hand_index % 2
        game = scenario_game(hand_stack, random.Random(seed + pair_index * 7_919), preflop_root_button_offset(root, learner_player), profile)
        if not prepare_preflop_root(game, root):
            root = "blind"
            game = scenario_game(hand_stack, random.Random(seed + pair_index * 7_919), learner_player, profile)
        scenario_counts[profile] += 1
        preflop_root_counts[root] += 1
        opponent_style = opponent if isinstance(opponent, str) else "exploiter" if paired_adversarial[pair_index] else "league"
        trajectory = HandTrajectory([], [], [], [], [], [], 0.0, 0, opponent_style=str(opponent_style), adversarial=paired_adversarial[pair_index], profile=profile, preflop_root=root, robust_weight=paired_robust_weights[pair_index])
        arena_hands.append(ArenaHand(game, learner_player, opponent, trajectory, profile))
    arena = BatchedRolloutArena(arena_hands)
    arena_setup_seconds = perf_counter() - arena_setup_started

    def value_leaf(branch: HeadsUpHoldem, focal_player: int) -> float:
        inputs = torch.tensor([[observation(branch, focal_player)]], dtype=torch.float32, device=device)
        with torch.inference_mode():
            _, _, _, _, _, distribution_logits, _, _ = target(inputs)
            value, uncertainty = value_distribution_moments(distribution_logits[0, 0])
        return float((value - 0.12 * uncertainty).item())

    def action_likelihood(history: list[list[float]]) -> list[list[list[float]]]:
        inputs = torch.tensor([history], dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits = learner.action_likelihood_sequence_logits(inputs)[0]
        return torch.softmax(logits, dim=-1).detach().cpu().tolist()

    def continuation(hand: ArenaHand, branch: HeadsUpHoldem, current: int) -> ActionChoice:
        model = learner if current == hand.learner_player else hand.opponent
        if model is None:
            return ActionChoice(heuristic_action(branch, current))
        if isinstance(model, str):
            return ActionChoice(style_action(branch, current, model))
        key = (current, tuple(round(value, 4) for value in observation(branch, current)))
        cached = hand.cache.get(key)
        if cached is not None:
            return cached
        _, _, action, _, _, _, fraction, _ = network_action(model, branch, current, None, greedy=True)
        choice = ActionChoice(action, fraction if action in RAISE_ACTIONS else None)
        if len(hand.cache) >= 384:
            hand.cache.pop(next(iter(hand.cache)))
        hand.cache[key] = choice
        return choice

    def finish_actions(pending: list[tuple[ArenaHand, int, int, float | None, list[float], list[list[float]]]]) -> None:
        nonlocal compiled_transition_actions
        if not pending:
            return
        rules_started = perf_counter()
        compiled_transition_actions += execute_actions_batch([(hand.game, player, action, fraction) for hand, player, action, fraction, _, _ in pending])
        for hand, player, action, _, context, history in pending:
            observed = int(hand.game.public_actions[-1].get("action_index", action)) if hand.game.public_actions else action
            likelihood_records.append(ActionLikelihoodRecord(context, [*history, context], hand_bucket(hand.game.hole_cards[player]), observed))
            arena.note_action(hand)
        phase_profile.rule_execution_seconds += perf_counter() - rules_started

    play_started = perf_counter()
    while not arena.complete:
        learner_hands = arena.select(lambda hand: hand.game.current_player == hand.learner_player)
        decisions = network_actions_batch(learner, [(hand.game, hand.learner_player, hand.learner_state) for hand in learner_hands], profile=phase_profile)
        learner_pending: list[tuple[ArenaHand, int, int, float | None, list[float], list[list[float]]]] = []
        for hand, decision in zip(learner_hands, decisions):
            features, mask, action, log_prob, value, range_bias, fraction, hand.learner_state = decision
            player = hand.learner_player
            calibration_active, all_in_target, _ = preflop_all_in_calibration(hand.game, player)
            teacher_eligible = preflop_3bet_teacher_eligible(hand.game, player, hand.path.preflop_root, not hand.path.actions)
            teacher_target, teacher_confidence, teacher_raise_advantage = [0.0] * ACTION_COUNT, 0.0, 0.0
            is_multi_raise_root = hand.path.preflop_root in PREFLOP_3BET_TEACHER_MULTI_RAISE_ROOTS
            is_facing_4bet_root = hand.path.preflop_root == "facing_4bet"
            is_shallow_open_root = hand.path.preflop_root in PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS
            is_focus_root = hand.path.preflop_root == focus_root
            teacher_budget_available = preflop_teacher_sampling_allowed(
                hand.path.preflop_root,
                focus_root,
                root_teacher_remaining,
                multi_raise_teacher_remaining,
                facing_4bet_teacher_remaining,
                shallow_open_teacher_remaining,
                shallow_open_teacher_samples,
                focus_teacher_remaining,
            )
            if teacher_eligible and teacher_budget_available and rng.random() < PREFLOP_3BET_TEACHER_SAMPLE_PROBABILITY:
                teacher_target, teacher_confidence, teacher_raise_advantage = preflop_3bet_teacher_target(hand.game, learner, target, hand.opponent, player)
                root_teacher_remaining -= 1
                if is_multi_raise_root:
                    multi_raise_teacher_remaining = max(0, multi_raise_teacher_remaining - 1)
                if is_facing_4bet_root:
                    facing_4bet_teacher_remaining = max(0, facing_4bet_teacher_remaining - 1)
                if is_shallow_open_root:
                    shallow_open_teacher_remaining = max(0, shallow_open_teacher_remaining - 1)
                    shallow_open_teacher_samples[hand.path.preflop_root] += 1
                if is_focus_root:
                    focus_teacher_remaining -= 1
            context = action_context_features(hand.game, player)
            history = [event_context_features(event, hand.game.initial_stack) for event in hand.game.public_actions if int(event.get("action_index", -1)) >= 0]
            hand.path.observations.append(features)
            hand.path.masks.append(mask)
            hand.path.actions.append(action)
            hand.path.log_probs.append(log_prob)
            hand.path.values.append(value)
            hand.path.raise_fractions.append(fraction)
            hand.path.streets.append(hand.game.street)
            hand.path.all_in_probability_targets.append(all_in_target)
            hand.path.all_in_calibration_active.append(calibration_active)
            hand.path.preflop_3bet_teacher_targets.append(teacher_target)
            hand.path.preflop_3bet_teacher_confidences.append(teacher_confidence)
            hand.path.preflop_3bet_teacher_eligible.append(teacher_eligible)
            hand.path.preflop_3bet_teacher_raise_advantages.append(teacher_raise_advantage)
            hand.path.preflop_teacher_root_codes.append(PREFLOP_TEACHER_ROOT_CODES.get(hand.path.preflop_root, PREFLOP_TEACHER_UNKNOWN_ROOT_CODE))
            if ENABLE_HEURISTIC_ORACLE:
                oracle_records.append(oracle.target(hand.game, player, mask, features))
            if abstraction_solver is not None and not hand.recorded_solver_teacher:
                solver_records.append(abstraction_solver.target(hand.game, player, mask, features))
                hand.recorded_solver_teacher = True
            search_probability = (0.10 + 0.07 * curriculum_stage + (0.08 if hand.game.street >= 2 else 0.0) + (0.04 if hand.game.street >= 3 else 0.0)) if ENABLE_APPROXIMATE_RESOLVER else 0.0
            if not hand.recorded_counterfactual and rng.random() < search_probability:
                record = external_sample_record(hand.game, player, features, mask, action, fraction if action in RAISE_ACTIONS else None, range_bias, action_likelihood, lambda branch, current, active_hand=hand: continuation(active_hand, branch, current), value_leaf, cfr_iteration, rng, world_samples=2 + curriculum_stage + int(hand.game.street >= 2), action_limit=5 + curriculum_stage + int(hand.game.street >= 2), depth_limit=9 + curriculum_stage * 3 + hand.game.street * 2, resolver_iterations=3 + curriculum_stage + hand.game.street)
                if record is not None:
                    cfr_records.append(record)
                    hand.recorded_counterfactual = True
            learner_pending.append((hand, player, action, fraction, context, history))
        finish_actions(learner_pending)

        simple_hands = arena.select(lambda hand: hand.game.current_player != hand.learner_player and not isinstance(hand.opponent, PolicyValueNetwork))
        simple_pending: list[tuple[ArenaHand, int, int, float | None, list[float], list[list[float]]]] = []
        for hand in simple_hands:
            player = hand.game.current_player
            assert player is not None
            context = action_context_features(hand.game, player)
            history = [event_context_features(event, hand.game.initial_stack) for event in hand.game.public_actions if int(event.get("action_index", -1)) >= 0]
            action = heuristic_action(hand.game, player) if hand.opponent is None else style_action(hand.game, player, hand.opponent)
            simple_pending.append((hand, player, action, None, context, history))
        finish_actions(simple_pending)

        neural_groups: dict[int, tuple[PolicyValueNetwork, list[ArenaHand]]] = {}
        for hand in arena.select(lambda item: item.game.current_player != item.learner_player and isinstance(item.opponent, PolicyValueNetwork)):
            assert isinstance(hand.opponent, PolicyValueNetwork)
            neural_groups.setdefault(id(hand.opponent), (hand.opponent, []))[1].append(hand)
        for opponent_model, grouped_hands in neural_groups.values():
            decisions = network_actions_batch(opponent_model, [(hand.game, int(hand.game.current_player), hand.opponent_state) for hand in grouped_hands], profile=phase_profile)
            opponent_pending: list[tuple[ArenaHand, int, int, float | None, list[float], list[list[float]]]] = []
            for hand, decision in zip(grouped_hands, decisions):
                _, _, action, _, _, _, fraction, hand.opponent_state = decision
                player = hand.game.current_player
                assert player is not None
                context = action_context_features(hand.game, player)
                history = [event_context_features(event, hand.game.initial_stack) for event in hand.game.public_actions if int(event.get("action_index", -1)) >= 0]
                opponent_pending.append((hand, player, action, fraction, context, history))
            finish_actions(opponent_pending)
    play_seconds = perf_counter() - play_started

    total_actions = 0
    for hand in arena.hands:
        hand.path.reward = (hand.game.stacks[hand.learner_player] - hand.game.initial_stack) / hand.game.big_blind
        hand.path.range_label = hand_bucket(hand.game.hole_cards[1 - hand.learner_player])
        if hand.path.actions:
            paths.append(hand.path)
        total_actions += hand.safety
    adversarial_hands = sum(1 for hand_index in range(hands) if paired_adversarial.get(hand_index // 2, False))
    return RolloutResult(
        paths,
        hands,
        total_actions,
        cfr_records,
        likelihood_records,
        scenario_counts,
        preflop_root_counts=preflop_root_counts,
        oracle_records=oracle_records,
        solver_records=solver_records,
        paired_hands=hands // 2 * 2,
        adversarial_hands=adversarial_hands,
        compiled_transition_actions=compiled_transition_actions,
        model_sync_seconds=model_sync_seconds,
        arena_setup_seconds=arena_setup_seconds,
        tensor_preparation_seconds=phase_profile.tensor_preparation_seconds,
        inference_dispatch_seconds=phase_profile.inference_dispatch_seconds,
        action_postprocess_seconds=phase_profile.action_postprocess_seconds,
        rule_execution_seconds=phase_profile.rule_execution_seconds,
        play_seconds=play_seconds,
        worker_seconds=perf_counter() - worker_started,
        cached_opponent_models=cached_opponent_models,
        opponent_revisions=opponent_revisions,
    )


def _load_evaluation_models(candidate_state: dict[str, Tensor], opponent_state: dict[str, Tensor]) -> tuple[PolicyValueNetwork, PolicyValueNetwork]:
    candidate = PolicyValueNetwork()
    opponent = PolicyValueNetwork()
    candidate.load_state_dict(candidate_state)
    opponent.load_state_dict(opponent_state)
    candidate.eval()
    opponent.eval()
    return candidate, opponent


def _evaluate_pair_with_models(candidate: PolicyValueNetwork, opponent: PolicyValueNetwork, hands: int, seed: int, curriculum_stage: int, profile: str = "balanced") -> MatchResult:
    """Run one fixed-seed paired match using each model's learned mixed policy."""
    stack_size = stack_for_stage(curriculum_stage)
    games = [scenario_game(250 if profile == "short_pressure" else stack_size, random.Random(seed + (hand // 2) * 17), hand % 2, profile) for hand in range(hands)]
    candidate_players = [hand % 2 for hand in range(hands)]
    candidate_histories: list[PolicyState | None] = [None] * hands
    opponent_histories: list[PolicyState | None] = [None] * hands
    safety = [0] * hands
    while any(not game.hand_complete for game in games):
        candidate_rows = [index for index, game in enumerate(games) if not game.hand_complete and game.current_player == candidate_players[index]]
        candidate_decisions = [(games[index], candidate_players[index], candidate_histories[index]) for index in candidate_rows]
        for index, decision in zip(candidate_rows, network_actions_batch(candidate, candidate_decisions, sample_raise=False)):
            _, _, action, _, _, _, raise_fraction, candidate_histories[index] = decision
            execute_action(games[index], candidate_players[index], action, raise_fraction)
            safety[index] += 1
        opponent_rows = [index for index, game in enumerate(games) if not game.hand_complete and game.current_player is not None and game.current_player != candidate_players[index]]
        opponent_decisions = [(games[index], int(games[index].current_player), opponent_histories[index]) for index in opponent_rows]
        for index, decision in zip(opponent_rows, network_actions_batch(opponent, opponent_decisions, sample_raise=False)):
            _, _, action, _, _, _, raise_fraction, opponent_histories[index] = decision
            player = games[index].current_player
            assert player is not None
            execute_action(games[index], player, action, raise_fraction)
            safety[index] += 1
        if any(value >= 100 and not games[index].hand_complete for index, value in enumerate(safety)):
            raise RuntimeError("Direct evaluation hand exceeded the 100-action safety limit")
    result = MatchResult()
    for game, candidate_player in zip(games, candidate_players):
        reward = (game.stacks[candidate_player] - game.initial_stack) / game.big_blind
        result.reward += reward
        result.returns_bb.append(reward)
        if game.winner is None:
            result.ties += 1
        elif game.winner == candidate_player:
            result.wins += 1
        else:
            result.losses += 1
    return result


def evaluate_pair(candidate_state: dict[str, Tensor], opponent_state: dict[str, Tensor], hands: int, seed: int, curriculum_stage: int, profile: str = "balanced") -> MatchResult:
    """Run paired-seat evaluation with batched policy inference across hands."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    candidate, opponent = _load_evaluation_models(candidate_state, opponent_state)
    return _evaluate_pair_with_models(candidate, opponent, hands, seed, curriculum_stage, profile)


def sequential_evaluate_pair(candidate_state: dict[str, Tensor], opponent_state: dict[str, Tensor], minimum_hands: int, maximum_hands: int, seed: int, curriculum_stage: int) -> MatchResult:
    """Spend extra promotion hands only while paired chip-EV remains ambiguous."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    candidate, opponent = _load_evaluation_models(candidate_state, opponent_state)
    aggregate = MatchResult()
    batch_index = 0
    while aggregate.hands < maximum_hands:
        batch_size = min(16, maximum_hands - aggregate.hands)
        batch_seed = seed + batch_index * 1_009
        torch.manual_seed(batch_seed)
        current = _evaluate_pair_with_models(candidate, opponent, batch_size, batch_seed, curriculum_stage)
        aggregate.wins += current.wins
        aggregate.losses += current.losses
        aggregate.ties += current.ties
        aggregate.reward += current.reward
        aggregate.returns_bb.extend(current.returns_bb)
        batch_index += 1
        if aggregate.hands < minimum_hands:
            continue
        lower, upper = bootstrap_bb_per_100_bounds(aggregate, seed + batch_index * 5_003)
        if lower >= 0.0 or upper <= 0.0:
            break
    return aggregate


def behavioral_policy_audit(candidate_state: dict[str, Tensor], champion_state: dict[str, Tensor], seed: int, curriculum_stage: int, states: int = BEHAVIORAL_AUDIT_STATES) -> dict[str, float | int]:
    """Measure observable candidate change on fixed legal roots, separate from chip EV."""
    torch.set_num_threads(1)
    candidate = PolicyValueNetwork()
    champion = PolicyValueNetwork()
    candidate.load_state_dict(candidate_state)
    champion.load_state_dict(champion_state)
    candidate.eval()
    champion.eval()
    agreements = 0
    raise_deltas: list[float] = []
    for index in range(states):
        profile = SCENARIO_PROFILES[index % len(SCENARIO_PROFILES)]
        stack = 250 if profile == "short_pressure" else stack_for_stage(curriculum_stage)
        game = scenario_game(stack, random.Random(seed + index * 97), index % 2, profile)
        player = index % 2
        _, _, candidate_action, _, _, _, candidate_fraction, _ = network_action(candidate, game, player, None, greedy=True)
        _, _, champion_action, _, _, _, champion_fraction, _ = network_action(champion, game, player, None, greedy=True)
        agreements += int(candidate_action == champion_action)
        if candidate_action in RAISE_ACTIONS and champion_action in RAISE_ACTIONS:
            raise_deltas.append(abs(candidate_fraction - champion_fraction))
    return {
        "states": states,
        "action_agreement": agreements / max(1, states),
        "action_change_rate": 1.0 - agreements / max(1, states),
        "raise_fraction_delta": sum(raise_deltas) / max(1, len(raise_deltas)),
        "matched_raises": len(raise_deltas),
    }


def preflop_population_behavior_audit(candidate_state: dict[str, Tensor], seed: int, curriculum_stage: int, hands_per_root: int = 16) -> dict[str, float | int]:
    """Fresh fixed-seed mixed-policy preflight used before selecting a member."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    candidate = PolicyValueNetwork()
    candidate.load_state_dict(candidate_state)
    candidate.eval()
    decisions: list[tuple[HeadsUpHoldem, int, PolicyState | None]] = []
    stack = stack_for_stage(curriculum_stage)
    for root_index, root in enumerate(PREFLOP_FORCED_ROOTS):
        for hand in range(max(4, hands_per_root)):
            player = hand % 2
            game = scenario_game(
                stack,
                random.Random(seed + root_index * 10_007 + hand * 97),
                preflop_root_button_offset(root, player),
                "balanced",
            )
            if prepare_preflop_root(game, root) and game.current_player == player:
                decisions.append((game, player, None))
    actions = [decision[2] for decision in network_actions_batch(candidate, decisions, sample_raise=False)]
    total = max(1, len(actions))
    return {
        "states": len(actions),
        "fold_rate": sum(action == 0 for action in actions) / total,
        "call_rate": sum(action == 1 for action in actions) / total,
        "raise_rate": sum(action == 2 for action in actions) / total,
        "all_in_rate": sum(action == 3 for action in actions) / total,
    }


def preflop_sizing_audit(candidate_state: dict[str, Tensor], seed: int, curriculum_stage: int, roots: int = PREFLOP_SIZING_AUDIT_HANDS) -> dict[str, float | int]:
    """Audit deployed mixed-policy normal-open and capped 3-bet sizing."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    candidate = PolicyValueNetwork()
    candidate.load_state_dict(candidate_state)
    candidate.eval()
    open_sizes_bb: list[float] = []
    cap_hits = 0
    oversized_opens = 0
    all_ins = 0
    three_bet_sizes_to_pot: list[float] = []
    three_bet_cap_hits = 0
    three_bet_cap_violations = 0
    three_bet_minimum_overrides = 0
    three_bet_all_ins = 0
    for index in range(roots):
        game = scenario_game(stack_for_stage(curriculum_stage), random.Random(seed + index * 149), index % 2, "balanced")
        player = game.current_player
        assert player is not None
        _, _, action, _, _, _, fraction, _ = network_action(candidate, game, player, None, sample_raise=False)
        if action == 3:
            all_ins += 1
            continue
        if action not in RAISE_ACTIONS:
            continue
        minimum, maximum = normal_raise_bounds(game, player)
        target = continuous_raise_target(game, player, fraction)
        size_bb = target / max(1, game.big_blind)
        open_sizes_bb.append(size_bb)
        cap_hits += int(maximum < int(game.legal_actions(player)["raise_max"]) and target >= maximum)
        oversized_opens += int(size_bb > 4.0)
    for index in range(roots):
        game = scenario_game(stack_for_stage(curriculum_stage), random.Random(seed + 50_003 + index * 151), index % 2, "balanced")
        opener = game.current_player
        assert opener is not None
        opening_legal = game.legal_actions(opener)
        open_to = min(int(opening_legal["raise_max"]), max(int(opening_legal["raise_min"]), game.big_blind * 3))
        game.act(opener, "raise", open_to)
        player = game.current_player
        assert player is not None
        _, _, action, _, _, _, fraction, _ = network_action(candidate, game, player, None, sample_raise=False)
        if action == 3:
            three_bet_all_ins += 1
            continue
        if action not in RAISE_ACTIONS:
            continue
        legal = game.legal_actions(player)
        legal_maximum = int(legal["raise_max"])
        minimum, maximum = normal_raise_bounds(game, player)
        raw_cap = round(game.pot * PREFLOP_THREE_BET_POT_CAP_MULTIPLIER)
        target = continuous_raise_target(game, player, fraction)
        three_bet_sizes_to_pot.append(target / max(1, game.pot))
        three_bet_cap_hits += int(maximum < legal_maximum and target >= maximum)
        three_bet_minimum_overrides += int(minimum > raw_cap)
        three_bet_cap_violations += int(minimum <= raw_cap and target > raw_cap)
    ordered_sizes = sorted(open_sizes_bb)
    p95_index = max(0, math.ceil(len(ordered_sizes) * 0.95) - 1)
    ordered_three_bet_sizes = sorted(three_bet_sizes_to_pot)
    three_bet_p95_index = max(0, math.ceil(len(ordered_three_bet_sizes) * 0.95) - 1)
    return {
        "roots": roots,
        "normal_raises": len(open_sizes_bb),
        "normal_raise_rate": len(open_sizes_bb) / max(1, roots),
        "mean_raise_bb": sum(open_sizes_bb) / max(1, len(open_sizes_bb)),
        "p95_raise_bb": ordered_sizes[p95_index] if ordered_sizes else 0.0,
        "oversized_open_rate": oversized_opens / max(1, len(open_sizes_bb)),
        "cap_hit_rate": cap_hits / max(1, len(open_sizes_bb)),
        "all_in_rate": all_ins / max(1, roots),
        "three_bet_roots": roots,
        "three_bet_normal_raises": len(three_bet_sizes_to_pot),
        "three_bet_normal_raise_rate": len(three_bet_sizes_to_pot) / max(1, roots),
        "three_bet_mean_raise_to_pot": sum(three_bet_sizes_to_pot) / max(1, len(three_bet_sizes_to_pot)),
        "three_bet_p95_raise_to_pot": ordered_three_bet_sizes[three_bet_p95_index] if ordered_three_bet_sizes else 0.0,
        "three_bet_cap_hit_rate": three_bet_cap_hits / max(1, len(three_bet_sizes_to_pot)),
        "three_bet_over_cap_rate": three_bet_cap_violations / max(1, len(three_bet_sizes_to_pot)),
        "three_bet_minimum_override_rate": three_bet_minimum_overrides / max(1, len(three_bet_sizes_to_pot)),
        "three_bet_all_in_rate": three_bet_all_ins / max(1, roots),
    }


def style_action(game: HeadsUpHoldem, player: int, style: str) -> int:
    """Fixed benchmark archetypes with intentionally different, reproducible leaks."""
    legal = game.legal_actions(player)
    ranks = sorted((card[0] for card in game.hole_cards[player]), reverse=True)
    pair = ranks[0] == ranks[1]
    strong = pair or ranks[0] >= 12
    facing = game.to_call(player) > 0
    if style == "calling_station":
        if legal.get("check") or legal.get("call"):
            return 1
        return 2
    if style == "loose_aggressive":
        if legal.get("raise") and (strong or not facing):
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "trapper":
        if strong and legal.get("check"):
            return 1
        if strong and legal.get("raise"):
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "pressure":
        if legal.get("raise") and (not facing or strong):
            return 2
        if facing and not strong:
            return 0
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "nit":
        if facing and not (pair or ranks[0] >= 13):
            return 0
        if legal.get("raise") and pair and ranks[0] >= 11:
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "maniac":
        if legal.get("raise"):
            return 2
        return 1 if legal.get("check") or legal.get("call") else 0
    if style == "river_hunter":
        if game.street == 3 and legal.get("raise") and (strong or not facing):
            return 2
        if facing and game.street == 3 and not strong:
            return 0
        return 1 if legal.get("check") or legal.get("call") else 0
    # Tight-aggressive default.
    if facing and not strong:
        return 0
    if strong and legal.get("raise"):
        return 2
    return 1 if legal.get("check") or legal.get("call") else 0


def _record_evaluation_outcome(result: MatchResult, game: HeadsUpHoldem, candidate_player: int) -> None:
    reward = (game.stacks[candidate_player] - game.initial_stack) / game.big_blind
    result.reward += reward
    result.returns_bb.append(reward)
    if game.winner is None:
        result.ties += 1
    elif game.winner == candidate_player:
        result.wins += 1
    else:
        result.losses += 1


def evaluate_style(candidate_state: dict[str, Tensor], style: str, hands: int, seed: int, curriculum_stage: int) -> MatchResult:
    """Run fixed-seed mixed-policy hands against one scripted style."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    candidate = PolicyValueNetwork()
    candidate.load_state_dict(candidate_state)
    candidate.eval()
    stack_size = stack_for_stage(curriculum_stage)
    games = [scenario_game(stack_size, random.Random(seed + (hand // 2) * 23), hand % 2, "balanced") for hand in range(hands)]
    candidate_players = [hand % 2 for hand in range(hands)]
    histories: list[PolicyState | None] = [None] * hands
    safety = [0] * hands
    while any(not game.hand_complete for game in games):
        candidate_rows = [index for index, game in enumerate(games) if not game.hand_complete and game.current_player == candidate_players[index]]
        decisions = [(games[index], candidate_players[index], histories[index]) for index in candidate_rows]
        for index, decision in zip(candidate_rows, network_actions_batch(candidate, decisions, sample_raise=False)):
            _, _, action, _, _, _, raise_fraction, histories[index] = decision
            execute_action(games[index], candidate_players[index], action, raise_fraction)
            safety[index] += 1
        opponent_rows = [index for index, game in enumerate(games) if not game.hand_complete and game.current_player is not None and game.current_player != candidate_players[index]]
        for index in opponent_rows:
            player = games[index].current_player
            assert player is not None
            execute_action(games[index], player, style_action(games[index], player, style), None)
            safety[index] += 1
        if any(value >= 100 and not games[index].hand_complete for index, value in enumerate(safety)):
            raise RuntimeError("Benchmark hand exceeded the 100-action safety limit")
    result = MatchResult()
    for game, candidate_player in zip(games, candidate_players):
        _record_evaluation_outcome(result, game, candidate_player)
    return result


def evaluate_preflop_scenario(candidate_state: dict[str, Tensor], style: str, root: str, hands: int, seed: int, curriculum_stage: int, inference_device: str = "cpu") -> dict[str, object]:
    """Evaluate one fixed public preflop root with the learned mixed policy."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    use_cuda = inference_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
        candidate = cached_inference_model("evaluation-preflop-candidate", candidate_state, device)
    else:
        candidate = PolicyValueNetwork()
        candidate.load_state_dict(candidate_state)
        candidate.eval()
    stack_size = stack_for_stage(curriculum_stage)
    candidate_players = [hand % 2 for hand in range(hands)]
    games = [scenario_game(stack_size, random.Random(seed + (hand // 2) * 29), preflop_root_button_offset(root, candidate_player), "balanced") for hand, candidate_player in enumerate(candidate_players)]
    for game in games:
        if not prepare_preflop_root(game, root):
            raise RuntimeError(f"Unable to construct preflop audit root {root}")
    prefix_actions = [len(game.public_actions) for game in games]
    histories: list[PolicyState | None] = [None] * hands
    first_candidate_decisions = [True] * hands
    safety = [0] * hands
    action_counts = {"fold": 0, "check_call": 0, "raise": 0, "all_in": 0}
    first_action_counts = {"fold": 0, "check_call": 0, "raise": 0, "all_in": 0}
    first_policy_probability_totals = {"fold": 0.0, "check_call": 0.0, "raise": 0.0, "all_in": 0.0}
    first_commitments: list[float] = []
    first_calibration_targets: list[float] = []
    first_calibration_eligible = 0
    first_calibrated_all_ins = 0
    terminal_streets = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
    raise_sizes_bb: list[float] = []
    while any(not game.hand_complete for game in games):
        candidate_rows = [index for index, game in enumerate(games) if not game.hand_complete and game.current_player == candidate_players[index]]
        first_calibrations = {
            index: preflop_all_in_calibration(games[index], candidate_players[index])
            for index in candidate_rows
            if first_candidate_decisions[index]
        }
        decisions = [(games[index], candidate_players[index], histories[index]) for index in candidate_rows]
        first_candidate_rows = [index for index in candidate_rows if first_candidate_decisions[index]]
        first_policy_probabilities = network_policy_probabilities_batch(candidate, [(games[index], candidate_players[index], histories[index]) for index in first_candidate_rows])
        first_policy_by_row = dict(zip(first_candidate_rows, first_policy_probabilities))
        for index, decision in zip(candidate_rows, network_actions_batch(candidate, decisions, sample_raise=False)):
            _, _, action, _, _, _, raise_fraction, histories[index] = decision
            if first_candidate_decisions[index]:
                calibration_active, all_in_target, commitment = first_calibrations[index]
                action_name = "fold" if action == 0 else "check_call" if action == 1 else "raise" if action == 2 else "all_in"
                first_action_counts[action_name] += 1
                for action_index, name in enumerate(("fold", "check_call", "raise", "all_in")):
                    first_policy_probability_totals[name] += float(first_policy_by_row[index][action_index])
                first_commitments.append(commitment)
                if calibration_active:
                    first_calibration_eligible += 1
                    first_calibration_targets.append(all_in_target)
                    first_calibrated_all_ins += int(action == 3)
                first_candidate_decisions[index] = False
            execute_action(games[index], candidate_players[index], action, raise_fraction)
            safety[index] += 1
        opponent_rows = [index for index, game in enumerate(games) if not game.hand_complete and game.current_player is not None and game.current_player != candidate_players[index]]
        for index in opponent_rows:
            player = games[index].current_player
            assert player is not None
            execute_action(games[index], player, style_action(games[index], player, style), None)
            safety[index] += 1
        if any(value >= 100 and not games[index].hand_complete for index, value in enumerate(safety)):
            raise RuntimeError("Preflop scenario audit hand exceeded the 100-action safety limit")
    result = MatchResult()
    for index, (game, candidate_player) in enumerate(zip(games, candidate_players)):
        _record_evaluation_outcome(result, game, candidate_player)
        terminal_streets[("preflop", "flop", "turn", "river")[game.street]] += 1
        for event in game.public_actions[prefix_actions[index]:]:
            if int(event.get("player", -1)) != candidate_player or int(event.get("action_index", -1)) < 0:
                continue
            action_index = int(event["action_index"])
            action_name = "fold" if action_index == 0 else "check_call" if action_index == 1 else "raise" if action_index == 2 else "all_in"
            action_counts[action_name] += 1
            if action_index in RAISE_ACTIONS:
                raise_sizes_bb.append(float(event.get("amount", 0)) / max(1, game.big_blind))
    lower, upper = bootstrap_bb_per_100_bounds(result, seed + 811)
    decisions = sum(action_counts.values())
    first_decisions = sum(first_action_counts.values())
    return {
        "hands": result.hands,
        "bb_per_100": result.bb_per_100,
        "bb_per_100_lower": lower,
        "bb_per_100_upper": upper,
        "win_rate": result.score,
        "action_mix": {name: count / max(1, decisions) for name, count in action_counts.items()},
        "first_decision": {
            "decisions": first_decisions,
            "all_in_actions": first_action_counts["all_in"],
            "action_mix": {name: count / max(1, first_decisions) for name, count in first_action_counts.items()},
            "policy_probability": {name: value / max(1, first_decisions) for name, value in first_policy_probability_totals.items()},
            "mean_commitment": sum(first_commitments) / max(1, len(first_commitments)),
            "calibration_eligible_decisions": first_calibration_eligible,
            "calibration_eligible_rate": first_calibration_eligible / max(1, first_decisions),
            "calibrated_all_in_actions": first_calibrated_all_ins,
            "calibrated_all_in_rate": first_calibrated_all_ins / max(1, first_calibration_eligible),
            "all_in_target": sum(first_calibration_targets) / max(1, len(first_calibration_targets)),
        },
        "terminal_streets": terminal_streets,
        "mean_raise_bb": sum(raise_sizes_bb) / max(1, len(raise_sizes_bb)),
        "p95_raise_bb": sorted(raise_sizes_bb)[max(0, math.ceil(len(raise_sizes_bb) * 0.95) - 1)] if raise_sizes_bb else 0.0,
    }


def evaluate_preflop_scenario_cuda(candidate_state: dict[str, Tensor], style: str, root: str, hands: int, seed: int, curriculum_stage: int) -> dict[str, object]:
    """Run a fixed preflop audit on the single coordinated CUDA inference lane."""
    try:
        return evaluate_preflop_scenario(candidate_state, style, root, hands, seed, curriculum_stage, "cuda")
    except RuntimeError as exc:
        # Evaluation is evidence, not an optional accelerator. A VRAM failure
        # must run the identical CPU audit rather than skip or weaken its gate.
        log_training_debug("cuda_preflop_evaluation_fallback", error=str(exc), root=root, style=style)
        return evaluate_preflop_scenario(candidate_state, style, root, hands, seed, curriculum_stage, "cpu")


def restricted_best_response_bb_per_100(candidate_state: dict[str, Tensor], seed: int, curriculum_stage: int) -> float:
    """A reproducible restricted best-response lower bound, not a claim of exact exploitability."""
    results = [evaluate_style(candidate_state, style, ADVERSARIAL_EVALUATION_HANDS, seed + index * 2_003, curriculum_stage) for index, style in enumerate(ADVERSARIAL_TRAINING_STYLES)]
    return min(result.reward / max(1, result.hands) * 100 for result in results)


def audit_exploitability_proxy_bb_per_100(candidate_state: dict[str, Tensor], seed: int, curriculum_stage: int) -> float:
    """Independent action-abstraction audit, not a claim of exact exploitability."""
    results = [evaluate_style(candidate_state, style, ADVERSARIAL_EVALUATION_HANDS, seed + index * 3_007, curriculum_stage) for index, style in enumerate(AUDIT_STYLES)]
    return min(result.reward / max(1, result.hands) * 100 for result in results)


@dataclass
class OpponentProfile:
    """Across-hand, showdown-safe opponent statistics for live play."""

    actions: int = 0
    raises: int = 0
    calls_or_checks: int = 0
    folds: int = 0
    showdown_strength_total: float = 0.0
    showdowns: int = 0
    _last_hand: tuple | None = None

    @property
    def confidence(self) -> float:
        return min(1.0, self.actions / 80.0)

    @property
    def raise_rate(self) -> float:
        return self.raises / max(1, self.actions)

    @property
    def showdown_strength(self) -> float | None:
        return self.showdown_strength_total / self.showdowns if self.showdowns else None

    def observe_completed_hand(self, game: HeadsUpHoldem, opponent: int) -> None:
        fingerprint = (game.hand_number, game.result, tuple((event["player"], event["action"], event["amount"], event["street"]) for event in game.public_actions))
        if fingerprint == self._last_hand:
            return
        self._last_hand = fingerprint
        for event in game.public_actions:
            if event["player"] != opponent or event["action"] == "blind":
                continue
            self.actions += 1
            if event["action"] == "raise":
                self.raises += 1
            elif event["action"] in {"check", "call"}:
                self.calls_or_checks += 1
            elif event["action"] == "fold":
                self.folds += 1
        # Opponent cards are read only after a real showdown exposed them.
        if game.showdown_scores:
            ranks = sorted((card[0] for card in game.hole_cards[opponent]), reverse=True)
            pair = float(ranks[0] == ranks[1])
            suited = float(game.hole_cards[opponent][0][1] == game.hole_cards[opponent][1][1])
            strength = min(1.0, (ranks[0] + ranks[1]) / 28 + 0.20 * pair + 0.04 * suited)
            self.showdown_strength_total += strength
            self.showdowns += 1

    def adapt_range_bias(self, game: HeadsUpHoldem, player: int, range_bias: list[float]) -> list[float]:
        """Apply a conservative posterior tilt from public, persistent evidence."""
        if self.confidence < 0.10 or len(range_bias) != RANGE_BUCKETS:
            return range_bias
        observed_strength = self.showdown_strength if self.showdown_strength is not None else 0.62
        aggression_shift = max(-0.20, min(0.20, self.raise_rate - 0.28))
        strength_shift = max(-0.18, min(0.18, observed_strength - 0.62))
        tilt = (strength_shift - aggression_shift) * self.confidence
        known = set(game.hole_cards[player] + game.community)
        adjusted = list(range_bias)
        for first, first_card in enumerate(new_deck()):
            if first_card in known:
                continue
            for second_card in new_deck()[first + 1:]:
                if second_card in known:
                    continue
                ranks = sorted((first_card[0], second_card[0]), reverse=True)
                pair = float(ranks[0] == ranks[1])
                suited = float(first_card[1] == second_card[1])
                strength = (ranks[0] + ranks[1]) / 28 + 0.35 * pair + 0.07 * suited
                bucket = hand_bucket((first_card, second_card))
                adjusted[bucket] *= math.exp(2.2 * tilt * (strength - 0.62))
        total = sum(adjusted)
        return [value / total for value in adjusted] if total > 1e-12 else range_bias

    def policy_adjustment(self, game: HeadsUpHoldem, player: int) -> list[float]:
        """Small exploitative tilt, capped so the blueprint remains dominant."""
        adjustment = [0.0] * ACTION_COUNT
        if self.confidence < 0.20 or game.to_call(player) <= 0:
            return adjustment
        pressure = min(1.0, game.to_call(player) / max(1, game.pot + game.to_call(player)))
        aggression = max(-0.18, min(0.18, self.raise_rate - 0.28)) * self.confidence
        adjustment[1] = aggression * (0.45 + 0.55 * pressure)
        adjustment[0] = -aggression * (0.30 + 0.40 * pressure)
        return adjustment


class NeuralAgent:
    """Thread-safe recurrent champion deployed to the human-facing table."""

    def __init__(self) -> None:
        self.model = PolicyValueNetwork()
        self.model.eval()
        self._strategy_models: list[PolicyValueNetwork] = []
        self._history: list[list[float]] = []
        self._hand_key: tuple | None = None
        self._raise_fraction: float | None = None
        self.ready = False
        self.resolver_uses = 0
        self.resolver_depth = 0
        self.search_leaf_evaluations = 0
        self.search_value_spread = 0.0
        self.search_confidence = 0.0
        self.search_action_width = 0
        self.search_endgame_worlds = 0
        self.search_safety_rejections = 0
        self.search_safety_margin = 0.0
        self.search_safety_confidence = 0.0
        self.search_confident_actions = 0
        self.search_iterations = 0
        self.search_strategy_peak = 0.0
        self.opponent_profile = OpponentProfile()
        self.average_strategy_weight = 0.0
        self._lock = RLock()

    def select(self, game: HeadsUpHoldem, player: int) -> int:
        with self._lock:
            if not self.ready:
                return heuristic_action(game, player)
            hand_key = (game.hand_number, *game.hole_cards[player])
            if hand_key != self._hand_key:
                self._history, self._hand_key, self._raise_fraction = [], hand_key, None
            current_observation = observation(game, player)
            features = torch.tensor([[*self._history[-(POLICY_HISTORY - 1):], current_observation]], dtype=torch.float32)
            mask = torch.tensor([[legal_action_mask(game, player)]], dtype=torch.bool)
            with torch.inference_mode():
                logits, average_logits, _, range_logits, advantage_logits, value_distribution_logits, raise_shapes, _ = self.model(features)
                average_weight = self.average_strategy_weight
                current_policy = (1.0 - average_weight) * logits[:, -1] + average_weight * average_logits[:, -1]
                current_policy = current_policy + torch.tensor(self.opponent_profile.policy_adjustment(game, player), dtype=current_policy.dtype).unsqueeze(0)
                raise_fractions = [0.5] * ACTION_COUNT
                raise_proposals: dict[int, list[float]] = {}
                for action in RAISE_ACTIONS:
                    raise_fractions[action] = float(raise_distribution(raise_shapes[0, -1], action).mean.clamp(0.005, 0.995).item())
                    raise_proposals[action] = raise_size_proposals(raise_shapes[0, -1], action)
                if self._strategy_models:
                    snapshot_policies: list[Tensor] = []
                    snapshot_sizes: dict[int, list[float]] = {action: [] for action in RAISE_ACTIONS}
                    for snapshot in self._strategy_models:
                        snapshot_logits, snapshot_average, _, _, _, _, snapshot_shapes, _ = snapshot(features)
                        snapshot_policies.append((1.0 - average_weight) * snapshot_logits[:, -1] + average_weight * snapshot_average[:, -1])
                        for action in RAISE_ACTIONS:
                            snapshot_sizes[action].append(float(raise_distribution(snapshot_shapes[0, -1], action).mean.clamp(0.005, 0.995).item()))
                    policy_logits = 0.64 * current_policy + 0.36 * torch.stack(snapshot_policies).mean(dim=0)
                    for action in RAISE_ACTIONS:
                        raise_fractions[action] = 0.70 * raise_fractions[action] + 0.30 * (sum(snapshot_sizes[action]) / len(snapshot_sizes[action]))
                else:
                    policy_logits = current_policy
                # Heads-up poker strategies must randomize.  Collapsing PPO's
                # learned distribution to argmax makes balanced ranges readable
                # and previously turned healthy 4-bet defense into mass folding.
                choice = int(masked_distribution(policy_logits, mask[:, 0]).sample().item())
                if choice in RAISE_ACTIONS:
                    self._raise_fraction = raise_fractions[choice]
                else:
                    self._raise_fraction = None
            self._history = [*self._history, current_observation][-POLICY_HISTORY:]
            call_amount = game.to_call(player)
            strategic_pressure = call_amount >= max(game.big_blind * 2, game.pot * 0.35) or (game.street >= 2 and len(game.public_actions) >= 4)
            if strategic_pressure and ENABLE_APPROXIMATE_RESOLVER:
                range_bias = self.opponent_profile.adapt_range_bias(game, player, torch.softmax(range_logits[0, -1], dim=-1).tolist())
                legal = legal_action_mask(game, player)
                value_uncertainty = float(value_distribution_moments(value_distribution_logits[0, -1])[1].item())

                def action_likelihood(history: list[list[float]]) -> list[list[list[float]]]:
                    inputs = torch.tensor([history], dtype=torch.float32)
                    with torch.no_grad():
                        likelihood_logits = self.model.action_likelihood_sequence_logits(inputs)[0]
                    return torch.softmax(likelihood_logits, dim=-1).tolist()

                def continuation(branch: HeadsUpHoldem, current: int) -> ActionChoice:
                    _, _, branch_action, _, _, _, branch_fraction, _ = network_action(self.model, branch, current, None, greedy=True)
                    return ActionChoice(branch_action, branch_fraction if branch_action in RAISE_ACTIONS else None)

                def value_leaf(branch: HeadsUpHoldem, focal_player: int) -> float:
                    inputs = torch.tensor([[observation(branch, focal_player)]], dtype=torch.float32)
                    with torch.inference_mode():
                        _, _, _, _, _, distribution_logits, _, _ = self.model(inputs)
                        mean, uncertainty = value_distribution_moments(distribution_logits[0, 0])
                    return float((mean - 0.14 * uncertainty).item())

                def counterfactual_leaf(branch: HeadsUpHoldem, focal_player: int, branch_belief: object) -> tuple[float, float]:
                    inputs = torch.tensor([[observation(branch, focal_player)]], dtype=torch.float32)
                    own_belief_tensor = torch.tensor([private_belief_features(branch.hole_cards[focal_player])], dtype=torch.float32)
                    belief_tensor = torch.tensor([belief_features(branch_belief)], dtype=torch.float32)
                    with torch.inference_mode():
                        values, uncertainty, opponent_values, opponent_uncertainty = self.model.public_belief_values(inputs, own_belief_tensor, belief_tensor)
                    weights = belief_tensor[0, :BELIEF_VALUE_CLASSES]
                    expected = (values[0, 0] * weights).sum()
                    expected_uncertainty = ((uncertainty[0, 0] + opponent_uncertainty[0, 0]) * weights).sum() * 0.5
                    return float(expected.item()), float(expected_uncertainty.item())

                search = robust_belief_search(
                    game,
                    player,
                    policy_logits[0].tolist(),
                    advantage_logits[0, -1].tolist(),
                    legal,
                    raise_fractions,
                    range_bias,
                    action_likelihood,
                    continuation,
                    value_leaf,
                    value_uncertainty,
                    random.Random(game.hand_number * 9_973 + game.street * 131 + len(game.public_actions)),
                    world_samples=2 + min(3, game.street) + min(2, int(value_uncertainty)),
                    depth_limit=4 + min(4, game.street) + int(value_uncertainty >= 1.2),
                    counterfactual_leaf=counterfactual_leaf,
                    resolver_iterations=3 + game.street,
                    raise_proposals=raise_proposals,
                )
                choice = search.action
                if choice in RAISE_ACTIONS:
                    self._raise_fraction = search.raise_fraction if search.raise_fraction is not None else raise_fractions[choice]
                self.resolver_uses += 1
                self.resolver_depth = search.depth
                self.search_leaf_evaluations = search.leaf_evaluations
                self.search_value_spread = search.value_spread
                self.search_confidence = search.confidence
                self.search_action_width = search.width
                self.search_endgame_worlds = search.endgame_worlds
                self.search_safety_rejections = search.safety_rejections
                self.search_safety_margin = search.safety_margin
                self.search_safety_confidence = search.safety_confidence
                self.search_confident_actions = search.confident_actions
                self.search_iterations = search.iterations
                self.search_strategy_peak = search.average_strategy_peak
            return choice

    def execute(self, game: HeadsUpHoldem, player: int, choice: int) -> None:
        execute_action(game, player, choice, self._raise_fraction)

    def observe_completed_hand(self, game: HeadsUpHoldem, player: int) -> None:
        with self._lock:
            if game.hand_complete:
                self.opponent_profile.observe_completed_hand(game, 1 - player)

    def replace_state(self, state: dict[str, Tensor], snapshots: list[dict[str, Tensor]] | None = None, ready: bool = True, average_strategy_weight: float = 0.0) -> None:
        with self._lock:
            self.model.load_state_dict(state)
            self.model.eval()
            self._strategy_models = []
            for snapshot_state in (snapshots or [])[-4:]:
                snapshot = PolicyValueNetwork()
                snapshot.load_state_dict(snapshot_state)
                snapshot.eval()
                self._strategy_models.append(snapshot)
            self._history = []
            self._hand_key = None
            self._raise_fraction = None
            self.average_strategy_weight = min(0.65, max(0.0, float(average_strategy_weight)))
            self.ready = ready

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())


class StrategicLeagueTrainer:
    """PPO learner with deterministic promotion gates and Elo-rated frozen champions."""

    def __init__(self, live_agent: NeuralAgent) -> None:
        self.live_agent = live_agent
        self.runtime = resolve_training_runtime()
        self.model = PolicyValueNetwork().to(self.runtime.device)
        self.target_model = PolicyValueNetwork().to(self.runtime.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer_backend = "adamw"
        self.mixed_precision_enabled = self.runtime.cuda_enabled
        self.amp_overflow_fallbacks = 0
        self._optimizer_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._range_coarse_index = RANGE_COARSE_INDEX.to(self.runtime.device)
        self.optimizer = self._new_optimizer(3e-4)
        self.grad_scaler = self._new_grad_scaler()
        self._lock = RLock()
        self._rng = random.Random()
        self._rollout_cached_opponent_revisions: set[str] = set()
        self.champion_state = clone_state(self.model)
        self.champion_id = "champion-0"
        self.champion_elo = 1_200.0
        self.league: list[dict] = []
        self.exploiters: list[dict] = []
        self.specialist_archive: list[dict] = []
        self.population_members: list[dict] = [
            {"id": "population-balanced", "state": clone_state(self.model), "target_state": clone_state(self.target_model), "optimizer_state": None, "grad_scaler_state": None, "lr_scale": 1.00, "entropy_scale": 1.00, "score": 0.50, "bb_per_100": 0.0, "adversarial_bb_per_100": 0.0, "preflop_worst_lcb_bb_per_100": 0.0, "preflop_allin_probability": 0.0, "quarantine_count": 0, "safety_regressions": 0, "recovery_cooldown_until": 0, "updates": 0},
            {"id": "population-press", "state": clone_state(self.model), "target_state": clone_state(self.target_model), "optimizer_state": None, "grad_scaler_state": None, "lr_scale": 1.16, "entropy_scale": 1.28, "score": 0.50, "bb_per_100": 0.0, "adversarial_bb_per_100": 0.0, "preflop_worst_lcb_bb_per_100": 0.0, "preflop_allin_probability": 0.0, "quarantine_count": 0, "safety_regressions": 0, "recovery_cooldown_until": 0, "updates": 0},
            {"id": "population-robust", "state": clone_state(self.model), "target_state": clone_state(self.target_model), "optimizer_state": None, "grad_scaler_state": None, "lr_scale": 0.82, "entropy_scale": 0.74, "score": 0.50, "bb_per_100": 0.0, "adversarial_bb_per_100": 0.0, "preflop_worst_lcb_bb_per_100": 0.0, "preflop_allin_probability": 0.0, "quarantine_count": 0, "safety_regressions": 0, "recovery_cooldown_until": 0, "updates": 0},
        ]
        self.active_population_index = 0
        self.recovery_anchor_state = clone_state(self.model)
        self.recovery_anchor_target_state = clone_state(self.target_model)
        self.recovery_anchor_score = float("-inf")
        self.recovery_anchor_metrics: dict[str, float] = {}
        self.recovery_anchor_updates = 0
        self.recovery_anchor_source = "unverified"
        self.recovery_safe_audits = 0
        self.recovery_baseline_verified = True
        self.recovery_baseline_metrics: dict[str, float] = {}
        self.last_recovery_candidate_metrics: dict[str, float] = {}
        self.last_fresh_warmup_fold_collapse = False
        self.last_final_audit_checkpoint_restored = False
        self.last_final_audit_restore_reason = ""
        self.recovery_halted = False
        self.recovery_revalidation_required = False
        self._recovery_anchor_probe_cache: dict[int, dict[str, list[list[float]]]] = {}
        self.strategy_snapshots: list[dict] = []
        self.payoff_matrix: dict[str, float] = {}
        self.mixture_regrets: dict[str, float] = {self.champion_id: 0.0, **{f"style-{style}": 0.0 for style in BENCHMARK_STYLES}}
        self.cfr_memory = ReservoirMemory()
        self.strategy_memory = StrategyMemory()
        self.search_value_memory = SearchValueMemory()
        self.counterfactual_value_memory = CounterfactualValueMemory()
        self.action_likelihood_memory = ActionLikelihoodMemory()
        self.imitation_memory = SelfImitationMemory()
        self.hard_spot_value_memory = HardSpotValueMemory()
        self.benchmarks: dict[str, dict[str, float]] = {}
        self.tournament_count = 0
        self.version = 0
        self.updates = 0
        self.trained_hands = 0
        self.resumed = False
        self.last_gate_passed = False
        self.last_challenger_status = "untrained"
        self.last_promotion_confidence = 0.0
        self.last_direct_bb_per_100 = 0.0
        self.last_evaluation_bb_per_100 = 0.0
        self.last_promotion_ci_lower = 0.0
        self.last_promotion_ci_upper = 0.0
        self.last_holdout_bb_per_100 = 0.0
        self.last_holdout_floor_bb_per_100 = 0.0
        self.last_opponent_pressure = 0.0
        self.last_rare_spot_rate = 0.0
        self.last_belief_confidence = 0.0
        self.last_leaf_evaluations = 0
        self.last_best_response_bb_per_100 = 0.0
        self.last_target_drift = 0.0
        self.curriculum_unlocked_stage = 0
        self.last_curriculum_readiness = 0.0
        self.last_training_lane = "population"
        self.last_replay_rare_fraction = 0.0
        self.last_replay_priority = 0.0
        self.last_replay_recent_fraction = 0.0
        self.last_exploiter_diversity = 0.0
        self.last_exploiter_threat = 0.0
        self.last_champion_vulnerability = 0.0
        self.exploiter_generations = 0
        self.exploiter_lane_remaining = 0
        self.ppo_learning_rate = 3e-4
        self.ppo_clip_epsilon = 0.20
        self.ppo_entropy_coefficient = 0.012
        self.ppo_kl_target = 0.012
        self.last_ppo_epochs = 0
        self.last_ppo_clip_fraction = 0.0
        self.ppo_recovery_updates = 0
        self.last_ppo_kl_limited = False
        self.last_ppo_hard_kl = 0.0
        self.last_ppo_epoch_budget = PPO_MAX_EPOCHS
        self.last_ppo_update_reverted = False
        self.last_ppo_rollback_phase = "none"
        self.last_ppo_post_step_retry_applied = False
        self.last_ppo_post_step_retry_accepted = False
        self.last_ppo_post_step_retry_kl = 0.0
        self.last_ppo_root_backoff_applied = False
        self.last_ppo_root_backoff_accepted = False
        self.last_ppo_root_backoff_scale = 0.0
        self.last_preflop_root_guarded = False
        self.last_preflop_root_guard_reason = "none"
        self.last_preflop_root_update_kl = 0.0
        self.last_preflop_root_anchor_kl = 0.0
        self.last_preflop_root_update_action_delta = 0.0
        self.last_preflop_root_anchor_action_delta = 0.0
        self.last_preflop_root_drift_root = "pending"
        self.ppo_post_step_retry_scale = PPO_POST_STEP_RETRY_SCALE
        self.run_ppo_safety = empty_ppo_safety_counters()
        self.last_imitation_loss = 0.0
        self.last_imitation_reward = 0.0
        self.last_evaluation_hands = 0
        self.last_evaluation_seconds = 0.0
        self.last_parallel_evaluation = False
        self.last_holdout_score = 0.0
        self.last_holdout_floor = 0.0
        self.last_continuous_raise_mean = 0.5
        self.last_sizing_cfr_loss = 0.0
        self.last_strategy_memory_size = 0
        self.last_scenario_coverage = 0.0
        self.last_restricted_br_bb_per_100 = 0.0
        self.last_adversarial_floor_bb_per_100 = 0.0
        self.last_adversarial_rollout_fraction = 0.0
        self.run_rollout_hands = 0
        self.run_adversarial_hands = 0
        self._rollout_diagnostic_window: list[tuple[int, int]] = []
        self.last_adversarial_focus = "pending"
        self.adversarial_style_bb_per_100 = {style: 0.0 for style in ADVERSARIAL_TRAINING_STYLES}
        self.last_compiled_transition_fraction = 0.0
        self.last_crossplay_robustness = 0.0
        self.last_population_continuity = 0.0
        self.last_ensemble_disagreement = 0.0
        self.last_search_value_loss = 0.0
        self.last_search_memory_size = 0
        self.last_snapshot_diversity = 0.0
        self.last_snapshot_min_distance = 0.0
        self.snapshot_rejections = 0
        self.last_adaptive_action_width = 0.0
        self.last_adversarial_ci_floor_bb_per_100 = 0.0
        self.last_adversarial_evaluation_hands = ADVERSARIAL_SCREENING_HANDS
        self.last_adversarial_confirmation_hands = 0
        self.last_final_audit_ran = False
        self.last_tail_loss_rate = 0.0
        self.last_tail_loss_bb = 0.0
        self.last_tail_policy_weight = 1.0
        self.last_tail_style_diagnostics: dict[str, dict[str, float | int]] = {}
        self.run_adversarial_paths = 0
        self.run_tail_paths = 0
        self.run_tail_loss_sum = 0.0
        self.run_tail_weight_sum = 0.0
        self.run_tail_style_totals: dict[str, dict[str, Any]] = {}
        self._tail_diagnostic_window: list[tuple[int, int, float, float]] = []
        self.last_hard_spot_value_loss = 0.0
        self.last_hard_spot_memory_size = 0
        self.last_behavior_action_agreement = 1.0
        self.last_behavior_action_change_rate = 0.0
        self.last_behavior_raise_fraction_delta = 0.0
        self.last_behavior_audit_states = 0
        self.last_preflop_sizing_audit: dict[str, float | int] = {"roots": 0, "normal_raises": 0, "normal_raise_rate": 0.0, "mean_raise_bb": 0.0, "p95_raise_bb": 0.0, "oversized_open_rate": 0.0, "cap_hit_rate": 0.0, "all_in_rate": 0.0, "three_bet_roots": 0, "three_bet_normal_raises": 0, "three_bet_normal_raise_rate": 0.0, "three_bet_mean_raise_to_pot": 0.0, "three_bet_p95_raise_to_pot": 0.0, "three_bet_cap_hit_rate": 0.0, "three_bet_over_cap_rate": 0.0, "three_bet_minimum_override_rate": 0.0, "three_bet_all_in_rate": 0.0}
        self.preflop_root_weakness = {root: 0.50 for root in PREFLOP_FORCED_ROOTS}
        self.last_preflop_root_fraction = 0.0
        self.last_preflop_scenario_audit: dict[str, dict[str, dict[str, float | int | dict[str, float]]]] = {}
        self.last_preflop_scenario_audit_hands = 0
        self.last_preflop_scenario_worst_lcb_bb_per_100 = 0.0
        self.last_preflop_scenario_worst_root = "pending"
        self.last_preflop_scenario_worst_style = "pending"
        self.last_preflop_allin_calibration_loss = 0.0
        self.last_preflop_allin_stability_loss = 0.0
        self.last_preflop_guarded_allin_probability = 0.0
        self.last_preflop_allin_target = 0.0
        self.last_preflop_guarded_state_fraction = 0.0
        self.last_preflop_immediate_allin_rate = 0.0
        self.last_preflop_immediate_allin_target = 0.0
        self.last_preflop_immediate_eligible_rate = 0.0
        self.last_preflop_3bet_teacher_loss = 0.0
        self.last_preflop_3bet_teacher_eligible_roots = 0
        self.last_preflop_3bet_teacher_samples = 0
        self.last_preflop_3bet_teacher_coverage = 0.0
        self.last_preflop_3bet_teacher_confidence = 0.0
        self.last_preflop_3bet_teacher_effective_coverage = 0.0
        self.last_preflop_3bet_teacher_effective_weight = 0.0
        self.last_preflop_3bet_teacher_raise_target = 0.0
        self.last_preflop_3bet_teacher_raise_advantage_bb = 0.0
        self.last_preflop_3bet_teacher_actual_raise_rate = 0.0
        self.last_preflop_3bet_teacher_allin_target = 0.0
        self.last_preflop_3bet_teacher_actual_allin_rate = 0.0
        self.last_preflop_3bet_teacher_allin_suppressed = 0
        self.last_preflop_3bet_teacher_multi_raise_samples = 0
        self.last_preflop_3bet_teacher_multi_raise_allin_target = 0.0
        self.last_preflop_3bet_teacher_multi_raise_actual_allin_rate = 0.0
        self.last_preflop_3bet_teacher_multi_raise_allin_vetoes = 0
        self.last_preflop_3bet_teacher_facing_4bet_samples = 0
        self.last_preflop_3bet_teacher_facing_4bet_target_actions = {name: 0.0 for name in PREFLOP_3BET_TEACHER_ACTION_NAMES}
        self.last_preflop_3bet_teacher_facing_4bet_actual_actions = {name: 0.0 for name in PREFLOP_3BET_TEACHER_ACTION_NAMES}
        self.last_preflop_3bet_teacher_facing_4bet_non_allin_vetoes = 0
        self.last_preflop_3bet_teacher_by_root: dict[str, dict[str, float | int | dict[str, float]]] = {}
        self.run_preflop_teacher_by_root: dict[str, dict[str, float | int | dict[str, float]]] = {}
        self.last_robust_policy_weight = 1.0
        self.last_rollout_inference_device = "cpu"
        self.scenario_weakness = {profile: 0.50 for profile in SCENARIO_PROFILES}
        self.opponent_weakness: dict[str, float] = {}
        self.last_training_focus = "balanced"
        self.last_weakness_score = 0.50
        self.last_adaptive_workers = 0
        self.last_adaptive_batch_hands = 0
        self.last_rollout_decisions_per_second = 0.0
        self.rollout_scale = 1.0
        self.teacher_data_report = HandHistoryReport("", 0, 0, "no local teacher data")
        self.teacher_data_records = 0
        self.audit_benchmarks: dict[str, dict[str, float]] = {}
        self.evaluation_history: list[dict[str, float | int]] = []
        self.last_audit_score = 0.0
        self.last_audit_exploitability_bb_per_100 = 0.0
        self.last_scenario_gate = 0.0
        self.last_ablation_delta = 0.0
        self.last_subgame_policy_loss = 0.0
        self.last_subgame_teacher_size = 0
        self.last_rollout_arena_width = 0
        self.last_average_strategy_weight = 0.0
        self.abstract_oracle = AbstractCfrOracle()
        self.abstract_oracle.solve(24)
        self.abstract_cfr_solver = HoldemAbstractionCfr()
        self.abstract_cfr_solver.solve(2)
        self.abstract_teacher_memory = AbstractTeacherMemory()
        self.last_oracle_policy_loss = 0.0
        self.last_oracle_value_loss = 0.0
        self.last_oracle_confidence = 0.0
        self.last_oracle_iterations = self.abstract_cfr_solver.iterations
        initial_abstraction_audit = self.abstract_cfr_solver.audit()
        self.last_abstraction_nash_conv = initial_abstraction_audit.nash_conv
        self.last_abstraction_value = initial_abstraction_audit.average_value
        self.last_abstraction_information_sets = initial_abstraction_audit.information_sets
        self.last_holdout_ci_floor_bb_per_100 = 0.0
        self.last_holdout_paired_variance = 0.0
        self.last_paired_deal_coverage = 0.0
        self.last_belief_posterior_support = 1.0
        self.last_resolver_replay_confidence = 0.0
        self.last_resolver_replay_size = 0
        self.last_blueprint_score = 0.0
        self.last_blueprint_confidence = 0.0
        self.last_blueprint_floor = 0.0
        self.last_blueprint_hands = 0
        self.last_kuhn_value_gap = 1.0
        self.last_blueprint_status = "not audited"
        self.last_counterfactual_value_loss = 0.0
        self.last_counterfactual_coverage = 0.0
        self.last_counterfactual_memory_size = 0
        self.last_public_belief_teacher_size = 0
        self.last_sizing_proposal_diversity = 0.0
        self._publish_locked()

    def _new_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        """Use the stable AdamW path unless fused mode was explicitly requested."""
        optimizer, self.optimizer_backend = build_ppo_optimizer(
            self.model.parameters(),
            learning_rate=learning_rate,
            cuda_enabled=self.runtime.cuda_enabled,
        )
        return optimizer

    def _new_grad_scaler(self, init_scale: float | None = None) -> torch.amp.GradScaler:
        options = {"enabled": self.mixed_precision_enabled}
        if init_scale is not None:
            options["init_scale"] = init_scale
        return torch.amp.GradScaler("cuda", **options)

    def _disable_mixed_precision_after_overflow(self) -> bool:
        """Switch the active trainer to FP32 after a candidate only overflows in AMP."""
        if not self.mixed_precision_enabled:
            return False
        self.mixed_precision_enabled = False
        self.grad_scaler = self._new_grad_scaler()
        self._optimizer_timing_events = []
        self.amp_overflow_fallbacks += 1
        return True

    def begin_learning_profile(self) -> None:
        """Reset one learner update's GPU timing events without changing training."""
        self._optimizer_timing_events = []

    def finish_learning_profile(self) -> dict[str, float | str]:
        """Collect one synchronized optimizer total after the update is complete."""
        if not self.runtime.cuda_enabled or not self._optimizer_timing_events:
            return {"optimizer_seconds": 0.0, "optimizer_backend": self.optimizer_backend}
        torch.cuda.synchronize(self.runtime.device)
        seconds = sum(start.elapsed_time(end) for start, end in self._optimizer_timing_events) / 1_000.0
        return {"optimizer_seconds": seconds, "optimizer_backend": self.optimizer_backend}

    def _publish_locked(self) -> None:
        snapshots = [snapshot["state"] for snapshot in self.strategy_snapshots[-4:] if isinstance(snapshot.get("state"), dict)]
        # The average-strategy head starts near random and is trained from sparse
        # solver replay.  Blend it into live play only after teacher confidence
        # and replay coverage provide evidence that it is useful.
        teacher_confidence = min(1.0, max(0.0, (self.last_oracle_confidence - 0.18) / 0.32))
        replay_coverage = min(1.0, len(self.strategy_memory.records) / 2_000.0)
        blueprint_verified = self.last_blueprint_status == "verified"
        self.last_average_strategy_weight = 0.65 * teacher_confidence * replay_coverage * float(blueprint_verified)
        self.live_agent.replace_state(
            self.champion_state,
            snapshots=snapshots,
            ready=self.version > 0,
            average_strategy_weight=self.last_average_strategy_weight,
        )

    @staticmethod
    def _policy_state_distance(left: dict[str, Tensor], right: dict[str, Tensor]) -> float:
        """Normalized RMS distance across the policy path, not a one-weight proxy."""
        prefixes = ("public_encoder.", "private_encoder.", "encoder.", "gru.", "sequence_attention.", "street_adapters.", "public_state_router.", "public_state_residuals.", "policy.", "average_strategy.", "raise_shapes.")
        squared_difference = 0.0
        squared_scale = 0.0
        for name, left_value in left.items():
            right_value = right.get(name)
            if not name.startswith(prefixes) or not isinstance(left_value, Tensor) or not isinstance(right_value, Tensor):
                continue
            if not left_value.is_floating_point() or not right_value.is_floating_point():
                continue
            left_float, right_float = left_value.float(), right_value.float()
            squared_difference += float((left_float - right_float).square().mean().item())
            squared_scale += 0.5 * float((left_float.square().mean() + right_float.square().mean()).item())
        return math.sqrt(squared_difference / max(1e-12, squared_scale))

    def _refresh_snapshot_diversity_locked(self) -> None:
        states = [snapshot["state"] for snapshot in self.strategy_snapshots if isinstance(snapshot.get("state"), dict)]
        distances = [
            self._policy_state_distance(left, right)
            for index, left in enumerate(states)
            for right in states[index + 1:]
        ]
        self.last_snapshot_min_distance = min(distances, default=0.0)
        mean_distance = sum(distances) / max(1, len(distances))
        self.last_snapshot_diversity = min(1.0, mean_distance / max(SNAPSHOT_MIN_DISTANCE * 4.0, 1e-12))

    def _retain_diverse_snapshots_locked(self, snapshots: list[dict]) -> list[dict]:
        """Keep the newest snapshot plus the policies farthest from one another."""
        if len(snapshots) <= 6:
            return snapshots
        newest = max(snapshots, key=lambda snapshot: int(snapshot.get("updates", -1)))
        selected = [newest]
        remaining = [snapshot for snapshot in snapshots if snapshot is not newest]
        while remaining and len(selected) < 6:
            candidate = max(
                remaining,
                key=lambda snapshot: min(
                    self._policy_state_distance(snapshot["state"], retained["state"])
                    for retained in selected
                ),
            )
            selected.append(candidate)
            remaining.remove(candidate)
        return sorted(selected, key=lambda snapshot: int(snapshot.get("updates", -1)))

    def refresh_strategy_snapshots(self) -> None:
        """Keep a small, diverse frozen policy ensemble for SD-CFR-style averaging."""
        if self.updates == 0 or self.updates % 3:
            return
        with self._lock:
            if self.strategy_snapshots and int(self.strategy_snapshots[-1].get("updates", -1)) == self.updates:
                return
            snapshot = {"id": f"strategy-{self.updates}", "state": clone_state(self.model), "updates": self.updates}
            distances = [
                self._policy_state_distance(snapshot["state"], retained["state"])
                for retained in self.strategy_snapshots
                if isinstance(retained.get("state"), dict)
            ]
            if distances and min(distances) < SNAPSHOT_MIN_DISTANCE:
                self.snapshot_rejections += 1
                self._refresh_snapshot_diversity_locked()
                return
            self.strategy_snapshots = self._retain_diverse_snapshots_locked([*self.strategy_snapshots, snapshot])
            self._refresh_snapshot_diversity_locked()

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(self.runtime.device)

    def _capture_active_member_locked(self) -> None:
        member = self.population_members[self.active_population_index]
        member["state"] = clone_state(self.model)
        member["target_state"] = clone_state(self.target_model)
        member["optimizer_state"] = copy.deepcopy(self.optimizer.state_dict())
        member["grad_scaler_state"] = copy.deepcopy(self.grad_scaler.state_dict())
        member["ppo_learning_rate"] = self.ppo_learning_rate
        member["ppo_clip_epsilon"] = self.ppo_clip_epsilon
        member["ppo_entropy_coefficient"] = self.ppo_entropy_coefficient

    def _retain_specialist_locked(self, state: dict[str, Tensor], metrics: dict[str, float], reason: str) -> None:
        """Preserve a safe non-dominated challenger for future specialist work."""
        if float(metrics.get("behavior_degeneracy", 0.0)) > 0.0 or float(metrics.get("fold_collapse", 0.0)) > 0.0:
            return
        entry = {
            "id": f"specialist-{self.updates}",
            "state": copy.deepcopy(state),
            "target_state": copy.deepcopy(state),
            "optimizer_state": None,
            "grad_scaler_state": None,
            "metrics": {metric: float(metrics.get(metric, -1_000.0)) for metric in PARETO_SPECIALIST_METRICS},
            "focus_root": self.last_preflop_scenario_worst_root,
            "reason": reason,
            "updates": self.updates,
            "resumes": 0,
        }
        self.specialist_archive = retain_pareto_specialists([*self.specialist_archive, entry], 8)

    def _specialist_seed_locked(self) -> dict | None:
        """Choose the least-resumed Pareto specialist without mutating its state."""
        candidates = [entry for entry in self.specialist_archive if isinstance(entry.get("state"), dict)]
        if not candidates:
            return None
        selected = min(candidates, key=lambda entry: (int(entry.get("resumes", 0)), -int(entry.get("updates", 0))))
        selected["resumes"] = int(selected.get("resumes", 0)) + 1
        return selected

    @staticmethod
    def _population_safety_locked(member: dict) -> float:
        """Prioritize profiles that are weak in adversarial and preflop safety audits."""
        return population_safety_score(member)

    @staticmethod
    def _population_member_is_catastrophic(member: dict) -> bool:
        """Recognize a member that lost every recent final-audit evaluation."""
        return population_member_is_catastrophic(member)

    def _trainable_population_indexes_locked(self) -> list[int]:
        """Return only members that are safe to receive another optimizer step."""
        return [
            index
            for index, member in enumerate(self.population_members)
            if population_member_is_trainable(member, self.updates)
        ]

    @classmethod
    def final_audit_population_index(cls, members: list[dict]) -> int:
        """Choose the safest viable saved member for a promotion-grade audit."""
        if not members:
            raise ValueError("Cannot select a final-audit member from an empty population.")
        viable_indexes = [
            index
            for index, member in enumerate(members)
            if not cls._population_member_is_catastrophic(member)
        ]
        candidates = viable_indexes or list(range(len(members)))
        return max(
            candidates,
            key=lambda index: (
                cls._population_safety_locked(members[index]),
                float(members[index].get("score", 0.0)),
                -index,
            ),
        )

    def _activate_population_member_locked(self, member_index: int) -> None:
        """Switch to one saved population member without losing the active state."""
        if member_index == self.active_population_index:
            return
        self._capture_active_member_locked()
        member = self.population_members[member_index]
        self.model.load_state_dict(member["state"])
        self.target_model.load_state_dict(member.get("target_state", member["state"]))
        self.target_model.eval()
        self.optimizer = self._new_optimizer(3e-4 * float(member["lr_scale"]))
        if isinstance(member.get("optimizer_state"), dict):
            self.optimizer.load_state_dict(member["optimizer_state"])
            self._move_optimizer_state_to_device()
        self.grad_scaler = self._new_grad_scaler()
        if isinstance(member.get("grad_scaler_state"), dict):
            self.grad_scaler.load_state_dict(member["grad_scaler_state"])
        self.ppo_learning_rate = float(member.get("ppo_learning_rate", 3e-4 * float(member["lr_scale"])))
        self.ppo_clip_epsilon = float(member.get("ppo_clip_epsilon", 0.20))
        self.ppo_entropy_coefficient = float(member.get("ppo_entropy_coefficient", 0.012 * float(member["entropy_scale"])))
        self.active_population_index = member_index

    def select_final_audit_member(self) -> dict[str, object]:
        """Select by fresh behavior, not stale metrics from a different member."""
        with self._lock:
            previous_index = self.active_population_index
            preflights = self._population_behavior_preflights_locked(hands_per_root=16)
            selected_index = population_behavior_selection_index(self.population_members, preflights, self.updates)
            self._activate_population_member_locked(selected_index)
            self._capture_active_member_locked()
            selected = self.population_members[selected_index]
            return {
                "previous_member": str(self.population_members[previous_index]["id"]),
                "selected_member": str(selected["id"]),
                "selected_score": float(selected.get("score", 0.0)),
                "selected_safety": self._population_safety_locked(selected),
                "selected_behavior": preflights[selected_index],
                "population_behavior_preflight": {
                    str(member["id"]): {
                        **preflight,
                        "degeneracy": population_behavior_degeneracy(preflight),
                    }
                    for member, preflight in zip(self.population_members, preflights)
                },
                "skipped_catastrophic_members": [
                    str(member["id"])
                    for member in self.population_members
                    if self._population_member_is_catastrophic(member)
                ],
            }

    def _population_behavior_preflights_locked(self, hands_per_root: int = 4) -> list[dict[str, float | int]]:
        """Refresh behavior safety from current weights before scheduling a learner."""
        self._capture_active_member_locked()
        preflights = [
            preflop_population_behavior_audit(
                member["state"],
                seed=1_460_000 + self.updates * 101 + index * 10_003,
                curriculum_stage=self.curriculum_unlocked_stage,
                hands_per_root=hands_per_root,
            )
            for index, member in enumerate(self.population_members)
        ]
        for member, preflight in zip(self.population_members, preflights):
            member["behavior_fold_rate"] = float(preflight["fold_rate"])
            member["behavior_all_in_rate"] = float(preflight["all_in_rate"])
            member["behavior_degeneracy"] = population_behavior_degeneracy(preflight)
        return preflights

    def _quarantine_active_population_member_locked(self, reasons: list[str]) -> dict[str, object]:
        """Rollback only a repeatedly regressing learner to its measured safety anchor."""
        member = self.population_members[self.active_population_index]
        recovery_learning_rate = min(self.ppo_learning_rate, 8e-5 * float(member["lr_scale"]))
        member_id = str(member["id"])
        quarantine_count = int(member.get("quarantine_count", 0)) + 1
        member.update({
            "state": copy.deepcopy(self.recovery_anchor_state),
            "target_state": copy.deepcopy(self.recovery_anchor_target_state),
            "optimizer_state": None,
            "grad_scaler_state": None,
            "score": 0.50,
            "bb_per_100": 0.0,
            "adversarial_bb_per_100": 0.0,
            "preflop_worst_lcb_bb_per_100": 0.0,
            "preflop_allin_probability": 0.0,
            "behavior_fold_rate": float(self.recovery_anchor_metrics.get("first_fold_rate", 0.0)),
            "behavior_all_in_rate": float(self.recovery_anchor_metrics.get("first_all_in_rate", 0.0)),
            "behavior_degeneracy": float(self.recovery_anchor_metrics.get("behavior_degeneracy", 0.0)),
            "quarantine_count": quarantine_count,
            "safety_regressions": 0,
            "recovery_cooldown_until": self.updates + RECOVERY_ANCHOR_COOLDOWN_UPDATES,
            "last_quarantine_reason": "; ".join(reasons),
            "updates": 0,
        })
        self.model.load_state_dict(self.recovery_anchor_state)
        self.target_model.load_state_dict(self.recovery_anchor_target_state)
        self.target_model.eval()
        self.optimizer = self._new_optimizer(recovery_learning_rate)
        self.grad_scaler = self._new_grad_scaler()
        self.ppo_learning_rate = recovery_learning_rate
        self.ppo_clip_epsilon = 0.20
        self.ppo_entropy_coefficient = 0.012 * float(member["entropy_scale"])
        self.ppo_recovery_updates = 0
        return {"member": member_id, "reasons": reasons, "quarantine_count": quarantine_count, "updates": self.updates}

    def _halt_and_restore_population_locked(self, reasons: list[str]) -> dict[str, object]:
        """Stop the run and restore every learner before another profile can train."""
        restored_members: list[str] = []
        for member in self.population_members:
            member["state"] = copy.deepcopy(self.recovery_anchor_state)
            member["target_state"] = copy.deepcopy(self.recovery_anchor_target_state)
            member["optimizer_state"] = None
            member["grad_scaler_state"] = None
            member["score"] = 0.50
            member["bb_per_100"] = 0.0
            member["adversarial_bb_per_100"] = 0.0
            member["preflop_worst_lcb_bb_per_100"] = 0.0
            member["preflop_allin_probability"] = 0.0
            member["behavior_fold_rate"] = float(self.recovery_anchor_metrics.get("first_fold_rate", 0.0))
            member["behavior_all_in_rate"] = float(self.recovery_anchor_metrics.get("first_all_in_rate", 0.0))
            member["behavior_degeneracy"] = float(self.recovery_anchor_metrics.get("behavior_degeneracy", 0.0))
            member["safety_regressions"] = 0
            member["recovery_cooldown_until"] = self.updates + RECOVERY_ANCHOR_COOLDOWN_UPDATES
            member["last_quarantine_reason"] = "; ".join(reasons)
            member["quarantine_count"] = int(member.get("quarantine_count", 0)) + 1
            member["updates"] = 0
            restored_members.append(str(member["id"]))
        self.active_population_index = 0
        active_member = self.population_members[self.active_population_index]
        self.model.load_state_dict(self.recovery_anchor_state)
        self.target_model.load_state_dict(self.recovery_anchor_target_state)
        self.target_model.eval()
        self.optimizer = self._new_optimizer(3e-4 * float(active_member["lr_scale"]))
        self.grad_scaler = self._new_grad_scaler()
        self.ppo_learning_rate = 3e-4 * float(active_member["lr_scale"])
        self.ppo_clip_epsilon = 0.20
        self.ppo_entropy_coefficient = 0.012 * float(active_member["entropy_scale"])
        self.ppo_recovery_updates = 0
        self.recovery_safe_audits = 0
        self.recovery_halted = True
        self.last_challenger_status = "safety-stopped and restored"
        return {"members": restored_members, "reasons": reasons, "updates": self.updates}

    @staticmethod
    def _recovery_safety_score(adversarial_lcb: float, preflop_lcb: float, holdout_lcb: float, all_in_probability: float, all_in_target: float) -> float:
        """Conservative rollback score; it is never used as a promotion gate."""
        excessive_all_in = max(0.0, all_in_probability - max(0.12, all_in_target * 1.5))
        return 0.40 * adversarial_lcb + 0.35 * preflop_lcb + 0.25 * holdout_lcb - 140.0 * excessive_all_in

    def _consider_recovery_anchor_locked(self, candidate: dict[str, Tensor], score: float, metrics: dict[str, float]) -> bool:
        if not self._recovery_candidate_is_safe(metrics):
            return False
        if score <= self.recovery_anchor_score:
            return False
        self.recovery_anchor_state = copy.deepcopy(candidate)
        self.recovery_anchor_target_state = clone_state(self.target_model)
        self.recovery_anchor_score = score
        self.recovery_anchor_metrics = metrics
        self.recovery_anchor_updates = self.updates
        self.recovery_anchor_source = "measured_candidate"
        self._recovery_anchor_probe_cache.clear()
        return True

    def _recovery_anchor_policy_signature(self, curriculum_stage: int) -> dict[str, list[list[float]]]:
        """Cache the verified anchor's fixed-root strategy for PPO trust regions."""
        cached = self._recovery_anchor_probe_cache.get(curriculum_stage)
        if cached is not None:
            return cached
        anchor = PolicyValueNetwork().to(self.runtime.device)
        anchor.load_state_dict(self.recovery_anchor_state)
        signature = preflop_root_policy_signature(anchor, curriculum_stage)
        self._recovery_anchor_probe_cache[curriculum_stage] = signature
        del anchor
        return signature

    @staticmethod
    def _recovery_candidate_is_safe(metrics: dict[str, float]) -> bool:
        """Reject degenerate all-fold/all-in candidates before they can be anchors."""
        behavior_safe = population_behavior_is_safe({
            "fold_rate": float(metrics.get("first_fold_rate", 1.0)),
            "all_in_rate": float(metrics.get("first_all_in_rate", 1.0)),
        })
        return (
            float(metrics.get("direct_lcb", -float("inf"))) >= -25.0
            and float(metrics.get("adversarial_lcb", -float("inf"))) >= -220.0
            and float(metrics.get("holdout_lcb", -float("inf"))) >= -220.0
            and float(metrics.get("preflop_lcb", -float("inf"))) >= -450.0
            and float(metrics.get("all_in_probability", 1.0)) <= 0.50
            and behavior_safe
        )

    @staticmethod
    def recovery_regression_requires_quarantine(final_audit: bool, safety_regressions: int) -> bool:
        """Keep a failed final audit from becoming the next live checkpoint."""
        return final_audit or safety_regressions >= 2

    def _has_verified_recovery_anchor_locked(self) -> bool:
        """Return whether a measured safe checkpoint is available for restoration.

        A fresh model deliberately has no such anchor.  Treating its random
        initialization as one turns an expected early audit failure into a
        misleading restore-and-stop loop.
        """
        return (
            self.recovery_anchor_source != "unverified"
            and bool(self.recovery_anchor_metrics)
            and self._recovery_candidate_is_safe(self.recovery_anchor_metrics)
        )

    def seed_recovery_anchor_from_current_checkpoint(self, source: str = "protected_checkpoint") -> None:
        """Pin a known checkpoint before a recovery run can produce a new anchor."""
        with self._lock:
            self.recovery_anchor_state = clone_state(self.model)
            self.recovery_anchor_target_state = clone_state(self.target_model)
            self.recovery_anchor_score = RECOVERY_ANCHOR_BOOTSTRAP_SCORE
            self.recovery_anchor_metrics = {}
            self.recovery_anchor_updates = self.updates
            self.recovery_anchor_source = source
            self.recovery_safe_audits = 0
            self.recovery_baseline_verified = False
            self.recovery_baseline_metrics = {}
            self.last_recovery_candidate_metrics = {}
            self.last_fresh_warmup_fold_collapse = False
            self.recovery_halted = False
            self._recovery_anchor_probe_cache.clear()
            for member in self.population_members:
                member["state"] = clone_state(self.model)
                member["target_state"] = clone_state(self.target_model)
                member["optimizer_state"] = None
                member["grad_scaler_state"] = None
                member["safety_regressions"] = 0
                member["recovery_cooldown_until"] = self.updates
            self._publish_locked()

    def verify_recovery_baseline(self, curriculum_stage: int, curriculum_phase: str) -> dict:
        """Audit a restored checkpoint before any new rollout can update it."""
        with self._lock:
            protected_source = self.recovery_anchor_source
        summary = self.evaluate_and_checkpoint(curriculum_stage, curriculum_phase, force_full=True)
        with self._lock:
            metrics = dict(self.last_recovery_candidate_metrics)
            self.recovery_baseline_metrics = copy.deepcopy(metrics)
            self.recovery_baseline_verified = self._recovery_candidate_is_safe(metrics)
            suffix = "verified" if self.recovery_baseline_verified else "rejected_safety"
            self.recovery_anchor_source = f"{protected_source}:{suffix}"
            if not self.recovery_baseline_verified:
                self.recovery_safe_audits = 0
            self._publish_locked()
            summary.update({
                "recovery_baseline_verified": self.recovery_baseline_verified,
                "recovery_baseline_metrics": copy.deepcopy(metrics),
            })
        return summary

    def _update_target_network(self, tau: float = 0.04) -> None:
        with torch.no_grad():
            squared_difference = 0.0
            parameter_count = 0
            for target_parameter, source_parameter in zip(self.target_model.parameters(), self.model.parameters()):
                squared_difference += float((target_parameter - source_parameter).square().mean().item())
                parameter_count += 1
                target_parameter.lerp_(source_parameter, tau)
            self.last_target_drift = math.sqrt(squared_difference / max(1, parameter_count))
        self.target_model.eval()

    def update_target_network(self) -> None:
        self._update_target_network()

    def _to_training_device(self, tensor: Tensor) -> Tensor:
        if not self.runtime.cuda_enabled:
            return tensor
        return tensor.pin_memory().to(self.runtime.device, non_blocking=True)

    def _autocast(self):
        # bf16 has fp32's dynamic range, eliminating the fp16 overflow class
        # that produced non-finite policy logits; fp16 remains the fallback
        # for pre-Ampere devices only.
        dtype = torch.bfloat16 if self.runtime.cuda_enabled and torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=self.mixed_precision_enabled)

    def _optimizer_step(self, loss: Tensor) -> None:
        """Apply one auxiliary update transactionally and fail closed.

        PPO has its own candidate rollback.  Replay/solver objectives share an
        optimizer but previously had no equivalent protection, so a single
        non-finite auxiliary gradient could poison the next rollout audit.
        """
        if not bool(torch.isfinite(loss.detach()).all().item()):
            self.optimizer.zero_grad(set_to_none=True)
            raise AuxiliaryUpdateRejected("auxiliary loss is non-finite")
        start_event = end_event = None
        if self.runtime.cuda_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.runtime.device))
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        active_parameters = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
        gradient_norm = nn.utils.clip_grad_norm_(active_parameters, 0.7)
        if not bool(torch.isfinite(gradient_norm).all().item()) or not all(bool(torch.isfinite(parameter.grad).all().item()) for parameter in active_parameters):
            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.update()
            raise AuxiliaryUpdateRejected("auxiliary gradients are non-finite")
        parameter_snapshot = [parameter.detach().clone() for parameter in active_parameters]
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if not all(bool(torch.isfinite(parameter).all().item()) for parameter in active_parameters):
            with torch.no_grad():
                for parameter, previous in zip(active_parameters, parameter_snapshot):
                    parameter.copy_(previous)
            current_lr = min(float(group.get("lr", self.ppo_learning_rate)) for group in self.optimizer.param_groups)
            self.optimizer = self._new_optimizer(current_lr)
            self.grad_scaler = self._new_grad_scaler()
            self.optimizer.zero_grad(set_to_none=True)
            raise AuxiliaryUpdateRejected("auxiliary optimizer produced non-finite parameters")
        if start_event is not None and end_event is not None:
            end_event.record(torch.cuda.current_stream(self.runtime.device))
            self._optimizer_timing_events.append((start_event, end_event))

    def _adapt_ppo_controller(self, kl_divergence: float, entropy: float, clip_fraction: float, kl_limited: bool = False) -> None:
        observed_kl = abs(kl_divergence)
        if kl_limited:
            self.ppo_learning_rate = max(4e-5, self.ppo_learning_rate * 0.60)
            self.ppo_clip_epsilon = max(0.08, self.ppo_clip_epsilon * 0.85)
            self.ppo_recovery_updates = max(self.ppo_recovery_updates, PPO_RECOVERY_UPDATES)
        elif self.ppo_recovery_updates > 0:
            self.ppo_recovery_updates -= 1
            # Keep the hard stop intact, but do not leave a recovering learner
            # permanently pinned at the minimum rate after clean updates.
            if observed_kl < self.ppo_kl_target * 0.85 and clip_fraction < 0.25:
                self.ppo_learning_rate = min(3e-4, self.ppo_learning_rate * 1.05)
                self.ppo_clip_epsilon = min(0.20, self.ppo_clip_epsilon * 1.01)
        elif observed_kl > self.ppo_kl_target * 1.65 or clip_fraction > 0.30:
            self.ppo_learning_rate = max(4e-5, self.ppo_learning_rate * 0.78)
            self.ppo_clip_epsilon = max(0.10, self.ppo_clip_epsilon * 0.92)
        elif observed_kl < self.ppo_kl_target * 0.45 and clip_fraction < 0.08:
            self.ppo_learning_rate = min(5e-4, self.ppo_learning_rate * 1.08)
            self.ppo_clip_epsilon = min(0.26, self.ppo_clip_epsilon * 1.03)
        if entropy < 0.55:
            self.ppo_entropy_coefficient = min(0.022, self.ppo_entropy_coefficient * 1.07)
        elif entropy > 1.25:
            self.ppo_entropy_coefficient = max(0.002, self.ppo_entropy_coefficient * 0.94)
        for group in self.optimizer.param_groups:
            group["lr"] = self.ppo_learning_rate

    def begin_run(self) -> None:
        with self._lock:
            if not self.imitation_memory.self_play_migration_complete:
                archives: list[str] = []
                archive_directory = MODEL_PATH.parent / "checkpoint_archives" / f"before-imitation-recovery-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                try:
                    existing_checkpoints = [path for path in (MODEL_PATH, MODEL_BACKUP_PATH) if path.exists()]
                    if existing_checkpoints:
                        archive_directory.mkdir(parents=True, exist_ok=False)
                        for checkpoint_path in existing_checkpoints:
                            archived_path = archive_directory / checkpoint_path.name
                            shutil.copy2(checkpoint_path, archived_path)
                            archives.append(str(archived_path))
                    removed = self.imitation_memory.discard_legacy_self_play()
                    log_training_debug("self_imitation_recovery_migrated", removed_records=removed, archives=archives)
                except OSError as exc:
                    # Do not discard the legacy buffer if its checkpoint was not preserved.
                    log_training_debug("self_imitation_recovery_archive_failed", error=str(exc))
            # An interrupted background task can leave GradScaler with an
            # optimizer marked as already unscaled. Keep its learned scale,
            # but start the next user-requested run with fresh transient state.
            scaler_scale = self.grad_scaler.get_scale()
            self.grad_scaler = self._new_grad_scaler(init_scale=scaler_scale)
            self._optimizer_timing_events = []
            self.run_rollout_hands = 0
            self.run_adversarial_hands = 0
            self._rollout_diagnostic_window = []
            self.run_adversarial_paths = 0
            self.run_tail_paths = 0
            self.run_tail_loss_sum = 0.0
            self.run_tail_weight_sum = 0.0
            self.run_tail_style_totals = {}
            self._tail_diagnostic_window = []
            self.last_adversarial_rollout_fraction = 0.0
            self.last_tail_loss_rate = 0.0
            self.last_tail_loss_bb = 0.0
            self.last_tail_policy_weight = 1.0
            self.last_tail_style_diagnostics = {}
            self.run_preflop_teacher_by_root = {}
            self.run_ppo_safety = empty_ppo_safety_counters()
            self.last_preflop_root_guarded = False
            self.last_preflop_root_guard_reason = "none"
            self.last_ppo_root_backoff_applied = False
            self.last_ppo_root_backoff_accepted = False
            self.last_ppo_root_backoff_scale = 0.0
            self.last_fresh_warmup_fold_collapse = False
            self.last_final_audit_checkpoint_restored = False
            self.last_final_audit_restore_reason = ""
        if self.runtime.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.runtime.device)

    def runtime_view(self) -> dict:
        if not self.runtime.cuda_enabled:
            return {"device": "cpu", "device_request": self.runtime.requested, "gpu_enabled": False, "gpu_name": self.runtime.name, "gpu_vram_total_mb": 0, "gpu_vram_allocated_mb": 0, "gpu_vram_peak_mb": 0, "device_reason": self.runtime.reason}
        return {
            "device": str(self.runtime.device),
            "device_request": self.runtime.requested,
            "gpu_enabled": True,
            "gpu_name": self.runtime.name,
            "gpu_vram_total_mb": self.runtime.total_memory_mb,
            "gpu_vram_allocated_mb": round(torch.cuda.memory_allocated(self.runtime.device) / 1024**2, 1),
            "gpu_vram_peak_mb": round(torch.cuda.max_memory_allocated(self.runtime.device) / 1024**2, 1),
            "device_reason": self.runtime.reason,
        }

    def rollout_batch_size(self, workers: int) -> int:
        base = max(384, workers * 192) if self.rollout_inference_device() == "cuda" else max(256, workers * 64) if self.runtime.cuda_enabled else max(64, workers * 40)
        return max(workers * 16, int(base * self.rollout_scale))

    def rollout_inference_device(self) -> str:
        """Use one CUDA collector only when explicitly available; workers otherwise stay CPU."""
        if ROLLOUT_INFERENCE_DEVICE == "cpu":
            return "cpu"
        if self.runtime.cuda_enabled and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def rollout_plan(self, remaining_hands: int) -> tuple[int, int]:
        """Keep batches long enough for efficient PPO but short enough for responsive gates."""
        try:
            configured_workers = int(os.environ.get("HOLDEM_ROLLOUT_WORKERS", "0"))
        except ValueError:
            configured_workers = 0
        inference_device = self.rollout_inference_device()
        available = 1 if inference_device == "cuda" else min(4, max(1, os.cpu_count() or 1))
        requested_workers = configured_workers if configured_workers > 0 else available
        workers = min(available, requested_workers, max(1, remaining_hands // 24))
        with self._lock:
            batch_hands = min(remaining_hands, self.rollout_batch_size(workers))
            remainder = remaining_hands - batch_hands
            if 0 < remainder < PPO_MIN_FINAL_ROLLOUT_HANDS:
                batch_hands = remaining_hands
            self.last_adaptive_workers = workers
            self.last_adaptive_batch_hands = batch_hands
        return workers, batch_hands

    def note_rollout_throughput(self, actions: int, elapsed: float, workers: int, batch_hands: int, inference_device: str = "cpu") -> None:
        with self._lock:
            self.last_rollout_decisions_per_second = actions / max(0.001, elapsed)
            self.last_adaptive_workers = workers
            self.last_adaptive_batch_hands = batch_hands
            self.last_rollout_arena_width = max(1, batch_hands // max(1, workers))
            self.last_rollout_inference_device = inference_device
            # Aim for a few seconds of CPU collection per PPO update. This reacts
            # to worker throughput without assuming a particular GPU or CPU.
            if elapsed < 3.0:
                self.rollout_scale = min(2.0, self.rollout_scale * 1.12)
            elif elapsed > 12.0:
                self.rollout_scale = max(0.55, self.rollout_scale * 0.86)

    def _champion_entry_locked(self) -> dict:
        return {"id": self.champion_id, "state": copy.deepcopy(self.champion_state), "state_revision": f"champion:{self.champion_id}:{id(self.champion_state)}", "rollout_cacheable": True, "elo": self.champion_elo, "games": 1_000, "kind": "model"}

    @staticmethod
    def _frozen_rollout_entry(entry: dict, source: str) -> dict:
        cloned = copy.deepcopy(entry)
        state = entry.get("state")
        if isinstance(state, dict):
            cloned["state_revision"] = f"{source}:{entry.get('id', 'unknown')}:{id(state)}"
            cloned["rollout_cacheable"] = True
        return cloned

    def _roster_locked(self) -> list[dict]:
        population = [
            {"id": str(member["id"]), "state": copy.deepcopy(member["state"]), "elo": 1_180.0 + 40 * float(member.get("score", 0.5)), "games": 1_000, "kind": "model"}
            for index, member in enumerate(self.population_members)
            if index != self.active_population_index and member["id"] != self.champion_id
        ]
        snapshots = [
            {"id": str(snapshot["id"]), "state": copy.deepcopy(snapshot["state"]), "state_revision": f"snapshot:{snapshot['id']}:{id(snapshot['state'])}", "rollout_cacheable": True, "elo": 1_150.0, "games": 600, "kind": "model"}
            for snapshot in self.strategy_snapshots[-4:]
            if isinstance(snapshot.get("state"), dict)
        ]
        league = [self._frozen_rollout_entry(entry, "league") for entry in self.league]
        exploiters = [self._frozen_rollout_entry(entry, "exploiter") for entry in self.exploiters]
        return [self._champion_entry_locked(), *league, *exploiters, *snapshots, *population]

    @staticmethod
    def _payoff_key(row_id: str, column_id: str) -> str:
        return f"{row_id}|{column_id}"

    def _payoff_locked(self, row_id: str, column_id: str) -> float:
        return self.payoff_matrix.get(self._payoff_key(row_id, column_id), 0.5)

    def _record_payoff_locked(self, row_id: str, column_id: str, score: float) -> None:
        self.payoff_matrix[self._payoff_key(row_id, column_id)] = min(1.0, max(0.0, float(score)))

    def _record_match_payoff_locked(self, row_id: str, column_id: str, result: MatchResult) -> None:
        score = bb_per_100_score(result.bb_per_100)
        self._record_payoff_locked(row_id, column_id, score)
        self._record_payoff_locked(column_id, row_id, 1.0 - score)

    def _mixture_weights_locked(self, entries: list[dict]) -> dict[str, float]:
        """Regret matching with payoff pressure and a small coverage floor."""
        if not entries:
            return {}
        raw: dict[str, float] = {}
        for entry in entries:
            policy_id = str(entry["id"])
            regret = max(0.0, self.mixture_regrets.get(policy_id, 0.0))
            pressure = max(0.0, 0.5 - self._payoff_locked(self.champion_id, policy_id))
            raw[policy_id] = regret + pressure
        total = sum(raw.values())
        uniform = 1.0 / len(entries)
        if total <= 1e-12:
            return {str(entry["id"]): uniform for entry in entries}
        coverage = min(0.05, 0.5 / len(entries))
        return {policy_id: coverage + (1 - coverage * len(entries)) * value / total for policy_id, value in raw.items()}

    def _adaptive_opponent_weights_locked(self, entries: list[dict]) -> dict[str, float]:
        base = self._mixture_weights_locked(entries)
        if not base:
            return base
        raw = {
            policy_id: base[policy_id]
            * (0.35 + 0.65 * max(0.08, 1.0 - abs(2.0 * self._payoff_locked(self.champion_id, policy_id) - 1.0)))
            * (0.60 + 1.40 * min(1.0, max(0.0, self.opponent_weakness.get(policy_id, 0.50))))
            * (1.0 + 1.80 * min(0.35, max(0.0, float(entry.get("threat", 0.0)))))
            for entry in entries
            for policy_id in [str(entry["id"])]
        }
        total = sum(raw.values())
        coverage = min(0.025, 0.35 / len(raw))
        return {policy_id: coverage + (1 - coverage * len(raw)) * value / max(1e-12, total) for policy_id, value in raw.items()}

    def _focused_adversarial_styles_locked(self) -> tuple[str, ...]:
        """Select stable BB/100 leaks, with measured large-pot tail severity."""
        def risk_adjusted_score(style: str) -> float:
            # BB/100 already includes every tail loss. Tail severity may break
            # close calls, but must not be subtracted at full size a second time
            # or profitable volatile styles displace an actual losing matchup.
            score = float(self.adversarial_style_bb_per_100.get(style, 0.0)) / 100.0
            tail = self.last_tail_style_diagnostics.get(style, {})
            hands = int(tail.get("hands", 0)) if isinstance(tail, dict) else 0
            if hands >= 8:
                tail_rate = max(0.0, float(tail.get("tail_rate", 0.0)))
                score -= min(0.08, 0.20 * max(0.0, tail_rate - 0.20))
            return score
        ranked = sorted(
            ADVERSARIAL_TRAINING_STYLES,
            key=risk_adjusted_score,
        )
        return tuple(ranked[:ADVERSARIAL_FOCUS_COUNT])

    def _hard_spot_styles_locked(self) -> tuple[str, ...]:
        """Keep critic replay aware of costly, under-focused opponent styles."""
        focused = self._focused_adversarial_styles_locked()
        tail_ranked = sorted(
            (style for style in ADVERSARIAL_TRAINING_STYLES if style not in focused),
            key=lambda style: float(self.last_tail_style_diagnostics.get(style, {}).get("tail_rate", 0.0)) * max(0.0, -float(self.last_tail_style_diagnostics.get(style, {}).get("tail_loss_bb", 0.0))),
            reverse=True,
        )
        return tuple(dict.fromkeys((*focused, *tail_ranked[:2])))

    def _eligible_curriculum_stage_locked(self) -> int:
        stage = 0
        for index in range(1, len(CURRICULUM_PHASES)):
            if self.trained_hands >= CURRICULUM_PHASES[index - 1][0]:
                stage = index
        return stage

    def _refresh_curriculum_gate_locked(self, full_evaluation: bool) -> None:
        direct_quality = bb_per_100_quality(self.last_promotion_ci_lower, -60.0, 0.0)
        league_quality = bb_per_100_quality(self.last_evaluation_bb_per_100, -50.0, 0.0)
        adversarial_quality = min(
            bb_per_100_quality(self.last_adversarial_floor_bb_per_100, -80.0, -5.0),
            bb_per_100_quality(self.last_restricted_br_bb_per_100, -80.0, -5.0),
            bb_per_100_quality(self.last_audit_exploitability_bb_per_100, -80.0, -5.0),
        )
        self.last_scenario_gate = min((1.0 - weakness for weakness in self.scenario_weakness.values()), default=0.0)
        self.last_curriculum_readiness = 0.24 * direct_quality + 0.22 * league_quality + 0.36 * adversarial_quality + 0.18 * self.last_scenario_gate
        eligible = self._eligible_curriculum_stage_locked()
        required = (0.0, 0.60, 0.64, 0.68)
        direct_floor = (0.0, -10.0, -8.0, -5.0)
        league_floor = (0.0, -12.0, -10.0, -8.0)
        adversarial_floor = (0.0, -25.0, -20.0, -15.0)
        restricted_floor = (0.0, -30.0, -25.0, -20.0)
        if full_evaluation and self.curriculum_unlocked_stage < eligible:
            next_stage = self.curriculum_unlocked_stage + 1
            if self.last_curriculum_readiness >= required[next_stage] and self.last_promotion_ci_lower >= direct_floor[next_stage] and self.last_evaluation_bb_per_100 >= league_floor[next_stage] and self.last_adversarial_floor_bb_per_100 >= adversarial_floor[next_stage] and self.last_restricted_br_bb_per_100 >= restricted_floor[next_stage] and self.last_scenario_gate >= (0.54, 0.57, 0.60, 0.63)[next_stage]:
                self.curriculum_unlocked_stage = next_stage

    def curriculum(self, pending_hands: int = 0) -> tuple[int, str]:
        with self._lock:
            stage = self.curriculum_unlocked_stage
        return stage, CURRICULUM_PHASES[stage][2]

    def select_training_lane(self) -> bool:
        with self._lock:
            preflights = self._population_behavior_preflights_locked(hands_per_root=4)
            behavior_safe_indexes = [
                index
                for index, preflight in enumerate(preflights)
                if population_behavior_is_safe(preflight)
            ]
            trainable_indexes = self._trainable_population_indexes_locked()
            candidates = [index for index in trainable_indexes if index in behavior_safe_indexes]
            if not candidates and self._has_verified_recovery_anchor_locked():
                anchor_preflight = preflop_population_behavior_audit(
                    self.recovery_anchor_state,
                    seed=1_565_000 + self.updates * 107,
                    curriculum_stage=self.curriculum_unlocked_stage,
                    hands_per_root=4,
                )
                if population_behavior_is_safe(anchor_preflight):
                    # A final-audit rollback can leave every specialist either
                    # cooling down or carrying stale catastrophic EV metadata,
                    # even though their restored anchor behavior is safe. Keep
                    # one neutral recovery member trainable so the scheduler can
                    # learn the newly measured leak instead of deadlocking.
                    member_index = min(
                        range(len(self.population_members)),
                        key=lambda index: (int(self.population_members[index].get("quarantine_count", 0)), index),
                    )
                    member = self.population_members[member_index]
                    recovery_learning_rate = min(self.ppo_learning_rate, 8e-5 * float(member["lr_scale"]))
                    member.update({
                        "state": copy.deepcopy(self.recovery_anchor_state),
                        "target_state": copy.deepcopy(self.recovery_anchor_target_state),
                        "optimizer_state": None,
                        "grad_scaler_state": None,
                        "score": 0.50,
                        "bb_per_100": 0.0,
                        "adversarial_bb_per_100": 0.0,
                        "preflop_worst_lcb_bb_per_100": 0.0,
                        "preflop_allin_probability": 0.0,
                        "behavior_fold_rate": float(anchor_preflight["fold_rate"]),
                        "behavior_all_in_rate": float(anchor_preflight["all_in_rate"]),
                        "behavior_degeneracy": population_behavior_degeneracy(anchor_preflight),
                        "safety_regressions": 0,
                        "recovery_cooldown_until": self.updates,
                        "last_quarantine_reason": "verified-anchor scheduler recovery",
                        "updates": 0,
                    })
                    self.active_population_index = member_index
                    self.model.load_state_dict(self.recovery_anchor_state)
                    self.target_model.load_state_dict(self.recovery_anchor_target_state)
                    self.target_model.eval()
                    self.optimizer = self._new_optimizer(recovery_learning_rate)
                    self.grad_scaler = self._new_grad_scaler()
                    self.ppo_learning_rate = recovery_learning_rate
                    self.ppo_clip_epsilon = 0.20
                    self.ppo_entropy_coefficient = 0.012 * float(member["entropy_scale"])
                    self.ppo_recovery_updates = PPO_RECOVERY_UPDATES
                    self.recovery_halted = False
                    preflights[member_index] = anchor_preflight
                    candidates = [member_index]
                    log_training_debug("population_scheduler_anchor_recovered", member=str(member["id"]), anchor_preflight=anchor_preflight, updates=self.updates)
            if not candidates:
                self.recovery_halted = True
                raise RuntimeError(
                    "Safety stop: no behavior-safe population member remains; "
                    "recover a checkpoint with balanced fixed-root fold and all-in rates."
                )
            continuing_exploiter_lane = self.exploiter_lane_remaining > 0 and self.active_population_index in candidates
            if self.exploiter_lane_remaining > 0 and not continuing_exploiter_lane:
                self.exploiter_lane_remaining = 0
            if continuing_exploiter_lane:
                member_index = self.active_population_index
            else:
                rotation_index = candidates[self.updates % len(candidates)]
                weakest_index = min(candidates, key=lambda index: self._population_safety_locked(self.population_members[index]))
                member_index = weakest_index if self.updates > 0 and self.updates % 5 == 0 else rotation_index
            if member_index != self.active_population_index:
                self._activate_population_member_locked(member_index)
            scheduled_best_response = self.updates > 0 and self.updates % 3 == 1
            active_updates = int(self.population_members[self.active_population_index].get("updates", 0))
            fresh_exploiter_lane = not continuing_exploiter_lane and scheduled_best_response and self.updates % EXPLOITER_REFRESH_UPDATES == 1 and active_updates >= POPULATION_SPECIALIST_MIN_UPDATES
            specialist_seed = self._specialist_seed_locked() if fresh_exploiter_lane else None
            seed_state = specialist_seed["state"] if specialist_seed is not None else self.champion_state
            if fresh_exploiter_lane:
                champion_preflight = preflop_population_behavior_audit(
                    seed_state,
                    seed=1_570_000 + self.updates * 103,
                    curriculum_stage=self.curriculum_unlocked_stage,
                    hands_per_root=4,
                )
                fresh_exploiter_lane = population_behavior_is_safe(champion_preflight)
            best_response_lane = continuing_exploiter_lane or scheduled_best_response
            if fresh_exploiter_lane:
                member = self.population_members[self.active_population_index]
                self.model.load_state_dict(seed_state)
                self.target_model.load_state_dict(specialist_seed.get("target_state", seed_state) if specialist_seed is not None else seed_state)
                self.target_model.eval()
                self.optimizer = self._new_optimizer(3e-4 * float(member["lr_scale"]))
                self.grad_scaler = self._new_grad_scaler()
                self.ppo_learning_rate = 3e-4 * float(member["lr_scale"])
                self.ppo_clip_epsilon = 0.20
                self.ppo_entropy_coefficient = 0.012 * float(member["entropy_scale"])
                # seed_state is already a serialized state-dict, while
                # clone_state accepts an nn.Module. Keep independent tensors
                # for the fresh exploiter without calling .state_dict() on it.
                member["state"] = copy.deepcopy(seed_state)
                member["target_state"] = copy.deepcopy(specialist_seed.get("target_state", seed_state) if specialist_seed is not None else seed_state)
                member["optimizer_state"] = None
                member["grad_scaler_state"] = None
                member["updates"] = 0
                member["bb_per_100"] = 0.0
                member["adversarial_bb_per_100"] = 0.0
                self.exploiter_generations += 1
                self.exploiter_lane_remaining = 3
            elif continuing_exploiter_lane:
                self.exploiter_lane_remaining -= 1
            lane_name = "resumed-specialist" if fresh_exploiter_lane and specialist_seed is not None else "fresh-exploiter" if fresh_exploiter_lane else "exploiter" if continuing_exploiter_lane else "best-response" if best_response_lane else "population"
            self.last_training_lane = f"{self.population_members[self.active_population_index]['id']} · {lane_name}"
            return best_response_lane

    def load(self) -> bool:
        """Restore the newest valid checkpoint, falling back to the prior atomic save."""
        return any(self._load_checkpoint(path) for path in (MODEL_PATH, MODEL_BACKUP_PATH))

    def _load_checkpoint(self, checkpoint_path: Path) -> bool:
        if not checkpoint_path.exists():
            return False
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            saved_format = int(payload.get("format", -1))
            if saved_format not in COMPATIBLE_MODEL_VERSIONS:
                return False
            checkpoint_migrated = migrate_checkpoint_policy_states(payload, clone_state(self.model))
            self.model.load_state_dict(payload["model"])
            self.target_model.load_state_dict(payload.get("target_model", payload["model"]))
            self.target_model.eval()
            self.champion_state = payload["champion"]
            self.mixed_precision_enabled = self.runtime.cuda_enabled and bool(payload.get("mixed_precision_enabled", self.runtime.cuda_enabled))
            self.amp_overflow_fallbacks = max(0, int(payload.get("amp_overflow_fallbacks", 0)))
            self.grad_scaler = self._new_grad_scaler()
            if not checkpoint_migrated:
                self.optimizer.load_state_dict(payload["optimizer"])
                self._move_optimizer_state_to_device()
            else:
                log_training_debug(
                    "checkpoint_model_architecture_migrated",
                    checkpoint=str(checkpoint_path),
                    saved_format=saved_format,
                    current_format=MODEL_VERSION,
                    optimizer_reset=True,
                )
            self.champion_id = str(payload["champion_id"])
            self.champion_elo = float(payload.get("champion_elo", 1_200.0))
            self.league = [entry for entry in payload.get("league", []) if isinstance(entry, dict) and isinstance(entry.get("state"), dict) and entry.get("id") != self.champion_id][-9:]
            loaded_exploiters = [entry for entry in payload.get("exploiters", []) if isinstance(entry, dict) and isinstance(entry.get("state"), dict) and entry.get("id") != self.champion_id]
            self.exploiters = retain_unique_entries_by_id(loaded_exploiters, 4)
            loaded_specialists = [entry for entry in payload.get("specialist_archive", []) if isinstance(entry, dict) and isinstance(entry.get("state"), dict)]
            self.specialist_archive = retain_pareto_specialists(loaded_specialists, 8)
            restored_population = [member for member in payload.get("population_members", []) if isinstance(member, dict) and isinstance(member.get("state"), dict) and isinstance(member.get("id"), str)]
            if len(restored_population) == 3:
                self.population_members = restored_population
            for member in self.population_members:
                member.setdefault("bb_per_100", 0.0)
                member.setdefault("adversarial_bb_per_100", float(member["bb_per_100"]))
                member.setdefault("preflop_worst_lcb_bb_per_100", 0.0)
                member.setdefault("preflop_allin_probability", 0.0)
                member.setdefault("behavior_fold_rate", 0.0)
                member.setdefault("behavior_all_in_rate", 0.0)
                member.setdefault("behavior_degeneracy", 0.0)
                member.setdefault("quarantine_count", 0)
                member.setdefault("last_quarantine_reason", "")
                member.setdefault("safety_regressions", 0)
                member.setdefault("recovery_cooldown_until", 0)
            saved_anchor_state = payload.get("recovery_anchor_state")
            self.recovery_anchor_state = copy.deepcopy(saved_anchor_state) if isinstance(saved_anchor_state, dict) else copy.deepcopy(self.champion_state)
            saved_anchor_target = payload.get("recovery_anchor_target_state")
            self.recovery_anchor_target_state = copy.deepcopy(saved_anchor_target) if isinstance(saved_anchor_target, dict) else copy.deepcopy(self.recovery_anchor_state)
            self.recovery_anchor_score = float(payload.get("recovery_anchor_score", float("-inf")))
            self.recovery_anchor_metrics = {str(key): float(value) for key, value in payload.get("recovery_anchor_metrics", {}).items()} if isinstance(payload.get("recovery_anchor_metrics"), dict) else {}
            self.recovery_anchor_updates = int(payload.get("recovery_anchor_updates", 0))
            self.recovery_anchor_source = str(payload.get("recovery_anchor_source", "unverified"))
            self.recovery_safe_audits = max(0, int(payload.get("recovery_safe_audits", 0)))
            self.recovery_baseline_verified = bool(payload.get("recovery_baseline_verified", True))
            self.recovery_baseline_metrics = {str(key): float(value) for key, value in payload.get("recovery_baseline_metrics", {}).items()} if isinstance(payload.get("recovery_baseline_metrics"), dict) else {}
            self.last_recovery_candidate_metrics = {str(key): float(value) for key, value in payload.get("last_recovery_candidate_metrics", {}).items()} if isinstance(payload.get("last_recovery_candidate_metrics"), dict) else {}
            self.last_fresh_warmup_fold_collapse = bool(payload.get("last_fresh_warmup_fold_collapse", False))
            self.last_final_audit_checkpoint_restored = bool(payload.get("last_final_audit_checkpoint_restored", False))
            self.last_final_audit_restore_reason = str(payload.get("last_final_audit_restore_reason", ""))
            self.recovery_halted = bool(payload.get("recovery_halted", False))
            if self.recovery_anchor_source.startswith("protected_pre_recovery_checkpoint") and self.recovery_baseline_metrics:
                self.recovery_baseline_verified = self.recovery_baseline_verified and self._recovery_candidate_is_safe(self.recovery_baseline_metrics)
            self.active_population_index = max(0, min(len(self.population_members) - 1, int(payload.get("active_population_index", 0))))
            self.payoff_matrix = {str(key): float(value) for key, value in payload.get("payoff_matrix", {}).items()}
            self.mixture_regrets = {str(key): float(value) for key, value in payload.get("mixture_regrets", {}).items()}
            self.cfr_memory.restore(payload.get("cfr_memory", {}))
            self.strategy_memory.restore(payload.get("strategy_memory", {}))
            self.search_value_memory.restore(payload.get("search_value_memory", {}))
            self.action_likelihood_memory.restore(payload.get("action_likelihood_memory", {}))
            self.imitation_memory.restore(payload.get("imitation_memory", {}))
            self.hard_spot_value_memory.restore(payload.get("hard_spot_value_memory", {}))
            if isinstance(payload.get("grad_scaler"), dict):
                self.grad_scaler.load_state_dict(payload["grad_scaler"])
            self.mixture_regrets.setdefault(self.champion_id, 0.0)
            for style in BENCHMARK_STYLES:
                self.mixture_regrets.setdefault(f"style-{style}", 0.0)
            self.benchmarks = {str(key): {str(metric): float(value) for metric, value in metrics.items()} for key, metrics in payload.get("benchmarks", {}).items()}
            self.tournament_count = int(payload.get("tournament_count", 0))
            self.version = int(payload.get("version", 0))
            self.updates = int(payload.get("updates", 0))
            self.trained_hands = int(payload.get("trained_hands", 0))
            self.last_challenger_status = str(payload.get("last_challenger_status", "resumed"))
            self.last_promotion_confidence = float(payload.get("last_promotion_confidence", 0.0))
            self.last_direct_bb_per_100 = float(payload.get("last_direct_bb_per_100", 0.0))
            self.last_evaluation_bb_per_100 = float(payload.get("last_evaluation_bb_per_100", 0.0))
            self.last_promotion_ci_lower = float(payload.get("last_promotion_ci_lower", 0.0))
            self.last_promotion_ci_upper = float(payload.get("last_promotion_ci_upper", 0.0))
            self.last_holdout_bb_per_100 = float(payload.get("last_holdout_bb_per_100", 0.0))
            self.last_holdout_floor_bb_per_100 = float(payload.get("last_holdout_floor_bb_per_100", 0.0))
            self.last_opponent_pressure = float(payload.get("last_opponent_pressure", 0.0))
            self.last_rare_spot_rate = float(payload.get("last_rare_spot_rate", 0.0))
            self.last_belief_confidence = float(payload.get("last_belief_confidence", 0.0))
            self.last_leaf_evaluations = int(payload.get("last_leaf_evaluations", 0))
            self.last_best_response_bb_per_100 = float(payload.get("last_best_response_bb_per_100", 0.0))
            self.last_target_drift = float(payload.get("last_target_drift", 0.0))
            self.curriculum_unlocked_stage = max(0, min(len(CURRICULUM_PHASES) - 1, int(payload.get("curriculum_unlocked_stage", 0))))
            self.last_curriculum_readiness = float(payload.get("last_curriculum_readiness", 0.0))
            self.last_training_lane = str(payload.get("last_training_lane", "population"))
            self.last_replay_rare_fraction = float(payload.get("last_replay_rare_fraction", 0.0))
            self.last_replay_priority = float(payload.get("last_replay_priority", 0.0))
            self.last_replay_recent_fraction = float(payload.get("last_replay_recent_fraction", 0.0))
            self.last_exploiter_diversity = float(payload.get("last_exploiter_diversity", 0.0))
            self.last_exploiter_threat = float(payload.get("last_exploiter_threat", 0.0))
            self.last_champion_vulnerability = float(payload.get("last_champion_vulnerability", 0.0))
            self.exploiter_generations = int(payload.get("exploiter_generations", 0))
            self.exploiter_lane_remaining = max(0, int(payload.get("exploiter_lane_remaining", 0)))
            self.ppo_learning_rate = float(payload.get("ppo_learning_rate", self.ppo_learning_rate))
            self.ppo_clip_epsilon = float(payload.get("ppo_clip_epsilon", self.ppo_clip_epsilon))
            self.ppo_entropy_coefficient = float(payload.get("ppo_entropy_coefficient", self.ppo_entropy_coefficient))
            self.ppo_kl_target = float(payload.get("ppo_kl_target", self.ppo_kl_target))
            self.last_ppo_epochs = int(payload.get("last_ppo_epochs", 0))
            self.last_ppo_clip_fraction = float(payload.get("last_ppo_clip_fraction", 0.0))
            self.ppo_recovery_updates = max(0, int(payload.get("ppo_recovery_updates", 0)))
            self.last_ppo_kl_limited = bool(payload.get("last_ppo_kl_limited", False))
            self.last_ppo_hard_kl = float(payload.get("last_ppo_hard_kl", self.ppo_kl_target * PPO_HARD_KL_MULTIPLIER))
            self.last_ppo_epoch_budget = max(1, int(payload.get("last_ppo_epoch_budget", PPO_MAX_EPOCHS)))
            self.last_ppo_update_reverted = bool(payload.get("last_ppo_update_reverted", False))
            self.last_ppo_rollback_phase = str(payload.get("last_ppo_rollback_phase", "none"))
            self.last_ppo_post_step_retry_applied = bool(payload.get("last_ppo_post_step_retry_applied", False))
            self.last_ppo_post_step_retry_accepted = bool(payload.get("last_ppo_post_step_retry_accepted", False))
            self.last_ppo_post_step_retry_kl = float(payload.get("last_ppo_post_step_retry_kl", 0.0))
            self.ppo_post_step_retry_scale = min(0.80, max(0.10, float(payload.get("ppo_post_step_retry_scale", PPO_POST_STEP_RETRY_SCALE))))
            self.last_imitation_loss = float(payload.get("last_imitation_loss", 0.0))
            self.last_imitation_reward = float(payload.get("last_imitation_reward", 0.0))
            self.last_evaluation_hands = int(payload.get("last_evaluation_hands", 0))
            self.last_holdout_score = float(payload.get("last_holdout_score", 0.0))
            self.last_holdout_floor = float(payload.get("last_holdout_floor", 0.0))
            self.last_continuous_raise_mean = float(payload.get("last_continuous_raise_mean", 0.5))
            self.last_sizing_cfr_loss = float(payload.get("last_sizing_cfr_loss", 0.0))
            self.last_strategy_memory_size = int(payload.get("last_strategy_memory_size", len(self.strategy_memory.records)))
            self.last_scenario_coverage = float(payload.get("last_scenario_coverage", 0.0))
            self.last_restricted_br_bb_per_100 = float(payload.get("last_restricted_br_bb_per_100", 0.0))
            self.last_adversarial_floor_bb_per_100 = float(payload.get("last_adversarial_floor_bb_per_100", 0.0))
            self.last_adversarial_rollout_fraction = float(payload.get("last_adversarial_rollout_fraction", 0.0))
            self.last_adversarial_focus = str(payload.get("last_adversarial_focus", "pending"))
            restored_style_results = payload.get("adversarial_style_bb_per_100", {})
            if isinstance(restored_style_results, dict):
                self.adversarial_style_bb_per_100 = {style: float(restored_style_results.get(style, 0.0)) for style in ADVERSARIAL_TRAINING_STYLES}
            self.last_compiled_transition_fraction = float(payload.get("last_compiled_transition_fraction", 0.0))
            self.last_crossplay_robustness = float(payload.get("last_crossplay_robustness", 0.0))
            self.last_population_continuity = float(payload.get("last_population_continuity", 0.0))
            self.last_ensemble_disagreement = float(payload.get("last_ensemble_disagreement", 0.0))
            self.last_search_value_loss = float(payload.get("last_search_value_loss", 0.0))
            self.last_search_memory_size = int(payload.get("last_search_memory_size", len(self.search_value_memory.records)))
            self.last_snapshot_diversity = float(payload.get("last_snapshot_diversity", 0.0))
            self.last_snapshot_min_distance = float(payload.get("last_snapshot_min_distance", 0.0))
            self.snapshot_rejections = max(0, int(payload.get("snapshot_rejections", 0)))
            self.last_adaptive_action_width = float(payload.get("last_adaptive_action_width", 0.0))
            self.last_adversarial_ci_floor_bb_per_100 = float(payload.get("last_adversarial_ci_floor_bb_per_100", 0.0))
            self.last_adversarial_evaluation_hands = max(0, int(payload.get("last_adversarial_evaluation_hands", ADVERSARIAL_SCREENING_HANDS)))
            self.last_adversarial_confirmation_hands = max(0, int(payload.get("last_adversarial_confirmation_hands", 0)))
            self.last_final_audit_ran = bool(payload.get("last_final_audit_ran", False))
            self.last_tail_loss_rate = float(payload.get("last_tail_loss_rate", 0.0))
            self.last_tail_loss_bb = float(payload.get("last_tail_loss_bb", 0.0))
            self.last_tail_policy_weight = float(payload.get("last_tail_policy_weight", 1.0))
            saved_tail_diagnostics = payload.get("last_tail_style_diagnostics", {})
            self.last_tail_style_diagnostics = {
                str(style): copy.deepcopy(metrics)
                for style, metrics in saved_tail_diagnostics.items()
                if isinstance(metrics, dict)
            }
            self.last_hard_spot_value_loss = float(payload.get("last_hard_spot_value_loss", 0.0))
            self.last_hard_spot_memory_size = int(payload.get("last_hard_spot_memory_size", len(self.hard_spot_value_memory.records)))
            self.last_behavior_action_agreement = float(payload.get("last_behavior_action_agreement", 1.0))
            self.last_behavior_action_change_rate = float(payload.get("last_behavior_action_change_rate", 0.0))
            self.last_behavior_raise_fraction_delta = float(payload.get("last_behavior_raise_fraction_delta", 0.0))
            self.last_behavior_audit_states = int(payload.get("last_behavior_audit_states", 0))
            saved_preflop_audit = payload.get("last_preflop_sizing_audit", {})
            if isinstance(saved_preflop_audit, dict):
                self.last_preflop_sizing_audit.update({str(key): float(value) if isinstance(value, (float, int)) else value for key, value in saved_preflop_audit.items()})
            saved_root_weakness = payload.get("preflop_root_weakness", {})
            if isinstance(saved_root_weakness, dict):
                self.preflop_root_weakness = {root: min(1.0, max(0.0, float(saved_root_weakness.get(root, 0.50)))) for root in PREFLOP_FORCED_ROOTS}
            saved_preflop_scenarios = payload.get("last_preflop_scenario_audit", {})
            if isinstance(saved_preflop_scenarios, dict):
                self.last_preflop_scenario_audit = copy.deepcopy(saved_preflop_scenarios)
            self.last_preflop_root_fraction = min(1.0, max(0.0, float(payload.get("last_preflop_root_fraction", 0.0))))
            self.last_preflop_scenario_audit_hands = max(0, int(payload.get("last_preflop_scenario_audit_hands", 0)))
            self.last_preflop_scenario_worst_lcb_bb_per_100 = float(payload.get("last_preflop_scenario_worst_lcb_bb_per_100", 0.0))
            self.last_preflop_scenario_worst_root = str(payload.get("last_preflop_scenario_worst_root", "pending"))
            self.last_preflop_scenario_worst_style = str(payload.get("last_preflop_scenario_worst_style", "pending"))
            if self.last_preflop_scenario_worst_root in self.preflop_root_weakness and self.last_preflop_scenario_worst_lcb_bb_per_100 < PREFLOP_ROOT_PROMOTION_LCB_FLOOR:
                severity = min(1.0, max(0.0, (PREFLOP_ROOT_PROMOTION_LCB_FLOOR - self.last_preflop_scenario_worst_lcb_bb_per_100) / 400.0))
                priority = 0.82 + 0.18 * severity
                self.preflop_root_weakness[self.last_preflop_scenario_worst_root] = max(self.preflop_root_weakness[self.last_preflop_scenario_worst_root], priority)
            self.last_preflop_allin_calibration_loss = float(payload.get("last_preflop_allin_calibration_loss", 0.0))
            self.last_preflop_allin_stability_loss = float(payload.get("last_preflop_allin_stability_loss", 0.0))
            self.last_preflop_guarded_allin_probability = float(payload.get("last_preflop_guarded_allin_probability", 0.0))
            self.last_preflop_allin_target = float(payload.get("last_preflop_allin_target", 0.0))
            self.last_preflop_guarded_state_fraction = float(payload.get("last_preflop_guarded_state_fraction", 0.0))
            self.last_preflop_immediate_allin_rate = float(payload.get("last_preflop_immediate_allin_rate", 0.0))
            self.last_preflop_immediate_allin_target = float(payload.get("last_preflop_immediate_allin_target", 0.0))
            self.last_preflop_immediate_eligible_rate = float(payload.get("last_preflop_immediate_eligible_rate", 0.0))
            self.last_preflop_3bet_teacher_loss = float(payload.get("last_preflop_3bet_teacher_loss", 0.0))
            self.last_preflop_3bet_teacher_eligible_roots = max(0, int(payload.get("last_preflop_3bet_teacher_eligible_roots", 0)))
            self.last_preflop_3bet_teacher_samples = max(0, int(payload.get("last_preflop_3bet_teacher_samples", 0)))
            self.last_preflop_3bet_teacher_coverage = float(payload.get("last_preflop_3bet_teacher_coverage", 0.0))
            self.last_preflop_3bet_teacher_confidence = float(payload.get("last_preflop_3bet_teacher_confidence", 0.0))
            self.last_preflop_3bet_teacher_effective_coverage = float(payload.get("last_preflop_3bet_teacher_effective_coverage", 0.0))
            self.last_preflop_3bet_teacher_effective_weight = float(payload.get("last_preflop_3bet_teacher_effective_weight", 0.0))
            self.last_preflop_3bet_teacher_raise_target = float(payload.get("last_preflop_3bet_teacher_raise_target", 0.0))
            self.last_preflop_3bet_teacher_raise_advantage_bb = float(payload.get("last_preflop_3bet_teacher_raise_advantage_bb", 0.0))
            self.last_preflop_3bet_teacher_actual_raise_rate = float(payload.get("last_preflop_3bet_teacher_actual_raise_rate", 0.0))
            self.last_preflop_3bet_teacher_allin_target = float(payload.get("last_preflop_3bet_teacher_allin_target", 0.0))
            self.last_preflop_3bet_teacher_actual_allin_rate = float(payload.get("last_preflop_3bet_teacher_actual_allin_rate", 0.0))
            self.last_preflop_3bet_teacher_allin_suppressed = max(0, int(payload.get("last_preflop_3bet_teacher_allin_suppressed", 0)))
            self.last_preflop_3bet_teacher_multi_raise_samples = max(0, int(payload.get("last_preflop_3bet_teacher_multi_raise_samples", 0)))
            self.last_preflop_3bet_teacher_multi_raise_allin_target = float(payload.get("last_preflop_3bet_teacher_multi_raise_allin_target", 0.0))
            self.last_preflop_3bet_teacher_multi_raise_actual_allin_rate = float(payload.get("last_preflop_3bet_teacher_multi_raise_actual_allin_rate", 0.0))
            self.last_preflop_3bet_teacher_multi_raise_allin_vetoes = max(0, int(payload.get("last_preflop_3bet_teacher_multi_raise_allin_vetoes", 0)))
            self.last_preflop_3bet_teacher_facing_4bet_samples = max(0, int(payload.get("last_preflop_3bet_teacher_facing_4bet_samples", 0)))
            saved_4bet_targets = payload.get("last_preflop_3bet_teacher_facing_4bet_target_actions", {})
            saved_4bet_actual = payload.get("last_preflop_3bet_teacher_facing_4bet_actual_actions", {})
            self.last_preflop_3bet_teacher_facing_4bet_target_actions = {name: float(saved_4bet_targets.get(name, 0.0)) for name in PREFLOP_3BET_TEACHER_ACTION_NAMES} if isinstance(saved_4bet_targets, dict) else {name: 0.0 for name in PREFLOP_3BET_TEACHER_ACTION_NAMES}
            self.last_preflop_3bet_teacher_facing_4bet_actual_actions = {name: float(saved_4bet_actual.get(name, 0.0)) for name in PREFLOP_3BET_TEACHER_ACTION_NAMES} if isinstance(saved_4bet_actual, dict) else {name: 0.0 for name in PREFLOP_3BET_TEACHER_ACTION_NAMES}
            self.last_preflop_3bet_teacher_facing_4bet_non_allin_vetoes = max(0, int(payload.get("last_preflop_3bet_teacher_facing_4bet_non_allin_vetoes", 0)))
            saved_teacher_by_root = payload.get("last_preflop_3bet_teacher_by_root", {})
            self.last_preflop_3bet_teacher_by_root = copy.deepcopy(saved_teacher_by_root) if isinstance(saved_teacher_by_root, dict) else {}
            self.last_robust_policy_weight = max(1.0, float(payload.get("last_robust_policy_weight", 1.0)))
            self.last_rollout_inference_device = str(payload.get("last_rollout_inference_device", "cpu"))
            self.scenario_weakness = {profile: min(1.0, max(0.0, float(payload.get("scenario_weakness", {}).get(profile, 0.50)))) for profile in SCENARIO_PROFILES}
            self.opponent_weakness = {str(key): min(1.0, max(0.0, float(value))) for key, value in payload.get("opponent_weakness", {}).items()}
            self.last_training_focus = str(payload.get("last_training_focus", "balanced"))
            self.last_weakness_score = float(payload.get("last_weakness_score", 0.50))
            self.last_adaptive_workers = int(payload.get("last_adaptive_workers", 0))
            self.last_adaptive_batch_hands = int(payload.get("last_adaptive_batch_hands", 0))
            self.last_rollout_decisions_per_second = float(payload.get("last_rollout_decisions_per_second", 0.0))
            self.rollout_scale = min(2.0, max(0.55, float(payload.get("rollout_scale", 1.0))))
            teacher_report = payload.get("teacher_data_report", {})
            self.teacher_data_report = HandHistoryReport(str(teacher_report.get("filename", "")), int(teacher_report.get("accepted", 0)), int(teacher_report.get("rejected", 0)), str(teacher_report.get("message", "no local teacher data")))
            self.teacher_data_records = int(payload.get("teacher_data_records", 0))
            self.audit_benchmarks = {str(key): {str(metric): float(value) for metric, value in metrics.items()} for key, metrics in payload.get("audit_benchmarks", {}).items()}
            self.evaluation_history = [dict(item) for item in payload.get("evaluation_history", []) if isinstance(item, dict)][-24:]
            self.last_audit_score = float(payload.get("last_audit_score", 0.0))
            self.last_audit_exploitability_bb_per_100 = float(payload.get("last_audit_exploitability_bb_per_100", 0.0))
            self.last_scenario_gate = float(payload.get("last_scenario_gate", 0.0))
            self.last_ablation_delta = float(payload.get("last_ablation_delta", 0.0))
            self.last_subgame_policy_loss = float(payload.get("last_subgame_policy_loss", 0.0))
            self.last_subgame_teacher_size = int(payload.get("last_subgame_teacher_size", 0))
            self.last_rollout_arena_width = int(payload.get("last_rollout_arena_width", 0))
            self.last_average_strategy_weight = float(payload.get("last_average_strategy_weight", 0.0))
            self.abstract_oracle.restore(payload.get("abstract_oracle", {}))
            saved_abstraction_solver = payload.get("abstract_cfr_solver")
            if isinstance(saved_abstraction_solver, dict) and saved_abstraction_solver:
                self.abstract_cfr_solver.restore(saved_abstraction_solver)
            self.abstract_teacher_memory.restore(payload.get("abstract_teacher_memory", {}))
            self.last_oracle_policy_loss = float(payload.get("last_oracle_policy_loss", 0.0))
            self.last_oracle_value_loss = float(payload.get("last_oracle_value_loss", 0.0))
            self.last_oracle_confidence = float(payload.get("last_oracle_confidence", 0.0))
            self.last_oracle_iterations = int(payload.get("last_oracle_iterations", self.abstract_cfr_solver.iterations))
            self.last_abstraction_nash_conv = float(payload.get("last_abstraction_nash_conv", self.last_abstraction_nash_conv))
            self.last_abstraction_value = float(payload.get("last_abstraction_value", self.last_abstraction_value))
            self.last_abstraction_information_sets = int(payload.get("last_abstraction_information_sets", self.last_abstraction_information_sets))
            self.last_holdout_ci_floor_bb_per_100 = float(payload.get("last_holdout_ci_floor_bb_per_100", 0.0))
            self.last_holdout_paired_variance = float(payload.get("last_holdout_paired_variance", 0.0))
            self.last_paired_deal_coverage = float(payload.get("last_paired_deal_coverage", 0.0))
            self.last_belief_posterior_support = float(payload.get("last_belief_posterior_support", 1.0))
            self.last_resolver_replay_confidence = float(payload.get("last_resolver_replay_confidence", 0.0))
            self.last_resolver_replay_size = int(payload.get("last_resolver_replay_size", 0))
            self.last_blueprint_score = float(payload.get("last_blueprint_score", 0.0))
            self.last_blueprint_confidence = float(payload.get("last_blueprint_confidence", 0.0))
            self.last_blueprint_floor = float(payload.get("last_blueprint_floor", 0.0))
            self.last_blueprint_hands = int(payload.get("last_blueprint_hands", 0))
            self.last_kuhn_value_gap = float(payload.get("last_kuhn_value_gap", 1.0))
            self.last_blueprint_status = str(payload.get("last_blueprint_status", "not audited"))
            self.counterfactual_value_memory.restore(payload.get("counterfactual_value_memory", {}))
            self.last_counterfactual_value_loss = float(payload.get("last_counterfactual_value_loss", 0.0))
            self.last_counterfactual_coverage = float(payload.get("last_counterfactual_coverage", 0.0))
            self.last_counterfactual_memory_size = int(payload.get("last_counterfactual_memory_size", len(self.counterfactual_value_memory.records)))
            self.last_public_belief_teacher_size = int(payload.get("last_public_belief_teacher_size", 0))
            self.last_sizing_proposal_diversity = float(payload.get("last_sizing_proposal_diversity", 0.0))
            self.strategy_snapshots = [snapshot for snapshot in payload.get("strategy_snapshots", []) if isinstance(snapshot, dict) and isinstance(snapshot.get("state"), dict)][-6:]
            self.recovery_revalidation_required = int(payload.get("policy_execution_version", 1)) < POLICY_EXECUTION_VERSION
            if self.recovery_revalidation_required:
                # Greedy-policy confidence scores and curriculum promotions are
                # not comparable with mixed-strategy deployment.  Rebuild the
                # safety baseline at foundation depth before accepting updates.
                current_state = clone_state(self.model)
                current_target_state = clone_state(self.target_model)
                self.curriculum_unlocked_stage = 0
                self.recovery_anchor_state = copy.deepcopy(current_state)
                self.recovery_anchor_target_state = copy.deepcopy(current_target_state)
                self.recovery_anchor_score = float("-inf")
                self.recovery_anchor_metrics = {}
                self.recovery_anchor_updates = self.updates
                self.recovery_anchor_source = f"policy_execution_v{POLICY_EXECUTION_VERSION}_revalidation"
                self.recovery_safe_audits = 0
                self.recovery_baseline_verified = False
                self.recovery_baseline_metrics = {}
                self.recovery_halted = False
                self.active_population_index = 0
                for member in self.population_members:
                    member.update({
                        "state": copy.deepcopy(current_state),
                        "target_state": copy.deepcopy(current_target_state),
                        "optimizer_state": None,
                        "grad_scaler_state": None,
                        "score": 0.50,
                        "bb_per_100": 0.0,
                        "adversarial_bb_per_100": 0.0,
                        "preflop_worst_lcb_bb_per_100": 0.0,
                        "preflop_allin_probability": 0.0,
                        "behavior_fold_rate": 0.0,
                        "behavior_all_in_rate": 0.0,
                        "behavior_degeneracy": 0.0,
                        "safety_regressions": 0,
                        "recovery_cooldown_until": self.updates,
                        "updates": 0,
                    })
                log_training_debug(
                    "checkpoint_policy_execution_revalidation_required",
                    checkpoint=str(checkpoint_path),
                    saved_version=int(payload.get("policy_execution_version", 1)),
                    required_version=POLICY_EXECUTION_VERSION,
                    curriculum_stage=0,
                )
            for group in self.optimizer.param_groups:
                group["lr"] = self.ppo_learning_rate
            self._rng.setstate(payload["python_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            if self.runtime.cuda_enabled and isinstance(payload.get("cuda_rng_state"), Tensor):
                torch.cuda.set_rng_state(payload["cuda_rng_state"], self.runtime.device)
            self.resumed = True
            with self._lock:
                self._refresh_snapshot_diversity_locked()
                self._publish_locked()
            return True
        except (AttributeError, EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as exc:
            log_training_debug(
                "checkpoint_load_failed",
                checkpoint=str(checkpoint_path),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )
            return False

    def save(self) -> None:
        """Atomically publish a checkpoint while retaining the immediately prior one."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._capture_active_member_locked()
            payload = {"format": MODEL_VERSION, "policy_execution_version": POLICY_EXECUTION_VERSION, "version": self.version, "updates": self.updates, "trained_hands": self.trained_hands, "model": clone_state(self.model), "target_model": clone_state(self.target_model), "optimizer": self.optimizer.state_dict(), "grad_scaler": self.grad_scaler.state_dict(), "champion": self.champion_state, "champion_id": self.champion_id, "champion_elo": self.champion_elo, "league": self.league[-9:], "exploiters": self.exploiters[-4:], "specialist_archive": self.specialist_archive[-8:], "population_members": self.population_members, "active_population_index": self.active_population_index, "strategy_snapshots": self.strategy_snapshots[-6:], "payoff_matrix": self.payoff_matrix, "mixture_regrets": self.mixture_regrets, "cfr_memory": self.cfr_memory.snapshot(), "strategy_memory": self.strategy_memory.snapshot(), "search_value_memory": self.search_value_memory.snapshot(), "action_likelihood_memory": self.action_likelihood_memory.snapshot(), "imitation_memory": self.imitation_memory.snapshot(), "hard_spot_value_memory": self.hard_spot_value_memory.snapshot(), "benchmarks": self.benchmarks, "tournament_count": self.tournament_count, "last_challenger_status": self.last_challenger_status, "last_promotion_confidence": self.last_promotion_confidence, "last_direct_bb_per_100": self.last_direct_bb_per_100, "last_evaluation_bb_per_100": self.last_evaluation_bb_per_100, "last_promotion_ci_lower": self.last_promotion_ci_lower, "last_promotion_ci_upper": self.last_promotion_ci_upper, "last_holdout_bb_per_100": self.last_holdout_bb_per_100, "last_holdout_floor_bb_per_100": self.last_holdout_floor_bb_per_100, "last_opponent_pressure": self.last_opponent_pressure, "last_rare_spot_rate": self.last_rare_spot_rate, "last_belief_confidence": self.last_belief_confidence, "last_leaf_evaluations": self.last_leaf_evaluations, "last_best_response_bb_per_100": self.last_best_response_bb_per_100, "last_target_drift": self.last_target_drift, "curriculum_unlocked_stage": self.curriculum_unlocked_stage, "last_curriculum_readiness": self.last_curriculum_readiness, "last_training_lane": self.last_training_lane, "last_replay_rare_fraction": self.last_replay_rare_fraction, "last_replay_priority": self.last_replay_priority, "last_replay_recent_fraction": self.last_replay_recent_fraction, "last_exploiter_diversity": self.last_exploiter_diversity, "ppo_learning_rate": self.ppo_learning_rate, "ppo_clip_epsilon": self.ppo_clip_epsilon, "ppo_entropy_coefficient": self.ppo_entropy_coefficient, "ppo_kl_target": self.ppo_kl_target, "last_ppo_epochs": self.last_ppo_epochs, "last_ppo_clip_fraction": self.last_ppo_clip_fraction, "last_imitation_loss": self.last_imitation_loss, "last_imitation_reward": self.last_imitation_reward, "last_evaluation_hands": self.last_evaluation_hands, "last_holdout_score": self.last_holdout_score, "last_holdout_floor": self.last_holdout_floor, "last_continuous_raise_mean": self.last_continuous_raise_mean, "last_sizing_cfr_loss": self.last_sizing_cfr_loss, "last_strategy_memory_size": self.last_strategy_memory_size, "last_scenario_coverage": self.last_scenario_coverage, "last_restricted_br_bb_per_100": self.last_restricted_br_bb_per_100, "last_adversarial_floor_bb_per_100": self.last_adversarial_floor_bb_per_100, "last_adversarial_rollout_fraction": self.last_adversarial_rollout_fraction, "last_crossplay_robustness": self.last_crossplay_robustness, "last_population_continuity": self.last_population_continuity, "last_ensemble_disagreement": self.last_ensemble_disagreement, "last_search_value_loss": self.last_search_value_loss, "last_search_memory_size": self.last_search_memory_size, "last_snapshot_diversity": self.last_snapshot_diversity, "last_snapshot_min_distance": self.last_snapshot_min_distance, "snapshot_rejections": self.snapshot_rejections, "last_adaptive_action_width": self.last_adaptive_action_width, "python_rng_state": self._rng.getstate(), "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(self.runtime.device) if self.runtime.cuda_enabled else None}
        payload.update({
            "scenario_weakness": self.scenario_weakness,
            "opponent_weakness": self.opponent_weakness,
            "last_training_focus": self.last_training_focus,
            "last_weakness_score": self.last_weakness_score,
            "last_adaptive_workers": self.last_adaptive_workers,
            "last_adaptive_batch_hands": self.last_adaptive_batch_hands,
            "last_rollout_decisions_per_second": self.last_rollout_decisions_per_second,
            "rollout_scale": self.rollout_scale,
            "teacher_data_records": self.teacher_data_records,
            "teacher_data_report": self.teacher_data_report.payload(),
            "audit_benchmarks": self.audit_benchmarks,
            "evaluation_history": self.evaluation_history[-24:],
            "last_audit_score": self.last_audit_score,
            "last_audit_exploitability_bb_per_100": self.last_audit_exploitability_bb_per_100,
            "last_scenario_gate": self.last_scenario_gate,
            "last_ablation_delta": self.last_ablation_delta,
            "last_subgame_policy_loss": self.last_subgame_policy_loss,
            "last_subgame_teacher_size": self.last_subgame_teacher_size,
            "last_rollout_arena_width": self.last_rollout_arena_width,
            "last_average_strategy_weight": self.last_average_strategy_weight,
            "abstract_oracle": self.abstract_oracle.snapshot(),
            "abstract_cfr_solver": self.abstract_cfr_solver.snapshot(),
            "abstract_teacher_memory": self.abstract_teacher_memory.snapshot(),
            "last_oracle_policy_loss": self.last_oracle_policy_loss,
            "last_oracle_value_loss": self.last_oracle_value_loss,
            "last_oracle_confidence": self.last_oracle_confidence,
            "last_oracle_iterations": self.last_oracle_iterations,
            "last_abstraction_nash_conv": self.last_abstraction_nash_conv,
            "last_abstraction_value": self.last_abstraction_value,
            "last_abstraction_information_sets": self.last_abstraction_information_sets,
            "last_holdout_ci_floor_bb_per_100": self.last_holdout_ci_floor_bb_per_100,
            "last_holdout_paired_variance": self.last_holdout_paired_variance,
            "last_paired_deal_coverage": self.last_paired_deal_coverage,
            "last_belief_posterior_support": self.last_belief_posterior_support,
            "last_resolver_replay_confidence": self.last_resolver_replay_confidence,
            "last_resolver_replay_size": self.last_resolver_replay_size,
            "last_blueprint_score": self.last_blueprint_score,
            "last_blueprint_confidence": self.last_blueprint_confidence,
            "last_blueprint_floor": self.last_blueprint_floor,
            "last_blueprint_hands": self.last_blueprint_hands,
            "last_kuhn_value_gap": self.last_kuhn_value_gap,
            "last_blueprint_status": self.last_blueprint_status,
            "counterfactual_value_memory": self.counterfactual_value_memory.snapshot(),
            "last_counterfactual_value_loss": self.last_counterfactual_value_loss,
            "last_counterfactual_coverage": self.last_counterfactual_coverage,
            "last_counterfactual_memory_size": self.last_counterfactual_memory_size,
            "last_public_belief_teacher_size": self.last_public_belief_teacher_size,
            "last_sizing_proposal_diversity": self.last_sizing_proposal_diversity,
            "last_exploiter_threat": self.last_exploiter_threat,
            "last_champion_vulnerability": self.last_champion_vulnerability,
            "exploiter_generations": self.exploiter_generations,
            "exploiter_lane_remaining": self.exploiter_lane_remaining,
            "ppo_recovery_updates": self.ppo_recovery_updates,
            "last_ppo_kl_limited": self.last_ppo_kl_limited,
            "last_ppo_hard_kl": self.last_ppo_hard_kl,
            "last_ppo_epoch_budget": self.last_ppo_epoch_budget,
            "last_ppo_update_reverted": self.last_ppo_update_reverted,
            "last_ppo_rollback_phase": self.last_ppo_rollback_phase,
            "last_ppo_post_step_retry_applied": self.last_ppo_post_step_retry_applied,
            "last_ppo_post_step_retry_accepted": self.last_ppo_post_step_retry_accepted,
            "last_ppo_post_step_retry_kl": self.last_ppo_post_step_retry_kl,
            "ppo_post_step_retry_scale": self.ppo_post_step_retry_scale,
            "mixed_precision_enabled": self.mixed_precision_enabled,
            "amp_overflow_fallbacks": self.amp_overflow_fallbacks,
            "adversarial_style_bb_per_100": self.adversarial_style_bb_per_100,
            "last_adversarial_focus": self.last_adversarial_focus,
            "last_compiled_transition_fraction": self.last_compiled_transition_fraction,
            "last_adversarial_ci_floor_bb_per_100": self.last_adversarial_ci_floor_bb_per_100,
            "last_adversarial_evaluation_hands": self.last_adversarial_evaluation_hands,
            "last_adversarial_confirmation_hands": self.last_adversarial_confirmation_hands,
            "last_final_audit_ran": self.last_final_audit_ran,
            "last_tail_loss_rate": self.last_tail_loss_rate,
            "last_tail_loss_bb": self.last_tail_loss_bb,
            "last_tail_policy_weight": self.last_tail_policy_weight,
            "last_tail_style_diagnostics": self.last_tail_style_diagnostics,
            "last_hard_spot_value_loss": self.last_hard_spot_value_loss,
            "last_hard_spot_memory_size": self.last_hard_spot_memory_size,
            "last_behavior_action_agreement": self.last_behavior_action_agreement,
            "last_behavior_action_change_rate": self.last_behavior_action_change_rate,
            "last_behavior_raise_fraction_delta": self.last_behavior_raise_fraction_delta,
            "last_behavior_audit_states": self.last_behavior_audit_states,
            "last_preflop_sizing_audit": self.last_preflop_sizing_audit,
            "preflop_root_weakness": self.preflop_root_weakness,
            "last_preflop_root_fraction": self.last_preflop_root_fraction,
            "last_preflop_scenario_audit": self.last_preflop_scenario_audit,
            "last_preflop_scenario_audit_hands": self.last_preflop_scenario_audit_hands,
            "last_preflop_scenario_worst_lcb_bb_per_100": self.last_preflop_scenario_worst_lcb_bb_per_100,
            "last_preflop_scenario_worst_root": self.last_preflop_scenario_worst_root,
            "last_preflop_scenario_worst_style": self.last_preflop_scenario_worst_style,
            "last_preflop_allin_calibration_loss": self.last_preflop_allin_calibration_loss,
            "last_preflop_allin_stability_loss": self.last_preflop_allin_stability_loss,
            "last_preflop_guarded_allin_probability": self.last_preflop_guarded_allin_probability,
            "last_preflop_allin_target": self.last_preflop_allin_target,
            "last_preflop_guarded_state_fraction": self.last_preflop_guarded_state_fraction,
            "last_preflop_immediate_allin_rate": self.last_preflop_immediate_allin_rate,
            "last_preflop_immediate_allin_target": self.last_preflop_immediate_allin_target,
            "last_preflop_immediate_eligible_rate": self.last_preflop_immediate_eligible_rate,
            "last_preflop_3bet_teacher_loss": self.last_preflop_3bet_teacher_loss,
            "last_preflop_3bet_teacher_eligible_roots": self.last_preflop_3bet_teacher_eligible_roots,
            "last_preflop_3bet_teacher_samples": self.last_preflop_3bet_teacher_samples,
            "last_preflop_3bet_teacher_coverage": self.last_preflop_3bet_teacher_coverage,
            "last_preflop_3bet_teacher_confidence": self.last_preflop_3bet_teacher_confidence,
            "last_preflop_3bet_teacher_effective_coverage": self.last_preflop_3bet_teacher_effective_coverage,
            "last_preflop_3bet_teacher_effective_weight": self.last_preflop_3bet_teacher_effective_weight,
            "last_preflop_3bet_teacher_raise_target": self.last_preflop_3bet_teacher_raise_target,
            "last_preflop_3bet_teacher_raise_advantage_bb": self.last_preflop_3bet_teacher_raise_advantage_bb,
            "last_preflop_3bet_teacher_actual_raise_rate": self.last_preflop_3bet_teacher_actual_raise_rate,
            "last_preflop_3bet_teacher_allin_target": self.last_preflop_3bet_teacher_allin_target,
            "last_preflop_3bet_teacher_actual_allin_rate": self.last_preflop_3bet_teacher_actual_allin_rate,
            "last_preflop_3bet_teacher_allin_suppressed": self.last_preflop_3bet_teacher_allin_suppressed,
            "last_preflop_3bet_teacher_multi_raise_samples": self.last_preflop_3bet_teacher_multi_raise_samples,
            "last_preflop_3bet_teacher_multi_raise_allin_target": self.last_preflop_3bet_teacher_multi_raise_allin_target,
            "last_preflop_3bet_teacher_multi_raise_actual_allin_rate": self.last_preflop_3bet_teacher_multi_raise_actual_allin_rate,
            "last_preflop_3bet_teacher_multi_raise_allin_vetoes": self.last_preflop_3bet_teacher_multi_raise_allin_vetoes,
            "last_preflop_3bet_teacher_facing_4bet_samples": self.last_preflop_3bet_teacher_facing_4bet_samples,
            "last_preflop_3bet_teacher_facing_4bet_target_actions": self.last_preflop_3bet_teacher_facing_4bet_target_actions,
            "last_preflop_3bet_teacher_facing_4bet_actual_actions": self.last_preflop_3bet_teacher_facing_4bet_actual_actions,
            "last_preflop_3bet_teacher_facing_4bet_non_allin_vetoes": self.last_preflop_3bet_teacher_facing_4bet_non_allin_vetoes,
            "last_preflop_3bet_teacher_by_root": self.last_preflop_3bet_teacher_by_root,
            "last_robust_policy_weight": self.last_robust_policy_weight,
            "last_rollout_inference_device": self.last_rollout_inference_device,
            "recovery_anchor_state": self.recovery_anchor_state,
            "recovery_anchor_target_state": self.recovery_anchor_target_state,
            "recovery_anchor_score": self.recovery_anchor_score,
            "recovery_anchor_metrics": self.recovery_anchor_metrics,
            "recovery_anchor_updates": self.recovery_anchor_updates,
            "recovery_anchor_source": self.recovery_anchor_source,
            "recovery_safe_audits": self.recovery_safe_audits,
            "recovery_baseline_verified": self.recovery_baseline_verified,
            "recovery_baseline_metrics": self.recovery_baseline_metrics,
            "last_recovery_candidate_metrics": self.last_recovery_candidate_metrics,
            "last_fresh_warmup_fold_collapse": self.last_fresh_warmup_fold_collapse,
            "last_final_audit_checkpoint_restored": self.last_final_audit_checkpoint_restored,
            "last_final_audit_restore_reason": self.last_final_audit_restore_reason,
            "recovery_halted": self.recovery_halted,
        })
        temporary_path = MODEL_PATH.with_name(f".{MODEL_PATH.name}.tmp")
        backup_temporary_path = MODEL_BACKUP_PATH.with_name(f".{MODEL_BACKUP_PATH.name}.tmp")
        try:
            torch.save(payload, temporary_path)
            with temporary_path.open("rb+") as checkpoint_file:
                os.fsync(checkpoint_file.fileno())
            if MODEL_PATH.exists():
                shutil.copyfile(MODEL_PATH, backup_temporary_path)
                with backup_temporary_path.open("rb+") as backup_file:
                    os.fsync(backup_file.fileno())
                os.replace(backup_temporary_path, MODEL_BACKUP_PATH)
            os.replace(temporary_path, MODEL_PATH)
        finally:
            temporary_path.unlink(missing_ok=True)
            backup_temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _json_report_value(value: object) -> object:
        """Convert metrics to strict portable JSON without serializing model state."""
        if isinstance(value, dict):
            return {str(key): StrategicLeagueTrainer._json_report_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [StrategicLeagueTrainer._json_report_value(item) for item in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        return str(value)

    @staticmethod
    def _write_json_atomically(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8", newline="\n") as report_file:
                json.dump(payload, report_file, indent=2, sort_keys=True, allow_nan=False)
                report_file.write("\n")
                report_file.flush()
                os.fsync(report_file.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _checkpoint_metadata(path: Path) -> dict[str, object]:
        try:
            stat = path.stat()
        except OSError:
            return {"filename": path.name, "exists": False}
        return {
            "filename": path.name,
            "exists": True,
            "bytes": stat.st_size,
            "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @staticmethod
    def _diagnose_training_report(telemetry: dict, requested_hands: int, completed_hands: int, error: str | None) -> list[dict[str, object]]:
        """Attach bounded diagnostics rather than inferring strength from self-play alone."""
        findings: list[dict[str, object]] = []

        def add(severity: str, code: str, message: str, **metrics: object) -> None:
            findings.append({"severity": severity, "code": code, "message": message, "metrics": metrics})

        if error:
            add("error", "training_failed", "The run ended with an error; do not treat its model checkpoint as a completed evaluation.", error=error)
        elif completed_hands < requested_hands:
            add("warning", "training_incomplete", "The run ended before the requested hand count.", requested_hands=requested_hands, completed_hands=completed_hands)
        else:
            add("info", "training_completed", "The requested self-play run completed.", requested_hands=requested_hands, completed_hands=completed_hands)

        if bool(telemetry.get("smoke_test", False)):
            add("info", "quick_smoke_test", "This 5K smoke test used routine safety screening and skipped the final promotion-grade audit; it cannot promote a model.")

        if not bool(telemetry.get("gate_passed", False)):
            add("warning", "promotion_gate_held", "Champion promotion was held by the evaluation gates; a 50% self-play result is not sufficient evidence of strength.", challenger_status=telemetry.get("challenger_status"), promotion_confidence=telemetry.get("promotion_confidence"))
        adversarial_lcb = float(telemetry.get("adversarial_ci_floor_bb_per_100", 0.0))
        if adversarial_lcb < ADVERSARIAL_PROMOTION_LCB_FLOOR:
            add("warning", "adversarial_lcb_below_floor", "The worst paired fixed-adversary lower confidence bound is below the promotion floor.", adversarial_lcb_bb_per_100=adversarial_lcb, required_floor_bb_per_100=ADVERSARIAL_PROMOTION_LCB_FLOOR)
        restricted_br = float(telemetry.get("restricted_br_bb_per_100", 0.0))
        if restricted_br < 0.0:
            add("warning", "restricted_best_response_loss", "The restricted fixed-style best-response proxy remains negative; this is a vulnerability signal, not formal Hold'em exploitability.", restricted_br_bb_per_100=restricted_br)
        audit_proxy = float(telemetry.get("audit_exploitability_bb_per_100", 0.0))
        if audit_proxy < 0.0:
            add("warning", "audit_proxy_loss", "The independent fixed-style audit proxy is negative.", audit_proxy_bb_per_100=audit_proxy)
        if int(telemetry.get("evaluation_history_size", 0)) == 0:
            add("warning", "no_full_evaluation", "No scheduled full evaluation was recorded, so promotion and adversarial metrics may still be provisional.")
        if int(telemetry.get("snapshot_count", 0)) >= 2 and float(telemetry.get("snapshot_diversity", 0.0)) < 0.25:
            add("warning", "low_snapshot_diversity", "Frozen strategy snapshots are too similar to provide a useful average-strategy ensemble.", snapshot_count=telemetry.get("snapshot_count"), normalized_diversity=telemetry.get("snapshot_diversity"), min_distance=telemetry.get("snapshot_min_distance"), rejected_snapshots=telemetry.get("snapshot_rejections"))
        if bool(telemetry.get("ppo_kl_limited", False)):
            add(
                "warning",
                "ppo_kl_recovery",
                "The latest PPO update triggered a policy-safety guard and entered conservative recovery.",
                observed_hard_kl=telemetry.get("ppo_hard_kl"),
                kl_target=telemetry.get("ppo_kl_target"),
                recovery_updates=telemetry.get("ppo_recovery_updates"),
                rollback_phase=telemetry.get("ppo_rollback_phase", "none"),
            )
        if bool(telemetry.get("preflop_root_guarded", False)):
            add("warning", "preflop_root_drift_recovery", "The latest PPO update was rolled back because a fixed preflop root moved outside its trust region.", reason=telemetry.get("preflop_root_guard_reason"), root=telemetry.get("preflop_root_drift_root"), update_kl=telemetry.get("preflop_root_update_kl"), anchor_kl=telemetry.get("preflop_root_anchor_kl"))
        if int(telemetry.get("teacher_data_records", 0)) == 0:
            add("info", "no_local_teacher_data", "No optional validated human hand-history teacher data was used in this run.")
        if not ENABLE_APPROXIMATE_RESOLVER:
            add("info", "approximate_resolver_disabled", "Approximate resolving is disabled by default because it is not a safe full no-limit solver; zero resolver targets are expected in this mode.")
        return findings

    def write_training_report(self, telemetry: dict, requested_hands: int, completed_hands: int, error: str | None) -> Path:
        """Write a portable end-of-run report, deliberately excluding weights and replay records."""
        generated_at = datetime.now(timezone.utc)
        telemetry = {
            **telemetry,
            "preflop_3bet_teacher_effective_coverage": self.last_preflop_3bet_teacher_effective_coverage,
            "preflop_3bet_teacher_effective_weight": self.last_preflop_3bet_teacher_effective_weight,
            "specialist_archive_size": len(self.specialist_archive),
            "preflop_root_guarded": self.last_preflop_root_guarded,
            "preflop_root_guard_reason": self.last_preflop_root_guard_reason,
            "preflop_root_update_kl": self.last_preflop_root_update_kl,
            "preflop_root_anchor_kl": self.last_preflop_root_anchor_kl,
            "preflop_root_update_action_delta": self.last_preflop_root_update_action_delta,
            "preflop_root_anchor_action_delta": self.last_preflop_root_anchor_action_delta,
            "preflop_root_drift_root": self.last_preflop_root_drift_root,
        }
        with self._lock:
            population = [
                {
                    "id": str(member.get("id", "unknown")),
                    "updates": int(member.get("updates", 0)),
                    "score": float(member.get("score", 0.0)),
                    "bb_per_100": float(member.get("bb_per_100", 0.0)),
                    "adversarial_bb_per_100": float(member.get("adversarial_bb_per_100", 0.0)),
                    "preflop_worst_lcb_bb_per_100": float(member.get("preflop_worst_lcb_bb_per_100", 0.0)),
                    "preflop_allin_probability": float(member.get("preflop_allin_probability", 0.0)),
                    "quarantine_count": int(member.get("quarantine_count", 0)),
                    "safety_regressions": int(member.get("safety_regressions", 0)),
                    "recovery_cooldown_until": int(member.get("recovery_cooldown_until", 0)),
                    "last_quarantine_reason": str(member.get("last_quarantine_reason", "")),
                    "learning_rate_scale": float(member.get("lr_scale", 1.0)),
                    "entropy_scale": float(member.get("entropy_scale", 1.0)),
                }
                for member in self.population_members
            ]
            league = [{key: entry[key] for key in ("id", "kind", "elo", "games", "threat", "champion_score") if key in entry} for entry in self.league]
            exploiters = [{key: entry[key] for key in ("id", "kind", "elo", "games", "threat", "champion_score", "generation", "signature") if key in entry} for entry in self.exploiters]
            trainer_state = {
                "model_format": MODEL_VERSION,
                "trainer_version": self.version,
                "updates": self.updates,
                "trained_hands_total": self.trained_hands,
                "champion": {"id": self.champion_id, "elo": self.champion_elo},
                "recovery_anchor": {"updates": self.recovery_anchor_updates, "score": self.recovery_anchor_score, "source": self.recovery_anchor_source, "safe_audits": self.recovery_safe_audits, "protection_active": self._has_verified_recovery_anchor_locked(), "metrics": copy.deepcopy(self.recovery_anchor_metrics)},
                "recovery_baseline": {"verified": self.recovery_baseline_verified, "metrics": copy.deepcopy(self.recovery_baseline_metrics), "fold_collapse_rate": PREFLOP_ROOT_FOLD_COLLAPSE_RATE},
                "last_recovery_candidate": {"metrics": copy.deepcopy(self.last_recovery_candidate_metrics), "fresh_warmup_fold_collapse": self.last_fresh_warmup_fold_collapse, "checkpoint_restored_after_final_audit": self.last_final_audit_checkpoint_restored, "checkpoint_restore_reason": self.last_final_audit_restore_reason, "safety_halted": self.recovery_halted},
                "optimizer_safety": {"hard_kl_limit": self.ppo_kl_target * PPO_HARD_KL_MULTIPLIER, "observed_peak_kl": self.last_ppo_hard_kl, "last_update_reverted": self.last_ppo_update_reverted, "rollback_phase": self.last_ppo_rollback_phase, "post_step_retry_applied": self.last_ppo_post_step_retry_applied, "post_step_retry_accepted": self.last_ppo_post_step_retry_accepted, "post_step_retry_kl": self.last_ppo_post_step_retry_kl, "post_step_retry_scale": self.ppo_post_step_retry_scale, "root_backoff_applied": self.last_ppo_root_backoff_applied, "root_backoff_accepted": self.last_ppo_root_backoff_accepted, "root_backoff_scale": self.last_ppo_root_backoff_scale, "mixed_precision_enabled": self.mixed_precision_enabled, "amp_overflow_fallbacks": self.amp_overflow_fallbacks, "preflop_root_trust_region": {"guarded": self.last_preflop_root_guarded, "reason": self.last_preflop_root_guard_reason, "root": self.last_preflop_root_drift_root, "update_kl": self.last_preflop_root_update_kl, "anchor_kl": self.last_preflop_root_anchor_kl, "update_action_delta": self.last_preflop_root_update_action_delta, "anchor_action_delta": self.last_preflop_root_anchor_action_delta}, "run": {**self.run_ppo_safety, "mean_retry_kl": self.run_ppo_safety["retry_kl_sum"] / max(1, self.run_ppo_safety["retry_attempts"])}},
                "teacher_safety": {"all_in_min_confidence": PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE, "low_confidence_all_in_targets_suppressed": self.last_preflop_3bet_teacher_allin_suppressed, "multi_raise_all_in_vetoes": self.last_preflop_3bet_teacher_multi_raise_allin_vetoes},
                "teacher_confidence_gate": {"worlds_per_target": PREFLOP_3BET_TEACHER_WORLDS, "minimum_confidence": PREFLOP_3BET_TEACHER_MIN_CONFIDENCE, "effective_coverage": self.last_preflop_3bet_teacher_effective_coverage, "mean_effective_weight": self.last_preflop_3bet_teacher_effective_weight},
                "population": population,
                "specialist_archive": [{"id": str(entry.get("id", "unknown")), "updates": int(entry.get("updates", 0)), "resumes": int(entry.get("resumes", 0)), "focus_root": str(entry.get("focus_root", "pending")), "reason": str(entry.get("reason", "")), "metrics": copy.deepcopy(entry.get("metrics", {}))} for entry in self.specialist_archive],
                "league": league,
                "exploiters": exploiters,
                "strategy_snapshots": [{"id": str(snapshot.get("id", "unknown")), "updates": int(snapshot.get("updates", 0))} for snapshot in self.strategy_snapshots],
                "adversarial_style_bb_per_100": copy.deepcopy(self.adversarial_style_bb_per_100),
                "scenario_weakness": copy.deepcopy(self.scenario_weakness),
                "benchmarks": copy.deepcopy(self.benchmarks),
                "audit_benchmarks": copy.deepcopy(self.audit_benchmarks),
                "evaluation_history": copy.deepcopy(self.evaluation_history[-24:]),
                "adversarial_rollouts": {"rolling_fraction": self.last_adversarial_rollout_fraction, "run_fraction": self.run_adversarial_hands / max(1, self.run_rollout_hands), "run_hands": self.run_adversarial_hands, "total_hands": self.run_rollout_hands, "window_updates": len(self._rollout_diagnostic_window)},
                "tail_risk": {"rolling": {"loss_rate": self.last_tail_loss_rate, "mean_tail_loss_bb": self.last_tail_loss_bb, "mean_policy_weight": self.last_tail_policy_weight, "window_updates": len(self._tail_diagnostic_window)}, "run": {"loss_rate": self.run_tail_paths / max(1, self.run_adversarial_paths), "mean_tail_loss_bb": self.run_tail_loss_sum / max(1, self.run_tail_paths), "mean_policy_weight": self.run_tail_weight_sum / max(1, self.run_tail_paths), "adversarial_paths": self.run_adversarial_paths, "tail_paths": self.run_tail_paths}, "by_style": copy.deepcopy(self.last_tail_style_diagnostics)},
                "hard_spot_value_replay": {"records": self.last_hard_spot_memory_size, "loss": self.last_hard_spot_value_loss},
                "behavioral_audit": {"states": self.last_behavior_audit_states, "action_agreement": self.last_behavior_action_agreement, "action_change_rate": self.last_behavior_action_change_rate, "raise_fraction_delta": self.last_behavior_raise_fraction_delta},
                "preflop_sizing": {"open_cap_bb": PREFLOP_OPEN_RAISE_CAP_BB, "three_bet_cap_pot_multiplier": PREFLOP_THREE_BET_POT_CAP_MULTIPLIER, **copy.deepcopy(self.last_preflop_sizing_audit)},
                "preflop_scenarios": {"forced_rollout_fraction": self.last_preflop_root_fraction, "weakness": copy.deepcopy(self.preflop_root_weakness), "hands_per_root_style": self.last_preflop_scenario_audit_hands, "worst_lcb_bb_per_100": self.last_preflop_scenario_worst_lcb_bb_per_100, "worst_root": self.last_preflop_scenario_worst_root, "worst_style": self.last_preflop_scenario_worst_style, "all_in_calibration": {"regularizer_loss": self.last_preflop_allin_calibration_loss, "stability_loss": self.last_preflop_allin_stability_loss, "guarded_policy_probability": self.last_preflop_guarded_allin_probability, "target_probability": self.last_preflop_allin_target, "guarded_state_fraction": self.last_preflop_guarded_state_fraction, "audit_immediate_all_in_rate": self.last_preflop_immediate_allin_rate, "audit_immediate_target": self.last_preflop_immediate_allin_target, "audit_immediate_eligible_rate": self.last_preflop_immediate_eligible_rate, "promotion_root_lcb_floor_bb_per_100": PREFLOP_ROOT_PROMOTION_LCB_FLOOR}, "three_bet_teacher": {"loss": self.last_preflop_3bet_teacher_loss, "eligible_roots": self.last_preflop_3bet_teacher_eligible_roots, "sampled_roots": self.last_preflop_3bet_teacher_samples, "coverage": self.last_preflop_3bet_teacher_coverage, "mean_confidence": self.last_preflop_3bet_teacher_confidence, "raise_target_probability": self.last_preflop_3bet_teacher_raise_target, "raise_advantage_bb": self.last_preflop_3bet_teacher_raise_advantage_bb, "sampled_raise_rate": self.last_preflop_3bet_teacher_actual_raise_rate, "all_in_target_probability": self.last_preflop_3bet_teacher_allin_target, "sampled_all_in_rate": self.last_preflop_3bet_teacher_actual_allin_rate, "multi_raise_samples": self.last_preflop_3bet_teacher_multi_raise_samples, "multi_raise_all_in_target_probability": self.last_preflop_3bet_teacher_multi_raise_allin_target, "multi_raise_sampled_all_in_rate": self.last_preflop_3bet_teacher_multi_raise_actual_allin_rate, "multi_raise_all_in_vetoes": self.last_preflop_3bet_teacher_multi_raise_allin_vetoes, "facing_4bet": {"samples": self.last_preflop_3bet_teacher_facing_4bet_samples, "target_action_mix": copy.deepcopy(self.last_preflop_3bet_teacher_facing_4bet_target_actions), "actual_action_mix": copy.deepcopy(self.last_preflop_3bet_teacher_facing_4bet_actual_actions), "non_all_in_vetoes": self.last_preflop_3bet_teacher_facing_4bet_non_allin_vetoes}, "style_focus_share": PREFLOP_3BET_STYLE_FOCUS_SHARE}, "audit": copy.deepcopy(self.last_preflop_scenario_audit)},
                "preflop_teacher_by_root": {"last_update": copy.deepcopy(self.last_preflop_3bet_teacher_by_root), "run": {root: preflop_teacher_root_metrics(totals) for root, totals in self.run_preflop_teacher_by_root.items()}},
                "rollout_collector": {"inference_device": self.last_rollout_inference_device, "workers": self.last_adaptive_workers, "batch_hands": self.last_adaptive_batch_hands, "decisions_per_second": self.last_rollout_decisions_per_second},
                "robust_training": {"policy_weight_cap": self.last_robust_policy_weight, "specialist_min_updates": POPULATION_SPECIALIST_MIN_UPDATES, "exploiter_refresh_updates": EXPLOITER_REFRESH_UPDATES, "self_imitation_migrated": self.imitation_memory.self_play_migration_complete},
                "adversarial_audit": {"hands_per_style": self.last_adversarial_evaluation_hands, "confirmation_hands": self.last_adversarial_confirmation_hands, "final_audit_ran": self.last_final_audit_ran, "styles_bb_per_100": copy.deepcopy(self.adversarial_style_bb_per_100), "focus": self.last_adversarial_focus},
                "teacher_data": self.teacher_data_report.payload(),
            }
        settings = {
            "checkpoint_interval_hands": CHECKPOINT_INTERVAL_HANDS,
            "evaluation_interval_updates": EVALUATION_INTERVAL,
            "evaluation_workers": EVALUATION_WORKERS,
            "policy_execution": {"version": POLICY_EXECUTION_VERSION, "training": "sampled mixed action and sizing", "evaluation": "fixed-seed sampled action with deterministic mean sizing", "live": "sampled action with deterministic mean sizing", "counterfactual_search": "greedy bounded continuation"},
            "promotion_hands": {"minimum": PROMOTION_MIN_HANDS, "maximum": PROMOTION_MAX_HANDS},
            "holdout_hands": HOLDOUT_HANDS,
            "adversarial": {"training_styles": ADVERSARIAL_TRAINING_STYLES, "rollout_fraction": ADVERSARIAL_ROLLOUT_FRACTION, "focus_share": ADVERSARIAL_FOCUS_SHARE, "rotation_share": ADVERSARIAL_ROTATION_SHARE, "focus_count": ADVERSARIAL_FOCUS_COUNT, "restricted_screening_hands_per_style": ADVERSARIAL_EVALUATION_HANDS, "full_screening_hands_per_style": ADVERSARIAL_SCREENING_HANDS, "confirmation_hands_per_style": ADVERSARIAL_CONFIRMATION_HANDS, "promotion_lcb_floor_bb_per_100": ADVERSARIAL_PROMOTION_LCB_FLOOR},
            "tail_risk": {"large_loss_bb": LARGE_LOSS_BB, "maximum_adversarial_policy_weight": ADVERSARIAL_TAIL_WEIGHT, "hard_spot_value_memory_capacity": self.hard_spot_value_memory.capacity, "rolling_updates": ROLLING_DIAGNOSTIC_UPDATES},
            "preflop_sizing": {"unopened_normal_raise_cap_bb": PREFLOP_OPEN_RAISE_CAP_BB, "three_bet_normal_raise_cap_pot_multiplier": PREFLOP_THREE_BET_POT_CAP_MULTIPLIER, "audit_roots": PREFLOP_SIZING_AUDIT_HANDS, "all_in_action_is_uncapped": True, "all_in_calibration_weight": PREFLOP_ALLIN_CALIBRATION_WEIGHT, "all_in_calibration_committed_fraction": PREFLOP_ALLIN_COMMITTED_FRACTION, "all_in_calibration_target_max": PREFLOP_ALLIN_TARGET_MAX, "root_promotion_lcb_floor_bb_per_100": PREFLOP_ROOT_PROMOTION_LCB_FLOOR, "root_fold_collapse_rate": PREFLOP_ROOT_FOLD_COLLAPSE_RATE, "teacher_allin_min_confidence": PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE, "three_bet_teacher_roots": PREFLOP_3BET_TEACHER_ROOTS, "three_bet_teacher_multi_raise_roots": PREFLOP_3BET_TEACHER_MULTI_RAISE_ROOTS, "three_bet_teacher_multi_raise_min_roots": PREFLOP_3BET_TEACHER_MULTI_RAISE_MIN_ROOTS, "three_bet_teacher_facing_4bet_min_roots": PREFLOP_3BET_TEACHER_FACING_4BET_MIN_ROOTS, "teacher_focus_min_roots": PREFLOP_TEACHER_FOCUS_MIN_ROOTS, "focus_root_weight_multiplier": PREFLOP_FOCUS_ROOT_WEIGHT_MULTIPLIER, "three_bet_teacher_style_focus_share": PREFLOP_3BET_STYLE_FOCUS_SHARE, "three_bet_teacher_sample_probability": PREFLOP_3BET_TEACHER_SAMPLE_PROBABILITY, "three_bet_teacher_max_roots_per_rollout": PREFLOP_3BET_TEACHER_MAX_ROOTS, "three_bet_teacher_depth": PREFLOP_3BET_TEACHER_DEPTH, "three_bet_teacher_weight": PREFLOP_3BET_TEACHER_WEIGHT, "three_bet_teacher_temperature_bb": PREFLOP_3BET_TEACHER_TEMPERATURE_BB, "teacher_allin_contrastive_weight": PREFLOP_TEACHER_ALLIN_CONTRASTIVE_WEIGHT, "forced_root_fraction": PREFLOP_FORCED_ROOT_FRACTION, "scenario_audit_hands_per_root_style": PREFLOP_SCENARIO_AUDIT_HANDS, "final_audit_multiplier": PREFLOP_FINAL_AUDIT_MULTIPLIER, "roots": PREFLOP_FORCED_ROOTS},
            "robust_training": {"style_policy_weight": ROBUST_STYLE_POLICY_WEIGHT, "population_specialist_min_updates": POPULATION_SPECIALIST_MIN_UPDATES, "exploiter_refresh_updates": EXPLOITER_REFRESH_UPDATES},
            "ppo_safety": {"hard_kl_multiplier": PPO_HARD_KL_MULTIPLIER, "post_step_retry_initial_scale": PPO_POST_STEP_RETRY_SCALE, "minibatch_hands": PPO_MINIBATCH_HANDS, "minimum_final_rollout_hands": PPO_MIN_FINAL_ROLLOUT_HANDS, "dropout_disabled_for_policy_ratio": True, "fused_adamw_opt_in": PPO_USE_FUSED_ADAMW, "preflop_root_probe_hands": PREFLOP_ROOT_PROBE_HANDS, "preflop_root_update_kl_limit": PREFLOP_ROOT_UPDATE_KL_LIMIT, "preflop_root_anchor_kl_limit": PREFLOP_ROOT_ANCHOR_KL_LIMIT, "preflop_root_update_action_delta_limit": PREFLOP_ROOT_UPDATE_ACTION_DELTA_LIMIT, "preflop_root_anchor_action_delta_limit": PREFLOP_ROOT_ANCHOR_ACTION_DELTA_LIMIT},
            "rollout_collector": {"requested_inference_device": ROLLOUT_INFERENCE_DEVICE, "cuda_uses_one_worker": True, "cuda_evaluation_enabled": CUDA_EVALUATION_ENABLED},
            "snapshot_min_distance": SNAPSHOT_MIN_DISTANCE,
            "blueprint_promotion": {"confidence_floor": BLUEPRINT_PROMOTION_CONFIDENCE, "score_floor": BLUEPRINT_PROMOTION_FLOOR, "metric": "paired chip-EV quality", "neutral_bb_per_100_quality": 0.5},
            "ppo": {"maximum_epochs": PPO_MAX_EPOCHS, "hard_kl_multiplier": PPO_HARD_KL_MULTIPLIER, "recovery_updates": PPO_RECOVERY_UPDATES, "post_update_rollback": True, "range_loss_weight": PPO_RANGE_LOSS_WEIGHT},
            "recovery_anchor": {"regression_margin": RECOVERY_ANCHOR_REGRESSION_MARGIN, "cooldown_updates": RECOVERY_ANCHOR_COOLDOWN_UPDATES, "bootstrap_score": RECOVERY_ANCHOR_BOOTSTRAP_SCORE, "recovery_full_evaluation_interval": RECOVERY_FULL_EVALUATION_INTERVAL},
            "safety_features": {"approximate_resolver_enabled": ENABLE_APPROXIMATE_RESOLVER, "heuristic_oracle_enabled": ENABLE_HEURISTIC_ORACLE, "abstract_cfr_teacher_mode": ABSTRACT_CFR_TEACHER_MODE},
        }
        settings["preflop_sizing"]["three_bet_teacher_min_confidence"] = PREFLOP_3BET_TEACHER_MIN_CONFIDENCE
        settings["preflop_sizing"]["three_bet_teacher_worlds"] = PREFLOP_3BET_TEACHER_WORLDS
        settings["preflop_sizing"]["low_confidence_teacher_weight"] = 0.0
        settings["holdout_confirmation_hands"] = HOLDOUT_CONFIRMATION_HANDS
        settings["preflop_root_confirmation_hands"] = PREFLOP_ROOT_CONFIRMATION_HANDS
        settings["public_state_experts"] = PUBLIC_STATE_EXPERTS
        settings["preflop_sizing"]["all_in_stability_weight"] = PREFLOP_ALLIN_STABILITY_WEIGHT
        settings["preflop_sizing"]["all_in_ranking_weight"] = PREFLOP_ALLIN_RANKING_WEIGHT
        settings["preflop_sizing"]["all_in_ranking_margin"] = PREFLOP_ALLIN_RANKING_MARGIN
        settings["preflop_sizing"]["three_bet_teacher_allin_disadvantage_weight"] = PREFLOP_3BET_TEACHER_ALLIN_DISADVANTAGE_WEIGHT
        settings["preflop_sizing"]["teacher_facing_4bet_call_contrastive_weight"] = PREFLOP_TEACHER_FACING_4BET_CALL_CONTRASTIVE_WEIGHT
        settings["preflop_sizing"]["teacher_facing_4bet_weight_multiplier"] = PREFLOP_TEACHER_FACING_4BET_WEIGHT_MULTIPLIER
        settings["preflop_sizing"]["teacher_shallow_open_min_roots"] = PREFLOP_TEACHER_SHALLOW_OPEN_MIN_ROOTS
        settings["preflop_sizing"]["teacher_shallow_all_in_margin_weight"] = PREFLOP_TEACHER_SHALLOW_ALLIN_MARGIN_WEIGHT
        settings["preflop_sizing"]["teacher_action_margin"] = PREFLOP_TEACHER_ACTION_MARGIN
        report = {
            "schema_version": TRAINING_REPORT_SCHEMA_VERSION,
            "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
            "run": {"outcome": "failed" if error else "completed" if completed_hands >= requested_hands else "incomplete", "smoke_test": bool(telemetry.get("smoke_test", False)), "promotion_eligible": not bool(telemetry.get("smoke_test", False)), "requested_hands": requested_hands, "completed_hands": completed_hands, "error": error},
            "telemetry": copy.deepcopy(telemetry),
            "run_final_evaluation": {key: telemetry.get(key) for key in ("direct_bb_per_100", "promotion_ci_lower_bb_per_100", "adversarial_floor_bb_per_100", "adversarial_ci_floor_bb_per_100", "restricted_br_bb_per_100", "audit_exploitability_bb_per_100", "holdout_ci_floor_bb_per_100", "preflop_scenario_worst_lcb_bb_per_100", "preflop_scenario_worst_root", "preflop_scenario_worst_style", "evaluation_seconds", "final_audit_ran")},
            "diagnostics": self._diagnose_training_report(telemetry, requested_hands, completed_hands, error),
            "trainer": trainer_state,
            "settings": settings,
            "checkpoints": {"primary": self._checkpoint_metadata(MODEL_PATH), "backup": self._checkpoint_metadata(MODEL_BACKUP_PATH)},
            "notes": {"includes": ["final telemetry", "evaluation history", "benchmark and audit results", "tail-risk diagnostics", "behavioral audit", "preflop sizing and root audits", "range calibration", "population metadata", "configuration and checkpoint metadata"], "excludes": ["model weights", "optimizer tensors", "replay records", "private hand histories"]},
        }
        tail_risk = trainer_state["tail_risk"]
        run_tail_risk = tail_risk["run"]
        if float(run_tail_risk["loss_rate"]) >= 0.12:
            report["diagnostics"].append({"severity": "warning", "code": "adversarial_large_pot_losses", "message": "Large adversarial losses are common enough to dominate chip EV despite hand-score results.", "metrics": {"tail_loss_rate": run_tail_risk["loss_rate"], "mean_tail_loss_bb": run_tail_risk["mean_tail_loss_bb"]}})
        behavioral_audit = trainer_state["behavioral_audit"]
        if int(behavioral_audit["states"]) and float(behavioral_audit["action_agreement"]) >= 0.98:
            report["diagnostics"].append({"severity": "info", "code": "low_behavioral_change", "message": "Candidate and champion choose nearly identical greedy actions on the fixed behavioral audit; zero direct EV may reflect limited deployment-policy change.", "metrics": behavioral_audit})
        preflop_sizing = trainer_state["preflop_sizing"]
        preflop_scenarios = trainer_state["preflop_scenarios"]
        last_recovery_candidate = trainer_state.get("last_recovery_candidate", {})
        if bool(last_recovery_candidate.get("checkpoint_restored_after_final_audit", False)):
            report["diagnostics"].append({"severity": "warning", "code": "final_audit_checkpoint_restored", "message": "The final audit materially regressed from the verified recovery anchor, so the rejected candidate was restored before checkpointing.", "metrics": {"reason": last_recovery_candidate.get("checkpoint_restore_reason", ""), **last_recovery_candidate.get("metrics", {})}})
        if bool(last_recovery_candidate.get("fresh_warmup_fold_collapse", False)):
            report["diagnostics"].append({"severity": "warning", "code": "fresh_warmup_fold_collapse", "message": "The fresh model collapsed to first-decision folds before a verified recovery anchor existed. Training continued, but promotion remains held until fixed-root audits recover.", "metrics": last_recovery_candidate.get("metrics", {})})
        if int(preflop_scenarios["hands_per_root_style"]) and float(preflop_scenarios["worst_lcb_bb_per_100"]) < -40.0:
            report["diagnostics"].append({"severity": "warning", "code": "preflop_root_vulnerability", "message": "A fixed preflop response root has a materially negative lower confidence bound; the forced-root curriculum will prioritize it.", "metrics": {"worst_lcb_bb_per_100": preflop_scenarios["worst_lcb_bb_per_100"], "root": preflop_scenarios["worst_root"], "style": preflop_scenarios["worst_style"]}})
        calibration = preflop_scenarios.get("all_in_calibration", {})
        if int(preflop_scenarios["hands_per_root_style"]) and float(preflop_scenarios["worst_lcb_bb_per_100"]) < float(calibration.get("promotion_root_lcb_floor_bb_per_100", PREFLOP_ROOT_PROMOTION_LCB_FLOOR)):
            report["diagnostics"].append({"severity": "warning", "code": "preflop_root_promotion_hold", "message": "A forced preflop root is below the promotion lower-confidence floor, so the challenger remains held while its root defense is trained.", "metrics": {"worst_lcb_bb_per_100": preflop_scenarios["worst_lcb_bb_per_100"], "promotion_floor": calibration.get("promotion_root_lcb_floor_bb_per_100", PREFLOP_ROOT_PROMOTION_LCB_FLOOR), "root": preflop_scenarios["worst_root"], "style": preflop_scenarios["worst_style"]}})
        if float(calibration.get("guarded_state_fraction", 0.0)) >= 0.01 and float(calibration.get("guarded_policy_probability", 0.0)) > float(calibration.get("target_probability", 1.0)) + 0.08:
            report["diagnostics"].append({"severity": "warning", "code": "preflop_early_allin_pressure", "message": "The policy still assigns too much all-in probability to low-commitment preflop decisions; the soft calibration loss remains active without masking legal 4-/5-bet shoves.", "metrics": calibration})
        three_bet_teacher = preflop_scenarios.get("three_bet_teacher", {})
        if int(three_bet_teacher.get("eligible_roots", 0)) and float(three_bet_teacher.get("coverage", 0.0)) < 0.15:
            report["diagnostics"].append({"severity": "warning", "code": "preflop_3bet_teacher_undercovered", "message": "Too few eligible small-open roots received a matched 3-bet comparison; increase bounded teacher sampling before judging the policy response.", "metrics": three_bet_teacher})
        if int(three_bet_teacher.get("sampled_roots", 0)) and float(three_bet_teacher.get("raise_target_probability", 0.0)) >= 0.35 and float(three_bet_teacher.get("sampled_raise_rate", 0.0)) <= 0.05:
            report["diagnostics"].append({"severity": "warning", "code": "preflop_3bet_policy_lag", "message": "Matched root branches favor legal 3-bets, but the sampled policy is still not taking them; continue focused root training before adding a stronger constraint.", "metrics": three_bet_teacher})
        if int(preflop_sizing["roots"]) and float(preflop_sizing["p95_raise_bb"]) > float(preflop_sizing["open_cap_bb"]) + 0.05:
            report["diagnostics"].append({"severity": "warning", "code": "preflop_open_cap_mismatch", "message": "The preflop sizing audit observed a normal open beyond the configured cap; check execution-path parity.", "metrics": preflop_sizing})
        if int(preflop_sizing["three_bet_roots"]) and float(preflop_sizing["three_bet_over_cap_rate"]) > 0.0:
            report["diagnostics"].append({"severity": "warning", "code": "preflop_three_bet_cap_mismatch", "message": "The preflop sizing audit observed a feasible normal 3-bet beyond the configured pot cap; check execution-path parity.", "metrics": preflop_sizing})
        elif int(preflop_sizing["roots"]) and float(preflop_sizing["all_in_rate"]) >= 0.08:
            report["diagnostics"].append({"severity": "warning", "code": "preflop_all_in_frequency", "message": "The independent all-in action remains unusually frequent on blind-only preflop roots.", "metrics": preflop_sizing})
        safe_report = self._json_report_value(report)
        timestamp = generated_at.strftime("%Y%m%dT%H%M%S%fZ")
        report_path = TRAINING_REPORT_DIRECTORY / f"training-{timestamp}-hands-{completed_hands}-updates-{trainer_state['updates']}.json"
        self._write_json_atomically(report_path, safe_report)
        self._write_json_atomically(TRAINING_REPORT_DIRECTORY / "latest.json", safe_report)
        return report_path

    def reset_rollout_worker_cache(self) -> None:
        with self._lock:
            self._rollout_cached_opponent_revisions.clear()

    def note_rollout_worker_cache(self, revisions: list[str]) -> None:
        with self._lock:
            self._rollout_cached_opponent_revisions.update(revision for revision in revisions if revision)

    def rollout_snapshot(self, best_response_lane: bool = False, use_cached_opponents: bool = False) -> tuple[dict[str, Tensor], dict[str, Tensor], list[dict], dict[str, float], dict[str, float], dict, dict]:
        with self._lock:
            entries = [self._champion_entry_locked()] if best_response_lane else self._roster_locked()
            focus_styles = set(self._focused_adversarial_styles_locked())
            style_pool = ADVERSARIAL_TRAINING_STYLES if best_response_lane else dict.fromkeys((*BENCHMARK_STYLES, *ADVERSARIAL_TRAINING_STYLES))
            entries.extend({"id": f"style-{style}", "kind": "style", "style": style, "adversarial": style in ADVERSARIAL_TRAINING_STYLES, "focus": style in focus_styles} for style in style_pool)
            if not best_response_lane:
                for entry in entries:
                    if entry.get("kind") == "exploiter" and float(entry.get("threat", 0.0)) >= 0.05:
                        entry["focus"] = True
            weights = self._adaptive_opponent_weights_locked(entries)
            for entry in entries:
                entry["weight"] = weights[str(entry["id"])]
                focused = bool(entry.get("focus", False))
                adversarial = bool(entry.get("adversarial", False)) or entry.get("kind") == "exploiter"
                entry["robust_weight"] = 1.0 + ROBUST_STYLE_POLICY_WEIGHT * (1.0 if focused else 0.50 if adversarial else 0.0)
                revision = str(entry.get("state_revision", ""))
                if use_cached_opponents and entry.get("rollout_cacheable") and revision in self._rollout_cached_opponent_revisions:
                    entry["state"] = None
            self.last_robust_policy_weight = max((float(entry["robust_weight"]) for entry in entries), default=1.0)
            return clone_state(self.model), clone_state(self.target_model), entries, dict(self.scenario_weakness), dict(self.preflop_root_weakness), self.abstract_oracle.snapshot(), self.abstract_cfr_solver.snapshot()

    def note_rollouts(self, hands: int) -> None:
        with self._lock:
            self.trained_hands += hands

    def note_scenarios(self, scenario_counts: list[dict[str, int]]) -> None:
        total = sum(sum(counts.values()) for counts in scenario_counts)
        targeted = sum(sum(count for profile, count in counts.items() if profile != "balanced") for counts in scenario_counts)
        with self._lock:
            self.last_scenario_coverage = targeted / max(1, total)

    def note_preflop_roots(self, root_counts: list[dict[str, int]]) -> None:
        total = sum(sum(counts.values()) for counts in root_counts)
        forced = sum(sum(count for root, count in counts.items() if root != "blind") for counts in root_counts)
        with self._lock:
            self.last_preflop_root_fraction = forced / max(1, total)

    def note_adversarial_rollouts(self, adversarial_hands: int, total_hands: int) -> None:
        with self._lock:
            adversarial_hands = max(0, int(adversarial_hands))
            total_hands = max(0, int(total_hands))
            self.run_adversarial_hands += adversarial_hands
            self.run_rollout_hands += total_hands
            self._rollout_diagnostic_window.append((adversarial_hands, total_hands))
            self._rollout_diagnostic_window = self._rollout_diagnostic_window[-ROLLING_DIAGNOSTIC_UPDATES:]
            window_adversarial = sum(adversarial for adversarial, _ in self._rollout_diagnostic_window)
            window_total = sum(total for _, total in self._rollout_diagnostic_window)
            self.last_adversarial_rollout_fraction = window_adversarial / max(1, window_total)

    def note_compiled_transitions(self, compiled_actions: int, total_actions: int) -> None:
        with self._lock:
            self.last_compiled_transition_fraction = compiled_actions / max(1, total_actions)

    def note_paired_deals(self, paired_hands: int, total_hands: int) -> None:
        """Report how much collection used same-deal, swapped-seat pairs."""
        with self._lock:
            self.last_paired_deal_coverage = paired_hands / max(1, total_hands)

    def import_teacher_data(self, filename: str) -> dict[str, int | str]:
        """Seed safe auxiliary memories from a user-provided local JSONL export."""
        targets, report = load_teacher_actions(filename)
        likelihood_records = [
            ActionLikelihoodRecord(target.context, target.history or [target.context], target.range_class, target.action)
            for target in targets
            if target.context is not None and target.range_class is not None
        ]
        imitation_records = [
            ImitationRecord(target.observation, target.mask, target.action, target.return_value, min(4.0, 0.75 + max(0.0, target.return_value)))
            for target in targets
            if target.return_value >= 0.35
        ]
        with self._lock:
            self.action_likelihood_memory.extend(likelihood_records, self._rng)
            self.imitation_memory.extend_records(imitation_records, self._rng)
            self.teacher_data_report = report
            self.teacher_data_records += len(targets)
        return report.payload()

    def _sequence_tensors(self, paths: list[HandTrajectory]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = len(paths)
        length = max(len(path.actions) for path in paths)
        observations = torch.zeros((batch_size, length, OBSERVATION_SIZE), dtype=torch.float32)
        masks = torch.zeros((batch_size, length, ACTION_COUNT), dtype=torch.bool)
        actions = torch.zeros((batch_size, length), dtype=torch.long)
        old_log_probs = torch.zeros((batch_size, length), dtype=torch.float32)
        returns = torch.zeros((batch_size, length), dtype=torch.float32)
        advantages = torch.zeros((batch_size, length), dtype=torch.float32)
        range_labels = torch.full((batch_size, length), -1, dtype=torch.long)
        raise_fractions = torch.full((batch_size, length), 0.5, dtype=torch.float32)
        raise_active = torch.zeros((batch_size, length), dtype=torch.bool)
        valid = torch.zeros((batch_size, length), dtype=torch.bool)
        tail_weights = torch.ones((batch_size, length), dtype=torch.float32)
        all_in_probability_targets = torch.ones((batch_size, length), dtype=torch.float32)
        all_in_calibration_active = torch.zeros((batch_size, length), dtype=torch.bool)
        preflop_3bet_teacher_targets = torch.zeros((batch_size, length, ACTION_COUNT), dtype=torch.float32)
        preflop_3bet_teacher_confidences = torch.zeros((batch_size, length), dtype=torch.float32)
        preflop_3bet_teacher_eligible = torch.zeros((batch_size, length), dtype=torch.bool)
        preflop_3bet_teacher_raise_advantages = torch.zeros((batch_size, length), dtype=torch.float32)
        preflop_teacher_root_codes = torch.full((batch_size, length), PREFLOP_TEACHER_UNKNOWN_ROOT_CODE, dtype=torch.long)
        for row, path in enumerate(paths):
            path_advantages = trajectory_advantages(path)
            path_returns = [advantage + value for advantage, value in zip(path_advantages, path.values)]
            current_length = len(path.actions)
            observations[row, :current_length] = torch.tensor(path.observations)
            masks[row, :current_length] = torch.tensor(path.masks)
            actions[row, :current_length] = torch.tensor(path.actions)
            old_log_probs[row, :current_length] = torch.tensor(path.log_probs)
            returns[row, :current_length] = torch.tensor(path_returns)
            advantages[row, :current_length] = torch.tensor(path_advantages)
            range_labels[row, :current_length] = path.range_label
            raise_fractions[row, :current_length] = torch.tensor(path.raise_fractions)
            raise_active[row, :current_length] = torch.tensor([action in RAISE_ACTIONS for action in path.actions], dtype=torch.bool)
            valid[row, :current_length] = True
            robust_weight = min(1.0 + ROBUST_STYLE_POLICY_WEIGHT, max(1.0, float(path.robust_weight)))
            tail_credit = adversarial_tail_credit_weights(
                base_weight=robust_weight,
                tail_weight=adversarial_tail_policy_weight(path.reward) if path.adversarial else 1.0,
                reward_bb=path.reward,
                large_loss_bb=LARGE_LOSS_BB,
                advantages=path_advantages,
                streets=path.streets,
                masks=path.masks,
            )
            tail_weights[row, :current_length] = torch.tensor(tail_credit, dtype=torch.float32)
            if len(path.all_in_probability_targets) == current_length:
                all_in_probability_targets[row, :current_length] = torch.tensor(path.all_in_probability_targets, dtype=torch.float32)
            if len(path.all_in_calibration_active) == current_length:
                all_in_calibration_active[row, :current_length] = torch.tensor(path.all_in_calibration_active, dtype=torch.bool)
            if len(path.preflop_3bet_teacher_targets) == current_length:
                preflop_3bet_teacher_targets[row, :current_length] = torch.tensor(path.preflop_3bet_teacher_targets, dtype=torch.float32)
            if len(path.preflop_3bet_teacher_confidences) == current_length:
                preflop_3bet_teacher_confidences[row, :current_length] = torch.tensor(path.preflop_3bet_teacher_confidences, dtype=torch.float32)
            if len(path.preflop_3bet_teacher_eligible) == current_length:
                preflop_3bet_teacher_eligible[row, :current_length] = torch.tensor(path.preflop_3bet_teacher_eligible, dtype=torch.bool)
            if len(path.preflop_3bet_teacher_raise_advantages) == current_length:
                preflop_3bet_teacher_raise_advantages[row, :current_length] = torch.tensor(path.preflop_3bet_teacher_raise_advantages, dtype=torch.float32)
            if len(path.preflop_teacher_root_codes) == current_length:
                preflop_teacher_root_codes[row, :current_length] = torch.tensor(path.preflop_teacher_root_codes, dtype=torch.long)
        valid_advantages = advantages[valid]
        advantages[valid] = (valid_advantages - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-8)
        return observations, masks, actions, old_log_probs, returns, advantages, range_labels, raise_fractions, raise_active, valid, tail_weights, all_in_probability_targets, all_in_calibration_active, preflop_3bet_teacher_targets, preflop_3bet_teacher_confidences, preflop_3bet_teacher_eligible, preflop_3bet_teacher_raise_advantages, preflop_teacher_root_codes

    def ppo_update(self, paths: list[HandTrajectory]) -> dict:
        if not paths:
            return {"policy_loss": 0.0, "value_loss": 0.0, "distributional_value_loss": 0.0, "ensemble_disagreement": 0.0, "entropy": 0.0, "kl_divergence": 0.0, "range_loss": 0.0, "range_accuracy": 0.0, "range_brier": 0.0, "range_ece": 0.0, "range_coarse_accuracy": 0.0, "range_coarse_brier": 0.0, "ppo_clip_fraction": 0.0, "ppo_epochs": 0.0, "ppo_learning_rate": self.ppo_learning_rate, "preflop_allin_calibration_loss": 0.0, "preflop_allin_stability_loss": 0.0, "preflop_guarded_allin_probability": 0.0, "preflop_allin_target": 0.0, "preflop_guarded_state_fraction": 0.0, "preflop_3bet_teacher_loss": 0.0, "preflop_3bet_teacher_coverage": 0.0, "preflop_3bet_teacher_samples": 0.0, "ppo_tensor_preparation_seconds": 0.0, "ppo_transfer_seconds": 0.0, "ppo_compute_seconds": 0.0}
        tensor_preparation_started = perf_counter()
        observations, masks, actions, old_log_probs, returns, advantages, range_labels, raise_fractions, raise_active, valid, tail_weights, all_in_probability_targets, all_in_calibration_active, preflop_3bet_teacher_targets, preflop_3bet_teacher_confidences, preflop_3bet_teacher_eligible, preflop_3bet_teacher_raise_advantages, preflop_teacher_root_codes = self._sequence_tensors(paths)
        tensor_preparation_seconds = perf_counter() - tensor_preparation_started
        transfer_started = perf_counter()
        observations, masks, actions, old_log_probs, returns, advantages, range_labels, raise_fractions, raise_active, valid, tail_weights, all_in_probability_targets, all_in_calibration_active, preflop_3bet_teacher_targets, preflop_3bet_teacher_confidences, preflop_3bet_teacher_eligible, preflop_3bet_teacher_raise_advantages, preflop_teacher_root_codes = (
            self._to_training_device(tensor)
            for tensor in (observations, masks, actions, old_log_probs, returns, advantages, range_labels, raise_fractions, raise_active, valid, tail_weights, all_in_probability_targets, all_in_calibration_active, preflop_3bet_teacher_targets, preflop_3bet_teacher_confidences, preflop_3bet_teacher_eligible, preflop_3bet_teacher_raise_advantages, preflop_teacher_root_codes)
        )
        transfer_seconds = perf_counter() - transfer_started
        ppo_compute_started = perf_counter()
        root_probe_stage = self.curriculum_unlocked_stage
        pre_update_root_signature = preflop_root_policy_signature(self.model, root_probe_stage)
        anchor_root_signature = self._recovery_anchor_policy_signature(root_probe_stage) if self._has_verified_recovery_anchor_locked() else {}
        policy_losses: list[float] = []
        value_losses: list[float] = []
        distributional_value_losses: list[float] = []
        entropies: list[float] = []
        kl_values: list[float] = []
        range_losses: list[float] = []
        range_accuracies: list[float] = []
        range_briers: list[float] = []
        range_eces: list[float] = []
        range_coarse_accuracies: list[float] = []
        range_coarse_briers: list[float] = []
        clip_fractions: list[float] = []
        ensemble_disagreements: list[float] = []
        all_in_calibration_losses: list[float] = []
        all_in_stability_losses: list[float] = []
        guarded_all_in_probabilities: list[float] = []
        all_in_targets: list[float] = []
        preflop_3bet_teacher_losses: list[float] = []
        preflop_teacher_allin_contrast_losses: list[float] = []
        preflop_teacher_facing_4bet_call_contrast_losses: list[float] = []
        preflop_teacher_shallow_allin_margin_losses: list[float] = []
        preflop_teacher_allin_suppressed: list[int] = []
        epochs_run = 0
        batch_size = observations.size(0)
        mini_batch_size = ppo_minibatch_size(batch_size, self.runtime.cuda_enabled)
        for group in self.optimizer.param_groups:
            group["lr"] = self.ppo_learning_rate
        # Rollout log-probabilities come from a deterministic policy. cuDNN
        # requires recurrent modules to remain in train mode for backward, so
        # disable dropout explicitly instead of switching the GRU to eval mode.
        prepare_deterministic_ppo_policy(self.model)
        stop_early = False
        kl_limited = False
        epoch_budget = PPO_RECOVERY_EPOCHS if self.ppo_recovery_updates > 0 else PPO_MAX_EPOCHS
        self.last_ppo_epoch_budget = epoch_budget
        hard_kl_limit = self.ppo_kl_target * PPO_HARD_KL_MULTIPLIER
        pre_update_state = clone_state(self.model)
        pre_update_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        pre_update_scaler_state = copy.deepcopy(self.grad_scaler.state_dict())
        update_reverted = False
        rollback_phase = "none"
        post_step_retry_applied = False
        post_step_retry_accepted = False
        post_step_retry_kl = 0.0
        root_backoff_applied = False
        root_backoff_accepted = False
        root_backoff_scale = 0.0
        pre_step_guarded = False
        post_step_guarded = False

        def restore_pre_update() -> None:
            """Restore model, optimizer, and scaler before a guarded PPO pass."""
            self.model.load_state_dict(pre_update_state)
            self.optimizer.load_state_dict(pre_update_optimizer_state)
            self._move_optimizer_state_to_device()
            self.grad_scaler.load_state_dict(pre_update_scaler_state)

        def rollback_ppo_update(phase: str) -> None:
            """Restore the complete PPO update, including earlier mini-batches."""
            nonlocal update_reverted, rollback_phase
            restore_pre_update()
            self.optimizer.zero_grad(set_to_none=True)
            update_reverted = True
            rollback_phase = phase

        for _ in range(epoch_budget):
            row_order = torch.randperm(batch_size, device=observations.device)
            for start in range(0, batch_size, mini_batch_size):
                rows = row_order[start:start + mini_batch_size]
                observation_batch = observations[rows]
                mask_batch = masks[rows]
                action_batch = actions[rows]
                old_log_prob_batch = old_log_probs[rows]
                return_batch = returns[rows]
                advantage_batch = advantages[rows]
                range_label_batch = range_labels[rows]
                raise_fraction_batch = raise_fractions[rows]
                raise_active_batch = raise_active[rows]
                valid_batch = valid[rows]
                tail_weight_batch = tail_weights[rows]
                all_in_target_batch = all_in_probability_targets[rows]
                all_in_calibration_active_batch = all_in_calibration_active[rows]
                preflop_3bet_target_batch = preflop_3bet_teacher_targets[rows]
                preflop_3bet_confidence_batch = preflop_3bet_teacher_confidences[rows]
                preflop_3bet_eligible_batch = preflop_3bet_teacher_eligible[rows]
                preflop_teacher_root_code_batch = preflop_teacher_root_codes[rows]
                with self._autocast():
                    logits, _, values, range_logits, _, value_distribution_logits, raise_shapes, _ = self.model(observation_batch, padding_mask=~valid_batch)
                    policy = masked_distribution(logits, mask_batch)
                    sizing = raise_distribution(raise_shapes, action_batch)
                    continuous_log_probs = sizing.log_prob(raise_fraction_batch.clamp(0.005, 0.995))
                    new_log_probs = policy.log_prob(action_batch) + torch.where(raise_active_batch, continuous_log_probs, torch.zeros_like(continuous_log_probs))
                    valid_ratio = (new_log_probs - old_log_prob_batch)[valid_batch].exp()
                    valid_advantages = advantage_batch[valid_batch]
                    unclipped = valid_ratio * valid_advantages
                    clipped_ratio = valid_ratio.clamp(1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon)
                    clipped = clipped_ratio * valid_advantages
                    valid_tail_weights = tail_weight_batch[valid_batch]
                    policy_loss = -(torch.minimum(unclipped, clipped) * valid_tail_weights).sum() / valid_tail_weights.sum().clamp_min(1.0)
                    value_targets = (return_batch[valid_batch] / VALUE_RETURN_SCALE_BB).unsqueeze(-1)
                    value_bootstrap = (torch.rand_like(values[valid_batch]) > 0.20).float()
                    value_loss = ((values[valid_batch] - value_targets).square() * value_bootstrap).sum() / value_bootstrap.sum().clamp_min(1.0)
                    distribution_targets = value_support_bins(return_batch[valid_batch])
                    valid_distributions = value_distribution_logits[valid_batch]
                    repeated_targets = distribution_targets.unsqueeze(-1).expand(-1, VALUE_ENSEMBLE_SIZE).reshape(-1)
                    distribution_error = nn.functional.cross_entropy(valid_distributions.reshape(-1, VALUE_BINS), repeated_targets, reduction="none").view(-1, VALUE_ENSEMBLE_SIZE)
                    distribution_bootstrap = (torch.rand_like(distribution_error) > 0.20).float()
                    distributional_value_loss = (distribution_error * distribution_bootstrap).sum() / distribution_bootstrap.sum().clamp_min(1.0)
                    range_valid = valid_batch & (range_label_batch >= 0)
                    if range_valid.any():
                        exact_range_loss = nn.functional.cross_entropy(range_logits[range_valid], range_label_batch[range_valid])
                        range_probabilities = torch.softmax(range_logits[range_valid], dim=-1)
                        range_predictions = range_probabilities.argmax(dim=-1)
                        range_accuracy = (range_predictions == range_label_batch[range_valid]).float().mean()
                        targets = nn.functional.one_hot(range_label_batch[range_valid], num_classes=RANGE_BUCKETS).float()
                        range_brier = (range_probabilities - targets).square().sum(dim=-1).mean()
                        coarse_labels = self._range_coarse_index[range_label_batch[range_valid]]
                        coarse_probabilities = range_coarse_probabilities(range_probabilities, self._range_coarse_index)
                        coarse_loss = nn.functional.nll_loss(coarse_probabilities.clamp_min(1e-8).log(), coarse_labels)
                        coarse_predictions = coarse_probabilities.argmax(dim=-1)
                        range_coarse_accuracy = (coarse_predictions == coarse_labels).float().mean()
                        coarse_targets = nn.functional.one_hot(coarse_labels, num_classes=RANGE_COARSE_BUCKETS).float()
                        range_coarse_brier = (coarse_probabilities - coarse_targets).square().sum(dim=-1).mean()
                        range_loss = hierarchical_range_objective(
                            exact_range_loss,
                            coarse_loss,
                            exact_buckets=RANGE_BUCKETS,
                            coarse_buckets=RANGE_COARSE_BUCKETS,
                        )
                        confidences = range_probabilities.max(dim=-1).values
                        correct = (range_predictions == range_label_batch[range_valid]).float()
                        range_ece = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        for lower in torch.linspace(0.0, 0.8, 5, device=self.runtime.device):
                            in_bin = (confidences >= lower) & (confidences < lower + 0.2)
                            if in_bin.any():
                                range_ece += in_bin.float().mean() * (correct[in_bin].mean() - confidences[in_bin].mean()).abs()
                    else:
                        range_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        range_accuracy = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        range_brier = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        range_ece = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        range_coarse_accuracy = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        range_coarse_brier = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                    teacher_sampled = valid_batch & (preflop_3bet_target_batch.sum(dim=-1) > 0)
                    teacher_allin_supported = torch.zeros_like(valid_batch)
                    if teacher_sampled.any():
                        sampled_targets = preflop_3bet_target_batch[teacher_sampled]
                        sampled_confidences = preflop_3bet_confidence_batch[teacher_sampled]
                        # Do not let the generic early-shove ceiling fight a
                        # high-confidence matched branch comparison that supports the shove.
                        teacher_allin_supported[teacher_sampled] = (sampled_targets[..., 3] >= sampled_targets[..., :3].max(dim=-1).values - 0.08) & (sampled_confidences >= PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE)
                    calibration_valid = valid_batch & all_in_calibration_active_batch & ~teacher_allin_supported
                    if calibration_valid.any():
                        guarded_all_in_probability = policy.probs[..., 3][calibration_valid]
                        guarded_all_in_target = all_in_target_batch[calibration_valid]
                        all_in_calibration_loss = nn.functional.relu(guarded_all_in_probability - guarded_all_in_target).square().mean()
                        all_in_tail_risk_loss = tail_all_in_risk_loss(
                            guarded_all_in_probability,
                            guarded_all_in_target,
                            tail_weight_batch[calibration_valid],
                            baseline_weight=1.0 + ROBUST_STYLE_POLICY_WEIGHT,
                        )
                        guarded_logits = logits[calibration_valid].masked_fill(~mask_batch[calibration_valid], torch.finfo(logits.dtype).min)
                        non_all_in_mask = mask_batch[calibration_valid].clone()
                        non_all_in_mask[..., 3] = False
                        guarded_all_in_log_odds = guarded_logits[..., 3] - torch.logsumexp(guarded_logits.masked_fill(~non_all_in_mask, torch.finfo(logits.dtype).min), dim=-1)
                        all_in_stability_loss = nn.functional.binary_cross_entropy_with_logits(guarded_all_in_log_odds, guarded_all_in_target)
                        all_in_ranking_loss = preflop_all_in_ranking_loss(
                            logits[calibration_valid],
                            mask_batch[calibration_valid],
                            PREFLOP_ALLIN_RANKING_MARGIN,
                        )
                        guarded_all_in_probability_mean = guarded_all_in_probability.mean()
                        guarded_all_in_target_mean = guarded_all_in_target.mean()
                    else:
                        all_in_calibration_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        all_in_tail_risk_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        all_in_stability_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        all_in_ranking_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        guarded_all_in_probability_mean = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        guarded_all_in_target_mean = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                    if teacher_sampled.any():
                        teacher_log_probability = policy.probs.clamp_min(1e-8).log()
                        teacher_targets = preflop_3bet_target_batch[teacher_sampled]
                        teacher_confidences = preflop_3bet_confidence_batch[teacher_sampled]
                        teacher_effective_confidences = preflop_teacher_confidence_weight(teacher_confidences)
                        facing_4bet_samples = preflop_teacher_root_code_batch[teacher_sampled] == PREFLOP_TEACHER_ROOT_CODES["facing_4bet"]
                        facing_4bet_teacher_weights = torch.where(
                            facing_4bet_samples,
                            torch.full_like(teacher_confidences, PREFLOP_TEACHER_FACING_4BET_WEIGHT_MULTIPLIER),
                            torch.ones_like(teacher_confidences),
                        )
                        effective_teacher_targets = teacher_targets.clone()
                        low_confidence_allin = (teacher_targets[..., 3] > 0.0) & (teacher_confidences < PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE)
                        if low_confidence_allin.any():
                            normal_targets = effective_teacher_targets[..., :3]
                            normal_mass = normal_targets.sum(dim=-1, keepdim=True)
                            normal_legal = mask_batch[..., :3][teacher_sampled].float()
                            fallback = normal_legal / normal_legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
                            adjusted_normal_targets = torch.where(normal_mass > 1e-6, normal_targets / normal_mass.clamp_min(1e-6), fallback)
                            effective_teacher_targets[low_confidence_allin, :3] = adjusted_normal_targets[low_confidence_allin]
                            effective_teacher_targets[low_confidence_allin, 3] = 0.0
                        sampled_teacher_cross_entropy = -(effective_teacher_targets * teacher_log_probability[teacher_sampled]).sum(dim=-1)
                        best_teacher_probability = effective_teacher_targets.max(dim=-1).values
                        all_in_disadvantage = (best_teacher_probability - effective_teacher_targets[..., 3]).clamp_min(0.0)
                        all_in_available = mask_batch[..., 3][teacher_sampled].float()
                        all_in_bonus = PREFLOP_3BET_TEACHER_ALLIN_DISADVANTAGE_WEIGHT * all_in_disadvantage * teacher_effective_confidences * all_in_available
                        teacher_weights = teacher_effective_confidences * (1.0 + all_in_bonus) * facing_4bet_teacher_weights
                        preflop_3bet_teacher_loss = (sampled_teacher_cross_entropy * teacher_weights).sum() / teacher_weights.sum().clamp_min(1e-6)
                        teacher_logits = logits[teacher_sampled].masked_fill(~mask_batch[teacher_sampled], torch.finfo(logits.dtype).min)
                        non_all_in_logits = teacher_logits[..., :3]
                        all_in_log_odds = teacher_logits[..., 3] - torch.logsumexp(non_all_in_logits, dim=-1)
                        best_normal_target = effective_teacher_targets[..., :3].max(dim=-1).values
                        all_in_target = effective_teacher_targets[..., 3]
                        all_in_is_inferior = (best_normal_target - all_in_target).clamp_min(0.0)
                        contrast_weights = all_in_is_inferior * teacher_effective_confidences * all_in_available
                        # Only push the shove logit down when the same deal and
                        # continuation make a normal legal action preferable.
                        preflop_teacher_allin_contrast_loss = (nn.functional.softplus(all_in_log_odds) * contrast_weights).sum() / contrast_weights.sum().clamp_min(1e-6)
                        # A 4-bet root is already too committed for the generic
                        # early-shove guard. Require its call logit to sit below
                        # teacher-supported legal alternatives by a small margin.
                        call_available = mask_batch[..., 1][teacher_sampled].float()
                        non_call_targets = effective_teacher_targets.clone()
                        non_call_targets[..., 1] = 0.0
                        supported_non_call_logits = teacher_logits.masked_fill(non_call_targets <= 0.0, torch.finfo(teacher_logits.dtype).min)
                        supported_non_call_logits = supported_non_call_logits + non_call_targets.clamp_min(1e-8).log()
                        supported_non_call = torch.logsumexp(supported_non_call_logits, dim=-1)
                        best_non_call_target = non_call_targets.max(dim=-1).values
                        call_is_inferior = (best_non_call_target - effective_teacher_targets[..., 1]).clamp_min(0.0)
                        high_confidence = (teacher_confidences >= PREFLOP_3BET_TEACHER_ALLIN_MIN_CONFIDENCE).float()
                        facing_4bet_call_weights = call_is_inferior * teacher_effective_confidences * high_confidence * call_available * facing_4bet_samples.float()
                        call_margin = teacher_logits[..., 1] - supported_non_call + PREFLOP_TEACHER_ACTION_MARGIN
                        preflop_teacher_facing_4bet_call_contrast_loss = (nn.functional.softplus(call_margin) * facing_4bet_call_weights).sum() / facing_4bet_call_weights.sum().clamp_min(1e-6)
                        shallow_root_codes = torch.tensor([PREFLOP_TEACHER_ROOT_CODES[root] for root in PREFLOP_TEACHER_SHALLOW_OPEN_ROOTS], device=self.runtime.device)
                        shallow_open_samples = (preflop_teacher_root_code_batch[teacher_sampled].unsqueeze(-1) == shallow_root_codes).any(dim=-1)
                        normal_targets = effective_teacher_targets[..., :3]
                        supported_normal_logits = teacher_logits[..., :3].masked_fill(normal_targets <= 0.0, torch.finfo(teacher_logits.dtype).min)
                        supported_normal_logits = supported_normal_logits + normal_targets.clamp_min(1e-8).log()
                        supported_normal = torch.logsumexp(supported_normal_logits, dim=-1)
                        best_normal_target = normal_targets.max(dim=-1).values
                        all_in_is_inferior = (best_normal_target - effective_teacher_targets[..., 3]).clamp_min(0.0)
                        shallow_allin_weights = all_in_is_inferior * teacher_effective_confidences * high_confidence * all_in_available * shallow_open_samples.float()
                        shallow_allin_margin = teacher_logits[..., 3] - supported_normal + PREFLOP_TEACHER_ACTION_MARGIN
                        preflop_teacher_shallow_allin_margin_loss = (nn.functional.softplus(shallow_allin_margin) * shallow_allin_weights).sum() / shallow_allin_weights.sum().clamp_min(1e-6)
                        fold_available = mask_batch[..., 0][teacher_sampled].float()
                        nonfold_preference = (teacher_targets[..., 1:].sum(dim=-1) - teacher_targets[..., 0]).clamp_min(0.0)
                        nonfold_logits = teacher_logits[..., 1:]
                        fold_log_odds = teacher_logits[..., 0] - torch.logsumexp(nonfold_logits, dim=-1)
                        fold_weights = nonfold_preference * teacher_effective_confidences * fold_available * facing_4bet_teacher_weights
                        preflop_teacher_fold_contrast_loss = (nn.functional.softplus(fold_log_odds) * fold_weights).sum() / fold_weights.sum().clamp_min(1e-6)
                        preflop_teacher_allin_suppressed.append(int(low_confidence_allin.sum().item()))
                    else:
                        preflop_3bet_teacher_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        preflop_teacher_allin_contrast_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        preflop_teacher_facing_4bet_call_contrast_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        preflop_teacher_shallow_allin_margin_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        preflop_teacher_fold_contrast_loss = torch.zeros((), dtype=torch.float32, device=self.runtime.device)
                        preflop_teacher_allin_suppressed.append(0)
                    total_entropy = policy.entropy() + torch.where(raise_active_batch, sizing.entropy(), torch.zeros_like(continuous_log_probs))
                    entropy = total_entropy[valid_batch].mean()
                    approx_kl = approximate_policy_kl(old_log_prob_batch[valid_batch], new_log_probs[valid_batch])
                    clip_fraction = (valid_ratio.sub(clipped_ratio).abs() > 1e-6).float().mean()
                    loss = policy_loss + 0.45 * value_loss + 0.16 * distributional_value_loss + PPO_RANGE_LOSS_WEIGHT * range_loss + PREFLOP_ALLIN_CALIBRATION_WEIGHT * all_in_calibration_loss + PREFLOP_TAIL_ALLIN_WEIGHT * all_in_tail_risk_loss + PREFLOP_ALLIN_STABILITY_WEIGHT * all_in_stability_loss + PREFLOP_ALLIN_RANKING_WEIGHT * all_in_ranking_loss + PREFLOP_3BET_TEACHER_WEIGHT * preflop_3bet_teacher_loss + PREFLOP_TEACHER_ALLIN_CONTRASTIVE_WEIGHT * preflop_teacher_allin_contrast_loss + PREFLOP_TEACHER_FACING_4BET_CALL_CONTRASTIVE_WEIGHT * preflop_teacher_facing_4bet_call_contrast_loss + PREFLOP_TEACHER_SHALLOW_ALLIN_MARGIN_WEIGHT * preflop_teacher_shallow_allin_margin_loss + PREFLOP_TEACHER_FOLD_CONTRASTIVE_WEIGHT * preflop_teacher_fold_contrast_loss - self.ppo_entropy_coefficient * entropy
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                distributional_value_losses.append(float(distributional_value_loss.item()))
                entropies.append(float(entropy.item()))
                kl_values.append(float(approx_kl.item()))
                range_losses.append(float(range_loss.item()))
                range_accuracies.append(float(range_accuracy.item()))
                range_briers.append(float(range_brier.item()))
                range_eces.append(float(range_ece.item()))
                range_coarse_accuracies.append(float(range_coarse_accuracy.item()))
                range_coarse_briers.append(float(range_coarse_brier.item()))
                ensemble_disagreements.append(float(value_distribution_moments(valid_distributions)[1].mean().item()))
                clip_fractions.append(float(clip_fraction.item()))
                all_in_calibration_losses.append(float(all_in_calibration_loss.item()))
                all_in_stability_losses.append(float(all_in_stability_loss.item()))
                guarded_all_in_probabilities.append(float(guarded_all_in_probability_mean.item()))
                all_in_targets.append(float(guarded_all_in_target_mean.item()))
                preflop_3bet_teacher_losses.append(float(preflop_3bet_teacher_loss.item()))
                preflop_teacher_allin_contrast_losses.append(float(preflop_teacher_allin_contrast_loss.item()))
                preflop_teacher_facing_4bet_call_contrast_losses.append(float(preflop_teacher_facing_4bet_call_contrast_loss.item()))
                preflop_teacher_shallow_allin_margin_losses.append(float(preflop_teacher_shallow_allin_margin_loss.item()))
                current_kl = float(approx_kl.item())
                if current_kl > hard_kl_limit:
                    # This can occur after earlier mini-batches already changed
                    # the model.  Stopping alone leaves those unsafe changes in
                    # place, so rollback must cover the entire PPO update.
                    rollback_ppo_update("pre_step")
                    pre_step_guarded = True
                    kl_limited = True
                    stop_early = True
                    break
                self._optimizer_step(loss)
                with torch.no_grad(), self._autocast():
                    post_logits, _, _, _, _, _, post_raise_shapes, _ = self.model(observation_batch, padding_mask=~valid_batch)
                if not ppo_candidate_is_finite(post_logits, post_raise_shapes):
                    # Check in FP32 before deciding whether the update corrupted
                    # the model or only overflowed in the FP16 verification pass.
                    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                        full_precision_logits, _, _, _, _, _, full_precision_raise_shapes, _ = self.model(observation_batch, padding_mask=~valid_batch)
                    precision_state = ppo_candidate_precision_state(post_logits, post_raise_shapes, full_precision_logits, full_precision_raise_shapes)
                    if precision_state == "nonfinite":
                        # A genuine FP32 non-finite candidate can be caused by a
                        # step that is simply too large. Retry once from the
                        # complete pre-update state, rather than discarding a
                        # potentially useful, bounded update outright.
                        restore_pre_update()
                        post_step_retry_applied = True
                        post_step_guarded = True
                        for parameter in self.model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(self.ppo_post_step_retry_scale)
                        retry_learning_rate = self.ppo_learning_rate * self.ppo_post_step_retry_scale
                        for group in self.optimizer.param_groups:
                            group["lr"] = retry_learning_rate
                        self.optimizer.step()
                        with torch.no_grad(), self._autocast():
                            retry_logits, _, _, _, _, _, retry_raise_shapes, _ = self.model(observation_batch, padding_mask=~valid_batch)
                        if not ppo_candidate_is_finite(retry_logits, retry_raise_shapes):
                            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                                full_precision_logits, _, _, _, _, _, full_precision_raise_shapes, _ = self.model(observation_batch, padding_mask=~valid_batch)
                            retry_precision_state = ppo_candidate_precision_state(retry_logits, retry_raise_shapes, full_precision_logits, full_precision_raise_shapes)
                            rollback_ppo_update("post_step_retry_amp_overflow" if retry_precision_state == "amp_overflow" else "post_step_retry_nonfinite")
                            amp_fallback = retry_precision_state == "amp_overflow" and self._disable_mixed_precision_after_overflow()
                            self.ppo_post_step_retry_scale = max(0.20, self.ppo_post_step_retry_scale * 0.80)
                            kl_limited = True
                            stop_early = True
                            log_training_debug("ppo_candidate_rolled_back", phase="post_step_retry_amp_overflow" if amp_fallback else "post_step_retry_nonfinite")
                            break
                        with torch.no_grad(), self._autocast():
                            retry_policy = masked_distribution(retry_logits, mask_batch)
                            retry_sizing = raise_distribution(retry_raise_shapes, action_batch)
                            retry_continuous_log_probs = retry_sizing.log_prob(raise_fraction_batch.clamp(0.005, 0.995))
                            retry_log_probs = retry_policy.log_prob(action_batch) + torch.where(raise_active_batch, retry_continuous_log_probs, torch.zeros_like(retry_continuous_log_probs))
                            post_step_retry_kl = float(approximate_policy_kl(old_log_prob_batch[valid_batch], retry_log_probs[valid_batch]).item())
                        kl_values.append(post_step_retry_kl)
                        kl_limited = True
                        stop_early = True
                        if ppo_retry_is_acceptable("finite", post_step_retry_kl, hard_kl_limit):
                            post_step_retry_accepted = True
                            self.ppo_post_step_retry_scale = min(0.60, self.ppo_post_step_retry_scale * 1.05)
                            rollback_phase = "post_step_nonfinite_backoff_accepted"
                            self.optimizer.zero_grad(set_to_none=True)
                            break
                        rollback_ppo_update("post_step_nonfinite_backoff")
                        self.ppo_post_step_retry_scale = max(0.20, self.ppo_post_step_retry_scale * 0.80)
                        break
                    rollback_ppo_update("post_step_amp_overflow")
                    amp_fallback = self._disable_mixed_precision_after_overflow()
                    self.ppo_post_step_retry_scale = max(0.20, self.ppo_post_step_retry_scale * 0.80)
                    post_step_guarded = True
                    kl_limited = True
                    stop_early = True
                    log_training_debug("ppo_candidate_rolled_back", phase="post_step_amp_overflow" if amp_fallback else "post_step_nonfinite")
                    break
                with torch.no_grad(), self._autocast():
                    post_policy = masked_distribution(post_logits, mask_batch)
                    post_sizing = raise_distribution(post_raise_shapes, action_batch)
                    post_continuous_log_probs = post_sizing.log_prob(raise_fraction_batch.clamp(0.005, 0.995))
                    post_log_probs = post_policy.log_prob(action_batch) + torch.where(raise_active_batch, post_continuous_log_probs, torch.zeros_like(post_continuous_log_probs))
                    post_kl = float(approximate_policy_kl(old_log_prob_batch[valid_batch], post_log_probs[valid_batch]).item())
                    kl_values.append(post_kl)
                if post_kl > hard_kl_limit:
                    # The first optimizer step already produced clipped, unscaled
                    # gradients. Restore its complete state, then retry that exact
                    # step at half strength. This is a bounded line search: it
                    # accepts only after re-checking the same hard-KL guard.
                    restore_pre_update()
                    post_step_retry_applied = True
                    post_step_guarded = True
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(self.ppo_post_step_retry_scale)
                    retry_learning_rate = self.ppo_learning_rate * self.ppo_post_step_retry_scale
                    for group in self.optimizer.param_groups:
                        group["lr"] = retry_learning_rate
                    self.optimizer.step()
                    with torch.no_grad(), self._autocast():
                        retry_logits, _, _, _, _, _, retry_raise_shapes, _ = self.model(observation_batch, padding_mask=~valid_batch)
                    if not ppo_candidate_is_finite(retry_logits, retry_raise_shapes):
                        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                            full_precision_logits, _, _, _, _, _, full_precision_raise_shapes, _ = self.model(observation_batch, padding_mask=~valid_batch)
                        precision_state = ppo_candidate_precision_state(retry_logits, retry_raise_shapes, full_precision_logits, full_precision_raise_shapes)
                        rollback_ppo_update("post_step_retry_amp_overflow" if precision_state == "amp_overflow" else "post_step_retry_nonfinite")
                        amp_fallback = precision_state == "amp_overflow" and self._disable_mixed_precision_after_overflow()
                        self.ppo_post_step_retry_scale = max(0.20, self.ppo_post_step_retry_scale * 0.80)
                        kl_limited = True
                        stop_early = True
                        log_training_debug("ppo_candidate_rolled_back", phase="post_step_retry_amp_overflow" if amp_fallback else "post_step_retry_nonfinite")
                        break
                    with torch.no_grad(), self._autocast():
                        retry_policy = masked_distribution(retry_logits, mask_batch)
                        retry_sizing = raise_distribution(retry_raise_shapes, action_batch)
                        retry_continuous_log_probs = retry_sizing.log_prob(raise_fraction_batch.clamp(0.005, 0.995))
                        retry_log_probs = retry_policy.log_prob(action_batch) + torch.where(raise_active_batch, retry_continuous_log_probs, torch.zeros_like(retry_continuous_log_probs))
                        post_step_retry_kl = float(approximate_policy_kl(old_log_prob_batch[valid_batch], retry_log_probs[valid_batch]).item())
                    kl_values.extend((post_kl, post_step_retry_kl))
                    kl_limited = True
                    stop_early = True
                    if ppo_retry_is_acceptable("finite", post_step_retry_kl, hard_kl_limit):
                        post_step_retry_accepted = True
                        self.ppo_post_step_retry_scale = min(0.60, self.ppo_post_step_retry_scale * 1.05)
                        rollback_phase = "post_step_backoff_accepted"
                        self.optimizer.zero_grad(set_to_none=True)
                        break
                    rollback_ppo_update("post_step")
                    self.ppo_post_step_retry_scale = max(0.20, self.ppo_post_step_retry_scale * 0.80)
                    break
            epochs_run += 1
            if stop_early:
                break
        self.model.eval()
        candidate_root_signature = preflop_root_policy_signature(self.model, root_probe_stage)
        update_root_drift = preflop_root_policy_drift(pre_update_root_signature, candidate_root_signature)
        pre_update_anchor_root_drift = preflop_root_policy_drift(anchor_root_signature, pre_update_root_signature) if anchor_root_signature else {
            "max_kl": 0.0,
            "max_kl_root": "pending",
            "max_action_delta": 0.0,
            "max_action_delta_root": "pending",
        }
        anchor_root_drift = preflop_root_policy_drift(anchor_root_signature, candidate_root_signature) if anchor_root_signature else {
            "max_kl": 0.0,
            "max_kl_root": "pending",
            "max_action_delta": 0.0,
            "max_action_delta_root": "pending",
        }
        focused_root = focused_preflop_root(self.preflop_root_weakness)
        anchor_protected_roots = {focused_root} if focused_root else set()
        if self.last_preflop_scenario_worst_root in self.preflop_root_weakness and self.preflop_root_weakness[self.last_preflop_scenario_worst_root] >= 0.72:
            anchor_protected_roots.add(self.last_preflop_scenario_worst_root)
        root_guard_reasons = ppo_root_drift_guard_reasons(
            update_root_drift,
            pre_update_anchor_root_drift,
            anchor_root_drift,
            protected_roots=anchor_protected_roots,
        )
        root_guarded = bool(root_guard_reasons) and not update_reverted
        if root_guarded:
            proposed_state = clone_state(self.model)
            original_root_guard_reasons = list(root_guard_reasons)
            root_backoff_applied = True
            for scale in (0.5, 0.25, 0.125, 0.0625):
                restore_pre_update()
                self.model.load_state_dict(interpolate_model_state(pre_update_state, proposed_state, scale))
                self.model.eval()
                backoff_signature = preflop_root_policy_signature(self.model, root_probe_stage)
                backoff_update_drift = preflop_root_policy_drift(pre_update_root_signature, backoff_signature)
                backoff_anchor_drift = preflop_root_policy_drift(anchor_root_signature, backoff_signature) if anchor_root_signature else {
                    "max_kl": 0.0,
                    "max_kl_root": "pending",
                    "max_action_delta": 0.0,
                    "max_action_delta_root": "pending",
                }
                backoff_reasons = ppo_root_drift_guard_reasons(
                    backoff_update_drift,
                    pre_update_anchor_root_drift,
                    backoff_anchor_drift,
                    protected_roots=anchor_protected_roots,
                )
                if not backoff_reasons:
                    root_backoff_accepted = True
                    root_backoff_scale = scale
                    update_root_drift = backoff_update_drift
                    anchor_root_drift = backoff_anchor_drift
                    rollback_phase = "preflop_root_backoff_accepted"
                    self.optimizer.zero_grad(set_to_none=True)
                    log_training_debug(
                        "ppo_preflop_root_backoff_accepted",
                        scale=scale,
                        original_reasons=original_root_guard_reasons,
                        update=update_root_drift,
                        anchor_before=pre_update_anchor_root_drift,
                        anchor=anchor_root_drift,
                        stage=root_probe_stage,
                    )
                    break
            if not root_backoff_accepted:
                rollback_ppo_update("preflop_root_drift")
                self.model.eval()
                post_step_retry_accepted = False
                log_training_debug(
                    "ppo_preflop_root_drift_rollback",
                    reasons=original_root_guard_reasons,
                    update=update_root_drift,
                    anchor_before=pre_update_anchor_root_drift,
                    anchor=anchor_root_drift,
                    stage=root_probe_stage,
                )
            kl_limited = True
        ppo_compute_seconds = perf_counter() - ppo_compute_started
        self.updates += 1
        with self._lock:
            active_member = self.population_members[self.active_population_index]
            if not update_reverted:
                active_member["updates"] = int(active_member.get("updates", 0)) + 1
        sampled_raise_fractions = [fraction for path in paths for action, fraction in zip(path.actions, path.raise_fractions) if action in RAISE_ACTIONS]
        if sampled_raise_fractions:
            self.last_continuous_raise_mean = sum(sampled_raise_fractions) / len(sampled_raise_fractions)
        average_kl = sum(kl_values) / max(1, len(kl_values))
        average_entropy = sum(entropies) / max(1, len(entropies))
        average_clip_fraction = sum(clip_fractions) / max(1, len(clip_fractions))
        self.last_preflop_allin_calibration_loss = sum(all_in_calibration_losses) / max(1, len(all_in_calibration_losses))
        self.last_preflop_allin_stability_loss = sum(all_in_stability_losses) / max(1, len(all_in_stability_losses))
        self.last_preflop_guarded_allin_probability = sum(guarded_all_in_probabilities) / max(1, len(guarded_all_in_probabilities))
        self.last_preflop_allin_target = sum(all_in_targets) / max(1, len(all_in_targets))
        self.last_preflop_guarded_state_fraction = sum(sum(path.all_in_calibration_active) for path in paths) / max(1, sum(len(path.actions) for path in paths))
        teacher_eligible_roots = sum(sum(path.preflop_3bet_teacher_eligible) for path in paths)
        teacher_samples = [
            (path.preflop_root, target, confidence, advantage, action)
            for path in paths
            for target, confidence, advantage, action in zip(path.preflop_3bet_teacher_targets, path.preflop_3bet_teacher_confidences, path.preflop_3bet_teacher_raise_advantages, path.actions)
            if sum(target) > 0.0
        ]
        self.last_preflop_3bet_teacher_loss = sum(preflop_3bet_teacher_losses) / max(1, len(preflop_3bet_teacher_losses))
        self.last_preflop_3bet_teacher_eligible_roots = teacher_eligible_roots
        self.last_preflop_3bet_teacher_samples = len(teacher_samples)
        self.last_preflop_3bet_teacher_coverage = len(teacher_samples) / max(1, teacher_eligible_roots)
        self.last_preflop_3bet_teacher_confidence = sum(confidence for _, _, confidence, _, _ in teacher_samples) / max(1, len(teacher_samples))
        effective_teacher_weights = [
            max(0.0, min(1.0, (confidence - PREFLOP_3BET_TEACHER_MIN_CONFIDENCE) / max(1e-6, 1.0 - PREFLOP_3BET_TEACHER_MIN_CONFIDENCE)))
            for _, _, confidence, _, _ in teacher_samples
        ]
        self.last_preflop_3bet_teacher_effective_coverage = sum(weight > 0.0 for weight in effective_teacher_weights) / max(1, teacher_eligible_roots)
        self.last_preflop_3bet_teacher_effective_weight = sum(effective_teacher_weights) / max(1, len(effective_teacher_weights))
        self.last_preflop_3bet_teacher_raise_target = sum(target[2] for _, target, _, _, _ in teacher_samples) / max(1, len(teacher_samples))
        self.last_preflop_3bet_teacher_raise_advantage_bb = sum(advantage for _, _, _, advantage, _ in teacher_samples) / max(1, len(teacher_samples))
        self.last_preflop_3bet_teacher_actual_raise_rate = sum(action == 2 for _, _, _, _, action in teacher_samples) / max(1, len(teacher_samples))
        self.last_preflop_3bet_teacher_allin_target = sum(target[3] for _, target, _, _, _ in teacher_samples) / max(1, len(teacher_samples))
        self.last_preflop_3bet_teacher_actual_allin_rate = sum(action == 3 for _, _, _, _, action in teacher_samples) / max(1, len(teacher_samples))
        self.last_preflop_3bet_teacher_allin_suppressed = sum(preflop_teacher_allin_suppressed)
        root_teacher_totals = {root: empty_preflop_teacher_root_totals() for root in PREFLOP_3BET_TEACHER_ROOTS}
        for path in paths:
            for eligible in path.preflop_3bet_teacher_eligible:
                if eligible and path.preflop_root in root_teacher_totals:
                    root_teacher_totals[path.preflop_root]["eligible_roots"] = int(root_teacher_totals[path.preflop_root]["eligible_roots"]) + 1
        for root, target, confidence, _, action in teacher_samples:
            if root not in root_teacher_totals:
                continue
            totals = root_teacher_totals[root]
            totals["sampled_roots"] = int(totals["sampled_roots"]) + 1
            totals["confidence_sum"] = float(totals["confidence_sum"]) + confidence
            target_sums = totals["target_action_sums"]
            actual_counts = totals["actual_action_counts"]
            assert isinstance(target_sums, dict) and isinstance(actual_counts, dict)
            for index, name in enumerate(PREFLOP_3BET_TEACHER_ACTION_NAMES):
                target_sums[name] += target[index]
                actual_counts[name] += float(action == index)
        self.last_preflop_3bet_teacher_by_root = {
            root: preflop_teacher_root_metrics(totals)
            for root, totals in root_teacher_totals.items()
            if int(totals["eligible_roots"]) > 0 or int(totals["sampled_roots"]) > 0
        }
        with self._lock:
            for root, totals in root_teacher_totals.items():
                if not int(totals["eligible_roots"]) and not int(totals["sampled_roots"]):
                    continue
                run_totals = self.run_preflop_teacher_by_root.setdefault(root, empty_preflop_teacher_root_totals())
                run_totals["eligible_roots"] = int(run_totals["eligible_roots"]) + int(totals["eligible_roots"])
                run_totals["sampled_roots"] = int(run_totals["sampled_roots"]) + int(totals["sampled_roots"])
                run_totals["confidence_sum"] = float(run_totals["confidence_sum"]) + float(totals["confidence_sum"])
                for key in ("target_action_sums", "actual_action_counts"):
                    source = totals[key]
                    destination = run_totals[key]
                    assert isinstance(source, dict) and isinstance(destination, dict)
                    for name in PREFLOP_3BET_TEACHER_ACTION_NAMES:
                        destination[name] += source[name]
        multi_raise_teacher_samples = [sample for sample in teacher_samples if sample[0] in PREFLOP_3BET_TEACHER_MULTI_RAISE_ROOTS]
        self.last_preflop_3bet_teacher_multi_raise_samples = len(multi_raise_teacher_samples)
        self.last_preflop_3bet_teacher_multi_raise_allin_target = sum(target[3] for _, target, _, _, _ in multi_raise_teacher_samples) / max(1, len(multi_raise_teacher_samples))
        self.last_preflop_3bet_teacher_multi_raise_actual_allin_rate = sum(action == 3 for _, _, _, _, action in multi_raise_teacher_samples) / max(1, len(multi_raise_teacher_samples))
        self.last_preflop_3bet_teacher_multi_raise_allin_vetoes = sum(
            target[3] + 1e-6 < max(target[:3])
            for _, target, confidence, _, _ in multi_raise_teacher_samples
            if confidence >= PREFLOP_3BET_TEACHER_MIN_CONFIDENCE
        )
        facing_4bet_teacher_samples = [sample for sample in teacher_samples if sample[0] == "facing_4bet"]
        self.last_preflop_3bet_teacher_facing_4bet_samples = len(facing_4bet_teacher_samples)
        self.last_preflop_3bet_teacher_facing_4bet_target_actions = {
            name: sum(target[index] for _, target, _, _, _ in facing_4bet_teacher_samples) / max(1, len(facing_4bet_teacher_samples))
            for index, name in enumerate(PREFLOP_3BET_TEACHER_ACTION_NAMES)
        }
        self.last_preflop_3bet_teacher_facing_4bet_actual_actions = {
            name: sum(action == index for _, _, _, _, action in facing_4bet_teacher_samples) / max(1, len(facing_4bet_teacher_samples))
            for index, name in enumerate(PREFLOP_3BET_TEACHER_ACTION_NAMES)
        }
        self.last_preflop_3bet_teacher_facing_4bet_non_allin_vetoes = sum(
            target[3] + 1e-6 < max(target[:3])
            for _, target, confidence, _, _ in facing_4bet_teacher_samples
            if confidence >= PREFLOP_3BET_TEACHER_MIN_CONFIDENCE
        )
        self.last_ppo_epochs = epochs_run
        self.last_ppo_clip_fraction = average_clip_fraction
        self.last_ppo_kl_limited = kl_limited
        self.last_ppo_hard_kl = max(kl_values, default=0.0)
        self.last_ppo_update_reverted = update_reverted
        self.last_ppo_rollback_phase = rollback_phase
        self.last_ppo_post_step_retry_applied = post_step_retry_applied
        self.last_ppo_post_step_retry_accepted = post_step_retry_accepted
        self.last_ppo_post_step_retry_kl = post_step_retry_kl
        self.last_ppo_root_backoff_applied = root_backoff_applied
        self.last_ppo_root_backoff_accepted = root_backoff_accepted
        self.last_ppo_root_backoff_scale = root_backoff_scale
        self.last_preflop_root_guarded = root_guarded and not root_backoff_accepted
        self.last_preflop_root_guard_reason = (
            f"backoff {root_backoff_scale:.4f} accepted after: {'; '.join(root_guard_reasons)}"
            if root_backoff_accepted
            else "; ".join(root_guard_reasons) if root_guard_reasons else "none"
        )
        self.last_preflop_root_update_kl = float(update_root_drift["max_kl"])
        self.last_preflop_root_anchor_kl = float(anchor_root_drift["max_kl"])
        self.last_preflop_root_update_action_delta = float(update_root_drift["max_action_delta"])
        self.last_preflop_root_anchor_action_delta = float(anchor_root_drift["max_action_delta"])
        self.last_preflop_root_drift_root = str(
            anchor_root_drift["max_kl_root"]
            if float(anchor_root_drift["max_kl"]) >= float(update_root_drift["max_kl"])
            else update_root_drift["max_kl_root"]
        )
        with self._lock:
            self.run_ppo_safety["updates"] += 1
            self.run_ppo_safety["pre_step_guards"] += int(pre_step_guarded)
            self.run_ppo_safety["post_step_guards"] += int(post_step_guarded)
            self.run_ppo_safety["root_guards"] += int(root_guarded)
            self.run_ppo_safety["retry_attempts"] += int(post_step_retry_applied)
            self.run_ppo_safety["retry_accepted"] += int(post_step_retry_accepted)
            self.run_ppo_safety["root_backoff_attempts"] += int(root_backoff_applied)
            self.run_ppo_safety["root_backoff_accepted"] += int(root_backoff_accepted)
            self.run_ppo_safety["reverted_updates"] += int(update_reverted)
            self.run_ppo_safety["retry_kl_sum"] += post_step_retry_kl if post_step_retry_applied else 0.0
        self._adapt_ppo_controller(average_kl, average_entropy, average_clip_fraction, kl_limited)
        adversarial_paths = [path for path in paths if path.adversarial]
        tail_paths = [path for path in adversarial_paths if path.reward <= -LARGE_LOSS_BB]
        tail_loss_sum = sum(path.reward for path in tail_paths)
        tail_weight_sum = sum(
            sum(
                adversarial_tail_credit_weights(
                    base_weight=min(1.0 + ROBUST_STYLE_POLICY_WEIGHT, max(1.0, float(path.robust_weight))),
                    tail_weight=adversarial_tail_policy_weight(path.reward),
                    reward_bb=path.reward,
                    large_loss_bb=LARGE_LOSS_BB,
                    advantages=trajectory_advantages(path),
                    streets=path.streets,
                    masks=path.masks,
                )
            ) / max(1, len(path.actions))
            for path in tail_paths
        )
        tail_by_style: dict[str, dict[str, Any]] = {}
        for style in sorted({path.opponent_style for path in adversarial_paths}):
            style_paths = [path for path in adversarial_paths if path.opponent_style == style]
            style_tail = [path for path in style_paths if path.reward <= -LARGE_LOSS_BB]
            terminal_actions = {"fold": 0, "check": 0, "call": 0, "raise": 0, "all_in": 0}
            terminal_streets = {"preflop": 0, "flop": 0, "turn": 0, "river": 0}
            for path in style_tail:
                if not path.actions:
                    continue
                action = path.actions[-1]
                if action == 0:
                    action_name = "fold"
                elif action == 1:
                    action_name = "call" if path.masks[-1][0] else "check"
                elif action == 2:
                    action_name = "raise"
                else:
                    action_name = "all_in"
                terminal_actions[action_name] += 1
                street = path.streets[-1] if path.streets else 0
                terminal_streets[("preflop", "flop", "turn", "river")[min(3, max(0, street))]] += 1
            tail_by_style[style] = {
                "hands": len(style_paths),
                "tail_hands": len(style_tail),
                "reward_sum": sum(path.reward for path in style_paths),
                "tail_loss_sum": sum(path.reward for path in style_tail),
                "terminal_actions": terminal_actions,
                "terminal_streets": terminal_streets,
            }
        with self._lock:
            self.run_adversarial_paths += len(adversarial_paths)
            self.run_tail_paths += len(tail_paths)
            self.run_tail_loss_sum += tail_loss_sum
            self.run_tail_weight_sum += tail_weight_sum
            self._tail_diagnostic_window.append((len(adversarial_paths), len(tail_paths), tail_loss_sum, tail_weight_sum))
            self._tail_diagnostic_window = self._tail_diagnostic_window[-ROLLING_DIAGNOSTIC_UPDATES:]
            for style, metrics in tail_by_style.items():
                totals = self.run_tail_style_totals.setdefault(style, {"hands": 0, "tail_hands": 0, "reward_sum": 0.0, "tail_loss_sum": 0.0, "terminal_actions": {}, "terminal_streets": {}})
                for key in ("hands", "tail_hands", "reward_sum", "tail_loss_sum"):
                    totals[key] = totals.get(key, 0) + metrics[key]
                for key in ("terminal_actions", "terminal_streets"):
                    destination = totals.setdefault(key, {})
                    for name, count in metrics[key].items():
                        destination[name] = destination.get(name, 0) + count
            window_adversarial = sum(item[0] for item in self._tail_diagnostic_window)
            window_tail = sum(item[1] for item in self._tail_diagnostic_window)
            window_tail_loss = sum(item[2] for item in self._tail_diagnostic_window)
            window_tail_weight = sum(item[3] for item in self._tail_diagnostic_window)
            self.last_tail_loss_rate = window_tail / max(1, window_adversarial)
            self.last_tail_loss_bb = window_tail_loss / max(1, window_tail)
            self.last_tail_policy_weight = window_tail_weight / max(1, window_tail)
            self.last_tail_style_diagnostics = {
                style: {
                    "hands": int(totals["hands"]),
                    "tail_rate": float(totals["tail_hands"]) / max(1, float(totals["hands"])),
                    "mean_reward_bb": float(totals["reward_sum"]) / max(1, float(totals["hands"])),
                    "tail_loss_bb": float(totals["tail_loss_sum"]) / max(1, float(totals["tail_hands"])),
                    "terminal_action_mix": {name: int(count) / max(1, int(totals["tail_hands"])) for name, count in dict(totals.get("terminal_actions", {})).items()},
                    "terminal_street_mix": {name: int(count) / max(1, int(totals["tail_hands"])) for name, count in dict(totals.get("terminal_streets", {})).items()},
                }
                for style, totals in self.run_tail_style_totals.items()
            }
        return {"policy_loss": sum(policy_losses) / len(policy_losses), "value_loss": sum(value_losses) / len(value_losses), "distributional_value_loss": sum(distributional_value_losses) / len(distributional_value_losses), "ensemble_disagreement": sum(ensemble_disagreements) / max(1, len(ensemble_disagreements)), "entropy": average_entropy, "kl_divergence": average_kl, "range_loss": sum(range_losses) / len(range_losses), "range_accuracy": sum(range_accuracies) / len(range_accuracies), "range_brier": sum(range_briers) / len(range_briers), "range_ece": sum(range_eces) / len(range_eces), "range_coarse_accuracy": sum(range_coarse_accuracies) / len(range_coarse_accuracies), "range_coarse_brier": sum(range_coarse_briers) / len(range_coarse_briers), "ppo_clip_fraction": average_clip_fraction, "ppo_epochs": float(epochs_run), "ppo_learning_rate": self.ppo_learning_rate, "ppo_kl_limited": float(kl_limited), "ppo_update_reverted": float(update_reverted), "ppo_rollback_phase": rollback_phase, "ppo_post_step_retry_applied": float(post_step_retry_applied), "ppo_post_step_retry_accepted": float(post_step_retry_accepted), "ppo_post_step_retry_kl": post_step_retry_kl, "tail_loss_rate": self.last_tail_loss_rate, "tail_policy_weight": self.last_tail_policy_weight, "preflop_allin_calibration_loss": self.last_preflop_allin_calibration_loss, "preflop_allin_stability_loss": self.last_preflop_allin_stability_loss, "preflop_guarded_allin_probability": self.last_preflop_guarded_allin_probability, "preflop_allin_target": self.last_preflop_allin_target, "preflop_guarded_state_fraction": self.last_preflop_guarded_state_fraction, "preflop_3bet_teacher_loss": self.last_preflop_3bet_teacher_loss, "preflop_3bet_teacher_coverage": self.last_preflop_3bet_teacher_coverage, "preflop_3bet_teacher_samples": float(self.last_preflop_3bet_teacher_samples), "ppo_tensor_preparation_seconds": tensor_preparation_seconds, "ppo_transfer_seconds": transfer_seconds, "ppo_compute_seconds": ppo_compute_seconds}

    def add_cfr_records(self, records: list[CFRRecord]) -> None:
        if not records:
            return
        with self._lock:
            self.cfr_memory.extend(records, self._rng)
            self.strategy_memory.extend(records, self._rng)
            self.search_value_memory.extend([SearchValueRecord.from_cfr(record) for record in records if record.search_depth > 0], self._rng)
            counterfactual_records = [record for source in records if (record := CounterfactualValueRecord.from_cfr(source)) is not None]
            self.counterfactual_value_memory.extend(counterfactual_records, self._rng)
            self.last_rare_spot_rate = sum(record.rare for record in records) / len(records)
            self.last_belief_confidence = sum(record.reach_weight for record in records) / len(records)
            self.last_belief_posterior_support = sum(record.belief_support for record in records) / len(records)
            resolved_records = [record for record in records if record.search_depth > 0]
            self.last_resolver_replay_confidence = sum(record.resolver_confidence for record in resolved_records) / max(1, len(resolved_records))
            self.last_leaf_evaluations = sum(record.leaf_evaluations for record in records)
            composition = self.cfr_memory.composition()
            self.last_replay_rare_fraction = composition["rare"]
            self.last_replay_priority = composition["priority"]
            self.last_replay_recent_fraction = composition["recent"]
            self.last_strategy_memory_size = len(self.strategy_memory.records)
            self.last_search_memory_size = len(self.search_value_memory.records)
            self.last_resolver_replay_size = sum(record.search_depth > 0 for record in self.strategy_memory.records)
            self.last_counterfactual_coverage = sum(len(record.classes) for record in counterfactual_records) / max(1, len(records) * BELIEF_VALUE_CLASSES)
            self.last_counterfactual_memory_size = len(self.counterfactual_value_memory.records)
            self.last_public_belief_teacher_size = sum(record.search_depth >= 11 for record in self.strategy_memory.records)
            raise_targets = [target for record in records for action, target in enumerate(record.sizing_targets or []) if action in RAISE_ACTIONS and (record.sizing_weights or [0.0] * ACTION_COUNT)[action] > 0]
            self.last_sizing_proposal_diversity = min(1.0, len({round(target, 2) for target in raise_targets}) / 8) if raise_targets else 0.0

    def add_oracle_records(self, records: list[AbstractTeacherRecord | SolverTeacherRecord]) -> None:
        if not ENABLE_ABSTRACT_CFR_TEACHER or not records:
            return
        with self._lock:
            self.abstract_teacher_memory.extend(records, self._rng)
            self.last_oracle_confidence = sum(record.confidence for record in records) / len(records)

    def advance_abstract_oracle(self) -> None:
        with self._lock:
            if ENABLE_HEURISTIC_ORACLE:
                self.abstract_oracle.solve(8)
            if ENABLE_ABSTRACT_CFR_TEACHER:
                self.abstract_cfr_solver.solve(1)
                self.last_oracle_iterations = self.abstract_cfr_solver.iterations
                if self.updates % 3 == 0:
                    audit = self.abstract_cfr_solver.audit()
                    self.last_abstraction_nash_conv = audit.nash_conv
                    self.last_abstraction_value = audit.average_value
                    self.last_abstraction_information_sets = audit.information_sets

    def add_action_likelihood_records(self, records: list[ActionLikelihoodRecord]) -> None:
        if not records:
            return
        with self._lock:
            self.action_likelihood_memory.extend(records, self._rng)

    def add_imitation_paths(self, paths: list[HandTrajectory]) -> None:
        if not paths:
            return
        with self._lock:
            self.imitation_memory.extend_paths(paths, self._rng)

    def add_hard_spot_paths(self, paths: list[HandTrajectory]) -> None:
        """Retain adversarial loss states for critic calibration, never policy cloning."""
        if not paths:
            return
        with self._lock:
            self.hard_spot_value_memory.extend_paths(paths, self._rng)
            self.last_hard_spot_memory_size = len(self.hard_spot_value_memory.records)

    def hard_spot_value_update(self) -> dict:
        """Teach the critic to price costly adversarial states without off-policy PPO."""
        with self._lock:
            records = self.hard_spot_value_memory.sample(384, self._rng, self._hard_spot_styles_locked())
        if not records:
            self.last_hard_spot_value_loss = 0.0
            return {"hard_spot_value_loss": 0.0, "hard_spot_memory_size": float(len(self.hard_spot_value_memory.records))}
        observations = torch.tensor([record.observation for record in records], dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor([max(-200.0, min(200.0, record.return_value)) for record in records], dtype=torch.float32)
        priorities = torch.tensor([record.priority for record in records], dtype=torch.float32)
        observations, targets, priorities = (self._to_training_device(tensor) for tensor in (observations, targets, priorities))
        weights = priorities / priorities.mean().clamp_min(1e-6)
        self.model.train()
        with self._autocast():
            values, distributions = self.model.detached_critic(observations)
            value_error = (values[:, 0] - (targets / VALUE_RETURN_SCALE_BB).unsqueeze(-1)).square()
            value_loss = (value_error * weights.unsqueeze(-1)).sum() / (weights.sum() * VALUE_ENSEMBLE_SIZE).clamp_min(1e-6)
            bins = value_support_bins(targets)
            repeated_bins = bins.unsqueeze(-1).expand(-1, VALUE_ENSEMBLE_SIZE).reshape(-1)
            distribution_error = nn.functional.cross_entropy(distributions[:, 0].reshape(-1, VALUE_BINS), repeated_bins, reduction="none").view(-1, VALUE_ENSEMBLE_SIZE)
            distribution_loss = (distribution_error * weights.unsqueeze(-1)).sum() / (weights.sum() * VALUE_ENSEMBLE_SIZE).clamp_min(1e-6)
            loss = 0.14 * value_loss + 0.07 * distribution_loss
        self._optimizer_step(loss)
        self.model.eval()
        self.last_hard_spot_value_loss = float(loss.item())
        self.last_hard_spot_memory_size = len(self.hard_spot_value_memory.records)
        return {"hard_spot_value_loss": self.last_hard_spot_value_loss, "hard_spot_memory_size": float(self.last_hard_spot_memory_size)}

    def deep_cfr_update(self) -> dict:
        """Fit CFR+ regrets, action-conditioned sizing, and SD-CFR average strategy."""
        with self._lock:
            records = self.cfr_memory.sample(512, self._rng)
            strategy_records = self.strategy_memory.sample(512, self._rng)
        if not records:
            return {"cfr_advantage_loss": 0.0, "average_strategy_loss": 0.0, "sizing_cfr_loss": 0.0, "cfr_memory_size": 0.0, "strategy_memory_size": 0.0, "cfr_effective_weight": 0.0}
        observations = torch.tensor([record.observation for record in records], dtype=torch.float32).unsqueeze(1)
        masks = torch.tensor([record.mask for record in records], dtype=torch.bool)
        sampled_masks = torch.tensor([record.sampled for record in records], dtype=torch.bool)
        advantage_targets = torch.tensor([record.advantages for record in records], dtype=torch.float32)
        sizing_targets = torch.tensor([record.sizing_targets or [0.5] * ACTION_COUNT for record in records], dtype=torch.float32)
        sizing_weights = torch.tensor([record.sizing_weights or [0.0] * ACTION_COUNT for record in records], dtype=torch.float32)
        newest_iteration = max(record.iteration for record in records)
        discounted_weights = torch.tensor([record.reach_weight * (0.45 + 0.55 * record.resolver_confidence) * min(2.5, math.sqrt(record.priority)) * (record.iteration / newest_iteration) ** 1.5 for record in records], dtype=torch.float32)
        strategy_observations = torch.tensor([record.observation for record in (strategy_records or records)], dtype=torch.float32).unsqueeze(1)
        strategy_masks = torch.tensor([record.mask for record in (strategy_records or records)], dtype=torch.bool)
        strategy_targets = torch.tensor([record.strategy for record in (strategy_records or records)], dtype=torch.float32)
        strategy_newest = max(record.iteration for record in (strategy_records or records))
        strategy_weights = torch.tensor([record.reach_weight * (0.45 + 0.55 * record.resolver_confidence) * (record.iteration / strategy_newest) ** 1.5 for record in (strategy_records or records)], dtype=torch.float32)
        observations, masks, sampled_masks, advantage_targets, sizing_targets, sizing_weights, discounted_weights, strategy_observations, strategy_masks, strategy_targets, strategy_weights = (
            self._to_training_device(tensor) for tensor in (observations, masks, sampled_masks, advantage_targets, sizing_targets, sizing_weights, discounted_weights, strategy_observations, strategy_masks, strategy_targets, strategy_weights)
        )
        advantage_losses: list[float] = []
        strategy_losses: list[float] = []
        sizing_losses: list[float] = []
        self.model.train()
        for _ in range(2):
            with self._autocast():
                _, advantage_logits, raise_shapes = self.model.detached_cfr_heads(observations)
                advantage_prediction = advantage_logits[:, 0]
                squared_advantage_error = nn.functional.smooth_l1_loss(advantage_prediction, advantage_targets, reduction="none")
                cfr_plus_error = (nn.functional.relu(advantage_prediction) - nn.functional.relu(advantage_targets)).square()
                sampled_weights = sampled_masks.float() * discounted_weights.unsqueeze(-1)
                advantage_loss = ((squared_advantage_error + 0.55 * cfr_plus_error) * sampled_weights).sum() / sampled_weights.sum().clamp_min(1e-6)
                raw_shapes = raise_shapes[:, 0].reshape(-1, RAISE_ACTION_COUNT, 2)
                sizing = Beta(nn.functional.softplus(raw_shapes[..., 0]) + 1.05, nn.functional.softplus(raw_shapes[..., 1]) + 1.05)
                target_sizes = sizing_targets[:, list(RAISE_ACTIONS)].clamp(0.005, 0.995)
                target_weights = sizing_weights[:, list(RAISE_ACTIONS)] * discounted_weights.unsqueeze(-1)
                sizing_loss = -(sizing.log_prob(target_sizes) * target_weights).sum() / target_weights.sum().clamp_min(1e-6)
                average_logits, _, _ = self.model.detached_cfr_heads(strategy_observations)
                masked_logits = average_logits[:, 0].masked_fill(~strategy_masks, torch.finfo(average_logits.dtype).min)
                per_record_strategy_loss = -(strategy_targets * torch.log_softmax(masked_logits, dim=-1)).sum(dim=-1)
                average_strategy_loss = (per_record_strategy_loss * strategy_weights).sum() / strategy_weights.sum().clamp_min(1e-6)
                loss = 0.42 * advantage_loss + 0.22 * average_strategy_loss + 0.12 * sizing_loss
            self._optimizer_step(loss)
            advantage_losses.append(float(advantage_loss.item()))
            strategy_losses.append(float(average_strategy_loss.item()))
            sizing_losses.append(float(sizing_loss.item()))
        self.model.eval()
        sizing_average = sum(sizing_losses) / len(sizing_losses)
        with self._lock:
            self.last_sizing_cfr_loss = sizing_average
            self.last_strategy_memory_size = len(self.strategy_memory.records)
        return {"cfr_advantage_loss": sum(advantage_losses) / len(advantage_losses), "average_strategy_loss": sum(strategy_losses) / len(strategy_losses), "sizing_cfr_loss": sizing_average, "cfr_memory_size": float(len(self.cfr_memory.records)), "strategy_memory_size": float(len(self.strategy_memory.records)), "cfr_effective_weight": float(discounted_weights.sum().item())}

    def subgame_policy_update(self) -> dict:
        """Distil search into the isolated average-strategy head.

        The final policy remains PPO-only; otherwise a small replay batch can
        overturn the trust-region update and collapse every fixed root.
        """
        with self._lock:
            deep_records = [record for record in self.strategy_memory.records if record.search_depth >= 11 and record.resolver_confidence >= 0.42]
            ranked = sorted(deep_records, key=lambda record: record.priority * max(1, record.search_depth), reverse=True)
            pool = ranked[:max(1, len(ranked) * 3 // 4)]
            records = self._rng.sample(pool, min(384, len(pool))) if pool else []
        if not records:
            self.last_subgame_policy_loss = 0.0
            self.last_subgame_teacher_size = 0
            return {"subgame_policy_loss": 0.0, "subgame_teacher_size": 0.0}
        observations = torch.tensor([record.observation for record in records], dtype=torch.float32).unsqueeze(1)
        masks = torch.tensor([record.mask for record in records], dtype=torch.bool)
        targets = torch.tensor([record.strategy for record in records], dtype=torch.float32)
        weights = torch.tensor([record.priority * (0.35 + 0.65 * record.resolver_confidence) * (1.0 + record.search_depth / 20.0) for record in records], dtype=torch.float32)
        observations, masks, targets, weights = (self._to_training_device(tensor) for tensor in (observations, masks, targets, weights))
        targets = targets.masked_fill(~masks, 0.0)
        targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.model.train()
        with self._autocast():
            average_logits, _, _ = self.model.detached_cfr_heads(observations)
            average_log = torch.log_softmax(average_logits[:, 0].masked_fill(~masks, torch.finfo(average_logits.dtype).min), dim=-1)
            average_loss = -(targets * average_log).sum(dim=-1)
            loss = (average_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        self._optimizer_step(loss)
        self.model.eval()
        self.last_subgame_policy_loss = float(loss.item())
        self.last_subgame_teacher_size = len(deep_records)
        return {"subgame_policy_loss": self.last_subgame_policy_loss, "subgame_teacher_size": float(self.last_subgame_teacher_size)}

    def abstract_oracle_update(self) -> dict:
        """Distil abstraction targets into isolated solver/value heads."""
        if not ENABLE_ABSTRACT_CFR_TEACHER:
            self.last_oracle_policy_loss = self.last_oracle_value_loss = 0.0
            return {"oracle_policy_loss": 0.0, "oracle_value_loss": 0.0, "oracle_teacher_size": 0.0, "oracle_confidence": 0.0}
        with self._lock:
            records = self.abstract_teacher_memory.sample(512, self._rng)
        if not records:
            self.last_oracle_policy_loss = self.last_oracle_value_loss = 0.0
            return {"oracle_policy_loss": 0.0, "oracle_value_loss": 0.0, "oracle_teacher_size": 0.0, "oracle_confidence": self.last_oracle_confidence}
        observations = torch.tensor([record.observation for record in records], dtype=torch.float32).unsqueeze(1)
        masks = torch.tensor([record.mask for record in records], dtype=torch.bool)
        strategies = torch.tensor([record.strategy for record in records], dtype=torch.float32)
        values = torch.tensor([record.value for record in records], dtype=torch.float32)
        weights = torch.tensor([record.confidence * (1.0 + record.street / 3) for record in records], dtype=torch.float32)
        observations, masks, strategies, values, weights = (self._to_training_device(tensor) for tensor in (observations, masks, strategies, values, weights))
        strategies = strategies.masked_fill(~masks, 0.0)
        strategies = strategies / strategies.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.model.train()
        with self._autocast():
            average_logits, _, raise_shapes = self.model.detached_cfr_heads(observations)
            value_heads, _ = self.model.detached_critic(observations)
            average_log = torch.log_softmax(average_logits[:, 0].masked_fill(~masks, torch.finfo(average_logits.dtype).min), dim=-1)
            average_error = -(strategies * average_log).sum(dim=-1)
            policy_loss = (average_error * weights).sum() / weights.sum().clamp_min(1e-6)
            value_error = (value_heads[:, 0].mean(dim=-1) - values / VALUE_RETURN_SCALE_BB).square()
            value_loss = (value_error * weights).sum() / weights.sum().clamp_min(1e-6)
            # Solver raises map to the half-pot and pot abstract actions.
            shapes = raise_shapes[:, 0].reshape(-1, RAISE_ACTION_COUNT, 2)
            sizing_mean = torch.sigmoid(shapes[..., 0] - shapes[..., 1]).mean(dim=-1)
            sizing_target = (0.25 + 0.50 * strategies[:, 2]).detach()
            sizing_loss = ((sizing_mean - sizing_target.unsqueeze(-1)).square() * weights.unsqueeze(-1)).sum() / weights.sum().clamp_min(1e-6)
            loss = ABSTRACT_CFR_TEACHER_WEIGHT * (0.20 * policy_loss + 0.10 * value_loss + 0.04 * sizing_loss)
        self._optimizer_step(loss)
        self.model.eval()
        self.last_oracle_policy_loss = float(policy_loss.item())
        self.last_oracle_value_loss = float(value_loss.item())
        return {"oracle_policy_loss": self.last_oracle_policy_loss, "oracle_value_loss": self.last_oracle_value_loss, "oracle_teacher_size": float(len(self.abstract_teacher_memory.records)), "oracle_confidence": self.last_oracle_confidence}

    def search_value_update(self) -> dict:
        """Distil resolved public-belief search values into a bootstrapped value ensemble."""
        with self._lock:
            records = self.search_value_memory.sample(384, self._rng)
        if not records:
            return {"search_value_loss": 0.0, "search_memory_size": 0.0, "ensemble_disagreement": 0.0}
        observations = torch.tensor([record.observation for record in records], dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor([max(-200.0, min(200.0, record.value)) for record in records], dtype=torch.float32)
        uncertainties = torch.tensor([max(0.0, min(200.0, record.uncertainty)) for record in records], dtype=torch.float32)
        priorities = torch.tensor([record.priority * (1.0 + min(1.0, record.depth / 12)) for record in records], dtype=torch.float32)
        observations, targets, uncertainties, priorities = (self._to_training_device(tensor) for tensor in (observations, targets, uncertainties, priorities))
        losses: list[float] = []
        disagreements: list[float] = []
        self.model.train()
        for _ in range(2):
            with self._autocast():
                values, distribution_logits = self.model.detached_critic(observations)
                values = values[:, 0]
                distributions = distribution_logits[:, 0]
                bootstrap = (torch.rand_like(values) > 0.20).float()
                weighted_error = (values - (targets / VALUE_RETURN_SCALE_BB).unsqueeze(-1)).square() * bootstrap * priorities.unsqueeze(-1)
                value_loss = weighted_error.sum() / (bootstrap * priorities.unsqueeze(-1)).sum().clamp_min(1.0)
                bins = value_support_bins(targets)
                repeated_bins = bins.unsqueeze(-1).expand(-1, VALUE_ENSEMBLE_SIZE).reshape(-1)
                distribution_error = nn.functional.cross_entropy(distributions.reshape(-1, VALUE_BINS), repeated_bins, reduction="none").view(-1, VALUE_ENSEMBLE_SIZE)
                distribution_bootstrap = (torch.rand_like(distribution_error) > 0.20).float()
                distribution_loss = (distribution_error * distribution_bootstrap * priorities.unsqueeze(-1)).sum() / (distribution_bootstrap * priorities.unsqueeze(-1)).sum().clamp_min(1.0)
                _, predicted_uncertainty = value_distribution_moments(distributions)
                uncertainty_loss = nn.functional.smooth_l1_loss(predicted_uncertainty, uncertainties)
                loss = 0.34 * value_loss + 0.16 * distribution_loss + 0.04 * uncertainty_loss
            self._optimizer_step(loss)
            losses.append(float(loss.item()))
            disagreements.append(float(predicted_uncertainty.mean().item()))
        self.model.eval()
        average_loss = sum(losses) / len(losses)
        average_disagreement = sum(disagreements) / len(disagreements)
        with self._lock:
            self.last_search_value_loss = average_loss
            self.last_search_memory_size = len(self.search_value_memory.records)
            self.last_ensemble_disagreement = average_disagreement
        return {"search_value_loss": average_loss, "search_memory_size": float(len(self.search_value_memory.records)), "ensemble_disagreement": average_disagreement}

    def counterfactual_value_update(self) -> dict:
        """Fit sparse solver targets without assigning values to unsampled range cells."""
        with self._lock:
            records = self.counterfactual_value_memory.sample(256, self._rng)
        if not records:
            self.last_counterfactual_value_loss = 0.0
            return {"counterfactual_value_loss": 0.0, "counterfactual_memory_size": 0.0, "counterfactual_coverage": self.last_counterfactual_coverage}
        observations = torch.tensor([record.observation for record in records], dtype=torch.float32).unsqueeze(1)
        own_beliefs = torch.tensor([record.own_belief for record in records], dtype=torch.float32)
        beliefs = torch.tensor([record.belief for record in records], dtype=torch.float32)
        targets = torch.zeros((len(records), BELIEF_VALUE_CLASSES), dtype=torch.float32)
        target_weights = torch.zeros((len(records), BELIEF_VALUE_CLASSES), dtype=torch.float32)
        confidence = torch.tensor([max(0.05, record.confidence) * (1.0 + record.depth / 16) for record in records], dtype=torch.float32)
        for row, record in enumerate(records):
            for kind, value, weight in zip(record.classes, record.values, record.weights):
                targets[row, kind] = max(-200.0, min(200.0, value))
                target_weights[row, kind] = max(target_weights[row, kind], weight)
        observations, own_beliefs, beliefs, targets, target_weights, confidence = (self._to_training_device(tensor) for tensor in (observations, own_beliefs, beliefs, targets, target_weights, confidence))
        active = target_weights > 0
        self.model.train()
        with self._autocast():
            predicted_values, predicted_uncertainty, predicted_opponent_values, predicted_opponent_uncertainty = self.model.detached_public_belief_values(observations, own_beliefs, beliefs)
            predicted_values, predicted_uncertainty = predicted_values[:, 0], predicted_uncertainty[:, 0]
            predicted_opponent_values, predicted_opponent_uncertainty = predicted_opponent_values[:, 0], predicted_opponent_uncertainty[:, 0]
            weights = target_weights * confidence.unsqueeze(-1)
            value_error = nn.functional.smooth_l1_loss(predicted_values, targets, reduction="none")
            value_loss = (value_error * weights).sum() / weights.sum().clamp_min(1e-6)
            opponent_error = nn.functional.smooth_l1_loss(predicted_opponent_values, -targets, reduction="none")
            opponent_value_loss = (opponent_error * weights).sum() / weights.sum().clamp_min(1e-6)
            uncertainty_target = (predicted_values.detach() - targets).abs()
            uncertainty_loss = (nn.functional.smooth_l1_loss(predicted_uncertainty, uncertainty_target, reduction="none") * weights).sum() / weights.sum().clamp_min(1e-6)
            opponent_uncertainty_target = (predicted_opponent_values.detach() + targets).abs()
            opponent_uncertainty_loss = (nn.functional.smooth_l1_loss(predicted_opponent_uncertainty, opponent_uncertainty_target, reduction="none") * weights).sum() / weights.sum().clamp_min(1e-6)
            loss = 0.15 * value_loss + 0.10 * opponent_value_loss + 0.035 * (uncertainty_loss + opponent_uncertainty_loss)
        self._optimizer_step(loss)
        self.model.eval()
        self.last_counterfactual_value_loss = float(loss.item())
        self.last_counterfactual_memory_size = len(self.counterfactual_value_memory.records)
        self.last_counterfactual_coverage = float(active.float().mean().item())
        return {"counterfactual_value_loss": self.last_counterfactual_value_loss, "counterfactual_memory_size": float(self.last_counterfactual_memory_size), "counterfactual_coverage": self.last_counterfactual_coverage}

    def belief_update(self) -> dict:
        """Learn P(action | public context, private range class) for Bayesian range updates."""
        with self._lock:
            records = self.action_likelihood_memory.sample(256, self._rng)
        if not records:
            return {"belief_log_loss": 0.0, "belief_action_accuracy": 0.0, "likelihood_memory_size": 0.0}
        histories = [record.history[-12:] if record.history else [record.context] for record in records]
        sequence_length = max(len(history) for history in histories)
        contexts = torch.zeros((len(records), sequence_length, ACTION_CONTEXT_SIZE), dtype=torch.float32)
        final_positions = torch.zeros(len(records), dtype=torch.long)
        for index, history in enumerate(histories):
            contexts[index, :len(history)] = torch.tensor(history, dtype=torch.float32)
            final_positions[index] = len(history) - 1
        range_classes = torch.tensor([record.range_class for record in records], dtype=torch.long)
        actions = torch.tensor([record.action for record in records], dtype=torch.long)
        contexts, final_positions, range_classes, actions = (self._to_training_device(tensor) for tensor in (contexts, final_positions, range_classes, actions))
        losses: list[float] = []
        accuracies: list[float] = []
        self.model.train()
        for _ in range(2):
            with self._autocast():
                logits = self.model.action_likelihood_sequence_logits(contexts)
                selected_logits = logits[torch.arange(len(records), device=self.runtime.device), final_positions, range_classes]
                loss = nn.functional.cross_entropy(selected_logits, actions)
            self._optimizer_step(loss)
            losses.append(float(loss.item()))
            accuracies.append(float((selected_logits.argmax(dim=-1) == actions).float().mean().item()))
        self.model.eval()
        return {"belief_log_loss": sum(losses) / len(losses), "belief_action_accuracy": sum(accuracies) / len(accuracies), "likelihood_memory_size": float(len(self.action_likelihood_memory.records))}

    def self_imitation_update(self) -> dict:
        """Distil winning replay into isolated average-strategy and critic heads."""
        with self._lock:
            records = self.imitation_memory.sample(768, self._rng)
        if not records:
            self.last_imitation_loss = 0.0
            self.last_imitation_reward = 0.0
            return {"imitation_loss": 0.0, "imitation_memory_size": 0.0, "imitation_reward": 0.0}
        observations = torch.tensor([record.observation for record in records], dtype=torch.float32).unsqueeze(1)
        masks = torch.tensor([record.mask for record in records], dtype=torch.bool)
        actions = torch.tensor([record.action for record in records], dtype=torch.long)
        returns = torch.tensor([record.return_value for record in records], dtype=torch.float32)
        priorities = torch.tensor([record.priority for record in records], dtype=torch.float32)
        observations, masks, actions, returns, priorities = (self._to_training_device(tensor) for tensor in (observations, masks, actions, returns, priorities))
        weights = priorities / priorities.mean().clamp_min(1e-6)
        self.model.train()
        with self._autocast():
            logits, _, _ = self.model.detached_cfr_heads(observations)
            values, _ = self.model.detached_critic(observations)
            policy = masked_distribution(logits[:, 0], masks)
            policy_loss = -(weights * policy.log_prob(actions)).mean()
            value_loss = nn.functional.smooth_l1_loss(values[:, 0], (returns / VALUE_RETURN_SCALE_BB).unsqueeze(-1).expand_as(values[:, 0]))
            loss = 0.08 * policy_loss + 0.06 * value_loss
        self._optimizer_step(loss)
        self.model.eval()
        self.last_imitation_loss = float(loss.item())
        self.last_imitation_reward = float(returns.mean().item())
        return {"imitation_loss": self.last_imitation_loss, "imitation_memory_size": float(len(self.imitation_memory.records)), "imitation_reward": self.last_imitation_reward}

    def _update_regrets_locked(self, results: dict[str, MatchResult]) -> None:
        if not results:
            return
        losses = {policy_id: 0.5 - bb_per_100_score(result.bb_per_100) for policy_id, result in results.items()}
        average_loss = sum(losses.values()) / len(losses)
        for policy_id, loss in losses.items():
            previous = self.mixture_regrets.get(policy_id, 0.0)
            self.mixture_regrets[policy_id] = min(12.0, max(-12.0, previous + loss - average_loss))

    def _update_weaknesses_locked(self, model_results: dict[str, MatchResult], style_results: dict[str, MatchResult], holdout_results: dict[str, MatchResult]) -> None:
        """Turn evaluator losses into bounded next-rollout priorities."""
        opponent_scores = {
            **{policy_id: bb_per_100_score(result.bb_per_100) for policy_id, result in model_results.items()},
            **{f"style-{style}": bb_per_100_score(result.bb_per_100) for style, result in style_results.items()},
        }
        for policy_id, score in opponent_scores.items():
            target = min(1.0, max(0.0, 1.0 - score))
            previous = self.opponent_weakness.get(policy_id, 0.50)
            self.opponent_weakness[policy_id] = 0.74 * previous + 0.26 * target
        by_profile = {profile: [] for profile in SCENARIO_PROFILES}
        for name, _, _, profile in HOLDOUT_SCENARIOS:
            if name in holdout_results:
                by_profile[profile].append(bb_per_100_score(holdout_results[name].bb_per_100))
        for profile, scores in by_profile.items():
            if not scores:
                continue
            target = min(1.0, max(0.0, 1.0 - sum(scores) / len(scores)))
            self.scenario_weakness[profile] = 0.74 * self.scenario_weakness.get(profile, 0.50) + 0.26 * target
        self.last_training_focus = max(self.scenario_weakness, key=self.scenario_weakness.get)
        self.last_weakness_score = self.scenario_weakness[self.last_training_focus]

    def _trim_payoff_matrix_locked(self) -> None:
        while len(self.payoff_matrix) > 2_048:
            self.payoff_matrix.pop(next(iter(self.payoff_matrix)))

    def evaluate_and_checkpoint(self, curriculum_stage: int, curriculum_phase: str, force_full: bool = False, final_audit: bool = False, executor: ProcessPoolExecutor | None = None) -> dict:
        """Evaluate a frozen candidate without changing its training workload.

        The jobs below are independent fixed-seed audits. They can therefore run
        concurrently while preserving every deal, hand count, and gate result.
        """
        evaluation_started = perf_counter()
        candidate = clone_state(self.model)
        with self._lock:
            champion = copy.deepcopy(self.champion_state)
            champion_id = self.champion_id
            champion_elo = self.champion_elo
            roster = self._roster_locked()
            audit_interval = RECOVERY_FULL_EVALUATION_INTERVAL if self.recovery_safe_audits < 2 else EVALUATION_INTERVAL
            full_evaluation = force_full or (self.updates > 0 and self.updates % audit_interval == 0)
            frozen = [entry for entry in roster if entry["id"] != champion_id]
            exploiter_entries = [copy.deepcopy(entry) for entry in self.exploiters]
            if full_evaluation:
                representatives = frozen
            else:
                weights = self._mixture_weights_locked(roster)
                representatives = sorted(frozen, key=lambda entry: weights.get(str(entry["id"]), 0.0), reverse=True)[:2]

        scenario_audit_hands = PREFLOP_SCENARIO_AUDIT_HANDS * (PREFLOP_FINAL_AUDIT_MULTIPLIER if final_audit else 1)
        evaluation_jobs = [
            EvaluationJob(
                "direct",
                sequential_evaluate_pair,
                (candidate, champion, PROMOTION_MIN_HANDS if full_evaluation else 64, PROMOTION_MAX_HANDS if full_evaluation else 128, 700_001 + self.updates, curriculum_stage),
            ),
            *[
                EvaluationJob(
                    f"league:{index}",
                    evaluate_pair,
                    (candidate, entry["state"], HOLDOUT_HANDS if full_evaluation else 32, 710_001 + self.updates + index, curriculum_stage),
                )
                for index, entry in enumerate(representatives)
            ],
        ]
        gpu_evaluation_jobs: list[EvaluationJob] = []
        if full_evaluation:
            evaluation_jobs.extend(
                [
                    *[
                        EvaluationJob(f"exploiter:{index}", evaluate_pair, (entry["state"], champion, 24, 705_001 + self.updates + index, curriculum_stage))
                        for index, entry in enumerate(exploiter_entries)
                    ],
                    *[
                        EvaluationJob(f"style:{style}", evaluate_style, (candidate, style, ADVERSARIAL_SCREENING_HANDS, 720_001 + self.updates + index, curriculum_stage))
                        for index, style in enumerate(BENCHMARK_STYLES)
                    ],
                    *[
                        EvaluationJob(f"audit:{style}", evaluate_style, (candidate, style, ADVERSARIAL_SCREENING_HANDS, 730_001 + self.updates + index, curriculum_stage))
                        for index, style in enumerate(AUDIT_STYLES)
                    ],
                    *[
                        EvaluationJob(f"blueprint:{style}", evaluate_style, (candidate, style, HOLDOUT_HANDS, seed, curriculum_stage))
                        for style, seed in BLUEPRINT_AUDITS
                    ],
                    *[
                        EvaluationJob(f"holdout:{name}", evaluate_pair, (candidate, champion, HOLDOUT_HANDS, seed, stage, profile))
                        for name, stage, seed, profile in HOLDOUT_SCENARIOS
                    ],
                    *[
                        EvaluationJob(
                            f"restricted_br:{style}",
                            evaluate_style,
                            (candidate, style, ADVERSARIAL_EVALUATION_HANDS, 740_001 + self.updates + index * 2_003, curriculum_stage),
                        )
                        for index, style in enumerate(ADVERSARIAL_TRAINING_STYLES)
                    ],
                    *[
                        EvaluationJob(
                            f"audit_proxy:{style}",
                            evaluate_style,
                            (candidate, style, ADVERSARIAL_EVALUATION_HANDS, 750_001 + self.updates + index * 3_007, curriculum_stage),
                        )
                        for index, style in enumerate(AUDIT_STYLES)
                    ],
                    EvaluationJob("behavior", behavioral_policy_audit, (candidate, champion, 790_001 + self.updates, curriculum_stage)),
                    EvaluationJob("sizing", preflop_sizing_audit, (candidate, 795_001 + self.updates, curriculum_stage)),
                ]
            )
            preflop_jobs = [
                EvaluationJob(
                    f"preflop:{root}:{style}",
                    evaluate_preflop_scenario_cuda if CUDA_EVALUATION_ENABLED and self.runtime.cuda_enabled else evaluate_preflop_scenario,
                    (candidate, style, root, scenario_audit_hands, 796_001 + self.updates * 97 + root_index * 13 + style_index, curriculum_stage),
                )
                for root_index, root in enumerate(PREFLOP_FORCED_ROOTS)
                for style_index, style in enumerate(("nit", "tight_aggressive"))
            ]
            if CUDA_EVALUATION_ENABLED and self.runtime.cuda_enabled:
                gpu_evaluation_jobs = preflop_jobs
            else:
                evaluation_jobs.extend(preflop_jobs)
        evaluation_results, parallel_evaluation = run_evaluation_jobs(executor, evaluation_jobs, gpu_evaluation_jobs)

        direct = evaluation_results["direct"]
        league_results = {
            str(entry["id"]): evaluation_results[f"league:{index}"]
            for index, entry in enumerate(representatives)
        }
        model_results = {champion_id: direct, **league_results}
        exploiter_ids = {str(entry["id"]) for entry in exploiter_entries}
        exploiter_adversarial_floor = min((result.bb_per_100 for policy_id, result in league_results.items() if policy_id in exploiter_ids), default=0.0)
        candidate_adversarial_floor = exploiter_adversarial_floor
        model_hands = sum(result.hands for result in model_results.values())
        league_score = sum(result.score * result.hands for result in model_results.values()) / max(1, model_hands)
        league_bb_per_100 = sum(result.reward for result in model_results.values()) / max(1, model_hands) * 100
        league_returns = [value for result in model_results.values() for value in result.returns_bb]
        league_result = MatchResult(reward=sum(league_returns), wins=model_hands, returns_bb=league_returns)
        league_lower_bb_per_100, _ = bootstrap_bb_per_100_bounds(league_result, 701_001 + self.updates)
        population_confidence = wilson_lower_bound(league_score, model_hands)
        direct_confidence = wilson_lower_bound(direct.score, direct.hands)
        exploiter_results = {str(entry["id"]): evaluation_results[f"exploiter:{index}"] for index, entry in enumerate(exploiter_entries)} if full_evaluation else {}
        style_results = {style: evaluation_results[f"style:{style}"] for style in BENCHMARK_STYLES} if full_evaluation else {}
        audit_results = {style: evaluation_results[f"audit:{style}"] for style in AUDIT_STYLES} if full_evaluation else {}
        adversarial_results = {
            style: style_results.get(style) or audit_results.get(style)
            for style in ADVERSARIAL_TRAINING_STYLES
            if style_results.get(style) is not None or audit_results.get(style) is not None
        }
        adversarial_style_floor = min((result.bb_per_100 for result in adversarial_results.values()), default=0.0)
        candidate_adversarial_floor = min(candidate_adversarial_floor, adversarial_style_floor)
        adversarial_ci_floor_bb_per_100 = min(
            (
                bootstrap_bb_per_100_bounds(result, 735_001 + self.updates + index)[0]
                for index, result in enumerate(adversarial_results.values())
            ),
            default=self.last_adversarial_ci_floor_bb_per_100,
        )
        blueprint_style_results = {style: evaluation_results[f"blueprint:{style}"] for style, _ in BLUEPRINT_AUDITS} if full_evaluation else {}
        holdout_results = {name: evaluation_results[f"holdout:{name}"] for name, _, _, _ in HOLDOUT_SCENARIOS} if full_evaluation else {}
        restricted_br = min((evaluation_results[f"restricted_br:{style}"].bb_per_100 for style in ADVERSARIAL_TRAINING_STYLES), default=self.last_restricted_br_bb_per_100) if full_evaluation else self.last_restricted_br_bb_per_100
        audit_exploitability = min((evaluation_results[f"audit_proxy:{style}"].bb_per_100 for style in AUDIT_STYLES), default=self.last_audit_exploitability_bb_per_100) if full_evaluation else self.last_audit_exploitability_bb_per_100
        confirmation_hands = 0
        near_confirmation = full_evaluation and direct.bb_per_100 >= -8.0 and candidate_adversarial_floor >= -64.0 and adversarial_ci_floor_bb_per_100 >= -64.0 and restricted_br >= -48.0 and audit_exploitability >= -48.0
        # A final diagnostic must retain every full screening/root audit, but a
        # clearly weak candidate cannot be promoted.  Reserve the costly
        # 2,048-hand/style confirmation for candidates close enough to the
        # promotion floors to make that evidence consequential.
        promotion_confirmation = near_confirmation
        if promotion_confirmation:
            confirmation_jobs = [
                EvaluationJob(f"confirmation:{style}", evaluate_style, (candidate, style, ADVERSARIAL_CONFIRMATION_HANDS, 780_001 + self.updates + index, curriculum_stage))
                for index, style in enumerate(ADVERSARIAL_TRAINING_STYLES)
            ]
            confirmation_results, confirmation_parallel = run_evaluation_jobs(executor, confirmation_jobs)
            parallel_evaluation = parallel_evaluation or confirmation_parallel
            adversarial_results = {style: confirmation_results[f"confirmation:{style}"] for style in ADVERSARIAL_TRAINING_STYLES}
            confirmation_hands = ADVERSARIAL_CONFIRMATION_HANDS
            candidate_adversarial_floor = min(exploiter_adversarial_floor, min((result.bb_per_100 for result in adversarial_results.values()), default=0.0))
            adversarial_ci_floor_bb_per_100 = min(
                (bootstrap_bb_per_100_bounds(result, 785_001 + self.updates + index)[0] for index, result in enumerate(adversarial_results.values())),
                default=adversarial_ci_floor_bb_per_100,
            )
        behavior_audit = evaluation_results["behavior"] if full_evaluation else None
        sizing_audit = evaluation_results["sizing"] if full_evaluation else None
        preflop_scenario_audit = {
            root: {
                style: evaluation_results[f"preflop:{root}:{style}"]
                for style_index, style in enumerate(("nit", "tight_aggressive"))
            }
            for root_index, root in enumerate(PREFLOP_FORCED_ROOTS)
        } if full_evaluation else None
        if final_audit and preflop_scenario_audit is not None:
            holdout_screen_metrics = {
                name: paired_bb_per_100_metrics(result, 760_001 + self.updates + index)
                for index, (name, result) in enumerate(holdout_results.items())
            }
            uncertain_holdouts = sorted(
                (
                    (metrics[1], name, stage, seed, profile)
                    for name, stage, seed, profile in HOLDOUT_SCENARIOS
                    if (metrics := holdout_screen_metrics.get(name)) is not None
                    and holdout_results[name].bb_per_100 >= -30.0
                    and metrics[1] < -18.0
                ),
                key=lambda item: item[0],
            )[:SEQUENTIAL_CONFIRMATION_SCENARIOS]
            uncertain_roots = sorted(
                (
                    (float(metrics["bb_per_100_lower"]), root, style)
                    for root, by_style in preflop_scenario_audit.items()
                    for style, metrics in by_style.items()
                    if float(metrics["bb_per_100"]) >= PREFLOP_ROOT_PROMOTION_LCB_FLOOR - 16.0
                    and float(metrics["bb_per_100_lower"]) < PREFLOP_ROOT_PROMOTION_LCB_FLOOR
                ),
                key=lambda item: item[0],
            )[:SEQUENTIAL_CONFIRMATION_SCENARIOS]
            sequential_jobs = [
                EvaluationJob(
                    f"holdout_confirmation:{name}",
                    evaluate_pair,
                    (candidate, champion, HOLDOUT_CONFIRMATION_HANDS, seed + 4_000_003, stage, profile),
                )
                for _, name, stage, seed, profile in uncertain_holdouts
            ]
            sequential_jobs.extend(
                EvaluationJob(
                    f"preflop_confirmation:{root}:{style}",
                    evaluate_preflop_scenario,
                    (candidate, style, root, PREFLOP_ROOT_CONFIRMATION_HANDS, 8_960_001 + self.updates * 97 + index, curriculum_stage),
                )
                for index, (_, root, style) in enumerate(uncertain_roots)
            )
            if sequential_jobs:
                sequential_results, sequential_parallel = run_evaluation_jobs(executor, sequential_jobs)
                parallel_evaluation = parallel_evaluation or sequential_parallel
                for _, name, _, _, _ in uncertain_holdouts:
                    holdout_results[name] = sequential_results[f"holdout_confirmation:{name}"]
                for _, root, style in uncertain_roots:
                    preflop_scenario_audit[root][style] = sequential_results[f"preflop_confirmation:{root}:{style}"]
        scenario_lcb_records = [
            (float(metrics["bb_per_100_lower"]), root, style)
            for root, by_style in (preflop_scenario_audit or {}).items()
            for style, metrics in by_style.items()
        ]
        scenario_worst_lcb, scenario_worst_root, scenario_worst_style = min(
            scenario_lcb_records,
            default=(self.last_preflop_scenario_worst_lcb_bb_per_100, self.last_preflop_scenario_worst_root, self.last_preflop_scenario_worst_style),
        )
        first_decision_records = [
            metrics.get("first_decision", {})
            for by_style in (preflop_scenario_audit or {}).values()
            for metrics in by_style.values()
        ]
        first_decisions = sum(int(record.get("decisions", 0)) for record in first_decision_records)
        first_all_in_actions = sum(int(record.get("all_in_actions", 0)) for record in first_decision_records)
        first_fold_actions = sum(round(float(dict(record.get("action_mix", {})).get("fold", 0.0)) * int(record.get("decisions", 0))) for record in first_decision_records)
        first_eligible_decisions = sum(int(record.get("calibration_eligible_decisions", 0)) for record in first_decision_records)
        first_calibrated_all_ins = sum(int(record.get("calibrated_all_in_actions", 0)) for record in first_decision_records)
        first_all_in_target = sum(float(record.get("all_in_target", 0.0)) * int(record.get("calibration_eligible_decisions", 0)) for record in first_decision_records) / max(1, first_eligible_decisions)
        first_fold_rate = first_fold_actions / max(1, first_decisions)
        fold_collapse = first_decisions > 0 and first_fold_rate >= PREFLOP_ROOT_FOLD_COLLAPSE_RATE
        holdout_score = sum(result.score for result in holdout_results.values()) / max(1, len(holdout_results))
        holdout_floor = min((result.score for result in holdout_results.values()), default=0.0)
        holdout_bb_per_100 = sum(result.bb_per_100 for result in holdout_results.values()) / max(1, len(holdout_results))
        holdout_floor_bb_per_100 = min((result.bb_per_100 for result in holdout_results.values()), default=0.0)
        holdout_metrics = {
            name: paired_bb_per_100_metrics(result, 760_001 + self.updates + index)
            for index, (name, result) in enumerate(holdout_results.items())
        }
        recovery_holdout_names = [
            name
            for name, stage, _, _ in HOLDOUT_SCENARIOS
            if stage <= curriculum_stage and name in holdout_metrics
        ]
        recovery_holdout_ci_floor_bb_per_100 = min(
            (holdout_metrics[name][1] for name in recovery_holdout_names),
            default=0.0,
        )
        blueprint_style_metrics = {
            style: paired_bb_per_100_metrics(result, 770_001 + self.updates + index)
            for index, (style, result) in enumerate(blueprint_style_results.items())
        }
        holdout_ci_floor_bb_per_100 = min((metrics[1] for metrics in holdout_metrics.values()), default=0.0)
        holdout_paired_variance = sum(metrics[3] for metrics in holdout_metrics.values()) / max(1, len(holdout_metrics))
        audit_score = sum(result.score for result in audit_results.values()) / max(1, len(audit_results))
        direct_lower_bb_per_100, direct_upper_bb_per_100 = bootstrap_bb_per_100_bounds(direct, 702_001 + self.updates)
        blueprint = score_blueprint({
            **{f"holdout-{name}": (metrics[0], metrics[1], holdout_results[name].hands) for name, metrics in holdout_metrics.items()},
            **{f"style-{style}": (metrics[0], metrics[1], blueprint_style_results[style].hands) for style, metrics in blueprint_style_metrics.items()},
        }) if full_evaluation else None
        kuhn_audit = kuhn_cfr_audit() if full_evaluation else None
        evaluation_seconds = perf_counter() - evaluation_started
        with self._lock:
            self.last_evaluation_seconds = evaluation_seconds
            self.last_parallel_evaluation = parallel_evaluation
            self.last_evaluation_hands = direct.hands
            self.last_direct_bb_per_100 = direct.bb_per_100
            self.last_evaluation_bb_per_100 = league_bb_per_100
            self.last_promotion_ci_lower = direct_lower_bb_per_100
            self.last_promotion_ci_upper = direct_upper_bb_per_100
            if exploiter_results:
                threats = {policy_id: min(0.35, max(0.0, result.bb_per_100 / 100.0)) for policy_id, result in exploiter_results.items()}
                self.last_exploiter_threat = max(threats.values())
                self.last_champion_vulnerability = self.last_exploiter_threat
                for entry in self.exploiters:
                    policy_id = str(entry["id"])
                    if policy_id in exploiter_results:
                        entry["champion_score"] = exploiter_results[policy_id].score
                        entry["threat"] = threats[policy_id]
                        self._record_match_payoff_locked(policy_id, champion_id, exploiter_results[policy_id])
            if holdout_results:
                self.last_holdout_score, self.last_holdout_floor = holdout_score, holdout_floor
                self.last_holdout_bb_per_100, self.last_holdout_floor_bb_per_100 = holdout_bb_per_100, holdout_floor_bb_per_100
                self.last_holdout_ci_floor_bb_per_100 = holdout_ci_floor_bb_per_100
                self.last_holdout_paired_variance = holdout_paired_variance
                self.last_restricted_br_bb_per_100 = restricted_br
                self.last_crossplay_robustness = league_score
            if adversarial_results:
                self.last_adversarial_ci_floor_bb_per_100 = adversarial_ci_floor_bb_per_100
                self.last_adversarial_evaluation_hands = confirmation_hands or ADVERSARIAL_SCREENING_HANDS
                self.last_adversarial_confirmation_hands = confirmation_hands
            self.last_final_audit_ran = final_audit
            for style in ADVERSARIAL_TRAINING_STYLES:
                result = adversarial_results.get(style)
                if result is not None:
                    self.adversarial_style_bb_per_100[style] = result.bb_per_100
            self.last_adversarial_focus = " · ".join(self._focused_adversarial_styles_locked())
            if behavior_audit is not None:
                self.last_behavior_action_agreement = float(behavior_audit["action_agreement"])
                self.last_behavior_action_change_rate = float(behavior_audit["action_change_rate"])
                self.last_behavior_raise_fraction_delta = float(behavior_audit["raise_fraction_delta"])
                self.last_behavior_audit_states = int(behavior_audit["states"])
            if sizing_audit is not None:
                self.last_preflop_sizing_audit = sizing_audit
            if preflop_scenario_audit is not None:
                self.last_preflop_scenario_audit = preflop_scenario_audit
                self.last_preflop_scenario_audit_hands = scenario_audit_hands
                self.last_preflop_scenario_worst_lcb_bb_per_100 = scenario_worst_lcb
                self.last_preflop_scenario_worst_root = scenario_worst_root
                self.last_preflop_scenario_worst_style = scenario_worst_style
                self.last_preflop_immediate_allin_rate = first_all_in_actions / max(1, first_decisions)
                self.last_preflop_immediate_allin_target = first_all_in_target
                self.last_preflop_immediate_eligible_rate = first_eligible_decisions / max(1, first_decisions)
                root_floors = {
                    root: min(float(metrics["bb_per_100_lower"]) for metrics in by_style.values())
                    for root, by_style in preflop_scenario_audit.items()
                }
                strongest_root_floor = max(root_floors.values(), default=0.0)
                weakest_root_floor = min(root_floors.values(), default=0.0)
                root_floor_spread = max(40.0, strongest_root_floor - weakest_root_floor)
                for root, root_floor in root_floors.items():
                    # Rank roots relative to the current audit instead of clipping every
                    # negative LCB to the same maximum weakness. This keeps the worst
                    # root (currently facing_4bet) meaningfully over-sampled.
                    relative_severity = min(1.0, max(0.0, (strongest_root_floor - root_floor) / root_floor_spread))
                    target = 0.25 + 0.75 * relative_severity
                    smoothed = 0.74 * self.preflop_root_weakness.get(root, 0.50) + 0.26 * target
                    if root == scenario_worst_root and root_floor < PREFLOP_ROOT_PROMOTION_LCB_FLOOR:
                        severity = min(1.0, max(0.0, (PREFLOP_ROOT_PROMOTION_LCB_FLOOR - root_floor) / 400.0))
                        smoothed = max(smoothed, 0.82 + 0.18 * severity)
                    self.preflop_root_weakness[root] = smoothed
            active_member = self.population_members[self.active_population_index]
            active_member["state"] = candidate
            active_member["score"] = league_score
            active_member["bb_per_100"] = league_bb_per_100
            if style_results:
                active_member["adversarial_bb_per_100"] = candidate_adversarial_floor
                self.last_adversarial_floor_bb_per_100 = candidate_adversarial_floor
            active_member["preflop_worst_lcb_bb_per_100"] = self.last_preflop_scenario_worst_lcb_bb_per_100
            active_member["preflop_allin_probability"] = self.last_preflop_guarded_allin_probability
            if full_evaluation:
                if final_audit:
                    self.last_final_audit_checkpoint_restored = False
                    self.last_final_audit_restore_reason = ""
                recovery_metrics = {
                    "direct_lcb": direct_lower_bb_per_100,
                    "adversarial_lcb": adversarial_ci_floor_bb_per_100,
                    "preflop_lcb": self.last_preflop_scenario_worst_lcb_bb_per_100,
                    # Recovery safety is stage-matched: deeper holdouts remain
                    # mandatory promotion gates, but cannot prevent a safe
                    # foundation model from training toward those stages.
                    "holdout_lcb": recovery_holdout_ci_floor_bb_per_100,
                    "all_in_probability": self.last_preflop_guarded_allin_probability,
                    "all_in_target": self.last_preflop_allin_target,
                    "first_fold_rate": first_fold_rate,
                    "first_all_in_rate": first_all_in_actions / max(1, first_decisions),
                    "behavior_degeneracy": population_behavior_degeneracy({
                        "fold_rate": first_fold_rate,
                        "all_in_rate": first_all_in_actions / max(1, first_decisions),
                    }),
                    "fold_collapse": float(fold_collapse),
                }
                active_member["behavior_fold_rate"] = recovery_metrics["first_fold_rate"]
                active_member["behavior_all_in_rate"] = recovery_metrics["first_all_in_rate"]
                active_member["behavior_degeneracy"] = recovery_metrics["behavior_degeneracy"]
                self.last_recovery_candidate_metrics = dict(recovery_metrics)
                recovery_score = self._recovery_safety_score(
                    recovery_metrics["adversarial_lcb"],
                    recovery_metrics["preflop_lcb"],
                    recovery_metrics["holdout_lcb"],
                    recovery_metrics["all_in_probability"],
                    recovery_metrics["all_in_target"],
                )
                self.recovery_safe_audits = self.recovery_safe_audits + 1 if self._recovery_candidate_is_safe(recovery_metrics) else 0
                anchored = self._consider_recovery_anchor_locked(candidate, recovery_score, recovery_metrics)
                if fold_collapse:
                    self.recovery_safe_audits = 0
                    if self._has_verified_recovery_anchor_locked():
                        rollback_event = self._quarantine_active_population_member_locked([
                            f"immediate fixed-root fold collapse ({first_fold_rate:.1%} first-decision folds)",
                        ])
                        log_training_debug("population_member_fold_collapse_rollback", **rollback_event, recovery_metrics=recovery_metrics)
                    else:
                        self.last_fresh_warmup_fold_collapse = True
                        active_member["recovery_cooldown_until"] = self.updates + RECOVERY_ANCHOR_COOLDOWN_UPDATES
                        active_member["last_quarantine_reason"] = f"fixed-root fold collapse ({first_fold_rate:.1%}) without a verified anchor"
                        self.last_challenger_status = "collapsed member quarantined; safe anchor unavailable"
                        log_training_debug(
                            "unanchored_population_member_fold_collapse",
                            member=str(active_member["id"]),
                            first_fold_rate=round(first_fold_rate, 6),
                            recovery_anchor_source=self.recovery_anchor_source,
                            recovery_metrics=recovery_metrics,
                        )
                elif not anchored and self._has_verified_recovery_anchor_locked() and self.updates >= int(active_member.get("recovery_cooldown_until", 0)) and recovery_score < self.recovery_anchor_score - RECOVERY_ANCHOR_REGRESSION_MARGIN:
                    active_member["safety_regressions"] = int(active_member.get("safety_regressions", 0)) + 1
                    catastrophic_regression = (
                        recovery_metrics["direct_lcb"] < -25.0
                        or recovery_metrics["adversarial_lcb"] < -220.0
                        or recovery_metrics["holdout_lcb"] < -220.0
                        or recovery_metrics["preflop_lcb"] < -450.0
                        or recovery_metrics["behavior_degeneracy"] > 0.0
                    )
                    if catastrophic_regression or self.recovery_regression_requires_quarantine(final_audit, int(active_member["safety_regressions"])):
                        reason = (
                            f"final-audit anchor regression (score={recovery_score:.2f}, anchor={self.recovery_anchor_score:.2f})"
                            if final_audit
                            else f"immediate catastrophic anchor regression (score={recovery_score:.2f}, anchor={self.recovery_anchor_score:.2f})"
                            if catastrophic_regression
                            else f"two consecutive anchor regressions (score={recovery_score:.2f}, anchor={self.recovery_anchor_score:.2f})"
                        )
                        quarantine_event = self._quarantine_active_population_member_locked([
                            reason,
                        ])
                        if final_audit:
                            self.last_final_audit_checkpoint_restored = True
                            self.last_final_audit_restore_reason = reason
                        log_training_debug("population_member_recovery_rollback", **quarantine_event, recovery_metrics=recovery_metrics)
                else:
                    active_member["safety_regressions"] = 0
            challenger_id = f"challenger-{self.updates}"
            for policy_id, result in model_results.items():
                self._record_match_payoff_locked(challenger_id, policy_id, result)
            for style, result in style_results.items():
                self._record_match_payoff_locked(challenger_id, f"style-{style}", result)
            self._trim_payoff_matrix_locked()
            regret_results = {**model_results, **{f"style-{style}": result for style, result in style_results.items()}}
            self._update_regrets_locked(regret_results)
            self._update_weaknesses_locked(model_results, style_results, holdout_results)
            for style, result in audit_results.items():
                policy_id = f"style-{style}"
                target = min(1.0, max(0.0, 1.0 - bb_per_100_score(result.bb_per_100)))
                self.opponent_weakness[policy_id] = 0.74 * self.opponent_weakness.get(policy_id, 0.50) + 0.26 * target
            if style_results:
                self.tournament_count += 1
                self.benchmarks = {
                    style: {"win_rate": result.score, "bb_per_100": result.reward / max(1, result.hands) * 100}
                    for style, result in style_results.items()
                }
            if audit_results:
                self.audit_benchmarks = {
                    style: {"win_rate": result.score, "bb_per_100": result.reward / max(1, result.hands) * 100}
                    for style, result in audit_results.items()
                }
                self.last_audit_score = audit_score
                self.last_audit_exploitability_bb_per_100 = audit_exploitability
                composite = (
                    0.28 * bb_per_100_quality(direct_lower_bb_per_100, -60.0, 0.0)
                    + 0.28 * bb_per_100_quality(adversarial_ci_floor_bb_per_100, -80.0, -5.0)
                    + 0.24 * bb_per_100_quality(restricted_br, -80.0, -5.0)
                    + 0.20 * bb_per_100_quality(audit_exploitability, -80.0, -5.0)
                )
                previous = float(self.evaluation_history[-1].get("composite", composite)) if self.evaluation_history else composite
                self.last_ablation_delta = composite - previous
                self.evaluation_history = [*self.evaluation_history, {"update": self.updates, "direct": direct.score, "direct_bb_per_100": direct.bb_per_100, "direct_ci_lower_bb_per_100": direct_lower_bb_per_100, "league": league_score, "league_bb_per_100": league_bb_per_100, "holdout": holdout_score, "holdout_bb_per_100": holdout_bb_per_100, "holdout_ci_floor_bb_per_100": holdout_ci_floor_bb_per_100, "holdout_paired_variance": holdout_paired_variance, "adversarial_floor_bb_per_100": candidate_adversarial_floor, "adversarial_ci_floor_bb_per_100": adversarial_ci_floor_bb_per_100, "audit": audit_score, "restricted_br": restricted_br, "audit_proxy": audit_exploitability, "blueprint": blueprint.score if blueprint else self.last_blueprint_score, "blueprint_lower": blueprint.lower_confidence if blueprint else self.last_blueprint_confidence, "blueprint_floor": blueprint.floor if blueprint else self.last_blueprint_floor, "composite": composite}][-24:]
            if audit_results and self.evaluation_history:
                self.evaluation_history[-1].update({
                    "adversarial_hands_per_style": confirmation_hands or ADVERSARIAL_SCREENING_HANDS,
                    "final_audit": final_audit,
                    "behavior_action_agreement": self.last_behavior_action_agreement,
                    "behavior_action_change_rate": self.last_behavior_action_change_rate,
                    "preflop_mean_raise_bb": float(self.last_preflop_sizing_audit["mean_raise_bb"]),
                    "preflop_oversized_open_rate": float(self.last_preflop_sizing_audit["oversized_open_rate"]),
                    "preflop_three_bet_p95_raise_to_pot": float(self.last_preflop_sizing_audit["three_bet_p95_raise_to_pot"]),
                    "preflop_three_bet_over_cap_rate": float(self.last_preflop_sizing_audit["three_bet_over_cap_rate"]),
                    "preflop_scenario_worst_lcb_bb_per_100": self.last_preflop_scenario_worst_lcb_bb_per_100,
                    "preflop_scenario_worst_root": self.last_preflop_scenario_worst_root,
                    "preflop_scenario_worst_style": self.last_preflop_scenario_worst_style,
                    "preflop_immediate_allin_rate": self.last_preflop_immediate_allin_rate,
                    "preflop_immediate_allin_target": self.last_preflop_immediate_allin_target,
                    "preflop_immediate_eligible_rate": self.last_preflop_immediate_eligible_rate,
                    "preflop_immediate_calibrated_allin_rate": first_calibrated_all_ins / max(1, first_eligible_decisions),
                    "preflop_3bet_teacher_samples": self.last_preflop_3bet_teacher_samples,
                    "preflop_3bet_teacher_coverage": self.last_preflop_3bet_teacher_coverage,
                    "preflop_3bet_teacher_raise_target": self.last_preflop_3bet_teacher_raise_target,
                    "preflop_3bet_teacher_actual_raise_rate": self.last_preflop_3bet_teacher_actual_raise_rate,
                    "preflop_3bet_teacher_allin_target": self.last_preflop_3bet_teacher_allin_target,
                    "preflop_3bet_teacher_actual_allin_rate": self.last_preflop_3bet_teacher_actual_allin_rate,
                })
            if blueprint is not None and kuhn_audit is not None:
                self.last_blueprint_score = blueprint.score
                self.last_blueprint_confidence = blueprint.lower_confidence
                self.last_blueprint_floor = blueprint.floor
                self.last_blueprint_hands = blueprint.hands
                self.last_kuhn_value_gap = kuhn_audit.value_gap
                self.last_blueprint_status = "verified" if blueprint.lower_confidence >= BLUEPRINT_PROMOTION_CONFIDENCE and blueprint.floor >= BLUEPRINT_PROMOTION_FLOOR and kuhn_audit.value_gap <= 0.14 else "held by fixed audit"
            self.last_promotion_confidence = min(direct_confidence, population_confidence)
            adversarial_floor = max(-12.0, -100.0 * self.last_champion_vulnerability - 3.0)
            self.last_gate_passed = full_evaluation and confirmation_hands >= ADVERSARIAL_CONFIRMATION_HANDS and direct.hands >= PROMOTION_MIN_HANDS and direct_lower_bb_per_100 >= 0.0 and league_lower_bb_per_100 >= 0.0 and holdout_floor_bb_per_100 >= -8.0 and holdout_ci_floor_bb_per_100 >= -18.0 and restricted_br >= -32.0 and audit_exploitability >= -34.0 and candidate_adversarial_floor >= adversarial_floor and adversarial_ci_floor_bb_per_100 >= ADVERSARIAL_PROMOTION_LCB_FLOOR and self.last_preflop_scenario_worst_lcb_bb_per_100 >= PREFLOP_ROOT_PROMOTION_LCB_FLOOR and blueprint is not None and blueprint.lower_confidence >= BLUEPRINT_PROMOTION_CONFIDENCE and blueprint.floor >= BLUEPRINT_PROMOTION_FLOOR and self.last_kuhn_value_gap <= 0.14
            if full_evaluation and not self.last_gate_passed:
                self._retain_specialist_locked(
                    candidate,
                    {
                        "direct_lcb": direct_lower_bb_per_100,
                        "adversarial_lcb": adversarial_ci_floor_bb_per_100,
                        "preflop_lcb": self.last_preflop_scenario_worst_lcb_bb_per_100,
                        "holdout_lcb": holdout_ci_floor_bb_per_100,
                        "restricted_br": restricted_br,
                        "audit_proxy": audit_exploitability,
                        "behavior_degeneracy": population_behavior_degeneracy({"fold_rate": first_fold_rate, "all_in_rate": first_all_in_actions / max(1, first_decisions)}),
                        "fold_collapse": float(fold_collapse),
                    },
                    "promotion gates not jointly satisfied",
                )
            self.last_challenger_status = "screening" if not full_evaluation else "promoted" if self.last_gate_passed else "confidence held"
            profile_signature = [bb_per_100_score(direct.bb_per_100), *(bb_per_100_score(style_results[style].bb_per_100) for style in BENCHMARK_STYLES if style in style_results)]
            retained_signatures = [entry.get("signature", []) for entry in self.exploiters]
            profile_distance = [sum(abs(left - right) for left, right in zip(profile_signature, signature)) / max(1, len(profile_signature)) for signature in retained_signatures if len(signature) == len(profile_signature)]
            diverse_exploiter = not profile_distance or min(profile_distance) >= 0.025
            if full_evaluation and not self.last_gate_passed and not self.last_final_audit_checkpoint_restored and direct.bb_per_100 >= 0.0 and direct_lower_bb_per_100 >= -12.0 and diverse_exploiter:
                exploiter_id = f"exploiter-{self.updates}"
                threat = min(0.35, max(0.0, direct.bb_per_100 / 100.0))
                retained_exploiter = {"id": exploiter_id, "state": candidate, "elo": champion_elo + 32 * (direct.score - 0.5), "games": direct.hands, "kind": "exploiter", "signature": profile_signature, "focus_styles": list(self._focused_adversarial_styles_locked()), "champion_score": direct.score, "threat": threat, "generation": self.exploiter_generations}
                self.exploiters = retain_unique_entries_by_id([*self.exploiters, retained_exploiter], 4)
                self.mixture_regrets[exploiter_id] = max(0.15, self.mixture_regrets.get(exploiter_id, 0.0))
                self._record_match_payoff_locked(exploiter_id, champion_id, direct)
                self.last_best_response_bb_per_100 = direct.reward / max(1, direct.hands) * 100
                self.last_exploiter_threat = max(self.last_exploiter_threat, threat)
                self.last_champion_vulnerability = max(self.last_champion_vulnerability, threat)
                self.last_challenger_status = "exploiter retained"
            if self.last_final_audit_checkpoint_restored:
                self.last_challenger_status = "final audit failed; checkpoint restored"
            if self.last_gate_passed:
                expected = 0.5
                candidate_elo = champion_elo + 32 * (direct.score - expected)
                previous_elo = champion_elo - 32 * (direct.score - expected)
                old_champion = self._champion_entry_locked()
                self.version += 1
                self.champion_id = f"champion-{self.version}"
                self.champion_state = candidate
                self.champion_elo = candidate_elo
                old_champion["elo"] = previous_elo
                old_champion["games"] = direct.hands
                self.league = [*self.league, old_champion][-9:]
                self.mixture_regrets.setdefault(self.champion_id, 0.0)
                for policy_id, result in model_results.items():
                    self._record_match_payoff_locked(self.champion_id, policy_id, result)
                for style, result in style_results.items():
                    self._record_match_payoff_locked(self.champion_id, f"style-{style}", result)
                self._publish_locked()
            mixed_entries = self._roster_locked()
            mixed_entries.extend({"id": f"style-{style}", "kind": "style", "style": style, "adversarial": style in ADVERSARIAL_TRAINING_STYLES} for style in dict.fromkeys((*BENCHMARK_STYLES, *ADVERSARIAL_TRAINING_STYLES)))
            mixed_weights = self._mixture_weights_locked(mixed_entries)
            self.last_opponent_pressure = sum(
                mixed_weights[str(entry["id"])] * max(0.0, 0.5 - self._payoff_locked(self.champion_id, str(entry["id"])))
                for entry in mixed_entries
            )
            signatures = [entry.get("signature", []) for entry in self.exploiters]
            signature_distances = [
                sum(abs(left - right) for left, right in zip(first, second)) / len(first)
                for index, first in enumerate(signatures)
                for second in signatures[index + 1:]
                if first and len(first) == len(second)
            ]
            self.last_exploiter_diversity = min(1.0, sum(signature_distances) / max(1, len(signature_distances)) / 0.12)
            self.last_population_continuity = sum(min(1.0, int(member.get("updates", 0)) / max(1, self.updates)) for member in self.population_members) / len(self.population_members)
            self._refresh_curriculum_gate_locked(full_evaluation)
            summary = self.summary_locked(curriculum_stage, curriculum_phase, direct.score, league_score)
            summary.update({
                "recovery_halted": self.recovery_halted,
                "ppo_post_step_retry_applied": self.last_ppo_post_step_retry_applied,
                "ppo_post_step_retry_accepted": self.last_ppo_post_step_retry_accepted,
                "ppo_post_step_retry_kl": self.last_ppo_post_step_retry_kl,
                "preflop_root_guarded": self.last_preflop_root_guarded,
                "preflop_root_guard_reason": self.last_preflop_root_guard_reason,
                "preflop_root_update_kl": self.last_preflop_root_update_kl,
                "preflop_root_anchor_kl": self.last_preflop_root_anchor_kl,
                "preflop_root_update_action_delta": self.last_preflop_root_update_action_delta,
                "preflop_root_anchor_action_delta": self.last_preflop_root_anchor_action_delta,
                "preflop_root_drift_root": self.last_preflop_root_drift_root,
                "adversarial_evaluation_hands": self.last_adversarial_evaluation_hands,
                "adversarial_confirmation_hands": self.last_adversarial_confirmation_hands,
                "final_audit_ran": self.last_final_audit_ran,
                "tail_loss_rate": self.last_tail_loss_rate,
                "tail_loss_bb": self.last_tail_loss_bb,
                "tail_policy_weight": self.last_tail_policy_weight,
                "hard_spot_value_loss": self.last_hard_spot_value_loss,
                "hard_spot_memory_size": self.last_hard_spot_memory_size,
                "behavior_action_agreement": self.last_behavior_action_agreement,
                "behavior_action_change_rate": self.last_behavior_action_change_rate,
                "behavior_raise_fraction_delta": self.last_behavior_raise_fraction_delta,
                "behavior_audit_states": self.last_behavior_audit_states,
                "adversarial_rollout_run_fraction": self.run_adversarial_hands / max(1, self.run_rollout_hands),
                "tail_loss_run_rate": self.run_tail_paths / max(1, self.run_adversarial_paths),
                "tail_loss_run_bb": self.run_tail_loss_sum / max(1, self.run_tail_paths),
                "tail_policy_run_weight": self.run_tail_weight_sum / max(1, self.run_tail_paths),
                "preflop_open_cap_bb": PREFLOP_OPEN_RAISE_CAP_BB,
                "preflop_sizing_audit_roots": int(self.last_preflop_sizing_audit["roots"]),
                "preflop_normal_raise_rate": float(self.last_preflop_sizing_audit["normal_raise_rate"]),
                "preflop_mean_raise_bb": float(self.last_preflop_sizing_audit["mean_raise_bb"]),
                "preflop_p95_raise_bb": float(self.last_preflop_sizing_audit["p95_raise_bb"]),
                "preflop_oversized_open_rate": float(self.last_preflop_sizing_audit["oversized_open_rate"]),
                "preflop_cap_hit_rate": float(self.last_preflop_sizing_audit["cap_hit_rate"]),
                "preflop_all_in_rate": float(self.last_preflop_sizing_audit["all_in_rate"]),
                "preflop_three_bet_cap_pot_multiplier": PREFLOP_THREE_BET_POT_CAP_MULTIPLIER,
                "preflop_three_bet_audit_roots": int(self.last_preflop_sizing_audit["three_bet_roots"]),
                "preflop_three_bet_normal_raise_rate": float(self.last_preflop_sizing_audit["three_bet_normal_raise_rate"]),
                "preflop_three_bet_mean_raise_to_pot": float(self.last_preflop_sizing_audit["three_bet_mean_raise_to_pot"]),
                "preflop_three_bet_p95_raise_to_pot": float(self.last_preflop_sizing_audit["three_bet_p95_raise_to_pot"]),
                "preflop_three_bet_cap_hit_rate": float(self.last_preflop_sizing_audit["three_bet_cap_hit_rate"]),
                "preflop_three_bet_over_cap_rate": float(self.last_preflop_sizing_audit["three_bet_over_cap_rate"]),
                "preflop_three_bet_minimum_override_rate": float(self.last_preflop_sizing_audit["three_bet_minimum_override_rate"]),
                "preflop_three_bet_all_in_rate": float(self.last_preflop_sizing_audit["three_bet_all_in_rate"]),
                "preflop_forced_root_fraction": self.last_preflop_root_fraction,
                "preflop_scenario_audit_hands": self.last_preflop_scenario_audit_hands,
                "preflop_scenario_worst_lcb_bb_per_100": self.last_preflop_scenario_worst_lcb_bb_per_100,
                "preflop_scenario_worst_root": self.last_preflop_scenario_worst_root,
                "preflop_scenario_worst_style": self.last_preflop_scenario_worst_style,
                "preflop_allin_calibration_loss": self.last_preflop_allin_calibration_loss,
                "preflop_allin_stability_loss": self.last_preflop_allin_stability_loss,
                "preflop_guarded_allin_probability": self.last_preflop_guarded_allin_probability,
                "preflop_allin_target": self.last_preflop_allin_target,
                "preflop_guarded_state_fraction": self.last_preflop_guarded_state_fraction,
                "preflop_immediate_allin_rate": self.last_preflop_immediate_allin_rate,
                "preflop_immediate_allin_target": self.last_preflop_immediate_allin_target,
                "preflop_immediate_eligible_rate": self.last_preflop_immediate_eligible_rate,
                "preflop_root_promotion_lcb_floor": PREFLOP_ROOT_PROMOTION_LCB_FLOOR,
                "preflop_3bet_teacher_loss": self.last_preflop_3bet_teacher_loss,
                "preflop_3bet_teacher_eligible_roots": self.last_preflop_3bet_teacher_eligible_roots,
                "preflop_3bet_teacher_samples": self.last_preflop_3bet_teacher_samples,
                "preflop_3bet_teacher_coverage": self.last_preflop_3bet_teacher_coverage,
                "preflop_3bet_teacher_confidence": self.last_preflop_3bet_teacher_confidence,
                "preflop_3bet_teacher_raise_target": self.last_preflop_3bet_teacher_raise_target,
                "preflop_3bet_teacher_raise_advantage_bb": self.last_preflop_3bet_teacher_raise_advantage_bb,
                "preflop_3bet_teacher_actual_raise_rate": self.last_preflop_3bet_teacher_actual_raise_rate,
                "preflop_3bet_teacher_allin_target": self.last_preflop_3bet_teacher_allin_target,
                "preflop_3bet_teacher_actual_allin_rate": self.last_preflop_3bet_teacher_actual_allin_rate,
                "preflop_3bet_teacher_multi_raise_samples": self.last_preflop_3bet_teacher_multi_raise_samples,
                "preflop_3bet_teacher_multi_raise_allin_target": self.last_preflop_3bet_teacher_multi_raise_allin_target,
                "preflop_3bet_teacher_multi_raise_actual_allin_rate": self.last_preflop_3bet_teacher_multi_raise_actual_allin_rate,
                "preflop_3bet_teacher_multi_raise_allin_vetoes": self.last_preflop_3bet_teacher_multi_raise_allin_vetoes,
                "robust_policy_weight": self.last_robust_policy_weight,
                "rollout_inference_device": self.last_rollout_inference_device,
            })
            return summary

    def summary_locked(self, curriculum_stage: int, curriculum_phase: str, direct_score: float, league_score: float) -> dict:
        entries = self._roster_locked()
        entries.extend({"id": f"style-{style}", "kind": "style", "style": style} for style in BENCHMARK_STYLES)
        weights = list(self._mixture_weights_locked(entries).values())
        diversity = -sum(weight * math.log(weight + 1e-12) for weight in weights) / math.log(max(2, len(weights)))
        benchmark_leader = max(self.benchmarks, key=lambda style: self.benchmarks[style]["bb_per_100"], default="pending")
        average_bb = sum(metrics["bb_per_100"] for metrics in self.benchmarks.values()) / max(1, len(self.benchmarks))
        population_diversity = sum(abs(float(left["lr_scale"]) - float(right["lr_scale"])) + abs(float(left["entropy_scale"]) - float(right["entropy_scale"])) for index, left in enumerate(self.population_members) for right in self.population_members[index + 1:]) / 6
        restricted_risk = min(1.0, max(0.0, -self.last_restricted_br_bb_per_100 / 40.0))
        return {"league_size": len(self._roster_locked()), "updates": self.updates, "champion_version": self.version, "champion_elo": self.champion_elo, "evaluation_win_rate": league_score, "direct_champion_score": direct_score, "direct_bb_per_100": self.last_direct_bb_per_100, "evaluation_bb_per_100": self.last_evaluation_bb_per_100, "promotion_ci_lower_bb_per_100": self.last_promotion_ci_lower, "promotion_ci_upper_bb_per_100": self.last_promotion_ci_upper, "gate_passed": self.last_gate_passed, "curriculum_stage": self.curriculum_unlocked_stage, "curriculum_phase": CURRICULUM_PHASES[self.curriculum_unlocked_stage][2], "curriculum_readiness": self.last_curriculum_readiness, "training_lane": self.last_training_lane, "population_size": len(self.population_members), "active_population": str(self.population_members[self.active_population_index]["id"]), "population_diversity": population_diversity, "population_continuity": self.last_population_continuity, "snapshot_count": len(self.strategy_snapshots), "snapshot_diversity": self.last_snapshot_diversity, "snapshot_min_distance": self.last_snapshot_min_distance, "snapshot_rejections": self.snapshot_rejections, "holdout_score": self.last_holdout_score, "holdout_floor": self.last_holdout_floor, "holdout_bb_per_100": self.last_holdout_bb_per_100, "holdout_floor_bb_per_100": self.last_holdout_floor_bb_per_100, "holdout_ci_floor_bb_per_100": self.last_holdout_ci_floor_bb_per_100, "holdout_paired_variance": self.last_holdout_paired_variance, "continuous_raise_mean": self.last_continuous_raise_mean, "scenario_coverage": self.last_scenario_coverage, "mixture_diversity": diversity, "opponent_pressure": self.last_opponent_pressure, "exploitability_proxy": restricted_risk, "restricted_br_bb_per_100": self.last_restricted_br_bb_per_100, "adversarial_floor_bb_per_100": self.last_adversarial_floor_bb_per_100, "adversarial_ci_floor_bb_per_100": self.last_adversarial_ci_floor_bb_per_100, "adversarial_evaluation_hands": ADVERSARIAL_EVALUATION_HANDS, "adversarial_rollout_fraction": self.last_adversarial_rollout_fraction, "adversarial_rollout_target": ADVERSARIAL_ROLLOUT_FRACTION, "crossplay_robustness": self.last_crossplay_robustness, "best_response_bb_per_100": self.last_best_response_bb_per_100, "exploiter_diversity": self.last_exploiter_diversity, "exploiter_threat": self.last_exploiter_threat, "champion_vulnerability": self.last_champion_vulnerability, "exploiter_generations": self.exploiter_generations, "promotion_confidence": self.last_promotion_confidence, "challenger_status": self.last_challenger_status, "evaluation_hands": self.last_evaluation_hands, "evaluation_seconds": self.last_evaluation_seconds, "parallel_evaluation": self.last_parallel_evaluation, "evaluation_workers": EVALUATION_WORKERS, "resolver_uses": self.live_agent.resolver_uses, "resolver_depth": self.live_agent.resolver_depth, "search_leaf_evaluations": self.live_agent.search_leaf_evaluations, "search_value_spread": self.live_agent.search_value_spread, "search_confidence": self.live_agent.search_confidence, "adaptive_action_width": self.live_agent.search_action_width, "endgame_worlds": self.live_agent.search_endgame_worlds, "resolver_safety_rejections": self.live_agent.search_safety_rejections, "resolver_safety_margin": self.live_agent.search_safety_margin, "resolver_safety_confidence": self.live_agent.search_safety_confidence, "resolver_confident_actions": self.live_agent.search_confident_actions, "resolver_iterations": self.live_agent.search_iterations, "resolver_strategy_peak": self.live_agent.search_strategy_peak, "public_belief_teacher_size": self.last_public_belief_teacher_size, "sizing_proposal_diversity": self.last_sizing_proposal_diversity, "rare_spot_rate": self.last_rare_spot_rate, "replay_rare_fraction": self.last_replay_rare_fraction, "replay_priority": self.last_replay_priority, "replay_recent_fraction": self.last_replay_recent_fraction, "belief_confidence": self.last_belief_confidence, "belief_posterior_support": self.last_belief_posterior_support, "resolver_replay_confidence": self.last_resolver_replay_confidence, "resolver_replay_size": self.last_resolver_replay_size, "leaf_evaluations": self.last_leaf_evaluations, "cfr_memory_size": len(self.cfr_memory.records), "strategy_memory_size": len(self.strategy_memory.records), "search_memory_size": self.last_search_memory_size, "counterfactual_memory_size": self.last_counterfactual_memory_size, "counterfactual_coverage": self.last_counterfactual_coverage, "sizing_cfr_loss": self.last_sizing_cfr_loss, "search_value_loss": self.last_search_value_loss, "ensemble_disagreement": self.last_ensemble_disagreement, "likelihood_memory_size": len(self.action_likelihood_memory.records), "imitation_memory_size": len(self.imitation_memory.records), "imitation_loss": self.last_imitation_loss, "imitation_reward": self.last_imitation_reward, "target_network_drift": self.last_target_drift, "ppo_learning_rate": self.ppo_learning_rate, "ppo_clip_epsilon": self.ppo_clip_epsilon, "ppo_entropy_coefficient": self.ppo_entropy_coefficient, "ppo_kl_target": self.ppo_kl_target, "ppo_epochs": self.last_ppo_epochs, "ppo_clip_fraction": self.last_ppo_clip_fraction, "mixed_precision_enabled": self.runtime.cuda_enabled, "gradient_scale": float(self.grad_scaler.get_scale()), "resumed": self.resumed, "benchmark_leader": benchmark_leader, "benchmark_bb_per_100": average_bb, "tournament_count": self.tournament_count, "training_focus": self.last_training_focus, "weakness_score": self.last_weakness_score, "adaptive_workers": self.last_adaptive_workers, "adaptive_batch_hands": self.last_adaptive_batch_hands, "rollout_decisions_per_second": self.last_rollout_decisions_per_second, "teacher_data_records": self.teacher_data_records, "teacher_data_filename": self.teacher_data_report.filename, "teacher_data_status": self.teacher_data_report.message, "audit_score": self.last_audit_score, "audit_exploitability_bb_per_100": self.last_audit_exploitability_bb_per_100, "scenario_gate": self.last_scenario_gate, "ablation_delta": self.last_ablation_delta, "evaluation_history_size": len(self.evaluation_history), "subgame_policy_loss": self.last_subgame_policy_loss, "subgame_teacher_size": self.last_subgame_teacher_size, "rollout_arena_width": self.last_rollout_arena_width, "average_strategy_weight": self.last_average_strategy_weight, "oracle_policy_loss": self.last_oracle_policy_loss, "oracle_value_loss": self.last_oracle_value_loss, "oracle_teacher_size": len(self.abstract_teacher_memory.records), "oracle_confidence": self.last_oracle_confidence, "oracle_iterations": self.last_oracle_iterations, "abstraction_nash_conv": self.last_abstraction_nash_conv, "abstraction_value": self.last_abstraction_value, "abstraction_information_sets": self.last_abstraction_information_sets, "paired_deal_coverage": self.last_paired_deal_coverage, "rollout_backend": active_rollout_capabilities().mode, "rollout_backend_reason": active_rollout_capabilities().reason, "blueprint_score": self.last_blueprint_score, "blueprint_confidence": self.last_blueprint_confidence, "blueprint_floor": self.last_blueprint_floor, "blueprint_hands": self.last_blueprint_hands, "blueprint_status": self.last_blueprint_status, "kuhn_value_gap": self.last_kuhn_value_gap, "approximate_resolver_enabled": ENABLE_APPROXIMATE_RESOLVER, "heuristic_oracle_enabled": ENABLE_HEURISTIC_ORACLE, "abstract_cfr_teacher_enabled": ENABLE_ABSTRACT_CFR_TEACHER, "abstract_cfr_teacher_mode": ABSTRACT_CFR_TEACHER_MODE, "adversarial_focus": self.last_adversarial_focus, "compiled_transition_fraction": self.last_compiled_transition_fraction, "ppo_kl_limited": self.last_ppo_kl_limited, "ppo_hard_kl": self.last_ppo_hard_kl, "ppo_epoch_budget": self.last_ppo_epoch_budget, "ppo_recovery_updates": self.ppo_recovery_updates, "ppo_update_reverted": self.last_ppo_update_reverted, "ppo_rollback_phase": self.last_ppo_rollback_phase, "ppo_post_step_retry_applied": self.last_ppo_post_step_retry_applied, "ppo_post_step_retry_accepted": self.last_ppo_post_step_retry_accepted, "ppo_post_step_retry_kl": self.last_ppo_post_step_retry_kl}


@dataclass
class TrainingStatus:
    running: bool = False
    smoke_test: bool = False
    episodes: int = 0
    completed: int = 0
    actions: int = 0
    league_size: int = 1
    updates: int = 0
    champion_version: int = 0
    champion_elo: float = 1_200.0
    evaluation_win_rate: float = 0.0
    direct_champion_score: float = 0.0
    direct_bb_per_100: float = 0.0
    evaluation_bb_per_100: float = 0.0
    promotion_ci_lower_bb_per_100: float = 0.0
    promotion_ci_upper_bb_per_100: float = 0.0
    gate_passed: bool = False
    curriculum_stage: int = 1
    curriculum_phase: str = "Foundation"
    curriculum_readiness: float = 0.0
    training_lane: str = "population"
    population_size: int = 1
    active_population: str = "population-balanced"
    population_diversity: float = 0.0
    population_continuity: float = 0.0
    snapshot_count: int = 0
    snapshot_diversity: float = 0.0
    snapshot_min_distance: float = 0.0
    snapshot_rejections: int = 0
    holdout_score: float = 0.0
    holdout_floor: float = 0.0
    holdout_bb_per_100: float = 0.0
    holdout_floor_bb_per_100: float = 0.0
    holdout_ci_floor_bb_per_100: float = 0.0
    holdout_paired_variance: float = 0.0
    continuous_raise_mean: float = 0.5
    scenario_coverage: float = 0.0
    restricted_br_bb_per_100: float = 0.0
    adversarial_floor_bb_per_100: float = 0.0
    adversarial_ci_floor_bb_per_100: float = 0.0
    adversarial_evaluation_hands: int = ADVERSARIAL_SCREENING_HANDS
    adversarial_confirmation_hands: int = 0
    final_audit_ran: bool = False
    adversarial_rollout_fraction: float = 0.0
    adversarial_rollout_run_fraction: float = 0.0
    adversarial_rollout_target: float = ADVERSARIAL_ROLLOUT_FRACTION
    adversarial_focus: str = "pending"
    compiled_transition_fraction: float = 0.0
    crossplay_robustness: float = 0.0
    mixture_diversity: float = 0.0
    opponent_pressure: float = 0.0
    exploitability_proxy: float = 0.0
    best_response_bb_per_100: float = 0.0
    exploiter_diversity: float = 0.0
    exploiter_threat: float = 0.0
    champion_vulnerability: float = 0.0
    exploiter_generations: int = 0
    promotion_confidence: float = 0.0
    challenger_status: str = "untrained"
    evaluation_hands: int = 0
    rollout_seconds: float = 0.0
    rollout_model_sync_seconds: float = 0.0
    rollout_arena_setup_seconds: float = 0.0
    rollout_tensor_preparation_seconds: float = 0.0
    rollout_inference_dispatch_seconds: float = 0.0
    rollout_action_postprocess_seconds: float = 0.0
    rollout_rule_execution_seconds: float = 0.0
    rollout_play_seconds: float = 0.0
    rollout_worker_seconds: float = 0.0
    rollout_dispatch_wait_seconds: float = 0.0
    rollout_cached_opponent_models: int = 0
    learning_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    ppo_tensor_preparation_seconds: float = 0.0
    ppo_transfer_seconds: float = 0.0
    ppo_compute_seconds: float = 0.0
    auxiliary_learning_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    optimizer_backend: str = "adamw"
    parallel_evaluation: bool = False
    evaluation_workers: int = EVALUATION_WORKERS
    resolver_uses: int = 0
    resolver_depth: int = 0
    search_leaf_evaluations: int = 0
    search_value_spread: float = 0.0
    search_confidence: float = 0.0
    adaptive_action_width: int = 0
    endgame_worlds: int = 0
    rare_spot_rate: float = 0.0
    replay_rare_fraction: float = 0.0
    replay_priority: float = 0.0
    replay_recent_fraction: float = 0.0
    belief_confidence: float = 0.0
    leaf_evaluations: int = 0
    cfr_memory_size: int = 0
    strategy_memory_size: int = 0
    search_memory_size: int = 0
    sizing_cfr_loss: float = 0.0
    search_value_loss: float = 0.0
    ensemble_disagreement: float = 0.0
    likelihood_memory_size: int = 0
    imitation_memory_size: int = 0
    imitation_loss: float = 0.0
    imitation_reward: float = 0.0
    tail_loss_rate: float = 0.0
    tail_loss_bb: float = 0.0
    tail_policy_weight: float = 1.0
    tail_loss_run_rate: float = 0.0
    tail_loss_run_bb: float = 0.0
    tail_policy_run_weight: float = 1.0
    hard_spot_value_loss: float = 0.0
    hard_spot_memory_size: int = 0
    behavior_action_agreement: float = 1.0
    behavior_action_change_rate: float = 0.0
    behavior_raise_fraction_delta: float = 0.0
    behavior_audit_states: int = 0
    preflop_open_cap_bb: float = PREFLOP_OPEN_RAISE_CAP_BB
    preflop_sizing_audit_roots: int = 0
    preflop_normal_raise_rate: float = 0.0
    preflop_mean_raise_bb: float = 0.0
    preflop_p95_raise_bb: float = 0.0
    preflop_oversized_open_rate: float = 0.0
    preflop_cap_hit_rate: float = 0.0
    preflop_all_in_rate: float = 0.0
    preflop_three_bet_cap_pot_multiplier: float = PREFLOP_THREE_BET_POT_CAP_MULTIPLIER
    preflop_three_bet_audit_roots: int = 0
    preflop_three_bet_normal_raise_rate: float = 0.0
    preflop_three_bet_mean_raise_to_pot: float = 0.0
    preflop_three_bet_p95_raise_to_pot: float = 0.0
    preflop_three_bet_cap_hit_rate: float = 0.0
    preflop_three_bet_over_cap_rate: float = 0.0
    preflop_three_bet_minimum_override_rate: float = 0.0
    preflop_three_bet_all_in_rate: float = 0.0
    preflop_forced_root_fraction: float = 0.0
    preflop_scenario_audit_hands: int = 0
    preflop_scenario_worst_lcb_bb_per_100: float = 0.0
    preflop_scenario_worst_root: str = "pending"
    preflop_scenario_worst_style: str = "pending"
    preflop_allin_calibration_loss: float = 0.0
    preflop_allin_stability_loss: float = 0.0
    preflop_guarded_allin_probability: float = 0.0
    preflop_allin_target: float = 0.0
    preflop_guarded_state_fraction: float = 0.0
    preflop_immediate_allin_rate: float = 0.0
    preflop_immediate_allin_target: float = 0.0
    preflop_immediate_eligible_rate: float = 0.0
    preflop_root_promotion_lcb_floor: float = PREFLOP_ROOT_PROMOTION_LCB_FLOOR
    preflop_3bet_teacher_loss: float = 0.0
    preflop_3bet_teacher_eligible_roots: int = 0
    preflop_3bet_teacher_samples: int = 0
    preflop_3bet_teacher_coverage: float = 0.0
    preflop_3bet_teacher_confidence: float = 0.0
    preflop_3bet_teacher_raise_target: float = 0.0
    preflop_3bet_teacher_raise_advantage_bb: float = 0.0
    preflop_3bet_teacher_actual_raise_rate: float = 0.0
    preflop_3bet_teacher_allin_target: float = 0.0
    preflop_3bet_teacher_actual_allin_rate: float = 0.0
    preflop_3bet_teacher_multi_raise_samples: int = 0
    preflop_3bet_teacher_multi_raise_allin_target: float = 0.0
    preflop_3bet_teacher_multi_raise_actual_allin_rate: float = 0.0
    preflop_3bet_teacher_multi_raise_allin_vetoes: int = 0
    robust_policy_weight: float = 1.0
    target_network_drift: float = 0.0
    ppo_learning_rate: float = 3e-4
    ppo_clip_epsilon: float = 0.20
    ppo_entropy_coefficient: float = 0.012
    ppo_kl_target: float = 0.012
    ppo_epochs: int = 0
    ppo_clip_fraction: float = 0.0
    ppo_kl_limited: bool = False
    ppo_hard_kl: float = 0.0
    ppo_epoch_budget: int = PPO_MAX_EPOCHS
    ppo_recovery_updates: int = 0
    ppo_update_reverted: bool = False
    ppo_rollback_phase: str = "none"
    ppo_post_step_retry_applied: bool = False
    ppo_post_step_retry_accepted: bool = False
    ppo_post_step_retry_kl: float = 0.0
    mixed_precision_enabled: bool = False
    gradient_scale: float = 1.0
    resumed: bool = False
    range_loss: float = 0.0
    range_accuracy: float = 0.0
    range_brier: float = 0.0
    range_ece: float = 0.0
    range_coarse_accuracy: float = 0.0
    range_coarse_brier: float = 0.0
    cfr_advantage_loss: float = 0.0
    average_strategy_loss: float = 0.0
    cfr_effective_weight: float = 0.0
    distributional_value_loss: float = 0.0
    belief_log_loss: float = 0.0
    belief_action_accuracy: float = 0.0
    benchmark_leader: str = "pending"
    benchmark_bb_per_100: float = 0.0
    tournament_count: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    steps_per_second: float = 0.0
    training_focus: str = "balanced"
    weakness_score: float = 0.50
    adaptive_workers: int = 0
    adaptive_batch_hands: int = 0
    rollout_decisions_per_second: float = 0.0
    teacher_data_records: int = 0
    teacher_data_filename: str = ""
    teacher_data_status: str = "no local teacher data"
    audit_score: float = 0.0
    audit_exploitability_bb_per_100: float = 0.0
    scenario_gate: float = 0.0
    ablation_delta: float = 0.0
    evaluation_history_size: int = 0
    subgame_policy_loss: float = 0.0
    subgame_teacher_size: int = 0
    rollout_arena_width: int = 0
    average_strategy_weight: float = 0.0
    oracle_policy_loss: float = 0.0
    oracle_value_loss: float = 0.0
    oracle_teacher_size: int = 0
    oracle_confidence: float = 0.0
    oracle_iterations: int = 0
    abstraction_nash_conv: float = 0.0
    abstraction_value: float = 0.0
    abstraction_information_sets: int = 0
    paired_deal_coverage: float = 0.0
    rollout_backend: str = "python-batched"
    rollout_inference_device: str = "cpu"
    belief_posterior_support: float = 1.0
    resolver_replay_confidence: float = 0.0
    resolver_replay_size: int = 0
    blueprint_score: float = 0.0
    blueprint_confidence: float = 0.0
    blueprint_floor: float = 0.0
    blueprint_hands: int = 0
    blueprint_status: str = "not audited"
    kuhn_value_gap: float = 1.0
    counterfactual_value_loss: float = 0.0
    counterfactual_memory_size: int = 0
    counterfactual_coverage: float = 0.0
    resolver_safety_rejections: int = 0
    resolver_safety_margin: float = 0.0
    resolver_safety_confidence: float = 0.0
    resolver_confident_actions: int = 0
    resolver_iterations: int = 0
    resolver_strategy_peak: float = 0.0
    public_belief_teacher_size: int = 0
    sizing_proposal_diversity: float = 0.0
    abstract_cfr_teacher_mode: str = ABSTRACT_CFR_TEACHER_MODE
    last_error: str | None = None
    report_filename: str = ""
    report_status: str = "not generated"
    _started_at: float = 0.0
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def begin(self, episodes: int, smoke_test: bool = False) -> bool:
        with self._lock:
            if self.running:
                return False
            self.running, self.smoke_test, self.episodes, self.completed, self.actions = True, smoke_test, episodes, 0, 0
            self.league_size, self.updates, self.champion_version, self.champion_elo = 1, 0, 0, 1_200.0
            self.evaluation_win_rate = self.direct_champion_score = self.direct_bb_per_100 = self.evaluation_bb_per_100 = self.promotion_ci_lower_bb_per_100 = self.promotion_ci_upper_bb_per_100 = self.policy_loss = self.value_loss = self.entropy = self.kl_divergence = self.steps_per_second = self.curriculum_readiness = self.mixture_diversity = self.opponent_pressure = self.exploitability_proxy = self.best_response_bb_per_100 = self.exploiter_diversity = self.promotion_confidence = self.search_value_spread = self.search_confidence = self.rare_spot_rate = self.replay_rare_fraction = self.replay_priority = self.replay_recent_fraction = self.belief_confidence = self.range_loss = self.range_accuracy = self.range_brier = self.range_ece = self.range_coarse_accuracy = self.range_coarse_brier = self.cfr_advantage_loss = self.average_strategy_loss = self.cfr_effective_weight = self.benchmark_bb_per_100 = 0.0
            self.gate_passed, self.curriculum_stage, self.curriculum_phase, self.training_lane, self.challenger_status, self.resolver_uses, self.resolver_depth, self.search_leaf_evaluations, self.leaf_evaluations, self.cfr_memory_size, self.resumed, self.benchmark_leader, self.tournament_count, self.last_error, self.report_filename, self.report_status, self._started_at = False, 1, "Foundation", "population", "untrained", 0, 0, 0, 0, 0, False, "pending", 0, None, "", "pending", perf_counter()
            self.likelihood_memory_size = 0
            self.target_network_drift = 0.0
            self.distributional_value_loss = self.belief_log_loss = self.belief_action_accuracy = 0.0
            self.evaluation_hands = self.imitation_memory_size = self.ppo_epochs = 0
            self.rollout_seconds = self.learning_seconds = self.evaluation_seconds = self.checkpoint_seconds = 0.0
            self.rollout_model_sync_seconds = self.rollout_arena_setup_seconds = self.rollout_tensor_preparation_seconds = self.rollout_inference_dispatch_seconds = 0.0
            self.rollout_action_postprocess_seconds = self.rollout_rule_execution_seconds = self.rollout_play_seconds = self.rollout_worker_seconds = self.rollout_dispatch_wait_seconds = 0.0
            self.rollout_cached_opponent_models = 0
            self.ppo_tensor_preparation_seconds = self.ppo_transfer_seconds = self.ppo_compute_seconds = self.auxiliary_learning_seconds = self.optimizer_seconds = 0.0
            self.optimizer_backend = "adamw"
            self.parallel_evaluation = False
            self.evaluation_workers = EVALUATION_WORKERS
            self.imitation_loss = self.imitation_reward = self.ppo_clip_fraction = 0.0
            self.tail_loss_rate = self.tail_loss_bb = self.hard_spot_value_loss = 0.0
            self.tail_policy_weight = 1.0
            self.tail_loss_run_rate = self.tail_loss_run_bb = 0.0
            self.tail_policy_run_weight = 1.0
            self.hard_spot_memory_size = self.behavior_audit_states = 0
            self.behavior_action_agreement, self.behavior_action_change_rate, self.behavior_raise_fraction_delta = 1.0, 0.0, 0.0
            self.preflop_open_cap_bb = PREFLOP_OPEN_RAISE_CAP_BB
            self.preflop_sizing_audit_roots = 0
            self.preflop_normal_raise_rate = self.preflop_mean_raise_bb = self.preflop_p95_raise_bb = 0.0
            self.preflop_oversized_open_rate = self.preflop_cap_hit_rate = self.preflop_all_in_rate = 0.0
            self.preflop_three_bet_cap_pot_multiplier = PREFLOP_THREE_BET_POT_CAP_MULTIPLIER
            self.preflop_three_bet_audit_roots = 0
            self.preflop_three_bet_normal_raise_rate = self.preflop_three_bet_mean_raise_to_pot = self.preflop_three_bet_p95_raise_to_pot = 0.0
            self.preflop_three_bet_cap_hit_rate = self.preflop_three_bet_over_cap_rate = self.preflop_three_bet_minimum_override_rate = self.preflop_three_bet_all_in_rate = 0.0
            self.preflop_forced_root_fraction = self.preflop_scenario_worst_lcb_bb_per_100 = 0.0
            self.preflop_scenario_audit_hands = 0
            self.preflop_scenario_worst_root = self.preflop_scenario_worst_style = "pending"
            self.preflop_allin_calibration_loss = self.preflop_allin_stability_loss = self.preflop_guarded_allin_probability = self.preflop_allin_target = self.preflop_guarded_state_fraction = self.preflop_immediate_allin_rate = self.preflop_immediate_allin_target = self.preflop_immediate_eligible_rate = 0.0
            self.preflop_root_promotion_lcb_floor = PREFLOP_ROOT_PROMOTION_LCB_FLOOR
            self.preflop_3bet_teacher_loss = self.preflop_3bet_teacher_coverage = self.preflop_3bet_teacher_confidence = self.preflop_3bet_teacher_raise_target = self.preflop_3bet_teacher_raise_advantage_bb = self.preflop_3bet_teacher_actual_raise_rate = self.preflop_3bet_teacher_allin_target = self.preflop_3bet_teacher_actual_allin_rate = self.preflop_3bet_teacher_multi_raise_allin_target = self.preflop_3bet_teacher_multi_raise_actual_allin_rate = 0.0
            self.preflop_3bet_teacher_eligible_roots = self.preflop_3bet_teacher_samples = self.preflop_3bet_teacher_multi_raise_samples = self.preflop_3bet_teacher_multi_raise_allin_vetoes = 0
            self.robust_policy_weight = 1.0
            self.ppo_learning_rate, self.ppo_clip_epsilon, self.ppo_entropy_coefficient, self.ppo_kl_target = 3e-4, 0.20, 0.012, 0.012
            self.ppo_kl_limited, self.ppo_hard_kl, self.ppo_epoch_budget, self.ppo_recovery_updates, self.ppo_update_reverted, self.ppo_rollback_phase = False, 0.0, PPO_MAX_EPOCHS, 0, False, "none"
            self.ppo_post_step_retry_applied, self.ppo_post_step_retry_accepted, self.ppo_post_step_retry_kl = False, False, 0.0
            self.mixed_precision_enabled, self.gradient_scale = False, 1.0
            self.population_size, self.active_population, self.population_diversity, self.population_continuity, self.holdout_score, self.holdout_floor, self.holdout_bb_per_100, self.holdout_floor_bb_per_100, self.continuous_raise_mean, self.scenario_coverage, self.restricted_br_bb_per_100, self.adversarial_floor_bb_per_100, self.adversarial_rollout_fraction, self.adversarial_rollout_target, self.crossplay_robustness = 1, "population-balanced", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, ADVERSARIAL_ROLLOUT_FRACTION, 0.0
            self.holdout_ci_floor_bb_per_100 = self.holdout_paired_variance = 0.0
            self.adversarial_ci_floor_bb_per_100 = 0.0
            self.adversarial_evaluation_hands = ADVERSARIAL_SCREENING_HANDS
            self.adversarial_confirmation_hands, self.final_audit_ran = 0, False
            self.adversarial_rollout_run_fraction = 0.0
            self.adversarial_focus, self.compiled_transition_fraction = "pending", 0.0
            self.snapshot_count = self.adaptive_action_width = self.endgame_worlds = 0
            self.snapshot_diversity = 0.0
            self.snapshot_min_distance = 0.0
            self.snapshot_rejections = 0
            self.strategy_memory_size = self.search_memory_size = 0
            self.sizing_cfr_loss = self.search_value_loss = self.ensemble_disagreement = 0.0
            self.training_focus, self.weakness_score = "balanced", 0.50
            self.adaptive_workers = self.adaptive_batch_hands = self.teacher_data_records = 0
            self.rollout_decisions_per_second = 0.0
            self.teacher_data_filename, self.teacher_data_status = "", "no local teacher data"
            self.audit_score = self.audit_exploitability_bb_per_100 = self.scenario_gate = self.ablation_delta = 0.0
            self.evaluation_history_size = 0
            self.subgame_policy_loss = 0.0
            self.subgame_teacher_size = self.rollout_arena_width = 0
            self.average_strategy_weight = 0.0
            self.oracle_policy_loss = self.oracle_value_loss = self.oracle_confidence = self.paired_deal_coverage = 0.0
            self.oracle_teacher_size = self.oracle_iterations = 0
            self.abstraction_nash_conv = self.abstraction_value = 0.0
            self.abstraction_information_sets = 0
            self.rollout_backend = "python-batched"
            self.rollout_inference_device = "cpu"
            self.belief_posterior_support = 1.0
            self.resolver_replay_confidence = self.blueprint_score = self.blueprint_confidence = self.blueprint_floor = 0.0
            self.resolver_replay_size = self.blueprint_hands = 0
            self.blueprint_status, self.kuhn_value_gap = "not audited", 1.0
            self.counterfactual_value_loss = self.counterfactual_coverage = 0.0
            self.counterfactual_memory_size = self.resolver_safety_rejections = 0
            self.exploiter_threat = self.champion_vulnerability = 0.0
            self.exploiter_generations = 0
            self.resolver_safety_margin = self.resolver_safety_confidence = 0.0
            self.resolver_confident_actions = 0
            self.resolver_iterations = self.public_belief_teacher_size = 0
            self.resolver_strategy_peak = self.sizing_proposal_diversity = 0.0
            self.abstract_cfr_teacher_mode = ABSTRACT_CFR_TEACHER_MODE
            return True

    def record(self, completed: int, actions: int, summary: dict, losses: dict) -> None:
        with self._lock:
            self.completed, self.actions = completed, self.actions + actions
            self.league_size, self.updates, self.champion_version = int(summary["league_size"]), int(summary["updates"]), int(summary["champion_version"])
            self.champion_elo, self.evaluation_win_rate, self.direct_champion_score = float(summary["champion_elo"]), float(summary["evaluation_win_rate"]), float(summary["direct_champion_score"])
            self.direct_bb_per_100, self.evaluation_bb_per_100, self.promotion_ci_lower_bb_per_100, self.promotion_ci_upper_bb_per_100 = float(summary["direct_bb_per_100"]), float(summary["evaluation_bb_per_100"]), float(summary["promotion_ci_lower_bb_per_100"]), float(summary["promotion_ci_upper_bb_per_100"])
            self.gate_passed, self.curriculum_stage, self.curriculum_phase, self.curriculum_readiness, self.training_lane = bool(summary["gate_passed"]), int(summary["curriculum_stage"]) + 1, str(summary["curriculum_phase"]), float(summary["curriculum_readiness"]), str(summary["training_lane"])
            self.population_size, self.active_population, self.population_diversity, self.population_continuity, self.holdout_score, self.holdout_floor, self.holdout_bb_per_100, self.holdout_floor_bb_per_100, self.continuous_raise_mean, self.scenario_coverage, self.restricted_br_bb_per_100, self.crossplay_robustness = int(summary["population_size"]), str(summary["active_population"]), float(summary["population_diversity"]), float(summary["population_continuity"]), float(summary["holdout_score"]), float(summary["holdout_floor"]), float(summary["holdout_bb_per_100"]), float(summary["holdout_floor_bb_per_100"]), float(summary["continuous_raise_mean"]), float(summary["scenario_coverage"]), float(summary["restricted_br_bb_per_100"]), float(summary["crossplay_robustness"])
            self.adversarial_floor_bb_per_100, self.adversarial_ci_floor_bb_per_100, self.adversarial_evaluation_hands, self.adversarial_rollout_fraction, self.adversarial_rollout_target = float(summary["adversarial_floor_bb_per_100"]), float(summary["adversarial_ci_floor_bb_per_100"]), int(summary["adversarial_evaluation_hands"]), float(summary["adversarial_rollout_fraction"]), float(summary["adversarial_rollout_target"])
            self.adversarial_confirmation_hands, self.final_audit_ran = int(summary["adversarial_confirmation_hands"]), bool(summary["final_audit_ran"])
            self.adversarial_rollout_run_fraction = float(summary["adversarial_rollout_run_fraction"])
            self.holdout_ci_floor_bb_per_100, self.holdout_paired_variance = float(summary["holdout_ci_floor_bb_per_100"]), float(summary["holdout_paired_variance"])
            self.snapshot_count, self.snapshot_diversity, self.snapshot_min_distance, self.snapshot_rejections = int(summary["snapshot_count"]), float(summary["snapshot_diversity"]), float(summary["snapshot_min_distance"]), int(summary["snapshot_rejections"])
            self.mixture_diversity, self.opponent_pressure, self.exploitability_proxy, self.best_response_bb_per_100, self.exploiter_diversity, self.promotion_confidence, self.challenger_status, self.resolver_uses, self.resolver_depth, self.search_leaf_evaluations, self.search_value_spread, self.search_confidence, self.rare_spot_rate, self.replay_rare_fraction, self.replay_priority, self.replay_recent_fraction, self.belief_confidence, self.leaf_evaluations, self.cfr_memory_size, self.resumed = float(summary["mixture_diversity"]), float(summary["opponent_pressure"]), float(summary["exploitability_proxy"]), float(summary["best_response_bb_per_100"]), float(summary["exploiter_diversity"]), float(summary["promotion_confidence"]), str(summary["challenger_status"]), int(summary["resolver_uses"]), int(summary["resolver_depth"]), int(summary["search_leaf_evaluations"]), float(summary["search_value_spread"]), float(summary["search_confidence"]), float(summary["rare_spot_rate"]), float(summary["replay_rare_fraction"]), float(summary["replay_priority"]), float(summary["replay_recent_fraction"]), float(summary["belief_confidence"]), int(summary["leaf_evaluations"]), int(summary["cfr_memory_size"]), bool(summary["resumed"])
            self.exploiter_threat, self.champion_vulnerability, self.exploiter_generations = float(summary["exploiter_threat"]), float(summary["champion_vulnerability"]), int(summary["exploiter_generations"])
            self.likelihood_memory_size = int(summary["likelihood_memory_size"])
            self.strategy_memory_size, self.search_memory_size = int(summary["strategy_memory_size"]), int(summary["search_memory_size"])
            self.adaptive_action_width, self.endgame_worlds = int(summary["adaptive_action_width"]), int(summary["endgame_worlds"])
            self.imitation_memory_size, self.imitation_loss, self.imitation_reward = int(summary["imitation_memory_size"]), float(summary["imitation_loss"]), float(summary["imitation_reward"])
            self.tail_loss_rate, self.tail_loss_bb, self.tail_policy_weight = float(summary["tail_loss_rate"]), float(summary["tail_loss_bb"]), float(summary["tail_policy_weight"])
            self.tail_loss_run_rate, self.tail_loss_run_bb, self.tail_policy_run_weight = float(summary["tail_loss_run_rate"]), float(summary["tail_loss_run_bb"]), float(summary["tail_policy_run_weight"])
            self.hard_spot_value_loss, self.hard_spot_memory_size = float(summary["hard_spot_value_loss"]), int(summary["hard_spot_memory_size"])
            self.behavior_action_agreement, self.behavior_action_change_rate, self.behavior_raise_fraction_delta, self.behavior_audit_states = float(summary["behavior_action_agreement"]), float(summary["behavior_action_change_rate"]), float(summary["behavior_raise_fraction_delta"]), int(summary["behavior_audit_states"])
            self.preflop_open_cap_bb, self.preflop_sizing_audit_roots, self.preflop_normal_raise_rate, self.preflop_mean_raise_bb, self.preflop_p95_raise_bb, self.preflop_oversized_open_rate, self.preflop_cap_hit_rate, self.preflop_all_in_rate = float(summary["preflop_open_cap_bb"]), int(summary["preflop_sizing_audit_roots"]), float(summary["preflop_normal_raise_rate"]), float(summary["preflop_mean_raise_bb"]), float(summary["preflop_p95_raise_bb"]), float(summary["preflop_oversized_open_rate"]), float(summary["preflop_cap_hit_rate"]), float(summary["preflop_all_in_rate"])
            self.preflop_three_bet_cap_pot_multiplier, self.preflop_three_bet_audit_roots, self.preflop_three_bet_normal_raise_rate, self.preflop_three_bet_mean_raise_to_pot, self.preflop_three_bet_p95_raise_to_pot, self.preflop_three_bet_cap_hit_rate, self.preflop_three_bet_over_cap_rate, self.preflop_three_bet_minimum_override_rate, self.preflop_three_bet_all_in_rate = float(summary["preflop_three_bet_cap_pot_multiplier"]), int(summary["preflop_three_bet_audit_roots"]), float(summary["preflop_three_bet_normal_raise_rate"]), float(summary["preflop_three_bet_mean_raise_to_pot"]), float(summary["preflop_three_bet_p95_raise_to_pot"]), float(summary["preflop_three_bet_cap_hit_rate"]), float(summary["preflop_three_bet_over_cap_rate"]), float(summary["preflop_three_bet_minimum_override_rate"]), float(summary["preflop_three_bet_all_in_rate"])
            self.preflop_forced_root_fraction, self.preflop_scenario_audit_hands, self.preflop_scenario_worst_lcb_bb_per_100, self.preflop_scenario_worst_root, self.preflop_scenario_worst_style, self.robust_policy_weight = float(summary["preflop_forced_root_fraction"]), int(summary["preflop_scenario_audit_hands"]), float(summary["preflop_scenario_worst_lcb_bb_per_100"]), str(summary["preflop_scenario_worst_root"]), str(summary["preflop_scenario_worst_style"]), float(summary["robust_policy_weight"])
            self.preflop_allin_calibration_loss, self.preflop_allin_stability_loss, self.preflop_guarded_allin_probability, self.preflop_allin_target, self.preflop_guarded_state_fraction = float(summary["preflop_allin_calibration_loss"]), float(summary.get("preflop_allin_stability_loss", 0.0)), float(summary["preflop_guarded_allin_probability"]), float(summary["preflop_allin_target"]), float(summary["preflop_guarded_state_fraction"])
            self.preflop_immediate_allin_rate, self.preflop_immediate_allin_target, self.preflop_immediate_eligible_rate, self.preflop_root_promotion_lcb_floor = float(summary["preflop_immediate_allin_rate"]), float(summary["preflop_immediate_allin_target"]), float(summary["preflop_immediate_eligible_rate"]), float(summary["preflop_root_promotion_lcb_floor"])
            self.preflop_3bet_teacher_loss, self.preflop_3bet_teacher_eligible_roots, self.preflop_3bet_teacher_samples, self.preflop_3bet_teacher_coverage = float(summary["preflop_3bet_teacher_loss"]), int(summary["preflop_3bet_teacher_eligible_roots"]), int(summary["preflop_3bet_teacher_samples"]), float(summary["preflop_3bet_teacher_coverage"])
            self.preflop_3bet_teacher_confidence, self.preflop_3bet_teacher_raise_target, self.preflop_3bet_teacher_raise_advantage_bb, self.preflop_3bet_teacher_actual_raise_rate, self.preflop_3bet_teacher_allin_target, self.preflop_3bet_teacher_actual_allin_rate = float(summary["preflop_3bet_teacher_confidence"]), float(summary["preflop_3bet_teacher_raise_target"]), float(summary["preflop_3bet_teacher_raise_advantage_bb"]), float(summary["preflop_3bet_teacher_actual_raise_rate"]), float(summary.get("preflop_3bet_teacher_allin_target", 0.0)), float(summary.get("preflop_3bet_teacher_actual_allin_rate", 0.0))
            self.preflop_3bet_teacher_multi_raise_samples = int(summary.get("preflop_3bet_teacher_multi_raise_samples", 0))
            self.preflop_3bet_teacher_multi_raise_allin_target = float(summary.get("preflop_3bet_teacher_multi_raise_allin_target", 0.0))
            self.preflop_3bet_teacher_multi_raise_actual_allin_rate = float(summary.get("preflop_3bet_teacher_multi_raise_actual_allin_rate", 0.0))
            self.preflop_3bet_teacher_multi_raise_allin_vetoes = int(summary.get("preflop_3bet_teacher_multi_raise_allin_vetoes", 0))
            self.target_network_drift = float(summary["target_network_drift"])
            self.ppo_learning_rate, self.ppo_clip_epsilon, self.ppo_entropy_coefficient, self.ppo_kl_target, self.ppo_epochs, self.ppo_clip_fraction = float(summary["ppo_learning_rate"]), float(summary["ppo_clip_epsilon"]), float(summary["ppo_entropy_coefficient"]), float(summary["ppo_kl_target"]), int(summary["ppo_epochs"]), float(summary["ppo_clip_fraction"])
            self.ppo_kl_limited, self.ppo_hard_kl, self.ppo_epoch_budget, self.ppo_recovery_updates, self.ppo_update_reverted, self.ppo_rollback_phase = bool(summary["ppo_kl_limited"]), float(summary["ppo_hard_kl"]), int(summary["ppo_epoch_budget"]), int(summary["ppo_recovery_updates"]), bool(summary.get("ppo_update_reverted", False)), str(summary.get("ppo_rollback_phase", "none"))
            self.ppo_post_step_retry_applied, self.ppo_post_step_retry_accepted, self.ppo_post_step_retry_kl = bool(summary.get("ppo_post_step_retry_applied", False)), bool(summary.get("ppo_post_step_retry_accepted", False)), float(summary.get("ppo_post_step_retry_kl", 0.0))
            self.adversarial_focus, self.compiled_transition_fraction = str(summary["adversarial_focus"]), float(summary["compiled_transition_fraction"])
            self.mixed_precision_enabled, self.gradient_scale, self.evaluation_hands = bool(summary["mixed_precision_enabled"]), float(summary["gradient_scale"]), int(summary["evaluation_hands"])
            self.rollout_seconds, self.learning_seconds, self.evaluation_seconds, self.checkpoint_seconds = float(losses["rollout_seconds"]), float(losses["learning_seconds"]), float(summary["evaluation_seconds"]), float(losses["checkpoint_seconds"])
            self.rollout_model_sync_seconds = float(losses.get("rollout_model_sync_seconds", 0.0))
            self.rollout_arena_setup_seconds = float(losses.get("rollout_arena_setup_seconds", 0.0))
            self.rollout_tensor_preparation_seconds = float(losses.get("rollout_tensor_preparation_seconds", 0.0))
            self.rollout_inference_dispatch_seconds = float(losses.get("rollout_inference_dispatch_seconds", 0.0))
            self.rollout_action_postprocess_seconds = float(losses.get("rollout_action_postprocess_seconds", 0.0))
            self.rollout_rule_execution_seconds = float(losses.get("rollout_rule_execution_seconds", 0.0))
            self.rollout_play_seconds = float(losses.get("rollout_play_seconds", 0.0))
            self.rollout_worker_seconds = float(losses.get("rollout_worker_seconds", 0.0))
            self.rollout_dispatch_wait_seconds = float(losses.get("rollout_dispatch_wait_seconds", 0.0))
            self.rollout_cached_opponent_models = int(losses.get("rollout_cached_opponent_models", 0))
            self.ppo_tensor_preparation_seconds, self.ppo_transfer_seconds, self.ppo_compute_seconds, self.auxiliary_learning_seconds, self.optimizer_seconds, self.optimizer_backend = float(losses["ppo_tensor_preparation_seconds"]), float(losses["ppo_transfer_seconds"]), float(losses["ppo_compute_seconds"]), float(losses["auxiliary_learning_seconds"]), float(losses["optimizer_seconds"]), str(losses["optimizer_backend"])
            self.parallel_evaluation, self.evaluation_workers = bool(summary["parallel_evaluation"]), int(summary["evaluation_workers"])
            self.benchmark_leader, self.benchmark_bb_per_100, self.tournament_count = str(summary["benchmark_leader"]), float(summary["benchmark_bb_per_100"]), int(summary["tournament_count"])
            self.policy_loss, self.value_loss, self.entropy, self.kl_divergence, self.range_loss, self.range_accuracy, self.range_brier, self.range_ece, self.range_coarse_accuracy, self.range_coarse_brier, self.cfr_advantage_loss, self.average_strategy_loss, self.cfr_effective_weight = (float(losses[key]) for key in ("policy_loss", "value_loss", "entropy", "kl_divergence", "range_loss", "range_accuracy", "range_brier", "range_ece", "range_coarse_accuracy", "range_coarse_brier", "cfr_advantage_loss", "average_strategy_loss", "cfr_effective_weight"))
            self.sizing_cfr_loss = float(losses["sizing_cfr_loss"])
            self.search_value_loss = float(losses["search_value_loss"])
            self.ensemble_disagreement = float(losses["ensemble_disagreement"])
            self.training_focus, self.weakness_score = str(summary["training_focus"]), float(summary["weakness_score"])
            self.adaptive_workers, self.adaptive_batch_hands, self.rollout_decisions_per_second = int(summary["adaptive_workers"]), int(summary["adaptive_batch_hands"]), float(summary["rollout_decisions_per_second"])
            self.teacher_data_records, self.teacher_data_filename, self.teacher_data_status = int(summary["teacher_data_records"]), str(summary["teacher_data_filename"]), str(summary["teacher_data_status"])
            self.audit_score, self.audit_exploitability_bb_per_100, self.scenario_gate, self.ablation_delta, self.evaluation_history_size = float(summary["audit_score"]), float(summary["audit_exploitability_bb_per_100"]), float(summary["scenario_gate"]), float(summary["ablation_delta"]), int(summary["evaluation_history_size"])
            self.subgame_policy_loss, self.subgame_teacher_size, self.rollout_arena_width, self.average_strategy_weight = float(summary["subgame_policy_loss"]), int(summary["subgame_teacher_size"]), int(summary["rollout_arena_width"]), float(summary["average_strategy_weight"])
            self.oracle_policy_loss, self.oracle_value_loss, self.oracle_teacher_size, self.oracle_confidence, self.oracle_iterations, self.paired_deal_coverage, self.rollout_backend, self.rollout_inference_device = float(losses["oracle_policy_loss"]), float(losses["oracle_value_loss"]), int(summary["oracle_teacher_size"]), float(summary["oracle_confidence"]), int(summary["oracle_iterations"]), float(summary["paired_deal_coverage"]), str(summary["rollout_backend"]), str(summary["rollout_inference_device"])
            self.abstraction_nash_conv, self.abstraction_value, self.abstraction_information_sets = float(summary["abstraction_nash_conv"]), float(summary["abstraction_value"]), int(summary["abstraction_information_sets"])
            self.belief_posterior_support, self.resolver_replay_confidence, self.resolver_replay_size = float(summary["belief_posterior_support"]), float(summary["resolver_replay_confidence"]), int(summary["resolver_replay_size"])
            self.blueprint_score, self.blueprint_confidence, self.blueprint_floor, self.blueprint_hands, self.blueprint_status, self.kuhn_value_gap = float(summary["blueprint_score"]), float(summary["blueprint_confidence"]), float(summary["blueprint_floor"]), int(summary["blueprint_hands"]), str(summary["blueprint_status"]), float(summary["kuhn_value_gap"])
            self.counterfactual_value_loss, self.counterfactual_memory_size, self.counterfactual_coverage, self.resolver_safety_rejections = float(losses["counterfactual_value_loss"]), int(summary["counterfactual_memory_size"]), float(summary["counterfactual_coverage"]), int(summary["resolver_safety_rejections"])
            self.resolver_safety_margin, self.resolver_safety_confidence, self.resolver_confident_actions = float(summary["resolver_safety_margin"]), float(summary["resolver_safety_confidence"]), int(summary["resolver_confident_actions"])
            self.resolver_iterations, self.resolver_strategy_peak = int(summary["resolver_iterations"]), float(summary["resolver_strategy_peak"])
            self.public_belief_teacher_size, self.sizing_proposal_diversity = int(summary["public_belief_teacher_size"]), float(summary["sizing_proposal_diversity"])
            self.abstract_cfr_teacher_mode = str(summary["abstract_cfr_teacher_mode"])
            self.distributional_value_loss = float(losses["distributional_value_loss"])
            self.belief_log_loss = float(losses["belief_log_loss"])
            self.belief_action_accuracy = float(losses["belief_action_accuracy"])
            self.steps_per_second = self.actions / max(perf_counter() - self._started_at, 0.001)

    def finish(self, error: str | None = None) -> None:
        with self._lock:
            self.last_error, self.running = error, False

    def note_report(self, filename: str, report_status: str) -> None:
        with self._lock:
            self.report_filename, self.report_status = filename, report_status

    def view(self, parameters: int, runtime: dict | None = None) -> dict:
        with self._lock:
            result = {"running": self.running, "smoke_test": self.smoke_test, "episodes": self.episodes, "completed": self.completed, "actions": self.actions, "progress": self.completed / self.episodes if self.episodes else 0, "league_size": self.league_size, "updates": self.updates, "champion_version": self.champion_version, "champion_elo": self.champion_elo, "evaluation_win_rate": self.evaluation_win_rate, "direct_champion_score": self.direct_champion_score, "direct_bb_per_100": self.direct_bb_per_100, "evaluation_bb_per_100": self.evaluation_bb_per_100, "promotion_ci_lower_bb_per_100": self.promotion_ci_lower_bb_per_100, "promotion_ci_upper_bb_per_100": self.promotion_ci_upper_bb_per_100, "gate_passed": self.gate_passed, "curriculum_stage": self.curriculum_stage, "curriculum_phase": self.curriculum_phase, "curriculum_readiness": self.curriculum_readiness, "training_lane": self.training_lane, "mixture_diversity": self.mixture_diversity, "opponent_pressure": self.opponent_pressure, "exploitability_proxy": self.exploitability_proxy, "best_response_bb_per_100": self.best_response_bb_per_100, "exploiter_diversity": self.exploiter_diversity, "promotion_confidence": self.promotion_confidence, "challenger_status": self.challenger_status, "evaluation_hands": self.evaluation_hands, "rollout_seconds": self.rollout_seconds, "learning_seconds": self.learning_seconds, "evaluation_seconds": self.evaluation_seconds, "checkpoint_seconds": self.checkpoint_seconds, "parallel_evaluation": self.parallel_evaluation, "evaluation_workers": self.evaluation_workers, "resolver_uses": self.resolver_uses, "resolver_depth": self.resolver_depth, "search_leaf_evaluations": self.search_leaf_evaluations, "search_value_spread": self.search_value_spread, "search_confidence": self.search_confidence, "rare_spot_rate": self.rare_spot_rate, "replay_rare_fraction": self.replay_rare_fraction, "replay_priority": self.replay_priority, "replay_recent_fraction": self.replay_recent_fraction, "belief_confidence": self.belief_confidence, "leaf_evaluations": self.leaf_evaluations, "cfr_memory_size": self.cfr_memory_size, "likelihood_memory_size": self.likelihood_memory_size, "imitation_memory_size": self.imitation_memory_size, "imitation_loss": self.imitation_loss, "imitation_reward": self.imitation_reward, "target_network_drift": self.target_network_drift, "ppo_learning_rate": self.ppo_learning_rate, "ppo_clip_epsilon": self.ppo_clip_epsilon, "ppo_entropy_coefficient": self.ppo_entropy_coefficient, "ppo_kl_target": self.ppo_kl_target, "ppo_epochs": self.ppo_epochs, "ppo_clip_fraction": self.ppo_clip_fraction, "mixed_precision_enabled": self.mixed_precision_enabled, "gradient_scale": self.gradient_scale, "resumed": self.resumed, "range_loss": self.range_loss, "range_accuracy": self.range_accuracy, "range_brier": self.range_brier, "range_ece": self.range_ece, "range_coarse_accuracy": self.range_coarse_accuracy, "range_coarse_brier": self.range_coarse_brier, "cfr_advantage_loss": self.cfr_advantage_loss, "average_strategy_loss": self.average_strategy_loss, "cfr_effective_weight": self.cfr_effective_weight, "distributional_value_loss": self.distributional_value_loss, "belief_log_loss": self.belief_log_loss, "belief_action_accuracy": self.belief_action_accuracy, "benchmark_leader": self.benchmark_leader, "benchmark_bb_per_100": self.benchmark_bb_per_100, "tournament_count": self.tournament_count, "policy_loss": self.policy_loss, "value_loss": self.value_loss, "entropy": self.entropy, "kl_divergence": self.kl_divergence, "steps_per_second": self.steps_per_second, "parameters": parameters, "last_error": self.last_error, "report_filename": self.report_filename, "report_status": self.report_status}
            result.update({"ppo_tensor_preparation_seconds": self.ppo_tensor_preparation_seconds, "ppo_transfer_seconds": self.ppo_transfer_seconds, "ppo_compute_seconds": self.ppo_compute_seconds, "auxiliary_learning_seconds": self.auxiliary_learning_seconds, "optimizer_seconds": self.optimizer_seconds, "optimizer_backend": self.optimizer_backend})
            result.update({"rollout_model_sync_seconds": self.rollout_model_sync_seconds, "rollout_arena_setup_seconds": self.rollout_arena_setup_seconds, "rollout_tensor_preparation_seconds": self.rollout_tensor_preparation_seconds, "rollout_inference_dispatch_seconds": self.rollout_inference_dispatch_seconds, "rollout_action_postprocess_seconds": self.rollout_action_postprocess_seconds, "rollout_rule_execution_seconds": self.rollout_rule_execution_seconds, "rollout_play_seconds": self.rollout_play_seconds, "rollout_worker_seconds": self.rollout_worker_seconds, "rollout_dispatch_wait_seconds": self.rollout_dispatch_wait_seconds, "rollout_cached_opponent_models": self.rollout_cached_opponent_models})
            result.update({"population_size": self.population_size, "active_population": self.active_population, "population_diversity": self.population_diversity, "population_continuity": self.population_continuity, "snapshot_count": self.snapshot_count, "snapshot_diversity": self.snapshot_diversity, "snapshot_min_distance": self.snapshot_min_distance, "snapshot_rejections": self.snapshot_rejections, "holdout_score": self.holdout_score, "holdout_floor": self.holdout_floor, "holdout_bb_per_100": self.holdout_bb_per_100, "holdout_floor_bb_per_100": self.holdout_floor_bb_per_100, "holdout_ci_floor_bb_per_100": self.holdout_ci_floor_bb_per_100, "holdout_paired_variance": self.holdout_paired_variance, "continuous_raise_mean": self.continuous_raise_mean, "scenario_coverage": self.scenario_coverage, "restricted_br_bb_per_100": self.restricted_br_bb_per_100, "adversarial_floor_bb_per_100": self.adversarial_floor_bb_per_100, "adversarial_ci_floor_bb_per_100": self.adversarial_ci_floor_bb_per_100, "adversarial_evaluation_hands": self.adversarial_evaluation_hands, "adversarial_rollout_fraction": self.adversarial_rollout_fraction, "adversarial_rollout_target": self.adversarial_rollout_target, "crossplay_robustness": self.crossplay_robustness, "strategy_memory_size": self.strategy_memory_size, "search_memory_size": self.search_memory_size, "sizing_cfr_loss": self.sizing_cfr_loss, "search_value_loss": self.search_value_loss, "ensemble_disagreement": self.ensemble_disagreement, "adaptive_action_width": self.adaptive_action_width, "endgame_worlds": self.endgame_worlds, "training_focus": self.training_focus, "weakness_score": self.weakness_score, "adaptive_workers": self.adaptive_workers, "adaptive_batch_hands": self.adaptive_batch_hands, "rollout_decisions_per_second": self.rollout_decisions_per_second, "teacher_data_records": self.teacher_data_records, "teacher_data_filename": self.teacher_data_filename, "teacher_data_status": self.teacher_data_status, "audit_score": self.audit_score, "audit_exploitability_bb_per_100": self.audit_exploitability_bb_per_100, "scenario_gate": self.scenario_gate, "ablation_delta": self.ablation_delta, "evaluation_history_size": self.evaluation_history_size, "subgame_policy_loss": self.subgame_policy_loss, "subgame_teacher_size": self.subgame_teacher_size, "rollout_arena_width": self.rollout_arena_width, "average_strategy_weight": self.average_strategy_weight, "oracle_policy_loss": self.oracle_policy_loss, "oracle_value_loss": self.oracle_value_loss, "oracle_teacher_size": self.oracle_teacher_size, "oracle_confidence": self.oracle_confidence, "oracle_iterations": self.oracle_iterations, "abstraction_nash_conv": self.abstraction_nash_conv, "abstraction_value": self.abstraction_value, "abstraction_information_sets": self.abstraction_information_sets, "paired_deal_coverage": self.paired_deal_coverage, "rollout_backend": self.rollout_backend, "belief_posterior_support": self.belief_posterior_support, "resolver_replay_confidence": self.resolver_replay_confidence, "resolver_replay_size": self.resolver_replay_size, "blueprint_score": self.blueprint_score, "blueprint_confidence": self.blueprint_confidence, "blueprint_floor": self.blueprint_floor, "blueprint_hands": self.blueprint_hands, "blueprint_status": self.blueprint_status, "kuhn_value_gap": self.kuhn_value_gap, "counterfactual_value_loss": self.counterfactual_value_loss, "counterfactual_memory_size": self.counterfactual_memory_size, "counterfactual_coverage": self.counterfactual_coverage, "resolver_safety_rejections": self.resolver_safety_rejections, "abstract_cfr_teacher_mode": self.abstract_cfr_teacher_mode})
            result.update({"exploiter_threat": self.exploiter_threat, "champion_vulnerability": self.champion_vulnerability, "exploiter_generations": self.exploiter_generations, "resolver_safety_margin": self.resolver_safety_margin, "resolver_safety_confidence": self.resolver_safety_confidence, "resolver_confident_actions": self.resolver_confident_actions, "resolver_iterations": self.resolver_iterations, "resolver_strategy_peak": self.resolver_strategy_peak, "public_belief_teacher_size": self.public_belief_teacher_size, "sizing_proposal_diversity": self.sizing_proposal_diversity})
            result.update({"adversarial_focus": self.adversarial_focus, "compiled_transition_fraction": self.compiled_transition_fraction, "ppo_kl_limited": self.ppo_kl_limited, "ppo_hard_kl": self.ppo_hard_kl, "ppo_epoch_budget": self.ppo_epoch_budget, "ppo_recovery_updates": self.ppo_recovery_updates, "ppo_update_reverted": self.ppo_update_reverted, "ppo_rollback_phase": self.ppo_rollback_phase, "ppo_post_step_retry_applied": self.ppo_post_step_retry_applied, "ppo_post_step_retry_accepted": self.ppo_post_step_retry_accepted, "ppo_post_step_retry_kl": self.ppo_post_step_retry_kl})
            result.update({"adversarial_confirmation_hands": self.adversarial_confirmation_hands, "final_audit_ran": self.final_audit_ran, "adversarial_rollout_run_fraction": self.adversarial_rollout_run_fraction, "tail_loss_rate": self.tail_loss_rate, "tail_loss_bb": self.tail_loss_bb, "tail_policy_weight": self.tail_policy_weight, "tail_loss_run_rate": self.tail_loss_run_rate, "tail_loss_run_bb": self.tail_loss_run_bb, "tail_policy_run_weight": self.tail_policy_run_weight, "hard_spot_value_loss": self.hard_spot_value_loss, "hard_spot_memory_size": self.hard_spot_memory_size, "behavior_action_agreement": self.behavior_action_agreement, "behavior_action_change_rate": self.behavior_action_change_rate, "behavior_raise_fraction_delta": self.behavior_raise_fraction_delta, "behavior_audit_states": self.behavior_audit_states, "preflop_open_cap_bb": self.preflop_open_cap_bb, "preflop_sizing_audit_roots": self.preflop_sizing_audit_roots, "preflop_normal_raise_rate": self.preflop_normal_raise_rate, "preflop_mean_raise_bb": self.preflop_mean_raise_bb, "preflop_p95_raise_bb": self.preflop_p95_raise_bb, "preflop_oversized_open_rate": self.preflop_oversized_open_rate, "preflop_cap_hit_rate": self.preflop_cap_hit_rate, "preflop_all_in_rate": self.preflop_all_in_rate, "preflop_three_bet_cap_pot_multiplier": self.preflop_three_bet_cap_pot_multiplier, "preflop_three_bet_audit_roots": self.preflop_three_bet_audit_roots, "preflop_three_bet_normal_raise_rate": self.preflop_three_bet_normal_raise_rate, "preflop_three_bet_mean_raise_to_pot": self.preflop_three_bet_mean_raise_to_pot, "preflop_three_bet_p95_raise_to_pot": self.preflop_three_bet_p95_raise_to_pot, "preflop_three_bet_cap_hit_rate": self.preflop_three_bet_cap_hit_rate, "preflop_three_bet_over_cap_rate": self.preflop_three_bet_over_cap_rate, "preflop_three_bet_minimum_override_rate": self.preflop_three_bet_minimum_override_rate, "preflop_three_bet_all_in_rate": self.preflop_three_bet_all_in_rate, "preflop_forced_root_fraction": self.preflop_forced_root_fraction, "preflop_scenario_audit_hands": self.preflop_scenario_audit_hands, "preflop_scenario_worst_lcb_bb_per_100": self.preflop_scenario_worst_lcb_bb_per_100, "preflop_scenario_worst_root": self.preflop_scenario_worst_root, "preflop_scenario_worst_style": self.preflop_scenario_worst_style, "preflop_allin_calibration_loss": self.preflop_allin_calibration_loss, "preflop_allin_stability_loss": self.preflop_allin_stability_loss, "preflop_guarded_allin_probability": self.preflop_guarded_allin_probability, "preflop_allin_target": self.preflop_allin_target, "preflop_guarded_state_fraction": self.preflop_guarded_state_fraction, "preflop_immediate_allin_rate": self.preflop_immediate_allin_rate, "preflop_immediate_allin_target": self.preflop_immediate_allin_target, "preflop_immediate_eligible_rate": self.preflop_immediate_eligible_rate, "preflop_root_promotion_lcb_floor": self.preflop_root_promotion_lcb_floor, "preflop_3bet_teacher_loss": self.preflop_3bet_teacher_loss, "preflop_3bet_teacher_eligible_roots": self.preflop_3bet_teacher_eligible_roots, "preflop_3bet_teacher_samples": self.preflop_3bet_teacher_samples, "preflop_3bet_teacher_coverage": self.preflop_3bet_teacher_coverage, "preflop_3bet_teacher_confidence": self.preflop_3bet_teacher_confidence, "preflop_3bet_teacher_raise_target": self.preflop_3bet_teacher_raise_target, "preflop_3bet_teacher_raise_advantage_bb": self.preflop_3bet_teacher_raise_advantage_bb, "preflop_3bet_teacher_actual_raise_rate": self.preflop_3bet_teacher_actual_raise_rate, "preflop_3bet_teacher_allin_target": self.preflop_3bet_teacher_allin_target, "preflop_3bet_teacher_actual_allin_rate": self.preflop_3bet_teacher_actual_allin_rate, "robust_policy_weight": self.robust_policy_weight, "rollout_inference_device": self.rollout_inference_device})
            capabilities = active_rollout_capabilities()
            result.update({"rollout_backend": capabilities.mode, "rollout_backend_reason": capabilities.reason})
            if runtime:
                result.update(runtime)
            return result


def train_neural(trainer: StrategicLeagueTrainer, status: TrainingStatus, episodes: int, smoke_test: bool = False) -> None:
    """Collect recurrent rollouts and learn through staged strategic self-play."""
    error: str | None = None
    completed = 0
    last_checkpoint_completed = 0
    last_losses: dict | None = None
    safety_stopped = False
    log_training_debug("trainer_entered", episodes=episodes, smoke_test=smoke_test, cuda_enabled=trainer.runtime.cuda_enabled, device=str(trainer.runtime.device))
    try:
        if trainer.recovery_revalidation_required:
            baseline_stage = 0
            baseline_phase = CURRICULUM_PHASES[baseline_stage][2]
            log_training_debug(
                "policy_execution_baseline_revalidation_started",
                policy_execution_version=POLICY_EXECUTION_VERSION,
                stage=baseline_stage,
                phase=baseline_phase,
            )
            baseline = trainer.verify_recovery_baseline(baseline_stage, baseline_phase)
            trainer.recovery_revalidation_required = not bool(baseline.get("recovery_baseline_verified", False))
            log_training_debug(
                "policy_execution_baseline_revalidation_completed",
                verified=not trainer.recovery_revalidation_required,
                metrics=baseline.get("recovery_baseline_metrics", {}),
            )
            if not trainer.recovery_revalidation_required:
                trainer.save()
        if not trainer.recovery_baseline_verified or trainer.recovery_halted:
            raise RuntimeError("Recovered checkpoint has not passed its full safety baseline; choose a safer checkpoint before training.")
        initial_workers, _ = trainer.rollout_plan(episodes)
        trainer.reset_rollout_worker_cache()
        use_cached_opponents = initial_workers == 1 and trainer.rollout_inference_device() == "cuda"
        log_training_debug("rollout_pool_opening", episodes=episodes, workers=initial_workers, evaluation_workers=EVALUATION_WORKERS)
        with ProcessPoolExecutor(max_workers=initial_workers) as executor, ProcessPoolExecutor(max_workers=EVALUATION_WORKERS) as evaluation_executor:
            log_training_debug("training_pools_opened", rollout_workers=initial_workers, evaluation_workers=EVALUATION_WORKERS)
            while completed < episodes:
                stage, phase = trainer.curriculum(completed)
                workers, rollout_size = trainer.rollout_plan(episodes - completed)
                batch_hands = min(rollout_size, episodes - completed)
                chunks = [batch_hands // workers] * workers
                for index in range(batch_hands % workers):
                    chunks[index] += 1
                best_response_lane = trainer.select_training_lane()
                state, target_state, opponents, adaptive_scenarios, preflop_root_weights, oracle_snapshot, solver_snapshot = trainer.rollout_snapshot(best_response_lane, use_cached_opponents=use_cached_opponents)
                inference_device = trainer.rollout_inference_device()
                log_training_debug("rollout_submitting", completed=completed, batch_hands=batch_hands, workers=workers, chunks=chunks, stage=stage, phase=phase, inference_device=inference_device)
                collection_started = perf_counter()
                futures = [executor.submit(collect_rollouts_batched, state, target_state, opponents, chunk, 44_101 + completed + index, stage, trainer.updates + 1, best_response_lane, adaptive_scenarios, oracle_snapshot, solver_snapshot, preflop_root_weights, inference_device) for index, chunk in enumerate(chunks) if chunk]
                log_training_debug("rollout_submitted", completed=completed, futures=len(futures))
                results = [future.result() for future in futures]
                collection_seconds = perf_counter() - collection_started
                trainer.note_rollout_worker_cache([revision for result in results for revision in result.opponent_revisions])
                worker_seconds = max((result.worker_seconds for result in results), default=0.0)
                phase_timings = {
                    "rollout_model_sync_seconds": max((result.model_sync_seconds for result in results), default=0.0),
                    "rollout_arena_setup_seconds": max((result.arena_setup_seconds for result in results), default=0.0),
                    "rollout_tensor_preparation_seconds": max((result.tensor_preparation_seconds for result in results), default=0.0),
                    "rollout_inference_dispatch_seconds": max((result.inference_dispatch_seconds for result in results), default=0.0),
                    "rollout_action_postprocess_seconds": max((result.action_postprocess_seconds for result in results), default=0.0),
                    "rollout_rule_execution_seconds": max((result.rule_execution_seconds for result in results), default=0.0),
                    "rollout_play_seconds": max((result.play_seconds for result in results), default=0.0),
                    "rollout_worker_seconds": worker_seconds,
                    "rollout_dispatch_wait_seconds": max(0.0, collection_seconds - worker_seconds),
                    "rollout_cached_opponent_models": max((result.cached_opponent_models for result in results), default=0),
                }
                log_training_debug("rollout_completed", completed=completed, hands=sum(result.hands for result in results), actions=sum(result.actions for result in results), seconds=round(collection_seconds, 3), inference_device=inference_device, **{key: round(value, 4) if isinstance(value, float) else value for key, value in phase_timings.items()})
                trainer.note_rollout_throughput(sum(result.actions for result in results), collection_seconds, workers, batch_hands, inference_device)
                paths = [path for result in results for path in result.paths]
                learning_started = perf_counter()
                log_training_debug("learning_started", completed=completed, paths=len(paths))
                trainer.begin_learning_profile()
                losses = trainer.ppo_update(paths)
                auxiliary_learning_started = perf_counter()
                trainer.add_cfr_records([record for result in results for record in result.cfr_records])
                trainer.add_oracle_records([record for result in results for record in result.oracle_records])
                trainer.add_oracle_records([record for result in results for record in result.solver_records])
                trainer.add_action_likelihood_records([record for result in results for record in result.likelihood_records])
                trainer.add_imitation_paths(paths)
                trainer.add_hard_spot_paths(paths)
                ppo_update_reverted = bool(losses.get("ppo_update_reverted", False))
                if ppo_update_reverted:
                    # Preserve the complete policy rollback. These auxiliary
                    # objectives can touch the shared policy representation and
                    # therefore may only run after an accepted PPO step.
                    losses.update({
                        "cfr_advantage_loss": 0.0,
                        "average_strategy_loss": 0.0,
                        "sizing_cfr_loss": 0.0,
                        "cfr_memory_size": float(len(trainer.cfr_memory.records)),
                        "strategy_memory_size": float(len(trainer.strategy_memory.records)),
                        "cfr_effective_weight": 0.0,
                        "subgame_policy_loss": 0.0,
                        "subgame_teacher_size": 0.0,
                        "oracle_policy_loss": 0.0,
                        "oracle_value_loss": 0.0,
                        "oracle_teacher_size": float(len(trainer.abstract_teacher_memory.records)),
                        "oracle_confidence": trainer.last_oracle_confidence,
                        "search_value_loss": 0.0,
                        "search_memory_size": float(len(trainer.search_value_memory.records)),
                        "ensemble_disagreement": 0.0,
                        "counterfactual_value_loss": 0.0,
                        "counterfactual_memory_size": float(len(trainer.counterfactual_value_memory.records)),
                        "counterfactual_coverage": trainer.last_counterfactual_coverage,
                        "imitation_loss": 0.0,
                        "imitation_memory_size": float(len(trainer.imitation_memory.records)),
                        "imitation_reward": 0.0,
                    })
                    log_training_debug("policy_auxiliary_updates_skipped", completed=completed, rollback_phase=losses.get("ppo_rollback_phase", "unknown"))
                else:
                    for objective_name, objective in (
                        ("deep_cfr", trainer.deep_cfr_update),
                        ("subgame_policy", trainer.subgame_policy_update),
                        ("abstract_oracle", trainer.abstract_oracle_update),
                        ("search_value", trainer.search_value_update),
                        ("counterfactual_value", trainer.counterfactual_value_update),
                        ("self_imitation", trainer.self_imitation_update),
                    ):
                        try:
                            losses.update(objective())
                        except AuxiliaryUpdateRejected as exc:
                            log_training_debug("auxiliary_update_rejected", objective=objective_name, reason=str(exc), completed=completed)
                for objective_name, objective in (("belief", trainer.belief_update), ("hard_spot_value", trainer.hard_spot_value_update)):
                    try:
                        losses.update(objective())
                    except AuxiliaryUpdateRejected as exc:
                        log_training_debug("auxiliary_update_rejected", objective=objective_name, reason=str(exc), completed=completed)
                # After a rollback, only the isolated likelihood model and the
                # detached critic heads changed. Refreshing the target therefore
                # keeps branch-value supervision current without moving its policy.
                trainer.update_target_network()
                trainer.refresh_strategy_snapshots()
                batch_completed = sum(result.hands for result in results)
                completed += batch_completed
                trainer.note_rollouts(batch_completed)
                trainer.note_scenarios([result.scenario_counts for result in results])
                trainer.note_preflop_roots([result.preflop_root_counts for result in results])
                trainer.note_adversarial_rollouts(sum(result.adversarial_hands for result in results), batch_completed)
                trainer.note_compiled_transitions(sum(result.compiled_transition_actions for result in results), sum(result.actions for result in results))
                trainer.note_paired_deals(sum(result.paired_hands for result in results), batch_completed)
                trainer.advance_abstract_oracle()
                losses["rollout_seconds"] = collection_seconds
                losses.update(phase_timings)
                losses["auxiliary_learning_seconds"] = perf_counter() - auxiliary_learning_started
                losses.update(trainer.finish_learning_profile())
                losses["learning_seconds"] = perf_counter() - learning_started
                log_training_debug("learning_completed", completed=completed, seconds=round(float(losses["learning_seconds"]), 3), ppo_epochs=losses.get("ppo_epochs"), optimizer_backend=losses.get("optimizer_backend"))
                checkpoint_seconds = 0.0
                summary = trainer.evaluate_and_checkpoint(stage, phase, executor=evaluation_executor)
                log_training_debug("evaluation_completed", completed=completed, seconds=round(float(summary["evaluation_seconds"]), 3), parallel=bool(summary["parallel_evaluation"]))
                if bool(summary.get("recovery_halted", False)):
                    checkpoint_started = perf_counter()
                    trainer.save()
                    checkpoint_seconds = perf_counter() - checkpoint_started
                    losses["checkpoint_seconds"] = checkpoint_seconds
                    status.record(completed, sum(result.actions for result in results), summary, losses)
                    last_losses = losses
                    safety_stopped = True
                    error = "Safety stop: fixed-root fold collapse detected; all population members were restored to the recovery anchor."
                    log_training_debug("training_safety_stopped", completed=completed, checkpoint_seconds=round(checkpoint_seconds, 3), recovery_metrics=trainer.last_recovery_candidate_metrics)
                    break
                if CHECKPOINT_INTERVAL_HANDS and completed - last_checkpoint_completed >= CHECKPOINT_INTERVAL_HANDS:
                    checkpoint_started = perf_counter()
                    trainer.save()
                    checkpoint_seconds = perf_counter() - checkpoint_started
                    last_checkpoint_completed = completed
                    log_training_debug("checkpoint_completed", completed=completed, seconds=round(checkpoint_seconds, 3))
                losses["checkpoint_seconds"] = checkpoint_seconds
                status.record(completed, sum(result.actions for result in results), summary, losses)
                last_losses = losses
                log_training_debug("training_update_recorded", completed=completed, updates=trainer.updates, preflop_allin_probability=round(float(losses.get("preflop_guarded_allin_probability", 0.0)), 6), preflop_allin_stability_loss=round(float(losses.get("preflop_allin_stability_loss", 0.0)), 6))
            if completed and last_losses is not None and not safety_stopped and not smoke_test:
                final_stage, final_phase = trainer.curriculum()
                final_member = trainer.select_final_audit_member()
                log_training_debug("final_evaluation_started", completed=completed, stage=final_stage, phase=final_phase, **final_member)
                final_summary = trainer.evaluate_and_checkpoint(final_stage, final_phase, force_full=True, final_audit=True, executor=evaluation_executor)
                final_losses = dict(last_losses)
                checkpoint_started = perf_counter()
                trainer.save()
                final_losses["checkpoint_seconds"] = perf_counter() - checkpoint_started
                status.record(completed, 0, final_summary, final_losses)
                log_training_debug("final_evaluation_completed", completed=completed, evaluation_seconds=round(float(final_summary["evaluation_seconds"]), 3), checkpoint_seconds=round(float(final_losses["checkpoint_seconds"]), 3))
            elif completed and last_losses is not None and not safety_stopped:
                checkpoint_started = perf_counter()
                trainer.save()
                log_training_debug("smoke_test_finalization", completed=completed, checkpoint_seconds=round(perf_counter() - checkpoint_started, 3), promotion_eligible=False)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_training_debug("trainer_exception", completed=completed, exception_type=type(exc).__name__, exception_message=str(exc), traceback=traceback.format_exc(limit=20))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_training_debug("trainer_cancelled", completed=completed, exception_type=type(exc).__name__, exception_message=str(exc), traceback=traceback.format_exc(limit=20))
        raise
    finally:
        log_training_debug("trainer_finalizing", completed=completed, error=error)
        status.finish(error)
        try:
            parameter_count = sum(parameter.numel() for parameter in trainer.model.parameters())
            report_path = trainer.write_training_report(status.view(parameter_count, trainer.runtime_view()), episodes, completed, error)
            status.note_report(report_path.name, "written")
            log_training_debug("training_report_written", completed=completed, error=error, report_filename=report_path.name)
        except Exception as report_error:
            status.note_report("", f"failed: {report_error}")
            log_training_debug("training_report_failed", completed=completed, error=error, exception_type=type(report_error).__name__, exception_message=str(report_error), traceback=traceback.format_exc(limit=20))
