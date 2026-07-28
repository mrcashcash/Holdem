"""Run the frozen Phase 4 exact-river on/off confirmation in the background."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DATA_DIR = ROOT / "backend" / "data" / "gpu_blueprint_200bb"
CHECKPOINT = DATA_DIR / "champion.npz"
REPORT = DATA_DIR / "evaluations" / "phase4-confirm-3000pairs-6s.json"
STATE = DATA_DIR / "phase4_confirm_3000_state.json"
LOG = DATA_DIR / "phase4_confirm_3000.log"
LOCK = DATA_DIR / "phase4_confirm_3000.lock"
PAIRS = 3_000
SEED = 95_041_337
ITERATIONS = 80
BUDGET_MS = 6_000


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_state(state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc_now()
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def _process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and str(pid) in result.stdout


def _acquire_lock() -> None:
    if LOCK.exists():
        try:
            owner = int(LOCK.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = -1
        if owner > 0 and _process_exists(owner):
            raise RuntimeError(f"Phase 4 confirmation is already active (PID {owner})")
        LOCK.unlink(missing_ok=True)
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))


def main() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(f"workspace Python not found: {PYTHON}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"frozen 200bb champion not found: {CHECKPOINT}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    _acquire_lock()
    state: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "phase4_exact_river_confirmation",
        "checkpoint": str(CHECKPOINT.resolve()),
        "pairs": PAIRS,
        "hands": PAIRS * 2,
        "seed": SEED,
        "resolver_iterations": ITERATIONS,
        "resolver_budget_ms": BUDGET_MS,
        "active_stage": "phase4_confirmation",
        "started_utc": _utc_now(),
    }
    _write_state(state)

    command = [
        str(PYTHON),
        "-m",
        "backend.eval.river_search_ab",
        "--checkpoint",
        str(CHECKPOINT),
        "--stack-bb",
        "200",
        "--pairs",
        str(PAIRS),
        "--seed",
        str(SEED),
        "--iterations",
        str(ITERATIONS),
        "--budget-ms",
        str(BUDGET_MS),
        "--output",
        str(REPORT),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["HOLDEM_SUBGAME_ITERS"] = "0"
    environment["HOLDEM_PHASE4_RIVER"] = "0"

    try:
        with LOG.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Phase 4 confirmation exited with code {result.returncode}"
            )
        if not REPORT.exists():
            raise RuntimeError("Phase 4 confirmation finished without a report")

        report = json.loads(REPORT.read_text(encoding="utf-8"))
        diagnostics = report["diagnostics"]["river_search"]["challenger"]
        state.update(
            {
                "active_stage": None,
                "completed_utc": _utc_now(),
                "report_path": str(REPORT.resolve()),
                "mean_bb_per_100": report.get("mean_bb_per_100"),
                "ci_low_bb_per_100": report.get("ci_low_bb_per_100"),
                "ci_high_bb_per_100": report.get("ci_high_bb_per_100"),
                "verdict": report.get("verdict"),
                "eligible": report.get("eligible", False),
                "river_attempts": diagnostics.get("attempts"),
                "river_resolved": diagnostics.get("resolved"),
                "river_fallbacks": diagnostics.get("fallbacks"),
                "river_fallback_rate": diagnostics.get("fallback_rate"),
                "mean_elapsed_ms": diagnostics.get("mean_elapsed_ms"),
                "max_elapsed_ms": diagnostics.get("max_elapsed_ms"),
            }
        )
        _write_state(state)
    except BaseException as error:
        state["active_stage"] = "failed"
        state["last_error"] = str(error)
        _write_state(state)
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
