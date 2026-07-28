"""Resume the clean 200bb no-limp blueprint to a milestone and run its gate.

This long-running job is designed for hidden/background execution. Durable
state is written beside the checkpoint so a Codex completion monitor needs to
read only one small JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DATA_DIR = ROOT / "backend" / "data" / "gpu_blueprint_200bb_nolimp"
CHECKPOINT = DATA_DIR / "checkpoint.npz"
INCUMBENT = ROOT / "backend" / "data" / "gpu_blueprint_200bb" / "champion.npz"
STATE_PATH = DATA_DIR / "experiment_20k_state.json"
EVENTS_PATH = DATA_DIR / "experiment_20k_events.jsonl"
LOG_PATH = DATA_DIR / "experiment_20k.log"
LOCK_PATH = DATA_DIR / "experiment_20k.lock"
TARGET = 20_000
ITERATION_LINE = re.compile(r"^iter\s+(\d+)\s+\|")
REPORT_LINE = re.compile(r"\breport=(.+)$")


def _configure_target(target: int) -> None:
    global TARGET, STATE_PATH, EVENTS_PATH, LOG_PATH, LOCK_PATH
    if target <= 0 or target % 1_000:
        raise ValueError("target must be a positive multiple of 1,000")
    TARGET = target
    milestone = f"{target // 1_000}k"
    STATE_PATH = DATA_DIR / f"experiment_{milestone}_state.json"
    EVENTS_PATH = DATA_DIR / f"experiment_{milestone}_events.jsonl"
    LOG_PATH = DATA_DIR / f"experiment_{milestone}.log"
    LOCK_PATH = DATA_DIR / f"experiment_{milestone}.lock"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _checkpoint_iteration() -> int:
    if not CHECKPOINT.exists():
        return 0
    with np.load(CHECKPOINT, allow_pickle=False) as payload:
        return int(payload["iteration"])


def _read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {
            "schema_version": 1,
            "experiment": f"clean_200bb_nolimp_to_{TARGET}",
            "created_utc": _utc_now(),
        }
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _write_state(state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc_now()
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def _event(kind: str, **details: Any) -> None:
    record = {"timestamp": _utc_now(), "event": kind, **details}
    with EVENTS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"MONITOR {kind} {details}", flush=True)


def _process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and str(pid) in result.stdout


def _acquire_lock() -> None:
    if LOCK_PATH.exists():
        try:
            owner = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner = -1
        if owner > 0 and _process_exists(owner):
            raise RuntimeError(
                f"no-limp {TARGET // 1_000}k runner is already active (PID {owner})"
            )
        LOCK_PATH.unlink(missing_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))


def _run_child(command: list[str], mode: str) -> list[str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["HOLDEM_SUBGAME_ITERS"] = "0"
    environment["HOLDEM_PHASE4_RIVER"] = "0"
    output: list[str] = []
    with LOG_PATH.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] START {subprocess.list2cmdline(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            output.append(line)
            log.write(line + "\n")
            log.flush()
            if mode == "train":
                match = ITERATION_LINE.match(line)
                if match:
                    _event("training_milestone", iteration=int(match.group(1)))
            else:
                print(line, flush=True)
        return_code = process.wait()
        log.write(f"[{_utc_now()}] EXIT {return_code}\n")
    if return_code != 0:
        raise RuntimeError(
            f"{mode} failed with exit code {return_code}\n"
            + "\n".join(output[-30:])
        )
    return output


def _train(state: dict[str, Any]) -> None:
    current = _checkpoint_iteration()
    if current > TARGET:
        raise RuntimeError(
            f"checkpoint is already past the {TARGET // 1_000}k gate at {current}"
        )
    if current == TARGET:
        _event("training_target_already_reached", iteration=current)
        return
    state["active_stage"] = f"train_to_{TARGET}"
    state["checkpoint_iteration"] = current
    _write_state(state)
    _event("training_started", current_iteration=current, target_iteration=TARGET)
    _run_child(
        [
            str(PYTHON),
            "-m",
            "backend.solver.gpu.train",
            "--iterations",
            str(TARGET - current),
            "--device",
            "cuda",
            "--stack-bb",
            "200",
            "--ruleset",
            "nolimp",
            "--abstraction",
            "histogram",
            "--tag",
            "nolimp",
            "--batch-boards",
            "1",
            "--save-every",
            "1000",
        ],
        "train",
    )
    reached = _checkpoint_iteration()
    if reached != TARGET:
        raise RuntimeError(f"training ended at {reached}, expected {TARGET}")
    state["checkpoint_iteration"] = reached
    _write_state(state)
    _event("training_target_reached", iteration=reached)


def _gate(state: dict[str, Any]) -> None:
    state["active_stage"] = f"gate_at_{TARGET}"
    _write_state(state)
    _event("gate_started", iteration=TARGET)
    output = _run_child(
        [
            str(PYTHON),
            "-m",
            "backend.eval.gate",
            "--data-dir",
            str(DATA_DIR),
            "--challenger",
            str(CHECKPOINT),
            "--incumbent",
            str(INCUMBENT),
            "--stack-bb",
            "200",
        ],
        "gate",
    )
    report_path: Path | None = None
    for line in reversed(output):
        match = REPORT_LINE.search(line)
        if match:
            report_path = Path(match.group(1).strip())
            break
    if report_path is None or not report_path.exists():
        raise RuntimeError(
            f"{TARGET // 1_000}k gate completed without a readable report"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    screen = report.get("screen") or {}
    confirm = report.get("confirm") or {}
    state["gate"] = {
        "report_path": str(report_path.resolve()),
        "screen_bb_per_100": screen.get("mean_bb_per_100"),
        "screen_ci_low": screen.get("ci_low_bb_per_100"),
        "screen_ci_high": screen.get("ci_high_bb_per_100"),
        "screen_verdict": screen.get("verdict"),
        "confirm_bb_per_100": confirm.get("mean_bb_per_100"),
        "confirm_ci_low": confirm.get("ci_low_bb_per_100"),
        "confirm_ci_high": confirm.get("ci_high_bb_per_100"),
        "confirm_verdict": confirm.get("verdict"),
        "mapping_ok": (report.get("mapping_gate") or {}).get("ok"),
        "lbr_ok": (report.get("lbr_gate") or {}).get("ok"),
        "eligible": (report.get("promotion") or {}).get("eligible", False),
    }
    _write_state(state)
    _event("gate_completed", report=state["gate"]["report_path"])


def main(target: int = TARGET) -> None:
    _configure_target(target)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON.exists():
        raise FileNotFoundError(f"workspace Python not found: {PYTHON}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"10k no-limp checkpoint not found: {CHECKPOINT}")
    if not INCUMBENT.exists():
        raise FileNotFoundError(f"118k incumbent not found: {INCUMBENT}")

    _acquire_lock()
    state = _read_state()
    try:
        state.pop("completed_utc", None)
        state.pop("last_error", None)
        state["active_stage"] = "starting"
        _write_state(state)
        _event("runner_started", pid=os.getpid(), checkpoint_iteration=_checkpoint_iteration())
        _train(state)
        _gate(state)
        state["active_stage"] = None
        state["completed_through"] = TARGET
        state["completed_utc"] = _utc_now()
        _write_state(state)
        _event("experiment_completed", checkpoint_iteration=_checkpoint_iteration())
    except BaseException as error:
        state["active_stage"] = "failed"
        state["last_error"] = str(error)
        _write_state(state)
        _event("experiment_failed", error=repr(error))
        raise
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET)
    arguments = parser.parse_args()
    main(arguments.target)
