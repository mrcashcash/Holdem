"""Run and monitor the 200bb no-limp blueprint-v3 + Phase 3 experiment.

The runner is deliberately resumable:

* train to 5,000 iterations, reporting only 1,000-iteration milestones;
* run the Phase 1 gate against the incumbent champion;
* resume training to 10,000 iterations; and
* run the second Phase 1 gate.

State, events, and full child-process output live beside the v3 checkpoint.
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
DATA_DIR = ROOT / "backend" / "data" / "gpu_blueprint_200bb_v3_phase3_nolimp"
CHECKPOINT = DATA_DIR / "checkpoint.npz"
SAMPLER_INIT = ROOT / "backend" / "data" / "gpu_blueprint_200bb_v3_nolimp" / "checkpoint.npz"
INCUMBENT = ROOT / "backend" / "data" / "gpu_blueprint_200bb_nolimp" / "champion.npz"
STATE_PATH = DATA_DIR / "experiment_state.json"
EVENTS_PATH = DATA_DIR / "experiment_events.jsonl"
LOG_PATH = DATA_DIR / "experiment.log"
LOCK_PATH = DATA_DIR / "experiment.lock"
TARGETS = (5_000, 10_000)
ITERATION_LINE = re.compile(r"^iter\s+(\d+)\s+\|")
REPORT_LINE = re.compile(r"\breport=(.+)$")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {
            "schema_version": 1,
            "experiment": "200bb_nolimp_v3_phase3",
            "created_utc": _utc_now(),
            "gates": {},
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
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"MONITOR {kind}{' ' + fields if fields else ''}", flush=True)


def _checkpoint_iteration() -> int:
    if not CHECKPOINT.exists():
        return 0
    with np.load(CHECKPOINT, allow_pickle=False) as payload:
        return int(payload["iteration"])


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
            raise RuntimeError(f"v3 experiment runner is already active (PID {owner})")
        LOCK_PATH.unlink(missing_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))


def _run_child(command: list[str], mode: str) -> list[str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
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
                    _event("training_milestone", iteration=int(match.group(1)), detail=line)
                elif (
                    "fitting recursive potential-aware" in line
                    or "compact CFR tables:" in line
                    or "loaded compact" in line
                    or "phase3 action profile:" in line
                    or "imported fitted sampler" in line
                ):
                    print(line, flush=True)
            else:
                print(line, flush=True)
        return_code = process.wait()
        log.write(f"[{_utc_now()}] EXIT {return_code}\n")
    if return_code != 0:
        tail = "\n".join(output[-20:])
        raise RuntimeError(f"{mode} process failed with exit code {return_code}\n{tail}")
    return output


def _train_to(target: int, state: dict[str, Any]) -> None:
    current = _checkpoint_iteration()
    if current > target:
        raise RuntimeError(
            f"checkpoint is already at {current:,}; cannot reconstruct the {target:,} gate checkpoint"
        )
    if current == target:
        _event("training_target_already_reached", iteration=target)
        return
    delta = target - current
    state["active_stage"] = f"train_to_{target}"
    _write_state(state)
    _event("training_started", current_iteration=current, target_iteration=target, delta=delta)
    command = [
        str(PYTHON),
        "-m",
        "backend.solver.gpu.train",
        "--iterations",
        str(delta),
        "--device",
        "cuda",
        "--stack-bb",
        "200",
        "--ruleset",
        "nolimp",
        "--abstraction",
        "v3",
        "--phase3-actions",
        "--tag",
        "v3_phase3_nolimp",
        "--sampler-init",
        str(SAMPLER_INIT),
        "--batch-boards",
        "1",
        "--save-every",
        "1000",
    ]
    _run_child(command, "train")
    reached = _checkpoint_iteration()
    if reached != target:
        raise RuntimeError(f"training ended at iteration {reached:,}, expected {target:,}")
    state["checkpoint_iteration"] = reached
    _write_state(state)
    _event("training_target_reached", iteration=reached)


def _gate_summary(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    screen = report.get("screen") or {}
    confirm = report.get("confirm") or {}
    promotion = report.get("promotion") or {}
    return {
        "report_path": str(report_path.resolve()),
        "screen_bb_per_100": screen.get("mean_bb_per_100"),
        "screen_verdict": screen.get("verdict"),
        "confirm_bb_per_100": confirm.get("mean_bb_per_100"),
        "confirm_ci_low": confirm.get("ci_low_bb_per_100"),
        "confirm_ci_high": confirm.get("ci_high_bb_per_100"),
        "confirm_verdict": confirm.get("verdict"),
        "mapping_ok": (report.get("mapping_gate") or {}).get("ok"),
        "lbr_ok": (report.get("lbr_gate") or {}).get("ok"),
        "eligible": promotion.get("eligible", False),
    }


def _run_gate(target: int, state: dict[str, Any]) -> None:
    gate_key = str(target)
    existing = state.setdefault("gates", {}).get(gate_key)
    if existing and existing.get("completed"):
        _event("gate_already_completed", iteration=target, report=existing.get("report_path"))
        return
    current = _checkpoint_iteration()
    if current != target:
        raise RuntimeError(f"gate {target:,} requires checkpoint iteration {target:,}, found {current:,}")
    state["active_stage"] = f"gate_at_{target}"
    _write_state(state)
    _event("gate_started", iteration=target)
    command = [
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
    ]
    output = _run_child(command, "gate")
    report_path: Path | None = None
    for line in reversed(output):
        match = REPORT_LINE.search(line)
        if match:
            report_path = Path(match.group(1).strip())
            break
    if report_path is None or not report_path.exists():
        raise RuntimeError("gate completed without a readable report path")
    summary = _gate_summary(report_path)
    state["gates"][gate_key] = {"completed": True, **summary}
    state["active_stage"] = None
    _write_state(state)
    _event(
        "gate_completed",
        iteration=target,
        eligible=summary["eligible"],
        report=summary["report_path"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the monitored v3 + Phase 3 experiment")
    parser.add_argument(
        "--stop-after",
        type=int,
        choices=TARGETS,
        default=10_000,
        help="last training/gate milestone to run in this invocation",
    )
    arguments = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON.exists():
        raise FileNotFoundError(f"workspace Python was not found: {PYTHON}")
    if not INCUMBENT.exists():
        raise FileNotFoundError(f"incumbent champion was not found: {INCUMBENT}")
    if not SAMPLER_INIT.exists():
        raise FileNotFoundError(f"fitted v3 sampler checkpoint was not found: {SAMPLER_INIT}")
    _acquire_lock()
    state = _read_state()
    try:
        state.pop("completed_utc", None)
        state.pop("last_error", None)
        state["requested_stop_after"] = arguments.stop_after
        _write_state(state)
        _event(
            "runner_started",
            pid=os.getpid(),
            checkpoint_iteration=_checkpoint_iteration(),
            stop_after=arguments.stop_after,
        )
        for target in (value for value in TARGETS if value <= arguments.stop_after):
            _train_to(target, state)
            _run_gate(target, state)
        state["active_stage"] = None
        state["completed_utc"] = _utc_now()
        state["completed_through"] = arguments.stop_after
        _write_state(state)
        _event(
            "requested_run_completed",
            checkpoint_iteration=_checkpoint_iteration(),
            completed_through=arguments.stop_after,
        )
    except BaseException as error:
        state["active_stage"] = "failed"
        state["last_error"] = str(error)
        _write_state(state)
        _event("experiment_failed", error=repr(error))
        raise
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
