"""Resume the native 20bb blueprint through 50k with promotion gates.

The current checkpoint is trained in increments to each 10k milestone.
Training saves every 1,000 iterations. At every milestone the full evaluation
gate compares the checkpoint with the last promoted 20bb champion and installs
the challenger only when all confirmatory gates pass.

Safe to restart: the checkpoint iteration determines the next increment and
completed milestone reports are not repeated.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "backend" / "data" / "gpu_blueprint_20bb"
CHECKPOINT = DATA_DIR / "checkpoint.npz"
TARGET = 50_000
MILESTONES = (10_000, 20_000, 30_000, 40_000, 50_000)
BASE_SEED = 20260729


def checkpoint_iteration() -> int:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    with np.load(CHECKPOINT, allow_pickle=False) as payload:
        return int(payload["iteration"])


def milestone_evaluated(iteration: int) -> bool:
    evaluations = DATA_DIR / "evaluations"
    return any(evaluations.glob(f"gate-*-iter{iteration}.json"))


def run(arguments: list[str]) -> None:
    print(f"$ {' '.join(arguments)}", flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    champion = DATA_DIR / "champion.npz"
    if not champion.exists():
        raise RuntimeError(
            "promote the eligible 5k bootstrap checkpoint before starting continuation"
        )

    for milestone in MILESTONES:
        current = checkpoint_iteration()
        if current < milestone:
            run(
                [
                    sys.executable,
                    "-m",
                    "backend.solver.gpu.train",
                    "--stack-bb",
                    "20",
                    "--abstraction",
                    "histogram",
                    "--iterations",
                    str(milestone - current),
                    "--save-every",
                    "1000",
                    "--batch-boards",
                    "1",
                    "--device",
                    "cuda",
                    "--seed",
                    str(BASE_SEED),
                ]
            )
        elif current > milestone:
            continue

        if not milestone_evaluated(milestone):
            run(
                [
                    sys.executable,
                    "-m",
                    "backend.eval.gate",
                    "--data-dir",
                    str(DATA_DIR),
                    "--stack-bb",
                    "20",
                    "--screen-pairs",
                    "750",
                    "--confirm-pairs",
                    "3000",
                    "--lbr-pairs",
                    "400",
                    "--seed",
                    str(BASE_SEED + milestone * 1000),
                    "--promote",
                ]
            )

    final_iteration = checkpoint_iteration()
    if final_iteration != TARGET:
        raise RuntimeError(f"expected iteration {TARGET}, found {final_iteration}")
    summary = {
        "data_dir": str(DATA_DIR),
        "iteration": final_iteration,
        "milestones": list(MILESTONES),
        "save_every": 1000,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
