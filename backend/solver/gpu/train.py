"""GPU blueprint training CLI (docs/GPU_CFR_PLAN.md §Phases 4-5).

    python -m backend.solver.gpu.train --iterations 2000 [--device cuda]

Artifacts in backend/data/gpu_blueprint/ (separate from the CPU blueprint —
the abstractions differ): checkpoint.npz holds the dense tensors plus the
tree/sampler configuration; telemetry.json appends one row per save.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gpu_blueprint"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.npz"
TELEMETRY_PATH = DATA_DIR / "telemetry.json"

DEFAULT_CONFIG = GpuActionConfig(
    preflop_fractions=(0.75, 1.5),
    postflop_fractions=(0.5, 1.0),
    max_raises_per_street=3,
    stack_bb=100.0,  # matches the serving game: 2000 chips at a 20-chip big blind
)
DEFAULT_SAMPLER = dict(flop_buckets=20, turn_buckets=20, river_buckets=20, flop_samples=8, turn_samples=8)


def configure_stack(stack_bb: float) -> None:
    """Retarget the trainer at a different stack depth with its own artifacts.

    100bb keeps the canonical paths (the serving default); other depths get
    backend/data/gpu_blueprint_<N>bb/ so multi-stack blueprints coexist.
    """
    global DEFAULT_CONFIG, DATA_DIR, CHECKPOINT_PATH, TELEMETRY_PATH
    from dataclasses import replace

    DEFAULT_CONFIG = replace(DEFAULT_CONFIG, stack_bb=float(stack_bb))
    if stack_bb != 100.0:
        DATA_DIR = DATA_DIR.parent / f"gpu_blueprint_{int(stack_bb)}bb"
        CHECKPOINT_PATH = DATA_DIR / "checkpoint.npz"
        TELEMETRY_PATH = DATA_DIR / "telemetry.json"


def build_solver(device: str = "cuda", seed: int = 0, batch_boards: int = 1) -> VectorCFR:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tree = BettingTree(DEFAULT_CONFIG)
    # averaging_delay: the earliest strategies are noise; keep them out of the
    # average (Supremus' DCFR+ delayed averaging).
    solver = VectorCFR(
        tree,
        DealSampler(**DEFAULT_SAMPLER),
        device=device,
        seed=seed,
        averaging_delay=1000,
        batch_boards=batch_boards,
    )
    if CHECKPOINT_PATH.exists():
        payload = np.load(CHECKPOINT_PATH, allow_pickle=False)
        stored = json.loads(str(payload["config"]))
        # JSON turns tuples into lists; normalize both sides before comparing.
        if stored != json.loads(json.dumps(asdict(DEFAULT_CONFIG))):
            raise RuntimeError(
                "gpu_blueprint checkpoint was trained with a different action config; "
                "delete backend/data/gpu_blueprint to start fresh"
            )
        solver.regrets = torch.tensor(payload["regrets"], device=solver.device)
        solver.strategy_sums = torch.tensor(payload["strategy_sums"], device=solver.device)
        solver.iteration = int(payload["iteration"])
        if "reach_normalized" not in payload:
            # One-time migration: reach is now probability-normalized, which
            # scales every future increment by ~1/1081; scale the stored
            # tensors by the same constant so past and future stay consistent.
            solver.regrets /= 1081.0
            solver.strategy_sums /= 1081.0
    return solver


def save_solver(solver: VectorCFR) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        regrets=solver.regrets.cpu().numpy(),
        strategy_sums=solver.strategy_sums.cpu().numpy(),
        iteration=solver.iteration,
        config=json.dumps(asdict(DEFAULT_CONFIG)),
        sampler=json.dumps(DEFAULT_SAMPLER),
        reach_normalized=True,
    )
    temporary.replace(CHECKPOINT_PATH)
    # Keep sparse history so convergence trends can be probed retroactively.
    if solver.iteration % 20_000 == 0:
        import shutil

        shutil.copy2(CHECKPOINT_PATH, DATA_DIR / f"checkpoint-{solver.iteration}.npz")


def train(
    iterations: int,
    device: str = "cuda",
    save_every: int = 200,
    seed: int = 0,
    progress: bool = True,
    batch_boards: int = 1,
) -> VectorCFR:
    solver = build_solver(device=device, seed=seed, batch_boards=batch_boards)
    completed = 0
    started = time.time()
    while completed < iterations:
        chunk = min(save_every, iterations - completed)
        chunk_started = time.time()
        solver.run(chunk)
        completed += chunk
        save_solver(solver)
        rate = chunk / (time.time() - chunk_started)
        # The gate costs ~2-3 minutes; every other checkpoint is plenty for
        # the trend while giving those minutes back to training.
        exploitability_mbb = None
        if (completed // save_every) % 2 == 1 or completed >= iterations:
            from backend.solver.gpu.exploit import cfr_br_exploitability

            exploitability_mbb = round(cfr_br_exploitability(solver, br_iterations=60, eval_boards=8), 2)
        record = {
            "iteration": solver.iteration,
            "iterations_per_second": round(rate, 3),
            "device": str(solver.device),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if exploitability_mbb is not None:
            record["exploitability_mbb_per_hand"] = exploitability_mbb
        history = json.loads(TELEMETRY_PATH.read_text(encoding="utf-8")) if TELEMETRY_PATH.exists() else []
        history.append(record)
        TELEMETRY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if progress:
            print(f"iter {solver.iteration} | {rate:.2f}/s | elapsed {time.time() - started:.0f}s")
    return solver


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dense GPU blueprint")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-boards", type=int, default=1, help="boards per iteration (mini-batch)")
    parser.add_argument("--stack-bb", type=float, default=100.0, help="stack depth; non-100 gets its own artifact dir")
    arguments = parser.parse_args()
    if arguments.stack_bb != 100.0:
        configure_stack(arguments.stack_bb)
    train(
        arguments.iterations,
        device=arguments.device,
        save_every=arguments.save_every,
        seed=arguments.seed,
        batch_boards=arguments.batch_boards,
    )


if __name__ == "__main__":
    main()
