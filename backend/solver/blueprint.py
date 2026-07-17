"""Blueprint training: Linear MCCFR over abstracted heads-up NLHE.

Run as a module for long training sessions (checkpointable and resumable):

    python -m backend.solver.blueprint --iterations 100000
    python -m backend.solver.blueprint --iterations 0 --status   # inspect

Artifacts under backend/data/blueprint/:
    abstraction.npz   card abstraction (fitted once, reused)
    blueprint.pkl     regret + average-strategy tables and iteration count
    telemetry.json    convergence telemetry appended per save interval

Convergence is monotone in expectation and every checkpoint is playable;
strength evaluation lives in backend/eval (fixed styles + LBR probe).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from backend.abstraction.actions import ActionAbstraction
from backend.abstraction.buckets import AbstractionConfig, CardAbstraction
from backend.solver.holdem import AbstractHoldem
from backend.solver.mccfr import LinearMCCFR

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "blueprint"
ABSTRACTION_PATH = DATA_DIR / "abstraction.npz"
BLUEPRINT_PATH = DATA_DIR / "blueprint.pkl"
TELEMETRY_PATH = DATA_DIR / "telemetry.json"

# NOTE: the serving game moved to 100 bb (2000 chips at 10/20). Existing CPU
# checkpoints were trained at 50 bb and are depth-mismatched at the table;
# the GPU blueprint (trained at 100 bb) supersedes them. Change this only
# together with a fresh backend/data/blueprint/ — checkpoints embed the depth.
STACK_BB = 50.0
PRUNING_THRESHOLD = -1.5
PRUNING_WARMUP = 20000


def load_or_fit_abstraction(progress: bool = True) -> CardAbstraction:
    if ABSTRACTION_PATH.exists():
        return CardAbstraction.load(ABSTRACTION_PATH)
    if progress:
        print("fitting card abstraction (one-time, a few minutes)...")
    abstraction = CardAbstraction(config=AbstractionConfig()).fit(progress=progress)
    abstraction.save(ABSTRACTION_PATH)
    return abstraction


def build_game(abstraction: CardAbstraction | None = None) -> AbstractHoldem:
    return AbstractHoldem(
        abstraction=abstraction or load_or_fit_abstraction(),
        actions=ActionAbstraction(),
        stack_bb=STACK_BB,
    )


def load_checkpoint(path: Path | None = None) -> tuple["StrategyTable", int] | None:
    """Load (table, iteration), migrating legacy tuple keys to packed bytes.

    A checkpoint written before key packing stores holdem infoset keys as
    (street, bucket, history) tuples; they are re-encoded once here and the
    migrated checkpoint is written back so other processes load it directly.
    """
    from backend.solver.holdem import pack_infoset_key
    from backend.solver.mccfr import StrategyTable

    checkpoint = path or BLUEPRINT_PATH
    if not checkpoint.exists():
        return None
    # Local trainer artifact only; see StrategyTable.save on the pickle trust boundary.
    with open(checkpoint, "rb") as handle:
        payload = pickle.load(handle)
    table: StrategyTable = payload["table"]
    iteration: int = payload["iteration"]
    sample = next(iter(table.rows), None)
    if isinstance(sample, tuple):
        table.remap_keys(lambda key: pack_infoset_key(key[0], key[1], key[2]))
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        with open(temporary, "wb") as migrated:
            pickle.dump({"table": table, "iteration": iteration}, migrated, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint)
    return table, iteration


def load_solver(game: AbstractHoldem, seed: int = 0) -> LinearMCCFR:
    solver = LinearMCCFR(
        game,
        seed=seed,
        pruning_threshold=PRUNING_THRESHOLD,
        pruning_warmup_iterations=PRUNING_WARMUP,
    )
    loaded = load_checkpoint()
    if loaded is not None:
        solver.table, solver.iteration = loaded
    return solver


def save_solver(solver: LinearMCCFR) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = BLUEPRINT_PATH.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump({"table": solver.table, "iteration": solver.iteration}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(BLUEPRINT_PATH)


def append_telemetry(record: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    if TELEMETRY_PATH.exists():
        history = json.loads(TELEMETRY_PATH.read_text(encoding="utf-8"))
    history.append(record)
    TELEMETRY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def train(iterations: int, save_every: int = 5000, seed: int = 0, progress: bool = True) -> LinearMCCFR:
    abstraction = load_or_fit_abstraction(progress=progress)
    game = build_game(abstraction)
    solver = load_solver(game, seed=seed)
    start_iteration = solver.iteration
    started = time.time()
    completed = 0
    while completed < iterations:
        chunk = min(save_every, iterations - completed)
        chunk_started = time.time()
        solver.run(chunk)
        completed += chunk
        save_solver(solver)
        elapsed = time.time() - chunk_started
        record = {
            "iteration": solver.iteration,
            "infosets": len(solver.table),
            "iterations_per_second": round(chunk / elapsed, 2),
            "bucket_cache": len(abstraction._cache),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        append_telemetry(record)
        if progress:
            total_rate = completed / max(time.time() - started, 1e-9)
            print(
                f"iter {solver.iteration} (+{completed}) | {len(solver.table)} infosets | "
                f"{record['iterations_per_second']}/s (avg {total_rate:.1f}/s)"
            )
    if progress and iterations:
        print(f"trained {iterations} iterations ({start_iteration} -> {solver.iteration})")
    return solver


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hold'em blueprint with Linear MCCFR")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--status", action="store_true", help="print checkpoint status and exit")
    arguments = parser.parse_args()
    if arguments.status:
        if BLUEPRINT_PATH.exists():
            with open(BLUEPRINT_PATH, "rb") as handle:
                payload = pickle.load(handle)
            print(f"iteration={payload['iteration']} infosets={len(payload['table'])}")
        else:
            print("no blueprint checkpoint yet")
        return
    train(arguments.iterations, save_every=arguments.save_every, seed=arguments.seed)


if __name__ == "__main__":
    main()
