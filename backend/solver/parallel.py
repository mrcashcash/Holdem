"""Multiprocess blueprint training: K workers with periodic delta merging.

Regret and average-strategy updates are additive, so workers can traverse
independently and exchange increments: each round every worker runs a chunk
of iterations recording its table delta, the coordinator merges all deltas
into the master table, and each worker receives the other workers' deltas to
stay in sync. Windows-safe (spawn): workers rebuild the game from artifact
paths. Per-worker bucket caches stay warm for the whole run, which also
attacks the dominant cost (equity computation on cache misses).

Memory: every worker holds a full table copy (~130 MB per million infosets).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

from backend.solver.mccfr import LinearMCCFR


def _worker_main(
    connection,
    worker_id: int,
    abstraction_path: str,
    blueprint_path: str,
    seed: int,
    pruning_threshold: float | None,
    pruning_warmup: int,
) -> None:
    """Persistent worker: apply peers' deltas, run a chunk, ship own delta."""
    from backend.abstraction.buckets import CardAbstraction
    from backend.solver.blueprint import STACK_BB, load_checkpoint
    from backend.solver.holdem import AbstractHoldem

    abstraction = CardAbstraction.load(abstraction_path)
    game = AbstractHoldem(abstraction, stack_bb=STACK_BB)
    solver = LinearMCCFR(
        game,
        seed=seed,
        pruning_threshold=pruning_threshold,
        pruning_warmup_iterations=pruning_warmup,
    )
    loaded = load_checkpoint(Path(blueprint_path))
    if loaded is not None:
        solver.table, solver.iteration = loaded

    while True:
        message = connection.recv()
        command = message[0]
        if command == "run":
            _, base_iteration, chunk, peer_delta = message
            if peer_delta:
                solver.table.apply_delta(peer_delta)
            solver.iteration = base_iteration
            solver.table.begin_delta()
            solver.run(chunk)
            connection.send((worker_id, solver.table.collect_delta()))
        elif command == "stop":
            connection.close()
            return


def train_parallel(
    iterations: int,
    workers: int = 4,
    chunk: int = 250,
    save_every: int = 5000,
    seed: int = 0,
    progress: bool = True,
) -> None:
    """Run ``iterations`` weight-clock iterations across ``workers`` processes.

    Each round advances the Linear CFR weight clock by ``chunk`` while doing
    ``chunk * workers`` iterations of traversal work — the standard parallel
    MCCFR trade: more samples per weight step, slightly different averaging.
    """
    from backend.solver import blueprint as bp

    abstraction = bp.load_or_fit_abstraction(progress=progress)
    game = bp.build_game(abstraction)
    master = bp.load_solver(game, seed=seed)

    context = mp.get_context("spawn")
    pipes = []
    processes = []
    for worker_id in range(workers):
        parent_end, child_end = context.Pipe()
        process = context.Process(
            target=_worker_main,
            args=(
                child_end,
                worker_id,
                str(bp.ABSTRACTION_PATH),
                str(bp.BLUEPRINT_PATH),
                seed * 7919 + worker_id + 1,
                bp.PRUNING_THRESHOLD,
                bp.PRUNING_WARMUP,
            ),
            daemon=True,
        )
        process.start()
        pipes.append(parent_end)
        processes.append(process)

    previous_deltas: list[dict] = [{} for _ in range(workers)]
    completed = 0
    started = time.time()
    since_save = 0
    try:
        while completed < iterations:
            round_chunk = min(chunk, iterations - completed)
            for worker_id, pipe in enumerate(pipes):
                peers = {}
                for other_id, delta in enumerate(previous_deltas):
                    if other_id == worker_id:
                        continue
                    for key, change in delta.items():
                        if key in peers:
                            peers[key] = peers[key] + change
                        else:
                            peers[key] = change
                pipe.send(("run", master.iteration, round_chunk, peers))
            fresh: list[dict] = [{} for _ in range(workers)]
            for pipe in pipes:
                worker_id, delta = pipe.recv()
                fresh[worker_id] = delta
                master.table.apply_delta(delta)
            previous_deltas = fresh
            master.iteration += round_chunk
            completed += round_chunk
            since_save += round_chunk
            if since_save >= save_every or completed >= iterations:
                bp.save_solver(master)
                elapsed = time.time() - started
                record = {
                    "iteration": master.iteration,
                    "infosets": len(master.table),
                    "iterations_per_second": round(completed / elapsed, 2),
                    "workers": workers,
                    "traversal_iterations": completed * workers,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                bp.append_telemetry(record)
                since_save = 0
                if progress:
                    print(
                        f"iter {master.iteration} | {len(master.table)} infosets | "
                        f"{record['iterations_per_second']}/s weight-clock x{workers} workers"
                    )
    finally:
        for pipe in pipes:
            try:
                pipe.send(("stop",))
            except (OSError, BrokenPipeError):
                pass
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Linear MCCFR blueprint training")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=max(2, mp.cpu_count() - 2))
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    train_parallel(
        arguments.iterations,
        workers=arguments.workers,
        chunk=arguments.chunk,
        save_every=arguments.save_every,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
