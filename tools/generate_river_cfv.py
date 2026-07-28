"""Resumable river CFV dataset generation (P3a).

A 1M-sample run is ~14 h on the 3060, so it must survive interruption: samples
are flushed to shards on disk and a manifest records progress, so re-invoking the
same command continues rather than restarting.

Defaults come from the measurements in docs/PLAN_V2_STRONGEST_PLAYER.md P3a:

* `--iterations 200` — target error is 0.93% of mean |value| against a
  4,000-iteration reference. River solves are deterministic, so that is pure
  convergence error with no Monte-Carlo floor beneath it. Supremus needed 4,000
  because its turn/flop targets had runout noise; the river does not.
* `--emit 0` — MEASURED DEFAULT. Multi-row-per-solve emission was tried twice and
  both variants produce biased targets: randomly blended ranges priced against
  the original solve are wrong by 50-171%, and interior-node harvesting (the
  correct form of the TurboReBeL idea) is still 12-17% off because a node's
  value under the solved average is the value of FOLLOWING that strategy, not
  the equilibrium value for the ranges arising there. Filtering by depth or
  reach mass does not help (16.2% near the root vs 17.0% deeper). Root rows
  are 0.93% accurate, so only they are generated.

Usage:
    python tools/generate_river_cfv.py --samples 1000000 --out backend/data/cfv/river
    python tools/generate_river_cfv.py --samples 1000000 --out backend/data/cfv/river   # resumes
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import backend.cfv.river_dataset as rd  # noqa: E402

SHARD_ROWS = 2_000
#: Also flush on a timer. A hard kill (Stop-Process /F, OOM, power loss) never
#: runs the `finally` block, so anything still buffered is lost — at 0.7 rows/s a
#: 2,000-row shard is ~48 minutes at risk. Two minutes is a cheap ceiling.
FLUSH_SECONDS = 120.0


def _manifest_path(out: Path, worker: int = 0) -> Path:
    return out / (f"manifest-w{worker}.json" if worker else "manifest.json")


def _load_manifest(out: Path, worker: int = 0) -> dict:
    path = _manifest_path(out, worker)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"shards": [], "rows": 0, "solves": 0, "seed_cursor": 0}


def _save_manifest(out: Path, manifest: dict, worker: int = 0) -> None:
    _manifest_path(out, worker).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _flush(out: Path, manifest: dict, buffer: list[dict], worker: int = 0) -> None:
    if not buffer:
        return
    index = len(manifest["shards"])
    # Worker-tagged names so parallel workers never collide on a shard file.
    path = out / f"river-w{worker}-{index:05d}.npz"
    np.savez_compressed(
        path,
        board=np.stack([row["board"] for row in buffer]),
        pot_bb=np.stack([row["pot_bb"] for row in buffer]),
        stack_bb=np.stack([row["stack_bb"] for row in buffer]),
        ranges=np.stack([row["ranges"] for row in buffer]),
        values=np.stack([row["values"] for row in buffer]),
        valid=np.packbits(np.stack([row["valid"] for row in buffer]), axis=1),
    )
    manifest["shards"].append(path.name)
    manifest["rows"] += len(buffer)
    buffer.clear()
    _save_manifest(out, manifest, worker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate river CFV training data")
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, default=Path("backend/data/cfv/river"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--emit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-every", type=int, default=200, help="solves between progress lines")
    parser.add_argument("--per-cell", type=int, default=250,
                        help="situations generated per (stack, pot) cell before moving on")
    # The GPU is latency-bound on these tiny trees: ~1.4 GB and ~90% of the card
    # idle per worker, so parallel workers interleave their kernels and scale
    # close to linearly. Workers take disjoint seed cursors and worker-tagged
    # shard names, so they never duplicate situations or collide on files.
    # Resuming is the default and deleting must be deliberate. Rows stay valid
    # across changes to worker count, per-cell size and report cadence; they are
    # invalidated ONLY by something that changes what a target means (the
    # postflop menu, the iteration count, or the pot normalisation).
    parser.add_argument("--reset", action="store_true",
                        help="discard existing shards first (only for target-invalidating changes)")
    parser.add_argument("--worker", type=int, default=0, help="this worker's index")
    parser.add_argument("--workers", type=int, default=1, help="total worker count")
    arguments = parser.parse_args()

    if arguments.reset and arguments.out.exists():
        import shutil

        shutil.rmtree(arguments.out)
    arguments.out.mkdir(parents=True, exist_ok=True)
    log_path = arguments.out / f"datagen-w{arguments.worker}.log"

    def emit(message: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(stamped, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    manifest = _load_manifest(arguments.out, arguments.worker)
    manifest.update(
        {"iterations": arguments.iterations, "emit": arguments.emit, "device": arguments.device}
    )
    if manifest["rows"]:
        emit(f"resuming: {manifest['rows']:,} rows in {len(manifest['shards'])} shards")

    buffer: list[dict] = []
    started = time.monotonic()
    last_flush = started
    start_rows = manifest["rows"]
    solves = 0
    # CELL-MAJOR generation. Each (stack, pot) cell pins a captured CUDA graph,
    # and only ~3 GB of this card is free, so caching all 36 cells filled VRAM
    # and collapsed throughput to 0.5 rows/s. Staying in one cell for a run of
    # situations amortises the ~0.5 s capture over many solves and keeps a
    # single solver alive. Cells are visited round-robin so coverage across
    # 20/50/100/200bb stays even if the run is interrupted.
    cells = rd.grid_cells()
    try:
        while manifest["rows"] + len(buffer) < arguments.samples // max(arguments.workers, 1):
            # Seed cursor persists in the manifest, so a resumed run explores new
            # situations instead of regenerating the ones already on disk.
            # Disjoint across workers: cursor c maps to global c * workers + worker.
            global_cursor = manifest["seed_cursor"] * arguments.workers + arguments.worker
            rng = random.Random(global_cursor * 2_654_435_761 + 12345)
            cell = cells[(global_cursor // arguments.per_cell) % len(cells)]
            manifest["seed_cursor"] += 1
            try:
                rows = rd.solve_situation(
                    rd.sample_situation(rng, cell=cell),
                    device=arguments.device,
                    iterations=arguments.iterations,
                    emit_iterates=arguments.emit,
                )
            except Exception as error:  # one bad situation must not end a 14 h run
                emit(f"  situation skipped: {type(error).__name__}: {str(error)[:100]}")
                continue
            buffer.extend(rows)
            solves += 1
            if len(buffer) >= SHARD_ROWS or (
                buffer and time.monotonic() - last_flush >= FLUSH_SECONDS
            ):
                _flush(arguments.out, manifest, buffer, arguments.worker)
                last_flush = time.monotonic()
            if solves % arguments.report_every == 0:
                done = manifest["rows"] + len(buffer)
                elapsed = time.monotonic() - started
                rate = (done - start_rows) / max(elapsed, 1e-9)
                remaining = (arguments.samples - done) / max(rate, 1e-9)
                emit(
                    f"  {done:,}/{arguments.samples:,} rows  {rate:.1f} rows/s  "
                    f"ETA {remaining / 3600:.1f} h"
                )
    except KeyboardInterrupt:
        print("\ninterrupted; flushing buffered rows")
    finally:
        _flush(arguments.out, manifest, buffer, arguments.worker)
        manifest["solves"] += solves
        _save_manifest(arguments.out, manifest, arguments.worker)

    elapsed = time.monotonic() - started
    print(
        f"\n{manifest['rows']:,} rows in {len(manifest['shards'])} shards "
        f"({manifest['solves']:,} solves total, {elapsed / 3600:.2f} h this session)"
    )


if __name__ == "__main__":
    main()
