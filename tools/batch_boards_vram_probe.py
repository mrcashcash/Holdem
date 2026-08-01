"""Measure how VRAM and throughput scale with --batch-boards.

The question this exists to answer: 50bb and 100bb leave 18 GB and 10 GB of a 24 GB
card idle. Can that be spent on a better solve?

`--batch-boards` is the only candidate. `tools/bucket_sizing_probe.py` rules out
the card abstraction on exact numbers -- buckets move a ~250 MiB term inside a
~23 GB budget, so no bucket count reaches the idle memory. Batching does scale:
B boards fold into the combo axis (width = B x NUM_COMBOS) as B chance samples of
one game sharing a single regret table, so VRAM is linear in B and each iteration's
regret update averages B boards instead of 1.

WHETHER THAT IS A WIN IS NOT SETTLED, and this probe is how to find out rather
than assume. `cfr.py:59` claims batching is nearly free because these solves are
latency-bound (~530 dependent ops at ~26.5us => ~14 ms/iteration). Measured here at
20bb/histogram on an RTX 3060, one iteration takes ~730 ms -- 52x that model -- and
the rate falls close to 1/B (1.37, 0.73, 0.49, 0.37 it/s at B = 1..4). If that
holds on a clean host, batching buys no extra board throughput at all: it only
trades more-and-noisier iterations for fewer-and-cleaner ones at constant cost,
which is a convergence question, not a free lunch. The comment's profiling was
evidently taken on a different tree or abstraction and does not describe this one.

Two caveats before trusting a number from here:

  * The rate column is confoundable. Deal sampling is CPU work proportional to B on
    ONE producer thread (cfr.py:311), so a loaded host starves the GPU and produces
    the same 1/B curve as a compute-bound card. The gpu-util column exists to tell
    those apart -- the first run of this probe was taken with 6 duel shards on 6 of
    8 cores, which is exactly the situation that fakes the result.
  * A short probe understates a long run's peak. The canonical 200bb job is living
    at ~66 KB/node while a 3-iteration probe of that shape reported 66.6 and older
    runs peaked near 75. Treat the memory number as a floor and leave headroom.

Run it at a depth that FITS the local card: the constant is per node-board, so 20bb
extrapolates to 50/100/200bb without needing a 24 GB card to measure it.

Usage:
    python tools/batch_boards_vram_probe.py --depth 20 --batches 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _start_utilization_sampler():
    """Poll GPU utilization on a thread so the probe can tell why the rate moved.

    A rate that falls as 1/B has two completely different causes -- a GPU that is
    genuinely compute-bound, and a starved deal producer on a busy host -- and they
    are indistinguishable from the rate alone.
    """
    import subprocess
    import threading

    samples: list[int] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(0.5):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip():
                    samples.append(int(out.stdout.strip().splitlines()[0]))
            except Exception:
                return

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    return thread, stop, samples


def _stop_utilization_sampler(handle) -> str:
    thread, stop, samples = handle
    stop.set()
    thread.join(timeout=3)
    if not samples:
        return "util n/a"
    return f"util {sum(samples) / len(samples):.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="VRAM and rate vs batch_boards")
    parser.add_argument("--depth", type=float, default=20.0)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler-init", type=Path,
                        default=Path("backend/data/gpu_blueprint_20bb/champion.npz"),
                        help="checkpoint to import fitted histogram centroids from; "
                             "pass '' to use the scalar sampler instead")
    arguments = parser.parse_args()

    import json

    import numpy as np
    import torch

    from backend.solver.gpu import train
    from backend.solver.gpu.cfr import VectorCFR
    from backend.solver.gpu.deals import DealSampler
    from backend.solver.gpu.tree import BettingTree

    def build_sampler() -> DealSampler:
        """Fitted histogram centroids if available, else the scalar sampler.

        The working set is bucket-count independent (see bucket_sizing_probe), so
        the scaling constant this probe measures is unaffected by which sampler is
        used -- but the histogram path REFUSES to run without fitted centroids, and
        refitting them costs minutes and host RAM. Importing them is the cheap way
        to measure the real code path.
        """
        if arguments.sampler_init and str(arguments.sampler_init):
            path = arguments.sampler_init
            if not path.exists():
                raise SystemExit(f"sampler-init checkpoint not found: {path}")
            with np.load(path, allow_pickle=False) as payload:
                if "sampler" not in payload:
                    raise SystemExit(f"no sampler state in {path}")
                sampler = DealSampler.from_state(json.loads(str(payload["sampler"])))
            print(f"imported fitted sampler from {path}: buckets={sampler.bucket_counts()}")
            return sampler
        print("using the scalar sampler (working set does not depend on buckets)")
        return DealSampler()

    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")

    config = (train.BLUEPRINT_CONFIG_20 if arguments.depth == 20.0
              else replace(train.DEFAULT_CONFIG, stack_bb=arguments.depth))
    tree = BettingTree(config)
    nodes = len(tree)
    total = (torch.cuda.get_device_properties(0).total_memory / 1024**3
             if arguments.device == "cuda" else 0.0)
    print(f"{arguments.depth:.0f}bb, {nodes:,} nodes, {config.num_actions} actions, "
          f"{arguments.iterations} iterations per point")
    print(f"device: {torch.cuda.get_device_name(0) if arguments.device == 'cuda' else 'cpu'} "
          f"({total:.1f} GB total)")
    print()
    print(f"{'B':>3}  {'alloc peak':>11}  {'reserved':>10}  {'KB/node/board':>14}  "
          f"{'rate':>10}  {'gpu':>9}")

    baseline = None
    for batch in arguments.batches:
        if arguments.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        solver = VectorCFR(tree, build_sampler(),
                           device=arguments.device, seed=0, batch_boards=batch)
        try:
            solver.run(2)  # warm the kernels outside the clock
            sampler_thread = _start_utilization_sampler()
            started = time.time()
            solver.run(arguments.iterations)
            elapsed = time.time() - started
            utilization = _stop_utilization_sampler(sampler_thread)
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:>3}  {'OOM':>11}")
            del solver
            continue
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        per = allocated * 1024**2 / (nodes * batch)
        rate = arguments.iterations / elapsed if elapsed else 0.0
        if baseline is None:
            baseline = allocated
        print(f"{batch:>3}  {allocated:>8.2f} GB  {reserved:>7.2f} GB  "
              f"{per:>11.1f} KB  {rate:>7.2f} it/s  {utilization:>9}")
        del solver

    if baseline:
        print()
        print("If KB/node/board is flat, VRAM is linear in B and the constant")
        print("extrapolates to any depth: budget = nodes x B x constant.")
        print()
        print("Read the util column before believing the rate column. Deal sampling is")
        print("CPU work proportional to B on ONE producer thread (cfr.py:311), so a busy")
        print("host starves the GPU and fakes a rate that falls as 1/B -- which is also")
        print("exactly what a genuinely compute-bound GPU looks like. High util means the")
        print("rate is real; low util means you measured your own background load.")


if __name__ == "__main__":
    main()
