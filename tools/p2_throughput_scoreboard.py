"""P2 throughput scoreboard — and the rent-a-GPU decision gate.

Measured facts this starts from (docs/PLAN_V2_STRONGEST_PLAYER.md P1.2/P2):

* a graph-captured 726-node exact turn solve runs at 13.5 ms/iteration, which is
  **123x above its ~0.11 ms bandwidth floor**, so these solves are
  occupancy-bound: thousands of tiny kernels, each leaving the GPU idle;
* CUDA-graph capture already removed launch overhead (2.6-12.5x), so what remains
  is kernel size, not kernel count.

The fix for an occupancy-bound workload is wider kernels. `VectorCFR.batch_boards`
already folds B boards into the combo axis, so every kernel becomes B x wider for
free. On a turn board each "board" is one of the 48 river runouts, so batching
also multiplies the CHANCE SAMPLES per iteration — convergence per unit time
improves twice over.

The headline metric is therefore not iterations/second but **chance samples per
second** (batch x iterations/s), because that is what drives convergence. A
secondary metric, iterations/second, still matters for the strategy update count.

This scoreboard is also the gate for the deferred rent-a-GPU decision: if the
per-solve budgets in the plan's section 4 are missed by more than ~3x after
batching, rent.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from backend.search.exact_turn import ExactTurnSampler  # noqa: E402
from backend.solver.gpu.cfr import VectorCFR  # noqa: E402
from backend.solver.gpu.graph import GraphRunner  # noqa: E402
from backend.solver.gpu.tree import (  # noqa: E402
    BettingRootState,
    BettingTree,
    GpuActionConfig,
)

BOARD = (0, 17, 30, 43)


def build_tree(fractions, cap, stack_bb, pot_bb):
    config = GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=fractions,
        max_raises_per_street=cap, stack_bb=stack_bb,
    )
    behind = stack_bb - pot_bb / 2.0
    root = BettingRootState(
        street=2, to_act=1, committed=(pot_bb / 2.0, pot_bb / 2.0),
        street_commit=(0.0, 0.0), stacks=(behind, behind),
        acted=(False, False), raises=0, last_increment=1.0,
    )
    return BettingTree(config, root_state=root)


def measure(tree, batch: int, iterations: int, device: str) -> dict:
    sampler = ExactTurnSampler(BOARD)
    record: dict = {"batch_boards": batch}
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        solver = VectorCFR(tree, sampler, device=device, seed=7,
                           averaging_delay=2, batch_boards=batch)
        if device == "cuda":
            runner = GraphRunner(solver, warmup=2)
            torch.cuda.synchronize()
            started = time.monotonic()
            runner.run(iterations, random.Random(11))
            torch.cuda.synchronize()
        else:
            started = time.monotonic()
            solver.run(iterations)
        elapsed = time.monotonic() - started
        per_iteration = elapsed / iterations
        record.update(
            {
                "s_per_iteration": round(per_iteration, 5),
                "iterations_per_s": round(1.0 / per_iteration, 1),
                # The metric that matters: chance outcomes consumed per second.
                "chance_samples_per_s": round(batch / per_iteration, 1),
                "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if device == "cuda" else None,
                "status": "ok",
            }
        )
        del solver
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        message = str(error)
        record.update({"status": "OOM" if "out of memory" in message.lower() else "error",
                       "error": message[:120]})
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 throughput scoreboard")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pot-bb", type=float, default=20.0)
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4, 8, 16, 24, 48])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    tree = build_tree((0.5, 1.0), 2, arguments.stack_bb, arguments.pot_bb)
    print(f"exact turn tree: {len(tree)} nodes @ {arguments.stack_bb:.0f}bb, "
          f"pot {arguments.pot_bb:.0f}bb, device {arguments.device}")
    print(f"{'batch':>6} {'s/iter':>9} {'iters/s':>9} {'chance/s':>10} {'speedup':>8} {'VRAM MiB':>9}")

    records = []
    baseline = None
    for batch in arguments.batches:
        record = measure(tree, batch, arguments.iterations, arguments.device)
        records.append(record)
        if record["status"] != "ok":
            print(f"{batch:>6} {record['status']}: {record.get('error', '')}")
            sys.stdout.flush()
            continue
        if baseline is None:
            baseline = record["chance_samples_per_s"]
        record["speedup_vs_batch1"] = round(record["chance_samples_per_s"] / baseline, 2)
        print(f"{batch:>6} {record['s_per_iteration']:>9.5f} {record['iterations_per_s']:>9.1f} "
              f"{record['chance_samples_per_s']:>10.1f} {record['speedup_vs_batch1']:>7.2f}x "
              f"{record['peak_vram_mib']:>9.1f}")
        sys.stdout.flush()

    ok = [r for r in records if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["chance_samples_per_s"])
        print(f"\nbest: batch={best['batch_boards']} -> {best['chance_samples_per_s']:.0f} chance samples/s "
              f"({best['speedup_vs_batch1']:.1f}x over batch=1), {best['peak_vram_mib']:.0f} MiB")
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"written to {arguments.output}")


if __name__ == "__main__":
    main()
