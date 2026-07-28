"""Does exact-card turn+river solving fit on a 12GB card, and how fast?

P1.2's decisive question. The plan forbids trading card exactness for tree depth,
so if a configuration does not fit, the answer is a narrower betting menu or a
lower raise cap -- never coarser buckets.

Sweeps (menu x raise cap x depth), reporting for each:
  * tree size and decision nodes per street;
  * persistent table VRAM (regrets + strategy sums, the compact layout);
  * measured peak VRAM during a real solve;
  * seconds per CFR iteration and the implied latency at a given budget.

Usage:
    python tools/exact_turn_probe.py --iterations 20
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
from backend.solver.gpu.tree import (  # noqa: E402
    DECISION,
    BettingRootState,
    BettingTree,
    GpuActionConfig,
)

BOARD = (0, 17, 30, 43)  # a fixed rainbow-ish turn board

CONFIGS = [
    # label,                fractions,            cap, stack_bb, pot_bb
    ("2 sizes cap2 100bb", (0.5, 1.0), 2, 100.0, 20.0),
    ("2 sizes cap2 200bb", (0.5, 1.0), 2, 200.0, 20.0),
    ("3 sizes cap2 200bb", (0.33, 0.75, 1.5), 2, 200.0, 20.0),
    ("2 sizes cap3 200bb", (0.5, 1.0), 3, 200.0, 20.0),
    ("4 sizes cap2 200bb", (0.25, 0.5, 1.0, 1.5), 2, 200.0, 20.0),
    ("3 sizes cap3 200bb", (0.33, 0.75, 1.5), 3, 200.0, 20.0),
    ("4 sizes cap3 200bb", (0.25, 0.5, 1.0, 1.5), 3, 200.0, 20.0),
]


def probe(label, fractions, cap, stack_bb, pot_bb, iterations, device):
    config = GpuActionConfig(
        preflop_fractions=(1.0,),
        postflop_fractions=fractions,
        max_raises_per_street=cap,
        stack_bb=stack_bb,
    )
    behind = stack_bb - pot_bb / 2.0
    root = BettingRootState(
        street=2,
        to_act=1,
        committed=(pot_bb / 2.0, pot_bb / 2.0),
        street_commit=(0.0, 0.0),
        stacks=(behind, behind),
        acted=(False, False),
        raises=0,
        last_increment=1.0,
    )
    tree = BettingTree(config, root_state=root)
    decisions_by_street = {
        street: int(((tree.kind == DECISION) & (tree.street == street)).sum())
        for street in (2, 3)
    }
    record = {
        "config": label,
        "fractions": list(fractions),
        "raise_cap": cap,
        "stack_bb": stack_bb,
        "nodes": int(len(tree)),
        "decisions_turn": decisions_by_street[2],
        "decisions_river": decisions_by_street[3],
        "actions": config.num_actions,
    }

    sampler = ExactTurnSampler(BOARD)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        solver = VectorCFR(tree, sampler, device=device, seed=7, averaging_delay=2)
        storage = solver.storage_report()
        record["table_mib"] = round(storage["table_bytes_total"] / 2**20, 1)
        record["stored_rows"] = storage["total_rows"]

        started = time.monotonic()
        solver.run(iterations)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        record["eager_s_per_iteration"] = round(elapsed / max(iterations, 1), 4)
        record["eager_iterations_in_2s"] = int(2.0 / max(elapsed / iterations, 1e-9))

        # These trees are tiny and launch-overhead bound, which is exactly what
        # CUDA-graph capture exists for (the river resolver already relies on
        # it). Capture cost is paid once per solve, so it is charged here too.
        if device == "cuda":
            from backend.solver.gpu.graph import GraphRunner

            capture_started = time.monotonic()
            runner = GraphRunner(solver, warmup=2)
            record["graph_capture_s"] = round(time.monotonic() - capture_started, 3)
            solver.regrets.zero_()
            solver.strategy_sums.zero_()
            solver.iteration = 0
            torch.cuda.synchronize()
            started = time.monotonic()
            runner.run(iterations, random.Random(11))
            torch.cuda.synchronize()
            graph_elapsed = time.monotonic() - started
            record["graph_s_per_iteration"] = round(graph_elapsed / max(iterations, 1), 5)
            record["graph_speedup"] = round(elapsed / max(graph_elapsed, 1e-9), 2)
            # What a 2s decision budget buys once capture is amortized.
            usable = max(2.0 - record["graph_capture_s"], 0.0)
            record["graph_iterations_in_2s"] = int(usable / max(graph_elapsed / iterations, 1e-9))
            del runner

        if device == "cuda":
            record["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        record["status"] = "ok"
        del solver
    except torch.cuda.OutOfMemoryError as error:
        record["status"] = "OOM"
        record["error"] = str(error)[:120]
    except RuntimeError as error:  # non-typed CUDA OOM on some builds
        record["status"] = "OOM" if "out of memory" in str(error).lower() else "error"
        record["error"] = str(error)[:120]
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-card turn solve VRAM/latency probe")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if arguments.device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"device: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")

    records = []
    for entry in CONFIGS:
        record = probe(*entry, iterations=arguments.iterations, device=arguments.device)
        records.append(record)
        if record["status"] == "ok":
            print(
                f"{record['config']:22s} nodes={record['nodes']:>7,} "
                f"(turn {record['decisions_turn']:>5,} / river {record['decisions_river']:>6,})  "
                f"tables={record['table_mib']:>7.1f} MiB  "
                f"peak={record.get('peak_vram_mib', 0):>7.1f} MiB  "
                f"eager {record['eager_s_per_iteration']:.3f}s/it  "
                f"graph {record.get('graph_s_per_iteration', float('nan')):.5f}s/it "
                f"({record.get('graph_speedup', 0):.1f}x, capture {record.get('graph_capture_s', 0):.2f}s)  "
                f"=> {record.get('graph_iterations_in_2s', 0)} iters in 2s"
            )
        else:
            print(f"{record['config']:22s} nodes={record['nodes']:>7,}  {record['status']}: {record.get('error', '')}")
        sys.stdout.flush()

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nwritten to {arguments.output}")


if __name__ == "__main__":
    main()
