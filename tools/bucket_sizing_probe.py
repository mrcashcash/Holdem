"""How many card buckets can each depth afford, and what does it actually cost?

The naive reading of "50bb only uses 5 GB of the 24 GB card, so raise the buckets
until it is full" is wrong about where the memory goes. Two very different terms:

  working set   nodes x NUM_COMBOS x 4 bytes x ~12 tensors  (reach, values,
                child_values, ...). Scales with the BETTING TREE and the hole-card
                combo axis. Measured at ~64 KB/node. **Independent of buckets.**
  tables        regrets + strategy_sums = total_rows x actions x 4 bytes x 2.
                Scales with BUCKETS. On the running canonical 200bb job this is
                253.3 MiB against a 23.1 GB peak -- about 1%.

So buckets are nearly free in VRAM, and "fill the card with buckets" would need
absurd counts to move the needle. The binding constraint on bucket count is
statistical, not spatial: every bucket splits the same sampled experience, so
doubling buckets halves the visits per bucket and needs more iterations to reach
the same per-bucket confidence. Spending idle VRAM is not free accuracy.

This probe reports both terms exactly. Layout row counts are computed on CPU from
the real CompactTableLayout, so they are not estimates; the working set uses the
measured 64 KB/node, which is itself a floor (peak has been seen at 75 KB/node,
and a 3-iteration probe understates a long run's peak).

Usage:
    python tools/bucket_sizing_probe.py --depths 50 100 200
    python tools/bucket_sizing_probe.py --depths 50 --grid
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.solver.gpu import train  # noqa: E402
from backend.solver.gpu.storage import CompactTableLayout  # noqa: E402
from backend.solver.gpu.tree import BettingTree  # noqa: E402

# Measured on real runs, not derived. The lower number is what the canonical 200bb
# job is currently living at; the higher is the worst peak observed.
KB_PER_NODE_TYPICAL = 64.0
KB_PER_NODE_PEAK = 75.0
CARD_GB = 24.0


def _table_mib(tree: BettingTree, buckets: tuple[int, int, int, int]) -> tuple[int, float]:
    layout = CompactTableLayout(tree, buckets)
    rows = int(layout.total_rows)
    # regrets + strategy_sums, float32, one column per action.
    mib = rows * tree.config.num_actions * 4 * 2 / 1024**2
    return rows, mib


def _config_for(depth: float):
    if depth == 20.0:
        return train.BLUEPRINT_CONFIG_20
    return replace(train.DEFAULT_CONFIG, stack_bb=depth)


def main() -> None:
    parser = argparse.ArgumentParser(description="bucket count vs VRAM, exactly")
    parser.add_argument("--depths", type=float, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--grid", action="store_true",
                        help="sweep candidate bucket counts, not just the current one")
    arguments = parser.parse_args()

    base = (train.HISTOGRAM_SAMPLER["flop_buckets"],
            train.HISTOGRAM_SAMPLER["turn_buckets"],
            train.HISTOGRAM_SAMPLER["river_buckets"])
    print(f"current histogram abstraction: 169 / {base[0]} / {base[1]} / {base[2]}")
    print(f"card budget: {CARD_GB:.0f} GB; working set charged at "
          f"{KB_PER_NODE_TYPICAL:.0f} KB/node (peak seen {KB_PER_NODE_PEAK:.0f})")
    print()

    candidates: list[tuple[int, int, int]] = [base]
    if arguments.grid:
        candidates += [
            (base[0], base[1], 60),      # river is the coarsest street by far
            (base[0], base[1], 120),
            (200, 200, 120),
            (256, 256, 180),
            (384, 384, 192),             # the v3 potential-aware target
            (512, 512, 384),
            (1024, 1024, 768),
        ]

    for depth in arguments.depths:
        config = _config_for(depth)
        tree = BettingTree(config)
        nodes = len(tree)
        working_gb = nodes * KB_PER_NODE_TYPICAL / 1024**2
        peak_gb = nodes * KB_PER_NODE_PEAK / 1024**2
        menu = "20bb exception" if depth == 20.0 else "canonical"
        print(f"=== {depth:.0f}bb ({menu}) — {nodes:,} nodes, "
              f"{config.num_actions} actions ===")
        print(f"    working set: {working_gb:.2f} GB typical, {peak_gb:.2f} GB at peak "
              f"-> headroom {CARD_GB - peak_gb:.2f} GB")
        print(f"    {'buckets':>18}  {'rows':>14}  {'tables':>10}  {'total@peak':>11}  fit")
        for flop, turn, river in candidates:
            rows, mib = _table_mib(tree, (169, flop, turn, river))
            total = peak_gb + mib / 1024
            mark = "ok" if total <= CARD_GB else "OVER"
            tag = "  <- current" if (flop, turn, river) == base else ""
            print(f"    {f'{flop}/{turn}/{river}':>18}  {rows:>14,}  "
                  f"{mib:>7.1f} MiB  {total:>8.2f} GB  {mark}{tag}")
        print()

    print("Reading this table: the tables column is the ONLY column buckets move.")
    print("Even a 1024/1024/768 abstraction — 7x the current flop/turn resolution —")
    print("costs a few GB, and none of it buys accuracy on its own: the same sampled")
    print("experience is being divided among more buckets. Raise buckets only")
    print("together with iterations, and only where a duel says the coarseness hurts.")


if __name__ == "__main__":
    main()
