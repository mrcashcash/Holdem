"""Which betting menu is affordable at each target depth (20/50/100/200bb)?

Target depths are 20, 50, 100 and 200bb, but there is no blueprint below 100bb —
shallow hands currently route to the 100bb model, which is badly mismatched.
Shallow trees are far smaller, so the shallow depths can afford a RICHER menu
than the deep ones, and this sizes that before any GPU time is spent.

The plan's standing rule applies: never trade card exactness for tree size; cap
the betting menu instead. Table memory assumes the v3 bucket profile
(169 preflop / 384 flop / 384 turn / 192 river) over decision nodes only.

Writes a durable report next to the other evaluations so the choice is auditable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.solver.gpu.tree import DECISION, BettingTree, GpuActionConfig  # noqa: E402

BUCKETS = (169, 384, 384, 192)
MENUS = (
    ("lean", (0.75, 1.5), (0.5, 1.0), 2),
    ("mid", (0.5, 1.0, 1.5), (0.33, 0.75, 1.5), 2),
    ("mid+cap3", (0.5, 1.0, 1.5), (0.33, 0.75, 1.5), 3),
    ("rich", (0.5, 0.75), (0.33, 0.66, 1.0, 1.5), 2),
)
# ~150k nodes is the documented comfort ceiling with the server up; tables must
# also leave room for transients on a 12 GB card shared with the desktop.
#
# Lowered from 200,000 on 2026-07-31 after a CUDA probe MEASURED the real peak at
# 66.5-67.5 KB/node: 195,751 nodes peaked at 12,896 MiB, which exceeds the RTX
# 3060's 12,287 MiB total outright. The old ceiling therefore reported "FITS" for
# a tree that cannot run at all. 12,287 MiB less ~2 GiB of display headroom over
# 66.6 KB/node is ~157,000 nodes, so 150,000 is both safe and the figure
# STATUS.md 6 already gives.
NODE_CEILING = 150_000
TABLE_MIB_CEILING = 2_500


def main() -> None:
    parser = argparse.ArgumentParser(description="Size blueprint menus per depth")
    parser.add_argument("--depths", type=float, nargs="*", default=[20.0, 50.0, 100.0, 200.0])
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/blueprint-menu-sizing.json"))
    arguments = parser.parse_args()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log = arguments.output.with_suffix(".log")

    def emit(message: str) -> None:
        stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(stamped, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    emit(f"sizing menus for depths {arguments.depths}")
    emit(f"{'depth':>6} {'menu':>9} {'nodes':>10} {'decisions':>11} {'tables MiB':>11}  verdict")
    records = []
    for stack in arguments.depths:
        best = None
        for name, preflop, postflop, cap in MENUS:
            config = GpuActionConfig(
                preflop_fractions=preflop, postflop_fractions=postflop,
                max_raises_per_street=cap, stack_bb=stack,
                no_donk_srp=(stack == 20.0 and name == "rich"),
            )
            started = time.monotonic()
            try:
                tree = BettingTree(config)
            except MemoryError:
                emit(f"{stack:>6.0f} {name:>9} {'-':>10} {'-':>11} {'-':>11}  OOM building tree")
                continue
            build_s = time.monotonic() - started
            decisions = int((tree.kind == DECISION).sum())
            rows = sum(
                int(((tree.kind == DECISION) & (tree.street == street)).sum()) * BUCKETS[street]
                for street in range(4)
            )
            mib = rows * config.num_actions * 4 * 2 / 2**20
            fits = len(tree) <= NODE_CEILING and mib <= TABLE_MIB_CEILING
            emit(f"{stack:>6.0f} {name:>9} {len(tree):>10,} {decisions:>11,} {mib:>11.0f}  "
                 f"{'FITS' if fits else 'too big'} (built in {build_s:.1f}s)")
            record = {
                "stack_bb": stack, "menu": name, "preflop": list(preflop),
                "postflop": list(postflop), "raise_cap": cap, "nodes": len(tree),
                "decisions": decisions, "table_mib": round(mib, 1), "fits": fits,
                "no_donk_srp": config.no_donk_srp,
            }
            records.append(record)
            if fits:
                best = record
        if best:
            emit(f"       -> richest affordable at {stack:.0f}bb: {best['menu']} "
                 f"({best['nodes']:,} nodes, {best['table_mib']:.0f} MiB)")
            best["selected"] = True

    arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    emit(f"written to {arguments.output}")


if __name__ == "__main__":
    main()
