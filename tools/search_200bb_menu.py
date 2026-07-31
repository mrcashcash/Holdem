"""Find the richest 200bb betting menu that still fits a 12 GB card.

Why this exists: `tools/size_blueprint_menus.py` tests four PRESET tiers, and at
200bb only the leanest fits -- 69,973 nodes against a 200,000-node comfort
ceiling. The next preset ("mid") is 537,733 nodes, so the presets jump 7.7x and
leave the entire 70k..200k band unexplored. That band is worth searching because
the 2026-07-31 LBR sweep put nearly all of the agent's weakness at depth:

    20bb  +13.34 [+6.93,+19.75]      100bb  +118.64 [+99.40,+137.87]
    50bb  +19.12 [+6.54,+31.70]      200bb  +252.45 [+223.34,+281.55]

and the one hardware fix (the 538k-node tier, ~37 GB once transients count) is
not available on a 12 GB card, nor on a 16 GB one.

The lever is `preflop_raise_cap`. Per the tree builder's own note and
docs/STATUS.md 6, preflop raise depth is cheap while postflop multiplies -- so a
richer preflop menu and a deeper preflop cap can be bought without paying the
postflop blowup. That is also where a known leak lives: opponent min-raises
exhaust the raise cap, and the resulting geometry mismatch is what produced the
24x-pot shove (STATUS.md 3.6). More preflop sizes and a deeper preflop cap attack
that structurally rather than with a sizing heuristic.

Ceilings match the preset tool so verdicts are comparable: 200,000 nodes and
2,500 MiB of tables.

Usage:
    python tools/search_200bb_menu.py --stack-bb 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.solver.gpu.tree import DECISION, BettingTree, GpuActionConfig  # noqa: E402

BUCKETS = (169, 384, 384, 192)
#: MEASURED, not assumed. A CUDA probe over three candidate trees on the RTX 3060
#: read a peak of 66.5-67.5 KB/node -- confirming STATUS.md 6's "empirically
#: ~60-75KB/node" -- with these peaks:
#:
#:    69,973 nodes ->  4,554 MiB      129,001 nodes ->  8,380 MiB
#:   195,751 nodes -> 12,896 MiB  (EXCEEDS the card's 12,287 MiB total)
#:
#: So a 200,000-node ceiling admits configurations that cannot run: 12,287 MiB
#: total, less ~2 GiB for the display, over 66.6 KB/node is ~157,000 nodes. Use
#: 150,000, which is also the comfort budget STATUS.md 6 already states.
NODE_CEILING = 150_000
TABLE_MIB_CEILING = 2_500

#: Candidate preflop menus, lean-first. The deployed 200bb lean menu is
#: (0.75, 1.5); everything after it adds sizes an opponent's raise war can land on.
PREFLOP_MENUS = (
    (0.75, 1.5),
    (0.5, 1.0, 1.5),
    (0.5, 0.75, 1.5),
    (0.5, 1.0, 1.5, 2.5),
)
#: Postflop stays at or near the deployed lean menu -- this is where nodes
#: multiply, so it is deliberately NOT the search axis.
POSTFLOP_MENUS = (
    (0.5, 1.0),
    (0.33, 0.75, 1.5),
)
#: (postflop cap, preflop cap). None means "same as postflop".
CAPS = ((2, None), (2, 3), (2, 4), (3, None))


def measure(stack: float, preflop, postflop, postflop_cap: int, preflop_cap: int | None):
    config = GpuActionConfig(
        preflop_fractions=preflop,
        postflop_fractions=postflop,
        max_raises_per_street=postflop_cap,
        preflop_raise_cap=preflop_cap,
        stack_bb=stack,
    )
    started = time.monotonic()
    try:
        tree = BettingTree(config)
    except MemoryError:
        return None
    build_s = time.monotonic() - started
    decisions = int((tree.kind == DECISION).sum())
    rows = sum(
        int(((tree.kind == DECISION) & (tree.street == street)).sum()) * BUCKETS[street]
        for street in range(4)
    )
    mib = rows * config.num_actions * 4 * 2 / 2**20
    return {
        "preflop": list(preflop),
        "postflop": list(postflop),
        "postflop_raise_cap": postflop_cap,
        "preflop_raise_cap": preflop_cap,
        "nodes": len(tree),
        "decisions": decisions,
        "table_mib": round(mib, 1),
        "num_actions": config.num_actions,
        "fits": len(tree) <= NODE_CEILING and mib <= TABLE_MIB_CEILING,
        "build_s": round(build_s, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the 200bb menu space")
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/menu-search-200bb.json"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"searching 200bb menu space against ceilings "
          f"{NODE_CEILING:,} nodes / {TABLE_MIB_CEILING:,} MiB tables")
    print(f"{'preflop':>22} {'postflop':>20} {'pf/pl cap':>10} "
          f"{'nodes':>10} {'decisions':>10} {'MiB':>7}  verdict")

    records = []
    for preflop, postflop, (postflop_cap, preflop_cap) in product(
            PREFLOP_MENUS, POSTFLOP_MENUS, CAPS):
        record = measure(arguments.stack_bb, preflop, postflop, postflop_cap, preflop_cap)
        if record is None:
            print(f"{str(preflop):>22} {str(postflop):>20} "
                  f"{f'{postflop_cap}/{preflop_cap}':>10} {'-':>10} {'-':>10} {'-':>7}  OOM")
            continue
        records.append(record)
        print(f"{str(preflop):>22} {str(postflop):>20} "
              f"{f'{postflop_cap}/{preflop_cap or postflop_cap}':>10} "
              f"{record['nodes']:>10,} {record['decisions']:>10,} "
              f"{record['table_mib']:>7.0f}  {'FITS' if record['fits'] else 'too big'}")

    affordable = [r for r in records if r["fits"]]
    # "Richest" = most decision nodes among those that fit; decision nodes are what
    # carry strategy, so they are the honest measure of resolution bought.
    affordable.sort(key=lambda r: r["decisions"], reverse=True)
    print()
    if affordable:
        best = affordable[0]
        baseline = next(
            (r for r in records
             if r["preflop"] == [0.75, 1.5] and r["postflop"] == [0.5, 1.0]
             and r["postflop_raise_cap"] == 2 and r["preflop_raise_cap"] is None),
            None,
        )
        print(f"RICHEST AFFORDABLE: preflop {best['preflop']} postflop {best['postflop']} "
              f"caps {best['postflop_raise_cap']}/{best['preflop_raise_cap'] or best['postflop_raise_cap']}")
        print(f"  {best['nodes']:,} nodes, {best['decisions']:,} decisions, "
              f"{best['table_mib']:.0f} MiB tables")
        if baseline:
            print(f"  deployed lean: {baseline['nodes']:,} nodes, "
                  f"{baseline['decisions']:,} decisions")
            gain = best["decisions"] / max(baseline["decisions"], 1)
            print(f"  -> {gain:.2f}x the decision nodes of the deployed menu")
            if gain < 1.05:
                print("  NOTE: no meaningful gain available in this space; the preset")
                print("  jump really is the binding constraint and richer 200bb play")
                print("  needs a larger card, not a better menu choice.")
        best["selected"] = True
    else:
        print("nothing in the searched space fits; lean remains the only option")

    arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"written to {arguments.output}")


if __name__ == "__main__":
    main()
