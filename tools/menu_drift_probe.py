"""Which betting menu survives a sequence of min-raises without drifting?

The river translation drift measured on 2026-07-31 (median 3.33 at 200bb, 100% of
river decisions above 1.5x) is a MENU-COVERAGE failure, not a choice failure. Branch
counts over 120 hands versus a min-raiser:

    forced to the smallest fraction (observed below the menu)   652
    exact match                                                198
    genuine choice between two fractions                         0

A min-raise is roughly 0.1x pot; the deployed 200bb menu's smallest postflop raise
is 0.5x. Every min-raise is forced up ~5x, and it compounds.

Testing a candidate menu by training a blueprint on it costs about four hours at
200bb. That is unnecessary: drift depends only on the TREE and the TRANSLATION, not
on the learned strategy. So drive a synthetic min-raise sequence through
`_translate_event` on each candidate tree and measure the geometry drift it produces.
Structural, exact, and free.

Reported per menu: node count (against the 150,000 ceiling measured at 66.6 KB/node
on a 12 GB card) and the drift after each successive min-raise.

Usage:
    python tools/menu_drift_probe.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent  # noqa: E402
from backend.solver.gpu.tree import (  # noqa: E402
    DECISION,
    STREET_END,
    BettingTree,
    GpuActionConfig,
)

NODE_CEILING = 150_000
KB_PER_NODE = 66.6

#: (label, preflop fractions, postflop fractions, postflop cap, preflop cap)
MENUS = (
    ("DEPLOYED 0.5/1.0 cap3", (0.75, 1.5), (0.5, 1.0), 3, None),
    ("0.33/1.0 cap2", (0.75, 1.5), (0.33, 1.0), 2, None),
    ("0.25/0.75 cap2", (0.75, 1.5), (0.25, 0.75), 2, None),
    ("0.25/1.0 cap2", (0.75, 1.5), (0.25, 1.0), 2, None),
    ("0.2/0.6 cap2", (0.75, 1.5), (0.2, 0.6), 2, None),
    ("0.33/1.0 cap2 pfcap3", (0.75, 1.5), (0.33, 1.0), 2, 3),
)

BIG_BLIND = 20


def _stub(tree: BettingTree):
    """Minimal object exposing exactly what _translate_event reads."""
    stub = SimpleNamespace(tree=tree, _node_pot_cache={})
    stub._translate_event = lambda node, game, event, rng: (
        GpuBlueprintAgent._translate_event(stub, node, game, event, rng)
    )
    stub._abstract_matched_pot = lambda node: GpuBlueprintAgent._abstract_matched_pot(
        stub, node
    )
    return stub


def _event(amount: int, pot_before: int, to_call_before: int, current_bet_before: int) -> dict:
    return {
        "action": "raise",
        "amount": amount,
        "pot_before": pot_before,
        "to_call_before": to_call_before,
        "current_bet_before": current_bet_before,
    }


def probe_menu(label, preflop, postflop, cap, preflop_cap, raises: int, stack_bb: float):
    config = GpuActionConfig(
        preflop_fractions=preflop, postflop_fractions=postflop,
        max_raises_per_street=cap, preflop_raise_cap=preflop_cap, stack_bb=stack_bb,
    )
    tree = BettingTree(config)
    stub = _stub(tree)
    rng = random.Random(0)

    stack_chips = int(stack_bb) * BIG_BLIND
    node = tree.root
    # A min-raise war: each player raises by exactly one big blind over the last.
    current_bet, pot, drifts = BIG_BLIND, 3 * BIG_BLIND // 2, []
    for index in range(raises):
        while tree.kind[node] == STREET_END:
            node = int(tree.children[node][0])
        if tree.kind[node] != DECISION:
            break
        target = current_bet + BIG_BLIND
        event = _event(target, pot, max(current_bet - (current_bet - BIG_BLIND), 0), current_bet - BIG_BLIND)
        action = stub._translate_event(node, SimpleNamespace(pot=pot), event, rng)
        child = int(tree.children[node][action])
        if child < 0:
            break
        node = child
        pot += target - (current_bet - BIG_BLIND)
        current_bet = target

        # Real geometry after this raise: matched pot is twice the smaller
        # commitment, and each player still has the stack minus their commitment.
        matched_real = 2.0 * (current_bet - BIG_BLIND) if index else 2.0 * BIG_BLIND
        matched_real = max(matched_real, float(BIG_BLIND))
        ratio_real = stack_chips / matched_real
        try:
            matched_abstract = float(stub._abstract_matched_pot(node))
        except Exception:
            break
        if matched_abstract <= 0:
            continue
        ratio_abstract = stack_bb / matched_abstract
        drifts.append(round(ratio_real / max(ratio_abstract, 1e-9), 3))
    return tree, drifts


def main() -> None:
    parser = argparse.ArgumentParser(description="Menu coverage vs translation drift")
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--raises", type=int, default=6,
                        help="successive min-raises to drive through the translation")
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/menu-drift-probe.json"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"synthetic min-raise war at {arguments.stack_bb:.0f}bb, "
          f"{arguments.raises} successive min-raises")
    print(f"drift 1.0 = located node matches reality's stack-to-pot geometry\n")
    print(f"{'menu':24} {'nodes':>9} {'GB':>5} {'fits':>5}  drift after each min-raise")

    records = []
    for label, preflop, postflop, cap, preflop_cap in MENUS:
        tree, drifts = probe_menu(label, preflop, postflop, cap, preflop_cap,
                                  arguments.raises, arguments.stack_bb)
        gb = len(tree) * KB_PER_NODE / 1024 / 1024
        fits = "yes" if len(tree) <= NODE_CEILING else "NO"
        worst = max(drifts) if drifts else float("nan")
        print(f"{label:24} {len(tree):>9,} {gb:>5.1f} {fits:>5}  "
              f"{[f'{d:.2f}' for d in drifts]}")
        records.append({
            "menu": label, "preflop": list(preflop), "postflop": list(postflop),
            "postflop_cap": cap, "preflop_cap": preflop_cap,
            "nodes": len(tree), "est_gb": round(gb, 2), "fits_12gb": len(tree) <= NODE_CEILING,
            "drifts": drifts, "worst_drift": None if not drifts else worst,
        })

    affordable = [r for r in records if r["fits_12gb"] and r["worst_drift"] is not None]
    print()
    if affordable:
        best = min(affordable, key=lambda r: r["worst_drift"])
        deployed = next((r for r in records if r["menu"].startswith("DEPLOYED")), None)
        print(f"lowest worst-drift among menus that fit: {best['menu']} "
              f"(worst {best['worst_drift']:.2f}, {best['nodes']:,} nodes)")
        if deployed and deployed["worst_drift"]:
            print(f"deployed menu worst drift: {deployed['worst_drift']:.2f} "
                  f"({deployed['nodes']:,} nodes)")
            if best["menu"] != deployed["menu"]:
                print(f"-> a candidate reduces worst drift "
                      f"{deployed['worst_drift'] / max(best['worst_drift'], 1e-9):.1f}x "
                      f"at {best['nodes'] / deployed['nodes']:.2f}x the tree size.")
                print("   Training it is now justified; drift alone is NOT a promotion")
                print("   signal, so a duel and an LBR run still decide.")
            else:
                print("-> no candidate beats the deployed menu on drift. The menu is not")
                print("   the lever, or this synthetic sequence does not capture the cause.")
    arguments.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nwritten to {arguments.output}")


if __name__ == "__main__":
    main()
