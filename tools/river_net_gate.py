"""P3a acceptance gate: does a net-priced river horizon change the turn decision?

The plan is explicit that lower validation loss is NOT the acceptance criterion.
CFV v0 had a defensible-looking net (9.3 bb val MAE against a 24 bb zero
baseline) that still lost 65 bb/100 when actually used. What matters is whether
substituting the net for the real river subtree changes the action the turn solve
would have chosen.

Two metrics, reported per situation and aggregated:

* **action agreement** — for each live combo at the turn root, does the
  net-horizon solve put its mass on the same action as the full turn+river solve?
  Weighted by the combo's range mass, because a disagreement on a combo that is
  never held does not matter.
* **policy L1** — mean |p_net - p_full| summed over actions, the same scale used
  for the convergence ladder in P1.2 (max 2.0).

Also reports the speedup actually realised, since 9x fewer nodes is the entire
justification for the net existing.

Usage:
    python tools/river_net_gate.py --net backend/data/cfv/river_net.pt --situations 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from backend.search.exact_turn import TURN_FRACTIONS, TURN_RAISE_CAP, ExactTurnSampler  # noqa: E402
from backend.search.river_horizon import RiverNetEvaluator  # noqa: E402
from backend.solver.gpu.cfr import VectorCFR  # noqa: E402
from backend.solver.gpu.deals import NUM_COMBOS, CARD_IN_COMBO  # noqa: E402
from backend.solver.gpu.tree import (  # noqa: E402
    DECISION,
    BettingRootState,
    BettingTree,
    GpuActionConfig,
)


def _config(stack_bb: float) -> GpuActionConfig:
    return GpuActionConfig(
        preflop_fractions=(1.0,), postflop_fractions=TURN_FRACTIONS,
        max_raises_per_street=TURN_RAISE_CAP, stack_bb=stack_bb,
    )


def _situation(rng: random.Random):
    board = tuple(rng.sample(range(52), 4))
    stack_bb, pot_bb = 200.0, float(rng.choice((14.0, 22.0, 34.0, 50.0)))
    behind = stack_bb - pot_bb / 2.0
    root = BettingRootState(
        street=2, to_act=1, committed=(pot_bb / 2.0, pot_bb / 2.0),
        street_commit=(0.0, 0.0), stacks=(behind, behind),
        acted=(False, False), raises=0, last_increment=1.0,
    )
    live = np.ones(NUM_COMBOS, dtype=bool)
    for card in board:
        live &= ~CARD_IN_COMBO[card]
    ranges = np.stack([live / live.sum(), live / live.sum()]).astype(np.float32)
    return board, root, ranges, live, stack_bb


def main() -> None:
    parser = argparse.ArgumentParser(description="River-net acceptance gate")
    parser.add_argument("--net", type=Path, default=Path("backend/data/cfv/river_net.pt"))
    parser.add_argument("--situations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if not arguments.net.exists():
        raise SystemExit(f"no trained net at {arguments.net}")
    from backend.cfv.river_net import RiverCfvNet
    from backend.search.depth_limited import DepthLimitedCFR

    payload = torch.load(arguments.net, map_location="cpu", weights_only=True)
    net = RiverCfvNet(hidden=payload["hidden"], layers=payload["layers"])
    net.load_state_dict(payload["state_dict"])
    net.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(arguments.seed)
    agreements, l1s, full_times, net_times = [], [], [], []

    for index in range(arguments.situations):
        board, root, ranges, live, stack_bb = _situation(rng)
        config = _config(stack_bb)

        full_tree = BettingTree(config, root_state=root)
        solver = VectorCFR(full_tree, ExactTurnSampler(board), device=device, seed=11,
                           averaging_delay=max(2, arguments.iterations // 6))
        solver.root_reach = torch.as_tensor(ranges, device=solver.device)
        started = time.monotonic()
        solver.run(arguments.iterations)
        if device == "cuda":
            torch.cuda.synchronize()
        full_times.append(time.monotonic() - started)
        full_policy = solver.average_strategy_tables()[full_tree.root]
        del solver

        horizon_tree = BettingTree(config, root_state=root, end_street=2)
        limited = DepthLimitedCFR(
            horizon_tree, ExactTurnSampler(board), device=device, seed=11,
            averaging_delay=max(2, arguments.iterations // 6),
            horizon_evaluator=RiverNetEvaluator(net, device, board, stack_bb),
        )
        limited.root_reach = torch.as_tensor(ranges, device=limited.device)
        started = time.monotonic()
        limited.run(arguments.iterations)
        if device == "cuda":
            torch.cuda.synchronize()
        net_times.append(time.monotonic() - started)
        net_policy = limited.average_strategy_tables()[horizon_tree.root]
        del limited

        # Root action sets must line up for the comparison to mean anything.
        legal = np.asarray(full_tree.legal[full_tree.root], dtype=bool)
        assert np.array_equal(legal, np.asarray(horizon_tree.legal[horizon_tree.root], dtype=bool))

        weights = ranges[0][live]
        weights = weights / max(weights.sum(), 1e-12)
        full_choice = full_policy[live][:, legal].argmax(axis=1)
        net_choice = net_policy[live][:, legal].argmax(axis=1)
        agreements.append(float((weights * (full_choice == net_choice)).sum()))
        l1s.append(float((weights * np.abs(full_policy[live] - net_policy[live]).sum(axis=1)).sum()))
        if index == 0:
            print(f"  full tree {len(full_tree)} nodes, horizon tree {len(horizon_tree)} nodes "
                  f"({len(full_tree) / len(horizon_tree):.1f}x)")

    report = {
        "gate": "river-net-acceptance",
        "net": str(arguments.net),
        "situations": arguments.situations,
        "iterations": arguments.iterations,
        "action_agreement_mean": round(statistics.fmean(agreements), 4),
        "action_agreement_min": round(min(agreements), 4),
        "policy_l1_mean": round(statistics.fmean(l1s), 4),
        "policy_l1_max": round(max(l1s), 4),
        "full_solve_s_mean": round(statistics.fmean(full_times), 3),
        "net_solve_s_mean": round(statistics.fmean(net_times), 3),
        "speedup": round(statistics.fmean(full_times) / max(statistics.fmean(net_times), 1e-9), 2),
    }
    print(json.dumps(report, indent=2))
    print()
    print("Interpretation: agreement is the share of range mass whose top action")
    print("is unchanged by pricing the river with the net instead of solving it.")
    print("The P1.2 convergence ladder puts 240 iterations at L1 ~0.21 from")
    print("converged, so a policy_l1 near that is at the solver's own noise level.")
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
