"""River-net acceptance gate, v2 — null-anchored and sensitivity-restricted.

WHY v1 IS RETIRED. `tools/river_net_gate.py` scores top-action agreement over all
range mass at the turn root. Measured 2026-07-31, an ALL-ZERO net — weights zeroed,
so it predicts nothing and is definitionally the zero-predictor — scored:

    all-zero NULL net    agreement 0.9269    policy L1 0.4657
    trained net          agreement 0.3766    policy L1 1.1474
    v1 requirement            >= 0.90            <= 0.30

The null PASSES the agreement bar and the trained net fails it. That is not a
surprising result about the net, it is a broken metric: most combos' top action is
unchanged by whatever prices the river, so the statistic is dominated by decisions
no horizon could affect and its floor sits at ~0.93. A threshold of 0.90 is BELOW
the floor, so it can be met by learning nothing.

WHAT v2 MEASURES INSTEAD. Three arms per situation rather than two:

    full  — full turn+river solve. Ground truth.
    net   — horizon priced by the candidate net.
    null  — horizon priced by ZERO. The floor, computed rather than assumed.

From those, two criteria that cannot be satisfied by an empty net:

1. **Sensitive-set accuracy** (primary). Restrict to combos where `full` and `null`
   choose DIFFERENT top actions — the only decisions the river horizon actually
   changes. On that subset the null scores 0 by construction, so the measure runs
   from a true 0 (no better than nothing) to 1 (perfect), with no inherited floor.
   This is the number that decides acceptance.

2. **Null-relative L1 skill**. `(l1_null - l1_net) / l1_null`: the fraction of the
   null horizon's policy error the net removes. Negative means the net is worse than
   pricing the river at zero — which is the present situation, and worth stating as
   a number rather than a footnote.

Both are reported with the raw v1-style figures alongside, so the retirement is
auditable rather than asserted.

A NOTE ON THRESHOLDS. v1's 0.90 was assumed and never measured (STATUS.md §7 says
so). v2 deliberately does NOT invent a replacement constant. It reports where a
candidate sits, and the acceptance bar should be derived from an
accuracy-versus-strength curve once one exists — the same mistake is not worth
making twice with a new number.

Usage:
    python tools/river_net_gate_v2.py --net backend/data/cfv/river_net/river_net.pt
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from backend.search.exact_turn import ExactTurnSampler  # noqa: E402
from backend.search.river_horizon import RiverNetEvaluator  # noqa: E402
from backend.solver.gpu.cfr import VectorCFR  # noqa: E402
from backend.solver.gpu.tree import BettingTree  # noqa: E402

# Reuse v1's situation sampler and tree config verbatim so the two gates describe
# the same population and any difference is the metric, not the setup.
from tools.river_net_gate import _config, _situation  # noqa: E402


def load_net(path: Path):
    from backend.cfv.river_net import RiverCfvNet

    payload = torch.load(path, map_location="cpu", weights_only=True)
    net = RiverCfvNet(hidden=payload["hidden"], layers=payload["layers"])
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net


def zero_net(reference):
    """An all-zero copy of `reference`: predicts 0 CFVs, i.e. the null horizon."""
    from backend.cfv.river_net import RiverCfvNet

    net = RiverCfvNet(hidden=reference.hidden, layers=reference.layers)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Null-anchored river-net gate")
    parser.add_argument("--net", type=Path,
                        default=Path("backend/data/cfv/river_net/river_net.pt"))
    parser.add_argument("--situations", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/river-gate-v2.json"))
    arguments = parser.parse_args()
    if not arguments.net.exists():
        raise SystemExit(f"no trained net at {arguments.net}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output.with_suffix(".log")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    from backend.search.depth_limited import DepthLimitedCFR

    candidate = load_net(arguments.net)
    null = zero_net(candidate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(arguments.seed)

    log(f"=== river-net gate v2 (null-anchored): {arguments.net} ===")
    log(f"{arguments.situations} situations x {arguments.iterations} iterations on {device}")
    log(f"durable log: {log_path}")

    rows = []
    for index in range(arguments.situations):
        board, root, ranges, live, stack_bb = _situation(rng)
        config = _config(stack_bb)

        def solve_full():
            tree = BettingTree(config, root_state=root)
            solver = VectorCFR(tree, ExactTurnSampler(board), device=device, seed=11,
                               averaging_delay=max(2, arguments.iterations // 6))
            solver.root_reach = torch.as_tensor(ranges, device=solver.device)
            solver.run(arguments.iterations)
            if device == "cuda":
                torch.cuda.synchronize()
            policy = solver.average_strategy_tables()[tree.root]
            del solver
            return tree, policy

        def solve_horizon(evaluator_net):
            tree = BettingTree(config, root_state=root, end_street=2)
            limited = DepthLimitedCFR(
                tree, ExactTurnSampler(board), device=device, seed=11,
                averaging_delay=max(2, arguments.iterations // 6),
                horizon_evaluator=RiverNetEvaluator(evaluator_net, device, board, stack_bb),
            )
            limited.root_reach = torch.as_tensor(ranges, device=limited.device)
            limited.run(arguments.iterations)
            if device == "cuda":
                torch.cuda.synchronize()
            policy = limited.average_strategy_tables()[tree.root]
            del limited
            return tree, policy

        started = time.monotonic()
        full_tree, full_policy = solve_full()
        full_s = time.monotonic() - started

        started = time.monotonic()
        net_tree, net_policy = solve_horizon(candidate)
        net_s = time.monotonic() - started

        _, null_policy = solve_horizon(null)

        legal = np.asarray(full_tree.legal[full_tree.root], dtype=bool)
        if not np.array_equal(legal, np.asarray(net_tree.legal[net_tree.root], dtype=bool)):
            raise SystemExit("root action sets differ; the comparison would be meaningless")

        weights = ranges[0][live]
        weights = weights / max(weights.sum(), 1e-12)
        full_choice = full_policy[live][:, legal].argmax(axis=1)
        net_choice = net_policy[live][:, legal].argmax(axis=1)
        null_choice = null_policy[live][:, legal].argmax(axis=1)

        # v1-style figures, over ALL mass, for both arms.
        agree_net_all = float((weights * (full_choice == net_choice)).sum())
        agree_null_all = float((weights * (full_choice == null_choice)).sum())
        l1_net = float((weights * np.abs(full_policy[live] - net_policy[live]).sum(axis=1)).sum())
        l1_null = float((weights * np.abs(full_policy[live] - null_policy[live]).sum(axis=1)).sum())

        # PRIMARY: the horizon-sensitive subset -- combos the river horizon moves.
        sensitive = full_choice != null_choice
        sensitive_mass = float(weights[sensitive].sum())
        if sensitive_mass > 1e-9:
            sensitive_weights = weights[sensitive] / sensitive_mass
            sensitive_accuracy = float(
                (sensitive_weights * (full_choice[sensitive] == net_choice[sensitive])).sum())
        else:
            sensitive_accuracy = float("nan")

        rows.append({
            "situation": index,
            "stack_bb": stack_bb,
            "sensitive_mass": round(sensitive_mass, 4),
            "sensitive_accuracy": (None if sensitive_mass <= 1e-9
                                   else round(sensitive_accuracy, 4)),
            "agreement_net_all_mass": round(agree_net_all, 4),
            "agreement_null_all_mass": round(agree_null_all, 4),
            "policy_l1_net": round(l1_net, 4),
            "policy_l1_null": round(l1_null, 4),
            "full_s": round(full_s, 2),
            "net_s": round(net_s, 2),
        })
        log(f"  s{index:<2} {stack_bb:>5.0f}bb  sensitive mass {sensitive_mass:>6.1%}  "
            f"sensitive acc {('n/a' if sensitive_mass <= 1e-9 else f'{sensitive_accuracy:.3f}'):>6}  "
            f"agree net/null {agree_net_all:.3f}/{agree_null_all:.3f}  "
            f"L1 net/null {l1_net:.3f}/{l1_null:.3f}")

    scored = [r for r in rows if r["sensitive_accuracy"] is not None]
    mean_sensitive_mass = statistics.fmean(r["sensitive_mass"] for r in rows)
    mean_l1_net = statistics.fmean(r["policy_l1_net"] for r in rows)
    mean_l1_null = statistics.fmean(r["policy_l1_null"] for r in rows)
    l1_skill = (mean_l1_null - mean_l1_net) / max(mean_l1_null, 1e-12)
    mean_sensitive_accuracy = (statistics.fmean(r["sensitive_accuracy"] for r in scored)
                               if scored else float("nan"))

    report = {
        "gate": "river-net-acceptance-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "net": str(arguments.net),
        "situations": arguments.situations,
        "iterations": arguments.iterations,
        "situations_with_sensitive_mass": len(scored),
        "mean_sensitive_mass": round(mean_sensitive_mass, 4),
        "sensitive_accuracy_mean": (None if not scored
                                    else round(mean_sensitive_accuracy, 4)),
        "policy_l1_net_mean": round(mean_l1_net, 4),
        "policy_l1_null_mean": round(mean_l1_null, 4),
        "l1_skill_vs_null": round(l1_skill, 4),
        "agreement_net_all_mass_mean": round(
            statistics.fmean(r["agreement_net_all_mass"] for r in rows), 4),
        "agreement_null_all_mass_mean": round(
            statistics.fmean(r["agreement_null_all_mass"] for r in rows), 4),
        "rows": rows,
    }

    log("")
    log("--- v2 summary ---")
    log(f"mean horizon-sensitive mass          : {mean_sensitive_mass:.1%}  "
        f"(share of range the river horizon actually moves)")
    if scored:
        log(f"SENSITIVE-SET ACCURACY (primary)     : {mean_sensitive_accuracy:.4f}  "
            f"(0 = no better than a zero horizon, 1 = perfect)")
    else:
        log("SENSITIVE-SET ACCURACY               : undefined -- no situation had any")
        log("  horizon-sensitive mass, so this population cannot test a river net at all.")
    log(f"policy L1, net vs null               : {mean_l1_net:.4f} vs {mean_l1_null:.4f}")
    log(f"L1 SKILL vs null                     : {l1_skill:+.4f}  "
        f"(fraction of the null's error removed; negative = worse than nothing)")
    log("")
    log(f"for audit, v1-style over ALL mass    : agreement net {report['agreement_net_all_mass_mean']:.4f} "
        f"vs null {report['agreement_null_all_mass_mean']:.4f}")
    log("  v1 required >= 0.90, which the NULL clears -- that is why v1 is retired.")
    log("")
    if scored and mean_sensitive_accuracy <= 0.0:
        log("VERDICT: the net gets NONE of the horizon-sensitive decisions right, and")
        log("is worse than pricing the river at zero. Not a data-volume problem.")
    elif l1_skill < 0:
        log("VERDICT: negative L1 skill -- the net adds more policy error than a zero")
        log("horizon. Any acceptance bar is moot until this is positive.")
    else:
        log("VERDICT: the net has positive skill over the null. An acceptance threshold")
        log("should now be derived from an accuracy-versus-strength curve, NOT assumed.")

    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    handle.close()


if __name__ == "__main__":
    main()
