"""What accuracy does the river net need before its DECISIONS survive?

`docs/STATUS.md` 7 lists this as an open measurement: "the agreement-vs-ratio
curve for the river net, since the 0.1 acceptance threshold was assumed and never
measured." Without it, nobody can say whether the CFV line is one data doubling
away from working or a thousand.

What is known (recovered from backend/data/cfv/river_net/, 76,411 rows, 120 epochs):

    val MAE 0.3397   zero-baseline 0.6984   ratio 0.4864
    action agreement 0.3766  (gate needs >= 0.90)
    policy L1        1.1474  (gate needs <= 0.30)

So the net halves the zero-predictor's error yet only reproduces 38% of the
solver's top actions. The missing link is the SHAPE of agreement-versus-ratio: if
0.90 agreement arrives around ratio 0.25 it is a data problem worth solving, and
if it needs ratio 0.05 the line is dead at any feasible row count.

Method: train at several data fractions -- which is the only honest way to vary
net quality without also changing the architecture -- then run the real acceptance
gate on each saved net. Reports (rows, ratio, agreement, policy L1) so the trend
can be extrapolated to the 0.90 requirement.

Every point uses identical epochs/architecture so the only varying input is data
volume. Validation is split BY BOARD inside train(), so no solve leaks across it.

Usage:
    python tools/river_agreement_curve.py --fractions 0.25 0.5 1.0 --epochs 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.cfv.river_net import dataset_rows, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement-vs-accuracy curve for the river net")
    parser.add_argument("--data", type=Path, default=Path("backend/data/cfv/river"))
    parser.add_argument("--fractions", type=float, nargs="*", default=[0.25, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--situations", type=int, default=12,
                        help="gate situations; 12 matches the recorded baseline run")
    parser.add_argument("--iterations", type=int, default=160,
                        help="gate solve iterations; 160 matches the recorded baseline")
    parser.add_argument("--workdir", type=Path,
                        default=Path("backend/data/cfv/agreement_curve"))
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/river-agreement-curve.json"))
    arguments = parser.parse_args()

    arguments.workdir.mkdir(parents=True, exist_ok=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output.with_suffix(".log")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    total_rows = dataset_rows(arguments.data)
    log(f"=== agreement-vs-ratio curve: {total_rows:,} rows available, "
        f"fractions {arguments.fractions}, {arguments.epochs} epochs each ===")
    log(f"gate: {arguments.situations} situations x {arguments.iterations} iterations")
    log(f"durable log: {log_path}")

    points = []
    for fraction in arguments.fractions:
        limit = max(1, int(total_rows * fraction))
        tag = f"f{fraction:g}"
        checkpoint_dir = arguments.workdir / tag
        # Restartable: a completed point is not retrained.
        point_path = checkpoint_dir / "point.json"
        if point_path.exists():
            points.append(json.loads(point_path.read_text(encoding="utf-8")))
            log(f"-- fraction {fraction:g} already complete; skipping")
            continue

        log(f"-- fraction {fraction:g}: training on <= {limit:,} rows")
        started = time.time()
        result = train(
            arguments.data,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            limit=limit,
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )
        train_s = round(time.time() - started, 1)
        # train() reports per-epoch rows in `history`, not top-level metrics, and
        # saves the checkpoint at the BEST ratio -- so score the same epoch the
        # saved net came from, otherwise the ratio and the gated net disagree.
        history = result.get("history") or []
        if not history:
            log("   ERROR: train() returned no history; skipping this point")
            continue
        best_epoch = min(history, key=lambda row: row["vs_baseline"])
        ratio = best_epoch["vs_baseline"]
        log(f"   trained in {train_s}s: val_mae={best_epoch['val_mae']:.4f} "
            f"baseline={best_epoch['zero_baseline_mae']:.4f} ratio={ratio:.4f} "
            f"(best of {len(history)} epochs)")

        net_path = checkpoint_dir / "river_net.pt"
        if not net_path.exists():
            log(f"   ERROR: train() left no net at {net_path}; skipping gate")
            continue

        gate_output = checkpoint_dir / "gate.json"
        log("   running acceptance gate ...")
        completed = subprocess.run(
            [sys.executable, "tools/river_net_gate.py",
             "--net", str(net_path),
             "--situations", str(arguments.situations),
             "--iterations", str(arguments.iterations),
             "--output", str(gate_output)],
            capture_output=True, text=True,
        )
        if completed.returncode != 0 or not gate_output.exists():
            log(f"   gate FAILED rc={completed.returncode}: "
                f"{(completed.stderr or completed.stdout)[-300:]}")
            continue
        gate = json.loads(gate_output.read_text(encoding="utf-8"))
        point = {
            "fraction": fraction,
            "rows": limit,
            "val_mae": best_epoch["val_mae"],
            "zero_baseline_mae": best_epoch["zero_baseline_mae"],
            "ratio_vs_baseline": ratio,
            "action_agreement": gate.get("action_agreement_mean"),
            "policy_l1": gate.get("policy_l1_mean"),
            "train_s": train_s,
        }
        point_path.write_text(json.dumps(point, indent=2), encoding="utf-8")
        points.append(point)
        log(f"   ratio={point['ratio_vs_baseline']} -> agreement="
            f"{point['action_agreement']} policy_l1={point['policy_l1']}")

    # The recorded 120-epoch run on the full dataset, for context.
    baseline_point = {
        "fraction": 1.0,
        "rows": total_rows,
        "note": "recorded 120-epoch run (backend/data/cfv/river_net)",
        "ratio_vs_baseline": 0.4864,
        "action_agreement": 0.3766,
        "policy_l1": 1.1474,
    }

    log("")
    log("--- curve (lower ratio = more accurate net) ---")
    log(f"{'rows':>10} {'ratio':>8} {'agreement':>10} {'policy L1':>10}")
    for point in sorted([*points, baseline_point], key=lambda p: -(p["ratio_vs_baseline"] or 0)):
        log(f"{point['rows']:>10,} {point['ratio_vs_baseline']:>8.4f} "
            f"{(point['action_agreement'] or 0):>10.4f} {(point['policy_l1'] or 0):>10.4f}")

    usable = [p for p in points if p.get("ratio_vs_baseline") and p.get("action_agreement")]
    log("")
    if len(usable) >= 2:
        # Linear fit of agreement on ratio: crude, but enough to tell "one
        # doubling away" from "hopeless". Reported with the caveat, not as truth.
        best, worst = min(usable, key=lambda p: p["ratio_vs_baseline"]), max(
            usable, key=lambda p: p["ratio_vs_baseline"])
        d_ratio = worst["ratio_vs_baseline"] - best["ratio_vs_baseline"]
        d_agree = best["action_agreement"] - worst["action_agreement"]
        if d_ratio > 1e-9 and d_agree > 1e-9:
            slope = d_agree / d_ratio  # agreement gained per unit ratio reduction
            needed = best["ratio_vs_baseline"] - (0.90 - best["action_agreement"]) / slope
            log(f"agreement rises {slope:.3f} per 1.0 of ratio reduction")
            log(f"linear extrapolation: 0.90 agreement needs ratio ~{needed:.4f}")
            if needed <= 0:
                log("NEGATIVE required ratio -> unreachable by accuracy alone. The gate")
                log("cannot be met by making this net more accurate; the representation or")
                log("the acceptance criterion itself has to change. PARK the CFV line.")
            else:
                log(f"that is {best['ratio_vs_baseline'] / max(needed, 1e-9):.1f}x more")
                log("accurate than the best point measured here.")
        else:
            log("agreement did not improve as accuracy improved over this range --")
            log("evidence that agreement is NOT accuracy-limited. PARK the CFV line.")
    else:
        log("fewer than two usable points; cannot fit a trend")

    arguments.output.write_text(json.dumps(
        {"points": points, "recorded_baseline": baseline_point}, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    handle.close()


if __name__ == "__main__":
    main()
