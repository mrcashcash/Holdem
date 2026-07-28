"""Does MORE river data actually help? Run this before committing days of GPU.

The decision this answers: a full-scale dataset is 1-7 days of generation, and it
is only worth it if the net's held-out error keeps falling as data grows. CFV v0
burned its budget on a net that never beat a zero-predictor, and nobody measured
the scaling curve first.

Trains on nested subsets and reports validation MAE as a RATIO to the
zero-predictor baseline (predicting 0 everywhere):

    ratio ~ 1.00  -> the net has learned nothing; more data will not fix a
                     representation problem, stop and rethink
    ratio falling -> data-limited; extrapolate the trend to size the run
    ratio flat but < 1 -> capacity- or target-limited, not data-limited; more
                     rows are the wrong purchase (try width/depth or better
                     targets instead)

Validation is split BY BOARD so rows from one solve cannot leak across the split.

Usage:
    python tools/river_net_scaling.py --data backend/data/cfv/river
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="River-net data scaling probe")
    parser.add_argument("--data", type=Path, default=Path("backend/data/cfv/river"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fractions", type=float, nargs="*", default=[0.25, 0.5, 1.0])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    from backend.cfv.river_net import dataset_rows, load_shards, train

    if not list(arguments.data.glob("manifest*.json")):
        raise SystemExit(f"no dataset at {arguments.data}")
    total = dataset_rows(arguments.data)
    if total < 400:
        raise SystemExit(f"only {total} rows; let generation run longer first")

    print(f"dataset: {total:,} rows at {arguments.data}")
    print(f"{'rows':>10}{'val MAE':>12}{'zero base':>12}{'ratio':>9}{'best epoch':>12}")
    results = []
    for fraction in arguments.fractions:
        limit = max(200, int(total * fraction))
        report = train(
            arguments.data, epochs=arguments.epochs, batch_size=arguments.batch_size,
            limit=limit, progress=False,
        )
        best = min(report["history"], key=lambda row: row["vs_baseline"])
        results.append({"rows": report["rows"], **{k: best[k] for k in
                        ("val_mae", "zero_baseline_mae", "vs_baseline", "epoch")}})
        print(f"{report['rows']:>10,}{best['val_mae']:>12.5f}{best['zero_baseline_mae']:>12.5f}"
              f"{best['vs_baseline']:>9.3f}{best['epoch']:>12}")
        sys.stdout.flush()

    print()
    if len(results) >= 2:
        first, last = results[0], results[-1]
        improvement = first["vs_baseline"] - last["vs_baseline"]
        growth = last["rows"] / max(first["rows"], 1)
        print(f"ratio {first['vs_baseline']:.3f} -> {last['vs_baseline']:.3f} "
              f"over a {growth:.1f}x data increase (delta {improvement:+.3f})")
        if last["vs_baseline"] > 0.95:
            verdict = "STOP: the net has not beaten a zero-predictor; more data is unlikely to help"
        elif improvement > 0.05:
            verdict = "SCALE: still data-limited, more rows should keep helping"
        else:
            verdict = "PLATEAU: not data-limited; spend on targets/capacity, not rows"
        print(f"VERDICT: {verdict}")
        results.append({"verdict": verdict})

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
