"""Run one CRN-coupled duel across every core, with a bit-exactness proof.

A duel is CPU-bound and there is no GPU path worth taking. With search disabled a
blueprint decision is a bucket lookup plus an array index -- not tensor math -- so
moving it to the GPU would add a kernel launch per decision and run *slower*.
`backend/eval/duel.py` disables CUDA outright (`CUDA_VISIBLE_DEVICES=""`) for that
reason plus one more: a duel must never contend with a training run for the card.
The only real GPU path would be a vectorized game engine that steps thousands of
hands in lockstep, which is a rewrite of `backend/poker.py`, not a flag.

So the speedup here is cores, not the card. Measured on this 8-core box:
5.35 pairs/s serial, i.e. 3.1 h for the 60,000 pairs needed to match the +/-6.7
bb/100 precision of the earlier champion comparisons. Sharded, that is ~25 min.

WHY THIS IS SAFE. Each pair's deal seed is `seed * 1_000_003 + pair`, a pure
function of the pair index, and the per-pair sample is a self-contained
(seat 0, seat 1) average. Shards over disjoint pair ranges, concatenated in pair
order, therefore reproduce the serial sample vector *exactly* -- not to within
tolerance. `--verify-pairs` proves it every run: the parent replays the first N
pairs serially in-process and refuses to report a total unless those samples match
the sharded ones bit for bit. A sharding bug that silently reordered or dropped
pairs would otherwise look like a strength result.

Usage:
    python tools/duel_sharded.py \
        --challenger backend/data/gpu_blueprint_200bb_hist/checkpoint-40000.npz \
        --incumbent  backend/data/gpu_blueprint_200bb_hist/checkpoint-20000.npz \
        --pairs 60000 --workers 6 --label 40k-vs-20k
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load(path: Path):
    """A blueprint-only agent: no search, no guard, so the checkpoint is the sole variable."""
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    agent = GpuBlueprintAgent.try_load(path)
    if agent is None:
        raise SystemExit(f"could not load {path}")
    agent.subgame_search = False
    agent.flop_search = False
    agent.exact_river_search = False
    agent.continual_search = False
    agent.all_in_geometry_guard = False
    return agent


def _describe(path: Path) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        sampler = json.loads(str(payload["sampler"]))
        config = json.loads(str(payload["config"]))
        return {
            "path": str(path),
            "iteration": int(payload["iteration"]),
            "histogram": bool(sampler.get("histogram")),
            "buckets": [169, sampler.get("flop_buckets"),
                        sampler.get("turn_buckets"), sampler.get("river_buckets")],
            "preflop": config.get("preflop_fractions"),
            "postflop": config["postflop_fractions"],
            "raise_cap": config["max_raises_per_street"],
            "stack_bb": config.get("stack_bb"),
        }


def _run_worker(arguments) -> None:
    """Play one pair slice and write its raw samples out as JSON."""
    from backend.eval.duel import head_to_head

    report = head_to_head(
        _load(arguments.challenger), _load(arguments.incumbent),
        stack_bb=arguments.stack_bb, pairs=arguments.pairs, seed=arguments.seed,
        common_random_numbers=True,
        pair_start=arguments.pair_start, pair_stop=arguments.pair_stop,
        return_samples=True,
    )
    arguments.shard_out.write_text(json.dumps(report), encoding="utf-8")


def _aggregate(samples: list[float]) -> dict:
    mean = statistics.fmean(samples)
    margin = (1.96 * statistics.stdev(samples) / math.sqrt(len(samples))
              if len(samples) > 1 else 0.0)
    return {
        "mean_bb_per_100": round(mean * 100.0, 2),
        "ci_low_bb_per_100": round((mean - margin) * 100.0, 2),
        "ci_high_bb_per_100": round((mean + margin) * 100.0, 2),
        "pairs": len(samples),
        "hands": len(samples) * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="core-sharded CRN duel")
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pairs", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = (cores - 2), leaving room for the OS and any GPU feeder")
    parser.add_argument("--verify-pairs", type=int, default=200,
                        help="serial control replayed in-process; 0 disables (not advised)")
    parser.add_argument("--null-pairs", type=int, default=400,
                        help="incumbent-vs-itself CRN null; must read exactly 0.00")
    parser.add_argument("--label", default="duel")
    parser.add_argument("--output", type=Path, default=None)
    # worker plumbing
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pair-start", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--pair-stop", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard-out", type=Path, default=None, help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.worker:
        _run_worker(arguments)
        return

    workers = arguments.workers or max(1, (os.cpu_count() or 4) - 2)
    output = arguments.output or Path(f"backend/data/evaluations/{arguments.label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = output.parent / f"{arguments.label}-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    challenger_info = _describe(arguments.challenger)
    incumbent_info = _describe(arguments.incumbent)
    log(f"=== {arguments.label}: {arguments.pairs:,} pairs @ {arguments.stack_bb:.0f}bb, "
        f"CRN, {workers} workers ===")
    for name, info in (("challenger", challenger_info), ("incumbent ", incumbent_info)):
        log(f"{name}: iter={info['iteration']:,} histogram={info['histogram']} "
            f"buckets={info['buckets']} preflop={info['preflop']} "
            f"postflop={info['postflop']} cap={info['raise_cap']}")
    for field in ("preflop", "postflop", "raise_cap", "buckets"):
        if challenger_info[field] != incumbent_info[field]:
            log(f"NOTE: {field} differs between the arms -- this duel does not isolate "
                f"training progress alone.")
    log(f"durable log: {log_path}")

    # ---------------------------------------------------------------- null
    # An uncoupled router duel carries ~+/-80 bb/100 of avoidable noise, so verify
    # the coupling actually reaches these agents before trusting any difference.
    log("")
    log(f"-- NULL: incumbent vs itself, {arguments.null_pairs} pairs, must be exactly +0.00 --")
    from backend.eval.duel import head_to_head

    null = head_to_head(_load(arguments.incumbent), _load(arguments.incumbent),
                        stack_bb=arguments.stack_bb, pairs=arguments.null_pairs,
                        seed=arguments.seed, common_random_numbers=True)
    log(f"   null: {null['mean_bb_per_100']:+.2f} bb/100")
    if abs(null["mean_bb_per_100"]) > 1e-9:
        log("   NULL FAILED -- coupling is not reaching these agents. Stopping rather")
        log("   than reporting a number that carries avoidable noise.")
        raise SystemExit(1)

    # ------------------------------------------------------------- shards
    bounds = [round(i * arguments.pairs / workers) for i in range(workers + 1)]
    slices = [(bounds[i], bounds[i + 1]) for i in range(workers) if bounds[i] < bounds[i + 1]]
    log("")
    log(f"-- launching {len(slices)} shards: " +
        ", ".join(f"[{a:,},{b:,})" for a, b in slices))

    # One thread per worker. Eight workers each spawning eight BLAS threads would
    # oversubscribe the box and run slower than serial.
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "", "HOLDEM_SUBGAME_ITERS": "0"})

    started = time.time()
    processes = []
    for index, (start, stop) in enumerate(slices):
        shard_out = shard_dir / f"shard-{index}.json"
        if shard_out.exists():
            shard_out.unlink()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--challenger", str(arguments.challenger),
                   "--incumbent", str(arguments.incumbent),
                   "--stack-bb", str(arguments.stack_bb),
                   "--pairs", str(arguments.pairs), "--seed", str(arguments.seed),
                   "--pair-start", str(start), "--pair-stop", str(stop),
                   "--shard-out", str(shard_out)]
        errors = (shard_dir / f"shard-{index}.err").open("w", encoding="utf-8")
        processes.append((index, start, stop, shard_out, errors,
                          subprocess.Popen(command, cwd=str(REPO), env=env,
                                           stdout=subprocess.DEVNULL, stderr=errors)))

    # ---------------------------------------------- serial control, in parallel
    # Replay the first pairs here while the shards run: it costs nothing in
    # wall-clock and it is what makes the sharded total trustworthy.
    control: list[float] | None = None
    if arguments.verify_pairs > 0:
        log(f"-- serial control: replaying pairs [0,{arguments.verify_pairs}) in-process --")
        control = head_to_head(
            _load(arguments.challenger), _load(arguments.incumbent),
            stack_bb=arguments.stack_bb, pairs=arguments.pairs, seed=arguments.seed,
            common_random_numbers=True, pair_start=0, pair_stop=arguments.verify_pairs,
            return_samples=True)["samples"]

    failed = False
    for index, start, stop, shard_out, errors, process in processes:
        code = process.wait()
        errors.close()
        detail = (shard_dir / f"shard-{index}.err").read_text(encoding="utf-8").strip()
        if code != 0 or not shard_out.exists():
            failed = True
            log(f"   shard {index} [{start:,},{stop:,}) FAILED rc={code}")
            if detail:
                log("   " + detail.splitlines()[-1])
        else:
            log(f"   shard {index} [{start:,},{stop:,}) done")
    if failed:
        log("one or more shards failed; refusing to report a partial total")
        raise SystemExit(1)
    elapsed = round(time.time() - started, 1)

    samples: list[float] = []
    for index, start, stop, shard_out, _errors, _process in processes:
        shard = json.loads(shard_out.read_text(encoding="utf-8"))
        if shard["pair_start"] != start or shard["pair_stop"] != stop:
            log(f"   shard {index} covered [{shard['pair_start']},{shard['pair_stop']}) "
                f"but was asked for [{start},{stop})")
            raise SystemExit(1)
        samples.extend(shard["samples"])
    if len(samples) != arguments.pairs:
        log(f"expected {arguments.pairs:,} samples, assembled {len(samples):,}")
        raise SystemExit(1)

    # ------------------------------------------------------------- exactness
    if control is not None:
        head = samples[:len(control)]
        if head != control:
            bad = next(i for i, (a, b) in enumerate(zip(head, control)) if a != b)
            log("")
            log(f"EXACTNESS FAILED at pair {bad}: sharded={head[bad]!r} serial={control[bad]!r}")
            log("The shards are not reproducing the serial run, so the total below would")
            log("be a sharding artifact rather than a strength measurement. Stopping.")
            raise SystemExit(1)
        log(f"   exactness: first {len(control)} pairs match the serial control bit for bit")

    result = _aggregate(samples)
    mean = result["mean_bb_per_100"]
    low, high = result["ci_low_bb_per_100"], result["ci_high_bb_per_100"]
    verdict = ("CHALLENGER BETTER" if low > 0 else
               "CHALLENGER WORSE" if high < 0 else "INCONCLUSIVE")
    rate = round(arguments.pairs / elapsed, 2) if elapsed else 0.0
    log("")
    log(f"   challenger minus incumbent: {mean:+.2f} bb/100 [{low:+.2f},{high:+.2f}]")
    log(f"   {arguments.pairs:,} pairs in {elapsed}s -> {rate} pairs/s across {len(slices)} shards")
    log(f"   VERDICT: {verdict}")

    output.write_text(json.dumps({
        "gate": arguments.label,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "challenger": challenger_info,
        "incumbent": incumbent_info,
        "stack_bb": arguments.stack_bb,
        "seed": arguments.seed,
        "workers": len(slices),
        "null_bb_per_100": null["mean_bb_per_100"],
        "exactness_verified_pairs": len(control) if control is not None else 0,
        "challenger_minus_incumbent_bb_per_100": mean,
        "ci_low": low,
        "ci_high": high,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "pairs_per_s": rate,
        **{k: v for k, v in result.items() if k in ("pairs", "hands")},
    }, indent=2), encoding="utf-8")
    log(f"written to {output}")
    handle.close()


if __name__ == "__main__":
    main()
