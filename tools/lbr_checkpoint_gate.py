"""Paired LBR between two checkpoints: which is closer to equilibrium?

Two independent LBR runs give two wide, overlapping intervals. At 200bb the
histogram@20k challenger read +223.34 [+192.01, +254.66] and the deployed
scalar@118k champion +252.45 [+223.34, +281.55] — a 29.1 bb/100 gap buried inside
±30 of card variance, which decides nothing.

The variance is shared, though: both arms probe the SAME duplicate deals when given
the same seed. Differencing per pair cancels it. That is exactly what made
`tools/lbr_guard_gate.py` resolve a −4.74 [−10.63, +1.14] effect out of arms whose
own intervals were ±30.

This runs the same paired design across two arbitrary checkpoints rather than one
checkpoint with a flag flipped, so a card-abstraction change can be judged on the
project's north-star instrument.

Chunked and checkpointed: a crash costs one chunk, not the whole arm. Both arms are
blueprint-only with every search path off, so this measures the blueprint itself.

Usage:
    python tools/lbr_checkpoint_gate.py --pairs 20000 \
        --challenger backend/data/gpu_blueprint_200bb_hist/checkpoint.npz \
        --incumbent  backend/data/gpu_blueprint_200bb/champion.npz
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _summary(samples: list[float]) -> dict:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(len(samples)) if samples else 0.0
    return {
        "bb_per_100": round(mean * 100, 2),
        "ci_low": round((mean - margin) * 100, 2),
        "ci_high": round((mean + margin) * 100, 2),
        "pairs": len(samples),
    }


def run_arm(checkpoint: Path, stack_bb: float, pairs: int, seed: int, chunk_pairs: int,
            log, resume: dict | None, on_chunk) -> dict:
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
    from backend.eval.lbr import local_best_response_probe

    agent = GpuBlueprintAgent.try_load(checkpoint)
    if agent is None:
        raise SystemExit(f"could not load {checkpoint}")
    agent.subgame_search = False
    agent.flop_search = False
    agent.exact_river_search = False
    agent.continual_search = False
    agent.all_in_geometry_guard = False

    samples: list[float] = list((resume or {}).get("pair_samples", []))
    fallbacks: list[tuple[int, float]] = [
        (int(count), float(rate)) for count, rate in (resume or {}).get("fallbacks", [])
    ]
    elapsed = float((resume or {}).get("elapsed_s", 0.0))
    done = int((resume or {}).get("chunks_done", 0))

    boundaries = list(range(0, pairs, chunk_pairs))
    for index in range(done, len(boundaries)):
        size = min(chunk_pairs, pairs - boundaries[index])
        started = time.time()
        # Chunk i uses seed+i in BOTH arms, so the arms probe identical deals.
        report = local_best_response_probe(
            agent, hands=size * 2, seed=seed + index, stack_bb=stack_bb)
        elapsed += time.time() - started
        samples.extend(report["pair_samples"])
        fallbacks.append((size, float(report["diagnostics"]["fallback_rate"])))
        log(f"    chunk {index + 1}/{len(boundaries)}: {len(samples):,} pairs, "
            f"running LBR {statistics.fmean(samples) * 100:+.2f} bb/100, "
            f"{elapsed / 60:.1f} min")
        on_chunk({"pair_samples": samples, "fallbacks": fallbacks,
                  "elapsed_s": round(elapsed, 1), "chunks_done": index + 1})

    total = sum(count for count, _ in fallbacks) or 1
    weighted = sum(count * rate for count, rate in fallbacks) / total
    summary = _summary(samples)
    return {**summary, "fallback_rate": round(weighted, 6),
            "elapsed_s": round(elapsed, 1), "pair_samples": samples,
            "fallbacks": fallbacks, "chunks_done": len(boundaries)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired LBR between two checkpoints")
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--chunk-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/lbr-checkpoint-gate.json"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output.with_suffix(".log")
    partial_path = arguments.output.with_suffix(".partial.json")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    log(f"=== paired LBR: {arguments.pairs:,} duplicate pairs @ "
        f"{arguments.stack_bb:.0f}bb, seed {arguments.seed} ===")
    log(f"challenger: {arguments.challenger}")
    log(f"incumbent : {arguments.incumbent}")
    log(f"durable log: {log_path}")

    partial: dict = {}
    if partial_path.exists():
        stored = json.loads(partial_path.read_text(encoding="utf-8"))
        if (stored.get("pairs") == arguments.pairs
                and stored.get("seed") == arguments.seed
                and stored.get("challenger") == str(arguments.challenger)
                and stored.get("incumbent") == str(arguments.incumbent)):
            partial = stored
            log(f"resuming: arms complete = {sorted(partial.get('arms', {}))}")
        else:
            log("existing checkpoint has a different configuration; ignoring it")
    arms: dict = partial.get("arms", {})

    def save() -> None:
        partial_path.write_text(json.dumps({
            "pairs": arguments.pairs, "seed": arguments.seed,
            "challenger": str(arguments.challenger),
            "incumbent": str(arguments.incumbent), "arms": arms,
        }, indent=2), encoding="utf-8")

    for name, path in (("incumbent", arguments.incumbent),
                       ("challenger", arguments.challenger)):
        existing = arms.get(name)
        if existing and "bb_per_100" in existing:
            log(f"arm '{name}' already complete; skipping")
            continue
        log(f"arm '{name}': {path.name}")

        def on_chunk(progress: dict, _name: str = name) -> None:
            arms[_name] = progress
            save()

        arms[name] = run_arm(path, arguments.stack_bb, arguments.pairs, arguments.seed,
                             arguments.chunk_pairs, log, existing, on_chunk)
        save()
        log(f"  arm done: LBR {arms[name]['bb_per_100']:+.2f} bb/100 "
            f"[{arms[name]['ci_low']:+.2f},{arms[name]['ci_high']:+.2f}] "
            f"fallback={arms[name]['fallback_rate']:.2%}")

    left = arms["incumbent"]["pair_samples"]
    right = arms["challenger"]["pair_samples"]
    if len(left) != len(right):
        raise SystemExit(f"arms disagree on pair count ({len(left)} vs {len(right)})")

    # LBR reports what the PROBE wins, so lower is better. A negative difference
    # means the challenger is LESS exploitable than the incumbent.
    deltas = [challenger - incumbent for incumbent, challenger in zip(left, right)]
    paired = _summary(deltas)
    identical = sum(1 for value in deltas if value == 0.0)
    verdict = ("CHALLENGER LESS EXPLOITABLE" if paired["ci_high"] < 0
               else "CHALLENGER MORE EXPLOITABLE" if paired["ci_low"] > 0
               else "INCONCLUSIVE")

    log("")
    log("--- paired difference (challenger minus incumbent), NEGATIVE is better ---")
    log(f"  {paired['bb_per_100']:+.2f} bb/100 "
        f"[{paired['ci_low']:+.2f}, {paired['ci_high']:+.2f}] over {paired['pairs']:,} pairs")
    log(f"  pairs where the two played identically: {identical:,}/{len(deltas):,}")
    log(f"  VERDICT: {verdict}")

    arguments.output.write_text(json.dumps({
        "gate": "lbr-paired-checkpoint",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "challenger": str(arguments.challenger),
        "incumbent": str(arguments.incumbent),
        "stack_bb": arguments.stack_bb, "seed": arguments.seed,
        "incumbent_arm": {k: v for k, v in arms["incumbent"].items()
                          if k not in ("pair_samples", "fallbacks")},
        "challenger_arm": {k: v for k, v in arms["challenger"].items()
                           if k not in ("pair_samples", "fallbacks")},
        "paired_delta": paired, "pairs_identical": identical, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    partial_path.unlink(missing_ok=True)
    handle.close()


if __name__ == "__main__":
    main()
