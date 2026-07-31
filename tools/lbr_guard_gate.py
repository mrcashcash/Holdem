"""P1 gate: is the 24x-pot jam a leak in ABSOLUTE terms?

`docs/STATUS.md` §3.6 and `docs/PLAN_V2_STRONGEST_PLAYER.md` both close on the same
admission: the all-in geometry guard is implemented, tested, and shipped OFF
because the deciding measurement was never taken. The head-to-head A/B said the
guard COSTS 124-269 bb/100 against a min-raiser -- but that opponent is a calling
station in disguise, and shoving 199bb into a station is correct exploitation, not
a leak. So a head-to-head cannot answer the question.

LBR can. It is a best responder, so it is the right judge of whether the jam is
exploitable rather than merely large. This gate runs LBR against the same
checkpoint with the guard ON and OFF.

Design notes, each load-bearing:

1. **Paired.** Both arms probe the same deals with the same seed, so per-pair
   differences cancel card variance. Unpaired LBR at 400 pairs is +-140 bb/100 --
   wide enough to hide the entire effect. The paired delta is the result; the
   per-arm absolutes are context.
2. **Blueprint only, both arms.** This isolates the guard. It also matches the
   frozen baseline (+291.23 bb/100 at 200bb was measured blueprint-only), so the
   per-arm numbers are comparable to a recorded instrument reading.
3. **What this does NOT measure.** Serving runs with exact-card resolving ON for
   flop/turn/river, and the resolver puts 0.16% on all-in at the reported spot
   because its tree uses the real geometry. So the guard's SERVED effect is
   smaller than measured here and concentrated PREFLOP -- which is where four of
   the eight observed overbets were, and which no postflop resolving can reach.
   A blueprint-only reading is therefore an upper bound on the guard's value, and
   the preflop share is the part that survives into serving.
4. **LBR models the agent as its blueprint.** With search off in both arms that
   mis-specification is absent, so this is a cleaner LBR reading than the
   resolver A/B could be.

Restartable: each arm's report is checkpointed to `<output>.partial.json` the
moment it finishes, so an interrupted run resumes instead of re-probing. A durable
timestamped log is written beside the output.

Usage:
    python tools/lbr_guard_gate.py --pairs 400 --stack-bb 200 \
        --output backend/data/evaluations/lbr-guard-gate-200bb.json
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


class DurableLog:
    """Timestamped, flushed-on-every-line log beside the output.

    stdout alone gets swallowed by pipes and `tail` buffers to EOF, which has
    already made a long run here look dead while it was fine.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def __call__(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp} {message}"
        print(line)
        sys.stdout.flush()
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _summary(samples: list[float]) -> dict:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * deviation / math.sqrt(len(samples)) if samples else 0.0
    return {
        "bb_per_100": round(mean * 100, 2),
        "ci_low": round((mean - margin) * 100, 2),
        "ci_high": round((mean + margin) * 100, 2),
        "stdev": round(deviation, 4),
        "pairs": len(samples),
    }


def load_arm_agent(checkpoint: Path, *, guard: bool, tolerance: float,
                   max_pot_multiple: float):
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    agent = GpuBlueprintAgent.try_load(checkpoint)
    if agent is None:
        raise SystemExit(f"could not load {checkpoint}")

    # Blueprint only: isolate the guard from every search mechanism.
    agent.subgame_search = False
    agent.flop_search = False
    agent.exact_river_search = False
    agent.continual_search = False

    agent.all_in_geometry_guard = guard
    agent.all_in_geometry_tolerance = tolerance
    agent.all_in_max_pot_multiple = max_pot_multiple

    # Assert the switch actually took, rather than trusting the attribute name.
    if bool(agent.all_in_geometry_guard) is not guard:
        raise SystemExit(f"guard flag did not take (wanted {guard})")
    return agent


def run_arm(checkpoint: Path, stack_bb: float, pairs: int, seed: int, *, guard: bool,
            tolerance: float, max_pot_multiple: float, log: DurableLog,
            chunk_pairs: int, resume: dict | None = None,
            on_chunk=None) -> dict:
    """Probe `pairs` duplicate pairs in chunks, checkpointing after each one.

    Chunking exists for restartability: a single 20,000-pair arm is over an hour,
    and losing it whole to an interrupted process already happened once. Chunk i
    uses seed `seed + i` in BOTH arms, so the paired design is preserved exactly
    -- the arms still probe identical deals, just in independently resumable
    blocks.
    """
    from backend.eval.lbr import local_best_response_probe

    agent = load_arm_agent(checkpoint, guard=guard, tolerance=tolerance,
                           max_pot_multiple=max_pot_multiple)

    samples: list[float] = list((resume or {}).get("pair_samples", []))
    fallbacks: list[tuple[int, float]] = [
        (int(p), float(r)) for p, r in (resume or {}).get("fallbacks", [])
    ]
    elapsed = float((resume or {}).get("elapsed_s", 0.0))
    chunks_done = int((resume or {}).get("chunks_done", 0))

    boundaries = list(range(0, pairs, chunk_pairs))
    if chunks_done:
        log(f"  arm resume: {chunks_done}/{len(boundaries)} chunks, "
            f"{len(samples):,} pairs already probed")
    else:
        log(f"  arm start: guard={'ON' if guard else 'OFF'} "
            f"tolerance={tolerance} cap={max_pot_multiple}x pot, "
            f"{len(boundaries)} chunks of {chunk_pairs}")

    for index in range(chunks_done, len(boundaries)):
        this_chunk = min(chunk_pairs, pairs - boundaries[index])
        started = time.time()
        report = local_best_response_probe(
            agent, hands=this_chunk * 2, seed=seed + index, stack_bb=stack_bb)
        elapsed += time.time() - started
        samples.extend(report["pair_samples"])
        fallbacks.append((this_chunk, float(report["diagnostics"]["fallback_rate"])))
        mean = statistics.fmean(samples) * 100
        log(f"    chunk {index + 1}/{len(boundaries)}: {len(samples):,} pairs, "
            f"running LBR {mean:+.2f} bb/100, {elapsed / 60:.1f} min")
        if on_chunk is not None:
            on_chunk({
                "pair_samples": samples,
                "fallbacks": fallbacks,
                "elapsed_s": round(elapsed, 1),
                "chunks_done": index + 1,
            })

    total_pairs = sum(count for count, _ in fallbacks) or 1
    weighted_fallback = sum(count * rate for count, rate in fallbacks) / total_pairs
    summary = _summary(samples)
    result = {
        "guard": guard,
        "guard_tolerance": tolerance,
        "guard_max_pot_multiple": max_pot_multiple,
        "lbr_bb_per_100": summary["bb_per_100"],
        "ci_low_bb_per_100": summary["ci_low"],
        "ci_high_bb_per_100": summary["ci_high"],
        "pairs": len(samples),
        "fallback_rate": round(weighted_fallback, 6),
        "elapsed_s": round(elapsed, 1),
        "pair_samples": samples,
        "chunks_done": len(boundaries),
        "fallbacks": fallbacks,
    }
    log(f"  arm done : LBR {result['lbr_bb_per_100']:+.2f} bb/100 "
        f"[{result['ci_low_bb_per_100']:+.2f},{result['ci_high_bb_per_100']:+.2f}] "
        f"fallback={weighted_fallback:.2%} over {len(samples):,} pairs "
        f"in {elapsed / 60:.1f} min")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired LBR gate for the all-in geometry guard (STATUS.md §3.6)")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("backend/data/gpu_blueprint_200bb/champion.npz"))
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--chunk-pairs", type=int, default=1000,
                        help="checkpoint granularity; a crash costs at most one chunk")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--tolerance", type=float, default=1.5,
                        help="all_in_geometry_tolerance for the ON arm")
    parser.add_argument("--max-pot-multiple", type=float, default=6.0,
                        help="all_in_max_pot_multiple for the ON arm")
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/lbr-guard-gate.json"))
    arguments = parser.parse_args()

    output = arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    log = DurableLog(output.with_suffix(".log"))
    partial_path = output.with_suffix(".partial.json")

    log(f"=== LBR guard gate: {arguments.pairs} duplicate pairs, seed {arguments.seed}, "
        f"{arguments.stack_bb:.0f}bb ===")
    log(f"checkpoint: {arguments.checkpoint}")
    log(f"durable log: {log.path}")

    partial: dict = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        # A checkpoint from a different configuration must not be reused.
        if (partial.get("pairs") != arguments.pairs
                or partial.get("seed") != arguments.seed
                or partial.get("stack_bb") != arguments.stack_bb
                or partial.get("checkpoint") != str(arguments.checkpoint)):
            log("existing checkpoint has a different configuration; ignoring it")
            partial = {}
        else:
            log(f"resuming: arms already complete = {sorted(partial.get('arms', {}))}")

    arms: dict = partial.get("arms", {})

    def checkpoint_arms() -> None:
        partial_path.write_text(json.dumps({
            "pairs": arguments.pairs,
            "seed": arguments.seed,
            "stack_bb": arguments.stack_bb,
            "checkpoint": str(arguments.checkpoint),
            "arms": arms,
        }, indent=2), encoding="utf-8")

    for name, guard in (("off", False), ("on", True)):
        existing = arms.get(name)
        if existing and existing.get("chunks_done", 0) and "lbr_bb_per_100" in existing:
            log(f"arm '{name}' already complete; skipping")
            continue
        log(f"arm '{name}' of 2")

        def save_chunk(progress: dict, _name: str = name, _guard: bool = guard) -> None:
            arms[_name] = {**progress, "guard": _guard}
            checkpoint_arms()

        arms[name] = run_arm(
            arguments.checkpoint, arguments.stack_bb, arguments.pairs, arguments.seed,
            guard=guard, tolerance=arguments.tolerance,
            max_pot_multiple=arguments.max_pot_multiple, log=log,
            chunk_pairs=arguments.chunk_pairs, resume=existing, on_chunk=save_chunk)
        checkpoint_arms()

    off, on = arms["off"], arms["on"]
    left, right = off["pair_samples"], on["pair_samples"]
    if len(left) != len(right):
        raise SystemExit(f"arms disagree on pair count ({len(left)} vs {len(right)})")

    # LBR's number is what the PROBE wins. Positive delta = the probe wins MORE
    # with the guard on, i.e. the guard made the agent more exploitable.
    # Negative = the guard reduced exploitability, which is the case for shipping it.
    deltas = [b - a for a, b in zip(left, right)]
    paired = _summary(deltas)
    identical = sum(1 for value in deltas if value == 0.0)
    verdict = (
        "GUARD REDUCES EXPLOITABILITY" if paired["ci_high"] < 0
        else "GUARD INCREASES EXPLOITABILITY" if paired["ci_low"] > 0
        else "INCONCLUSIVE"
    )

    report = {
        "gate": "lbr-all-in-geometry-guard-on-vs-off",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(arguments.checkpoint),
        "stack_bb": arguments.stack_bb,
        "seed": arguments.seed,
        "search": "blueprint only (all search mechanisms off in both arms)",
        "guard_tolerance": arguments.tolerance,
        "guard_max_pot_multiple": arguments.max_pot_multiple,
        "off": {k: v for k, v in off.items() if k not in ("pair_samples", "fallbacks")},
        "on": {k: v for k, v in on.items() if k not in ("pair_samples", "fallbacks")},
        "paired_delta": paired,
        "pairs_identical": identical,
        "verdict": verdict,
    }

    log("")
    log("--- paired difference (guard ON minus OFF), NEGATIVE means the guard helps ---")
    log(f"  {paired['bb_per_100']:+.2f} bb/100 "
        f"[{paired['ci_low']:+.2f}, {paired['ci_high']:+.2f}] over {paired['pairs']} pairs")
    log(f"  pairs where the guard changed nothing: {identical}/{len(deltas)}")
    if identical == len(deltas):
        log("  WARNING: the guard never fired. Either the probe's lines do not exhaust")
        log("  the raise cap, or the flag is not reaching the decision path. Treat a")
        log("  0.00 delta as UNMEASURED, not as evidence the guard is neutral.")
    log(f"  VERDICT: {verdict}")

    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"written to {output}")
    partial_path.unlink(missing_ok=True)
    log("=== gate complete ===")
    log.close()


if __name__ == "__main__":
    main()
