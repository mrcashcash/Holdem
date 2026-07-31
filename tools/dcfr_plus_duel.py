"""P2 decision gate: train DCFR and DCFR+ blueprints, then duel them with CRN.

Why a duel and not exploitability: `abstract_exploitability_mbb` cannot measure
this change. On the fixed-river control game it reads exactly 0.00 mbb for BOTH
averaging rules at 1,000 and 4,000 iterations, raises IndexError below ~500
iterations on an under-trained strategy, and raises IndexError outright on a
larger tree. It is a converged-strategy instrument with no dynamic range for a
convergence-SPEED claim (see tools/dcfr_plus_gate.py).

A head-to-head duel does have range, and it is this project's NULL-tested
promotion instrument (`tests/test_duel_null.py`, and `head_to_head`'s CRN null
reads exactly +0.00 once coupling reaches the agent under test).

Both arms are trained from scratch at the same depth, seed, iteration count and
abstraction, into tag-isolated artifact directories so no champion is touched.
The ONLY difference is the average-policy weighting:

    control   iteration t weighted t**gamma   (DCFR, every champion to date)
    challenger                max{0, t - d}   (DCFR+, Supremus, d = 100)

20bb is chosen because it is the cheapest real config (36,906 nodes; 5,000
iterations took 1,433 s when the native 20bb blueprint was trained), so the
answer arrives in about an hour instead of most of a day.

Usage:
    python tools/dcfr_plus_duel.py --iterations 5000 --pairs 3000
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

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="DCFR vs DCFR+ trained-blueprint duel")
    parser.add_argument("--stack-bb", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=5000)
    # MATCHED to build_solver's hardcoded averaging_delay=1000 on purpose.
    # DCFR zeroes the strategy sums until t > averaging_delay; DCFR+ zeroes them
    # until t > d. Using Supremus's d=100 against the trainer's delay of 1000
    # would change TWO things at once -- how many early iterations are discarded
    # AND the weighting rule -- and the discard difference dominates: at 300
    # iterations the DCFR control accumulates nothing at all (300 < 1000) while
    # DCFR+ averages 200 iterations, which alone read +142 bb/100 in a smoke run.
    # Matching the delays leaves the weighting rule as the only variable.
    parser.add_argument("--delay", type=int, default=1000,
                        help="DCFR+ d. Defaults to 1000 to match the trainer's "
                             "averaging_delay so only the weighting rule differs; "
                             "pass 100 for Supremus's value, accepting the confound")
    parser.add_argument("--pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260731,
                        help="solver seed, IDENTICAL in both arms")
    parser.add_argument("--duel-seed", type=int, default=4242)
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/dcfr-plus-duel.json"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output.with_suffix(".log")
    handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {message}"
        print(line)
        sys.stdout.flush()
        handle.write(line + "\n")
        handle.flush()

    # Both arms discard their first `delay` iterations, so a run that barely
    # exceeds it compares two nearly-empty average policies. Require real
    # headroom rather than silently producing a meaningless duel.
    if arguments.iterations < 3 * arguments.delay:
        raise SystemExit(
            f"--iterations {arguments.iterations} is too few for delay "
            f"{arguments.delay}: both arms discard the first {arguments.delay} "
            f"iterations, so at least {3 * arguments.delay} is needed for the "
            f"averaged policies to be meaningful."
        )

    depth = int(arguments.stack_bb)
    arms = {
        "dcfr": {"tag": f"dcfr_ctl{arguments.iterations}", "delay": None},
        "dcfr_plus": {"tag": f"dcfrplus{arguments.delay}_{arguments.iterations}",
                      "delay": arguments.delay},
    }

    log(f"=== DCFR vs DCFR+ duel: {depth}bb, {arguments.iterations:,} iterations, "
        f"d={arguments.delay}, solver seed {arguments.seed} ===")
    log(f"durable log: {log_path}")

    for name, spec in arms.items():
        data_dir = REPO / "backend" / "data" / f"gpu_blueprint_{depth}bb_{spec['tag']}"
        spec["data_dir"] = data_dir
        checkpoint = data_dir / "checkpoint.npz"
        spec["checkpoint"] = checkpoint
        # Resume on ITERATION COUNT, never on mere existence. A killed run leaves
        # a valid checkpoint at whatever multiple of --save-every it reached, so
        # "exists" would silently duel an under-trained challenger against a
        # finished control and attribute the gap to the averaging rule. This
        # actually happened: a kill left the control at 5,000 and the challenger
        # at 4,500.
        remaining = arguments.iterations
        if checkpoint.exists():
            import numpy as np

            with np.load(checkpoint, allow_pickle=False) as payload:
                have = int(payload["iteration"]) if "iteration" in payload else 0
            if have == arguments.iterations:
                log(f"-- {name}: complete at {have:,} iterations; reusing")
                continue
            if have > arguments.iterations:
                raise SystemExit(
                    f"{name} checkpoint is at {have:,} iterations, beyond the "
                    f"requested {arguments.iterations:,}; training cannot be undone. "
                    f"Use a different --tag or delete {checkpoint.parent}."
                )
            # The trainer's --iterations is an INCREMENT, not an absolute target
            # (docs/20BB_BLUEPRINT_PLAN.md), so ask only for the shortfall.
            remaining = arguments.iterations - have
            log(f"-- {name}: resuming from {have:,}; training {remaining:,} more")
        command = [
            sys.executable, "-m", "backend.solver.gpu.train",
            "--stack-bb", str(arguments.stack_bb),
            "--abstraction", "histogram",
            "--iterations", str(remaining),
            "--save-every", "500",
            "--batch-boards", "1",
            "--device", "cuda",
            "--seed", str(arguments.seed),
            "--tag", spec["tag"],
        ]
        if spec["delay"] is not None:
            command += ["--dcfr-plus-delay", str(spec["delay"])]
        log(f"-- training {name}: {' '.join(command[2:])}")
        started = time.time()
        completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        elapsed = round(time.time() - started, 1)
        if completed.returncode != 0:
            log(f"   TRAINING FAILED rc={completed.returncode}")
            log(f"   {(completed.stderr or completed.stdout)[-800:]}")
            raise SystemExit(1)
        log(f"   trained in {elapsed}s -> {checkpoint}")
        spec["train_s"] = elapsed

    # -- duel ------------------------------------------------------------------
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
    from backend.eval.duel import head_to_head

    # Final guard before any number is produced: both arms must sit at exactly
    # the requested iteration count. Unequal training is the one confound that
    # would masquerade most convincingly as an averaging-rule effect.
    import numpy as np

    for name, spec in arms.items():
        with np.load(spec["checkpoint"], allow_pickle=False) as payload:
            have = int(payload["iteration"]) if "iteration" in payload else -1
        if have != arguments.iterations:
            raise SystemExit(
                f"{name} is at {have:,} iterations, not {arguments.iterations:,}. "
                f"Refusing to duel unequally-trained arms."
            )
        spec["iteration"] = have
    log(f"both arms verified at exactly {arguments.iterations:,} iterations")

    def load(path: Path):
        agent = GpuBlueprintAgent.try_load(path)
        if agent is None:
            raise SystemExit(f"could not load {path}")
        # Blueprint only: this measures the averaging rule, not any search path.
        agent.subgame_search = False
        agent.flop_search = False
        agent.exact_river_search = False
        agent.continual_search = False
        agent.all_in_geometry_guard = False
        return agent

    log("")
    log("-- NULL first: control vs ITSELF must read exactly +0.00 with CRN --")
    null = head_to_head(
        load(arms["dcfr"]["checkpoint"]), load(arms["dcfr"]["checkpoint"]),
        stack_bb=arguments.stack_bb, pairs=min(400, arguments.pairs),
        seed=arguments.duel_seed, common_random_numbers=True,
    )
    log(f"   null: {null['mean_bb_per_100']:+.2f} bb/100 "
        f"[{null['ci_low_bb_per_100']:+.2f},{null['ci_high_bb_per_100']:+.2f}]")
    if abs(null["mean_bb_per_100"]) > 1e-9:
        log("   NULL FAILED -- coupling is not reaching these agents. The duel below")
        log("   would carry avoidable noise; do not believe it. Stopping.")
        raise SystemExit(1)

    log("")
    log(f"-- duel: DCFR+ (challenger) vs DCFR (control), {arguments.pairs} pairs, CRN --")
    started = time.time()
    result = head_to_head(
        load(arms["dcfr_plus"]["checkpoint"]), load(arms["dcfr"]["checkpoint"]),
        stack_bb=arguments.stack_bb, pairs=arguments.pairs,
        seed=arguments.duel_seed, common_random_numbers=True,
    )
    duel_s = round(time.time() - started, 1)
    mean = result["mean_bb_per_100"]
    low, high = result["ci_low_bb_per_100"], result["ci_high_bb_per_100"]
    log(f"   DCFR+ minus DCFR: {mean:+.2f} bb/100 [{low:+.2f},{high:+.2f}] in {duel_s}s")

    verdict = ("DCFR+ BETTER" if low > 0 else
               "DCFR+ WORSE" if high < 0 else "INCONCLUSIVE")
    log(f"   VERDICT: {verdict}")
    if verdict == "INCONCLUSIVE":
        log("   The interval spans zero, so the averaging rule is not shown to matter")
        log("   at this depth and iteration count. Do NOT change the default on it.")

    report = {
        "gate": "dcfr-plus-vs-dcfr-duel",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stack_bb": arguments.stack_bb,
        "iterations": arguments.iterations,
        "dcfr_plus_delay": arguments.delay,
        "solver_seed": arguments.seed,
        "duel_seed": arguments.duel_seed,
        "pairs": arguments.pairs,
        "null_bb_per_100": null["mean_bb_per_100"],
        "challenger_minus_control_bb_per_100": mean,
        "ci_low": low,
        "ci_high": high,
        "verdict": verdict,
        "arms": {name: {"tag": spec["tag"], "train_s": spec.get("train_s")}
                 for name, spec in arms.items()},
    }
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    handle.close()


if __name__ == "__main__":
    main()
