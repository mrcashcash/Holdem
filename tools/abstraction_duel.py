"""Does the histogram card abstraction beat the legacy scalar one at 200bb?

`docs/STATUS.md` (2026-07-31) records that the deployed 200bb champion is the one
depth still on the superseded scalar abstraction:

    20bb  histogram-EMD  169, 150, 150, 30   LBR  +13.34
   100bb  histogram-EMD  169, 150, 150, 30   LBR +118.64
   200bb  scalar         169,  20,  20, 20   LBR +252.45

7.5x coarser on flop and turn than either other depth, at the depth where the
agent is weakest by a factor of nineteen. §2.3 already credits histogram-EMD with
reaching parity against a 3.3x-trained scalar model and fixing a whole leak class,
but scalar@118k predates that switch.

The challenger was trained with the deployed MENU untouched (0.5/1.0, cap 3,
147,349 nodes) and the 100bb champion's fitted sampler imported, so the card
abstraction is the only variable.

`backend.eval.duel`'s CLI compares one artifact directory's checkpoint against its
own champion, which cannot express "these two checkpoints, in different directories,
with different configs" -- hence this script. It uses the same NULL-tested
`head_to_head` and refuses to report a number if the CRN null is not exactly zero,
because an uncoupled router duel carries about +/-80 bb/100 of avoidable noise
(§4 bug 5).

Usage:
    python tools/abstraction_duel.py --pairs 3000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="histogram vs scalar abstraction at 200bb")
    parser.add_argument("--challenger", type=Path,
                        default=Path("backend/data/gpu_blueprint_200bb_hist/checkpoint.npz"))
    parser.add_argument("--incumbent", type=Path,
                        default=Path("backend/data/gpu_blueprint_200bb/champion.npz"))
    parser.add_argument("--stack-bb", type=float, default=200.0)
    parser.add_argument("--pairs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", type=Path,
                        default=Path("backend/data/evaluations/abstraction-duel-200bb.json"))
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

    import numpy as np

    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
    from backend.eval.duel import head_to_head

    def describe(path: Path) -> dict:
        with np.load(path, allow_pickle=False) as payload:
            sampler = json.loads(str(payload["sampler"]))
            config = json.loads(str(payload["config"]))
            return {
                "path": str(path),
                "iteration": int(payload["iteration"]),
                "histogram": bool(sampler.get("histogram")),
                "buckets": [169, sampler.get("flop_buckets"),
                            sampler.get("turn_buckets"), sampler.get("river_buckets")],
                "postflop": config["postflop_fractions"],
                "raise_cap": config["max_raises_per_street"],
            }

    def load(path: Path):
        agent = GpuBlueprintAgent.try_load(path)
        if agent is None:
            raise SystemExit(f"could not load {path}")
        # Blueprint only: this measures the card abstraction, not any search path.
        agent.subgame_search = False
        agent.flop_search = False
        agent.exact_river_search = False
        agent.continual_search = False
        agent.all_in_geometry_guard = False
        return agent

    challenger_info = describe(arguments.challenger)
    incumbent_info = describe(arguments.incumbent)
    log(f"=== abstraction duel @ {arguments.stack_bb:.0f}bb, {arguments.pairs} pairs, CRN ===")
    for label, info in (("challenger", challenger_info), ("incumbent ", incumbent_info)):
        log(f"{label}: iter={info['iteration']:,} histogram={info['histogram']} "
            f"buckets={info['buckets']} postflop={info['postflop']} cap={info['raise_cap']}")
    if challenger_info["postflop"] != incumbent_info["postflop"] or \
            challenger_info["raise_cap"] != incumbent_info["raise_cap"]:
        log("WARNING: the betting menus differ, so this duel does NOT isolate the")
        log("card abstraction. Interpret accordingly.")
    log(f"durable log: {log_path}")

    log("")
    log("-- NULL: incumbent vs ITSELF must read exactly +0.00 with CRN --")
    null = head_to_head(load(arguments.incumbent), load(arguments.incumbent),
                        stack_bb=arguments.stack_bb, pairs=min(400, arguments.pairs),
                        seed=arguments.seed, common_random_numbers=True)
    log(f"   null: {null['mean_bb_per_100']:+.2f} bb/100 "
        f"[{null['ci_low_bb_per_100']:+.2f},{null['ci_high_bb_per_100']:+.2f}]")
    if abs(null["mean_bb_per_100"]) > 1e-9:
        log("   NULL FAILED — coupling is not reaching these agents, so the duel below")
        log("   would carry avoidable noise. Stopping rather than reporting it.")
        raise SystemExit(1)

    log("")
    log("-- duel: histogram (challenger) vs scalar (incumbent) --")
    started = time.time()
    result = head_to_head(load(arguments.challenger), load(arguments.incumbent),
                          stack_bb=arguments.stack_bb, pairs=arguments.pairs,
                          seed=arguments.seed, common_random_numbers=True)
    elapsed = round(time.time() - started, 1)
    mean = result["mean_bb_per_100"]
    low, high = result["ci_low_bb_per_100"], result["ci_high_bb_per_100"]
    verdict = ("HISTOGRAM BETTER" if low > 0 else
               "HISTOGRAM WORSE" if high < 0 else "INCONCLUSIVE")
    log(f"   histogram minus scalar: {mean:+.2f} bb/100 [{low:+.2f},{high:+.2f}] in {elapsed}s")
    log(f"   VERDICT: {verdict}")
    log("")
    if verdict == "HISTOGRAM BETTER":
        log("   A duel win is necessary, not sufficient. Promotion still needs LBR at")
        log("   20,000 pairs against the frozen +252.45 [+223.34, +281.55], plus the")
        log("   mapping and fallback gates. 5,000 iterations may also be undertrained")
        log("   on a 147,349-node tree; continuing to 10k is the cheaper next step than")
        log("   promoting early.")
    elif verdict == "INCONCLUSIVE":
        log("   The interval spans zero. Do NOT promote. The likeliest cause is")
        log("   undertraining rather than the abstraction being worthless: this tree is")
        log("   4x the one where histogram@5k won at 20bb.")

    arguments.output.write_text(json.dumps({
        "gate": "abstraction-duel-200bb",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "challenger": challenger_info,
        "incumbent": incumbent_info,
        "pairs": arguments.pairs,
        "seed": arguments.seed,
        "null_bb_per_100": null["mean_bb_per_100"],
        "challenger_minus_incumbent_bb_per_100": mean,
        "ci_low": low,
        "ci_high": high,
        "verdict": verdict,
        "elapsed_s": elapsed,
    }, indent=2), encoding="utf-8")
    log(f"written to {arguments.output}")
    handle.close()


if __name__ == "__main__":
    main()
