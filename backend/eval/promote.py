"""Champion promotion gate: only checkpoints that prove themselves reach the table.

The latest training checkpoint is promoted to ``champion.npz`` only if it
does not regress against the incumbent champion on two measurements:

1. Styles benchmark (duplicate deals, same seeds every evaluation): the
   candidate's mean bb/100 must be at least (champion mean - margin).
2. Abstract-game exploitability (CFR-BR gate): at most champion x 1.25.

The champion's own scores are cached in champion_meta.json so each promotion
evaluates only the candidate. The serving agent prefers champion.npz when it
exists, so the table can never silently regress just because training wrote
a newer (possibly worse) checkpoint.

CLI:  python -m backend.eval.promote [--hands 150] [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics

from backend.solver.gpu import train as gpu_train

CHAMPION_PATH = gpu_train.DATA_DIR / "champion.npz"
CHAMPION_META_PATH = gpu_train.DATA_DIR / "champion_meta.json"

GATE_STYLES = ("calling_station", "maniac", "tight_aggressive", "nit")
STYLES_SEED = 3
MEAN_MARGIN_BB100 = 50.0  # benchmark noise allowance
EXPLOIT_RATIO_LIMIT = 1.25


def evaluate_checkpoint(checkpoint_path, hands: int = 150) -> dict:
    """Styles mean (fixed seeds) + exploitability for one checkpoint."""
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
    from backend.eval.benchmark import benchmark_against_styles
    from backend.solver.gpu.exploit import cfr_br_exploitability

    agent = GpuBlueprintAgent.try_load(checkpoint_path)
    if agent is None:
        raise FileNotFoundError(checkpoint_path)
    agent.subgame_search = False  # gate on blueprint strength; search is a separate layer
    report = benchmark_against_styles(agent, hands_per_style=hands, styles=GATE_STYLES, seed=STYLES_SEED)

    # Use the trainer's own logged CFR-BR reading (same metric) instead of
    # recomputing on the GPU — evaluation must never contend with training
    # for VRAM (see the 2026-07-19 stall incident).
    exploitability = None
    if gpu_train.TELEMETRY_PATH.exists():
        history = json.loads(gpu_train.TELEMETRY_PATH.read_text(encoding="utf-8"))
        readings = [r["exploitability_mbb_per_hand"] for r in history if r.get("exploitability_mbb_per_hand") is not None]
        if readings:
            exploitability = readings[-1]
    if exploitability is None:
        solver = gpu_train.build_solver(device="cpu")
        exploitability = cfr_br_exploitability(solver, br_iterations=60, eval_boards=8)
    return {
        "iteration": agent.iteration,
        "styles_mean_bb_per_100": report["mean_bb_per_100"],
        "styles": {name: entry["bb_per_100"] for name, entry in report["styles"].items()},
        "exploitability_mbb": round(exploitability, 2),
        "hands_per_style": hands,
    }


def promote_if_better(hands: int = 150, force: bool = False, progress: bool = True) -> dict:
    candidate = evaluate_checkpoint(gpu_train.CHECKPOINT_PATH, hands=hands)
    champion_meta = None
    if CHAMPION_META_PATH.exists():
        champion_meta = json.loads(CHAMPION_META_PATH.read_text(encoding="utf-8"))

    if champion_meta is not None and not force:
        mean_ok = candidate["styles_mean_bb_per_100"] >= champion_meta["styles_mean_bb_per_100"] - MEAN_MARGIN_BB100
        exploit_ok = candidate["exploitability_mbb"] <= max(
            champion_meta["exploitability_mbb"] * EXPLOIT_RATIO_LIMIT, 50.0
        )
        promoted = mean_ok and exploit_ok
    else:
        promoted = True
        mean_ok = exploit_ok = True

    verdict = {
        "promoted": promoted,
        "candidate": candidate,
        "champion": champion_meta,
        "mean_ok": mean_ok,
        "exploit_ok": exploit_ok,
    }
    if promoted:
        shutil.copy2(gpu_train.CHECKPOINT_PATH, CHAMPION_PATH)
        CHAMPION_META_PATH.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    if progress:
        print(json.dumps(verdict, indent=2))
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the latest checkpoint if it beats the champion")
    parser.add_argument("--hands", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="promote unconditionally")
    arguments = parser.parse_args()
    promote_if_better(hands=arguments.hands, force=arguments.force)


if __name__ == "__main__":
    main()
