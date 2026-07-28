"""Promotion-grade evaluation orchestrator for GPU blueprints.

The gate separates a cheap screening block from a disjoint confirmatory block,
checks retained champions for non-transitive regressions, optionally compares
GPU-LBR results, records serving/translation diagnostics, and writes a
reproducible manifest for every invocation.

CLI:
    python -m backend.eval.gate \
        --data-dir backend/data/gpu_blueprint_200bb_nolimp \
        --stack-bb 200 --promote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from backend.eval.duel import head_to_head, promote

EVALUATOR_VERSION = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state(root: Path) -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"revision": revision, "dirty": bool(status), "changed_paths": status}
    except Exception as error:
        return {"revision": None, "dirty": None, "error": str(error)}


def _load_agent(path: Path):
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    agent = GpuBlueprintAgent.try_load(path)
    if agent is None:
        raise FileNotFoundError(path)
    agent.subgame_search = False
    return agent


def _agent_manifest(path: Path, agent) -> dict:
    config = asdict(agent.tree.config)
    sampler = agent.sampler.state()
    abstraction = {"config": config, "sampler": sampler}
    sampler_summary = {
        key: value
        for key, value in sampler.items()
        if key not in {"hist_centroids", "std_edges", "potential_state"}
    }
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "abstraction_sha256": _json_sha256(abstraction),
        "bytes": path.stat().st_size,
        "iteration": int(agent.iteration),
        "config": config,
        "sampler": sampler_summary,
    }


def _retained_paths(data_dir: Path, incumbent: Path, limit: int) -> list[Path]:
    candidates = sorted(
        data_dir.glob("champion-*-backup.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [path for path in candidates if path.resolve() != incumbent.resolve()][: max(0, limit)]


def _mapping_gate(confirm: dict) -> dict:
    diagnostics = confirm.get("diagnostics")
    if not diagnostics:
        return {"ok": False, "reason": "confirmatory duel did not collect diagnostics"}
    challenger = diagnostics["challenger"]
    champion = diagnostics["champion"]
    fallback_limit = max(0.01, champion["fallback_rate"] + 0.005)
    translation_limit = champion["mean_translation_gap_pot"] + 0.05
    fallback_ok = challenger["fallback_rate"] <= fallback_limit
    translation_ok = challenger["mean_translation_gap_pot"] <= translation_limit
    return {
        "ok": fallback_ok and translation_ok,
        "fallback_ok": fallback_ok,
        "translation_ok": translation_ok,
        "challenger_fallback_rate": challenger["fallback_rate"],
        "champion_fallback_rate": champion["fallback_rate"],
        "fallback_limit": round(fallback_limit, 6),
        "challenger_mean_translation_gap_pot": challenger["mean_translation_gap_pot"],
        "champion_mean_translation_gap_pot": champion["mean_translation_gap_pot"],
        "translation_limit": round(translation_limit, 6),
    }


def _lbr_gate(challenger, incumbent, pairs: int, seed: int, stack_bb: float) -> dict:
    if pairs <= 0:
        return {"enabled": False, "ok": True}
    from backend.eval.lbr import local_best_response_probe

    challenger_report = local_best_response_probe(
        challenger,
        hands=pairs * 2,
        seed=seed,
        stack_bb=stack_bb,
    )
    incumbent_report = local_best_response_probe(
        incumbent,
        hands=pairs * 2,
        seed=seed,
        stack_bb=stack_bb,
    )
    # Reject only a statistically clear exploitability regression. A noisy tie
    # keeps the incumbent comparison neutral rather than pretending precision.
    ok = challenger_report["ci_low_bb_per_100"] <= incumbent_report["ci_high_bb_per_100"]
    return {
        "enabled": True,
        "ok": ok,
        "seed": seed,
        "challenger": challenger_report,
        "incumbent": incumbent_report,
    }


def evaluate_gate(
    data_dir: Path,
    stack_bb: float,
    challenger_path: Path | None = None,
    incumbent_path: Path | None = None,
    screen_pairs: int = 750,
    confirm_pairs: int = 3000,
    retained_pairs: int = 1500,
    retained_limit: int = 3,
    lbr_pairs: int = 100,
    seed: int | None = None,
    install: bool = False,
) -> dict:
    if stack_bb <= 0:
        raise ValueError("stack_bb must be positive")
    for label, value in (
        ("screen_pairs", screen_pairs),
        ("confirm_pairs", confirm_pairs),
        ("retained_pairs", retained_pairs),
    ):
        if value <= 0:
            raise ValueError(f"{label} must be positive")
    for label, value in (("retained_limit", retained_limit), ("lbr_pairs", lbr_pairs)):
        if value < 0:
            raise ValueError(f"{label} cannot be negative")

    root = Path(__file__).resolve().parents[2]
    data_dir = data_dir.resolve()
    challenger_path = (challenger_path or data_dir / "checkpoint.npz").resolve()
    incumbent_path = (incumbent_path or data_dir / "champion.npz").resolve()
    base_seed = int(seed if seed is not None else secrets.randbits(30))
    seeds = {
        "screen": base_seed,
        "confirm": base_seed + 1_000_003,
        "retained": base_seed + 2_000_003,
        "lbr": base_seed + 3_000_017,
    }

    challenger = _load_agent(challenger_path)
    incumbent = _load_agent(incumbent_path)
    for label, agent in (("challenger", challenger), ("incumbent", incumbent)):
        trained_depth = float(agent.tree.config.stack_bb)
        if abs(trained_depth - stack_bb) > 1e-6:
            raise ValueError(f"{label} was trained for {trained_depth:g}bb, not {stack_bb:g}bb")

    started = time.time()
    report = {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_dir": str(data_dir),
        "stack_bb": float(stack_bb),
        "seeds": seeds,
        "budgets": {
            "screen_pairs": screen_pairs,
            "confirm_pairs": confirm_pairs,
            "retained_pairs": retained_pairs,
            "retained_limit": retained_limit,
            "lbr_pairs": lbr_pairs,
        },
        "git": _git_state(root),
        "challenger": _agent_manifest(challenger_path, challenger),
        "incumbent": _agent_manifest(incumbent_path, incumbent),
    }

    print(
        f"screen: {screen_pairs} pairs seed={seeds['screen']} "
        f"(challenger iter={challenger.iteration}, incumbent iter={incumbent.iteration})",
        flush=True,
    )
    screen = head_to_head(
        challenger,
        incumbent,
        stack_bb,
        pairs=screen_pairs,
        seed=seeds["screen"],
        collect_diagnostics=True,
    )
    report["screen"] = screen
    screen_pass = screen["mean_bb_per_100"] > 0 and screen["verdict"] != "REGRESSION"
    report["screen_pass"] = screen_pass
    print(
        f"screen: {screen['mean_bb_per_100']:+.2f} bb/100 "
        f"[{screen['ci_low_bb_per_100']:+.2f}, {screen['ci_high_bb_per_100']:+.2f}] "
        f"{screen['verdict']} pass={screen_pass}",
        flush=True,
    )

    if not screen_pass:
        report.update(
            {
                "confirm": None,
                "retained_crossplay": [],
                "mapping_gate": {"ok": False, "reason": "screen did not pass"},
                "lbr_gate": {"enabled": False, "ok": True, "reason": "screen did not pass"},
                "promotion": {"eligible": False, "installed": False, "reason": "screen did not pass"},
            }
        )
    else:
        print(f"confirm: {confirm_pairs} pairs seed={seeds['confirm']}", flush=True)
        confirm = head_to_head(
            challenger,
            incumbent,
            stack_bb,
            pairs=confirm_pairs,
            seed=seeds["confirm"],
            collect_diagnostics=True,
        )
        report["confirm"] = confirm
        mapping = _mapping_gate(confirm)
        report["mapping_gate"] = mapping
        print(
            f"confirm: {confirm['mean_bb_per_100']:+.2f} bb/100 "
            f"[{confirm['ci_low_bb_per_100']:+.2f}, {confirm['ci_high_bb_per_100']:+.2f}] "
            f"{confirm['verdict']}; mapping_ok={mapping['ok']}",
            flush=True,
        )

        retained_reports = []
        retained_ok = True
        for index, path in enumerate(_retained_paths(data_dir, incumbent_path, retained_limit)):
            retained = _load_agent(path)
            if abs(float(retained.tree.config.stack_bb) - stack_bb) > 1e-6:
                retained_reports.append({"path": str(path), "skipped": "stack-depth mismatch"})
                continue
            result = head_to_head(
                challenger,
                retained,
                stack_bb,
                pairs=retained_pairs,
                seed=seeds["retained"] + index * 100_003,
            )
            result["opponent"] = _agent_manifest(path, retained)
            retained_reports.append(result)
            retained_ok = retained_ok and result["verdict"] != "REGRESSION"
        report["retained_crossplay"] = retained_reports
        report["retained_ok"] = retained_ok
        print(
            f"retained cross-play: opponents={len(retained_reports)} ok={retained_ok}",
            flush=True,
        )

        print(f"LBR: {lbr_pairs} pairs per model seed={seeds['lbr']}", flush=True)
        lbr = _lbr_gate(challenger, incumbent, lbr_pairs, seeds["lbr"], stack_bb)
        report["lbr_gate"] = lbr
        print(f"LBR: enabled={lbr['enabled']} ok={lbr['ok']}", flush=True)

        eligible = (
            confirm["verdict"] == "PROMOTE"
            and retained_ok
            and mapping["ok"]
            and lbr["ok"]
        )
        installed = False
        reason = "all confirmatory gates passed" if eligible else "one or more confirmatory gates failed"
        if eligible and install:
            if challenger_path != (data_dir / "checkpoint.npz").resolve():
                raise ValueError("--promote requires the challenger's path to be data-dir/checkpoint.npz")
            promote(
                data_dir,
                confirm,
                challenger_iteration=int(challenger.iteration),
                champion_iteration=int(incumbent.iteration),
            )
            installed = True
        report["promotion"] = {"eligible": eligible, "installed": installed, "reason": reason}

    report["elapsed_seconds"] = round(time.time() - started, 3)
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_dir = data_dir / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = output_dir / f"gate-{stamp}-iter{challenger.iteration}.json"
    report["report_path"] = str(output.resolve())
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Promotion-grade GPU blueprint gate")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--stack-bb", type=float, required=True)
    parser.add_argument("--challenger", type=str, default=None)
    parser.add_argument("--incumbent", type=str, default=None)
    parser.add_argument("--screen-pairs", type=int, default=750)
    parser.add_argument("--confirm-pairs", type=int, default=3000)
    parser.add_argument("--retained-pairs", type=int, default=1500)
    parser.add_argument("--retained-limit", type=int, default=3)
    parser.add_argument("--lbr-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--promote", action="store_true")
    arguments = parser.parse_args()

    report = evaluate_gate(
        data_dir=Path(arguments.data_dir),
        stack_bb=arguments.stack_bb,
        challenger_path=Path(arguments.challenger) if arguments.challenger else None,
        incumbent_path=Path(arguments.incumbent) if arguments.incumbent else None,
        screen_pairs=arguments.screen_pairs,
        confirm_pairs=arguments.confirm_pairs,
        retained_pairs=arguments.retained_pairs,
        retained_limit=arguments.retained_limit,
        lbr_pairs=arguments.lbr_pairs,
        seed=arguments.seed,
        install=arguments.promote,
    )
    promotion = report["promotion"]
    print(
        f"gate eligible={promotion['eligible']} installed={promotion['installed']} "
        f"report={report['report_path']}"
    )


if __name__ == "__main__":
    main()
