#!/usr/bin/env python3
"""Validate and summarize a Holdem training report without changing project state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def add_finding(findings: list[dict[str, Any]], priority: str, category: str, code: str, evidence: dict[str, Any]) -> None:
    findings.append({"priority": priority, "category": category, "code": code, "evidence": evidence})


def inspect(report: dict[str, Any], path: Path) -> dict[str, Any]:
    errors: list[str] = []
    for key in ("run", "diagnostics", "telemetry", "settings", "trainer"):
        if key not in report:
            errors.append(f"missing top-level key: {key}")

    run = report.get("run") if isinstance(report.get("run"), dict) else {}
    telemetry = report.get("telemetry") if isinstance(report.get("telemetry"), dict) else {}
    settings = report.get("settings") if isinstance(report.get("settings"), dict) else {}
    diagnostics = report.get("diagnostics") if isinstance(report.get("diagnostics"), list) else []
    findings: list[dict[str, Any]] = []

    requested = run.get("requested_hands")
    completed = run.get("completed_hands")
    if run.get("outcome") != "completed" or run.get("error") is not None or requested != completed:
        add_finding(findings, "blocker", "run_validity", "run_not_completed", {"outcome": run.get("outcome"), "error": run.get("error"), "requested_hands": requested, "completed_hands": completed})
    if run.get("smoke_test"):
        add_finding(findings, "high", "run_validity", "smoke_run_is_not_final_evidence", {"promotion_eligible": run.get("promotion_eligible"), "final_audit_ran": telemetry.get("final_audit_ran")})
    elif not telemetry.get("final_audit_ran"):
        add_finding(findings, "blocker", "run_validity", "full_run_missing_final_audit", {"final_audit_ran": telemetry.get("final_audit_ran")})

    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "unknown"))
        code = str(item.get("code", "unnamed_diagnostic"))
        if severity in {"warning", "error"}:
            add_finding(findings, "high" if severity == "warning" else "blocker", "reported_diagnostic", code, {"message": item.get("message"), "metrics": item.get("metrics", {})})

    adversarial = settings.get("adversarial") if isinstance(settings.get("adversarial"), dict) else {}
    preflop = settings.get("preflop_sizing") if isinstance(settings.get("preflop_sizing"), dict) else {}
    checks = (
        ("adversarial_ci_floor_bb_per_100", adversarial.get("promotion_lcb_floor_bb_per_100"), "robustness", "adversarial_lcb_below_configured_floor"),
        ("preflop_scenario_worst_lcb_bb_per_100", preflop.get("root_promotion_lcb_floor_bb_per_100"), "policy_behavior", "preflop_root_lcb_below_configured_floor"),
    )
    for metric, floor, category, code in checks:
        value = as_float(telemetry.get(metric))
        floor_number = as_float(floor)
        if value is not None and floor_number is not None and value < floor_number:
            add_finding(findings, "high", category, code, {"metric": metric, "value": value, "required_floor": floor_number})

    for metric, category, code in (
        ("restricted_br_bb_per_100", "robustness", "restricted_best_response_proxy_negative"),
        ("audit_exploitability_bb_per_100", "robustness", "fixed_style_audit_proxy_negative"),
        ("holdout_bb_per_100", "strength", "holdout_negative"),
        ("benchmark_bb_per_100", "strength", "benchmark_negative"),
        ("evaluation_bb_per_100", "strength", "evaluation_negative"),
    ):
        value = as_float(telemetry.get(metric))
        if value is not None and value < 0:
            add_finding(findings, "high", category, code, {"metric": metric, "value": value})

    hard_kl = as_float(telemetry.get("ppo_hard_kl"))
    kl_target = as_float(telemetry.get("ppo_kl_target"))
    if hard_kl is not None and kl_target is not None and hard_kl > kl_target:
        add_finding(findings, "high", "learning_safety", "ppo_hard_kl_above_target", {"ppo_hard_kl": hard_kl, "ppo_kl_target": kl_target, "rollback_phase": telemetry.get("ppo_rollback_phase"), "recovery_updates": telemetry.get("ppo_recovery_updates")})
    if telemetry.get("ppo_update_reverted"):
        add_finding(findings, "high", "learning_safety", "ppo_update_reverted", {"rollback_phase": telemetry.get("ppo_rollback_phase"), "post_step_retry_accepted": telemetry.get("ppo_post_step_retry_accepted")})

    guarded = as_float(telemetry.get("preflop_guarded_allin_probability"))
    target = as_float(telemetry.get("preflop_allin_target"))
    if guarded is not None and target is not None and guarded > target + 0.08:
        add_finding(findings, "high", "policy_behavior", "early_preflop_allin_probability_excess", {"guarded_probability": guarded, "target_probability": target, "guarded_state_fraction": telemetry.get("preflop_guarded_state_fraction")})

    return {
        "report_path": str(path),
        "schema_version": report.get("schema_version"),
        "generated_at_utc": report.get("generated_at_utc"),
        "valid": not errors,
        "validation_errors": errors,
        "run": run,
        "diagnostic_count": len(diagnostics),
        "warning_codes": [item.get("code") for item in diagnostics if isinstance(item, dict) and item.get("severity") in {"warning", "error"}],
        "key_metrics": {key: telemetry.get(key) for key in ("final_audit_ran", "gate_passed", "promotion_confidence", "holdout_bb_per_100", "holdout_ci_floor_bb_per_100", "benchmark_bb_per_100", "adversarial_ci_floor_bb_per_100", "restricted_br_bb_per_100", "audit_exploitability_bb_per_100", "tail_loss_rate", "tail_loss_bb", "preflop_scenario_worst_lcb_bb_per_100", "preflop_scenario_worst_root", "preflop_scenario_worst_style", "preflop_guarded_allin_probability", "preflop_allin_target", "ppo_hard_kl", "ppo_kl_target", "ppo_recovery_updates", "ppo_update_reverted")},
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", default="backend/data/training_reports/latest.json", type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"report_path": str(args.report), "valid": False, "validation_errors": [str(exc)]}, indent=2))
        return 2
    if not isinstance(document, dict):
        print(json.dumps({"report_path": str(args.report), "valid": False, "validation_errors": ["report root must be a JSON object"]}, indent=2))
        return 2
    result = inspect(document, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
