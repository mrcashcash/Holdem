# Holdem training project contract

## Scope

Work from the repository root. The authoritative current report is `backend/data/training_reports/latest.json`; timestamped result reports are stored in the same directory. The trainer debug log is `backend/data/training_reports/training-debug.jsonl`.

`backend/learning.py` owns training, evaluation, diagnostics, checkpoint/report writes, and the report schema. `backend/main.py` owns the local FastAPI lifecycle and background worker. Check the actual source before assuming a configuration constant or metric still has its current meaning.

## Live API (verified from OpenAPI and server)

Base URL: `http://127.0.0.1:8000`

| Operation | Contract |
| --- | --- |
| `GET /api/health` | Liveness check. |
| `GET /api/training/status` | Returns current/last run state, `running`, `report_filename`, and `report_status`. |
| `POST /api/training/start` | Requires JSON object `TrainingRequest`. `episodes` is integer 10–1,000,000 (default 50,000); `fresh` and `smoke_test` are booleans (default false). |

Use PowerShell to start the recommended full validation:

```powershell
$base = 'http://127.0.0.1:8000'
$payload = @{ episodes = 10000; fresh = $false; smoke_test = $false } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post -Uri "$base/api/training/start" -ContentType 'application/json' -Body $payload
```

For a fresh smoke preflight, use `$payload = @{ fresh = $true; smoke_test = $true } | ConvertTo-Json -Compress`. The server forces smoke runs to exactly 5,000 episodes, ignoring a supplied `episodes` value.

Poll completion with:

```powershell
python .codex/skills/holdem-training-improvement/scripts/wait_for_training.py --base-url http://127.0.0.1:8000
```

The start endpoint begins a background thread and returns immediately. It rejects another request while a run is active with HTTP 409. A fresh start copies existing checkpoints to `backend/data/checkpoint_archives/before-fresh-<UTC>/` before replacing the in-memory trainer.

## Result-file behavior

The trainer writes a timestamped `training-<UTC>-hands-<n>-updates-<n>.json` report at the end of each run, then atomically overwrites `latest.json`. It does this from the trainer finalization path even after a caught training failure; report-write failure itself is surfaced as `report_status` failure. Do not treat the old `latest.json` as the requested run's result until status identifies a written report and its timestamp/contents have been verified.

For non-smoke completed runs, `train_neural` calls its final evaluation with `force_full=True` and `final_audit=True`. For smoke runs it saves after the routine updates and deliberately skips that final audit. Consequently a smoke result has `run.promotion_eligible: false` and cannot validate model quality.

## Baseline interpretation

The currently supplied report is schema version 7 and a completed 5,000-hand smoke run. It lacks a final audit and reports multiple actionable warnings: failed promotion screening, adverse fixed-style confidence bounds, negative audit and restricted-best-response proxies, PPO hard-KL recovery, tail losses, a `facing_4bet` root vulnerability against `nit`, and overly frequent low-commitment preflop all-ins. Treat these as starting hypotheses, not permanent facts: recompute from the report read at invocation and trace them into code before changing anything.

Important metric interpretation:

- Compare adversarial and preflop **lower confidence bounds** to their report settings; do not replace them with raw win rates.
- Treat `restricted_br_bb_per_100` and `audit_exploitability_bb_per_100` as the explicitly documented restricted/fixed-style proxies, not formal exploitability.
- Treat `final_audit_ran`, smoke mode, completed/requested hands, and report timestamp as validity gates before comparing quality metrics.
- Consider the absent local teacher data informational unless the diagnosed change depends on teacher-data coverage.
- Consider an `approximate_resolver_disabled` diagnostic expected under the current safety setting, not a standalone regression.
