---
name: holdem-training-improvement
description: Diagnose and autonomously improve the Holdem self-play trainer from `backend/data/training_reports/latest.json`. Use when asked to investigate poor Holdem training results, trace report warnings into the Python backend, implement the justified fixes, or run and assess local training validation through the API on port 8000. Accept an optional positive integer iteration argument to repeat the entire improvement workflow that many times.
---

# Holdem Training Improvement

> **FROZEN (2026-07-17): Do not run this workflow.** Per-report-warning patching of the
> PPO trainer is what accreted the interacting subsystems that caused the metric decline
> (see `docs/REDESIGN_PLAN.md` §1.3). The trainer is being replaced by the CFR blueprint +
> search architecture in that plan. Any new loss term or subsystem now requires an A/B
> ablation with confidence intervals, not a report-warning trigger.

Run the whole workflow without an approval checkpoint. Preserve unrelated working-tree changes and never claim that a model improved merely because a run completed.

## Iteration contract

Accept an optional positive integer as the iteration count. No number means one iteration; a request such as `$holdem-training-improvement 3` means three consecutive, complete iterations. Treat the number as the total count, not the number of extra retries. Require a positive whole number; ask for clarification rather than silently rounding, clamping, or treating zero as one.

For every iteration, repeat every workflow step below in order: establish a new baseline, inspect and trace the current report, classify findings, state the implementation plan, implement only justified changes, run code validation, start and assess a new training validation, and report the result. Do not collapse multiple iterations into one diagnosis or one training run. If an iteration finds no justified code change, make none; still run and assess that iteration's training validation.

Before each iteration, record the current timestamped report and its key metrics as that iteration's baseline. Compare its validation report to that baseline, then use the newly written report as the next iteration's baseline. Keep report paths and metrics distinct in the final summary so that no result is attributed to the wrong iteration.

Read [the project contract](references/project-contract.md) before using the API or interpreting a report. Use `scripts/inspect_training_report.py` for a structured first pass and `scripts/wait_for_training.py` after starting an API run.

## Workflow

1. Establish the repository root and require the current report at `backend/data/training_reports/latest.json`. Inspect its JSON schema, `run`, `diagnostics`, `telemetry`, `trainer`, and the most recent evaluation history; do not rely only on a warning summary.
2. Run the report inspector. Treat its output as an index, then verify every warning or adverse metric in the raw report and trace it to its producing logic. Inspect `backend/learning.py` for learning, rollout, evaluation, promotion, report-generation, and checkpoint code; inspect `backend/main.py` for lifecycle and API behavior; inspect `backend/poker.py`, `backend/rl_env.py`, and `backend/vectorized_engine.py` when an issue could be environmental or action-legality related.
3. Classify findings before changing code:
   - **Run validity:** incomplete/error run, smoke-only evidence, missing final audit, insufficient sample size, stale report, or invalid report write.
   - **Learning safety:** hard-KL recovery, rollback/retry, skipped updates, exploding losses, device/worker failures, or checkpoint safety.
   - **Strength and robustness:** negative holdout, benchmark, fixed-adversary lower confidence bound, restricted best-response proxy, audit proxy, tail-loss concentration, and promotion-gate evidence.
   - **Policy behavior:** illegal or oversized sizing, fold collapse, excessive low-commitment all-ins, weak forced preflop roots, range/calibration quality, and inadequate teacher coverage.
   - **Expected limitations:** do not call an intentionally disabled approximate resolver a defect. It is a limitation only if the requested improvement requires a sound resolver.
4. Build a complete implementation plan in the working response, ordered by severity and expected causal impact. For every item state the report evidence, likely root cause, files/functions, concrete change, regression test, and metric that must improve. Do not weaken promotion floors, hide diagnostics, or relabel a proxy as formal no-limit Hold'em exploitability to make a report pass.
5. Implement all justified improvements. Add focused automated tests for deterministic logic and API/report contracts alongside the affected code. Keep report fields backward-compatible; if a schema change is unavoidable, bump the schema version and make the producer and consumers consistent.
6. Validate the code before expensive training: run the affected tests, then the full available test suite, and run `python -m compileall backend`. Exercise changed API validation or lifecycle behavior against the reloading server when applicable. Fix failures rather than proceeding with a known-bad build.
7. Choose and run the post-change training validation using the decision rule below. Start it via the API, wait for its report, verify that it is a newly produced file, and analyze it against that iteration's pre-change baseline.
8. Report the iteration number, implementation changes, commands/tests run, exact report file, validity checks, before/after metrics, remaining failures, and the next highest-value improvement. A completed training run is a result even when it fails its quality gates; say so plainly.

## Training validation decision

Default to a **5,000-hand full-audit run** after a successful code verification:

```json
{"episodes": 5000, "fresh": false, "smoke_test": false}
```

Use this default for every iteration unless a 10,000-hand run is genuinely necessary. Although it is smaller, it remains a full audit: `smoke_test: false` causes the trainer to perform its final full evaluation and audit before writing the final report.

Escalate an iteration to a **10,000-hand full-audit run** only when at least one of these conditions applies: the 5,000-hand result leaves a promotion gate or regression decision materially uncertain; confidence bounds overlap enough that the before/after conclusion would be unreliable; a problem appears only in longer runs or tail behavior; or the user explicitly needs promotion-grade/high-confidence evidence. State the concrete reason before escalating; do not use 10,000 hands merely by habit.

```json
{"episodes": 10000, "fresh": false, "smoke_test": false}
```

Use a 5,000-hand fresh smoke run only as a preceding safety gate when changing model construction, fresh-start/checkpoint behavior, tensor shapes, rollout initialization, or CUDA worker setup:

```json
{"episodes": 5000, "fresh": true, "smoke_test": true}
```

After that smoke run completes cleanly, automatically run the selected 5,000- or justified 10,000-hand full-audit continuation with `fresh: false`. Do not use the smoke report as the final verdict.

Avoid `fresh: true` for the full evaluation unless the change cannot be meaningfully tested from the existing checkpoint. Fresh mode replaces the live trainer and archives the current checkpoint pair, but the normal recovery endpoint does not select those `before-fresh-*` archives. Record their paths if fresh mode is necessary.

Use the API commands and completion contract in the project reference. Before posting, require `GET /api/training/status` to report `running: false`; a concurrent request returns HTTP 409. Once `report_status` is `written`, require all of the following from the generated report:

- `run.outcome == "completed"`, `run.error == null`, and completed hands equal requested hands;
- the timestamped report named by status exists under `backend/data/training_reports/` and is newer than the request;
- `run.smoke_test == false`, `run.promotion_eligible == true`, and `telemetry.final_audit_ran == true` for the final verdict;
- `latest.json` parses and has the same `generated_at_utc` as the timestamped final report.

Compare exact metrics and confidence bounds to the baseline. Keep failed gates visible; a relative improvement that remains below a required floor is still a failure.

## Completion standard

Finish only after the selected full-audit run has written and been checked, or after a concrete server/test failure prevents it. On failure, inspect the debug log and newest report, fix the cause when it is in scope, and rerun the validation. Do not fabricate a result file; the trainer writes a timestamped JSON report and atomically updates `latest.json` in its `finally` path.
