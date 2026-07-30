"""GPU blueprint training CLI (docs/GPU_CFR_PLAN.md §Phases 4-5).

    python -m backend.solver.gpu.train --iterations 2000 [--device cuda]

Artifacts in backend/data/gpu_blueprint/ (separate from the CPU blueprint —
    the abstractions differ): checkpoint.npz holds compact street-sharded
    tables plus the tree/sampler configuration; telemetry.json appends one row
    per save.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from backend.solver.gpu.cfr import VectorCFR
from backend.solver.gpu.action_profile import PHASE3_STATIC_PROFILE, load_compiled_profile
from backend.solver.gpu.deals import DealSampler
from backend.solver.gpu.tree import BettingTree, GpuActionConfig

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gpu_blueprint"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.npz"
TELEMETRY_PATH = DATA_DIR / "telemetry.json"
SAMPLER_INIT_PATH: Path | None = None

DEFAULT_CONFIG = GpuActionConfig(
    preflop_fractions=(0.75, 1.5),
    postflop_fractions=(0.5, 1.0),
    max_raises_per_street=3,
    stack_bb=100.0,  # matches the serving game: 2000 chips at a 20-chip big blind
)
# Native shallow-stack blueprint.  Menu sizing on 2026-07-29 measured 36,906
# nodes / 13,706 decision nodes / ~176 MiB of v3-sized CFR tables at 20bb, well
# below the documented training ceilings.  Shallow trees can afford this richer
# menu; using DEFAULT_CONFIG here would preserve the exact 100bb mismatch this
# blueprint is intended to remove.
BLUEPRINT_CONFIG_20 = GpuActionConfig(
    preflop_fractions=(0.5, 0.75),
    postflop_fractions=(0.33, 0.66, 1.0, 1.5),
    max_raises_per_street=2,
    stack_bb=20.0,
    no_donk_srp=True,
)
# House ruleset (2026-07-25 user directive): never limp preflop (raise or
# fold) + a wider sizing menu. Selected with --ruleset nolimp.
NO_LIMP_CONFIG = GpuActionConfig(
    preflop_fractions=(0.75, 1.0, 1.5),
    postflop_fractions=(0.33, 0.66, 1.0, 1.5),
    max_raises_per_street=3,
    stack_bb=100.0,
    no_limp=True,
)
# 200bb variant: full menu = 1.3M nodes (~40GB), impossible on the 3060;
# 310k nodes measured 12GB+spill (2026-07-25 overflow). Final shape uses the
# per-street cap: preflop cap 4 (sized opens/3bets/4bets/5bets, 6bet = jam,
# calls at every level) where the tree is cheap, postflop cap 2 with the wide
# 3-size menu where the blowup lives. 143k nodes ~ 6GB with real headroom.
NO_LIMP_CONFIG_200 = GpuActionConfig(
    preflop_fractions=(0.75, 1.5),
    postflop_fractions=(0.33, 0.75, 1.5),
    max_raises_per_street=2,
    stack_bb=200.0,
    no_limp=True,
    preflop_raise_cap=4,
)
# Phase 3: a broad, stable candidate-id pool with only two or three sized
# raises exposed at each structural state. The static-v1 selector varies the
# menu by street, position, SPR, pot, raise number, and facing-bet pressure;
# compiled local-EV overrides can replace individual structural classes.
NO_LIMP_PHASE3_CONFIG_200 = GpuActionConfig(
    preflop_fractions=(0.5, 0.75, 1.0, 1.5, 2.25),
    postflop_fractions=(0.25, 0.33, 0.5, 0.75, 1.0, 1.5),
    max_raises_per_street=2,
    stack_bb=200.0,
    no_limp=True,
    preflop_raise_cap=4,
    action_profile=PHASE3_STATIC_PROFILE,
    max_sized_raises_per_node=3,
)
# Distribution-aware buckets: 30 mean bins x 4 std (drawiness) bins = 120
# flop/turn buckets, separating draws from made hands and
# air at equal current equity. More runout samples sharpen the mean/std.
DEFAULT_SAMPLER = dict(
    flop_buckets=30,
    turn_buckets=30,
    river_buckets=30,
    flop_samples=12,
    turn_samples=12,
    distributional=True,
    std_bins=4,
)
# Histogram-EMD abstraction (Ganzfried & Sandholm 2014): full runout-equity
# histograms clustered by EMD k-means — separates draw TYPES (nut vs dominated)
# that mean+std collapses. Selected with --histogram; needs more runout samples
# for stable histogram shapes.
ACTIVE_SAMPLER = DEFAULT_SAMPLER  # swapped by --histogram
HISTOGRAM_SAMPLER = dict(
    flop_buckets=150,
    turn_buckets=150,
    river_buckets=30,
    flop_samples=24,
    turn_samples=16,
    histogram=True,
    hist_bins=10,
)
# Blueprint-v3 recursive potential-aware abstraction. Turn/river counts are
# doubled by two-bin prior-street recall, yielding (169, 384, 384, 192).
V3_SAMPLER = dict(
    flop_buckets=384,
    turn_buckets=192,
    river_buckets=96,
    potential_aware=True,
    recall_bins=2,
    flop_transition_samples=6,
    turn_transition_samples=12,
    flop_landmarks=24,
    turn_landmarks=16,
    potential_seed=20260725,
)
MILESTONE_ITERATIONS = frozenset((5_000, 10_000, 20_000))


def configure_stack(stack_bb: float, tag: str | None = None) -> None:
    """Retarget the trainer at a stack depth (and optional experiment tag).

    100bb + no tag keeps the canonical paths (the serving default). Other
    depths get gpu_blueprint_<N>bb/; a tag adds a _<tag> suffix so an
    experiment (e.g. a new abstraction) trains into its own directory without
    clobbering — or silently resuming from — an incompatible checkpoint.
    """
    global DEFAULT_CONFIG, DATA_DIR, CHECKPOINT_PATH, TELEMETRY_PATH
    DEFAULT_CONFIG = replace(DEFAULT_CONFIG, stack_bb=float(stack_bb))
    name = "gpu_blueprint" if stack_bb == 100.0 else f"gpu_blueprint_{int(stack_bb)}bb"
    if tag:
        name = f"{name}_{tag}"
    if name != "gpu_blueprint":
        DATA_DIR = DATA_DIR.parent / name
        CHECKPOINT_PATH = DATA_DIR / "checkpoint.npz"
        TELEMETRY_PATH = DATA_DIR / "telemetry.json"


def build_solver(device: str = "cuda", seed: int = 0, batch_boards: int = 1) -> VectorCFR:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    checkpoint = None
    sampler = DealSampler(**ACTIVE_SAMPLER)
    sampler_imported = False
    if CHECKPOINT_PATH.exists():
        with np.load(CHECKPOINT_PATH, allow_pickle=False) as payload:
            stored = json.loads(str(payload["config"]))
            # Older checkpoints predate optional no-limp/Phase 3 fields.
            # Reconstructing the dataclass fills their defaults before the
            # compatibility comparison, while still rejecting any genuinely
            # different betting tree.
            stored_config = GpuActionConfig(**stored)
            if asdict(stored_config) != asdict(DEFAULT_CONFIG):
                raise RuntimeError(
                    "gpu_blueprint checkpoint was trained with a different action config; "
                    "use a new --tag to start an incompatible experiment"
                )
            if "sampler" in payload:
                sampler = DealSampler.from_state(json.loads(str(payload["sampler"])))
            checkpoint = {
                "regrets": np.asarray(payload["regrets"]),
                "strategy_sums": np.asarray(payload["strategy_sums"]),
                "iteration": int(payload["iteration"]),
                "reach_normalized": "reach_normalized" in payload,
            }
    elif SAMPLER_INIT_PATH is not None:
        if not SAMPLER_INIT_PATH.exists():
            raise FileNotFoundError(f"sampler initialization checkpoint not found: {SAMPLER_INIT_PATH}")
        with np.load(SAMPLER_INIT_PATH, allow_pickle=False) as payload:
            if "sampler" not in payload:
                raise ValueError(f"sampler initialization checkpoint has no sampler state: {SAMPLER_INIT_PATH}")
            imported = DealSampler.from_state(json.loads(str(payload["sampler"])))
        if imported.bucket_counts() != sampler.bucket_counts():
            raise ValueError(
                "sampler initialization bucket counts do not match the selected abstraction: "
                f"{imported.bucket_counts()} != {sampler.bucket_counts()}"
            )
        if bool(imported.potential_aware) != bool(sampler.potential_aware):
            raise ValueError("sampler initialization uses a different abstraction family")
        sampler = imported
        sampler_imported = True
    tree = BettingTree(DEFAULT_CONFIG)
    # averaging_delay: the earliest strategies are noise; keep them out of the
    # average (Supremus' DCFR+ delayed averaging).
    solver = VectorCFR(
        tree,
        sampler,
        device=device,
        seed=seed,
        averaging_delay=1000,
        batch_boards=batch_boards,
    )
    storage = solver.storage_report()
    print(
        "compact CFR tables: "
        f"{storage['total_rows']:,} rows, "
        f"{storage['table_bytes_total'] / (1024**2):.1f} MiB, "
        f"{storage['row_reduction_fraction']:.1%} fewer rows than dense",
        flush=True,
    )
    if DEFAULT_CONFIG.action_profile == PHASE3_STATIC_PROFILE:
        description = tree.describe()
        print(
            "phase3 action profile: "
            f"{description['action_profile']} "
            f"sized_raise_nodes={description['sized_raise_nodes']} "
            f"overrides={len(DEFAULT_CONFIG.phase3_overrides)}",
            flush=True,
        )
    newly_fitted = False
    if checkpoint is not None:
        source = solver.load_tables(
            checkpoint["regrets"],
            checkpoint["strategy_sums"],
        )
        solver.iteration = checkpoint["iteration"]
        print(f"loaded {source} checkpoint at iteration {solver.iteration}", flush=True)
        if not checkpoint["reach_normalized"]:
            # One-time migration: reach is now probability-normalized, which
            # scales every future increment by ~1/1081; scale the stored
            # tensors by the same constant so past and future stay consistent.
            solver.regrets /= 1081.0
            solver.strategy_sums /= 1081.0
    elif (
        getattr(solver.sampler, "potential_aware", False)
        and solver.sampler._potential is not None
        and not solver.sampler._potential.fitted
    ):
        print("fitting recursive potential-aware v3 abstraction (one-time)...", flush=True)
        solver.sampler.fit_potential_abstraction(
            boards_per_street=12,
            seed=solver.sampler.potential_seed,
            iterations=12,
        )
        newly_fitted = True
    elif getattr(solver.sampler, "histogram", False) and not solver.sampler._hist_centroids:
        # Fresh histogram run: fit the EMD k-means centroids once (~minutes);
        # they persist in the first checkpoint so serving buckets identically.
        print("fitting histogram-EMD centroids (one-time)...")
        solver.sampler.fit_hist_centroids(boards=150, seed=seed)
        newly_fitted = True
    elif solver.sampler.distributional and not solver.sampler._std_edges:
        # Fresh run: fit the drawiness (equity-std) quantile edges once; they
        # persist in the first checkpoint so serving buckets identically.
        solver.sampler.fit_std_edges(samples=400, seed=seed)
        newly_fitted = True
    if sampler_imported:
        fitted = (
            solver.sampler._potential is not None
            and solver.sampler._potential.fitted
        )
        if solver.sampler.potential_aware and not fitted:
            raise ValueError("sampler initialization checkpoint contains an unfitted v3 abstraction")
        print(f"imported fitted sampler from {SAMPLER_INIT_PATH}", flush=True)
    if newly_fitted or sampler_imported:
        # Persist expensive abstraction fitting before the first CFR chunk.
        save_solver(solver)
    return solver


def save_solver(solver: VectorCFR) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        regrets=solver.regrets.cpu().numpy(),
        strategy_sums=solver.strategy_sums.cpu().numpy(),
        iteration=solver.iteration,
        config=json.dumps(asdict(DEFAULT_CONFIG)),
        sampler=json.dumps(solver.sampler.state()),
        storage=json.dumps(solver.storage_report()),
        reach_normalized=True,
    )
    temporary.replace(CHECKPOINT_PATH)
    # Keep sparse history so convergence trends can be probed retroactively.
    if solver.iteration in MILESTONE_ITERATIONS or solver.iteration % 20_000 == 0:
        import shutil

        shutil.copy2(CHECKPOINT_PATH, DATA_DIR / f"checkpoint-{solver.iteration}.npz")


def train(
    iterations: int,
    device: str = "cuda",
    save_every: int = 200,
    seed: int = 0,
    progress: bool = True,
    batch_boards: int = 1,
) -> VectorCFR:
    solver = build_solver(device=device, seed=seed, batch_boards=batch_boards)
    completed = 0
    started = time.time()
    while completed < iterations:
        chunk = min(save_every, iterations - completed)
        chunk_started = time.time()
        solver.run(chunk)
        completed += chunk
        save_solver(solver)
        rate = chunk / (time.time() - chunk_started)
        # The in-loop CFR-BR exploitability probe was removed 2026-07-23: the
        # (NULL-tested) head-to-head duel gate is the real promotion/plateau
        # signal, the probe's readings were noisy-to-misleading (negative on
        # converged strategies), it cost 2-3 minutes per probe, and building
        # its responder on the big tree ratcheted the CUDA caching pool (VRAM
        # incidents). For on-demand diagnostics use
        # backend.solver.gpu.exploit.cfr_br_exploitability directly.
        record = {
            "iteration": solver.iteration,
            "iterations_per_second": round(rate, 3),
            "device": str(solver.device),
            "storage": solver.storage_report(),
            "action_profile": {
                "name": solver.tree.config.action_profile,
                "profile_sha256": solver.tree.config.phase3_profile_sha256,
                "override_count": len(solver.tree.config.phase3_overrides),
                "sized_raise_nodes": solver.tree.describe()["sized_raise_nodes"],
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        history = json.loads(TELEMETRY_PATH.read_text(encoding="utf-8")) if TELEMETRY_PATH.exists() else []
        history.append(record)
        TELEMETRY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if progress:
            print(f"iter {solver.iteration} | {rate:.2f}/s | elapsed {time.time() - started:.0f}s")
    return solver


def main() -> None:
    global ACTIVE_SAMPLER, DEFAULT_CONFIG, SAMPLER_INIT_PATH

    parser = argparse.ArgumentParser(description="Train the compact GPU blueprint")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-boards", type=int, default=1, help="boards per iteration (mini-batch)")
    parser.add_argument("--stack-bb", type=float, default=100.0, help="stack depth; non-100 gets its own artifact dir")
    parser.add_argument("--tag", type=str, default=None, help="experiment tag; isolates artifacts (e.g. a new abstraction)")
    parser.add_argument("--histogram", action="store_true", help="histogram-EMD abstraction (fresh runs; use with --tag)")
    parser.add_argument(
        "--abstraction",
        choices=["legacy", "histogram", "v3"],
        default="legacy",
        help="card abstraction; v3 enables recursive potential-aware clustering",
    )
    parser.add_argument(
        "--phase3-actions",
        action="store_true",
        help="state-dependent two/three-size Phase 3 action abstraction (200bb no-limp)",
    )
    parser.add_argument(
        "--action-profile",
        type=Path,
        default=None,
        help="compiled Phase 3 local-EV profile JSON; rules are embedded in the checkpoint",
    )
    parser.add_argument(
        "--sampler-init",
        type=Path,
        default=None,
        help="reuse only a fitted card sampler from another checkpoint; CFR tables remain zero",
    )
    parser.add_argument("--ruleset", type=str, default=None, choices=["nolimp"], help="house ruleset (never-limp + wide sizing menu)")
    arguments = parser.parse_args()
    if arguments.histogram and arguments.abstraction not in ("legacy", "histogram"):
        parser.error("--histogram cannot be combined with --abstraction v3")
    if arguments.histogram or arguments.abstraction == "histogram":
        ACTIVE_SAMPLER = HISTOGRAM_SAMPLER
    elif arguments.abstraction == "v3":
        ACTIVE_SAMPLER = V3_SAMPLER
    if arguments.phase3_actions and not (
        arguments.ruleset == "nolimp" and arguments.stack_bb == 200.0
    ):
        parser.error("--phase3-actions currently requires --ruleset nolimp --stack-bb 200")
    if arguments.action_profile is not None and not arguments.phase3_actions:
        parser.error("--action-profile requires --phase3-actions")
    if arguments.ruleset == "nolimp":
        if arguments.phase3_actions:
            DEFAULT_CONFIG = NO_LIMP_PHASE3_CONFIG_200
        else:
            DEFAULT_CONFIG = NO_LIMP_CONFIG_200 if arguments.stack_bb == 200.0 else NO_LIMP_CONFIG
    elif arguments.stack_bb == 20.0:
        DEFAULT_CONFIG = BLUEPRINT_CONFIG_20
    if arguments.action_profile is not None:
        overrides, profile_sha256 = load_compiled_profile(arguments.action_profile.resolve())
        DEFAULT_CONFIG = replace(
            DEFAULT_CONFIG,
            phase3_overrides=overrides,
            phase3_profile_sha256=profile_sha256,
        )
    tag = arguments.tag
    if arguments.phase3_actions and not tag:
        prefix = "v3_" if arguments.abstraction == "v3" else ""
        tag = f"{prefix}phase3_nolimp"
    elif arguments.abstraction == "v3" and not tag:
        tag = "v3_nolimp" if arguments.ruleset == "nolimp" else "v3"
    if arguments.stack_bb != 100.0 or tag:
        configure_stack(arguments.stack_bb, tag=tag)
    if arguments.sampler_init is not None:
        SAMPLER_INIT_PATH = arguments.sampler_init.resolve()
    elif arguments.phase3_actions and arguments.abstraction == "v3":
        base_name = (
            "gpu_blueprint"
            if arguments.stack_bb == 100.0
            else f"gpu_blueprint_{int(arguments.stack_bb)}bb"
        )
        previous_v3 = DATA_DIR.parent / f"{base_name}_v3_nolimp" / "checkpoint.npz"
        if previous_v3.exists():
            SAMPLER_INIT_PATH = previous_v3
    train(
        arguments.iterations,
        device=arguments.device,
        save_every=arguments.save_every,
        seed=arguments.seed,
        batch_boards=arguments.batch_boards,
    )


if __name__ == "__main__":
    main()
