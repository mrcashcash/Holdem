"""Shared serving-agent selection for the API and screen decision pipeline.

The serving configuration is the best one the measurements support, and every
choice below is tied to a number in docs/PLAN_V2_STRONGEST_PLAYER.md.

**Bucketed subgame search is OFF.** It used to default ON here
(`HOLDEM_SUBGAME_ITERS=120`), which quietly served the one search variant that
was retired for measuring WORSE than the blueprint: -31 bb/100 [-95, +33] at 120
iterations and -86 [-150, -22] at 500. Re-solving at the blueprint's own
150/30-bucket resolution cannot beat it — there is no information edge — so more
iterations converge harder onto a less-informed problem.

**Exact-card continual resolving is ON.** Every solve uses one bucket per private
combo, which is a genuine information edge over the blueprint's 150 turn / 30
river buckets. Evidence:

* it demonstrably fixes a documented leak class — the 7s5s draw on 4s5c6sKc folds
  83% under the blueprint and 3.2% when solved exactly, while trash still folds
  0.92-0.99, so it is discrimination rather than looseness;
* reliability is 1,244/1,244 resolves with zero fallbacks across three stack
  depths and both entry streets;
* its on/off duel is INCONCLUSIVE, not negative: +17.82, +54.45 and +28.12 bb/100
  at 60/240/240(CRN) iterations, all with intervals spanning zero. Nothing has
  ever measured it as harmful, and the convergence ladder shows 60 iterations sits
  L1 0.418 from converged, so the arms tested were handicapped.

**The river value net is OFF.** It beats a zero-predictor (ratio 0.301 with
strength-ordered inputs) but FAILS its acceptance gate: only 38% action agreement
with a full solve and policy L1 1.15 against ~0.21 solver noise. Serving it would
repeat CFV v0's mistake of shipping a net on loss rather than on decisions.

**The ALL-IN geometry guard is OFF.** A live hand on 2026-07-29 jammed 3,980 chips
into a 166-chip pot (24x) because ALL-IN was the only size-bearing action executed
literally, while `_locate` matches nodes by action sequence and never compares
pot/stack geometry. `GpuBlueprintAgent._all_in_size` repairs that translation and
cuts the worst jam from 15.4x to 4.6x pot -- but the only A/B taken says it costs
-268.82 bb/100 at 200bb and -124.00 at 100bb against the min-raiser whose lines
trigger it, and is neutral otherwise. Against a station that calls any jam, shoving
199bb is correct exploitation rather than a bug, so the guard is available per agent
(`all_in_geometry_guard`) and not served. The structural fix is the resolver, which
puts 0.16% on all-in at that spot because its tree uses the REAL geometry.

Iteration count is the quality/latency dial. `HOLDEM_CONTINUAL_ITERS=120` is the
default and 60 is the serving quality floor. Actual latency remains
geometry-dependent and is recorded per decision; resource admission prevents a
large flop from allocating first and discovering the limit only by timing out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend.agents.blueprint_agent import BlueprintAgent
from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.agents.multistack_agent import MultiStackBlueprintAgent
from backend.search.resources import ResolverResourceLimits
from backend.styles import HeuristicAgent

GPU_SERVE_MIN_ITERATIONS = 5_000

#: Exact-card resolving iterations per decision.
DEFAULT_CONTINUAL_ITERS = 120
#: Do not silently trade away most of the resolver's quality to make an
#: unrealistic latency target. The measured 30-iteration policy was still far
#: from converged; callers can explicitly lower this floor for experiments.
DEFAULT_CONTINUAL_MIN_ITERS = 60
#: Hard wall-clock ceiling per decision; on expiry the agent fails closed to the
#: frozen blueprint for the REST OF THE HAND rather than serving a partial solve.
#: Generous on purpose: one blown deadline poisons every later street.
DEFAULT_CONTINUAL_BUDGET_MS = 45_000

#: All postflop streets are eligible. Flop trees use a richest-safe menu ladder
#: and explicit node/VRAM admission; turn and river retain exact-card solving to
#: showdown. The optional neural river horizon remains gated off until its
#: action-quality report passes.
DEFAULT_RESOLVE_STREETS = "flop,turn,river"
_STREET_IDS = {"flop": 1, "turn": 2, "river": 3}


def river_net_gate_status() -> dict:
    """Whether the optional river-horizon network is safe to serve.

    A low training loss is insufficient. Only an action-agreement gate can
    promote the network into the turn resolver.
    """

    checkpoint = Path(
        os.environ.get(
            "HOLDEM_RIVER_NET_PATH",
            "backend/data/cfv/river_net/river_net.pt",
        )
    )
    gate_path = Path(
        os.environ.get(
            "HOLDEM_RIVER_NET_GATE",
            "backend/data/cfv/river_net/gate.json",
        )
    )
    minimum_agreement = float(
        os.environ.get("HOLDEM_RIVER_NET_MIN_AGREEMENT", "0.90")
    )
    maximum_l1 = float(
        os.environ.get("HOLDEM_RIVER_NET_MAX_POLICY_L1", "0.30")
    )
    report = None
    error = None
    try:
        report = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        error = str(exc)
    agreement = (
        float(report.get("action_agreement_mean", 0.0))
        if isinstance(report, dict)
        else None
    )
    policy_l1 = (
        float(report["policy_l1_mean"])
        if isinstance(report, dict) and "policy_l1_mean" in report
        else None
    )
    eligible = bool(
        checkpoint.exists()
        and report
        and report.get("gate") == "river-net-acceptance"
        and agreement is not None
        and agreement >= minimum_agreement
        and policy_l1 is not None
        and policy_l1 <= maximum_l1
    )
    if error is None and not checkpoint.exists():
        error = f"checkpoint not found: {checkpoint}"
    if error is None and not eligible:
        error = (
            f"gate failed: agreement {agreement!r} needs >= {minimum_agreement:.2f}; "
            f"policy L1 {policy_l1!r} needs <= {maximum_l1:.2f}"
        )
    return {
        "eligible": eligible,
        "reason": None if eligible else error,
        "checkpoint": str(checkpoint),
        "gate_report": str(gate_path),
        "action_agreement_mean": agreement,
        "policy_l1_mean": policy_l1,
        "minimum_agreement": minimum_agreement,
        "maximum_policy_l1": maximum_l1,
    }


def serving_configuration() -> dict:
    """The resolved serving knobs, so the API can report exactly what is playing."""
    names = [
        name.strip().lower()
        for name in os.environ.get("HOLDEM_RESOLVE_STREETS", DEFAULT_RESOLVE_STREETS).split(",")
        if name.strip()
    ]
    streets = sorted({_STREET_IDS[name] for name in names if name in _STREET_IDS})
    minimum_iterations = max(
        12,
        int(
            os.environ.get(
                "HOLDEM_CONTINUAL_MIN_ITERS",
                DEFAULT_CONTINUAL_MIN_ITERS,
            )
        ),
    )
    requested_iterations = int(
        os.environ.get("HOLDEM_CONTINUAL_ITERS", DEFAULT_CONTINUAL_ITERS)
    )
    river_net_requested = os.environ.get("HOLDEM_RIVER_NET", "0") == "1"
    river_gate = river_net_gate_status()
    limits = ResolverResourceLimits.from_env()
    return {
        "continual_search": os.environ.get("HOLDEM_CONTINUAL", "1") != "0",
        "resolve_streets": [
            label for label, value in sorted(_STREET_IDS.items(), key=lambda kv: kv[1])
            if value in streets
        ],
        "resolve_street_ids": streets,
        "continual_iterations": max(minimum_iterations, requested_iterations),
        "continual_min_iterations": minimum_iterations,
        "continual_budget_ms": max(1, int(os.environ.get("HOLDEM_CONTINUAL_BUDGET_MS", DEFAULT_CONTINUAL_BUDGET_MS))),
        # Opt-in only, and both default off for the reasons in the module docstring.
        "bucketed_subgame_search": int(os.environ.get("HOLDEM_SUBGAME_ITERS", "0")) > 0,
        "river_net_requested": river_net_requested,
        "river_net": river_net_requested and river_gate["eligible"],
        "river_net_gate": river_gate,
        "resolver_resources": {
            "max_vram_mib": limits.physical_budget_bytes // (1024**2),
            "required_headroom_mib": (
                limits.required_free_headroom_bytes // (1024**2)
            ),
            "flop_node_budget": limits.flop_node_budget,
            "showdown_workspace_mib": max(
                32,
                int(os.environ.get("HOLDEM_SHOWDOWN_WORKSPACE_MB", "384")),
            ),
        },
        "resolver_optimizations": {
            "safety_price_cuda_graph": (
                os.environ.get("HOLDEM_SAFETY_PRICE_GRAPH", "1") != "0"
            ),
            "deal_prefetch": (
                os.environ.get("HOLDEM_RESOLVER_PREFETCH", "1") != "0"
            ),
            "session_runout_cache": (
                os.environ.get("HOLDEM_SESSION_RUNOUT_CACHE", "1") != "0"
            ),
            "startup_warmup": (
                os.environ.get("HOLDEM_RESOLVER_WARMUP", "1") != "0"
            ),
        },
    }


def load_serving_agent():
    """Load the strongest configuration the measurements support."""
    config = serving_configuration()

    river_net = None
    river_net_error = None
    if config["river_net"]:
        try:
            import torch

            from backend.cfv.river_net import RiverCfvNet

            path = Path(config["river_net_gate"]["checkpoint"])
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # older PyTorch
                payload = torch.load(path, map_location="cpu")
            river_net = RiverCfvNet(
                hidden=int(payload.get("hidden", 500)),
                layers=int(payload.get("layers", 7)),
            )
            river_net.load_state_dict(payload["state_dict"])
            river_net.eval()
        except Exception as exc:
            river_net_error = str(exc)
            config["river_net"] = False
            config["river_net_gate"] = {
                **config["river_net_gate"],
                "eligible": False,
                "reason": f"eligible checkpoint could not be loaded: {exc}",
            }

    def apply(agent):
        if agent is None:
            return None
        targets = [agent, *getattr(agent, "agents", {}).values()]
        for target in targets:
            if not hasattr(target, "continual_search"):
                continue
            target.continual_search = config["continual_search"]
            target.continual_streets = tuple(config["resolve_street_ids"])
            target.continual_iterations = config["continual_iterations"]
            target.continual_budget_ms = config["continual_budget_ms"]
            # A failed or missing action-agreement gate can never be bypassed
            # merely by setting HOLDEM_RIVER_NET=1.
            target.resolver_river_net = river_net
            target.resolver_river_net_error = river_net_error
            # Exact-card resolving supersedes both older search paths; never
            # layer them, or the retired bucketed resolver silently takes the
            # turn decisions.
            target.exact_river_search = False
            target.subgame_search = (
                config["bucketed_subgame_search"] and not config["continual_search"]
            )
            if target.subgame_search:
                target.subgame_iterations = int(os.environ["HOLDEM_SUBGAME_ITERS"])
        return agent

    router = MultiStackBlueprintAgent.try_load()
    if (
        router is not None
        and len(router.agents) >= 2
        and router.iteration >= GPU_SERVE_MIN_ITERATIONS
    ):
        return apply(router)

    gpu_agent = apply(GpuBlueprintAgent.try_load())
    if gpu_agent is not None and gpu_agent.iteration >= GPU_SERVE_MIN_ITERATIONS:
        return gpu_agent
    cpu_agent = BlueprintAgent.try_load()
    if cpu_agent is not None:
        return cpu_agent
    if gpu_agent is not None:
        return gpu_agent
    return HeuristicAgent()
