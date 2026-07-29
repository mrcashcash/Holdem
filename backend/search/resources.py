"""Resource planning for exact-card real-time resolves.

The old flop admission rule used only a public-node ceiling.  That is not a
VRAM ceiling: exact-card table rows scale with decision nodes, traversal
workspaces scale with all nodes, and CUDA graph capture needs a second private
workspace.  This module makes the estimate explicit and compares it with both
the card's physical capacity and the memory currently free before a solver is
constructed.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from backend.solver.gpu.deals import NUM_COMBOS
from backend.solver.gpu.storage import CompactTableLayout
from backend.solver.gpu.tree import DECISION, SHOWDOWN, BettingTree

MIB = 1024**2


@dataclass(frozen=True)
class ResolverResourceLimits:
    """Serving limits, intentionally conservative on a display-attached GPU."""

    physical_budget_bytes: int
    required_free_headroom_bytes: int
    flop_node_budget: int

    @classmethod
    def from_env(cls) -> "ResolverResourceLimits":
        return cls(
            physical_budget_bytes=max(
                1024,
                int(os.environ.get("HOLDEM_RESOLVER_MAX_VRAM_MB", "9500")),
            )
            * MIB,
            required_free_headroom_bytes=max(
                512,
                int(os.environ.get("HOLDEM_RESOLVER_VRAM_HEADROOM_MB", "2048")),
            )
            * MIB,
            # A second, independent guard.  It bounds latency as well as memory
            # and keeps the 20bb exact path while rejecting the 40k-60k node
            # deep-stack flops responsible for the live overflow.
            flop_node_budget=max(
                500,
                int(os.environ.get("HOLDEM_FLOP_NODE_BUDGET", "12000")),
            ),
        )


@dataclass(frozen=True)
class SolverResourceEstimate:
    nodes: int
    decision_nodes: int
    showdown_nodes: int
    actions: int
    compact_rows: int
    table_bytes: int
    frozen_policy_bytes: int
    static_bytes: int
    traversal_bytes: int
    showdown_workspace_bytes: int
    graph_workspace_bytes: int
    output_bytes: int
    estimated_peak_bytes: int

    def diagnostics(self) -> dict:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if key.endswith("_bytes"):
                result[key.replace("_bytes", "_mib")] = round(value / MIB, 1)
                del result[key]
        return result


@dataclass(frozen=True)
class ResourceDecision:
    allowed: bool
    reason: str | None
    estimate: SolverResourceEstimate
    free_bytes: int | None
    total_bytes: int | None
    current_allocated_bytes: int
    current_reserved_bytes: int
    limits: ResolverResourceLimits

    def diagnostics(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "free_mib": (
                round(self.free_bytes / MIB, 1) if self.free_bytes is not None else None
            ),
            "total_mib": (
                round(self.total_bytes / MIB, 1) if self.total_bytes is not None else None
            ),
            "current_allocated_mib": round(self.current_allocated_bytes / MIB, 1),
            "current_reserved_mib": round(self.current_reserved_bytes / MIB, 1),
            "physical_budget_mib": round(self.limits.physical_budget_bytes / MIB, 1),
            "required_headroom_mib": round(
                self.limits.required_free_headroom_bytes / MIB, 1
            ),
            "flop_node_budget": self.limits.flop_node_budget,
            "estimate": self.estimate.diagnostics(),
        }


class ResolverResourceError(RuntimeError):
    """Raised before CUDA allocation when a proposed solve cannot fit safely."""

    def __init__(self, decision: ResourceDecision) -> None:
        self.resource_diagnostics = decision.diagnostics()
        super().__init__(decision.reason or "resolver resource admission failed")


def _max_level_decisions(tree: BettingTree) -> int:
    depth = np.zeros(len(tree), dtype=np.int32)
    for node in range(len(tree)):
        for child in tree.children[node]:
            if child >= 0:
                depth[int(child)] = depth[node] + 1
    decisions = np.flatnonzero(tree.kind == DECISION)
    if not decisions.size:
        return 0
    counts = np.bincount(depth[decisions])
    return int(counts.max(initial=0))


def estimate_exact_solver(
    tree: BettingTree,
    bucket_counts: tuple[int, int, int, int],
    *,
    graph_capture: bool = True,
    compact_frozen_policy: bool = True,
    root_only_output: bool = True,
) -> SolverResourceEstimate:
    """Conservative peak estimate for a batch-one exact-card solve.

    The estimate intentionally overstates reusable temporaries.  A false
    rejection falls back to the frozen blueprint; a false admission can push a
    WDDM card into shared memory and stall a live decision for minutes.
    """

    nodes = len(tree)
    actions = tree.config.num_actions
    decisions = int((tree.kind == DECISION).sum())
    showdowns = int((tree.kind == SHOWDOWN).sum())
    layout = CompactTableLayout(tree, bucket_counts)
    compact_rows = int(layout.total_rows)

    table_bytes = compact_rows * actions * 4 * 2
    frozen_policy_bytes = (
        compact_rows * actions * 4
        if compact_frozen_policy
        else nodes * max(bucket_counts) * actions * 4
    )

    # Tree tensors, layout maps, card masks and graph input buffers.
    static_bytes = (
        nodes * (actions * (8 + 1) + 8 * 8)
        + 4 * NUM_COMBOS * 8
        + 52 * NUM_COMBOS * 4
    )

    width = NUM_COMBOS
    # reach[2,N,C] + values[N,C], plus the largest level's strategy and
    # child-value tensors.  Fused-forward edge intermediates are covered by a
    # further 25% margin.
    max_level = _max_level_decisions(tree)
    core = 3 * nodes * width * 4
    level = 2 * max_level * width * actions * 4
    traversal_bytes = int((core + level) * 1.25)

    # Showdown blocker correction is tiled by card channel.  Include the
    # configured tile ceiling plus the persistent worse/better/correction
    # tensors.  This matches the bounded implementation in gpu/cfr.py instead
    # of assuming that all 52 channels coexist.
    showdown_cap = max(
        32,
        int(os.environ.get("HOLDEM_SHOWDOWN_WORKSPACE_MB", "384")),
    ) * MIB
    full_channels = showdowns * width * 52 * 4 * 6
    persistent_showdown = showdowns * width * 4 * 4
    showdown_workspace_bytes = min(full_channels, showdown_cap) + persistent_showdown

    iteration_workspace = traversal_bytes + showdown_workspace_bytes
    graph_workspace_bytes = iteration_workspace if graph_capture else 0
    output_bytes = (
        NUM_COMBOS * actions * 4
        if root_only_output
        else nodes * max(bucket_counts) * actions * 4
    )
    # Ten percent covers allocator bins, CUDA graph metadata, and small
    # temporaries not worth modelling individually.
    estimated_peak = int(
        (
            table_bytes
            + frozen_policy_bytes
            + static_bytes
            + iteration_workspace
            + graph_workspace_bytes
            + output_bytes
        )
        * 1.10
    )
    return SolverResourceEstimate(
        nodes=nodes,
        decision_nodes=decisions,
        showdown_nodes=showdowns,
        actions=actions,
        compact_rows=compact_rows,
        table_bytes=table_bytes,
        frozen_policy_bytes=frozen_policy_bytes,
        static_bytes=static_bytes,
        traversal_bytes=traversal_bytes,
        showdown_workspace_bytes=showdown_workspace_bytes,
        graph_workspace_bytes=graph_workspace_bytes,
        output_bytes=output_bytes,
        estimated_peak_bytes=estimated_peak,
    )


def decide_exact_solver(
    tree: BettingTree,
    bucket_counts: tuple[int, int, int, int],
    *,
    street: int,
    device: str,
    graph_capture: bool = True,
    limits: ResolverResourceLimits | None = None,
) -> ResourceDecision:
    limits = limits or ResolverResourceLimits.from_env()
    estimate = estimate_exact_solver(
        tree,
        bucket_counts,
        graph_capture=graph_capture,
        compact_frozen_policy=True,
        root_only_output=True,
    )

    free_bytes: int | None = None
    total_bytes: int | None = None
    current_allocated_bytes = 0
    current_reserved_bytes = 0
    if device == "cuda" and torch.cuda.is_available():
        # Release reusable blocks from earlier decisions before measuring.  On
        # WDDM, cudaMemGetInfo can overstate globally free physical VRAM, so the
        # per-process allocated+estimate budget below is the primary guard.
        torch.cuda.empty_cache()
        current_allocated_bytes = int(torch.cuda.memory_allocated())
        current_reserved_bytes = int(torch.cuda.memory_reserved())
        free_bytes, total_bytes = (int(value) for value in torch.cuda.mem_get_info())

    reason = None
    if street == 1 and estimate.nodes > limits.flop_node_budget:
        reason = (
            f"flop tree has {estimate.nodes:,} nodes; serving budget is "
            f"{limits.flop_node_budget:,}"
        )
    elif (
        current_allocated_bytes + estimate.estimated_peak_bytes
        > limits.physical_budget_bytes
    ):
        reason = (
            f"current allocations plus estimated peak "
            f"{(current_allocated_bytes + estimate.estimated_peak_bytes) / MIB:.0f} "
            f"MiB exceed the {limits.physical_budget_bytes / MIB:.0f} MiB "
            f"resolver budget"
        )
    elif free_bytes is not None:
        usable = max(free_bytes - limits.required_free_headroom_bytes, 0)
        if estimate.estimated_peak_bytes > usable:
            reason = (
                f"estimated peak {estimate.estimated_peak_bytes / MIB:.0f} MiB exceeds "
                f"currently usable VRAM {usable / MIB:.0f} MiB"
            )

    return ResourceDecision(
        allowed=reason is None,
        reason=reason,
        estimate=estimate,
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        current_allocated_bytes=current_allocated_bytes,
        current_reserved_bytes=current_reserved_bytes,
        limits=limits,
    )
