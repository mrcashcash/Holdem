"""CUDA-graph capture for VectorCFR iterations (docs/GPU_CFR_PLAN.md follow-up).

Small subgame trees are launch-overhead-bound: a ~1,100-node turn tree spends
most of each iteration dispatching thousands of tiny kernels (~30us each on
Windows WDDM). The level plans and tensor shapes are static, so the whole
[iterate(traverser=0), iterate(traverser=1), discount] sequence can be
captured once as a CUDA graph and replayed per iteration with fresh deal data
copied into fixed input buffers — bit-identical math, an order of magnitude
fewer launches.

Excluded from the graph: the every-500-iteration float32 rescale guard (it
needs host syncs) — the runner executes it eagerly at the same cadence.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.solver.gpu.cfr import NUM_COMBOS, VectorCFR
from backend.solver.gpu.deals import Deal


class GraphRunner:
    """Capture-once / replay-many driver for a fixed-shape VectorCFR."""

    def __init__(self, solver: VectorCFR, warmup: int = 3) -> None:
        if solver.device.type != "cuda":
            raise ValueError("CUDA graphs require a cuda solver")
        self.solver = solver
        batch = solver.batch_boards
        device = solver.device
        width = batch * NUM_COMBOS

        # Static input buffers the graph reads from.
        self.valid_buf = torch.zeros(width, dtype=torch.bool, device=device)
        self.buckets_buf = torch.zeros((4, width), dtype=torch.long, device=device)
        self.scores_buf = torch.zeros((batch, NUM_COMBOS), dtype=torch.long, device=device)
        self.positive_factor = torch.ones((), dtype=torch.float32, device=device)
        self.negative_factor = torch.ones((), dtype=torch.float32, device=device)
        self.strategy_factor = torch.ones((), dtype=torch.float32, device=device)

        self._install_buffered_iterate()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._captured_body()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._captured_body()

    # -- the captured computation -------------------------------------------

    def _install_buffered_iterate(self) -> None:
        """Point _iterate's tensor construction at the static buffers."""
        solver = self.solver

        def buffered_inputs(_deals):
            return self.valid_buf, self.buckets_buf, self.scores_buf

        solver._graph_inputs = buffered_inputs  # consumed by _iterate

    def _captured_body(self) -> None:
        solver = self.solver
        sentinel = _BufferDeal(solver.batch_boards)
        solver._iterate(sentinel, traverser=0)
        solver._iterate(sentinel, traverser=1)
        # Discount with device-resident factors — strictly IN-PLACE so the
        # regret tensor keeps its identity across graph replays (reassigning
        # would leave subsequent replays writing into a stale buffer).
        solver.regrets.mul_(
            torch.where(solver.regrets > 0, self.positive_factor, self.negative_factor)
        )
        solver.strategy_sums.mul_(self.strategy_factor)

    # -- per-iteration replay --------------------------------------------------

    def run(self, iterations: int, rng) -> None:
        solver = self.solver
        for _ in range(iterations):
            solver.iteration += 1
            deals = [solver.sampler.sample(rng) for _ in range(solver.batch_boards)]
            self._fill_buffers(deals)
            t = float(solver.iteration)
            self.positive_factor.fill_(t**solver.discount_alpha / (t**solver.discount_alpha + 1.0))
            self.negative_factor.fill_(t**solver.discount_beta / (t**solver.discount_beta + 1.0))
            if solver.iteration > solver.averaging_delay:
                self.strategy_factor.fill_((t / (t + 1.0)) ** solver.discount_gamma)
            else:
                self.strategy_factor.fill_(0.0)
            self.graph.replay()
            # The float32 headroom guard needs host syncs — run it eagerly.
            if solver.iteration % 500 == 0:
                peak = float(solver.regrets.abs().max().item())
                if peak > 1e7:
                    solver.regrets *= 1e6 / peak
                sums_peak = float(solver.strategy_sums.abs().max().item())
                if sums_peak > 1e7:
                    solver.strategy_sums *= 1e6 / sums_peak

    def _fill_buffers(self, deals: list[Deal]) -> None:
        valid = np.concatenate([d.valid for d in deals])
        buckets = np.concatenate([d.buckets for d in deals], axis=1)
        scores = np.stack([d.river_scores for d in deals])
        self.valid_buf.copy_(torch.from_numpy(valid), non_blocking=True)
        self.buckets_buf.copy_(torch.from_numpy(buckets).clamp_min(0), non_blocking=True)
        self.scores_buf.copy_(torch.from_numpy(scores), non_blocking=True)


class _BufferDeal:
    """Sentinel telling _iterate to read the graph runner's buffers."""

    __slots__ = ("batch",)

    def __init__(self, batch: int) -> None:
        self.batch = batch
