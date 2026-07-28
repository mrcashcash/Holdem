"""Shared serving-agent selection for the API and screen decision pipeline."""

from __future__ import annotations

import os

from backend.agents.blueprint_agent import BlueprintAgent
from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
from backend.agents.multistack_agent import MultiStackBlueprintAgent
from backend.styles import HeuristicAgent


GPU_SERVE_MIN_ITERATIONS = 5_000


def load_serving_agent():
    """Load the same preferred brain without importing the FastAPI application."""

    subgame_iterations = int(os.environ.get("HOLDEM_SUBGAME_ITERS", "120"))

    def apply_search(agent):
        if agent is None:
            return None
        # Phase 4 replaces, rather than layers on top of, the retired bucketed
        # turn/river resolver. This keeps its on/off evaluation isolated.
        agent.subgame_search = (
            subgame_iterations > 0
            and not getattr(agent, "exact_river_search", False)
        )
        agent.subgame_iterations = subgame_iterations or agent.subgame_iterations
        return agent

    router = MultiStackBlueprintAgent.try_load()
    if (
        router is not None
        and len(router.agents) >= 2
        and router.iteration >= GPU_SERVE_MIN_ITERATIONS
    ):
        return apply_search(router)

    gpu_agent = apply_search(GpuBlueprintAgent.try_load())
    if gpu_agent is not None and gpu_agent.iteration >= GPU_SERVE_MIN_ITERATIONS:
        return gpu_agent
    cpu_agent = BlueprintAgent.try_load()
    if cpu_agent is not None:
        return cpu_agent
    if gpu_agent is not None:
        return gpu_agent
    return HeuristicAgent()
