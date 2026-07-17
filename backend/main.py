"""FastAPI application serving the game and the blueprint trainer.

The table is served by the MCCFR blueprint agent (backend/agents) whenever
its artifacts exist under backend/data/blueprint/; a simple heuristic agent
fills in before the first blueprint checkpoint is written. The training
endpoints drive the Linear MCCFR blueprint trainer (backend/solver) — the
legacy PPO league trainer was retired to legacy/ (docs/REDESIGN_PLAN.md).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Thread, current_thread
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agents import BlueprintAgent
from .poker import HeadsUpHoldem, InvalidAction
from .solver import blueprint as blueprint_trainer
from .styles import HeuristicAgent

app = FastAPI(title="Text Hold'em API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEBUG_LOG_PATH = Path(__file__).resolve().parent / "data" / "server-debug.jsonl"


def log_debug(event: str, **fields) -> None:
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


# The GPU blueprint overtakes the CPU one quickly (600 GPU iterations matched
# 12k CPU MCCFR iterations on the styles benchmark), and it is the only
# blueprint trained at the serving game's 100 bb depth — the frozen CPU table
# is a 50 bb model. Prefer the GPU checkpoint as soon as it has meaningful
# training.
GPU_SERVE_MIN_ITERATIONS = 5_000


def load_serving_agent():
    import os

    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent

    gpu_agent = GpuBlueprintAgent.try_load()
    if gpu_agent is not None:
        # Turn/river re-solving costs ~20-30s per decision until the CUDA-graph
        # optimization lands — too slow for live play, so it is off at the
        # table unless explicitly enabled (eval CLIs enable it themselves).
        subgame_iterations = int(os.environ.get("HOLDEM_SUBGAME_ITERS", "0"))
        gpu_agent.subgame_search = subgame_iterations > 0
        gpu_agent.subgame_iterations = subgame_iterations or gpu_agent.subgame_iterations
    if gpu_agent is not None and gpu_agent.iteration >= GPU_SERVE_MIN_ITERATIONS:
        return gpu_agent
    cpu_agent = BlueprintAgent.try_load()
    if cpu_agent is not None:
        return cpu_agent
    if gpu_agent is not None:
        return gpu_agent
    return HeuristicAgent()


game = HeadsUpHoldem()
serving_agent = load_serving_agent()
game_lock = RLock()
log_debug("serving_agent_selected", kind=type(serving_agent).__name__)


class TrainingState:
    """Status of the in-process blueprint training worker."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.worker: Thread | None = None
        self.running = False
        self.requested_iterations = 0
        self.completed_iterations = 0
        self.started_at = 0.0
        self.last_error: str | None = None

    def begin(self, iterations: int) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.requested_iterations = iterations
            self.completed_iterations = 0
            self.started_at = time.time()
            self.last_error = None
            return True

    def finish(self, error: str | None) -> None:
        with self.lock:
            self.running = False
            self.last_error = error

    def view(self) -> dict:
        checkpoint = {"iteration": 0, "infosets": 0}
        telemetry: list[dict] = []
        try:
            if blueprint_trainer.TELEMETRY_PATH.exists():
                telemetry = json.loads(blueprint_trainer.TELEMETRY_PATH.read_text(encoding="utf-8"))
                if telemetry:
                    checkpoint = {
                        "iteration": telemetry[-1].get("iteration", 0),
                        "infosets": telemetry[-1].get("infosets", 0),
                    }
        except (OSError, ValueError):
            pass
        with self.lock:
            progress = 0.0
            if self.requested_iterations > 0:
                progress = min(1.0, self.completed_iterations / self.requested_iterations)
            recent_rate = telemetry[-1].get("iterations_per_second", 0.0) if telemetry else 0.0
            return {
                "running": self.running,
                "episodes": self.requested_iterations,
                "completed": self.completed_iterations,
                "progress": round(progress, 4),
                "last_error": self.last_error,
                "updates": checkpoint["iteration"],
                "parameters": checkpoint["infosets"],
                "iterations_per_second": recent_rate,
                "serving_agent": type(serving_agent).__name__,
                "river_search": getattr(serving_agent, "river_search", False),
                "artifacts": {
                    "abstraction": blueprint_trainer.ABSTRACTION_PATH.exists(),
                    "blueprint": blueprint_trainer.BLUEPRINT_PATH.exists(),
                },
                "trainer": "linear-mccfr-blueprint",
            }


training = TrainingState()


class ActionRequest(BaseModel):
    action: Literal["fold", "check", "call", "raise", "all_in"]
    amount: int | None = Field(default=None, ge=0)


class TrainingRequest(BaseModel):
    episodes: int = Field(default=50_000, ge=10, le=100_000_000, description="MCCFR iterations to run")


def play_agent_turns() -> None:
    """Keep playing until the browser player is due to act or the hand finishes."""
    safety = 0
    while not game.hand_complete and game.current_player == 1 and safety < 100:
        choice = serving_agent.select(game, 1)
        serving_agent.execute(game, 1, choice)
        safety += 1
    if safety == 100:
        raise RuntimeError("Agent action safety limit reached")
    if game.hand_complete:
        serving_agent.observe_completed_hand(game, 1)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "serving_agent": type(serving_agent).__name__}


@app.get("/api/game")
def get_game() -> dict:
    with game_lock:
        return game.snapshot()


@app.post("/api/game/new")
def new_game() -> dict:
    with game_lock:
        game.new_match()
        play_agent_turns()
        return game.snapshot()


@app.post("/api/game/next")
def next_hand() -> dict:
    with game_lock:
        if not game.hand_complete:
            raise HTTPException(status_code=400, detail="Finish the current hand before dealing the next one.")
        game.new_hand()
        play_agent_turns()
        return game.snapshot()


@app.post("/api/game/action")
def player_action(request: ActionRequest) -> dict:
    try:
        with game_lock:
            game.act(0, request.action, request.amount)
            play_agent_turns()
            return game.snapshot()
    except (InvalidAction, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/training/status")
def training_status() -> dict:
    return training.view()


def run_training_worker(iterations: int) -> None:
    log_debug("blueprint_training_started", iterations=iterations)
    error: str | None = None
    try:
        blueprint_trainer.train(
            iterations,
            save_every=min(5000, max(500, iterations // 10)),
            progress=False,
        )
        with training.lock:
            training.completed_iterations = iterations
    except BaseException as exc:  # surfaced through status, not lost with the thread
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        training.finish(error)
        with training.lock:
            if training.worker is current_thread():
                training.worker = None
        log_debug("blueprint_training_finished", iterations=iterations, error=error)


@app.post("/api/training/start")
def start_training(request: TrainingRequest) -> dict:
    if not training.begin(request.episodes):
        raise HTTPException(status_code=409, detail="Blueprint training is already running.")
    worker = Thread(target=run_training_worker, args=(request.episodes,), name="blueprint-training", daemon=True)
    with training.lock:
        training.worker = worker
    try:
        worker.start()
    except RuntimeError as exc:
        training.finish(str(exc))
        with training.lock:
            training.worker = None
        raise HTTPException(status_code=500, detail="Unable to start blueprint training.") from exc
    return training.view()


@app.post("/api/training/reload-last")
def reload_last_training_model() -> dict:
    """Re-read the blueprint artifacts so the table serves the newest checkpoint."""
    global serving_agent
    if training.running:
        raise HTTPException(status_code=409, detail="Wait for the current training run to finish before reloading.")
    with game_lock:
        serving_agent = load_serving_agent()
    kind = type(serving_agent).__name__
    log_debug("serving_agent_reloaded", kind=kind)
    if isinstance(serving_agent, HeuristicAgent):
        raise HTTPException(status_code=500, detail="No blueprint checkpoint exists yet — train first.")
    return {"ok": True, "source": str(blueprint_trainer.BLUEPRINT_PATH), "status": training.view()}
