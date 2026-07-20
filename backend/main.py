"""FastAPI application serving the game and the blueprint trainer.

The table is served by the MCCFR blueprint agent (backend/agents) whenever
its artifacts exist under backend/data/blueprint/; a simple heuristic agent
fills in before the first blueprint checkpoint is written. The training
endpoints drive the Linear MCCFR blueprint trainer (backend/solver) — the
legacy PPO league trainer was retired to legacy/ (docs/REDESIGN_PLAN.md).
"""

from __future__ import annotations

import copy
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
from .poker import Card, HeadsUpHoldem, InvalidAction, new_deck
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
        # Turn/river re-solving is ON by default: CUDA-graph replay brought
        # solves to ~3s turn / <1s river (once per hand, cached). Set
        # HOLDEM_SUBGAME_ITERS=0 to disable, or tune the iteration count.
        subgame_iterations = int(os.environ.get("HOLDEM_SUBGAME_ITERS", "120"))
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


class ChampionHistoryAction(BaseModel):
    player: Literal[0, 1]
    action: Literal["fold", "check", "call", "raise", "all_in"]
    amount: int | None = Field(default=None, ge=0)


class ChampionSpotRequest(BaseModel):
    hero_cards: list[str] = Field(default_factory=list)
    board: list[str] = Field(default_factory=list)
    button: Literal[0, 1] = 0
    stacks: list[int] = Field(default_factory=lambda: [2_000, 2_000])
    actions: list[ChampionHistoryAction] = Field(default_factory=list)


class ChampionQueryRequest(ChampionSpotRequest):
    current: bool = False


champion_query_lock = RLock()
champion_query_agent = None
champion_query_mtime_ns: int | None = None


def parse_query_card(value: str) -> Card:
    cleaned = value.strip().replace("10", "T")
    if len(cleaned) != 2:
        raise ValueError(f"Invalid card '{value}'. Use notation such as As, Kh, or 7d.")
    rank_label, suit_label = cleaned[0].upper(), cleaned[1].lower()
    ranks = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10, "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2}
    suits = {
        "s": "\u2660",
        "h": "\u2665",
        "d": "\u2666",
        "c": "\u2663",
        "\u2660": "\u2660",
        "\u2665": "\u2665",
        "\u2666": "\u2666",
        "\u2663": "\u2663",
    }
    if rank_label not in ranks or suit_label not in suits:
        raise ValueError(f"Invalid card '{value}'. Use notation such as As, Kh, or 7d.")
    return ranks[rank_label], suits[suit_label]


def build_champion_query_game(
    request: ChampionSpotRequest,
    *,
    allow_partial_board: bool = False,
    require_hero_turn: bool = True,
) -> tuple[HeadsUpHoldem, list[Card], int]:
    if len(request.hero_cards) != 2:
        raise ValueError("Exactly two hero cards are required.")
    if len(request.board) > 5 or (not allow_partial_board and len(request.board) not in {0, 3, 4, 5}):
        raise ValueError("The board must contain 0, 3, 4, or 5 cards.")
    if len(request.stacks) != 2 or any(stack < 20 for stack in request.stacks):
        raise ValueError("Provide two starting stacks of at least one big blind (20 chips).")

    hero_cards = [parse_query_card(card) for card in request.hero_cards]
    board = [parse_query_card(card) for card in request.board]
    known_cards = hero_cards + board
    if len(set(known_cards)) != len(known_cards):
        raise ValueError("A card cannot appear more than once.")

    query_game = HeadsUpHoldem(initial_stack=max(request.stacks))
    query_game.stacks = list(request.stacks)
    query_game.hand_number = 0
    query_game.button_offset = request.button
    query_game.new_hand()

    available = [card for card in new_deck() if card not in set(known_cards)]
    opponent_cards = available[:2]
    future_cards = [card for card in available[2:] if card not in set(board)]
    query_game.hole_cards = [hero_cards, opponent_cards]
    query_game.deck = future_cards + list(reversed(board))

    for index, action in enumerate(request.actions, start=1):
        try:
            query_game.act(action.player, action.action, action.amount)
        except (InvalidAction, ValueError) as exc:
            raise ValueError(f"Action {index} is invalid: {exc}") from exc
        if query_game.hand_complete and index != len(request.actions):
            raise ValueError(f"Action {index} completes the hand; later actions cannot be applied.")

    missing_board_cards = max(0, len(query_game.community) - len(board))
    if not allow_partial_board and missing_board_cards:
        raise ValueError(f"Choose {missing_board_cards} more board card(s) to reach the {query_game.active_street}.")
    if require_hero_turn and query_game.hand_complete:
        raise ValueError("The action history completes the hand, so there is no decision to query.")
    if require_hero_turn and query_game.current_player != 0:
        raise ValueError("The action history must end when Hero (player 0) is due to act.")
    return query_game, board, missing_board_cards


def champion_spot_payload(
    spot_game: HeadsUpHoldem,
    request: ChampionSpotRequest,
    supplied_board: list[Card],
    missing_board_cards: int,
) -> dict:
    current_player = spot_game.current_player
    legal = spot_game.legal_actions(current_player) if current_player is not None else {}
    to_call = spot_game.to_call(current_player) if current_player is not None else 0
    visible_count = min(len(spot_game.community), len(supplied_board))
    visible_board = request.board[:visible_count]
    effective_stack = min(spot_game.stacks) if spot_game.stacks else 0
    pot_odds = to_call / (spot_game.pot + to_call) if to_call > 0 else 0.0
    spr = effective_stack / spot_game.pot if spot_game.pot > 0 else 0.0
    hand_strength = None
    if not missing_board_cards and visible_count >= 3:
        hand_strength = spot_game.snapshot(0).get("hero_hand_strength")
    return {
        "hero_cards": request.hero_cards,
        "board": visible_board,
        "staged_board": request.board[visible_count:],
        "button": spot_game.button,
        "street": spot_game.active_street,
        "current_player": current_player,
        "starting_stacks": request.stacks,
        "stacks": list(spot_game.stacks),
        "round_bets": list(spot_game.round_bets),
        "pot": int(spot_game.pot),
        "to_call": int(to_call),
        "legal_actions": legal,
        "complete": spot_game.hand_complete,
        "result": spot_game.result,
        "required_board_count": len(spot_game.community) if missing_board_cards else None,
        "metrics": {
            "effective_stack": int(effective_stack),
            "effective_stack_bb": round(effective_stack / spot_game.big_blind, 1),
            "spr": round(spr, 2),
            "pot_odds_percent": round(pot_odds * 100, 1),
            "hand_strength": hand_strength,
        },
    }


def load_champion_query_agent():
    global champion_query_agent, champion_query_mtime_ns
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
    from backend.solver.gpu import train as gpu_trainer

    champion_path = gpu_trainer.DATA_DIR / "champion.npz"
    if not champion_path.exists():
        raise FileNotFoundError("No promoted GPU champion exists yet.")
    modified = champion_path.stat().st_mtime_ns
    with champion_query_lock:
        if champion_query_agent is None or champion_query_mtime_ns != modified:
            champion_query_agent = GpuBlueprintAgent.try_load(champion_path)
            champion_query_mtime_ns = modified
        if champion_query_agent is None:
            raise FileNotFoundError("The promoted GPU champion could not be loaded.")
        return champion_query_agent, champion_path


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


@app.post("/api/champion/query")
def query_champion(request: ChampionQueryRequest) -> dict:
    """Inspect the promoted champion's mixed strategy without changing a game."""
    try:
        if request.current:
            with game_lock:
                if game.current_player != 0 or game.hand_complete:
                    raise ValueError("The current table can only be queried when it is your turn.")
                query_game = copy.deepcopy(game)
            spot_payload = None
        else:
            query_game, supplied_board, missing_board_cards = build_champion_query_game(request)
            spot_payload = champion_spot_payload(query_game, request, supplied_board, missing_board_cards)

        agent, source = load_champion_query_agent()
        with champion_query_lock:
            result = agent.strategy_for_state(query_game, 0)
        for action in result["actions"]:
            action["percentage"] = round(action["probability"] * 100, 1)
        recommended = result["actions"][0]
        return {
            "source": str(source),
            "iteration": int(agent.iteration),
            "street": query_game.active_street,
            "position": "button" if query_game.button == 0 else "big_blind",
            "pot": int(query_game.pot),
            "to_call": int(query_game.to_call(0)),
            "legal_actions": query_game.legal_actions(0),
            "spot": spot_payload,
            **result,
            "recommended": recommended,
        }
    except (FileNotFoundError, InvalidAction, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/champion/spot")
def preview_champion_spot(request: ChampionSpotRequest) -> dict:
    """Reconstruct any intermediate lab state without loading or querying a model."""
    try:
        spot_game, supplied_board, missing_board_cards = build_champion_query_game(
            request,
            allow_partial_board=True,
            require_hero_turn=False,
        )
        return champion_spot_payload(spot_game, request, supplied_board, missing_board_cards)
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
    from backend.agents.gpu_blueprint_agent import GpuBlueprintAgent
    from backend.solver.gpu import train as gpu_trainer

    kind = type(serving_agent).__name__
    log_debug("serving_agent_reloaded", kind=kind)
    if isinstance(serving_agent, HeuristicAgent):
        raise HTTPException(status_code=500, detail="No blueprint checkpoint exists yet — train first.")
    if isinstance(serving_agent, GpuBlueprintAgent):
        source = str(gpu_trainer.CHECKPOINT_PATH)
        iteration = int(serving_agent.iteration)
    else:
        source = str(blueprint_trainer.BLUEPRINT_PATH)
        iteration = None
    return {
        "ok": True,
        "agent": kind,
        "source": source,
        "iteration": iteration,
        "status": training.view(),
    }
