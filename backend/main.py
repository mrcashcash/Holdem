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

from .agents.serving import load_serving_agent
from .poker import Card, HeadsUpHoldem, InvalidAction, card_text, new_deck
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


game = HeadsUpHoldem()
serving_agent = load_serving_agent()
game_lock = RLock()
log_debug("serving_agent_selected", kind=type(serving_agent).__name__)


def serving_model_view() -> dict:
    """Describe the model making table decisions, separate from CPU trainer telemetry."""
    with game_lock:
        agent = serving_agent
        selected_depth: float | None = None
        selected_iteration = getattr(agent, "iteration", None)
        available_depths: list[dict[str, float | int]] = []

        if hasattr(agent, "depth_summary") and hasattr(agent, "selected_depth"):
            depth_summary = agent.depth_summary()
            selected_depth = float(agent.selected_depth(game, 1))
            selected_iteration = depth_summary.get(selected_depth)
            available_depths = [
                {"depth_bb": float(depth), "iteration": int(iteration)}
                for depth, iteration in sorted(depth_summary.items())
            ]
        elif hasattr(agent, "tree") and hasattr(agent.tree, "config"):
            depth = getattr(agent.tree.config, "stack_bb", None)
            if depth is not None:
                selected_depth = float(depth)
                available_depths = [
                    {
                        "depth_bb": selected_depth,
                        "iteration": int(selected_iteration or 0),
                    }
                ]

        legacy_search_enabled = bool(
            getattr(agent, "subgame_search", getattr(agent, "river_search", False))
        )
        exact_river_enabled = bool(getattr(agent, "exact_river_search", False))
        search_enabled = legacy_search_enabled or exact_river_enabled
        search_iterations = getattr(
            agent,
            "exact_river_iterations" if exact_river_enabled else "subgame_iterations",
            getattr(agent, "river_iterations", None),
        )
        return {
            "kind": type(agent).__name__,
            "selected_depth_bb": selected_depth,
            "iteration": int(selected_iteration) if selected_iteration is not None else None,
            "available_depths": available_depths,
            "search_enabled": search_enabled,
            "search_iterations": (
                int(search_iterations) if search_iterations is not None else None
            ),
            "search_mode": (
                "exact-card-safe-river-v1"
                if exact_river_enabled
                else ("legacy-bucketed" if legacy_search_enabled else "blueprint-only")
            ),
            "river_budget_ms": (
                int(getattr(agent, "exact_river_budget_ms"))
                if exact_river_enabled
                else None
            ),
        }


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
        serving_model = serving_model_view()
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
                "serving_model": serving_model,
                "river_search": bool(
                    getattr(serving_agent, "exact_river_search", False)
                ),
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


class GameSettingsRequest(BaseModel):
    initial_stack: int = Field(ge=1, le=1_000_000_000)
    small_blind: int = Field(ge=1, le=1_000_000_000)
    big_blind: int = Field(ge=2, le=1_000_000_000)


class CashReloadRequest(BaseModel):
    player: Literal[0, 1, "both"]
    amount: int = Field(ge=1, le=1_000_000_000)


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


def observe_agent_hand() -> None:
    """Let the serving agent learn from a hand that has just completed."""
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
        return game.snapshot()


@app.post("/api/game/settings")
def update_game_settings(request: GameSettingsRequest) -> dict:
    """Apply play-table settings and begin a fresh match.

    Training owns its own stack-depth configuration under backend/solver and
    deliberately does not read these live game values.
    """
    global game

    if request.big_blind != request.small_blind * 2:
        raise HTTPException(status_code=400, detail="Big blind must be exactly twice the small blind.")
    if request.initial_stack < request.big_blind:
        raise HTTPException(status_code=400, detail="Starting stack must be at least one big blind.")

    with game_lock:
        game = HeadsUpHoldem(
            initial_stack=request.initial_stack,
            small_blind=request.small_blind,
            big_blind=request.big_blind,
        )
        return game.snapshot()


@app.post("/api/game/reload-cash")
def reload_game_cash(request: CashReloadRequest) -> dict:
    """Add play money between hands without resetting the match or trainer."""
    try:
        with game_lock:
            players = [0, 1] if request.player == "both" else [request.player]
            game.reload_cash(players, request.amount)
            return game.snapshot()
    except (InvalidAction, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/game/next")
def next_hand() -> dict:
    with game_lock:
        if not game.hand_complete:
            raise HTTPException(status_code=400, detail="Finish the current hand before dealing the next one.")
        game.new_hand()
        return game.snapshot()


@app.post("/api/game/action")
def player_action(request: ActionRequest) -> dict:
    try:
        with game_lock:
            game.act(0, request.action, request.amount)
            observe_agent_hand()
            return game.snapshot()
    except (InvalidAction, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def log_agent_decision(g: HeadsUpHoldem) -> None:
    """Best-effort record of the served agent's full reasoning for ONE decision
    (routed depth, effective stack, node, bucket, exact_match, and the whole
    action distribution + cards) so a fold can be explained after the fact.
    Must be called while it is still the agent's turn (before execute), and must
    never raise into the serving path.
    """
    try:
        agent = serving_agent
        routed = agent._route(g, 1) if hasattr(agent, "_route") else agent
        info = routed.strategy_for_state(g, 1)
        to_call = int(g.to_call(1))
        log_debug(
            "agent_decision",
            hand=g.hand_number,
            street=g.active_street,
            board=[card_text(c) for c in g.community],
            agent_cards=[card_text(c) for c in g.hole_cards[1]],
            pot=int(g.pot),
            to_call=to_call,
            pot_odds_pct=round(to_call / (g.pot + to_call) * 100, 1) if to_call else 0.0,
            eff_stack_bb=round(agent._effective_stack_bb(g, 1), 1) if hasattr(agent, "_effective_stack_bb") else None,
            routed_depth_bb=agent.selected_depth(g, 1) if hasattr(agent, "selected_depth") else None,
            node=info.get("node"),
            bucket=info.get("bucket"),
            exact_match=info.get("exact_match"),
            # If True, the actual action may have come from a subgame re-solve,
            # and this blueprint distribution is NOT the acting strategy.
            search_active=bool(
                getattr(routed, "subgame_search", False)
                or getattr(routed, "exact_river_search", False)
            ),
            search_mode=(
                "exact-card-safe-river-v1"
                if getattr(routed, "exact_river_search", False)
                else (
                    "legacy-bucketed"
                    if getattr(routed, "subgame_search", False)
                    else "blueprint-only"
                )
            ),
            river_resolve=getattr(routed, "last_river_search", None),
            actions=[{"a": a["action"], "amt": a.get("amount"), "p": round(a["probability"], 4)} for a in info.get("actions", [])],
            warnings=info.get("warnings", []),
        )
    except Exception as exc:  # diagnostics must never break serving
        log_debug("agent_decision_error", error=str(exc))


@app.post("/api/game/agent-action")
def agent_action() -> dict:
    """Apply exactly one agent action so the client can pace opponent turns."""
    try:
        with game_lock:
            if game.hand_complete:
                raise InvalidAction("The hand is already complete.")
            if game.current_player != 1:
                raise InvalidAction("It is not the agent's turn.")
            choice = serving_agent.select(game, 1)
            log_agent_decision(game)  # capture reasoning while it is still the agent's turn
            serving_agent.execute(game, 1, choice)
            observe_agent_hand()
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


@app.post("/api/champion/reload")
def reload_champion_query_agent() -> dict:
    """Force the promoted champion to be re-read from disk on the next query.

    The champion query agent normally hot-reloads when champion.npz changes, but
    this lets a client force it (e.g. after promoting a new champion) so the live
    screen decision overlay serves the newest strategy without restarting the
    server."""
    global champion_query_agent, champion_query_mtime_ns
    with champion_query_lock:
        champion_query_agent = None
        champion_query_mtime_ns = None
    try:
        agent, source = load_champion_query_agent()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    log_debug("champion_query_agent_reloaded", source=str(source))
    return {"reloaded": True, "source": str(source), "iteration": int(agent.iteration)}


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
    if hasattr(serving_agent, "depth_summary"):  # multi-stack router
        source = "multi-stack champions: " + ", ".join(
            f"{depth:.0f}bb@{iters}" for depth, iters in sorted(serving_agent.depth_summary().items())
        )
        iteration = int(serving_agent.iteration)
    elif isinstance(serving_agent, GpuBlueprintAgent):
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
