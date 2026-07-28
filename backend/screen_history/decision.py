"""Validated screen-state input and final-action output for the poker brain."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from backend.agents.serving import load_serving_agent
from backend.poker import HeadsUpHoldem, InvalidAction, card_text


@dataclass(frozen=True)
class DecisionRequest:
    game: HeadsUpHoldem
    hand_number: int | None
    captured_at: str
    recognition_confidence: float
    recognition_ms: int | None
    state_key: tuple[Any, ...]
    action_signature: tuple[tuple[int, str, int | None, str], ...]


@dataclass(frozen=True)
class BrainDecision:
    decision_id: str
    hand_number: int | None
    captured_at: str
    decided_at: str
    action: str
    amount: int | None
    all_in: bool
    model: str
    iteration: int | None
    selected_depth_bb: float | None
    recognition_confidence: float
    street: str
    pot: int
    to_call: int
    hero_cards: tuple[str, ...]
    board: tuple[str, ...]
    stacks: tuple[int, int]
    legal_actions: dict[str, int | bool]
    state_fingerprint: str
    action_signature: tuple[tuple[int, str, int | None, str], ...]
    warnings: tuple[str, ...] = ()
    strategy: tuple[dict[str, Any], ...] = ()
    latency_ms: int | None = None
    total_latency_ms: int | None = None
    source: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _state_fingerprint(request: DecisionRequest) -> str:
    fingerprint_payload = {
        "hand_number": request.hand_number,
        "state_key": request.state_key,
        "public_actions": request.game.public_actions,
    }
    return hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _total_latency_ms(captured_at: str) -> int | None:
    try:
        captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        return max(
            0,
            int(
                round(
                    (
                        datetime.now(timezone.utc)
                        - captured.astimezone(timezone.utc)
                    ).total_seconds()
                    * 1000
                )
            ),
        )
    except (TypeError, ValueError):
        return None


class BrainDecisionEngine:
    """Load the serving brain and sample one final legal Hero action."""

    def __init__(self) -> None:
        self.agent = load_serving_agent()
        self.model_name = type(self.agent).__name__

    def decide(self, request: DecisionRequest) -> BrainDecision:
        game = copy.deepcopy(request.game)
        if request.hand_number is not None:
            game.hand_number = request.hand_number
        if game.hand_complete or game.current_player != 0:
            raise ValueError("The reconstructed game is not waiting for Hero.")
        if len(game.hole_cards[0]) != 2:
            raise ValueError("Two validated Hero cards are required for a decision.")
        legal = game.legal_actions(0)
        if not legal:
            raise ValueError("The reconstructed state has no legal Hero actions.")

        public_action_count = len(game.public_actions)
        try:
            choice = self.agent.select(game, 0)
            self.agent.execute(game, 0, choice)
        except Exception as exc:
            raise RuntimeError(f"The serving brain could not decide: {exc}") from exc
        if len(game.public_actions) != public_action_count + 1:
            raise RuntimeError("The serving brain did not produce exactly one public action.")

        event = game.public_actions[-1]
        engine_action = str(event["action"])
        raw_amount = int(event.get("amount", 0))
        amount = raw_amount if engine_action in {"raise", "call"} else None
        all_in = bool(
            (engine_action == "raise" and raw_amount >= int(legal["raise_max"]))
            or (
                engine_action == "call"
                and int(legal["to_call"]) >= int(request.game.stacks[0])
            )
        )
        action = "all_in" if all_in else engine_action

        selected_depth: float | None = None
        if hasattr(self.agent, "selected_depth"):
            try:
                selected_depth = float(self.agent.selected_depth(request.game, 0))
            except (RuntimeError, TypeError, ValueError):
                selected_depth = None
        iteration_value = getattr(self.agent, "iteration", None)
        iteration = int(iteration_value) if iteration_value is not None else None
        warnings: list[str] = []
        if self.model_name == "HeuristicAgent":
            warnings.append("No trained blueprint was available; heuristic fallback was used.")

        fingerprint = _state_fingerprint(request)
        decided_at = datetime.now(timezone.utc).isoformat()
        return BrainDecision(
            decision_id=f"{request.hand_number or 'unknown'}-{fingerprint}",
            hand_number=request.hand_number,
            captured_at=request.captured_at,
            decided_at=decided_at,
            action=action,
            amount=amount,
            all_in=all_in,
            model=self.model_name,
            iteration=iteration,
            selected_depth_bb=selected_depth,
            recognition_confidence=request.recognition_confidence,
            street=request.game.active_street,
            pot=int(request.game.pot),
            to_call=int(request.game.to_call(0)),
            hero_cards=tuple(card_text(card) for card in request.game.hole_cards[0]),
            board=tuple(card_text(card) for card in request.game.community),
            stacks=(int(request.game.stacks[0]), int(request.game.stacks[1])),
            legal_actions=dict(legal),
            state_fingerprint=fingerprint,
            action_signature=request.action_signature,
            warnings=tuple(warnings),
            total_latency_ms=_total_latency_ms(request.captured_at),
        )


class ServerBrainDecisionEngine:
    """Query the running champion API and return one locally validated action."""

    model_name = "Champion server"

    def __init__(self, server_url: str, timeout: float = 10.0) -> None:
        base_url = server_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Decision server URL must begin with http:// or https://.")
        if timeout <= 0:
            raise ValueError("Decision server timeout must be positive.")
        self.server_url = base_url
        self.timeout = timeout

    @staticmethod
    def _scale_chips(value: int, scale: float) -> int:
        return max(0, int(round(float(value) * scale)))

    def _request_payload(
        self,
        request: DecisionRequest,
        scale: float,
    ) -> dict[str, Any]:
        game = request.game
        starting_stacks = [
            int(game.stacks[player]) + int(game.contributions[player])
            for player in (0, 1)
        ]
        actions: list[dict[str, Any]] = []
        for event in game.public_actions:
            action = str(event["action"])
            if action == "blind":
                continue
            amount = event.get("amount")
            actions.append(
                {
                    "player": int(event["player"]),
                    "action": action,
                    "amount": (
                        self._scale_chips(int(amount), scale)
                        if action == "raise" and amount is not None
                        else None
                    ),
                }
            )
        return {
            "hero_cards": [card_text(card) for card in game.hole_cards[0]],
            "board": [card_text(card) for card in game.community],
            "button": int(game.button),
            "stacks": [
                self._scale_chips(stack, scale) for stack in starting_stacks
            ],
            "actions": actions,
            "current": False,
        }

    def _query(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        request_body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.server_url}/api/champion/query",
            data=request_body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
                detail = error_payload.get("detail", str(exc))
            except (UnicodeDecodeError, json.JSONDecodeError):
                detail = str(exc)
            raise RuntimeError(f"Champion server rejected the live hand: {detail}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Champion server is unavailable: {reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Champion server decision timed out.") from exc
        latency_ms = int(round((time.monotonic() - started) * 1000))
        if not isinstance(result, dict):
            raise RuntimeError("Champion server returned an invalid response.")
        return result, latency_ms

    @staticmethod
    def _local_action(
        game: HeadsUpHoldem,
        candidate: dict[str, Any],
        scale: float,
    ) -> tuple[str, int | None, bool]:
        action = str(candidate.get("action", "")).lower()
        if action not in {"fold", "check", "call", "raise", "all_in"}:
            raise ValueError(f"Champion server returned unknown action '{action}'.")
        legal = game.legal_actions(0)
        if not legal.get(action):
            raise ValueError(f"Champion server action '{action}' is not legal locally.")

        amount: int | None = None
        if action == "raise":
            remote_amount = candidate.get("amount")
            if remote_amount is None:
                raise ValueError("Champion server returned a raise without an amount.")
            amount = int(round(float(remote_amount) / scale))
            amount = max(
                int(legal["raise_min"]),
                min(int(legal["raise_max"]), amount),
            )
        elif action == "call":
            amount = min(int(legal["to_call"]), int(game.stacks[0]))
        elif action == "all_in":
            amount = int(legal["raise_max"])

        verification = copy.deepcopy(game)
        try:
            verification.act(0, action, amount if action == "raise" else None)
        except (InvalidAction, ValueError) as exc:
            raise ValueError(
                f"Champion server action '{action}' failed local validation: {exc}"
            ) from exc
        return action, amount, action == "all_in"

    @staticmethod
    def _converted_strategy(
        actions: list[dict[str, Any]],
        scale: float,
    ) -> tuple[dict[str, Any], ...]:
        converted: list[dict[str, Any]] = []
        for candidate in actions:
            item = dict(candidate)
            remote_amount = item.get("amount")
            if remote_amount is not None:
                item["server_amount"] = remote_amount
                item["amount"] = int(round(float(remote_amount) / scale))
            converted.append(item)
        return tuple(converted)

    def decide(self, request: DecisionRequest) -> BrainDecision:
        game = copy.deepcopy(request.game)
        if request.hand_number is not None:
            game.hand_number = request.hand_number
        if game.hand_complete or game.current_player != 0:
            raise ValueError("The reconstructed game is not waiting for Hero.")
        if len(game.hole_cards[0]) != 2:
            raise ValueError("Two validated Hero cards are required for a decision.")
        legal = game.legal_actions(0)
        if not legal:
            raise ValueError("The reconstructed state has no legal Hero actions.")
        if game.big_blind <= 0:
            raise ValueError("The reconstructed game has an invalid big blind.")

        scale = 20.0 / float(game.big_blind)
        payload = self._request_payload(request, scale)
        result, latency_ms = self._query(payload)
        raw_actions = result.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise RuntimeError("Champion server returned no strategy actions.")

        validation_errors: list[str] = []
        selected: tuple[str, int | None, bool] | None = None
        for candidate in raw_actions:
            if not isinstance(candidate, dict):
                continue
            try:
                selected = self._local_action(game, candidate, scale)
                break
            except ValueError as exc:
                validation_errors.append(str(exc))
        if selected is None:
            detail = "; ".join(validation_errors) or "No valid action candidates."
            raise RuntimeError(
                f"Champion server strategy had no locally legal action: {detail}"
            )

        action, amount, all_in = selected
        warnings = [str(value) for value in result.get("warnings", [])]
        if abs((game.small_blind / game.big_blind) - 0.5) > 1e-9:
            warnings.append(
                "Live chips were normalized to the champion's 10/20 blind abstraction; "
                "the opening small blind differs from the trained 0.5 BB size."
            )
        warnings.extend(validation_errors)
        source_value = result.get("source")
        source = str(source_value) if source_value is not None else None
        iteration_value = result.get("iteration")
        iteration = int(iteration_value) if iteration_value is not None else None

        return BrainDecision(
            decision_id=f"{request.hand_number or 'unknown'}-{_state_fingerprint(request)}",
            hand_number=request.hand_number,
            captured_at=request.captured_at,
            decided_at=datetime.now(timezone.utc).isoformat(),
            action=action,
            amount=amount,
            all_in=all_in,
            model=self.model_name,
            iteration=iteration,
            selected_depth_bb=None,
            recognition_confidence=request.recognition_confidence,
            street=game.active_street,
            pot=int(game.pot),
            to_call=int(game.to_call(0)),
            hero_cards=tuple(card_text(card) for card in game.hole_cards[0]),
            board=tuple(card_text(card) for card in game.community),
            stacks=(int(game.stacks[0]), int(game.stacks[1])),
            legal_actions=dict(legal),
            state_fingerprint=_state_fingerprint(request),
            action_signature=request.action_signature,
            warnings=tuple(warnings),
            strategy=self._converted_strategy(raw_actions, scale),
            latency_ms=latency_ms,
            total_latency_ms=_total_latency_ms(request.captured_at),
            source=source,
        )
