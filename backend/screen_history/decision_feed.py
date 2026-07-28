"""Localhost-only HTTP feed for displaying live screen decisions in the lab UI."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .runtime import RuntimeEvent


_LOCAL_ORIGIN = re.compile(r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$")
_STALE_DECISION_SECONDS = 12.0
_HISTORY_HAND_LIMIT = 12
_HISTORY_STEP_LIMIT = 48
_STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_EXPECTED_BOARD_CARDS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_payload(event: RuntimeEvent) -> dict[str, Any] | None:
    state = event.state
    if state is None:
        return None
    return {
        "captured_at": state.captured_at,
        "hand_number": state.hand_number,
        "street": state.street,
        "pot": state.pot,
        "stacks": list(state.stacks),
        "round_bets": list(state.round_bets),
        "hero_cards": list(state.hero_cards),
        "board": list(state.board),
        "button": state.button,
        "current_player": state.current_player,
        "complete": state.complete,
        "stable": state.stable,
        "history_stable": state.history_stable,
        "confidence": state.confidence,
        "recognition_ms": state.recognition_ms,
        "warnings": list(state.warnings),
        "players": list(state.players),
        "visible_actions": [asdict(action) for action in state.visible_actions],
        "timeline_starts_at_hand": state.timeline_starts_at_hand,
    }


def _decision_matches_table(
    decision: dict[str, Any],
    table: dict[str, Any],
) -> bool:
    table_actions = tuple(
        (
            int(action.get("player", -1)),
            str(action.get("action", "")),
            (
                int(action["amount"])
                if action.get("amount") is not None
                else None
            ),
            str(action.get("street", "")),
        )
        for action in table.get("visible_actions", ())
        if isinstance(action, dict)
    )
    decision_actions = tuple(
        (
            int(action[0]),
            str(action[1]),
            int(action[2]) if action[2] is not None else None,
            str(action[3]),
        )
        for action in decision.get("action_signature", ())
        if isinstance(action, (list, tuple)) and len(action) == 4
    )
    return bool(
        table.get("hand_number") == decision.get("hand_number")
        and table.get("street") == decision.get("street")
        and table.get("hero_cards") == list(decision.get("hero_cards", ()))
        and table.get("board") == list(decision.get("board", ()))
        and table.get("current_player") == 0
        and not table.get("complete")
        and table_actions == decision_actions
    )


class LiveDecisionFeed:
    """Publish the watcher's latest safe recommendation on loopback only."""

    def __init__(self, port: int = 8765, amount_scale: int = 1) -> None:
        if not 1 <= port <= 65_535:
            raise ValueError("Decision feed port must be between 1 and 65535.")
        if amount_scale <= 0:
            raise ValueError("Decision feed amount scale must be positive.")
        self.port = port
        self.amount_scale = amount_scale
        self._lock = threading.RLock()
        self._history_sequence = 0
        self._payload: dict[str, Any] = {
            "connected": True,
            "status": "waiting",
            "message": "Waiting for a validated Hero turn.",
            "updated_at": _now(),
            "amount_scale": amount_scale,
            "table": None,
            "decision": None,
            "history": [],
            "history_gap_count": 0,
        }
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._stale_since: float | None = None
        self._condition = threading.Condition(self._lock)
        self._version = 0
        self._closed = threading.Event()

    def _expire_stale_locked(self) -> None:
        if (
            self._payload.get("status") == "stale"
            and self._stale_since is not None
            and time.monotonic() - self._stale_since >= _STALE_DECISION_SECONDS
        ):
            self._payload = {
                **self._payload,
                "status": "waiting",
                "message": "Waiting for a validated Hero turn.",
                "decision": None,
                "updated_at": _now(),
            }
            self._stale_since = None
            self._version += 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            self._expire_stale_locked()
            return json.loads(json.dumps(self._payload))

    def _replace(self, **values: Any) -> None:
        with self._condition:
            self._payload = {
                **self._payload,
                **values,
                "connected": True,
                "updated_at": _now(),
            }
            self._version += 1
            self._condition.notify_all()

    def _wait_for_update(
        self,
        last_version: int,
        timeout: float = 10.0,
    ) -> tuple[int, dict[str, Any] | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._version != last_version or self._closed.is_set(),
                timeout=timeout,
            )
            if self._closed.is_set():
                return self._version, None
            self._expire_stale_locked()
            return self._version, json.loads(json.dumps(self._payload))

    def _finalize_previous_hand_locked(self, next_hand_number: int | None) -> None:
        history = self._payload["history"]
        if not history:
            return
        previous = history[-1]
        if previous["hand_number"] == next_hand_number:
            return
        if previous["status"] == "in_progress":
            if previous["complete"]:
                previous["status"] = "verified"
                previous["verification_message"] = "Complete hand verified."
            elif (
                not previous["unrecoverable_gap"]
                and any(
                    step.get("verified") and step.get("actions")
                    for step in previous["steps"]
                )
            ):
                previous["status"] = "verified_actions"
                previous["verification_message"] = (
                    "All visible Dealer Chat actions were verified. "
                    "The terminal result was not shown before the next hand."
                )
            else:
                previous["status"] = "partial"
                previous["verification_message"] = (
                    "The next hand appeared before a complete final state was observed."
                )

    def _new_history_hand_locked(
        self,
        table: dict[str, Any],
    ) -> dict[str, Any]:
        self._finalize_previous_hand_locked(table.get("hand_number"))
        self._history_sequence += 1
        entry = {
            "id": f"live-{self._history_sequence}",
            "hand_number": table.get("hand_number"),
            "started_at": table["captured_at"],
            "updated_at": table["captured_at"],
            "status": "in_progress",
            "verification_message": "Reading the current hand.",
            "hero_cards": list(table.get("hero_cards", ())),
            "players": list(table.get("players", ())),
            "button": table.get("button"),
            "complete": bool(table.get("complete")),
            "recovered": False,
            "unrecoverable_gap": False,
            "steps": [],
            "decisions": [],
        }
        history = self._payload["history"]
        history.append(entry)
        if len(history) > _HISTORY_HAND_LIMIT:
            del history[: len(history) - _HISTORY_HAND_LIMIT]
        return entry

    def _active_history_hand_locked(
        self,
        table: dict[str, Any],
        transition_status: str,
    ) -> dict[str, Any]:
        history = self._payload["history"]
        hand_number = table.get("hand_number")
        if (
            not history
            or transition_status == "new_hand"
            or history[-1]["hand_number"] != hand_number
        ):
            return self._new_history_hand_locked(table)
        return history[-1]

    def _record_state_locked(
        self,
        event: RuntimeEvent,
        table: dict[str, Any],
    ) -> None:
        if not table.get("history_stable"):
            return
        transition = event.transition
        transition_status = transition.status if transition is not None else "observed"
        entry = self._active_history_hand_locked(table, transition_status)
        entry["updated_at"] = table["captured_at"]
        entry["hero_cards"] = list(table.get("hero_cards", ()))
        entry["players"] = list(table.get("players", ()))
        entry["button"] = table.get("button")
        entry["complete"] = bool(table.get("complete"))

        warnings = list(table.get("warnings", ()))
        if transition is not None and transition.warning:
            warnings.append(transition.warning)
        warnings = list(dict.fromkeys(warnings))
        street = table.get("street")
        board = list(table.get("board", ()))
        actions = list(table.get("visible_actions", ()))
        verified = bool(
            table.get("history_stable")
            and table.get("hand_number") is not None
            and street in _STREET_INDEX
            and len(table.get("hero_cards", ())) == 2
            and len(board) == _EXPECTED_BOARD_CARDS.get(street)
            and transition_status
            not in {"ambiguous", "unmatched", "untracked", "transient"}
        )
        recovered = transition_status == "resynced"
        gap_reasons: list[str] = []
        steps = entry["steps"]
        previous = steps[-1] if steps else None
        if previous is not None:
            previous_street = previous.get("street")
            if (
                street in _STREET_INDEX
                and previous_street in _STREET_INDEX
                and _STREET_INDEX[street] < _STREET_INDEX[previous_street]
            ):
                gap_reasons.append(
                    f"Street moved backward from {previous_street} to {street}."
                )
            previous_board = list(previous.get("board", ()))
            if len(board) < len(previous_board) or board[: len(previous_board)] != previous_board:
                gap_reasons.append("The verified board did not continue from the previous step.")
            previous_actions = list(previous.get("actions", ()))
            if (
                len(actions) < len(previous_actions)
                or actions[: len(previous_actions)] != previous_actions
            ):
                if recovered:
                    entry["recovered"] = True
                    entry["verification_message"] = (
                        "Recovered from the complete Dealer Chat timeline."
                    )
                else:
                    gap_reasons.append(
                        "The Dealer Chat action sequence did not continue from the previous step."
                    )

        step_key = (
            street,
            table.get("pot"),
            tuple(board),
            table.get("current_player"),
            bool(table.get("complete")),
            tuple(
                (
                    action.get("player"),
                    action.get("action"),
                    action.get("amount"),
                    action.get("street"),
                )
                for action in actions
            ),
        )
        previous_key = (
            (
                previous.get("street"),
                previous.get("pot"),
                tuple(previous.get("board", ())),
                previous.get("current_player"),
                bool(previous.get("complete")),
                tuple(
                    (
                        action.get("player"),
                        action.get("action"),
                        action.get("amount"),
                        action.get("street"),
                    )
                    for action in previous.get("actions", ())
                ),
            )
            if previous is not None
            else None
        )
        if previous is not None and previous_key == step_key:
            previous.update(
                {
                    "captured_at": table["captured_at"],
                    "confidence": table.get("confidence", 0.0),
                    "recognition_ms": table.get("recognition_ms"),
                    "verified": verified and not gap_reasons,
                    "warnings": warnings,
                }
            )
        else:
            self._history_sequence += 1
            steps.append(
                {
                    "id": f"step-{self._history_sequence}",
                    "captured_at": table["captured_at"],
                    "street": street,
                    "pot": table.get("pot"),
                    "stacks": list(table.get("stacks", ())),
                    "board": board,
                    "current_player": table.get("current_player"),
                    "complete": bool(table.get("complete")),
                    "confidence": table.get("confidence", 0.0),
                    "recognition_ms": table.get("recognition_ms"),
                    "transition": transition_status,
                    "verified": verified and not gap_reasons,
                    "recovered": recovered,
                    "warnings": warnings,
                    "actions": actions,
                    "decision": None,
                }
            )
            if len(steps) > _HISTORY_STEP_LIMIT:
                del steps[: len(steps) - _HISTORY_STEP_LIMIT]

        if gap_reasons or not verified:
            if gap_reasons or transition_status in {
                "ambiguous",
                "unmatched",
                "untracked",
            }:
                entry["status"] = "gap"
                if gap_reasons or transition_status in {"ambiguous", "unmatched"}:
                    entry["unrecoverable_gap"] = True
                entry["verification_message"] = " ".join(
                    gap_reasons
                    or [
                        transition.warning
                        if transition is not None and transition.warning
                        else f"The {transition_status} transition could not be verified."
                    ]
                )
        elif not entry["unrecoverable_gap"]:
            if table.get("complete"):
                entry["status"] = "verified"
                entry["verification_message"] = (
                    "Complete hand verified"
                    + (" after Dealer Chat recovery." if entry["recovered"] else ".")
                )
            else:
                entry["status"] = "in_progress"
                entry["verification_message"] = (
                    "All observed steps are verified"
                    + (" after Dealer Chat recovery." if entry["recovered"] else ".")
                )

    def _record_history_gap_locked(self, message: str) -> None:
        self._payload["history_gap_count"] += 1
        history = self._payload["history"]
        if not history:
            return
        entry = history[-1]
        entry["status"] = "gap"
        entry["unrecoverable_gap"] = True
        entry["verification_message"] = message

    def _record_decision_locked(self, decision: dict[str, Any]) -> None:
        hand_number = decision.get("hand_number")
        matching = next(
            (
                entry
                for entry in reversed(self._payload["history"])
                if entry.get("hand_number") == hand_number
            ),
            None,
        )
        if matching is None:
            return
        if not any(
            item.get("decision_id") == decision.get("decision_id")
            for item in matching["decisions"]
        ):
            matching["decisions"].append(decision)
        for step in reversed(matching["steps"]):
            if (
                step.get("street") == decision.get("street")
                and step.get("board") == list(decision.get("board", ()))
            ):
                step["decision"] = decision
                break

    def publish(self, event: RuntimeEvent) -> None:
        table = _table_payload(event)
        if event.kind == "state" and table is not None:
            with self._condition:
                self._record_state_locked(event, table)
            with self._lock:
                current_decision = self._payload.get("decision")
            if isinstance(current_decision, dict) and _decision_matches_table(
                current_decision, table
            ):
                with self._lock:
                    self._stale_since = None
                self._replace(
                    status="ready",
                    message="Live recommendation ready.",
                    table=table,
                )
            elif isinstance(current_decision, dict):
                with self._lock:
                    if self._stale_since is None:
                        self._stale_since = time.monotonic()
                self._replace(
                    status="stale",
                    message=(
                        "Last validated action retained after the table advanced. "
                        "Do not treat it as a current recommendation."
                    ),
                    table=table,
                    decision=current_decision,
                )
            else:
                with self._lock:
                    self._stale_since = None
                self._replace(
                    status="waiting",
                    message=(
                        "The table advanced; waiting for a new validated recommendation."
                        if current_decision
                        else "Waiting for a validated Hero turn."
                    ),
                    table=table,
                    decision=None,
                )
            return

        if event.kind == "brain_thinking":
            with self._lock:
                self._stale_since = None
            self._replace(
                status="thinking",
                message=event.message or "Champion server is evaluating the live hand.",
                table=table,
                decision=None,
            )
        elif event.kind == "brain_decision" and event.decision is not None:
            decision = asdict(event.decision)
            with self._condition:
                self._record_decision_locked(decision)
            with self._lock:
                self._stale_since = None
            self._replace(
                status="ready",
                message=event.message or "Live recommendation ready.",
                decision=decision,
            )
        elif event.kind == "history_gap":
            with self._condition:
                self._record_history_gap_locked(
                    event.message or "A recognition milestone was not processed."
                )
            self._replace(
                message=event.message or "The live hand history contains a capture gap.",
            )
        elif event.kind == "brain_stale":
            with self._lock:
                current_decision = self._payload.get("decision")
                if isinstance(current_decision, dict) and self._stale_since is None:
                    self._stale_since = time.monotonic()
            self._replace(
                status="stale",
                message=(
                    "Last validated action retained after the table advanced. "
                    "Do not treat it as a current recommendation."
                    if isinstance(current_decision, dict)
                    else event.message or "The table advanced before the answer arrived."
                ),
                decision=current_decision if isinstance(current_decision, dict) else None,
            )
        elif event.kind == "brain_error":
            with self._lock:
                self._stale_since = None
            self._replace(
                status="error",
                message=event.message or "The champion server could not decide.",
                decision=None,
            )
        elif event.kind == "brain_ready":
            with self._lock:
                self._stale_since = None
            self._replace(
                status="waiting",
                message="Connected. Waiting for a validated Hero turn.",
                decision=None,
            )
        elif event.kind == "brain_skipped" and table is not None and table.get("stable"):
            with self._lock:
                current_decision = self._payload.get("decision")
                if isinstance(current_decision, dict):
                    if self._stale_since is None:
                        self._stale_since = time.monotonic()
                else:
                    self._stale_since = None
            if isinstance(current_decision, dict):
                self._replace(
                    status="stale",
                    message=(
                        "Last validated action retained after the table advanced. "
                        "Do not treat it as a current recommendation."
                    ),
                    table=table,
                    decision=current_decision,
                )
            else:
                self._replace(
                    status="waiting",
                    message=event.message or "This table state is not eligible for a decision.",
                    table=table,
                    decision=None,
                )

    def start(self) -> None:
        if self._server is not None:
            return
        self._closed.clear()
        feed = self

        class Handler(BaseHTTPRequestHandler):
            def _cors(self) -> None:
                origin = self.headers.get("Origin", "")
                if _LOCAL_ORIGIN.fullmatch(origin):
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._cors()
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                if path == "/events":
                    self.send_response(200)
                    self._cors()
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Connection", "keep-alive")
                    self.send_header(
                        "Content-Type",
                        "text/event-stream; charset=utf-8",
                    )
                    self.end_headers()
                    version = -1
                    try:
                        while not feed._closed.is_set():
                            next_version, payload = feed._wait_for_update(version)
                            if payload is None:
                                break
                            if next_version == version:
                                self.wfile.write(b": keepalive\n\n")
                            else:
                                body = json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                ).encode("utf-8")
                                self.wfile.write(b"data: " + body + b"\n\n")
                                version = next_version
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return
                if path != "/latest":
                    self.send_error(404)
                    return
                body = json.dumps(feed.snapshot(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self._cors()
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="screen-decision-feed",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        self._closed.set()
        with self._condition:
            self._condition.notify_all()
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
