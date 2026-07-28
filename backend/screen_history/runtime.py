"""Reusable lifecycle for live screen capture and hand reconstruction."""

from __future__ import annotations

import copy
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .autoplay import AutoPlayer, AutoPlayResult, AutoPlaySettings
from .capture import CaptureRect, ScreenCapture
from .decision import (
    BrainDecision,
    BrainDecisionEngine,
    DecisionRequest,
    ServerBrainDecisionEngine,
)
from .stream_capture import FrameStream, StreamFrame, create_frame_stream
from .watcher import (
    LiveHandTracker,
    LiveTableRecognizer,
    ScreenHistoryWriter,
    TransitionResult,
    VisibleTableState,
    load_profile,
    profile_change_score,
)


@dataclass(frozen=True)
class RuntimeSettings:
    """Validated configuration shared by the command-line and GUI watchers."""

    asset_directory: Path
    output_directory: Path
    profile: str | Path = "default"
    interval: float = 1.0
    capture_fps: float = 15.0
    stability_seconds: float = 0.3
    capture_backend: str = "auto"
    blinds: tuple[int, int] = (10, 20)
    maximum_transition_actions: int = 4
    brain_decisions: bool = False
    minimum_decision_confidence: float = 0.85
    decision_source: str = "local"
    decision_server_url: str = "http://127.0.0.1:8000"
    window_title: str | None = None
    region: CaptureRect | None = None
    monitor: int | None = None
    # When true, the recognizer records every region it reads and the runtime
    # ships those boxes (in desktop coordinates, with their accuracy) on each
    # "state" event so the GUI can draw a live red-boxed inspection overlay on
    # top of the captured table. Off by default; there is no GUI control yet, so
    # it is toggled via the saved settings file or in code.
    show_inspection_boxes: bool = False
    # Optional desktop input. Disabled by default, and dry-run by default even
    # once enabled; see backend/screen_history/autoplay.py.
    auto_play: AutoPlaySettings = field(default_factory=AutoPlaySettings)

    def validate(self) -> None:
        selected_sources = sum(
            value is not None for value in (self.window_title, self.region, self.monitor)
        )
        if selected_sources != 1:
            raise ValueError("Choose exactly one capture source: window, monitor, or region.")
        if self.window_title is not None and not self.window_title.strip():
            raise ValueError("Window title must not be empty.")
        if self.monitor is not None and self.monitor <= 0:
            raise ValueError("Monitor number must be positive.")
        if self.interval <= 0:
            raise ValueError("Capture interval must be greater than zero.")
        if not 1.0 <= self.capture_fps <= 60.0:
            raise ValueError("Capture FPS must be between 1 and 60.")
        if not 0.1 <= self.stability_seconds <= 2.0:
            raise ValueError("Stability delay must be between 0.1 and 2 seconds.")
        if self.capture_backend not in {"auto", "windows", "mss"}:
            raise ValueError("Capture backend must be auto, windows, or mss.")
        small_blind, big_blind = self.blinds
        if small_blind <= 0 or big_blind <= 0 or small_blind >= big_blind:
            raise ValueError("Blinds must be positive values with small blind < big blind.")
        if self.maximum_transition_actions <= 0:
            raise ValueError("Maximum transition actions must be positive.")
        if not 0.0 <= self.minimum_decision_confidence <= 1.0:
            raise ValueError("Minimum decision confidence must be between 0 and 1.")
        if self.decision_source not in {"local", "server"}:
            raise ValueError("Decision source must be local or server.")
        if self.decision_source == "server" and not self.decision_server_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("Decision server URL must begin with http:// or https://.")
        self.auto_play.validate()
        if self.auto_play.enabled and not self.brain_decisions:
            raise ValueError("Auto-play needs brain decisions to be enabled.")
        if self.auto_play.enabled and self.auto_play.minimum_confidence < (
            self.minimum_decision_confidence
        ):
            raise ValueError(
                "Auto-play confidence must not be below the decision confidence minimum."
            )
        if not self.asset_directory.is_dir():
            raise FileNotFoundError(f"Card asset directory was not found: {self.asset_directory}")

    def capture(self) -> ScreenCapture:
        return ScreenCapture(
            window_title=self.window_title,
            region=self.region,
            monitor=self.monitor,
        )


@dataclass(frozen=True)
class RuntimeEvent:
    """Thread-safe status update emitted by :class:`WatchRuntime`."""

    kind: str
    message: str = ""
    state: VisibleTableState | None = None
    transition: TransitionResult | None = None
    frame: Any = None
    paths: tuple[Path, Path] | None = None
    rect: CaptureRect | None = None
    pending_frames: int = 0
    stream_fps: float = 0.0
    backend: str = ""
    capture_count: int = 0
    decision: BrainDecision | None = None
    auto_play: AutoPlayResult | None = None
    # Inspected regions for the live overlay: (label, left, top, right, bottom,
    # confidence) in desktop pixel coordinates. Empty unless
    # RuntimeSettings.show_inspection_boxes is enabled.
    regions: tuple[tuple[str, float, float, float, float, float], ...] = ()
    # Which recognition path the frame took: "fast", "slow — <reason>", or "".
    recognition_path: str = ""


EventCallback = Callable[[RuntimeEvent], None]


def capture_preview(settings: RuntimeSettings) -> tuple[Any, VisibleTableState]:
    """Capture and recognize one frame without creating history output."""

    settings.validate()
    profile = load_profile(settings.profile)
    recognizer = LiveTableRecognizer(
        profile,
        settings.asset_directory,
        collect_inspection_boxes=settings.show_inspection_boxes,
    )
    stream = create_frame_stream(
        backend=settings.capture_backend,
        fps=settings.capture_fps,
        window_title=settings.window_title,
        region=settings.region,
        monitor=settings.monitor,
    )
    captured: StreamFrame | None = None
    try:
        for attempt in range(2):
            try:
                stream.start()
                deadline = time.monotonic() + 5.0
                while captured is None and time.monotonic() < deadline:
                    captured = stream.next_frame(timeout=0.1)
                if captured is None:
                    raise RuntimeError(
                        "The capture stream did not produce a frame within five seconds."
                    )
                break
            except (RuntimeError, ValueError, OSError):
                if (
                    attempt > 0
                    or settings.capture_backend != "auto"
                    or stream.backend_name == "MSS fallback"
                ):
                    raise
                stream.stop()
                stream.wait()
                stream = create_frame_stream(
                    backend="mss",
                    fps=settings.capture_fps,
                    window_title=settings.window_title,
                    region=settings.region,
                    monitor=settings.monitor,
                )
    finally:
        stream.stop()
        stream.wait()
    assert captured is not None
    recognition_started = time.monotonic()
    state = recognizer.recognize(captured.image, captured.rect)
    state = replace(
        state,
        captured_at=captured.captured_at,
        recognition_ms=int(round((time.monotonic() - recognition_started) * 1000)),
    )
    return captured.image, state


class WatchRuntime:
    """Run screen recognition until stopped, emitting UI-neutral events."""

    def __init__(
        self,
        settings: RuntimeSettings,
        callback: EventCallback | None = None,
    ) -> None:
        self.settings = settings
        self.callback = callback or (lambda _event: None)
        self._stop_event = threading.Event()
        self.running = False

    def stop(self) -> None:
        self._stop_event.set()

    def _emit(self, kind: str, **values: Any) -> None:
        self.callback(RuntimeEvent(kind=kind, **values))

    def run(self, *, once: bool = False) -> bool:
        """Run synchronously in the caller's thread; return whether setup succeeded."""

        tracker: LiveHandTracker | None = None
        writer: ScreenHistoryWriter | None = None
        stream: FrameStream | None = None
        failure_reason: str | None = None
        self.running = True
        try:
            self.settings.validate()
            profile = load_profile(self.settings.profile)
            settle_seconds = (
                min(self.settings.stability_seconds, 0.025)
                if profile.name.lower() == "coinpoker"
                else self.settings.stability_seconds
            )
            recognizer = LiveTableRecognizer(
                profile,
                self.settings.asset_directory,
                collect_inspection_boxes=self.settings.show_inspection_boxes,
            )
            recognizer.warm_up()
            writer = ScreenHistoryWriter(self.settings.output_directory.resolve())
            tracker = LiveHandTracker(
                writer,
                blinds=self.settings.blinds,
                maximum_transition_actions=self.settings.maximum_transition_actions,
            )
            # Preserve stable CoinPoker milestones in capture order. The
            # decision worker still keeps only its newest request, so retaining
            # hand-history states does not put old recommendations back on the
            # latency path.
            frame_queue: queue.Queue[Any] = queue.Queue(
                maxsize=16 if profile.name.lower() == "coinpoker" else 32
            )
            queue_sentinel = object()
            processing_failure: list[str] = []
            tracker_lock = threading.RLock()
            decision_queue: queue.Queue[Any] | None = (
                queue.Queue(maxsize=1) if self.settings.brain_decisions else None
            )
            decision_sentinel = object()
            auto_player: AutoPlayer | None = None
            if self.settings.auto_play.enabled:

                def announce_auto_play(result: AutoPlayResult) -> None:
                    writer.event("auto_play", auto_play=result.payload())
                    self._emit("auto_play", message=result.message, auto_play=result)

                auto_player = AutoPlayer(
                    self.settings.auto_play,
                    controls=profile.action_controls or None,
                    notify=announce_auto_play,
                )
            # CoinPoker chips are tracked in hundredths; a typed bet has to go
            # back to the decimals the client shows.
            auto_play_scale = 100 if profile.name.lower() == "coinpoker" else 1
            latest_state_lock = threading.Lock()
            # The desktop window the recognizer is reading, published from the
            # recognition thread for the decision thread's auto-play clicks.
            latest_table_window: list[Any] = [None]
            handled_decision_states: dict[tuple[Any, ...], float] = {}
            requested_decision_states: set[tuple[Any, ...]] = set()
            # Last temporary skip reported per spot, so retrying a spot every
            # frame does not repeat the same line in the log.
            pending_skip_reasons: dict[tuple[Any, ...], str] = {}

            def action_signature(
                state: VisibleTableState,
            ) -> tuple[tuple[int, str, int | None, str], ...]:
                return tuple(
                    (
                        int(action.player),
                        str(action.action),
                        int(action.amount) if action.amount is not None else None,
                        str(action.street),
                    )
                    for action in state.visible_actions
                )

            def state_identity(state: VisibleTableState) -> tuple[Any, ...]:
                # Only fields that change the validated poker decision belong
                # here. Raw pot/stack OCR, confidence, and transient stability
                # flags can fluctuate while Hero is still facing the exact
                # same action.
                return (
                    state.hand_number,
                    state.street,
                    state.hero_cards,
                    state.board,
                    state.button,
                    state.current_player,
                    state.complete,
                    action_signature(state),
                )

            def spot_reason(request: DecisionRequest) -> str | None:
                """Has the spot this decision was made for actually moved on?

                Asks the tracked rules engine, which only advances on validated
                transitions. Comparing recognized frames instead looks the same
                most of the time and then throws away good answers: a transient
                frame carries no cards and no acting player, so it reads as a
                completely different spot while Hero is still sitting on the
                very same decision.
                """

                hand = request.hand_number
                acted = len(request.game.public_actions)
                with tracker_lock:
                    tracked = tracker.current
                    if tracked is None or tracked.engine is None:
                        return "the hand is no longer being tracked"
                    if (
                        hand is not None
                        and tracked.hand_number is not None
                        and tracked.hand_number != hand
                    ):
                        return f"hand #{tracked.hand_number} is now in play, not #{hand}"
                    game = tracked.engine
                    if game.hand_complete:
                        return "the hand finished"
                    if game.current_player != 0:
                        return "it is no longer Hero's turn"
                    now_acted = len(game.public_actions)
                    if now_acted != acted:
                        return (
                            f"the action history moved from {acted} to "
                            f"{now_acted} actions"
                        )
                return None

            def record_decision_skip(state: VisibleTableState, reason: str) -> None:
                writer.event(
                    "brain_skipped",
                    hand_number=state.hand_number,
                    reason=reason,
                    recognition_confidence=state.confidence,
                )
                self._emit("brain_skipped", message=reason, state=state)

            def queue_brain_decision(
                state: VisibleTableState,
                transition: TransitionResult,
            ) -> None:
                if decision_queue is None:
                    return
                identity = state_identity(state)
                if identity in requested_decision_states:
                    return
                previous_confidence = handled_decision_states.get(identity)
                if (
                    previous_confidence is not None
                    and state.confidence <= previous_confidence
                ):
                    return

                reason: str | None = None
                # A reason drawn from the state's own identity can never change
                # for that identity, so it is final. Anything else — the tracker
                # still catching up, an ambiguous transition, a transient read —
                # is temporary, and the same spot must stay eligible.
                final = True
                if not state.decision_ready:
                    reason = "The recognized frame is transient."
                    final = False
                elif state.current_player != 0:
                    reason = "The recognized table is not waiting for Hero."
                elif state.complete:
                    reason = "The hand is already complete."
                elif len(state.hero_cards) != 2:
                    reason = "Two Hero cards were not recognized."
                elif state.confidence < self.settings.minimum_decision_confidence:
                    reason = (
                        f"Recognition confidence {state.confidence:.0%} is below the "
                        f"{self.settings.minimum_decision_confidence:.0%} decision minimum."
                    )
                elif transition.status in {"ambiguous", "unmatched", "untracked", "transient"}:
                    reason = f"The rules transition is {transition.status}."
                    final = False

                with tracker_lock:
                    tracked = tracker.current
                    if reason is None and (tracked is None or tracked.engine is None):
                        # Normal at the very start of a hand: the tracker has not
                        # rebuilt the engine yet. It arrives a frame or two later
                        # — as long as this spot is not written off in the
                        # meantime, which is what used to lose the whole preflop.
                        reason = "No complete rules-engine state is available."
                        final = False
                    if reason is None and tracked is not None and tracked.engine is not None:
                        game = copy.deepcopy(tracked.engine)
                    else:
                        game = None

                def note_skip(message: str, retryable: bool) -> None:
                    if not retryable:
                        handled_decision_states[identity] = state.confidence
                    # Report a temporary reason once per spot; retrying every
                    # frame must not fill the log with the same line.
                    if retryable and pending_skip_reasons.get(identity) == message:
                        return
                    pending_skip_reasons[identity] = message
                    record_decision_skip(state, message)

                if reason is not None or game is None:
                    note_skip(
                        reason or "No validated decision state is available.",
                        not final,
                    )
                    return
                if game.current_player != 0 or game.hand_complete:
                    note_skip(
                        "The reconstructed rules engine is not waiting for Hero.",
                        True,
                    )
                    return
                handled_decision_states[identity] = state.confidence

                request = DecisionRequest(
                    game=game,
                    hand_number=state.hand_number,
                    captured_at=state.captured_at,
                    recognition_confidence=state.confidence,
                    recognition_ms=state.recognition_ms,
                    state_key=identity,
                    action_signature=action_signature(state),
                )
                requested_decision_states.add(identity)
                try:
                    decision_queue.put_nowait(request)
                except queue.Full:
                    try:
                        replaced = decision_queue.get_nowait()
                        decision_queue.task_done()
                    except queue.Empty:
                        replaced = None
                    if isinstance(replaced, DecisionRequest):
                        writer.event(
                            "brain_stale",
                            hand_number=replaced.hand_number,
                            reason="A newer validated table state replaced the queued decision.",
                        )
                    decision_queue.put_nowait(request)
                writer.event(
                    "brain_requested",
                    hand_number=state.hand_number,
                    recognition_confidence=state.confidence,
                )
                self._emit(
                    "brain_thinking",
                    message=f"Brain evaluating hand #{state.hand_number or '?'}.",
                    state=state,
                )

            def process_decisions() -> None:
                assert decision_queue is not None
                try:
                    try:
                        if self.settings.decision_source == "server":
                            engine = ServerBrainDecisionEngine(
                                self.settings.decision_server_url
                            )
                        else:
                            engine = BrainDecisionEngine()
                    except Exception as exc:
                        message = f"Brain unavailable: {exc}"
                        writer.event("brain_unavailable", error=str(exc))
                        self._emit("brain_error", message=message)
                        while True:
                            queued = decision_queue.get()
                            decision_queue.task_done()
                            if queued is decision_sentinel:
                                return
                    self._emit(
                        "brain_ready",
                        message=f"Brain ready: {engine.model_name}.",
                    )
                    writer.event("brain_ready", model=engine.model_name)
                    while True:
                        queued = decision_queue.get()
                        try:
                            if queued is decision_sentinel:
                                return
                            assert isinstance(queued, DecisionRequest)
                            try:
                                decision = engine.decide(queued)
                                advanced = spot_reason(queued)
                                if advanced is not None:
                                    message = (
                                        "Brain result discarded because the table "
                                        f"advanced: {advanced}."
                                    )
                                    writer.event(
                                        "brain_stale",
                                        hand_number=queued.hand_number,
                                        decision=decision.payload(),
                                        reason=message,
                                    )
                                    self._emit("brain_stale", message=message)
                                    continue
                                payload = decision.payload()
                                with tracker_lock:
                                    attached = tracker.add_decision(
                                        queued.hand_number,
                                        payload,
                                    )
                                if not attached:
                                    message = (
                                        "Brain result discarded because its hand is no longer active."
                                    )
                                    writer.event(
                                        "brain_stale",
                                        hand_number=queued.hand_number,
                                        decision=payload,
                                        reason=message,
                                    )
                                    self._emit("brain_stale", message=message)
                                    continue
                                writer.event("brain_decision", decision=payload)
                                self._emit(
                                    "brain_decision",
                                    message=(
                                        f"Brain: {decision.action}"
                                        f"{f' {decision.amount}' if decision.amount is not None else ''}"
                                    ),
                                    decision=decision,
                                )
                                if auto_player is not None:
                                    # Only here: the decision is fresh, attached
                                    # to the live hand, and already re-validated
                                    # against a copy of the rules engine. The
                                    # same spot check that gated it above keeps
                                    # gating it right up to the click.
                                    def spot_changed(
                                        request: DecisionRequest = queued,
                                    ) -> str | None:
                                        return spot_reason(request)

                                    with latest_state_lock:
                                        table_window = latest_table_window[0]
                                    result = auto_player.execute(
                                        decision,
                                        amount_scale=auto_play_scale,
                                        spot_changed=spot_changed,
                                        table_window=table_window,
                                    )
                                    writer.event(
                                        "auto_play",
                                        hand_number=queued.hand_number,
                                        auto_play=result.payload(),
                                    )
                                    self._emit(
                                        "auto_play",
                                        message=result.message,
                                        auto_play=result,
                                        decision=decision,
                                    )
                            except (RuntimeError, ValueError, OSError) as exc:
                                writer.event(
                                    "brain_error",
                                    hand_number=queued.hand_number,
                                    error=str(exc),
                                )
                                self._emit("brain_error", message=str(exc))
                        finally:
                            decision_queue.task_done()
                except Exception as exc:
                    writer.event("brain_error", error=str(exc))
                    self._emit("brain_error", message=f"Brain worker failed: {exc}")

            coalesce_frames = profile.name.lower() == "coinpoker"
            # Idle backoff: when no hand has been seen for a while (an empty /
            # between-hands table that only shimmers), stop hammering the slow
            # recognizer on every animation frame — require a much larger change
            # to wake, and slow the heartbeat. Recognition thread stamps
            # hand_activity whenever it sees a real hand; the capture loop reads
            # it. A single-element list so the closure can mutate it.
            hand_activity = [time.monotonic()]
            idle_after_seconds = 2.5

            def process_frames() -> None:
                try:
                    while True:
                        queued = frame_queue.get()
                        drained_sentinel = False
                        try:
                            if queued is queue_sentinel:
                                return
                            # Coalesce a backlog: the CoinPoker Dealer Chat is
                            # cumulative, so an older queued frame's actions are
                            # all present in the newest one. When recognition
                            # falls behind (e.g. the slow full-frame OCR path),
                            # skip straight to the freshest frame instead of
                            # grinding through 16 stale ones — this keeps latency
                            # low and empties the "awaiting OCR" queue.
                            if coalesce_frames:
                                while True:
                                    try:
                                        newer = frame_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                    if newer is queue_sentinel:
                                        frame_queue.task_done()
                                        drained_sentinel = True
                                        break
                                    frame_queue.task_done()
                                    queued = newer
                            assert isinstance(queued, StreamFrame)
                            try:
                                recognition_started = time.monotonic()
                                state = recognizer.recognize(
                                    queued.image,
                                    queued.rect,
                                )
                                state = replace(
                                    state,
                                    captured_at=queued.captured_at,
                                    recognition_ms=int(
                                        round(
                                            (
                                                time.monotonic()
                                                - recognition_started
                                            )
                                            * 1000
                                        )
                                    ),
                                )
                                with tracker_lock:
                                    transition = tracker.observe(state)
                                with latest_state_lock:
                                    if recognizer.last_table_window is not None:
                                        latest_table_window[0] = (
                                            recognizer.last_table_window
                                        )
                                # Stamp activity whenever this frame looks like a
                                # real hand, so the capture loop keeps sampling
                                # at full rate during play and only backs off
                                # once the table has genuinely gone quiet.
                                if (
                                    state.history_stable
                                    or state.visible_actions
                                    or state.hero_cards
                                    or state.board
                                ):
                                    hand_activity[0] = time.monotonic()
                                regions: tuple[
                                    tuple[str, float, float, float, float, float],
                                    ...,
                                ] = ()
                                if (
                                    self.settings.show_inspection_boxes
                                    and queued.rect is not None
                                ):
                                    origin_left = queued.rect.left
                                    origin_top = queued.rect.top
                                    regions = tuple(
                                        (
                                            box.label,
                                            origin_left + box.left,
                                            origin_top + box.top,
                                            origin_left + box.right,
                                            origin_top + box.bottom,
                                            box.confidence,
                                        )
                                        for box in recognizer.last_inspection_boxes
                                    )
                                self._emit(
                                    "state",
                                    state=state,
                                    transition=transition,
                                    frame=queued.image,
                                    pending_frames=frame_queue.qsize(),
                                    rect=queued.rect,
                                    regions=regions,
                                    recognition_path=recognizer.last_recognition_path,
                                )
                                if auto_player is not None:
                                    # The validated history is the only proof a
                                    # click landed; an unconfirmed one turns
                                    # auto-play off rather than clicking again.
                                    confirmation = auto_player.confirm_from_state(state)
                                    if confirmation is not None:
                                        writer.event(
                                            "auto_play",
                                            hand_number=state.hand_number,
                                            auto_play=confirmation.payload(),
                                        )
                                        self._emit(
                                            "auto_play",
                                            message=confirmation.message,
                                            auto_play=confirmation,
                                            state=state,
                                        )
                                queue_brain_decision(state, transition)
                            except (RuntimeError, ValueError, OSError) as exc:
                                writer.event("recognition_error", error=str(exc))
                                self._emit("recognition_error", message=str(exc))
                        finally:
                            frame_queue.task_done()
                        if drained_sentinel:
                            return
                except Exception as exc:  # keep capture/UI alive long enough to report failure
                    processing_failure.append(str(exc))
                    self._stop_event.set()

            stream = create_frame_stream(
                backend=self.settings.capture_backend,
                fps=self.settings.capture_fps,
                window_title=self.settings.window_title,
                region=self.settings.region,
                monitor=self.settings.monitor,
            )
            try:
                stream.start()
            except (RuntimeError, ValueError, OSError) as exc:
                if self.settings.capture_backend != "auto" or stream.backend_name == "MSS fallback":
                    raise
                self._emit(
                    "fallback",
                    message=f"Windows capture unavailable ({exc}); using MSS fallback.",
                )
                stream = create_frame_stream(
                    backend="mss",
                    fps=self.settings.capture_fps,
                    window_title=self.settings.window_title,
                    region=self.settings.region,
                    monitor=self.settings.monitor,
                )
                stream.start()

            last_milestone: StreamFrame | None = None
            candidate: StreamFrame | None = None
            candidate_last_motion = 0.0
            capture_count = 0
            frame_times: deque[float] = deque()
            last_status_emit = 0.0
            first_frame_deadline = time.monotonic() + 3.0
            settle_threshold = max(0.35, profile.pixel_change_threshold * 0.28)
            # Heartbeat: even when pixel-change detection sees nothing, re-run
            # recognition on the current frame every heartbeat_seconds so the
            # cumulative Dealer-Chat timeline is always re-read — a change that
            # slipped under the motion threshold can never hide an action. Cheap
            # thanks to the region/rank caches (unchanged frame ≈ a few ms). Only
            # for CoinPoker, whose chat is cumulative; 0 disables it.
            heartbeat_seconds = 1.0 if profile.name.lower() == "coinpoker" else 0.0
            last_milestone_at = time.monotonic()

            def queue_milestone(selected: StreamFrame) -> bool:
                nonlocal last_milestone, last_milestone_at
                try:
                    frame_queue.put_nowait(selected)
                    last_milestone = selected
                    last_milestone_at = time.monotonic()
                    return True
                except queue.Full:
                    message = (
                        "Recognition milestone queue is full; a stable state was dropped. "
                        "The live history was marked with an explicit gap."
                    )
                    writer.event("frame_queue_overflow", error=message)
                    self._emit("history_gap", message=message)
                    return False

            self._emit(
                "started",
                message=(
                    f"Streaming {stream.description()} at up to {self.settings.capture_fps:g} FPS "
                    f"using {stream.backend_name}, profile '{profile.name}'. "
                    f"Output: {writer.output_directory}"
                ),
                backend=stream.backend_name,
            )
            if auto_player is not None:
                auto_player.start()
                writer.event("auto_play_ready", detail=auto_player.describe())
                self._emit("auto_play_ready", message=auto_player.describe())
            recognition_thread = threading.Thread(
                target=process_frames,
                name="screen-history-recognition",
                daemon=True,
            )
            decision_thread: threading.Thread | None = None
            if decision_queue is not None:
                decision_thread = threading.Thread(
                    target=process_decisions,
                    name="screen-history-brain",
                    daemon=True,
                )
                decision_thread.start()
            recognition_thread.start()
            try:
                while not self._stop_event.is_set():
                    try:
                        frame = stream.next_frame(timeout=0.05)
                    except RuntimeError as exc:
                        if (
                            self.settings.capture_backend == "auto"
                            and stream.backend_name != "MSS fallback"
                        ):
                            stream.stop()
                            stream.wait()
                            self._emit(
                                "fallback",
                                message=(
                                    f"Windows capture stream failed ({exc}); "
                                    "continuing with MSS fallback."
                                ),
                            )
                            stream = create_frame_stream(
                                backend="mss",
                                fps=self.settings.capture_fps,
                                window_title=self.settings.window_title,
                                region=self.settings.region,
                                monitor=self.settings.monitor,
                            )
                            stream.start()
                            frame_times.clear()
                            first_frame_deadline = time.monotonic() + 3.0
                            continue
                        raise
                    now = time.monotonic()
                    if (
                        frame is None
                        and capture_count == 0
                        and now >= first_frame_deadline
                    ):
                        if (
                            self.settings.capture_backend == "auto"
                            and stream.backend_name != "MSS fallback"
                        ):
                            stream.stop()
                            stream.wait()
                            self._emit(
                                "fallback",
                                message=(
                                    "Windows capture produced no frames; continuing with "
                                    "MSS fallback."
                                ),
                            )
                            stream = create_frame_stream(
                                backend="mss",
                                fps=self.settings.capture_fps,
                                window_title=self.settings.window_title,
                                region=self.settings.region,
                                monitor=self.settings.monitor,
                            )
                            stream.start()
                            frame_times.clear()
                            first_frame_deadline = time.monotonic() + 3.0
                            continue
                        raise RuntimeError(
                            "The capture stream did not produce a frame within three seconds."
                        )
                    if frame is not None:
                        capture_count += 1
                        frame_times.append(frame.captured_monotonic)
                        while frame_times and now - frame_times[0] > 1.0:
                            frame_times.popleft()

                        if once:
                            candidate = frame
                            candidate_last_motion = now - settle_seconds
                        elif last_milestone is None:
                            if candidate is None:
                                candidate_last_motion = now
                            elif (
                                profile_change_score(candidate.image, frame.image, profile)
                                >= settle_threshold
                            ):
                                candidate_last_motion = now
                            candidate = frame
                        else:
                            milestone_change = profile_change_score(
                                last_milestone.image,
                                frame.image,
                                profile,
                            )
                            # When the table has gone idle (no hand for a while),
                            # demand a much bigger change to re-recognize, so the
                            # felt shimmer / avatar idle animation of an empty
                            # table no longer triggers constant slow OCR. A real
                            # new hand (cards dealt, seats fill) far exceeds this.
                            idle = (now - hand_activity[0]) > idle_after_seconds
                            wake_threshold = profile.pixel_change_threshold * (
                                4.0 if idle else 1.0
                            )
                            if milestone_change >= wake_threshold:
                                # Pixel motion used to invalidate any in-flight
                                # recommendation here. It cost real decisions:
                                # dealing the flop moves plenty of pixels while
                                # Hero still faces the very same spot, so a good
                                # answer was thrown away as stale. Whether the
                                # spot actually moved is now decided by the
                                # tracked rules engine (see spot_reason).
                                if candidate is None:
                                    candidate_last_motion = now
                                elif (
                                    profile_change_score(candidate.image, frame.image, profile)
                                    >= settle_threshold
                                ):
                                    candidate_last_motion = now
                                candidate = frame
                            elif candidate is not None:
                                candidate = None

                    if (
                        candidate is not None
                        and now - candidate_last_motion >= settle_seconds
                    ):
                        queue_milestone(candidate)
                        candidate = None
                        if once:
                            break

                    # Heartbeat re-read: nothing is settling and the table has
                    # been static past the interval — re-recognize the current
                    # frame so the cumulative chat is refreshed and no action can
                    # be silently missed. Skipped until the first real milestone.
                    # Much slower while idle (empty table): a relaxed check for a
                    # new hand rather than a busy every-second re-scan.
                    heartbeat_interval = heartbeat_seconds * (
                        5.0 if (now - hand_activity[0]) > idle_after_seconds else 1.0
                    )
                    if (
                        heartbeat_seconds
                        and not once
                        and frame is not None
                        and candidate is None
                        and last_milestone is not None
                        and now - last_milestone_at >= heartbeat_interval
                    ):
                        queue_milestone(frame)

                    if frame is not None and now - last_status_emit >= 0.2:
                        self._emit(
                            "capture",
                            message=f"Frame {capture_count}",
                            rect=frame.rect,
                            pending_frames=frame_queue.qsize(),
                            stream_fps=float(len(frame_times)),
                            backend=stream.backend_name,
                            capture_count=capture_count,
                        )
                        last_status_emit = now
            finally:
                stream.stop()
                stream.wait()
                stream = None
                if candidate is not None:
                    queue_milestone(candidate)
                if recognition_thread.is_alive():
                    frame_queue.put(queue_sentinel)
                    recognition_thread.join()
                if (
                    decision_queue is not None
                    and decision_thread is not None
                    and decision_thread.is_alive()
                ):
                    decision_queue.put(decision_sentinel)
                    decision_thread.join()
                if auto_player is not None:
                    auto_player.stop()
            if processing_failure:
                raise RuntimeError(f"Recognition worker failed: {processing_failure[0]}")
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            failure_reason = str(exc)
            if writer is not None:
                writer.event("capture_error", error=failure_reason)
            self._emit("error", message=failure_reason)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.wait()
                except (RuntimeError, OSError):
                    pass
            if tracker is not None:
                reason = (
                    "single capture requested"
                    if once
                    else "capture failed" if failure_reason else "watcher stopped"
                )
                if "tracker_lock" in locals():
                    with tracker_lock:
                        paths = tracker.finalize(reason)
                else:
                    paths = tracker.finalize(reason)
                if paths is not None:
                    self._emit("finalized", message=f"Saved {paths[0].name}", paths=paths)
            self.running = False
            self._emit("stopped", message="Watcher stopped.")
        return failure_reason is None
