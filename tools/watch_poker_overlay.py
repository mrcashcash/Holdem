"""Run the screen watcher headless with the live inspection overlay only.

This starts the same WatchRuntime the control panel uses, but WITHOUT the Tkinter
control window: a hidden root drives just the red capture border and the
inspection overlay (red boxes + accuracy over each region the recognizer reads)
directly on top of the captured table. Capture source, FPS, profile, blinds and
output all come from the control panel's saved settings
(backend/data/screen_history_gui.json), so it mirrors whatever was last selected
there. Stop with Ctrl+C.
"""

from __future__ import annotations

import json
import queue
import signal
import sys
import threading
import tkinter as tk
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.screen_history.capture import parse_region
from backend.screen_history.gui import (
    ASSET_DIRECTORY,
    GUI_SETTINGS_PATH,
    CaptureBorder,
    InspectionOverlay,
)
from backend.screen_history.runtime import RuntimeEvent, RuntimeSettings, WatchRuntime
from backend.screen_history.watcher import DEFAULT_OUTPUT_DIRECTORY


def _load_settings() -> RuntimeSettings:
    """Build runtime settings from the control panel's saved configuration.

    The inspection overlay is force-enabled here (that is the whole point of this
    launcher); everything else follows whatever the GUI last saved."""

    payload: dict = {}
    if GUI_SETTINGS_PATH.is_file():
        try:
            payload = json.loads(GUI_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}

    source = payload.get("source_mode", "monitor")
    try:
        fps = float(payload.get("capture_fps", 10) or 10)
    except (TypeError, ValueError):
        fps = 10.0
    region = (
        parse_region(payload["region"])
        if source == "region" and payload.get("region")
        else None
    )
    return RuntimeSettings(
        asset_directory=ASSET_DIRECTORY,
        output_directory=Path(
            payload.get("output_directory", str(DEFAULT_OUTPUT_DIRECTORY))
        ).expanduser().resolve(),
        profile=payload.get("profile", "default") or "default",
        interval=1.0 / fps,
        capture_fps=fps,
        stability_seconds=float(payload.get("stability_ms", 300)) / 1000.0,
        capture_backend=payload.get("capture_backend", "auto"),
        blinds=(
            int(payload.get("small_blind", 10)),
            int(payload.get("big_blind", 20)),
        ),
        maximum_transition_actions=int(payload.get("max_actions", 4)),
        brain_decisions=bool(payload.get("brain_decisions", False)),
        minimum_decision_confidence=(
            float(payload.get("minimum_decision_confidence", 85)) / 100.0
        ),
        decision_source=payload.get("decision_source", "local") or "local",
        decision_server_url=payload.get(
            "decision_server_url", "http://127.0.0.1:8000"
        ),
        window_title=payload.get("window_title") if source == "window" else None,
        monitor=int(payload.get("monitor", 1)) if source == "monitor" else None,
        region=region,
        show_inspection_boxes=True,
    )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        settings = _load_settings()
        settings.validate()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    events: queue.Queue[RuntimeEvent] = queue.Queue()
    root = tk.Tk()
    root.withdraw()  # no control panel — the overlay windows are all that show
    capture_border = CaptureBorder(root)
    inspection_overlay = InspectionOverlay(root)
    runtime = WatchRuntime(settings, events.put)
    worker = threading.Thread(target=runtime.run, name="headless-watch", daemon=True)

    amount_scale = 100 if str(settings.profile).lower() == "coinpoker" else 1

    def format_decision(decision: Any) -> str:
        action = decision.action.replace("_", " ").title()
        if decision.amount is not None:
            action += (
                f" {decision.amount / amount_scale:.2f}"
                if amount_scale != 1
                else f" {decision.amount}"
            )
        probability = ""
        for option in decision.strategy or ():
            if option.get("action") == decision.action and (
                decision.amount is None or option.get("amount") == decision.amount
            ):
                value = option.get("probability")
                if value is not None:
                    probability = f"   {float(value):.0%}"
                break
        return action + probability

    def drain_events() -> None:
        try:
            while True:
                event = events.get_nowait()
                if event.kind == "capture" and event.rect is not None:
                    capture_border.show(event.rect)
                elif event.kind == "state":
                    if event.regions and event.rect is not None:
                        inspection_overlay.show(event.rect, event.regions)
                    else:
                        inspection_overlay.hide()
                    # A recommendation only makes sense on Hero's live turn; drop
                    # it the moment the table is no longer waiting for Hero so a
                    # stale action never lingers on screen.
                    if event.state is not None and (
                        event.state.current_player != 0 or event.state.complete
                    ):
                        inspection_overlay.set_decision(None)
                elif event.kind == "brain_thinking":
                    inspection_overlay.set_decision("Thinking…", "#f1c75b")
                elif event.kind == "brain_decision" and event.decision is not None:
                    inspection_overlay.set_decision(
                        format_decision(event.decision), "#8ef16b"
                    )
                    print(f"Brain: {format_decision(event.decision)}")
                elif event.kind in {"brain_skipped", "brain_stale"}:
                    inspection_overlay.set_decision(None)
                elif event.kind == "brain_error":
                    inspection_overlay.set_decision(None)
                    if event.message:
                        print(f"Brain error: {event.message}", file=sys.stderr)
                elif event.kind in {"error", "stopped"}:
                    capture_border.hide()
                    inspection_overlay.hide()
                    if event.message:
                        print(event.message)
                elif event.kind in {"started", "fallback", "finalized", "brain_ready"}:
                    print(event.message)
        except queue.Empty:
            pass
        if not worker.is_alive():
            capture_border.destroy()
            inspection_overlay.destroy()
            root.destroy()
            return
        root.after(80, drain_events)

    def request_stop(*_args) -> None:
        runtime.stop()

    signal.signal(signal.SIGINT, request_stop)
    if settings.window_title is not None:
        source_label = f"window '{settings.window_title}'"
    elif settings.monitor is not None:
        source_label = f"monitor {settings.monitor}"
    else:
        source_label = "the selected region"
    worker.start()
    print(
        f"Watching {source_label} at {settings.capture_fps:g} FPS with the "
        "inspection overlay. Press Ctrl+C to stop."
    )
    root.after(80, drain_events)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        runtime.stop()
    if worker.is_alive():
        worker.join(timeout=10.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
