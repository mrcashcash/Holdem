"""Continuously reconstruct poker hands from a visible window or screen region."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.screen_history.autoplay import AutoPlaySettings
from backend.screen_history.capture import list_windows, parse_region
from backend.screen_history.decision_feed import LiveDecisionFeed
from backend.screen_history.runtime import RuntimeEvent, RuntimeSettings, WatchRuntime
from backend.screen_history.watcher import (
    CUSTOM_PROFILE_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    calibrate_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a visible poker simulator window, recognize its table state, and "
            "reconstruct uniquely determined hand actions without using the simulator API."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--window-title",
        help="Capture the client area of a visible window containing this title",
    )
    source.add_argument(
        "--region",
        help="Capture explicit desktop coordinates as left,top,width,height",
    )
    source.add_argument(
        "--monitor",
        type=int,
        help="Capture an entire monitor by one-based MSS monitor number",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List visible Windows application titles and exit",
    )
    parser.add_argument(
        "--calibrate",
        type=Path,
        metavar="SCREENSHOT",
        help="Create/update a normalized recognition profile from a screenshot and exit",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile name or JSON path (default: default)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        help="Legacy polling interval; when supplied it overrides --fps",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Maximum capture stream frames per second (default: 15)",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "windows", "mss"),
        default="auto",
        help="Capture backend (default: auto, with MSS fallback)",
    )
    parser.add_argument(
        "--stability-ms",
        type=int,
        default=300,
        help="Milliseconds a changed view must settle before OCR (default: 300)",
    )
    parser.add_argument(
        "--blinds",
        nargs=2,
        type=int,
        metavar=("SMALL", "BIG"),
        default=(10, 20),
        help="Blind sizes used by transition validation (default: 10 20)",
    )
    parser.add_argument(
        "--max-transition-actions",
        type=int,
        default=4,
        help="Maximum actions searched between recognized frames (default: 4)",
    )
    parser.add_argument(
        "--brain-decisions",
        action="store_true",
        help="Feed validated Hero-turn states to the serving brain",
    )
    parser.add_argument(
        "--decision-source",
        choices=("local", "server"),
        default="local",
        help="Read decisions from the local model or running champion server (default: local)",
    )
    parser.add_argument(
        "--decision-server-url",
        default="http://127.0.0.1:8000",
        help="Base URL used by --decision-source server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--decision-feed-port",
        type=int,
        default=8765,
        help="Localhost UI feed port; use 0 to disable it (default: 8765)",
    )
    parser.add_argument(
        "--min-decision-confidence",
        type=float,
        default=85.0,
        metavar="PERCENT",
        help="Minimum recognition confidence for brain decisions (default: 85)",
    )
    parser.add_argument(
        "--auto-play",
        action="store_true",
        help=(
            "Press the poker client's buttons for accepted decisions. Dry run "
            "unless --auto-play-live is also given"
        ),
    )
    parser.add_argument(
        "--auto-play-live",
        action="store_true",
        help="Turn off the auto-play dry run and actually click",
    )
    parser.add_argument(
        "--auto-play-min-confidence",
        type=float,
        default=90.0,
        metavar="PERCENT",
        help="Minimum recognition confidence for a click (default: 90)",
    )
    parser.add_argument(
        "--auto-play-actions",
        default="fold,check,call,raise,all_in",
        help="Comma-separated actions auto-play may press (default: all)",
    )
    parser.add_argument(
        "--auto-play-max-per-hand",
        type=int,
        default=4,
        help="Maximum auto-play clicks in one hand (default: 4)",
    )
    parser.add_argument(
        "--auto-play-max-per-session",
        type=int,
        default=0,
        help="Maximum auto-play clicks before it switches off (default: 0, unlimited)",
    )
    parser.add_argument(
        "--auto-play-delay",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.8, 2.4),
        help="Randomized think time before a click (default: 0.8 2.4; use 0 0 to act at once)",
    )
    parser.add_argument(
        "--auto-play-allow-warnings",
        action="store_true",
        help="Let auto-play click decisions that carry recognition warnings",
    )
    parser.add_argument(
        "--auto-play-no-yield",
        action="store_true",
        help=(
            "Never hand the pointer back. By default auto-play takes the mouse "
            "immediately but gives up if you fight it for more than a second"
        ),
    )
    parser.add_argument(
        "--auto-play-click-method",
        choices=("input", "message"),
        default="input",
        help=(
            "How clicks are delivered: input = synthesized mouse input; "
            "message = posted window messages, which bypass low-level mouse "
            "hooks in clients that ignore injected input (default: input)"
        ),
    )
    parser.add_argument(
        "--auto-play-panic-key",
        default="0x7B",
        help="Virtual-key code that stops auto-play immediately (default: 0x7B, F12)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for live-events.jsonl and completed hand files",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture and recognize one frame, then exit",
    )
    return parser


def _profile_output(requested: str) -> Path:
    path = Path(requested)
    if path.suffix.lower() == ".json":
        return path.resolve()
    return (CUSTOM_PROFILE_DIRECTORY / f"{requested}.json").resolve()


def _print_windows() -> int:
    try:
        windows = list_windows()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for window in windows:
        suffix = " [minimized]" if window.minimized else ""
        print(f"{window.title}{suffix}")
    return 0


def _print_state(state, transition) -> None:
    cards = " ".join(state.hero_cards) or "unknown"
    board = " ".join(state.board) or "-"
    stacks = "/".join("?" if value is None else str(value) for value in state.stacks)
    recognition = (
        f" | vision {state.recognition_ms} ms"
        if state.recognition_ms is not None
        else ""
    )
    print(
        f"Hand #{state.hand_number or '?'} | {state.street or '?'} | pot {state.pot} | "
        f"stacks {stacks} | hero {cards} | board {board} | "
        f"{transition.status}{recognition}"
    )
    if transition.actions:
        for action in transition.actions:
            amount = f" {action.amount}" if action.amount is not None else ""
            print(f"  Player {action.player + 1}: {action.action}{amount} ({action.street})")
    if transition.warning:
        print(f"  Warning: {transition.warning}")
    for warning in state.warnings:
        print(f"  Recognition: {warning}")


def _print_event(event: RuntimeEvent) -> None:
    if event.kind == "state" and event.state is not None and event.transition is not None:
        _print_state(event.state, event.transition)
    elif event.kind in {"started", "fallback", "finalized", "stopped"}:
        print(event.message)
    elif event.kind == "brain_decision" and event.decision is not None:
        amount = f" {event.decision.amount}" if event.decision.amount is not None else ""
        probability = ""
        if event.decision.strategy:
            selected = next(
                (
                    option
                    for option in event.decision.strategy
                    if option.get("action") == event.decision.action
                    and (
                        event.decision.amount is None
                        or option.get("amount") == event.decision.amount
                    )
                ),
                event.decision.strategy[0],
            )
            if selected.get("probability") is not None:
                probability = f", {float(selected['probability']):.1%}"
        latency = (
            f", server {event.decision.latency_ms} ms"
            if event.decision.latency_ms is not None
            else ""
        )
        total_latency = (
            f", live {event.decision.total_latency_ms} ms"
            if event.decision.total_latency_ms is not None
            else ""
        )
        print(
            f"Brain decision: {event.decision.action}{amount} "
            f"({event.decision.model}, confidence "
            f"{event.decision.recognition_confidence:.0%}{probability}"
            f"{latency}{total_latency})"
        )
    elif event.kind == "auto_play":
        status = event.auto_play.status if event.auto_play is not None else "auto-play"
        stream = (
            sys.stderr
            if status in {"aborted", "unconfirmed", "disabled"}
            else sys.stdout
        )
        print(f"Auto-play [{status}]: {event.message}", file=stream)
    elif event.kind in {"brain_ready", "brain_thinking", "brain_stale", "auto_play_ready"}:
        print(event.message)
    elif event.kind in {"brain_skipped", "brain_error"}:
        print(f"Brain: {event.message}", file=sys.stderr)
    elif event.kind in {"recognition_error", "error"}:
        label = "Recognition error" if event.kind == "recognition_error" else "Capture failed"
        print(f"{label}: {event.message}", file=sys.stderr)


def main() -> int:
    for output_stream in (sys.stdout, sys.stderr):
        if hasattr(output_stream, "reconfigure"):
            output_stream.reconfigure(encoding="utf-8", errors="replace")
    arguments = build_parser().parse_args()
    if arguments.list_windows:
        return _print_windows()
    if arguments.interval is not None and arguments.interval <= 0:
        print("--interval must be greater than zero.", file=sys.stderr)
        return 2
    effective_fps = 1.0 / arguments.interval if arguments.interval else arguments.fps
    if not 1 <= effective_fps <= 60:
        print("--fps must be between 1 and 60.", file=sys.stderr)
        return 2
    if not 100 <= arguments.stability_ms <= 2000:
        print("--stability-ms must be between 100 and 2000.", file=sys.stderr)
        return 2
    if any(value <= 0 for value in arguments.blinds) or arguments.blinds[0] >= arguments.blinds[1]:
        print("--blinds requires positive SMALL BIG values with SMALL < BIG.", file=sys.stderr)
        return 2
    if arguments.max_transition_actions <= 0:
        print("--max-transition-actions must be positive.", file=sys.stderr)
        return 2
    if not 0 <= arguments.min_decision_confidence <= 100:
        print("--min-decision-confidence must be between 0 and 100.", file=sys.stderr)
        return 2
    if not 0 <= arguments.decision_feed_port <= 65_535:
        print("--decision-feed-port must be between 0 and 65535.", file=sys.stderr)
        return 2
    if arguments.auto_play and not arguments.brain_decisions:
        print("--auto-play requires --brain-decisions.", file=sys.stderr)
        return 2
    if not 0 <= arguments.auto_play_min_confidence <= 100:
        print("--auto-play-min-confidence must be between 0 and 100.", file=sys.stderr)
        return 2
    try:
        panic_key = int(str(arguments.auto_play_panic_key), 0)
    except ValueError:
        print("--auto-play-panic-key must be a virtual-key code.", file=sys.stderr)
        return 2
    if not 1 <= panic_key <= 0xFF:
        print("--auto-play-panic-key must be between 1 and 255.", file=sys.stderr)
        return 2
    auto_play_actions = tuple(
        part.strip().lower()
        for part in str(arguments.auto_play_actions).split(",")
        if part.strip()
    )

    if arguments.calibrate is not None:
        screenshot = arguments.calibrate.resolve()
        if not screenshot.is_file():
            print(f"Calibration screenshot does not exist: {screenshot}", file=sys.stderr)
            return 2
        name = Path(arguments.profile).stem
        destination = _profile_output(arguments.profile)
        try:
            profile = calibrate_profile(screenshot, destination, name)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"Calibration failed: {exc}", file=sys.stderr)
            return 1
        print(f"Profile '{profile.name}' saved to {destination}")
        return 0

    asset_dir = REPOSITORY_ROOT / "frontend" / "public" / "assets" / "casino-cards"
    try:
        region = parse_region(arguments.region) if arguments.region else None
        settings = RuntimeSettings(
            asset_directory=asset_dir,
            output_directory=arguments.output_dir.resolve(),
            profile=arguments.profile,
            interval=1.0 / effective_fps,
            capture_fps=effective_fps,
            capture_backend=arguments.capture_backend,
            stability_seconds=arguments.stability_ms / 1000.0,
            window_title=arguments.window_title or (
                None if region is not None or arguments.monitor is not None else "Text Hold'em"
            ),
            region=region,
            monitor=arguments.monitor,
            blinds=tuple(arguments.blinds),
            maximum_transition_actions=arguments.max_transition_actions,
            brain_decisions=arguments.brain_decisions,
            minimum_decision_confidence=(
                arguments.min_decision_confidence / 100.0
            ),
            decision_source=arguments.decision_source,
            decision_server_url=arguments.decision_server_url,
            auto_play=AutoPlaySettings(
                enabled=arguments.auto_play,
                dry_run=not arguments.auto_play_live,
                minimum_confidence=arguments.auto_play_min_confidence / 100.0,
                allowed_actions=auto_play_actions,
                maximum_clicks_per_hand=arguments.auto_play_max_per_hand,
                maximum_clicks_per_session=arguments.auto_play_max_per_session,
                minimum_delay_seconds=float(arguments.auto_play_delay[0]),
                maximum_delay_seconds=float(arguments.auto_play_delay[1]),
                allow_warned_decisions=arguments.auto_play_allow_warnings,
                contest_seconds=0.0 if arguments.auto_play_no_yield else 1.0,
                click_method=arguments.auto_play_click_method,
                panic_virtual_key=panic_key,
            ),
        )
        settings.validate()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    feed: LiveDecisionFeed | None = None
    if arguments.brain_decisions and arguments.decision_feed_port:
        profile_name = Path(str(arguments.profile)).stem.lower()
        feed = LiveDecisionFeed(
            port=arguments.decision_feed_port,
            amount_scale=100 if profile_name == "coinpoker" else 1,
        )
        try:
            feed.start()
        except OSError as exc:
            print(f"Could not start the live UI feed: {exc}", file=sys.stderr)
            return 1
        print(
            f"Live decision UI feed: http://127.0.0.1:{arguments.decision_feed_port}/latest"
        )

    def handle_event(event: RuntimeEvent) -> None:
        _print_event(event)
        if feed is not None:
            feed.publish(event)

    runtime = WatchRuntime(settings, callback=handle_event)
    try:
        succeeded = runtime.run(once=arguments.once)
    except KeyboardInterrupt:
        runtime.stop()
        print("Watcher stopped.")
        return 0
    finally:
        if feed is not None:
            feed.stop()
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
