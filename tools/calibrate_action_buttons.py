"""Check (or re-derive) the auto-play click regions for the poker client.

Auto-play never clicks a stored coordinate: it OCRs the action strip and clicks
the button whose text matches the intended action. The stored regions only say
*where to look*. This tool shows exactly what the watcher would see there, so a
layout change is caught before a hand is played rather than during one.

Arrange the client with Hero facing a decision (the Fold / Check / Call / Bet
buttons visible), then run:

    python tools/calibrate_action_buttons.py

Use --scan when the buttons are not inside the current strip: it OCRs the whole
bottom of the window and prints normalized bounds to paste into the profile's
"action_controls".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.screen_history.autoplay import (
    ACTION_PATTERNS,
    DEFAULT_ACTION_CONTROLS,
    AutoPlaySettings,
    _median_saturation,
    list_table_candidates,
    ocr_controls,
    region_rect,
)
from backend.screen_history.capture import CaptureRect, ScreenCapture
from backend.screen_history.watcher import load_profile


def _normalized(window: CaptureRect, left: float, top: float, right: float, bottom: float) -> str:
    return (
        f"[{(left - window.left) / window.width:.3f}, "
        f"{(top - window.top) / window.height:.3f}, "
        f"{(right - window.left) / window.width:.3f}, "
        f"{(bottom - window.top) / window.height:.3f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="coinpoker",
        help="Profile whose action_controls are checked (default: coinpoker)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        metavar="N",
        help="Which candidate window to inspect (default: 1, the best guess)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="OCR the bottom third of the window instead of the stored strip",
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        help="Write the captured region to this path for inspection",
    )
    arguments = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = AutoPlaySettings(enabled=True, dry_run=True)
    candidates = list_table_candidates(settings)
    if not candidates:
        print(
            "No visible poker table window was found "
            f"(looking for {', '.join(settings.table_window_keywords)}).",
            file=sys.stderr,
        )
        return 1
    # CoinPoker titles its lobby and its tables identically, so this pick is a
    # guess. Always show the alternatives and how to select one.
    if len(candidates) > 1:
        print("Candidate windows (--window N to pick another):")
        for index, (named, _area, other, other_rect) in enumerate(candidates, start=1):
            tag = " [title names a game]" if named else ""
            print(
                f"  {index}. \"{other.title}\" {other_rect.width}x{other_rect.height} "
                f"at {other_rect.left},{other_rect.top}{tag}"
            )
    choice = arguments.window
    if not 1 <= choice <= len(candidates):
        print(f"--window must be between 1 and {len(candidates)}.", file=sys.stderr)
        return 2
    window, rect = candidates[choice - 1][2], candidates[choice - 1][3]
    print(f'\nUsing window {choice}: "{window.title}"')
    print(f"  desktop rect: {rect.left},{rect.top} {rect.width}x{rect.height}")

    try:
        controls = dict(load_profile(arguments.profile).action_controls)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Could not load profile: {exc}", file=sys.stderr)
        return 1
    if not controls:
        controls = dict(DEFAULT_ACTION_CONTROLS)
        print(f"  profile '{arguments.profile}' has no action_controls; using defaults")

    bounds = (0.0, 0.66, 1.0, 1.0) if arguments.scan else controls["button_strip"]
    strip = region_rect(rect, bounds)
    print(
        f"  {'scan area' if arguments.scan else 'button strip'}: "
        f"{list(round(value, 3) for value in bounds)} -> "
        f"{strip.left},{strip.top} {strip.width}x{strip.height}"
    )

    with ScreenCapture(region=strip) as capture:
        image = capture.grab(strip)
    if arguments.save_image is not None:
        import cv2  # type: ignore

        cv2.imwrite(str(arguments.save_image), image)
        print(f"  saved {arguments.save_image}")

    # The same upscaled read auto-play performs, so this tool reports exactly
    # what the watcher will see rather than a more pessimistic raw OCR pass.
    lines = ocr_controls(image)
    if not lines:
        print("\nNo text was recognized. Is Hero actually facing a decision?")
        return 3

    print("\nRecognized controls (normalized bounds are relative to the window):")
    found: set[str] = set()
    for line in lines:
        text = str(line.text).strip()
        upper = text.upper()
        actions = sorted(
            name for name, pattern in ACTION_PATTERNS.items() if pattern.search(upper)
        )
        found.update(actions)
        saturation, value = _median_saturation(image, line.box)
        button = "button" if saturation >= 60 and value >= 60 else "flat  "
        print(
            f"  {text:<18} {button} sat {saturation:5.0f} val {value:5.0f}  "
            f"click {strip.left + int(line.box.center_x)},"
            f"{strip.top + int(line.box.center_y)}  "
            + _normalized(
                rect,
                strip.left + line.box.left,
                strip.top + line.box.top,
                strip.left + line.box.right,
                strip.top + line.box.bottom,
            )
            + (f"  <- {'/'.join(actions)}" if actions else "")
        )

    missing = {"fold", "check", "call", "raise"} - found
    print()
    if found:
        print(f"Actions clickable right now: {', '.join(sorted(found))}")
    if missing:
        print(
            "Not visible in this frame: "
            f"{', '.join(sorted(missing))} (normal — the client only shows the legal ones)"
        )
    if not found:
        print(
            "No action button was matched. Re-run with --scan and copy the bounds of "
            "the row holding Fold/Check/Call into the profile's "
            '"action_controls" -> "button_strip".'
        )
        return 3
    print(
        "\nAmount field currently read from "
        f"{list(round(value, 3) for value in controls.get('amount_field', ()))}; "
        "with --scan, use the bounds printed for the bet-size box."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
