"""CLI for extracting a validated hand history from a simulator screenshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.screenshot_history import extract_hand_history, readable_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract visible cards and betting actions from a Hold'em simulator screenshot, "
            "then validate them against the local rules engine."
        )
    )
    parser.add_argument("screenshot", type=Path, help="PNG/JPEG/WebP screenshot to inspect")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="JSON destination (default: <screenshot>.hand-history.json)",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        help="Readable text destination (default: <screenshot>.hand-history.txt)",
    )
    parser.add_argument(
        "--starting-stacks",
        nargs=2,
        type=int,
        metavar=("PLAYER_1", "PLAYER_2"),
        default=(2_000, 2_000),
        help="Starting stacks used for rules validation (default: 2000 2000)",
    )
    parser.add_argument(
        "--timeline-crop",
        help=(
            "Optional left,top,right,bottom timeline region. Values from 0 to 1 are "
            "fractions; larger values are pixels."
        ),
    )
    parser.add_argument(
        "--minimum-card-score",
        type=float,
        default=0.58,
        help="Minimum normalized template score for card recognition (default: 0.58)",
    )
    parser.add_argument(
        "--include-ocr",
        action="store_true",
        help="Include recognized timeline text and coordinates in the JSON output",
    )
    return parser


def output_path(source: Path, requested: Path | None, suffix: str) -> Path:
    return requested or source.with_name(f"{source.stem}.hand-history{suffix}")


def main() -> int:
    arguments = build_parser().parse_args()
    screenshot = arguments.screenshot.resolve()
    if not screenshot.is_file():
        print(f"Screenshot does not exist: {screenshot}", file=sys.stderr)
        return 2
    if not 0.0 <= arguments.minimum_card_score <= 1.0:
        print("--minimum-card-score must be between 0 and 1.", file=sys.stderr)
        return 2

    asset_dir = REPOSITORY_ROOT / "frontend" / "public" / "assets" / "casino-cards"
    try:
        result = extract_hand_history(
            image_path=screenshot,
            asset_dir=asset_dir,
            starting_stacks=arguments.starting_stacks,
            timeline_crop=arguments.timeline_crop,
            minimum_card_score=arguments.minimum_card_score,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    json_path = output_path(screenshot, arguments.json_output, ".json").resolve()
    text_path = output_path(screenshot, arguments.text_output, ".txt").resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result.to_dict(include_ocr=arguments.include_ocr), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    text_path.write_text(readable_text(result), encoding="utf-8")

    ready = result.validation.valid and result.timeline_complete
    status = "valid and complete" if ready else "needs review"
    print(f"Hand history: {status} ({result.confidence:.1%} confidence)")
    print(f"JSON: {json_path}")
    print(f"Text: {text_path}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    return 0 if ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
