"""Extract a best-effort Hold'em hand history from simulator screenshots.

The extractor deliberately separates recognition from validation. OCR and card
matching may be uncertain; the existing rules engine is the authority for turn
order, legal amounts, street transitions, and the resulting pot. Invisible
actions are never synthesized.
"""

from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from ..poker import Card, HeadsUpHoldem, InvalidAction, RANK_LABELS, new_deck


SUIT_CODES = {"s": "\u2660", "h": "\u2665", "d": "\u2666", "c": "\u2663"}
SUIT_ALIASES = {
    "\u2660": "\u2660",
    "\u2665": "\u2665",
    "\u2666": "\u2666",
    "\u2663": "\u2663",
    "s": "\u2660",
    "h": "\u2665",
    "d": "\u2666",
    "c": "\u2663",
}
RANK_VALUES = {label: rank for rank, label in RANK_LABELS.items()}
_OCR_LOCK = threading.RLock()


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    def intersects(self, other: "Box") -> bool:
        return not (
            self.right < other.left
            or self.left > other.right
            or self.bottom < other.top
            or self.top > other.bottom
        )


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class DetectedCard:
    card: str
    confidence: float
    margin: float
    box: Box


@dataclass(frozen=True)
class ParsedAction:
    player: int
    action: str
    amount: int | None
    street: str
    confidence: float
    raw_text: str


@dataclass
class ValidationResult:
    valid: bool = False
    applied_actions: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    resulting_state: dict[str, Any] | None = None
    readable_history: list[str] = field(default_factory=list)


@dataclass
class ScreenshotHandHistory:
    source_image: str
    layout: str
    hand_number: int | None
    button: int | None
    blinds: list[int]
    starting_stacks: list[int]
    hero_cards: list[str]
    opponent_cards: list[str]
    board: list[str]
    actions: list[ParsedAction]
    timeline_complete: bool
    confidence: float
    warnings: list[str]
    validation: ValidationResult
    ocr_lines: list[OcrLine]
    players: list[str] = field(default_factory=lambda: ["Hero", "Opponent"])
    current_stacks: list[int | None] = field(default_factory=lambda: [None, None])
    amount_scale: int = 1
    currency: str | None = None

    def to_dict(self, include_ocr: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_ocr:
            payload.pop("ocr_lines", None)
        return payload


def _load_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Screenshot recognition dependencies are missing. Run "
            "`pip install -r backend/requirements.txt`."
        ) from exc
    return cv2, np


def load_image(path: Path):
    cv2, np = _load_cv2()
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


@lru_cache(maxsize=1)
def _rapid_ocr_engine():
    try:
        from rapidocr import RapidOCR  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR is not installed. Run `pip install -r backend/requirements.txt`."
        ) from exc
    return RapidOCR()


def run_ocr(image) -> list[OcrLine]:
    """Run cached RapidOCR while supporting both current and older result shapes."""
    # Card-corner recognition temporarily switches the shared engine to
    # recognition-only mode. RapidOCR keeps those flags for later calls, so
    # explicitly restore full detection/classification/recognition here.
    with _OCR_LOCK:
        result = _rapid_ocr_engine()(
            image,
            use_det=True,
            use_cls=True,
            use_rec=True,
        )
    rows: list[tuple[Any, str, float]] = []
    if hasattr(result, "boxes"):
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        boxes = [] if boxes is None else boxes
        texts = [] if texts is None else texts
        scores = [] if scores is None else scores
        rows = list(zip(boxes, texts, scores))
    elif isinstance(result, tuple) and result:
        for row in result[0] or []:
            if len(row) >= 3:
                rows.append((row[0], str(row[1]), float(row[2])))

    lines: list[OcrLine] = []
    for points, text, score in rows:
        if not text or points is None:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        lines.append(
            OcrLine(
                text=str(text).strip(),
                confidence=float(score),
                box=Box(min(xs), min(ys), max(xs), max(ys)),
            )
        )
    return sorted(lines, key=lambda line: (line.box.top, line.box.left))


def recognize_text_strip(image) -> tuple[str, float]:
    """Recognize one already-localized text strip without text detection.

    CoinPoker's live path knows the exact row/seat rectangles after the two
    windows are located. Skipping the detector turns a multi-second full-frame
    OCR pass into a small recognition call that normally completes in tens of
    milliseconds.
    """

    if image is None or getattr(image, "size", 0) == 0:
        return "", 0.0
    with _OCR_LOCK:
        result = _rapid_ocr_engine()(
            image,
            use_det=False,
            use_cls=False,
            use_rec=True,
        )
    texts = getattr(result, "txts", ()) or ()
    scores = getattr(result, "scores", ()) or ()
    candidates = [
        (str(text).strip(), float(score))
        for text, score in zip(texts, scores)
        if str(text).strip()
    ]
    return max(candidates, key=lambda item: item[1], default=("", 0.0))


def augment_coinpoker_seat_ocr(
    image,
    lines: Sequence[OcrLine],
) -> list[OcrLine]:
    """Add focused OCR for the two CoinPoker name/stack plaques.

    Full-monitor OCR can miss the small stylized stack amount even when it
    detects the surrounding Dealer Chat. Tight seat crops preserve enough
    character detail for reliable decimal recognition.
    """

    height, width = image.shape[:2]
    crops = (
        (0.60, 0.08, 0.82, 0.25),
        (0.60, 0.57, 0.82, 0.79),
    )
    combined = list(lines)
    for left_ratio, top_ratio, right_ratio, bottom_ratio in crops:
        left = max(0, min(width - 1, int(width * left_ratio)))
        top = max(0, min(height - 1, int(height * top_ratio)))
        right = max(left + 1, min(width, int(width * right_ratio)))
        bottom = max(top + 1, min(height, int(height * bottom_ratio)))
        crop = image[top:bottom, left:right]
        for line in run_ocr(crop):
            offset = OcrLine(
                text=line.text,
                confidence=line.confidence,
                box=Box(
                    line.box.left + left,
                    line.box.top + top,
                    line.box.right + left,
                    line.box.bottom + top,
                ),
            )
            duplicate = any(
                existing.text.casefold() == offset.text.casefold()
                and abs(existing.box.center_x - offset.box.center_x) <= 8.0
                and abs(existing.box.center_y - offset.box.center_y) <= 8.0
                for existing in combined
            )
            if not duplicate:
                combined.append(offset)
    return sorted(combined, key=lambda line: (line.box.top, line.box.left))


def _merge_ocr_lines(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Join OCR fragments that occupy the same visual text row."""
    if not lines:
        return []
    heights = [line.box.height for line in lines if line.box.height > 0]
    tolerance = max(5.0, (median(heights) if heights else 12.0) * 0.55)
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: (item.box.center_y, item.box.left)):
        target = next(
            (
                row
                for row in reversed(rows)
                if abs(median(item.box.center_y for item in row) - line.box.center_y)
                <= tolerance
            ),
            None,
        )
        if target is None:
            rows.append([line])
        else:
            target.append(line)

    merged: list[OcrLine] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.box.left)
        segments: list[list[OcrLine]] = []
        for item in ordered:
            if not segments:
                segments.append([item])
                continue
            previous = segments[-1][-1]
            maximum_gap = max(80.0, max(previous.box.height, item.box.height) * 6.0)
            if item.box.left - previous.box.right > maximum_gap:
                segments.append([item])
            else:
                segments[-1].append(item)
        for segment in segments:
            merged.append(
                OcrLine(
                    text=" ".join(item.text for item in segment),
                    confidence=sum(item.confidence for item in segment) / len(segment),
                    box=Box(
                        min(item.box.left for item in segment),
                        min(item.box.top for item in segment),
                        max(item.box.right for item in segment),
                        max(item.box.bottom for item in segment),
                    ),
                )
            )
    return sorted(merged, key=lambda line: (line.box.top, line.box.left))


def _parse_crop(value: str | None, width: int, height: int) -> Box | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Crop must be left,top,right,bottom.")
    if all(0.0 <= part <= 1.0 for part in parts):
        left, top, right, bottom = parts
        crop = Box(left * width, top * height, right * width, bottom * height)
        if crop.width <= 0 or crop.height <= 0:
            raise ValueError("Crop right/bottom must be greater than left/top.")
        return crop
    left, top, right, bottom = parts
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError("Pixel crop falls outside the screenshot.")
    crop = Box(left, top, right, bottom)
    if crop.width <= 0 or crop.height <= 0:
        raise ValueError("Crop right/bottom must be greater than left/top.")
    return crop


def _default_timeline_crop(lines: Sequence[OcrLine], width: int, height: int) -> tuple[str, Box | None]:
    replay_marker = next(
        (line for line in lines if re.search(r"HAND\s+REPLAY", line.text, re.IGNORECASE)),
        None,
    )
    if replay_marker is None:
        history_marker = next(
            (line for line in lines if re.search(r"HAND\s+HISTORY", line.text, re.IGNORECASE)),
            None,
        )
        return ("history-dialog" if history_marker else "unknown", None)

    # Mirrors the desktop replay CSS: a centered 1080x760 dialog whose right
    # 260px column contains the complete action timeline.
    dialog_width = min(1080.0, max(0.0, width - 48.0))
    dialog_height = min(760.0, max(0.0, height - 48.0))
    left = (width - dialog_width) / 2.0
    top = (height - dialog_height) / 2.0
    crop = Box(
        left + dialog_width * (820.0 / 1080.0),
        top + dialog_height * (65.0 / 760.0),
        left + dialog_width,
        top + dialog_height * (630.0 / 760.0),
    )
    return "replay-dialog", crop


def _normalize_text(value: str) -> str:
    normalized = value.replace("\u00b7", " ").replace("\u2022", " ")
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^\d+[.)]\s*", "", normalized)
    return normalized


def _amount(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9]", "", value)
    return int(cleaned) if cleaned else None


def _actor_from_text(text: str) -> int | None:
    match = re.search(r"\bPLAYER\s*([12])\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1)) - 1
    if re.search(r"\b(YOU|HERO)\b", text, re.IGNORECASE):
        return 0
    if re.search(r"\b(AGENT|VILLAIN|OPPONENT)\b", text, re.IGNORECASE):
        return 1
    return None


def parse_timeline(lines: Sequence[OcrLine]) -> tuple[
    int | None,
    int | None,
    list[int],
    list[ParsedAction],
    bool,
    list[str],
]:
    hand_number: int | None = None
    button: int | None = None
    small_blind = 10
    big_blind = 20
    saw_small_blind = False
    saw_big_blind = False
    street = "preflop"
    actions: list[ParsedAction] = []
    warnings: list[str] = []

    semantic_lines: list[OcrLine] = []
    for line in lines:
        text = _normalize_text(line.text)
        if re.search(
            r"HAND|PLAYER|YOU|HERO|AGENT|VILLAIN|OPPONENT|FLOP|TURN|RIVER|"
            r"FOLD|CHECK|CALL|RAISE|ALL.?IN|POSTS|SHOWDOWN|WINS|SPLIT",
            text,
            re.IGNORECASE,
        ):
            semantic_lines.append(OcrLine(text, line.confidence, line.box))

    # Exact adjacent duplicates are common when a callout repeats the selected
    # action. Keep the timeline ordering but remove those repeats.
    deduplicated: list[OcrLine] = []
    for line in semantic_lines:
        key = re.sub(r"[^A-Z0-9]", "", line.text.upper())
        if deduplicated:
            previous = re.sub(r"[^A-Z0-9]", "", deduplicated[-1].text.upper())
            if key == previous:
                continue
        deduplicated.append(line)

    for line in deduplicated:
        text = line.text
        number_match = re.search(r"\bHAND\s*#?\s*(\d+)\b", text, re.IGNORECASE)
        if number_match and hand_number is None:
            hand_number = int(number_match.group(1))

        street_match = re.search(r"\b(FLOP|TURN|RIVER)\b", text, re.IGNORECASE)
        if street_match:
            street = street_match.group(1).lower()
            continue

        actor = _actor_from_text(text)
        if actor is None:
            continue

        blind_match = re.search(
            r"POSTS?\s+(SMALL|BIG)\s+BLIND\s*[:=]?\s*([\d,.]+)",
            text,
            re.IGNORECASE,
        )
        if blind_match:
            amount = _amount(blind_match.group(2))
            if blind_match.group(1).lower() == "small":
                small_blind = amount or small_blind
                saw_small_blind = True
                button = actor
            else:
                big_blind = amount or big_blind
                saw_big_blind = True
            continue

        action: str | None = None
        amount: int | None = None
        if re.search(r"\bFOLDS?\b", text, re.IGNORECASE):
            action = "fold"
        elif re.search(r"\bCHECKS?\b", text, re.IGNORECASE):
            action = "check"
        else:
            call_match = re.search(r"\bCALLS?\s*([\d,.]+)?", text, re.IGNORECASE)
            raise_match = re.search(
                r"\b(?:RAISES?\s+TO|BETS?)\s*([\d,.]+)?",
                text,
                re.IGNORECASE,
            )
            if call_match:
                action = "call"
                amount = _amount(call_match.group(1))
            elif raise_match:
                action = "raise"
                amount = _amount(raise_match.group(1))
            elif re.search(r"\bALL[ -]?IN\b", text, re.IGNORECASE):
                action = "all_in"

        if action is None:
            continue
        actions.append(
            ParsedAction(
                player=actor,
                action=action,
                amount=amount,
                street=street,
                confidence=line.confidence,
                raw_text=text,
            )
        )

    if not saw_small_blind or not saw_big_blind:
        warnings.append(
            "The visible timeline does not contain both blind posts; earlier actions may be clipped."
        )
    if not actions:
        warnings.append("No betting actions were recognized in the visible timeline.")
    timeline_starts_at_hand = saw_small_blind and saw_big_blind
    return hand_number, button, [small_blind, big_blind], actions, timeline_starts_at_hand, warnings


@lru_cache(maxsize=4)
def _render_card_templates(asset_dir: Path, output_size: tuple[int, int] = (125, 180)):
    cv2, np = _load_cv2()
    try:
        import resvg_py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The SVG card renderer is not installed. Run "
            "`pip install -r backend/requirements.txt`."
        ) from exc

    templates: dict[str, Any] = {}
    for path in sorted(asset_dir.glob("*.svg")):
        if not re.fullmatch(r"[2-9TJQKA][shdc]\.svg", path.name):
            continue
        png = resvg_py.svg_to_bytes(svg_string=path.read_text(encoding="utf-8"))
        template = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if template is not None:
            template = cv2.resize(template, output_size, interpolation=cv2.INTER_AREA)
            code = path.stem
            templates[f"{code[0]}{SUIT_CODES[code[1]]}"] = template
    if len(templates) != 52:
        raise RuntimeError(f"Expected 52 card assets in {asset_dir}, found {len(templates)}.")
    return templates


def _order_quad(points):
    _, np = _load_cv2()
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _warp_candidate(image, contour, output_size: tuple[int, int] = (125, 180)):
    cv2, np = _load_cv2()
    rectangle = cv2.minAreaRect(contour)
    points = _order_quad(cv2.boxPoints(rectangle))
    width, height = output_size
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(image, matrix, output_size)


def _card_match_score(candidate, template) -> float:
    """Score the full face plus the two rank-heavy regions in the card artwork."""
    cv2, _ = _load_cv2()
    height, width = candidate.shape[:2]

    def normalized_match(first, second) -> float:
        return float(cv2.matchTemplate(first, second, cv2.TM_CCOEFF_NORMED)[0, 0])

    # The complete face preserves the suit-specific background signal. The
    # corner separates rank/suit combinations, while the large lower glyph
    # makes same-suit ranks much harder to confuse.
    corner = (slice(0, int(height * 0.43)), slice(0, int(width * 0.48)))
    large_rank = (slice(int(height * 0.34), height), slice(int(width * 0.12), int(width * 0.88)))
    full_score = normalized_match(candidate, template)
    corner_score = normalized_match(candidate[corner], template[corner])
    large_rank_score = normalized_match(candidate[large_rank], template[large_rank])
    return full_score * 0.25 + corner_score * 0.30 + large_rank_score * 0.45


def _iou(first: Box, second: Box) -> float:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def detect_cards(
    image,
    asset_dir: Path,
    minimum_score: float = 0.58,
) -> tuple[list[DetectedCard], list[str]]:
    cv2, _ = _load_cv2()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    templates = _render_card_templates(asset_dir)

    candidates: list[tuple[Box, Any, float]] = []
    minimum_width = max(22.0, width * 0.012)
    maximum_width = width * 0.13
    for contour in contours:
        center, (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        short, long = sorted((float(rect_width), float(rect_height)))
        if short < minimum_width or short > maximum_width or long <= 0:
            continue
        ratio = long / short
        if not 1.25 <= ratio <= 1.65:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if area < short * long * 0.55:
            continue
        box = Box(
            max(0.0, center[0] - short / 2.0),
            max(0.0, center[1] - long / 2.0),
            min(float(width), center[0] + short / 2.0),
            min(float(height), center[1] + long / 2.0),
        )
        if any(_iou(box, previous[0]) > 0.72 for previous in candidates):
            continue
        candidates.append((box, contour, area))

    detections: list[DetectedCard] = []
    for box, contour, _ in sorted(candidates, key=lambda item: item[2], reverse=True):
        warped = _warp_candidate(image, contour)
        scores: list[tuple[float, str]] = []
        for card, template in templates.items():
            score = _card_match_score(warped, template)
            # Some contours arrive upside-down depending on minAreaRect point order.
            rotated_score = _card_match_score(cv2.rotate(warped, cv2.ROTATE_180), template)
            scores.append((max(score, rotated_score), card))
        scores.sort(reverse=True)
        best_score, best_card = scores[0]
        second_score = scores[1][0]
        if best_score < minimum_score:
            continue
        detection = DetectedCard(
            card=best_card,
            confidence=best_score,
            margin=best_score - second_score,
            box=box,
        )
        if any(_iou(detection.box, existing.box) > 0.45 for existing in detections):
            continue
        detections.append(detection)

    warnings: list[str] = []
    ambiguous = [card for card in detections if card.margin < 0.015]
    if ambiguous:
        warnings.append(f"{len(ambiguous)} card match(es) have a small first/second-choice margin.")
    duplicate_names = {
        card.card for card in detections if sum(other.card == card.card for other in detections) > 1
    }
    if duplicate_names:
        warnings.append(
            "Duplicate card detections were discarded from seat assignment: "
            + ", ".join(sorted(duplicate_names))
        )
        detections = [card for card in detections if card.card not in duplicate_names]
    return sorted(detections, key=lambda card: (card.box.center_y, card.box.center_x)), warnings


def assign_cards(cards: Sequence[DetectedCard], image_height: int) -> tuple[list[str], list[str], list[str]]:
    if not cards:
        return [], [], []
    typical_height = median(card.box.height for card in cards)
    tolerance = max(12.0, typical_height * 0.65)
    groups: list[list[DetectedCard]] = []
    for card in sorted(cards, key=lambda item: item.box.center_y):
        target = next(
            (
                group
                for group in groups
                if abs(median(item.box.center_y for item in group) - card.box.center_y) <= tolerance
            ),
            None,
        )
        if target is None:
            groups.append([card])
        else:
            target.append(card)
    for group in groups:
        group.sort(key=lambda item: item.box.center_x)

    board_group = max((group for group in groups if len(group) >= 3), key=len, default=[])
    hole_groups = [group for group in groups if group is not board_group and len(group) == 2]
    hole_groups.sort(key=lambda group: median(item.box.center_y for item in group))

    opponent: list[str] = []
    hero: list[str] = []
    if len(hole_groups) >= 2:
        opponent = [card.card for card in hole_groups[0]]
        hero = [card.card for card in hole_groups[-1]]
    elif len(hole_groups) == 1:
        center_y = median(item.box.center_y for item in hole_groups[0])
        target = hero if center_y >= image_height / 2.0 else opponent
        target.extend(card.card for card in hole_groups[0])

    board = [card.card for card in board_group]
    return hero, opponent, board


def parse_card(value: str) -> Card:
    cleaned = value.strip().replace("10", "T")
    if len(cleaned) != 2:
        raise ValueError(f"Invalid card: {value}")
    rank = RANK_VALUES.get(cleaned[0].upper())
    suit = SUIT_ALIASES.get(cleaned[1].lower(), SUIT_ALIASES.get(cleaned[1]))
    if rank is None or suit is None:
        raise ValueError(f"Invalid card: {value}")
    return rank, suit


def _prepare_validation_game(
    starting_stacks: Sequence[int],
    blinds: Sequence[int],
    button: int,
    hero_cards: Sequence[str],
    opponent_cards: Sequence[str],
    board: Sequence[str],
) -> HeadsUpHoldem:
    known_hero = [parse_card(card) for card in hero_cards]
    known_opponent = [parse_card(card) for card in opponent_cards]
    known_board = [parse_card(card) for card in board]
    known = known_hero + known_opponent + known_board
    if len(set(known)) != len(known):
        raise ValueError("Recognized cards contain duplicates.")

    game = HeadsUpHoldem(
        initial_stack=max(starting_stacks),
        small_blind=int(blinds[0]),
        big_blind=int(blinds[1]),
    )
    game.stacks = list(starting_stacks)
    game.hand_number = 0
    game.button_offset = button
    game.new_hand()

    available = [card for card in new_deck() if card not in set(known)]
    hero = list(known_hero)
    opponent = list(known_opponent)
    while len(hero) < 2:
        hero.append(available.pop(0))
    while len(opponent) < 2:
        opponent.append(available.pop(0))
    game.hole_cards = [hero[:2], opponent[:2]]
    future = [card for card in available if card not in set(known_board)]
    game.deck = future + list(reversed(known_board))
    return game


def validate_history(
    actions: Sequence[ParsedAction],
    starting_stacks: Sequence[int],
    blinds: Sequence[int],
    button: int | None,
    hero_cards: Sequence[str],
    opponent_cards: Sequence[str],
    board: Sequence[str],
) -> ValidationResult:
    if button is None:
        return ValidationResult(error="The dealer/button could not be inferred from the small blind post.")
    try:
        game = _prepare_validation_game(
            starting_stacks,
            blinds,
            button,
            hero_cards,
            opponent_cards,
            board,
        )
    except ValueError as exc:
        return ValidationResult(error=str(exc))

    warnings: list[str] = []
    for index, action in enumerate(actions, start=1):
        if game.active_street != action.street:
            return ValidationResult(
                applied_actions=index - 1,
                error=(
                    f"Action {index} was read as {action.street}, but the rules engine is on "
                    f"{game.active_street}; the screenshot likely omits or misread an action."
                ),
                warnings=warnings,
                readable_history=list(game.history),
            )
        if action.action == "raise" and action.amount is None:
            return ValidationResult(
                applied_actions=index - 1,
                error=f"Action {index} is a raise but its raise-to amount was not recognized.",
                warnings=warnings,
                readable_history=list(game.history),
            )
        if game.current_player != action.player:
            return ValidationResult(
                applied_actions=index - 1,
                error=(
                    f"Action {index} assigns Player {action.player + 1}, but Player "
                    f"{(game.current_player or 0) + 1} is due to act."
                ),
                warnings=warnings,
                readable_history=list(game.history),
            )
        try:
            game.act(action.player, action.action, action.amount)
        except (InvalidAction, ValueError) as exc:
            return ValidationResult(
                applied_actions=index - 1,
                error=f"Action {index} is invalid: {exc}",
                warnings=warnings,
                readable_history=list(game.history),
            )

    if board:
        recognized = [parse_card(card) for card in board]
        visible = game.community[: len(recognized)]
        if visible != recognized[: len(visible)]:
            warnings.append("Recognized board cards do not match the replayed street order.")
        elif len(game.community) < len(recognized):
            warnings.append(
                "The recognized board is later than the last replayed action; the visible timeline may end early."
            )

    state = game.snapshot(0)
    state.pop("session_stats", None)
    return ValidationResult(
        valid=True,
        applied_actions=len(actions),
        warnings=warnings,
        resulting_state=state,
        readable_history=list(game.history),
    )


def extract_hand_history(
    image_path: Path,
    asset_dir: Path,
    starting_stacks: Sequence[int] | None = None,
    timeline_crop: str | None = None,
    minimum_card_score: float = 0.58,
    layout_hint: str = "auto",
) -> ScreenshotHandHistory:
    if layout_hint not in {"auto", "default", "coinpoker"}:
        raise ValueError("Layout must be auto, default, or coinpoker.")
    if starting_stacks is not None and (
        len(starting_stacks) != 2 or any(stack <= 0 for stack in starting_stacks)
    ):
        raise ValueError("Starting stacks must contain two positive chip amounts.")
    image = load_image(image_path)
    height, width = image.shape[:2]
    raw_ocr = run_ocr(image)
    from .layouts.coinpoker import detect_coinpoker_layout, extract_coinpoker_layout

    use_coinpoker = layout_hint == "coinpoker" or (
        layout_hint == "auto" and detect_coinpoker_layout(raw_ocr)
    )
    if use_coinpoker:
        recognized = extract_coinpoker_layout(
            image,
            raw_ocr,
            width,
            height,
            starting_stacks_override=starting_stacks,
        )
        if len(recognized.players) < 2 or any(
            stack is None for stack in recognized.current_stacks
        ):
            raw_ocr = augment_coinpoker_seat_ocr(image, raw_ocr)
            recognized = extract_coinpoker_layout(
                image,
                raw_ocr,
                width,
                height,
                starting_stacks_override=starting_stacks,
            )
        actions = [ParsedAction(**action) for action in recognized.actions]
        warnings = list(recognized.warnings)
        validation = validate_history(
            actions,
            recognized.starting_stacks,
            recognized.blinds,
            recognized.button,
            recognized.hero_cards,
            recognized.opponent_cards,
            recognized.board,
        )
        warnings.extend(validation.warnings)
        if validation.error:
            warnings.append(validation.error)
        resulting_state = validation.resulting_state or {}
        timeline_complete = recognized.timeline_starts_at_hand and bool(
            resulting_state.get("complete")
        )
        if recognized.timeline_starts_at_hand and validation.valid and not timeline_complete:
            warnings.append(
                "The current CoinPoker hand is valid so far but has not finished."
            )
        confidence_parts = recognized.action_confidences + recognized.card_confidences
        confidence = (
            sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.0
        )
        if not validation.valid:
            confidence *= 0.65
        if not recognized.timeline_starts_at_hand:
            confidence *= 0.75
        return ScreenshotHandHistory(
            source_image=str(image_path.resolve()),
            layout="coinpoker-dealer-chat",
            hand_number=recognized.hand_number,
            button=recognized.button,
            blinds=recognized.blinds,
            starting_stacks=recognized.starting_stacks,
            hero_cards=recognized.hero_cards,
            opponent_cards=recognized.opponent_cards,
            board=recognized.board,
            actions=actions,
            timeline_complete=timeline_complete,
            confidence=round(confidence, 4),
            warnings=list(dict.fromkeys(warnings)),
            validation=validation,
            ocr_lines=recognized.chat_lines,
            players=recognized.players or ["Hero", "Opponent"],
            current_stacks=recognized.current_stacks,
            amount_scale=100,
        )

    effective_starting_stacks = list(starting_stacks or (2_000, 2_000))
    layout, default_crop = _default_timeline_crop(raw_ocr, width, height)
    crop = _parse_crop(timeline_crop, width, height) or default_crop
    timeline_fragments = (
        [line for line in raw_ocr if crop and line.box.intersects(crop)]
        if crop is not None
        else raw_ocr
    )
    timeline_lines = _merge_ocr_lines(timeline_fragments)
    if crop is not None and not timeline_lines:
        timeline_lines = _merge_ocr_lines(raw_ocr)

    hand_number, button, blinds, actions, timeline_starts_at_hand, warnings = parse_timeline(
        timeline_lines
    )
    if hand_number is None:
        full_hand_number, _, _, _, _, _ = parse_timeline(_merge_ocr_lines(raw_ocr))
        hand_number = full_hand_number
    cards, card_warnings = detect_cards(image, asset_dir, minimum_card_score)
    hero_cards, opponent_cards, board = assign_cards(cards, height)
    warnings.extend(card_warnings)
    if not hero_cards:
        warnings.append("Hero hole cards were not recognized.")
    if not board and any(action.street != "preflop" for action in actions):
        warnings.append("Postflop actions were recognized, but no board cards were matched.")

    validation = validate_history(
        actions,
        effective_starting_stacks,
        blinds,
        button,
        hero_cards,
        opponent_cards,
        board,
    )
    warnings.extend(validation.warnings)
    if validation.error:
        warnings.append(validation.error)

    resulting_state = validation.resulting_state or {}
    timeline_complete = timeline_starts_at_hand and bool(resulting_state.get("complete"))
    if timeline_starts_at_hand and validation.valid and not timeline_complete:
        warnings.append(
            "The recognized timeline starts with the blinds but does not complete the hand."
        )

    confidence_parts = [action.confidence for action in actions]
    confidence_parts.extend(card.confidence for card in cards)
    confidence = sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.0
    if not validation.valid:
        confidence *= 0.65
    if not timeline_starts_at_hand:
        confidence *= 0.75

    return ScreenshotHandHistory(
        source_image=str(image_path.resolve()),
        layout=layout,
        hand_number=hand_number,
        button=button,
        blinds=blinds,
        starting_stacks=effective_starting_stacks,
        hero_cards=hero_cards,
        opponent_cards=opponent_cards,
        board=board,
        actions=actions,
        timeline_complete=timeline_complete,
        confidence=round(confidence, 4),
        warnings=list(dict.fromkeys(warnings)),
        validation=validation,
        ocr_lines=timeline_lines,
    )


def readable_text(history: ScreenshotHandHistory) -> str:
    def amount(value: int) -> str:
        if history.amount_scale == 1:
            return str(value)
        return f"{value / history.amount_scale:.2f}"

    if history.layout == "coinpoker-dealer-chat":
        players = history.players if len(history.players) == 2 else ["Hero", "Opponent"]
        button_name = (
            players[history.button]
            if history.button is not None and history.button < len(players)
            else "unknown"
        )
        lines = [
            f"Hand #{history.hand_number if history.hand_number is not None else 'unknown'}",
            "Game: Heads-Up No-Limit Hold'em",
            f"Blinds: {amount(history.blinds[0])}/{amount(history.blinds[1])}",
            f"Seat 1: {players[0]} ({amount(history.starting_stacks[0])})",
            f"Seat 2: {players[1]} ({amount(history.starting_stacks[1])})",
            f"Button: {button_name}",
            f"Dealt to {players[0]} [{' '.join(history.hero_cards) if history.hero_cards else 'unknown'}]",
            "",
            "*** PRE-FLOP ***",
        ]
        if history.button is not None:
            lines.extend(
                [
                    f"{players[history.button]} posts small blind {amount(history.blinds[0])}",
                    f"{players[1 - history.button]} posts big blind {amount(history.blinds[1])}",
                ]
            )
        street = "preflop"
        for action in history.actions:
            if action.street != street:
                street = action.street
                visible_board = history.board[: {"flop": 3, "turn": 4, "river": 5}[street]]
                lines.extend(["", f"*** {street.upper()} *** [{' '.join(visible_board)}]"])
            actor = players[action.player]
            if action.action == "raise":
                verb = (
                    "bets"
                    if action.street != "preflop" and re.search(r"\bBET", action.raw_text, re.IGNORECASE)
                    else "raises to"
                )
            else:
                verb = {
                    "fold": "folds",
                    "check": "checks",
                    "call": "calls",
                    "all_in": "is all-in",
                }.get(action.action, action.action)
            suffix = f" {amount(action.amount)}" if action.amount is not None else ""
            lines.append(f"{actor} {verb}{suffix}")
        if history.validation.resulting_state:
            lines.extend(
                [
                    "",
                    f"Pot: {amount(int(history.validation.resulting_state.get('pot', 0)))}",
                    "Status: complete" if history.timeline_complete else "Status: hand in progress",
                ]
            )
        if history.warnings:
            lines.extend(["", "Warnings:", *(f"- {warning}" for warning in history.warnings)])
        return "\n".join(lines).rstrip() + "\n"

    lines = [
        f"Hand #{history.hand_number if history.hand_number is not None else 'unknown'}",
        f"Blinds: {history.blinds[0]}/{history.blinds[1]}",
        f"Button: {'Player ' + str(history.button + 1) if history.button is not None else 'unknown'}",
        f"Hero: {' '.join(history.hero_cards) if history.hero_cards else 'unknown'}",
        f"Opponent: {' '.join(history.opponent_cards) if history.opponent_cards else 'hidden/unknown'}",
        f"Board: {' '.join(history.board) if history.board else 'none recognized'}",
        "",
    ]
    if history.validation.readable_history:
        lines.extend(history.validation.readable_history)
    else:
        current_street = "preflop"
        for action in history.actions:
            if action.street != current_street:
                current_street = action.street
                lines.append(f"{current_street.title()}:")
            amount = f" {action.amount}" if action.amount is not None else ""
            lines.append(f"Player {action.player + 1} {action.action}{amount}")
    if history.warnings:
        lines.extend(["", "Warnings:", *(f"- {warning}" for warning in history.warnings)])
    return "\n".join(lines).rstrip() + "\n"
