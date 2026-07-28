"""Continuous pixel-only recognition and hand tracking for the poker table."""

from __future__ import annotations

import copy
import difflib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..poker import HeadsUpHoldem, InvalidAction, STREETS, best_score, new_deck
from .recognition import (
    Box,
    OcrLine,
    ParsedAction,
    _merge_ocr_lines,
    augment_coinpoker_seat_ocr,
    assign_cards,
    detect_cards,
    load_image,
    parse_card,
    recognize_text_strip,
    run_ocr,
    validate_history,
)
from .layouts.coinpoker import extract_coinpoker_layout
from .capture import CaptureRect, WindowInfo, list_windows, window_outer_rect


STREET_INDEX = {street: index for index, street in enumerate(STREETS)}
BACKEND_DIRECTORY = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "default.json"
PROFILE_DIRECTORY = Path(__file__).resolve().parent / "profiles"
CUSTOM_PROFILE_DIRECTORY = BACKEND_DIRECTORY / "data" / "screen_profiles"
DEFAULT_OUTPUT_DIRECTORY = BACKEND_DIRECTORY / "data" / "screen_hand_history"
COINPOKER_STACK_FALLBACK_WARNING = (
    "Starting stacks could not be inferred; validation used the 20.00 fallback."
)


@dataclass(frozen=True)
class ScreenProfile:
    name: str
    version: int
    content_anchor_patterns: tuple[str, ...]
    fallback_content_bounds: tuple[float, float, float, float]
    regions: dict[str, tuple[float, float, float, float]]
    pixel_change_threshold: float = 1.25
    minimum_card_score: float = 0.58
    # Optional click targets for the auto-play module, normalized to the poker
    # TABLE WINDOW rather than to the captured frame, so they survive the window
    # being moved between monitors. Recognition never reads them.
    action_controls: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScreenProfile":
        regions = {
            str(name): tuple(float(value) for value in bounds)
            for name, bounds in payload.get("regions", {}).items()
        }
        action_controls = {
            str(name): tuple(float(value) for value in bounds)
            for name, bounds in payload.get("action_controls", {}).items()
        }
        for name, bounds in {**regions, **action_controls}.items():
            if len(bounds) != 4 or not all(0.0 <= value <= 1.0 for value in bounds):
                raise ValueError(f"Profile region {name} must contain four values from 0 to 1.")
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError(f"Profile region {name} has invalid right/bottom bounds.")
        fallback = tuple(float(value) for value in payload.get("fallback_content_bounds", (0, 0, 1, 1)))
        if len(fallback) != 4:
            raise ValueError("fallback_content_bounds must contain four values.")
        return cls(
            name=str(payload.get("name", "custom")),
            version=int(payload.get("version", 1)),
            content_anchor_patterns=tuple(payload.get("content_anchor_patterns", ())),
            fallback_content_bounds=fallback,
            regions=regions,
            pixel_change_threshold=float(payload.get("pixel_change_threshold", 1.25)),
            minimum_card_score=float(payload.get("minimum_card_score", 0.58)),
            action_controls=action_controls,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "version": self.version,
            "content_anchor_patterns": list(self.content_anchor_patterns),
            "fallback_content_bounds": list(self.fallback_content_bounds),
            "regions": {name: list(bounds) for name, bounds in self.regions.items()},
            "pixel_change_threshold": self.pixel_change_threshold,
            "minimum_card_score": self.minimum_card_score,
        }
        if self.action_controls:
            payload["action_controls"] = {
                name: list(bounds) for name, bounds in self.action_controls.items()
            }
        return payload


@dataclass(frozen=True)
class InferredAction:
    player: int
    action: str
    amount: int | None
    street: str


@dataclass(frozen=True)
class InspectionBox:
    """A region the recognizer read this frame, in captured-image pixel
    coordinates, with the recognition score (0..1) that came out of it.

    Collected only when a recognizer is created with
    ``collect_inspection_boxes=True`` (driven by the
    ``RuntimeSettings.show_inspection_boxes`` flag) and consumed by the optional
    live desktop overlay, which draws a red box + accuracy readout over every
    area the watcher is inspecting. It never affects recognition itself."""

    label: str
    left: float
    top: float
    right: float
    bottom: float
    confidence: float


@dataclass(frozen=True)
class VisibleTableState:
    captured_at: str
    hand_number: int | None
    street: str | None
    pot: int | None
    stacks: tuple[int | None, int | None]
    round_bets: tuple[int | None, int | None]
    hero_cards: tuple[str, ...]
    opponent_cards: tuple[str, ...]
    board: tuple[str, ...]
    button: int | None
    current_player: int | None
    complete: bool
    stable: bool
    history_stable: bool
    decision_ready: bool
    confidence: float
    warnings: tuple[str, ...]
    source_layout: str = "default"
    players: tuple[str, ...] = ("Hero", "Opponent")
    starting_stacks: tuple[int | None, int | None] = (None, None)
    visible_actions: tuple[InferredAction, ...] = ()
    timeline_starts_at_hand: bool = False
    recognition_ms: int | None = None
    # Blinds auto-detected from the Dealer Chat (SB, BB) in the recognizer's
    # smallest unit. (0, 0) means "not detected" -> the tracker falls back to the
    # launch --blinds. The reconstruction engine MUST use these, not the launch
    # flag, or a table at a different blind level cannot be reconstructed.
    blinds: tuple[int, int] = (0, 0)

    def state_key(self) -> tuple[Any, ...]:
        return (
            self.hand_number,
            self.street,
            self.pot,
            self.stacks,
            self.round_bets,
            self.hero_cards,
            self.opponent_cards,
            self.board,
            self.button,
            self.current_player,
            self.complete,
            self.stable,
            self.history_stable,
            self.source_layout,
            self.visible_actions,
        )


@dataclass
class TransitionResult:
    status: str
    actions: list[InferredAction] = field(default_factory=list)
    candidates: int = 0
    warning: str | None = None


@dataclass
class _CoinPokerFastContext:
    """Cached geometry and OCR strips for the arranged CoinPoker workspace."""

    rect_key: tuple[int, ...]
    dealer_box: Box
    table_box: Box
    players: tuple[str, str] | None = None
    fields: dict[str, tuple[int, str, float]] = field(default_factory=dict)
    rows: dict[tuple[str, int], tuple[int, str, str, float]] = field(
        default_factory=dict
    )


@dataclass
class TrackedHand:
    hand_number: int | None
    started_at: str
    button: int | None
    blinds: tuple[int, int]
    starting_stacks: tuple[int | None, int | None]
    hero_cards: tuple[str, ...]
    source_layout: str = "default"
    players: tuple[str, ...] = ("Hero", "Opponent")
    amount_scale: int = 1
    actions: list[InferredAction] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[VisibleTableState] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    complete: bool = False
    finalized: bool = False
    engine: HeadsUpHoldem | None = field(default=None, repr=False)

    def reported_warnings(self) -> list[str]:
        """Reconcile warnings to the hand's CLEANEST trustworthy read.

        A hand that had a clean decision frame should read clean, not inherit
        transient cross-check flicker (stack/pot mismatch) from a later
        animation frame. So the per-frame warnings come from the decision-ready/
        stable observation with the FEWEST warnings (typically the frame the
        decision was actually made on), not merely the most recent one. If no
        frame was ever stable, the last observation's warnings are used as-is.

        Structural, hand-level warnings are surfaced too, EXCEPT the transient
        engine-rebuild failure: if the hand produced at least one decision the
        engine demonstrably worked, so a later rebuild hiccup on an animation
        frame is not a hand-level problem worth flagging.
        """

        stable_obs = [o for o in self.observations if o.decision_ready or o.stable]
        if stable_obs:
            frame_warnings = min(stable_obs, key=lambda o: len(o.warnings)).warnings
        elif self.observations:
            frame_warnings = self.observations[-1].warnings
        else:
            frame_warnings = ()
        structural = list(self.warnings)
        if self.decisions:
            structural = [
                w for w in structural
                if "could not rebuild a decision rules state" not in w
            ]
        return list(dict.fromkeys([*structural, *frame_warnings]))

    def _determine_outcome(
        self, board: Sequence[str], opponent_cards: Sequence[str]
    ) -> tuple[int | None, str]:
        """Who won this hand (0=hero, 1=opponent, None=split/unknown) and how.

        A hand ends two ways: a FOLD (the folder loses; in heads-up the other
        player wins outright) or a SHOWDOWN. For a showdown we need the five
        board cards plus BOTH hole pairs — CoinPoker flips the opponent's cards
        face-up on the table at showdown, so the card detector supplies
        opponent_cards — and then the tested `best_score` evaluator decides it.
        Everything is wrapped so a bad read can never crash the hand record."""
        folds = [
            action.player
            for action in self.actions
            if getattr(action, "action", None) == "fold"
        ]
        if folds:
            folder = folds[-1]
            return (1 - folder if folder in (0, 1) else None), "fold"
        hero = list(self.hero_cards)
        opp = list(opponent_cards)
        if len(board) >= 5 and len(hero) == 2 and len(opp) == 2:
            try:
                cards = list(board)
                hero_score = best_score([parse_card(c) for c in hero + cards])
                opp_score = best_score([parse_card(c) for c in opp + cards])
                if hero_score > opp_score:
                    return 0, "showdown"
                if opp_score > hero_score:
                    return 1, "showdown"
                return None, "showdown-split"
            except Exception:
                return None, "showdown-unresolved"
        return None, "unknown"

    def payload(self) -> dict[str, Any]:
        board = list(self.observations[-1].board) if self.observations else []
        opponent_cards = (
            list(self.observations[-1].opponent_cards) if self.observations else []
        )
        winner, outcome = self._determine_outcome(board, opponent_cards)
        winner_name = (
            self.players[winner]
            if winner is not None and winner < len(self.players)
            else None
        )
        return {
            "hand_number": self.hand_number,
            "started_at": self.started_at,
            "button": self.button,
            "blinds": list(self.blinds),
            "starting_stacks": list(self.starting_stacks),
            "hero_cards": list(self.hero_cards),
            "source_layout": self.source_layout,
            "players": list(self.players),
            "amount_scale": self.amount_scale,
            "board": board,
            "opponent_cards": opponent_cards,
            "actions": [asdict(action) for action in self.actions],
            "decisions": list(self.decisions),
            "complete": self.complete,
            "winner": winner,
            "winner_name": winner_name,
            "outcome": outcome,
            "warnings": self.reported_warnings(),
            "observations": [asdict(observation) for observation in self.observations],
        }


def load_profile(name_or_path: str | Path = "default") -> ScreenProfile:
    requested = Path(name_or_path)
    candidates = [requested]
    if requested.suffix.lower() != ".json":
        candidates = [CUSTOM_PROFILE_DIRECTORY / f"{requested}.json"]
        candidates.append(PROFILE_DIRECTORY / f"{requested}.json")
        candidates.append(Path.cwd() / f"{requested}.json")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Screen profile was not found: {name_or_path}")
    return ScreenProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _normalized_box(bounds: Sequence[float], parent: Box) -> Box:
    left, top, right, bottom = bounds
    return Box(
        parent.left + parent.width * left,
        parent.top + parent.height * top,
        parent.left + parent.width * right,
        parent.top + parent.height * bottom,
    )


def _clip_box(box: Box, width: int, height: int) -> Box:
    return Box(
        max(0.0, min(float(width), box.left)),
        max(0.0, min(float(height), box.top)),
        max(0.0, min(float(width), box.right)),
        max(0.0, min(float(height), box.bottom)),
    )


def locate_content(lines: Sequence[OcrLine], width: int, height: int, profile: ScreenProfile) -> Box:
    # Patterns are ordered from most simulator-specific to least specific. This
    # prevents a browser tab named "Text Hold'em" from winning over the actual
    # SELF-PLAY LAB anchor inside the captured page.
    anchors: list[OcrLine] = []
    for pattern in profile.content_anchor_patterns:
        anchors = [
            line for line in lines if re.search(pattern, line.text, re.IGNORECASE)
        ]
        if anchors:
            break
    if anchors:
        anchor = max(
            anchors,
            key=lambda line: (line.box.height, line.box.width, -line.box.top),
        )
        left = max(0.0, anchor.box.left - max(14.0, anchor.box.height * 0.7))
        top = max(0.0, anchor.box.top - max(14.0, anchor.box.height * 1.1))
        right = max(left + 1.0, width - left)
        bottom = max(top + 1.0, height - max(8.0, left * 0.25))
        return _clip_box(Box(left, top, right, bottom), width, height)
    fallback = _normalized_box(profile.fallback_content_bounds, Box(0, 0, width, height))
    return _clip_box(fallback, width, height)


def calibrate_profile(image_path: Path, output_path: Path, name: str) -> ScreenProfile:
    image = load_image(image_path)
    height, width = image.shape[:2]
    base = load_profile("default")
    lines = run_ocr(image)
    content = locate_content(lines, width, height, base)
    calibrated = ScreenProfile(
        name=name,
        version=base.version,
        content_anchor_patterns=base.content_anchor_patterns,
        fallback_content_bounds=(
            content.left / width,
            content.top / height,
            content.right / width,
            content.bottom / height,
        ),
        regions=base.regions,
        pixel_change_threshold=base.pixel_change_threshold,
        minimum_card_score=base.minimum_card_score,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibrated.to_dict(), indent=2), encoding="utf-8")
    return calibrated


def _lines_in_box(lines: Sequence[OcrLine], box: Box) -> list[OcrLine]:
    return _merge_ocr_lines(
        [
            line
            for line in lines
            if box.left <= line.box.center_x <= box.right
            and box.top <= line.box.center_y <= box.bottom
        ]
    )


def _numbers(lines: Sequence[OcrLine]) -> list[tuple[int, float]]:
    results: list[tuple[int, float]] = []
    for line in lines:
        for match in re.finditer(r"(?<![#\d])\d[\d,.]*", line.text):
            cleaned = re.sub(r"\D", "", match.group(0))
            if cleaned:
                score = line.confidence * max(1.0, line.box.height)
                results.append((int(cleaned), score))
    return results


def _best_number(lines: Sequence[OcrLine], default: int | None = None) -> int | None:
    values = _numbers(lines)
    return max(values, key=lambda item: item[1])[0] if values else default


def _hand_number(lines: Sequence[OcrLine]) -> int | None:
    for line in lines:
        match = re.search(r"(?:HAND\s*)?#\s*(\d+)", line.text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return _best_number(lines)


def _street(lines: Sequence[OcrLine]) -> str | None:
    joined = " ".join(line.text.upper() for line in lines)
    for street in reversed(STREETS):
        if re.search(rf"\b{street.upper()}\b", joined):
            return street
    return None


class LiveTableRecognizer:
    def __init__(
        self,
        profile: ScreenProfile,
        asset_dir: Path,
        collect_inspection_boxes: bool = False,
    ) -> None:
        self.profile = profile
        self.asset_dir = asset_dir
        # When true, every region read during recognize() is recorded into
        # last_inspection_boxes (captured-image pixel coordinates + score) for
        # the optional live desktop overlay. Off by default so the normal
        # latency path allocates nothing extra.
        self.collect_inspection_boxes = collect_inspection_boxes
        self.last_inspection_boxes: list[InspectionBox] = []
        # Which recognition path the last frame took, for a live status readout:
        # "fast" (targeted window OCR) or a "slow — <reason>" full-frame-OCR
        # fallback. Empty for non-CoinPoker profiles.
        self.last_recognition_path: str = ""
        # The desktop window the CoinPoker table was last read from. Sticky: a
        # frame that falls back to full-screen OCR does not clear it, because
        # the window is still the same one. Auto-play uses this instead of
        # searching for the window itself.
        self.last_table_window: WindowInfo | None = None
        self._coinpoker_hand_sequence = 0
        self._coinpoker_actions: tuple[InferredAction, ...] = ()
        self._coinpoker_hero_cards: tuple[str, ...] = ()
        self._coinpoker_board: tuple[str, ...] = ()
        # The raw (hero, board) read of the PREVIOUS frame that passed the
        # per-frame verification bar. A card set only commits to the cache / is
        # allowed to drive a decision when TWO consecutive verified frames agree
        # on it, so a lone hard-frame misread that happens to clear 0.82 (glow /
        # occlusion / animation) can never reach the record or the live view —
        # its clean neighbours disagree with it and it is dropped.
        self._coinpoker_prev_verified_cards: tuple[
            tuple[str, ...], tuple[str, ...]
        ] | None = None
        # Debug: number of committing hero-pair frames saved this session, so a
        # confident MISREAD (e.g. 4s read as 8d, hand #66) can be inspected on
        # the exact frame that produced it. Capped so it never fills the disk.
        self._hero_debug_count = 0
        self._coinpoker_button: int | None = None
        self._coinpoker_starting_stacks: tuple[int, int] | None = None
        self._coinpoker_rank_cache: dict[
            tuple[str, bytes],
            tuple[str, float],
        ] = {}
        # Region cache for the colour card scan: skips _locate_colored_cards
        # (~24 ms) when the scan area is pixel-identical to the previous frame.
        self._coinpoker_locate_cache: dict[str, Any] = {}
        self._coinpoker_fast_context: _CoinPokerFastContext | None = None

    def warm_up(self) -> None:
        """Load OCR models before the first live frame enters the latency path."""

        if self.profile.name.lower() != "coinpoker":
            return
        import numpy as np  # type: ignore

        recognize_text_strip(np.zeros((32, 128, 3), dtype=np.uint8))

    def _record_inspection(self, label: str, box: Box, confidence: float) -> None:
        """Note one inspected region for the live overlay (no-op unless enabled)."""

        if not self.collect_inspection_boxes:
            return
        self.last_inspection_boxes.append(
            InspectionBox(
                label,
                float(box.left),
                float(box.top),
                float(box.right),
                float(box.bottom),
                float(confidence),
            )
        )

    @staticmethod
    def _relative_box(bounds: Box, ratios: tuple[float, float, float, float]) -> Box:
        left, top, right, bottom = ratios
        return Box(
            bounds.left + bounds.width * left,
            bounds.top + bounds.height * top,
            bounds.left + bounds.width * right,
            bounds.top + bounds.height * bottom,
        )

    @staticmethod
    def _crop(image, bounds: Box):
        height, width = image.shape[:2]
        left = max(0, min(width - 1, int(round(bounds.left))))
        top = max(0, min(height - 1, int(round(bounds.top))))
        right = max(left + 1, min(width, int(round(bounds.right))))
        bottom = max(top + 1, min(height, int(round(bounds.bottom))))
        return image[top:bottom, left:right]

    @staticmethod
    def _strip_signature(image) -> int:
        import cv2  # type: ignore

        if image is None or image.size == 0:
            return 0
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        tiny = cv2.resize(gray, (48, 12), interpolation=cv2.INTER_AREA)
        return hash(tiny.tobytes())

    @staticmethod
    def _has_visible_text(image) -> bool:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if image is None or image.size == 0:
            return False
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        return int(np.count_nonzero(gray >= 135)) >= max(5, gray.size // 180)

    def _cached_strip(
        self,
        context: _CoinPokerFastContext,
        key: str,
        image,
        bounds: Box,
        *,
        require_visible_text: bool = False,
    ) -> tuple[str, float]:
        crop = self._crop(image, bounds)
        signature = self._strip_signature(crop)
        cached = context.fields.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        if require_visible_text and not self._has_visible_text(crop):
            result = ("", 0.0)
        else:
            result = recognize_text_strip(crop)
        context.fields[key] = (signature, result[0], result[1])
        return result

    @staticmethod
    def _clean_player_name(value: str) -> str | None:
        matches = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,23}", value)
        excluded = {
            "BB",
            "BTN",
            "SB",
            "FOLD",
            "CALL",
            "CHECK",
            "RAISE",
            "BET",
        }
        return next(
            (
                match
                for match in matches
                if match.upper() not in excluded and not match.replace(".", "").isdigit()
            ),
            None,
        )

    @staticmethod
    def _match_player(value: str, players: tuple[str, str]) -> str | None:
        cleaned = LiveTableRecognizer._clean_player_name(value)
        if cleaned is None:
            return None
        exact = next(
            (player for player in players if player.casefold() == cleaned.casefold()),
            None,
        )
        if exact is not None:
            return exact
        candidate = max(
            players,
            key=lambda player: difflib.SequenceMatcher(
                None, player.casefold(), cleaned.casefold()
            ).ratio(),
        )
        similarity = difflib.SequenceMatcher(
            None, candidate.casefold(), cleaned.casefold()
        ).ratio()
        return candidate if similarity >= 0.68 else None

    def _coinpoker_window_context(
        self,
        image,
        capture_rect: CaptureRect | None,
    ) -> _CoinPokerFastContext | None:
        if capture_rect is None:
            return None
        image_height, image_width = image.shape[:2]
        candidates: list[tuple[str, Box, int, WindowInfo]] = []
        try:
            windows = list_windows()
        except RuntimeError:
            return None
        for window in windows:
            title = window.title.strip()
            upper = title.upper()
            if "NLH" not in upper and "COINPOKER" not in upper:
                continue
            try:
                outer = window_outer_rect(window)
            except RuntimeError:
                continue
            local = Box(
                float(outer.left - capture_rect.left),
                float(outer.top - capture_rect.top),
                float(outer.left + outer.width - capture_rect.left),
                float(outer.top + outer.height - capture_rect.top),
            )
            overlap_width = max(
                0.0, min(float(image_width), local.right) - max(0.0, local.left)
            )
            overlap_height = max(
                0.0, min(float(image_height), local.bottom) - max(0.0, local.top)
            )
            area = int(overlap_width * overlap_height)
            if area <= 20_000:
                continue
            kind = "dealer" if "DEALER CHAT" in upper else "table"
            candidates.append((kind, local, area, window))
        dealer = max(
            (entry for entry in candidates if entry[0] == "dealer"),
            key=lambda entry: entry[2],
            default=None,
        )
        table = max(
            (entry for entry in candidates if entry[0] == "table"),
            key=lambda entry: entry[2],
            default=None,
        )
        if dealer is None or table is None:
            return None
        dealer_box = dealer[1]
        table_box = table[1]
        if dealer_box.width < 400 or dealer_box.height < 300:
            return None
        if table_box.width < 600 or table_box.height < 450:
            return None
        # Remember which desktop window the table reads came from. Auto-play
        # needs exactly this window, and inheriting the recognizer's choice is
        # the only reliable way to get it: CoinPoker titles both its lobby and
        # its tables "CoinPoker", so no title rule can tell them apart.
        self.last_table_window = table[3]
        rect_key = tuple(
            int(round(value))
            for value in (
                dealer_box.left,
                dealer_box.top,
                dealer_box.right,
                dealer_box.bottom,
                table_box.left,
                table_box.top,
                table_box.right,
                table_box.bottom,
            )
        )
        context = self._coinpoker_fast_context
        if context is None or context.rect_key != rect_key:
            context = _CoinPokerFastContext(
                rect_key=rect_key,
                dealer_box=dealer_box,
                table_box=table_box,
            )
            self._coinpoker_fast_context = context
        return context

    def _coinpoker_fast_lines(
        self,
        image,
        capture_rect: CaptureRect | None,
    ) -> list[OcrLine] | None:
        context = self._coinpoker_window_context(image, capture_rect)
        if context is None:
            return None
        dealer = context.dealer_box
        table = context.table_box
        lines: list[OcrLine] = []

        top_name_box = self._relative_box(table, (0.43, 0.185, 0.59, 0.225))
        bottom_name_box = self._relative_box(table, (0.43, 0.845, 0.59, 0.895))
        top_name_raw, top_name_score = self._cached_strip(
            context, "table-top-name", image, top_name_box
        )
        bottom_name_raw, bottom_name_score = self._cached_strip(
            context, "table-bottom-name", image, bottom_name_box
        )
        top_name = self._clean_player_name(top_name_raw)
        bottom_name = self._clean_player_name(bottom_name_raw)
        if bottom_name and top_name and bottom_name.casefold() != top_name.casefold():
            context.players = (bottom_name, top_name)
        players = context.players
        if players is None:
            return []
        hero_name, opponent_name = players
        if top_name is not None:
            opponent_name = self._match_player(top_name, players) or opponent_name
        if bottom_name is not None:
            hero_name = self._match_player(bottom_name, players) or hero_name
        context.players = (hero_name, opponent_name)

        lines.extend(
            (
                OcrLine(opponent_name, max(0.8, top_name_score), top_name_box),
                OcrLine(hero_name, max(0.8, bottom_name_score), bottom_name_box),
            )
        )
        self._record_inspection("opponent name", top_name_box, top_name_score)
        self._record_inspection("hero name", bottom_name_box, bottom_name_score)
        strip_labels = {
            "table-top-stack": "opponent stack",
            "table-bottom-stack": "hero stack",
            "table-pot": "pot",
        }
        for key, bounds in (
            ("table-top-stack", (0.43, 0.225, 0.59, 0.275)),
            ("table-bottom-stack", (0.43, 0.89, 0.59, 0.945)),
            ("table-pot", (0.43, 0.34, 0.60, 0.405)),
        ):
            box = self._relative_box(table, bounds)
            text, score = self._cached_strip(
                context,
                key,
                image,
                box,
                require_visible_text=True,
            )
            self._record_inspection(strip_labels[key], box, score if text else 0.0)
            if text:
                lines.append(OcrLine(text, score, box))

        street_labels = ("Pre-Flop", "Flop", "Turn", "River")
        column_width = dealer.width / 4.0
        header_top = dealer.top + dealer.height * 0.145
        header_bottom = dealer.top + dealer.height * 0.198
        for index, label in enumerate(street_labels):
            left = dealer.left + column_width * index + column_width * 0.04
            right = dealer.left + column_width * (index + 1) - column_width * 0.08
            header_box = Box(left, header_top, right, header_bottom)
            lines.append(OcrLine(label, 1.0, header_box))
            self._record_inspection(f"street {label}", header_box, 1.0)

        row_top = dealer.top + dealer.height * 0.264
        row_pitch = dealer.height * 0.113
        row_height = dealer.height * 0.094
        maximum_rows = max(
            1,
            min(
                8,
                int(
                    (
                        dealer.bottom
                        - dealer.height * 0.025
                        - row_top
                    )
                    / row_pitch
                )
                + 1,
            ),
        )
        for column, street in enumerate(("preflop", "flop", "turn", "river")):
            column_left = dealer.left + column_width * column
            for row_index in range(maximum_rows):
                top = row_top + row_pitch * row_index
                if top + row_height > dealer.bottom:
                    break
                name_box = Box(
                    column_left + column_width * 0.34,
                    top + row_height * 0.08,
                    column_left + column_width * 0.94,
                    top + row_height * 0.49,
                )
                action_box = Box(
                    column_left + column_width * 0.34,
                    top + row_height * 0.47,
                    column_left + column_width * 0.94,
                    top + row_height * 0.91,
                )
                combined = self._crop(
                    image,
                    Box(name_box.left, name_box.top, action_box.right, action_box.bottom),
                )
                signature = self._strip_signature(combined)
                cache_key = (street, row_index)
                cached = context.rows.get(cache_key)
                if cached is None or cached[0] != signature:
                    name_crop = self._crop(image, name_box)
                    if self._has_visible_text(name_crop):
                        raw_name, name_score = recognize_text_strip(name_crop)
                        matched_name = self._match_player(raw_name, context.players)
                    else:
                        matched_name = None
                        name_score = 0.0
                    if matched_name is not None:
                        action_text, action_score = recognize_text_strip(
                            self._crop(image, action_box)
                        )
                        upper_action = action_text.upper()
                        if (
                            street == "preflop"
                            and row_index == 0
                            and not re.search(
                                r"\b(?:BTN|BET|CALL|CHECK|FOLD|RAISE|ALL[ -]?IN)\b",
                                upper_action,
                            )
                        ):
                            action_text = f"BTN {action_text}".strip()
                        elif (
                            street == "preflop"
                            and row_index == 1
                            and not re.search(
                                r"\b(?:BB|BET|CALL|CHECK|FOLD|RAISE|ALL[ -]?IN)\b",
                                upper_action,
                            )
                        ):
                            action_text = f"BB {action_text}".strip()
                        score = min(name_score, action_score)
                    else:
                        action_text = ""
                        score = 0.0
                    cached = (
                        signature,
                        matched_name or "",
                        action_text,
                        score,
                    )
                    context.rows[cache_key] = cached
                _, name_text, action_text, score = cached
                if not name_text:
                    continue
                lines.append(OcrLine(name_text, max(0.75, score), name_box))
                self._record_inspection(f"{street} {name_text}", name_box, score)
                if action_text:
                    lines.append(OcrLine(action_text, max(0.7, score), action_box))
                    self._record_inspection(action_text, action_box, score)
        return sorted(lines, key=lambda line: (line.box.top, line.box.left))

    def recognize(
        self,
        image,
        capture_rect: CaptureRect | None = None,
    ) -> VisibleTableState:
        height, width = image.shape[:2]
        if self.collect_inspection_boxes:
            self.last_inspection_boxes = []
        if self.profile.name.lower() == "coinpoker":
            fast_lines = self._coinpoker_fast_lines(image, capture_rect)
            if fast_lines:
                self.last_recognition_path = "fast"
                return self._recognize_coinpoker(
                    image,
                    fast_lines,
                    width,
                    height,
                    allow_seat_fallback=False,
                )
            # No usable fast path -> the SLOW full-frame OCR fallback. Record why
            # so the status line can tell the user their setup is off: None means
            # the CoinPoker table/Dealer-Chat windows were not found on the
            # captured monitor; [] means they were found but the seats could not
            # be read.
            self.last_recognition_path = (
                "slow — CoinPoker table/Dealer-Chat windows not found on this monitor"
                if fast_lines is None
                else "slow — windows found but seats not read"
            )
            # Saved screenshots and non-Windows capture sources do not expose
            # native window rectangles, so retain the complete OCR fallback.
            # NOTE: do NOT crop the image before run_ocr to "save scan area" --
            # RapidOCR resizes internally, so a crop changes detections and can
            # silently DROP Dealer Chat action rows (measured: frame_0008 lost a
            # raise, breaking the reconstruction and the whole hand record). The
            # full frame is required for a complete, valid action timeline.
            raw_lines = run_ocr(image)
            return self._recognize_coinpoker(
                image,
                raw_lines,
                width,
                height,
                allow_seat_fallback=True,
            )
        self.last_recognition_path = ""
        raw_lines = run_ocr(image)
        content = locate_content(raw_lines, width, height, self.profile)
        boxes = {
            name: _normalized_box(bounds, content)
            for name, bounds in self.profile.regions.items()
        }
        region_lines = {
            name: _lines_in_box(raw_lines, box)
            for name, box in boxes.items()
        }
        if self.collect_inspection_boxes:
            for name, box in boxes.items():
                found = region_lines.get(name, ())
                region_confidence = (
                    sum(line.confidence for line in found) / len(found)
                    if found
                    else 0.0
                )
                self._record_inspection(name, box, region_confidence)

        hand_number = _hand_number(region_lines.get("header_hand", ()))
        street = _street(region_lines.get("header_street", ()))
        pot = _best_number(region_lines.get("header_pot", ()))
        if pot is None:
            pot = _best_number(region_lines.get("center_pot", ()))
        agent_stack = _best_number(region_lines.get("agent_stack", ()))
        hero_stack = _best_number(region_lines.get("hero_stack", ()))
        agent_bet = _best_number(region_lines.get("opponent_wager", ()))
        hero_bet = _best_number(region_lines.get("hero_wager", ()))

        status_text = " ".join(line.text for line in region_lines.get("action_status", ()))
        busy = bool(re.search(r"AGENT.*ACT|THINK|WAIT", status_text, re.IGNORECASE))
        hero_turn = bool(re.search(r"YOUR\s+ACTION", status_text, re.IGNORECASE))
        if hero_turn:
            current_player = 0
        else:
            current_player = None
        completion_text = " ".join(
            [
                status_text,
                *(line.text for line in region_lines.get("header_pot", ())),
                *(line.text for line in region_lines.get("center_pot", ())),
            ]
        )
        complete = bool(
            re.search(r"NEXT\s+HAND|LAST\s+POT|SETTLED", completion_text, re.IGNORECASE)
        )
        status_stable = (hero_turn or complete) and not busy

        agent_text = " ".join(line.text for line in region_lines.get("agent_plaque", ()))
        hero_text = " ".join(line.text for line in region_lines.get("hero_plaque", ()))
        button = 1 if re.search(r"\bBUTTON\b", agent_text, re.IGNORECASE) else None
        if re.search(r"\bBUTTON\b", hero_text, re.IGNORECASE):
            button = 0

        content_image = image[
            int(content.top) : int(content.bottom),
            int(content.left) : int(content.right),
        ]
        cards, card_warnings = detect_cards(
            content_image,
            self.asset_dir,
            self.profile.minimum_card_score,
        )
        hero_cards, opponent_cards, board = assign_cards(cards, content_image.shape[0])
        if self.collect_inspection_boxes:
            for card in cards:
                # detect_cards works on the cropped content image; shift each
                # card box back into full-frame coordinates for the overlay.
                shifted = Box(
                    content.left + card.box.left,
                    content.top + card.box.top,
                    content.left + card.box.right,
                    content.top + card.box.bottom,
                )
                self._record_inspection(f"card {card.card}", shifted, card.confidence)

        warnings = list(card_warnings)
        required = {
            "hand number": hand_number,
            "street": street,
            "pot": pot,
            "hero stack": hero_stack,
            "agent stack": agent_stack,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            warnings.append("Could not recognize: " + ", ".join(missing) + ".")
        if not hero_cards:
            warnings.append("Hero cards were not recognized in this frame.")
        stable = status_stable and not missing and len(hero_cards) == 2
        if not status_stable and not busy:
            warnings.append("The action-status area was not stable enough to track this frame.")
        elif status_stable and not stable:
            warnings.append(
                "The table is still visually settling; this incomplete frame will not be tracked."
            )

        confidences = [line.confidence for line in raw_lines]
        confidences.extend(card.confidence for card in cards)
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        if missing:
            confidence *= max(0.3, 1.0 - len(missing) * 0.12)
        return VisibleTableState(
            captured_at=datetime.now(timezone.utc).isoformat(),
            hand_number=hand_number,
            street=street,
            pot=pot,
            stacks=(hero_stack, agent_stack),
            round_bets=(hero_bet, agent_bet),
            hero_cards=tuple(hero_cards),
            opponent_cards=tuple(opponent_cards),
            board=tuple(board),
            button=button,
            current_player=current_player,
            complete=complete,
            stable=stable,
            history_stable=stable,
            decision_ready=stable,
            confidence=round(confidence, 4),
            warnings=tuple(warnings),
        )

    def _save_hero_debug(self, image, hero_cards) -> None:
        """Save the exact frame that committed a hero pair, so a confident
        misread can be inspected on the frame that produced it. ROLLING buffer:
        always keep the newest ~40 committing frames and delete older ones, so
        disk stays bounded AND a recently-reported misread is always present.
        Best-effort — never raises into the recognition path."""
        try:
            import cv2  # type: ignore

            suit_map = {"\u2660": "s", "\u2665": "h", "\u2666": "d", "\u2663": "c"}
            safe = "_".join(
                "".join(suit_map.get(ch, ch) for ch in card) for card in hero_cards
            )
            debug_dir = Path(
                r"C:\Users\dudiz\AppData\Local\Temp\claude\C--Users-dudiz-Documents-Holdem"
                r"\9e5663a2-e556-4a15-a1c0-8890726f4305\scratchpad\hero_debug"
            )
            debug_dir.mkdir(parents=True, exist_ok=True)
            name = f"hand{self._coinpoker_hand_sequence}_{safe}_{self._hero_debug_count}.png"
            cv2.imwrite(str(debug_dir / name), image)
            self._hero_debug_count += 1
            # Rolling cap: keep only the newest 40 frames by modification time.
            existing = sorted(
                debug_dir.glob("*.png"), key=lambda p: p.stat().st_mtime
            )
            for old in existing[:-40]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except Exception:
            pass

    def _recognize_coinpoker(
        self,
        image,
        raw_lines: Sequence[OcrLine],
        width: int,
        height: int,
        *,
        allow_seat_fallback: bool,
    ) -> VisibleTableState:
        # On the live fast path the table window is known, so pin ONE small box
        # over each hole card's RANK CORNER (left card, right card), anchored to
        # the table window. The rank glyph is what OCR reads, so the box only
        # needs to cover the upper-left corner — but sized to absorb the measured
        # fan/overlap drift (left-card corner moves ~29px x / ~23px y between
        # hands; right card is near-fixed). A card is selected by its top-left
        # CORNER falling in its box, so the dealer-button "D" (left of ~0.40),
        # board, chips and the action badge/glow below are never picked. Only set
        # when the fast context is valid; offline keeps the geometry-free path.
        hero_card_boxes: tuple[tuple[float, float, float, float], ...] | None = None
        context = self._coinpoker_fast_context
        if not allow_seat_fallback and context is not None:
            table = context.table_box

            def _table_box(
                left: float, top: float, right: float, bottom: float
            ) -> tuple[float, float, float, float]:
                return (
                    table.left + table.width * left,
                    table.top + table.height * top,
                    table.left + table.width * right,
                    table.top + table.height * bottom,
                )

            hero_card_boxes = (
                _table_box(0.435, 0.728, 0.495, 0.780),  # left card rank corner
                _table_box(0.496, 0.728, 0.560, 0.770),  # right card rank corner
            )
            # Draw each rank-corner box on the overlay so it can be eyeballed
            # against the real cards. Confidence -1 flags a region marker (not a
            # read), letting the overlay style it apart.
            for box in hero_card_boxes:
                self._record_inspection("hero card", Box(*box), -1.0)
            # Anchor for the recommended-action banner: the empty felt on the
            # LEFT side of the table, mid-height. Confidence -2 marks it as an
            # invisible anchor (the overlay positions the banner here but never
            # draws the box).
            # Left felt, on the SAME vertical line as the hero cards (~0.72 of
            # the table), so the banner sits bottom-left next to the cards.
            self._record_inspection(
                "decision anchor",
                Box(*_table_box(0.03, 0.72, 0.28, 0.88)),
                -2.0,
            )
        recognized = extract_coinpoker_layout(
            image,
            raw_lines,
            width,
            height,
            rank_cache=self._coinpoker_rank_cache,
            hero_card_boxes=hero_card_boxes,
            locate_cache=self._coinpoker_locate_cache,
        )
        if allow_seat_fallback and (
            len(recognized.players) < 2
            or any(stack is None for stack in recognized.current_stacks)
        ):
            focused_lines = augment_coinpoker_seat_ocr(image, raw_lines)
            recognized = extract_coinpoker_layout(
                image,
                focused_lines,
                width,
                height,
                rank_cache=self._coinpoker_rank_cache,
                hero_card_boxes=hero_card_boxes,
                locate_cache=self._coinpoker_locate_cache,
            )
        for label, left, top, right, bottom, confidence in recognized.card_boxes:
            self._record_inspection(label, Box(left, top, right, bottom), confidence)
        parsed_actions = [ParsedAction(**action) for action in recognized.actions]
        visible_actions = tuple(
            InferredAction(
                player=action.player,
                action=action.action,
                amount=action.amount,
                street=action.street,
            )
            for action in parsed_actions
        )
        # A fold ends a heads-up hand outright — the Dealer Chat shows it as a
        # FOLD row (in any column; the parser already scans the whole row), e.g.
        # "Zodiak007 BTN FOLD". Treat that as hand-complete directly from the
        # chat so the hand finalizes and records promptly instead of waiting for
        # the next hand to appear, and so no recommendation is requested for a
        # hand that is already over.
        hand_folded = any(action.action == "fold" for action in visible_actions)
        raw_hero_cards = tuple(recognized.hero_cards)
        raw_board = tuple(recognized.board)
        previous_is_prefix = (
            len(visible_actions) >= len(self._coinpoker_actions)
            and visible_actions[: len(self._coinpoker_actions)] == self._coinpoker_actions
        )
        board_reset = bool(
            self._coinpoker_board
            and (
                len(raw_board) < len(self._coinpoker_board)
                or tuple(raw_board[: len(self._coinpoker_board)])
                != tuple(self._coinpoker_board)
            )
        )
        # A confident hero pair that differs from the cached one means a NEW hand
        # (or a corrected read) — but only trust it as a reset when corroborated,
        # so a single high-confidence mid-hand misread cannot wipe a verified
        # pair. Corroboration = no board yet (fresh preflop), the board itself
        # reset, or the Dealer Chat timeline is no longer a continuation. The old
        # `not raw_board` rule blocked this whenever a board was showing, which
        # let the previous hand's hole cards persist into a hand first seen at the
        # flop and be served at fabricated confidence — acting on the wrong hand.
        confident_new_hole_cards = bool(
            len(raw_hero_cards) == 2
            and len(recognized.card_confidences) >= 2
            and min(recognized.card_confidences[:2]) >= 0.94
            and self._coinpoker_hero_cards
            and raw_hero_cards != self._coinpoker_hero_cards
            and (not raw_board or board_reset or not previous_is_prefix)
        )
        # A Dealer Chat showing ONLY the SB + BB posts, an empty board and no
        # betting actions is a freshly dealt hand (user-confirmed signal). This
        # is STATE-based, so it survives a dropped transition frame from a queue
        # overflow: if we still hold a previous hand's community board while the
        # chat has reset to blinds-only, a new hand began. Guarding on the cached
        # board — which the reset below clears immediately — means it cannot
        # re-fire while we sit in the blinds-only state at the start of the hand.
        new_hand_from_blinds = bool(
            recognized.timeline_starts_at_hand
            and not visible_actions
            and not raw_board
            and self._coinpoker_board
        )
        # A shrinking / diverging Dealer-Chat timeline is only a RELIABLE new-hand
        # signal when it bottoms out at a fresh-hand level (<= 2 rows: the SB/BB
        # posts). A mid-hand dip that stays above that is almost always a
        # transient OCR miss (a row not read this frame) — treating it as a reset
        # needlessly tore down the hero/board locks mid-hand. Requiring <= 2 here
        # keeps the locks intact through a flickery frame; genuine new hands are
        # still caught by the board-reset / blinds-only / button-change signals
        # below, which don't depend on the action count.
        timeline_reset = (
            len(visible_actions) < len(self._coinpoker_actions)
            or not previous_is_prefix
        ) and len(visible_actions) <= 2
        reset = self._coinpoker_hand_sequence > 0 and (
            timeline_reset
            or confident_new_hole_cards
            or new_hand_from_blinds
            # A board that shrank or whose prefix changed is a reliable new-hand
            # signal that does NOT depend on the (often flaky) hero-card read.
            # Within a hand the board only grows prefix-preserving, so this only
            # fires across hands — and it clears the stale hero cache below so the
            # PREVIOUS hand's hole cards can never be served into a new hand (the
            # A2 vs cached-K4 desync).
            or board_reset
            or (
                self._coinpoker_button is not None
                and recognized.button != self._coinpoker_button
                and len(visible_actions) <= 2
            )
        )
        # A RELIABLE new-hand boundary advances the hand counter. This is a
        # subset of `reset`: it excludes board_reset alone (a transient
        # board-read glitch must not split one hand into two numbers) and
        # requires an action-count drop to bottom out at <=2 (a fresh hand). The
        # counter used to advance only on a history_stable frame, but a new
        # hand's opening frames are often not yet stable, so several real hands
        # collapsed under one number (three hands merged as "Hand #9"/"#17").
        hand_boundary = self._coinpoker_hand_sequence > 0 and (
            new_hand_from_blinds
            or confident_new_hole_cards
            or (
                self._coinpoker_button is not None
                and recognized.button != self._coinpoker_button
                and len(visible_actions) <= 2
            )
            or (
                len(visible_actions) < len(self._coinpoker_actions)
                and len(visible_actions) <= 2
            )
        )
        if reset:
            if hand_boundary:
                self._coinpoker_hand_sequence += 1
            self._coinpoker_starting_stacks = None
            # Clear the cached hero pair and board so a stale previous-hand read
            # is never served into the new hand; the new hand re-reads from
            # scratch (and declines until it reads cleanly, rather than showing
            # the wrong cards). Also clear the action/button caches so the
            # boundary signals cannot immediately re-fire and double-increment.
            self._coinpoker_hero_cards = ()
            self._coinpoker_board = ()
            self._coinpoker_prev_verified_cards = None
            self._coinpoker_actions = ()
            self._coinpoker_button = None
        elif self._coinpoker_starting_stacks is not None:
            # CoinPoker frequently hides one or both seat plaques behind the
            # cards/action glow after preflop. Keep the first stack-verified
            # values authoritative instead of replacing them with the generic
            # 20.00 fallback on later streets.
            recognized.starting_stacks = list(self._coinpoker_starting_stacks)
            recognized.warnings = [
                warning
                for warning in recognized.warnings
                if warning != COINPOKER_STACK_FALLBACK_WARNING
            ]
        cards_consistent = True
        if not reset:
            if self._coinpoker_hero_cards:
                if (
                    len(raw_hero_cards) == 2
                    and raw_hero_cards != self._coinpoker_hero_cards
                ):
                    cards_consistent = False
                    recognized.warnings.append(
                        "Hero cards conflicted with the cards already verified for this hand."
                    )
                elif len(raw_hero_cards) != 2:
                    # Hero hole cards do NOT change during a hand. Once the pair
                    # is verified it is cached for the hand (and reset clears it
                    # across hands), so a LATER STREET must not re-scan or
                    # re-verify it. CoinPoker hides a hole card behind the action
                    # glow postflop, which previously forced "still being
                    # verified" and blocked the flop/turn/river recommendation
                    # (user-reported, hand #14: verified preflop, then made to
                    # re-verify on the flop). A partial (0- or 1-card) read cannot
                    # contradict the locked pair, so serve it at decision-eligible
                    # confidence; only a full 2-card read that DIFFERS (handled
                    # above) un-verifies it. The board is still verified per
                    # street separately below — only the hero pair is locked.
                    recognized.card_confidences = [
                        0.86,
                        0.86,
                        *recognized.card_confidences[len(raw_hero_cards) :],
                    ]
                recognized.hero_cards = list(self._coinpoker_hero_cards)
            if self._coinpoker_board:
                board_prefix = raw_board[: len(self._coinpoker_board)]
                if (
                    len(raw_board) < len(self._coinpoker_board)
                    or board_prefix != self._coinpoker_board
                ):
                    cards_consistent = False
                    recognized.warnings.append(
                        "Community cards conflicted with the verified board; "
                        "the frame was blocked from decision use."
                    )
                recognized.board = [
                    *self._coinpoker_board,
                    *raw_board[len(self._coinpoker_board) :],
                ]
                # Already-verified board cards do not change within the hand
                # either: lock the cached prefix at verified confidence so a
                # later street (or a glow frame) does not force a re-scan of the
                # flop it already read (same principle as the hero lock — user,
                # hand #14). Only the freshly-dealt suffix beyond the cache keeps
                # its raw per-frame confidence and must read cleanly to verify.
                confs = list(recognized.card_confidences)
                locked_through = 2 + len(self._coinpoker_board)
                for index in range(2, min(locked_through, len(confs))):
                    confs[index] = 0.86
                while len(confs) < 2 + len(recognized.board):
                    confs.append(0.86)
                recognized.card_confidences = confs
        validation = validate_history(
            parsed_actions,
            recognized.starting_stacks,
            recognized.blinds,
            recognized.button,
            recognized.hero_cards,
            recognized.opponent_cards,
            recognized.board,
        )
        warnings = list(recognized.warnings)
        warnings.extend(validation.warnings)
        if validation.error:
            warnings.append(validation.error)
        state = validation.resulting_state or {}
        reconstructed_pot = state.get("pot")
        round_bets = tuple(state.get("round_bets", (0, 0)))
        live_wager = (
            max(int(value) for value in round_bets)
            if (
                len(round_bets) == 2
                and all(value is not None for value in round_bets)
                and int(round_bets[0]) != int(round_bets[1])
            )
            else 0
        )
        visible_pot = (
            int(recognized.visible_pot)
            if recognized.visible_pot is not None
            else None
        )
        reconstructed_pot_value = (
            int(reconstructed_pot)
            if reconstructed_pot is not None
            else None
        )
        coinpoker_live_wager_total = bool(
            visible_pot is not None
            and reconstructed_pot_value is not None
            and state.get("street") != "preflop"
            and live_wager > 0
            and visible_pot == reconstructed_pot_value + live_wager
        )
        pot_matches = (
            visible_pot is None
            or reconstructed_pot_value is None
            or visible_pot == reconstructed_pot_value
            or coinpoker_live_wager_total
        )
        if not pot_matches:
            warnings.append(
                "The CoinPoker table pot does not match the visible Dealer Chat actions."
            )
        reconstructed_stacks = tuple(state.get("stacks", (None, None)))
        stacks_match = all(
            visible is None or reconstructed_stacks[index] == visible
            for index, visible in enumerate(recognized.current_stacks)
        ) if len(reconstructed_stacks) == 2 else False
        if not stacks_match:
            warnings.append(
                "The CoinPoker seat stacks do not match the visible Dealer Chat actions."
            )
        stacks_verified = bool(
            stacks_match or self._coinpoker_starting_stacks is not None
        )
        board = tuple(recognized.board)
        street_name = state.get("street")
        expected_board_cards = {
            "preflop": 0,
            "flop": 3,
            "turn": 4,
            "river": 5,
        }.get(str(street_name) if street_name is not None else "")
        expected_card_count = len(recognized.hero_cards) + len(board)
        cards_verified = bool(
            cards_consistent
            and len(recognized.hero_cards) == 2
            and (
                expected_board_cards is None
                or len(board) == expected_board_cards
            )
            and len(recognized.card_confidences) >= expected_card_count
            and all(
                # Per-card verification floor. 0.82 -> 0.78 -> 0.76: a black
                # SPADE face reads a correct rank but its dark corner OCRs low
                # (~0.77 live), so it kept failing at 0.78 and blocked every
                # decision on any hand with a spade. 2-frame corroboration + the
                # colour-derived suit guard the read; uncertain suits are still
                # capped at 0.75 in the layout, so they stay below this floor and
                # cannot verify. A card that clears this then locks at 0.86,
                # which also clears the decision-confidence gate on the next
                # frame.
                confidence >= 0.76
                for confidence in recognized.card_confidences[:expected_card_count]
            )
        )
        if not cards_verified:
            warnings.append(
                "Visible cards are still being verified; no recommendation was requested."
            )
        # Two-consecutive-frame corroboration of the raw card read. A verified
        # card set only earns decision-readiness / a cache commit when the
        # immediately preceding verified frame read the SAME raw (hero, board):
        # a lone hard-frame misread that clears 0.82 (glow / occlusion /
        # animation) is dropped because its clean neighbours disagree with it.
        # Clean cards read identically every frame, so a stable set corroborates
        # on the very next frame (~33ms at 30fps) and then stays corroborated
        # with no further lag; only the first frame of each new card set waits.
        # Key corroboration on the EFFECTIVE cards (after the cache-serve above),
        # not the raw read: before the first commit the cache is empty so this
        # equals the raw read (a lone misread still needs a second matching frame
        # to commit), but once a pair is committed, a partial frame that serves
        # the cached pair carries the same key as the full-pair frames and stays
        # corroborated instead of being re-blocked by the occluded-card frame.
        effective_cards_key = (
            tuple(recognized.hero_cards),
            tuple(recognized.board),
        )
        cards_corroborated = bool(
            cards_verified
            and self._coinpoker_prev_verified_cards == effective_cards_key
        )
        if cards_verified:
            self._coinpoker_prev_verified_cards = effective_cards_key
        # Only ~8.5% of frames read BOTH hero cards (the action glow hides one),
        # so requiring two verified frames to agree barely establishes the pair
        # within a hand and starves decisions. A SINGLE strong read — both raw
        # hole cards at >=0.90 this frame — is trustworthy enough to establish
        # the pair on its own (user directive: favour more decisions). Borderline
        # reads (0.78-0.90) still need corroboration. Either path sets `cards_trusted`,
        # which now gates the cache commit and decision-readiness.
        high_confidence_pair = bool(
            cards_verified
            and len(raw_hero_cards) == 2
            and len(recognized.card_confidences) >= 2
            and min(recognized.card_confidences[:2]) >= 0.90
        )
        cards_trusted = cards_corroborated or high_confidence_pair
        # RECORDING a hand depends on the AUTHORITATIVE Dealer Chat timeline
        # ALONE: the chat supplies the players, blinds, board and every betting
        # action, so a hand is worth tracking and writing to the hand history
        # even when the flaky *table* hero-card OCR has not verified — including
        # when we join mid-hand (the chat still shows the whole hand). Requiring
        # cards_verified here previously discarded entire valid chat hands
        # (e.g. hand 11) as "transient" with no record. cards_verified now only
        # gates the DECISION, not the record.
        # timeline_starts_at_hand is NOT required to record: if we joined a hand
        # mid-way or a long hand scrolled its blind rows off the top of the chat,
        # the visible actions + board + players are still worth capturing (a
        # partial record beats a dropped hand). A valid reconstruction with both
        # players identified is the bar. When the blinds were not seen, the
        # "Dealer Chat does not show both blind rows" warning already flags the
        # record as begun mid-hand, so nothing is silently mis-stated.
        history_stable = bool(
            validation.valid
            and len(recognized.players) == 2
        )
        stable = bool(
            history_stable
            and pot_matches
            and stacks_verified
        )
        # A recommendation needs the reconstructed rules state (from the chat
        # timeline + stack-verified stacks) AND the hero's hole cards read
        # confidently (cards_verified). The visible table pot is only a
        # cross-check and must not gate the decision.
        decision_ready = bool(
            history_stable
            and cards_verified
            and cards_trusted
            and stacks_verified
            and not hand_folded
        )

        if history_stable:
            # The very first hand starts the counter; all later increments happen
            # at the reliable hand_boundary in the reset block above.
            if self._coinpoker_hand_sequence == 0:
                self._coinpoker_hand_sequence = 1
            self._coinpoker_actions = visible_actions
            # Only cache hero cards / board when they were actually VERIFIED, so
            # an unverified frame (history_stable but not cards_verified) cannot
            # poison the reused pair or the verified board.
            if cards_verified and cards_trusted:
                newly_committed = not self._coinpoker_hero_cards
                self._coinpoker_hero_cards = tuple(recognized.hero_cards)
                self._coinpoker_board = board
                if newly_committed:
                    self._save_hero_debug(image, recognized.hero_cards)
            # Cache the board INDEPENDENTLY of the hero read. The community cards
            # are their own entity; locking them as soon as they read cleanly —
            # even while the hero pair is still flickering — stops the board from
            # shrinking on a later frame (the all-or-nothing detector dropping a
            # card), which was the source of the spurious "verified board did not
            # continue" GAP, and keeps the board verified for postflop decisions.
            # Only ever EXTEND the cached board (never shrink); reset clears it
            # across hands, and cards_consistent already guards a prefix conflict.
            board_confs = recognized.card_confidences[2 : 2 + len(board)]
            board_verified = bool(
                cards_consistent
                and expected_board_cards
                and len(board) == expected_board_cards
                and len(board_confs) == len(board)
                and all(conf >= 0.76 for conf in board_confs)
            )
            if board_verified and len(board) > len(self._coinpoker_board):
                self._coinpoker_board = board
            self._coinpoker_button = recognized.button
        if history_stable and self._coinpoker_starting_stacks is None:
            # Cache starting stacks on any VALID-history frame, not only a fully
            # `stable` one. Requiring pot_matches AND exact live-stack OCR match
            # to bootstrap was a chicken-and-egg trap: a single off-by-one stack
            # OCR early in the hand meant the starting stacks never cached, so
            # stacks_verified stayed strict forever and no decision could fire.
            # history_stable already means validate_history accepted these
            # starting stacks (a valid reconstruction), so they are trustworthy;
            # the live pot/stack readings are only cross-checks, exactly as
            # pot_matches was decoupled from the decision gate.
            verified_stacks = tuple(int(value) for value in recognized.starting_stacks)
            if len(verified_stacks) == 2 and all(value > 0 for value in verified_stacks):
                self._coinpoker_starting_stacks = verified_stacks

        # Recognition confidence gates the DECISION (runtime compares it to
        # --min-decision-confidence). Base it on the WEAKEST card we would act on
        # (the MINIMUM card confidence), not a blunt average. The old average of
        # action+card scores diluted good frames below 0.85 and blocked postflop;
        # but the mean of CARDS is also wrong the other way — postflop the five
        # high-confidence board cards mask a weak/unstable HERO read, firing a
        # decision on cards we are not sure of. The min blocks exactly that: a
        # shaky hole card keeps confidence low so no decision rides on it, while a
        # cleanly-read hand (all cards high) clears the gate.
        card_confidences = list(recognized.card_confidences)
        if card_confidences:
            confidence = min(card_confidences)
        else:
            other = recognized.action_confidences
            confidence = sum(other) / len(other) if other else 0.0
        if not validation.valid:
            confidence *= 0.65
        # NOTE: the `not stable` (pot/stack cross-check) penalty is intentionally
        # NOT applied here — decision_ready already requires history_stable +
        # verified+corroborated cards + stacks, so the flaky visible-pot match
        # must not tank a decision the cards fully support.
        street = state.get("street")
        if street is None:
            street = (
                visible_actions[-1].street
                if visible_actions
                else "flop" if len(board) >= 3 else "preflop"
            )
        pot = recognized.visible_pot
        if pot is None and reconstructed_pot is not None:
            pot = int(reconstructed_pot)
        round_bets = tuple(state.get("round_bets", (None, None)))
        if len(round_bets) != 2:
            round_bets = (None, None)
        current_player = state.get("current_player")
        complete = bool(state.get("complete", False)) or hand_folded
        return VisibleTableState(
            captured_at=datetime.now(timezone.utc).isoformat(),
            hand_number=self._coinpoker_hand_sequence or None,
            street=str(street) if street is not None else None,
            pot=pot,
            stacks=(
                reconstructed_stacks
                if len(reconstructed_stacks) == 2
                else tuple(recognized.current_stacks)
            ),
            round_bets=round_bets,
            hero_cards=tuple(recognized.hero_cards),
            opponent_cards=tuple(recognized.opponent_cards),
            board=board,
            button=recognized.button,
            current_player=int(current_player) if current_player is not None else None,
            complete=complete,
            stable=stable,
            history_stable=history_stable,
            decision_ready=decision_ready,
            confidence=round(confidence, 4),
            warnings=tuple(dict.fromkeys(warnings)),
            source_layout="coinpoker",
            players=tuple(recognized.players or ("Hero", "Opponent")),
            starting_stacks=tuple(recognized.starting_stacks),
            visible_actions=visible_actions,
            timeline_starts_at_hand=recognized.timeline_starts_at_hand,
            blinds=(
                (int(recognized.blinds[0]), int(recognized.blinds[1]))
                if len(recognized.blinds) == 2
                else (0, 0)
            ),
        )


def frame_change_score(previous, current) -> float:
    if previous is None or previous.shape != current.shape:
        return float("inf")
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame change detection.") from exc
    old = cv2.resize(previous, (64, 36), interpolation=cv2.INTER_AREA)
    new = cv2.resize(current, (64, 36), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(old, new)))


def profile_change_score(previous, current, profile: ScreenProfile) -> float:
    """Return the strongest change in poker-specific screen regions."""

    if previous is None or previous.shape != current.shape:
        return float("inf")
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame change detection.") from exc

    height, width = current.shape[:2]
    full_image = Box(0, 0, width, height)
    content = _normalized_box(profile.fallback_content_bounds, full_image)
    if profile.name.lower() == "coinpoker":
        focus_bounds = list(profile.regions.values())
    else:
        focus_bounds = [
            profile.regions[name]
            for name in (
                "header_hand",
                "header_street",
                "header_pot",
                "action_status",
                "opponent_wager",
                "hero_wager",
            )
            if name in profile.regions
        ]
    # Board and hero cards get tighter focus than the broad table region so a
    # single revealed card is not diluted by the static felt around it.
    if profile.name.lower() != "coinpoker":
        focus_bounds.extend(
            (
                (0.36, 0.35, 0.64, 0.66),
                (0.35, 0.66, 0.60, 0.91),
            )
        )

    scores: list[float] = []
    for bounds in focus_bounds:
        box = _clip_box(_normalized_box(bounds, content), width, height)
        left = int(box.left)
        top = int(box.top)
        right = int(box.right)
        bottom = int(box.bottom)
        if right <= left or bottom <= top:
            continue
        old_crop = previous[top:bottom, left:right]
        new_crop = current[top:bottom, left:right]
        old_small = cv2.resize(old_crop, (96, 48), interpolation=cv2.INTER_AREA)
        new_small = cv2.resize(new_crop, (96, 48), interpolation=cv2.INTER_AREA)
        scores.append(float(np.mean(cv2.absdiff(old_small, new_small))))
    return max(scores, default=0.0)


def _known_pair(values: Sequence[int | None]) -> tuple[int, int] | None:
    if len(values) != 2 or any(value is None for value in values):
        return None
    return int(values[0]), int(values[1])


def _inject_known_cards(game: HeadsUpHoldem, state: VisibleTableState) -> None:
    known_hero = [parse_card(card) for card in state.hero_cards]
    known_opponent = [parse_card(card) for card in state.opponent_cards]
    known_board = [parse_card(card) for card in state.board]
    known = known_hero + known_opponent + known_board
    if len(set(known)) != len(known):
        return
    available = [card for card in new_deck() if card not in set(known)]
    while len(known_hero) < 2:
        known_hero.append(available.pop(0))
    while len(known_opponent) < 2:
        known_opponent.append(available.pop(0))
    game.hole_cards = [known_hero[:2], known_opponent[:2]]
    if known_board:
        future = [card for card in available if card not in set(known_board)]
        game.deck = future + list(reversed(known_board))


def _visible_matches(game: HeadsUpHoldem, target: VisibleTableState) -> bool:
    if target.street is not None and game.active_street != target.street:
        return False
    stacks = _known_pair(target.stacks)
    if stacks is not None and tuple(game.stacks) != stacks:
        return False
    bets = _known_pair(target.round_bets)
    if bets is not None and tuple(game.round_bets) != bets:
        return False
    if target.pot is not None:
        displayed_pot = game.last_pot if game.hand_complete else game.pot
        if displayed_pot != target.pot:
            return False
    if target.current_player is not None and game.current_player != target.current_player:
        return False
    if target.complete and not game.hand_complete:
        return False
    return True


def _normalized_action(game: HeadsUpHoldem, action: str, amount: int | None) -> InferredAction:
    player = int(game.current_player or 0)
    street = game.active_street
    if action == "all_in":
        to_call = game.to_call(player)
        maximum = game.round_bets[player] + game.stacks[player]
        if game.stacks[player] <= to_call:
            return InferredAction(player, "call", min(to_call, game.stacks[player]), street)
        return InferredAction(player, "raise", maximum, street)
    if action == "call":
        return InferredAction(player, action, min(game.to_call(player), game.stacks[player]), street)
    return InferredAction(player, action, amount, street)


def _candidate_actions(game: HeadsUpHoldem, target: VisibleTableState) -> list[tuple[str, int | None]]:
    player = game.current_player
    if player is None:
        return []
    legal = game.legal_actions(player)
    candidates: list[tuple[str, int | None]] = []
    for action in ("fold", "check", "call"):
        if legal.get(action):
            candidates.append((action, None))
    if legal.get("raise"):
        amounts = {int(legal["raise_min"]), int(legal["raise_max"])}
        target_bets = _known_pair(target.round_bets)
        if target_bets is not None and target.street == game.active_street:
            amounts.add(target_bets[player])
        target_stacks = _known_pair(target.stacks)
        if target_stacks is not None and target_stacks[player] <= game.stacks[player]:
            amounts.add(game.round_bets[player] + game.stacks[player] - target_stacks[player])
        minimum = int(legal["raise_min"])
        maximum = int(legal["raise_max"])
        for amount in sorted(value for value in amounts if minimum <= value < maximum):
            candidates.append(("raise", amount))
    if legal.get("all_in"):
        candidates.append(("all_in", None))
    return candidates


def infer_transition(
    game: HeadsUpHoldem,
    target: VisibleTableState,
    maximum_actions: int = 4,
) -> tuple[TransitionResult, HeadsUpHoldem | None]:
    if _visible_matches(game, target):
        return TransitionResult(status="unchanged"), game
    queue: list[tuple[HeadsUpHoldem, list[InferredAction]]] = [(copy.deepcopy(game), [])]
    matches: dict[tuple[Any, ...], tuple[HeadsUpHoldem, list[InferredAction]]] = {}
    target_street = STREET_INDEX.get(target.street or "", 99)

    while queue:
        candidate_game, path = queue.pop(0)
        if len(path) >= maximum_actions or candidate_game.hand_complete:
            continue
        for action, amount in _candidate_actions(candidate_game, target):
            next_game = copy.deepcopy(candidate_game)
            normalized = _normalized_action(next_game, action, amount)
            try:
                next_game.act(int(next_game.current_player), action, amount)
            except (InvalidAction, ValueError, TypeError):
                continue
            next_path = [*path, normalized]
            if _visible_matches(next_game, target):
                key = tuple((item.player, item.action, item.amount, item.street) for item in next_path)
                matches[key] = (next_game, next_path)
                continue
            if STREET_INDEX.get(next_game.active_street, 99) > target_street:
                continue
            target_stacks = _known_pair(target.stacks)
            if target_stacks is not None and not next_game.hand_complete:
                if any(next_game.stacks[index] < target_stacks[index] for index in (0, 1)):
                    continue
            queue.append((next_game, next_path))

    if len(matches) == 1:
        matched_game, path = next(iter(matches.values()))
        return TransitionResult(status="unique", actions=path, candidates=1), matched_game
    if matches:
        return (
            TransitionResult(
                status="ambiguous",
                candidates=len(matches),
                warning=f"{len(matches)} legal action sequences match the next visible state.",
            ),
            None,
        )
    return (
        TransitionResult(
            status="unmatched",
            warning="No legal action sequence connects the previous and current visible states.",
        ),
        None,
    )


class ScreenHistoryWriter:
    def __init__(self, output_directory: Path = DEFAULT_OUTPUT_DIRECTORY) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.event_path = self.output_directory / "live-events.jsonl"
        self._write_lock = threading.Lock()

    def event(self, event: str, **payload: Any) -> None:
        record = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self._write_lock:
            with open(self.event_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def hand(self, tracked: TrackedHand) -> tuple[Path, Path]:
        label = str(tracked.hand_number) if tracked.hand_number is not None else "unknown"
        json_path = self.output_directory / f"hand-{label}.json"
        text_path = self.output_directory / f"hand-{label}.txt"
        payload = tracked.payload()
        with self._write_lock:
            json_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        def amount(value: int) -> str:
            if tracked.amount_scale == 1:
                return str(value)
            return f"{value / tracked.amount_scale:.2f}"

        players = (
            tracked.players
            if len(tracked.players) == 2
            else ("Hero", "Opponent")
        )
        lines = [
            f"Hand #{label}",
            f"Source: {tracked.source_layout}",
            f"Button: {players[tracked.button] if tracked.button is not None else 'unknown'}",
            f"Blinds: {amount(tracked.blinds[0])}/{amount(tracked.blinds[1])}",
            f"Hero: {players[0]} {' '.join(tracked.hero_cards) if tracked.hero_cards else 'unknown'}",
            "",
        ]
        for action in tracked.actions:
            action_amount = (
                f" {amount(action.amount)}" if action.amount is not None else ""
            )
            lines.append(
                f"{action.street.title()}: {players[action.player]} "
                f"{action.action}{action_amount}"
            )
        if tracked.decisions:
            lines.extend(["", "Brain decisions:"])
            for decision in tracked.decisions:
                decision_amount = (
                    f" {amount(int(decision['amount']))}"
                    if decision.get("amount") is not None
                    else ""
                )
                lines.append(
                    f"- {decision.get('action', 'unknown')}{decision_amount} "
                    f"({decision.get('model', 'unknown model')})"
                )
        reported_warnings = tracked.reported_warnings()
        if reported_warnings:
            lines.extend(["", "Warnings:", *(f"- {warning}" for warning in reported_warnings)])
        with self._write_lock:
            text_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return json_path, text_path


class LiveHandTracker:
    def __init__(
        self,
        writer: ScreenHistoryWriter,
        blinds: tuple[int, int] = (10, 20),
        maximum_transition_actions: int = 4,
    ) -> None:
        self.writer = writer
        self.blinds = blinds
        self.maximum_transition_actions = maximum_transition_actions
        self.current: TrackedHand | None = None
        self.last_state: VisibleTableState | None = None

    def add_decision(self, hand_number: int | None, payload: dict[str, Any]) -> bool:
        """Attach a non-stale brain result to the active, unfinished hand."""

        tracked = self.current
        if tracked is None or tracked.finalized or tracked.hand_number != hand_number:
            return False
        decision_id = payload.get("decision_id")
        if any(item.get("decision_id") == decision_id for item in tracked.decisions):
            return False
        tracked.decisions.append(payload)
        return True

    def _initial_stacks(self, state: VisibleTableState) -> tuple[int | None, int | None]:
        if state.source_layout == "coinpoker":
            return state.starting_stacks
        stacks = _known_pair(state.stacks)
        bets = _known_pair(state.round_bets)
        if stacks is None or state.street != "preflop":
            return state.stacks
        if state.button is None:
            return (
                (stacks[0] + bets[0], stacks[1] + bets[1])
                if bets is not None
                else state.stacks
            )
        forced = (
            self.blinds[0] if state.button == 0 else self.blinds[1],
            self.blinds[0] if state.button == 1 else self.blinds[1],
        )
        contributions = (
            (max(bets[0], forced[0]), max(bets[1], forced[1]))
            if bets is not None
            else forced
        )
        return stacks[0] + contributions[0], stacks[1] + contributions[1]

    def _effective_blinds(self, state: VisibleTableState) -> tuple[int, int]:
        """Blinds to reconstruct with: the ones auto-detected from the Dealer
        Chat when available, else the launch --blinds fallback. Reconstruction
        must use the table's actual blinds, or a table at a different blind level
        parses valid actions yet fails to build an engine (no recommendation)."""
        detected = getattr(state, "blinds", (0, 0))
        if (
            state.source_layout == "coinpoker"
            and len(detected) == 2
            and all(int(value) > 0 for value in detected)
        ):
            sb, bb = int(detected[0]), int(detected[1])
            stacks = getattr(state, "starting_stacks", ()) or ()
            min_stack = min(
                (int(s) for s in stacks if int(s) > 0), default=0
            )
            # A blind can never exceed a player's stack, and in heads-up the big
            # blind is twice the small blind. A Dealer-Chat OCR slip (an 80.02
            # stack read as the BB, i.e. blinds [1, 8002]) violates both — it
            # must never reach the recorded hand. Keep a plausible small blind
            # and derive bb = 2*sb; only when the small blind is itself
            # implausible do we fall back to the launch --blinds value.
            sb_ok = sb > 0 and (min_stack == 0 or sb <= min_stack)
            bb_ok = sb <= bb <= 3 * sb and (min_stack == 0 or bb <= min_stack)
            if sb_ok and bb_ok:
                return (sb, bb)
            if sb_ok:
                return (sb, sb * 2)
        return self.blinds

    def _new_engine(self, tracked: TrackedHand, state: VisibleTableState) -> HeadsUpHoldem | None:
        starting = _known_pair(tracked.starting_stacks)
        if (
            starting is None
            or tracked.button is None
            or (
                state.source_layout != "coinpoker"
                and state.street != "preflop"
            )
        ):
            return None
        game = HeadsUpHoldem(
            initial_stack=max(starting),
            small_blind=tracked.blinds[0],
            big_blind=tracked.blinds[1],
        )
        game.stacks = list(starting)
        game.hand_number = 0
        game.button_offset = tracked.button
        game.new_hand()
        # Preserve the visible hand identity so serving-agent caches cannot
        # collide across separate screen-tracked hands.
        game.hand_number = tracked.hand_number or game.hand_number
        _inject_known_cards(game, state)
        if state.source_layout == "coinpoker":
            for action in state.visible_actions:
                if (
                    game.current_player != action.player
                    or game.active_street != action.street
                ):
                    return None
                try:
                    game.act(action.player, action.action, action.amount)
                except (InvalidAction, TypeError, ValueError):
                    return None
            if not _visible_matches(game, state):
                return None
        return game

    def _start(self, state: VisibleTableState) -> TrackedHand:
        tracked = TrackedHand(
            hand_number=state.hand_number,
            started_at=state.captured_at,
            button=state.button,
            blinds=self._effective_blinds(state),
            starting_stacks=self._initial_stacks(state),
            hero_cards=state.hero_cards,
            source_layout=state.source_layout,
            players=state.players,
            amount_scale=100 if state.source_layout == "coinpoker" else 1,
            actions=(
                list(state.visible_actions)
                if state.source_layout == "coinpoker"
                else []
            ),
            observations=[state],
            warnings=list(state.warnings),
            complete=state.complete,
        )
        tracked.engine = (
            self._new_engine(tracked, state)
            if state.source_layout != "coinpoker" or state.stable
            else None
        )
        if tracked.engine is None:
            if state.source_layout == "coinpoker":
                tracked.warnings.append(
                    "Dealer Chat history is available, but decision reconstruction "
                    "is waiting for a stack-verified frame."
                )
            else:
                tracked.warnings.append(
                    "Tracking began after preflop or without enough visible state; "
                    "earlier actions are unknown."
                )
        elif state.source_layout != "coinpoker":
            transition, matched = infer_transition(
                tracked.engine,
                state,
                self.maximum_transition_actions,
            )
            if transition.status == "unique" and matched is not None:
                tracked.actions.extend(transition.actions)
                tracked.engine = matched
            elif transition.status not in {"unchanged", "unique"}:
                tracked.warnings.append(transition.warning or "Initial state could not be reconstructed.")
                tracked.engine = None
        self.writer.event("hand_started", state=asdict(state))
        return tracked

    def observe(self, state: VisibleTableState) -> TransitionResult:
        history_stable = (
            state.history_stable
            if state.source_layout == "coinpoker"
            else state.stable
        )
        if not history_stable:
            self.writer.event("transient_frame", state=asdict(state))
            return TransitionResult(
                status="transient",
                warning="The simulator is still acting; this frame was not added to hand state.",
            )
        if self.last_state is not None and state.state_key() == self.last_state.state_key():
            return TransitionResult(status="unchanged")

        if self.current is None or (
            state.hand_number is not None
            and self.current.hand_number is not None
            and state.hand_number != self.current.hand_number
        ):
            if self.current is not None and not self.current.finalized:
                self.finalize("next hand appeared")
            self.current = self._start(state)
            self.last_state = state
            if state.complete:
                self.finalize("complete state recognized")
            return TransitionResult(status="new_hand")

        tracked = self.current
        tracked.observations.append(state)
        # Per-frame recognition warnings are NOT accumulated here: early frames
        # in a hand routinely report transient misreads (flickering pot, a
        # half-drawn hero pair) that later stable frames resolve. Unioning them
        # made every finalized hand look broken. tracked.warnings now holds only
        # hand-level/structural warnings; the reported set is reconciled against
        # the freshest good observation in TrackedHand.reported_warnings().
        if state.source_layout == "coinpoker":
            previous_actions = tuple(tracked.actions)
            authoritative = tuple(state.visible_actions)
            if (
                len(authoritative) >= len(previous_actions)
                and authoritative[: len(previous_actions)] == previous_actions
            ):
                # Timeline grew (or is unchanged): adopt it.
                added = list(authoritative[len(previous_actions) :])
                result = TransitionResult(
                    status="unique" if added else "unchanged",
                    actions=added,
                    candidates=1 if added else 0,
                )
                tracked.actions = list(authoritative)
            elif len(authoritative) < len(previous_actions):
                # This frame read FEWER Dealer-Chat rows than we have already
                # recorded. The chat only grows within a hand, so this is a
                # transient OCR miss (a row not read this frame), NOT a real
                # change — KEEP the fuller recorded timeline so a hero (or any)
                # action already seen is never dropped. A genuine new hand comes
                # in via a hand_number change (handled above), not here.
                result = TransitionResult(status="unchanged")
            else:
                # Same or greater length but a diverging prefix: a later, cleaner
                # read corrected an earlier misparse (e.g. an amount-less raise
                # that just gained its amount). Adopt the corrected timeline.
                result = TransitionResult(
                    status="resynced",
                    actions=list(authoritative),
                    candidates=1,
                    warning="The full visible Dealer Chat timeline replaced an earlier OCR reading.",
                )
                tracked.warnings.append(result.warning)
                tracked.actions = list(authoritative)

            # Keep the STRONGEST hero read seen this hand. tracked.hero_cards is
            # seeded from the first frame, which is often a half-drawn single
            # card (recorded as e.g. ["7♦"]). Upgrade to a full pair when one
            # appears, and let a corroborated (decision_ready) pair correct an
            # earlier uncorroborated read — but never downgrade a known pair back
            # to a partial frame.
            hero_read = list(state.hero_cards)
            if len(hero_read) == 2 and (
                len(tracked.hero_cards) != 2 or state.decision_ready
            ):
                tracked.hero_cards = hero_read

            if state.decision_ready:
                # decision_ready == history_stable AND stacks_verified, so the
                # starting stacks and action timeline are trustworthy even when
                # the visible pot OCR momentarily disagrees. Rebuild the engine
                # here (not only on fully-stable frames) so pot-flicker frames
                # can still produce a recommendation.
                if tracked.engine is None:
                    tracked.starting_stacks = self._initial_stacks(state)
                    tracked.button = state.button
                rebuilt = self._new_engine(tracked, state)
                if rebuilt is None:
                    tracked.warnings.append(
                        "The current stack-verified frame could not rebuild a "
                        "decision rules state."
                    )
                tracked.engine = rebuilt
            else:
                # Dealer Chat remains authoritative for history, but an
                # unverified stack reading must invalidate any engine used for
                # recommendations.
                tracked.engine = None
        elif tracked.engine is None:
            result = TransitionResult(
                status="untracked",
                warning="Rules-engine tracking is unavailable for this partial hand.",
            )
        else:
            result, matched = infer_transition(
                tracked.engine,
                state,
                self.maximum_transition_actions,
            )
            if result.status == "unique" and matched is not None:
                tracked.actions.extend(result.actions)
                tracked.engine = matched
            elif result.status in {"ambiguous", "unmatched"}:
                tracked.warnings.append(result.warning or result.status)
                tracked.engine = None
        tracked.complete = tracked.complete or state.complete
        self.last_state = state
        self.writer.event("state_changed", state=asdict(state), transition=asdict(result))
        if state.complete and not tracked.finalized:
            self.finalize("complete state recognized")
        return result

    # Reasons that prove the hand actually ended. A subsequent hand cannot be
    # dealt until the previous one is resolved (fold or showdown), so "next hand
    # appeared" is as conclusive as an explicitly recognized terminal state.
    _HAND_ENDED_REASONS = frozenset(
        {"next hand appeared", "complete state recognized"}
    )

    def finalize(self, reason: str) -> tuple[Path, Path] | None:
        if self.current is None or self.current.finalized:
            return None
        # Final re-merge: adopt the last observed frame's Dealer-Chat timeline
        # when it extends what we recorded, so a trailing action that landed on
        # the very last frame before the hand ended is never lost. Grows only —
        # a shorter last frame (transient miss) leaves the record untouched.
        last_state = self.last_state
        if (
            last_state is not None
            and last_state.source_layout == "coinpoker"
            and last_state.hand_number == self.current.hand_number
        ):
            latest = list(last_state.visible_actions)
            recorded = self.current.actions
            if len(latest) > len(recorded) and latest[: len(recorded)] == recorded:
                self.current.actions = latest
        self.current.finalized = True
        engine = self.current.engine
        engine_complete = bool(engine is not None and engine.hand_complete)
        hand_ended = reason in self._HAND_ENDED_REASONS or engine_complete
        if hand_ended:
            # The hand is over. Mark it complete even if we never captured the
            # exact terminal frame, since fast tables reset before the final
            # fold/showdown lingers on screen.
            self.current.complete = True
            if not engine_complete:
                self.current.warnings.append(
                    "The winning action was not captured before the next hand "
                    "began; the recorded hand may omit its final street."
                )
        else:
            # Only shutdown/capture-failure paths reach here mid-hand.
            self.current.warnings.append(
                f"Hand finalized before completing: {reason}."
            )
        paths = self.writer.hand(self.current)
        self.writer.event(
            "hand_finalized",
            reason=reason,
            json_path=str(paths[0]),
            text_path=str(paths[1]),
        )
        return paths
