"""Recognition helpers for CoinPoker Dealer Chat screenshots.

The Dealer Chat window is the authoritative visible action timeline. The table
window supplies the hero/opponent identity, current stacks, and four-color
cards. This adapter intentionally returns only information that is visible in
the pixels; an active hand remains partial.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from statistics import median
from typing import Any, Sequence


SUITS = {
    "spades": "\u2660",
    "hearts": "\u2665",
    "diamonds": "\u2666",
    "clubs": "\u2663",
}
_VALID_SUITS = frozenset(SUITS.values())
STREETS = ("preflop", "flop", "turn", "river")
STREET_LABELS = {
    "PRE-FLOP": "preflop",
    "PREFLOP": "preflop",
    "FLOP": "flop",
    "TURN": "turn",
    "RIVER": "river",
}
NON_NAME_TEXT = {
    "ALL-IN",
    "BB",
    "BET",
    "BTN",
    "CALL",
    "CHECK",
    "COINPOKER",
    "DEALER CHAT",
    "FLOP",
    "FOLD",
    "HIGH CARD",
    "MAX",
    "NLH",
    "POT",
    "PRE-FLOP",
    "PREFLOP",
    "RAISE",
    "RESET",
    "RIVER",
    "SB",
    "SEARCH",
    "SPLASH",
    "SWITCH TO BB",
    "TURN",
}


@dataclass
class CoinPokerLayoutResult:
    chat_lines: list[Any]
    players: list[str] = field(default_factory=list)
    hand_number: int | None = None
    button: int | None = None
    blinds: list[int] = field(default_factory=lambda: [2, 5])
    starting_stacks: list[int] = field(default_factory=lambda: [2_000, 2_000])
    current_stacks: list[int | None] = field(default_factory=lambda: [None, None])
    visible_pot: int | None = None
    hero_cards: list[str] = field(default_factory=list)
    opponent_cards: list[str] = field(default_factory=list)
    board: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    timeline_starts_at_hand: bool = False
    action_confidences: list[float] = field(default_factory=list)
    card_confidences: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # (label, left, top, right, bottom, confidence) for every card region the
    # detector inspected this frame, in captured-image pixel coordinates. Only
    # populated for the optional live inspection overlay; empty otherwise.
    card_boxes: list[tuple[str, float, float, float, float, float]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class _CardBox:
    left: int
    top: int
    right: int
    bottom: int
    suit: str
    fill: float
    is_dark: bool = False
    suit_confidence: float = 1.0
    uncertain: bool = False
    inferred: bool = False

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u2013", "-").replace("\u2014", "-")).strip()


def _key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def detect_coinpoker_layout(lines: Sequence[Any]) -> bool:
    keys = {_key(line.text) for line in lines}
    has_chat = any("DEALERCHAT" in key for key in keys)
    has_preflop = any(key == "PREFLOP" for key in keys)
    has_postflop = any(key in {"FLOP", "TURN", "RIVER"} for key in keys)
    return has_chat and has_preflop and has_postflop


def _decimal_units(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(?<!\d)(\d+)[.,](\d{1,2})(?!\d)", value)
    if not match:
        return None
    try:
        amount = Decimal(f"{match.group(1)}.{match.group(2)}")
    except InvalidOperation:
        return None
    return int(amount * 100)


def _fuzzy_street_label(label: str) -> str | None:
    """Fuzzy map of a street-header OCR string, keyed on the FIRST letter (the
    four headers have distinct initials P-re/F-lop/T-urn/R-iver). The stylized
    header font mangles the tail (e.g. "TURN" -> "TUM"), dropping a column. Only
    used for tokens that sit ON the established header row (see _street_headers),
    so stray table text ("TIME", desktop icons) cannot invent a phantom column.
    """
    if not (3 <= len(label) <= 8):
        return None
    if label.startswith("PRE"):
        return "preflop"
    if label[0] == "F" and "L" in label:
        return "flop"
    if label[0] == "T" and label != "TIME":
        return "turn"
    if label[0] == "R" and "V" in label:
        return "river"
    return None


def _street_headers(lines: Sequence[Any]) -> list[tuple[str, Any]]:
    # Pass 1: exact matches establish the header ROW (its y position).
    result: dict[str, Any] = {}
    for line in lines:
        label = _normalized(line.text).upper().replace(" ", "")
        street = STREET_LABELS.get(label)
        if street is not None and street not in result:
            result[street] = line
    # Pass 2: fuzzy-recover a missing column (e.g. a mangled "Turn"), but ONLY
    # from a token sitting on the same header row as the exact matches, so a
    # stray "T"/"R"/"F" word elsewhere on screen cannot fabricate a phantom
    # column and wreck the per-street x-bands (which emptied the action list).
    if result:
        tops = sorted(line.box.top for line in result.values())
        heights = sorted(line.box.height for line in result.values())
        header_top = tops[len(tops) // 2]
        header_h = heights[len(heights) // 2] or 20.0
        for line in lines:
            label = _normalized(line.text).upper().replace(" ", "")
            if STREET_LABELS.get(label) is not None:
                continue
            street = _fuzzy_street_label(label)
            if street is None or street in result:
                continue
            if abs(line.box.top - header_top) <= max(30.0, header_h * 1.5):
                result[street] = line
    return [(street, result[street]) for street in STREETS if street in result]


def _chat_right(lines: Sequence[Any], headers: Sequence[tuple[str, Any]], width: int) -> float:
    table_titles = [
        line
        for line in lines
        if line.box.left > width * 0.25
        and re.search(r"\bNLH\b.*\d+(?:[.,]\d+)?\s*[/\\].*\d+(?:[.,]\d+)?", line.text, re.IGNORECASE)
        and "DEALER" not in line.text.upper()
    ]
    if table_titles:
        return max(1.0, min(line.box.left for line in table_titles) - 18.0)
    if any(
        _key(line.text) == "NLH" and line.box.left < width * 0.12
        for line in lines
    ):
        # A table-only screenshot has no Dealer Chat divider. Treat the whole
        # image as table content instead of excluding its left 42 percent.
        return 0.0
    centers = sorted(line.box.center_x for _, line in headers)
    if not centers:
        has_dealer_chat = any(
            "DEALERCHAT" in _key(line.text)
            for line in lines
        )
        return width * 0.42 if has_dealer_chat else 0.0
    spacing = median(
        centers[index] - centers[index - 1] for index in range(1, len(centers))
    ) if len(centers) > 1 else width * 0.1
    return min(float(width), centers[-1] + spacing * 0.85)


def _is_name(value: str) -> bool:
    text = _normalized(value)
    upper = text.upper()
    if upper in NON_NAME_TEXT or any(token in upper for token in ("POT ", "SPLASH ", "%")):
        return False
    if _decimal_units(text) is not None or re.fullmatch(r"\d+", text):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_. -]{1,23}", text))


def _identify_players(
    lines: Sequence[Any],
    chat_right: float,
    header_bottom: float,
    height: int,
) -> tuple[list[str], dict[str, Any], list[str]]:
    warnings: list[str] = []
    chat_names = [
        _normalized(line.text)
        for line in lines
        if line.box.right < chat_right
        and line.box.top > header_bottom
        and line.box.top < height * 0.68
        and _is_name(line.text)
    ]
    counts = Counter(chat_names)
    table_names = {
        _normalized(line.text)
        for line in lines
        if line.box.left >= chat_right and _is_name(line.text)
    }
    candidates = sorted(
        (
            name
            for name, count in counts.items()
            if count >= 2 or name in table_names
        ),
        key=lambda name: (name in table_names, counts[name]),
        reverse=True,
    )[:2]
    if len(candidates) < 2:
        warnings.append("Could not identify both Dealer Chat player names reliably.")
        return [], {}, warnings

    table_occurrences = {
        name: [
            line
            for line in lines
            if line.box.left >= chat_right and _normalized(line.text) == name
        ]
        for name in candidates
    }
    lower_name = max(
        (
            (line.box.center_y, name, line)
            for name, occurrences in table_occurrences.items()
            for line in occurrences
        ),
        # Compare by vertical position ONLY. Without a key, max() falls through
        # to comparing the trailing OcrLine when two occurrences tie on
        # (center_y, name) — a duplicate name read produces exactly that tie and
        # raised "'>' not supported between instances of 'OcrLine'", crashing the
        # recognition worker and stopping the watcher (dropping every later hand).
        key=lambda item: item[0],
        default=None,
    )
    if lower_name is None:
        warnings.append("Hero identity could not be confirmed from the lower table seat.")
        return candidates, {}, warnings

    hero_name = lower_name[1]
    opponent_name = next(name for name in candidates if name != hero_name)
    anchors = {
        hero_name: lower_name[2],
        opponent_name: min(
            table_occurrences.get(opponent_name, []),
            key=lambda line: line.box.center_y,
            default=None,
        ),
    }
    return [hero_name, opponent_name], anchors, warnings


def _column_bounds(
    headers: Sequence[tuple[str, Any]],
    chat_right: float,
) -> dict[str, tuple[float, float, float]]:
    ordered = sorted(headers, key=lambda item: item[1].box.center_x)
    result: dict[str, tuple[float, float, float]] = {}
    for index, (street, line) in enumerate(ordered):
        left = (
            0.0
            if index == 0
            else (ordered[index - 1][1].box.center_x + line.box.center_x) / 2.0
        )
        right = (
            chat_right
            if index == len(ordered) - 1
            else (line.box.center_x + ordered[index + 1][1].box.center_x) / 2.0
        )
        result[street] = (left, right, line.box.bottom)
    return result


def _parse_actions(
    lines: Sequence[Any],
    players: Sequence[str],
    columns: dict[str, tuple[float, float, float]],
    chat_right: float,
) -> tuple[int | None, list[int], list[dict[str, Any]], bool, list[float], list[str]]:
    button: int | None = None
    small_blind: int | None = None
    big_blind: int | None = None
    actions: list[dict[str, Any]] = []
    confidences: list[float] = []
    warnings: list[str] = []
    player_index = {name: index for index, name in enumerate(players)}

    for street in STREETS:
        if street not in columns:
            continue
        left, right, header_bottom = columns[street]
        names = sorted(
            (
                line
                for line in lines
                if left <= line.box.center_x < right
                and line.box.right < chat_right
                and line.box.top > header_bottom + 8.0
                and _normalized(line.text) in player_index
            ),
            key=lambda line: line.box.center_y,
        )
        for row_index, name_line in enumerate(names):
            row_bottom = (
                names[row_index + 1].box.top - 2.0
                if row_index + 1 < len(names)
                else name_line.box.bottom + 58.0
            )
            row = [
                line
                for line in lines
                if left <= line.box.center_x < right
                and line.box.right < chat_right
                and name_line.box.top - 4.0 <= line.box.center_y <= row_bottom
            ]
            row_text = " ".join(
                _normalized(line.text) for line in sorted(row, key=lambda line: (line.box.top, line.box.left))
            )
            upper = row_text.upper()
            actor = player_index[_normalized(name_line.text)]
            amount_candidates = [
                (line, _decimal_units(line.text))
                for line in row
                if _decimal_units(line.text) is not None
            ]
            amount = amount_candidates[-1][1] if amount_candidates else None

            if re.search(r"\bANTE\b", upper):
                # An ante row is neither a blind nor a betting action. Skip it
                # entirely so a 0.2 ante is never mistaken for the small/big
                # blind: on ante tables (e.g. "0.5/1 (0.2)") the ante rows carry
                # BTN/BB position tags too, which corrupted the recorded blinds
                # (0.5/1 read as 20/20 or 50/5). NOTE: the reconstruction engine
                # still does not model antes, so pot/stack reconciliation on ante
                # tables remains approximate.
                continue

            action: str | None = None
            if re.search(r"\bFOLD(?:S|ED)?\b", upper):
                action = "fold"
            elif re.search(r"\bCHECK(?:S|ED)?\b", upper):
                action = "check"
            elif re.search(r"\bCALL(?:S|ED)?\b", upper):
                action = "call"
            elif re.search(r"\b(?:RAISE(?:S|D)?|BET(?:S)?)\b", upper):
                action = "raise"
            elif re.search(r"\bALL[ -]?IN\b", upper):
                action = "all_in"

            has_button = bool(re.search(r"\bBTN\b", upper))
            has_small_blind = bool(re.search(r"\bSB\b", upper))
            has_big_blind = bool(re.search(r"\bBB\b", upper))
            if street == "preflop" and action is None and amount is not None:
                # The small-blind post identifies the button: some tables tag the
                # row "BTN", others only "SB" (heads-up, the SB IS the button).
                # Checking only BTN missed the SB-tagged tables and left blinds at
                # the wrong default (e.g. a 1/2 table recorded as 2/5).
                if has_button or has_small_blind:
                    button = actor
                    small_blind = amount
                    continue
                if has_big_blind:
                    big_blind = amount
                    continue
            if (has_button or has_small_blind) and button is None:
                button = actor
            if action is None:
                continue

            relevant = [
                line.confidence
                for line in row
                if _normalized(line.text) != _normalized(name_line.text)
            ]
            confidence = (
                (name_line.confidence + sum(relevant)) / (len(relevant) + 1)
                if relevant
                else name_line.confidence
            )
            actions.append(
                {
                    "player": actor,
                    "action": action,
                    "amount": amount,
                    "street": street,
                    "confidence": float(confidence),
                    "raw_text": row_text,
                }
            )
            confidences.append(float(confidence))

    # Infer a missing raise-to amount from the following CALL on the same street.
    # CoinPoker sometimes OCRs a raise row without its number; heads-up, the next
    # player's call matches the raise-to, so the call amount IS the raise-to.
    # Without this, validate_history aborts the ENTIRE hand on the amount-less
    # raise (a 3-bet that dropped its number lost the whole history + decisions).
    for index, act in enumerate(actions):
        if act["action"] == "raise" and act["amount"] is None:
            for following in actions[index + 1 :]:
                if following["street"] != act["street"]:
                    break
                if following["action"] == "call" and following["amount"] is not None:
                    act["amount"] = following["amount"]
                    break

    starts_at_hand = small_blind is not None and big_blind is not None
    if not starts_at_hand:
        warnings.append(
            "Dealer Chat does not show both blind rows; the visible timeline may begin mid-hand."
        )
    if not actions and not starts_at_hand:
        # Only a real concern when the timeline did NOT capture the hand's start:
        # then betting actions are expected but missing. When both blind rows ARE
        # visible (starts_at_hand), a fresh hand legitimately has no voluntary
        # actions yet — the opener is first to act, or hero IS the opener making
        # its first decision (e.g. an SB open). Warning there was spurious and
        # made correct hands look broken (flagged as GAP).
        warnings.append("No Dealer Chat betting actions were recognized.")
    return button, [small_blind or 2, big_blind or 5], actions, starts_at_hand, confidences, warnings


def _rank(value: str) -> str | None:
    text = re.sub(r"[^A-Z0-9]", "", value.upper())
    aliases = {"10": "T", "IO": "T", "I0": "T"}
    text = aliases.get(text, text)
    return text if text in set("23456789TJQKA") else None


def _classify_suit(image: Any, line: Any) -> tuple[str | None, float]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    height, width = image.shape[:2]
    glyph_height = max(8.0, line.box.height)
    left = min(width - 1, max(0, int(line.box.right + 3)))
    right = min(width, max(left + 1, int(line.box.left + glyph_height * 2.05)))
    top = min(height - 1, max(0, int(line.box.top + 4)))
    bottom = min(height, max(top + 1, int(line.box.bottom - 4)))
    patch = image[top:bottom, left:right]
    if patch.size == 0:
        return None, 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, _ = np.median(hsv.reshape(-1, 3), axis=0)
    hue = float(hue)
    saturation = float(saturation)
    if saturation < 90.0:
        return SUITS["spades"], min(1.0, 0.65 + (90.0 - saturation) / 180.0)
    if hue <= 18.0 or hue >= 165.0:
        distance = min(hue, 180.0 - hue)
        return SUITS["hearts"], max(0.55, 1.0 - distance / 30.0)
    if 38.0 <= hue <= 84.0:
        return SUITS["clubs"], max(0.55, 1.0 - abs(hue - 62.0) / 45.0)
    if 88.0 <= hue <= 138.0:
        return SUITS["diamonds"], max(0.55, 1.0 - abs(hue - 110.0) / 50.0)
    return None, 0.0


def _drop_card(
    cards: Sequence[str],
    confidences: Sequence[float],
    target: str,
) -> tuple[list[str], list[float]]:
    """Return the card/confidence lists with every occurrence of ``target`` removed."""

    kept_cards: list[str] = []
    kept_conf: list[float] = []
    for card, confidence in zip(cards, confidences):
        if card == target:
            continue
        kept_cards.append(card)
        kept_conf.append(confidence)
    return kept_cards, kept_conf


def _box_iou(first: _CardBox, second: _CardBox) -> float:
    left = max(first.left, second.left)
    top = max(first.top, second.top)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    overlap = max(0, right - left) * max(0, bottom - top)
    union = first.width * first.height + second.width * second.height - overlap
    return overlap / union if union else 0.0


def _region_signature(region: Any) -> int:
    """A cheap, change-sensitive fingerprint of a pixel region.

    Downscales to a tiny thumbnail and hashes it, so an identical region hashes
    identically while any real change (a new card, dealt cards, glow) changes
    enough thumbnail pixels to change the hash. Sub-millisecond — used to skip
    the colour scan when the card area has not moved since the last frame."""

    import cv2  # type: ignore

    if region is None or region.size == 0:
        return 0
    gray = (
        cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        if region.ndim == 3
        else region
    )
    tiny = cv2.resize(gray, (96, 32), interpolation=cv2.INTER_AREA)
    return hash(tiny.tobytes())


def _dense_column_bands(
    mask: Any,
    left: int,
    top: int,
    blob_width: int,
    blob_height: int,
    *,
    merge_gap: int,
    minimum_band: int,
) -> list[tuple[int, int]]:
    """Split a colour blob into the column runs that look like card FACES.

    CoinPoker draws a cyan-blue glow arc BEHIND the hole cards of whoever is to
    act. Its hue is the diamonds hue, so on a blue (diamond) hole card the glow
    fuses with the card into one blob far wider than a card: measured 314 px and
    275 px for a ~74 px card pair, i.e. beyond the ``maximum_width * 2`` split
    cap, so every width test rejected it and the hero pair was dropped outright
    on the frames where the glow pulsed brightest.

    A card column is densely masked over the blob's height (~0.6-0.9 of it); the
    glow band that extends left/right of the cards is only ~25 px tall inside a
    ~110 px blob (<= 0.25). Keeping the columns at >= 45% of the blob's peak
    column (and >= 30% of its height) therefore trims the glow wings while
    leaving the cards intact. Gaps shorter than ``merge_gap`` are bridged so a
    tall white rank glyph — which thins the columns it crosses — cannot cut a
    single card in two; the real card/glow boundary is far wider than that. A
    band still spanning two touching cards is split by the caller's rank-corner
    probe, exactly as before.

    Returns ``[(left, width), ...]`` in mask coordinates, falling back to the
    untrimmed blob when the profile yields nothing usable.
    """

    import numpy as np  # type: ignore

    whole = [(left, blob_width)]
    region = mask[top : top + blob_height, left : left + blob_width]
    if region.size == 0:
        return whole
    columns = np.count_nonzero(region, axis=0)
    peak = float(columns.max())
    if peak <= 0.0:
        return whole
    threshold = max(0.45 * peak, 0.30 * blob_height)
    dense = columns >= threshold
    # Run boundaries from the flank transitions of the padded mask, so the
    # per-column work stays in numpy (this runs for every over-wide contour).
    padded = np.concatenate(([False], dense, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    bands: list[list[int]] = []
    for start, stop in zip(edges[0::2].tolist(), edges[1::2].tolist()):
        if bands and start - (bands[-1][1] + 1) <= merge_gap:
            bands[-1][1] = stop - 1
        else:
            bands.append([start, stop - 1])
    bands = [band for band in bands if band[1] - band[0] + 1 >= minimum_band]
    if bands:
        # Never shave the blob's own edges by a hair: a card's outermost columns
        # are thinned by its rounded corners/anti-aliasing (measured 2-3 px below
        # the density bar), and cutting those moved the rank-corner crop enough to
        # lose an otherwise clean read. A glow wing is an order wider (>= 40 px),
        # so only gaps of at least ``merge_gap`` are treated as real wings.
        if bands[0][0] <= merge_gap:
            bands[0][0] = 0
        if blob_width - 1 - bands[-1][1] <= merge_gap:
            bands[-1][1] = blob_width - 1
    trimmed = [(left + start, end - start + 1) for start, end in bands]
    return trimmed or whole


def _locate_colored_cards(
    image: Any,
    chat_right: float,
    width: int,
    height: int,
    locate_cache: dict[str, Any] | None = None,
) -> list[_CardBox]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    table_width = max(1.0, width - chat_right)
    x_min = chat_right + table_width * 0.20
    x_max = width - table_width * 0.16
    y_min = height * 0.20
    # Cap the card-search band just below the hero seat. Measured card TOPs are
    # rock-consistent: board ~0.32h, hero pair ~0.57h; every phantom (the fixed
    # dealer-button/timer blob ~0.67h and the red Fold/Call/Raise button strip
    # ~0.72h+) sits below 0.63h. Excluding top > 0.63h removes those phantom
    # "card" boxes without touching any real card.
    y_max = height * 0.63
    # Saturation/value floors for the COLOUR suits are set to exclude the
    # hero-turn glow: a cyan-blue arc that sweeps BEHIND the hole cards, whose
    # hue (~105) sits squarely inside the diamonds band. Measured on the glow
    # arc vs the card faces it crosses:
    #   glow        S median  77-119 (p90 ~145), V median 57-67 (p90 ~110)
    #   card faces  S median 161-189,            V median 123-198
    #   shaded card bottoms (under the seat plate) S >= 153, V >= 111
    # Floors of S >= 145 / V >= 100 keep >= 0.64 of even the most shaded card
    # face while admitting <= 0.07 of the glow, so the glow no longer fuses with
    # a blue card into one over-wide blob (see _dense_column_bands).
    color_ranges = {
        SUITS["hearts"]: [
            ((0, 145, 100), (12, 255, 255)),
            ((170, 145, 100), (179, 255, 255)),
        ],
        SUITS["diamonds"]: [((95, 145, 100), (120, 255, 255))],
        SUITS["clubs"]: [((50, 145, 100), (75, 255, 255))],
        # The dark-suit detector must also separate a black card from the DARK
        # TABLE RAIL it sits on. With S < 90 / V > 35 the rail passed at 0.23-0.35
        # density — enough to bridge — so a non-glowing spade hole card merged into
        # a 552x582 rail blob and was never detected at all (the hero-turn glow
        # happened to ring the card and break that bridge, so the bug only showed
        # when it was NOT hero's turn). Measured over 9 spade faces and 7 rail /
        # seat-plate patches: S < 75 / V > 50 passes 0.63-0.86 of a card face but
        # only 0.02-0.20 of the rail, which leaves the rail as sparse speckle that
        # cannot connect. The face's shadowed bottom drops out, so a dark card's
        # box is ~30 px shorter; every consumer derives its geometry from
        # max(height, width * 1.35), so the rank crops are unaffected.
        SUITS["spades"]: [((0, 0, 50), (179, 75, 180))],
    }
    minimum_width = max(30, int(width * 0.016))
    maximum_width = int(width * 0.075)
    minimum_height = max(45, int(height * 0.04))
    maximum_height = int(height * 0.16)
    roi_left = max(0, int(x_min) - maximum_width)
    roi_top = max(0, int(y_min) - maximum_height)
    roi_right = min(width, int(x_max) + maximum_width * 2)
    roi_bottom = min(height, int(y_max) + maximum_height)
    roi = image[roi_top:roi_bottom, roi_left:roi_right]
    # Region cache: if this exact scan area is pixel-identical to the last frame
    # (same geometry + fingerprint), reuse the cards found then and skip the
    # ~24 ms colour scan. Only the raw detection is cached; hero/board selection
    # still runs on the reused list, so it stays correct as the ROI filter or
    # anchors change. Signature is computed only when a cache was supplied.
    signature = 0
    cache_key = (roi_left, roi_top, roi_right, roi_bottom)
    if locate_cache is not None:
        signature = _region_signature(roi)
        entry = locate_cache.get("cards")
        if entry is not None and entry[0] == cache_key and entry[1] == signature:
            return entry[2]
    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV,
    )

    def corner_ink_score(
        candidate_left: int,
        candidate_top: int,
        candidate_width: int,
        candidate_height: int,
    ) -> float:
        corner = image[
            candidate_top : min(
                height,
                candidate_top + max(1, int(candidate_height * 0.42)),
            ),
            candidate_left : min(
                width,
                candidate_left + max(1, int(candidate_width * 0.52)),
            ),
        ]
        if corner.size == 0:
            return 0.0
        corner_hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
        rank_ink = (
            (corner_hsv[:, :, 1] < 125)
            & (corner_hsv[:, :, 2] > 155)
        )
        return float(np.count_nonzero(rank_ink)) / max(1.0, rank_ink.size)

    candidates: list[_CardBox] = []
    for suit, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            component = cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            )
            mask = component if mask is None else cv2.bitwise_or(mask, component)
        if suit == SUITS["spades"]:
            # The tightened dark range leaves the table rail as isolated speckle
            # instead of one bridging mass — correct, but it means ~1500 one-pixel
            # contours per frame, and the per-contour work below then dominates the
            # scan. A 3x3 opening erases the speckle (contours 1500 -> 150) without
            # touching a card face, which is solid over tens of pixels.
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
            )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            blob_left, blob_top, blob_width, card_height = cv2.boundingRect(contour)
            top = blob_top + roi_top
            area = float(cv2.contourArea(contour))
            fill = area / max(1.0, blob_width * card_height)
            # Card-face scale from the blob height (a card is ~1.42x taller than
            # wide). Kept separate from ``expected_face_width`` below because the
            # trimming tolerances must stay at the true face scale.
            face_scale = max(
                minimum_width,
                min(maximum_width, int(round(card_height / 1.42))),
            )
            expected_face_width = face_scale
            if card_height < blob_width * 1.2:
                # A card is ~1.4x taller than wide, so a blob shorter than that
                # has a TRUNCATED height — the dark range drops a black card's
                # shadowed bottom, leaving e.g. 76x62 for a full-width card.
                # height / 1.42 then under-estimates the face width and the split
                # below fires on a single card, cropping it to a 44 px sliver, so
                # trust the observed width. Genuinely fused pairs are still split:
                # they exceed maximum_width, which is a separate clause below.
                expected_face_width = max(face_scale, min(maximum_width, blob_width))
            # Trim the residual hero-turn glow off the blob first: what is left
            # is one card, or the two touching hole cards, at a sane width. Only
            # blobs WIDER than one card face can carry a glow wing, and only ones
            # of card HEIGHT can yield a card at all — profiling just those keeps
            # the scan at its original cost (the dark-suit mask alone yields
            # hundreds of contours per frame, and profiling all of them doubled
            # the ~36 ms colour scan). The dark suit is excluded outright: the glow
            # is bright and saturated so it never enters that mask, while a black
            # face's big white glyph thins the columns it crosses enough for the
            # profile to cut the card itself (a board 9-spade lost half its width).
            if (
                suit != SUITS["spades"]
                and blob_width > face_scale * 1.15
                and minimum_height <= card_height <= maximum_height
            ):
                bands = _dense_column_bands(
                    mask,
                    blob_left,
                    blob_top,
                    blob_width,
                    card_height,
                    merge_gap=max(6, int(face_scale * 0.22)),
                    minimum_band=max(8, int(minimum_width * 0.5)),
                )
            else:
                bands = [(blob_left, blob_width)]
            for band_left, band_width in bands:
                left = band_left + roi_left
                card_width = band_width
                # Normally one card per band. A ~2-card-wide band is split below.
                sub_cards: list[tuple[int, int]] = [(left, card_width)]
                if (
                    card_width <= maximum_width * 2
                    and (
                        card_width > maximum_width
                        or card_width > expected_face_width * 1.55
                    )
                ):
                    # TWO same-suit hole cards touch and their colour masks merge
                    # into one ~2-card-wide blob. Probe the top-left rank corner
                    # of each HALF (splitting at the midpoint, since the blob
                    # height can be truncated by the glow and under-estimate the
                    # card width): a real side-by-side pair carries rank ink in
                    # BOTH halves (~0.10-0.14), a single card fused with the
                    # Hero-seat glow only in one (~0).
                    half = int(round(card_width / 2.0))
                    left_half = left
                    right_half = left + card_width - half
                    left_ink = corner_ink_score(left_half, top, half, card_height)
                    right_ink = corner_ink_score(right_half, top, half, card_height)
                    if left_ink >= 0.04 and right_ink >= 0.04:
                        each = min(maximum_width, half)
                        sub_cards = [
                            (left_half, each),
                            (right_half, each),
                        ]
                    else:
                        # Single card fused with glow: keep the inked end at a
                        # normal card width.
                        if left_ink >= right_ink:
                            sub_cards = [(left, expected_face_width)]
                        else:
                            sub_cards = [
                                (
                                    left + card_width - expected_face_width,
                                    expected_face_width,
                                )
                            ]
                for sub_left, sub_width in sub_cards:
                    if not (
                        x_min <= sub_left <= x_max
                        and y_min <= top <= y_max
                        and minimum_width <= sub_width <= maximum_width
                        and minimum_height <= card_height <= maximum_height
                        and fill >= 0.32
                    ):
                        continue
                    # Require a hint of RANK INK in the top-left corner. The glow
                    # keeps a card-sized core after trimming, and such a phantom
                    # used to enter grouping, inflate the hero pair to a 3-box
                    # cluster and steal the board slot (hero pair then read as the
                    # board). Measured over every saved frame: real cards — spades
                    # included — score 0.079-0.283, glow cores and rank-less
                    # phantoms 0.000-0.011. The 0.02 bar is deliberately far below
                    # the weakest real card so an occluded corner still passes.
                    if corner_ink_score(sub_left, top, sub_width, card_height) < 0.02:
                        continue
                    # The per-suit masks are used only as DETECTORS. The emitted
                    # suit is re-derived from the dominant saturated face hue, so
                    # an occluded colour card is never labelled a spade because a
                    # dark shadow blob overlapped it, and a low-saturation shadow
                    # is admitted as a spade only when a rank is later read.
                    probe = _CardBox(
                        left=sub_left,
                        top=top,
                        right=sub_left + sub_width,
                        bottom=top + card_height,
                        suit=suit,
                        fill=fill,
                    )
                    derived_suit, suit_conf, is_dark, suit_uncertain = (
                        _reclassify_suit(image, probe)
                    )
                    if derived_suit is None:
                        continue
                    candidates.append(
                        _CardBox(
                            left=sub_left,
                            top=top,
                            right=sub_left + sub_width,
                            bottom=top + card_height,
                            suit=derived_suit,
                            fill=fill,
                            is_dark=is_dark,
                            suit_confidence=suit_conf,
                            uncertain=suit_uncertain,
                        )
                    )

    selected: list[_CardBox] = []
    for candidate in sorted(
        candidates,
        # Prefer a colour (rank-bearing) face over an overlapping dark blob so an
        # occluded 9d/4c wins dedup instead of the phantom spade that shadows it.
        key=lambda card: (
            0 if card.is_dark else 1,
            card.width * card.height * card.fill,
        ),
        reverse=True,
    ):
        if any(_box_iou(candidate, existing) > 0.45 for existing in selected):
            continue
        selected.append(candidate)
    result = sorted(selected, key=lambda card: (card.center_y, card.center_x))
    if locate_cache is not None:
        locate_cache["cards"] = (cache_key, signature, result)
    return result


def _box_suit(image: Any, card: _CardBox) -> str | None:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    patch = image[
        card.top + 5 : max(card.top + 6, card.bottom - 8),
        card.left + int(card.width * 0.58) : max(
            card.left + int(card.width * 0.58) + 1,
            card.right - 4,
        ),
    ]
    if patch.size == 0:
        return None
    hue, saturation, _ = np.median(
        cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3),
        axis=0,
    )
    if saturation < 90:
        return SUITS["spades"]
    if hue <= 18 or hue >= 165:
        return SUITS["hearts"]
    if 38 <= hue <= 84:
        return SUITS["clubs"]
    if 88 <= hue <= 138:
        return SUITS["diamonds"]
    return None


def _reclassify_suit(image: Any, box: _CardBox) -> tuple[str | None, float, bool, bool]:
    """Re-derive suit from the dominant SATURATED face hue.

    The suit is decoupled from whichever detector mask happened to find the
    blob. Spades are admitted only when the face is genuinely dark/desaturated
    (``is_dark``); such a box must still yield a readable rank before it is
    accepted as a card, so shadows/gaps/UI do not become phantom spades.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    inset_x = max(1, int(box.width * 0.16))
    inset_y = max(1, int(box.height * 0.16))
    top = max(0, box.top + inset_y)
    bottom = min(image.shape[0], box.bottom - inset_y)
    left = max(0, box.left + inset_x)
    right = min(image.shape[1], box.right - inset_x)
    if bottom <= top or right <= left:
        return None, 0.0, False, False
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].reshape(-1)
    hue = hsv[:, :, 0].reshape(-1)
    value = hsv[:, :, 2].reshape(-1)
    total = max(1, saturation.size)
    saturated = saturation >= 90
    # PRIMARY spade gate: MEDIAN saturation. A heart/diamond/club face is a
    # strong, uniform colour (median saturation ~190). A spade face is dark and
    # neutral; CoinPoker's blue hero glow adds a FRINGE of saturated pixels that
    # inflated the old saturated-FRACTION test into voting a colour (a glow-lit
    # Q-spade read as a diamond, hand #7), but the face's MEDIAN saturation stays
    # low. Measured: spades (incl. glow-lit) sat_median ~50-90, colour faces
    # ~190; split at 130, clear of both so neither regresses.
    sat_median = float(np.median(saturation))
    val_median = float(np.median(value))
    if sat_median < 130:
        if val_median < 120:
            # Dark, neutral face -> confident spade (survives the hero glow).
            return SUITS["spades"], 0.85, True, False
        # Weakly saturated but brighter -> possibly a washed colour card; admit
        # as spade but UNCERTAIN (capped) so the frame declines if it is wrong.
        return SUITS["spades"], 0.45, True, True
    saturated_fraction = float(np.count_nonzero(saturated)) / total
    if saturated_fraction < 0.15:
        # Below the strong-colour bar. Distinguish a GENUINE spade — even one
        # brightened by CoinPoker's blue hero-turn glow ring — from a muted
        # colour card. Brightness/median-saturation do NOT separate them (a
        # glowing spade and a muted diamond both sit ~med_val 66 / med_sat 47).
        # But a neutral spade face has almost no MODERATELY saturated pixels,
        # while a muted diamond/club keeps substantial mid-saturation colour.
        # Measured cleanly: genuine spades (incl. glowing) mid_sat_fraction
        # ~0.00-0.26; muted colour faces ~0.73-0.97 -> split at 0.45.
        mid_sat_fraction = float(np.count_nonzero(saturation >= 40)) / total
        if mid_sat_fraction < 0.45:
            # Neutral face -> confident spade (verifies even under the glow).
            return SUITS["spades"], 0.85, True, False
        # Substantial residual colour -> a muted colour face bucketed as spade.
        # Admit it as a spade but flag UNCERTAIN (low suit confidence) so the
        # frame declines rather than acting on a possibly-wrong suit.
        return SUITS["spades"], 0.40, True, True
    saturated_hue = hue[saturated]
    hearts = int(np.count_nonzero((saturated_hue <= 18) | (saturated_hue >= 165)))
    clubs = int(np.count_nonzero((saturated_hue >= 38) & (saturated_hue <= 84)))
    diamonds = int(np.count_nonzero((saturated_hue >= 88) & (saturated_hue <= 138)))
    best_name, best_count = max(
        (("hearts", hearts), ("clubs", clubs), ("diamonds", diamonds)),
        key=lambda item: item[1],
    )
    if best_count == 0:
        return None, 0.0, False, False
    confidence = best_count / max(1, hearts + clubs + diamonds)
    return SUITS[best_name], max(0.55, min(1.0, confidence)), False, False


def _white_ink_mask(patch: Any) -> Any:
    """Binary mask of the white glyph ink on a card face."""

    import cv2  # type: ignore

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] < 125) & (hsv[:, :, 2] > 155)).astype("uint8") * 255


def _glyph_line_crop(crop: Any) -> Any | None:
    """Tighten a card sub-region to the text LINE of white rank ink.

    RapidOCR reads a rank far more reliably from a tight glyph strip than from the
    whole corner, which also contains the suit pip and a slab of card face.
    Measured over every saved frame: the three Q-corners where a variant read "A"
    (a Q's tail reads as the crossbar) become unanimous Q, and weak J/Q reads rise
    from ~0.4-0.8 to 1.0 — no card reads a *different* rank than before.

    Components are clustered on the vertical band of the TALLEST one, not simply
    the largest blob: the corner carries the suit pip BELOW the rank (a
    largest-blob crop can land on the pip and lose the rank entirely) and a "10"
    is two glyphs side by side (which a single-blob crop would halve).
    """

    import cv2  # type: ignore

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        _white_ink_mask(crop), connectivity=8
    )
    boxes = [
        (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_TOP]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
        )
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 20
    ]
    if not boxes:
        return None
    seed = max(boxes, key=lambda box: box[3])
    seed_top, seed_bottom = seed[1], seed[1] + seed[3]
    kept = [
        box
        for box in boxes
        if min(seed_bottom, box[1] + box[3]) - max(seed_top, box[1])
        >= 0.5 * min(seed[3], box[3])
    ]
    left = min(box[0] for box in kept)
    top = min(box[1] for box in kept)
    right = max(box[0] + box[2] for box in kept)
    bottom = max(box[1] + box[3] for box in kept)
    pad_x = max(2, int((right - left) * 0.22))
    pad_y = max(2, int((bottom - top) * 0.18))
    return crop[
        max(0, top - pad_y) : min(crop.shape[0], bottom + pad_y),
        max(0, left - pad_x) : min(crop.shape[1], right + pad_x),
    ]


def _variant_rank_readings(crop: Any) -> list[tuple[str, float]]:
    """Read a rank from one crop with the colour/gray/Otsu transform set.

    A rank glyph is always WHITE ink on the card face, so a fourth variant that
    keeps only white pixels (inverted to dark-on-light) reads it more reliably
    than Otsu, which merges the glyph into a mid-bright face. It is only run when
    the first three disagree or come up empty — precisely the cases that fall
    through to the low-trust branches of :func:`_cropped_rank` — so clean cards
    (three identical reads) never pay for the extra pass.
    """

    import cv2  # type: ignore

    from ..recognition import recognize_text_strip

    enlarged = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    readings: list[tuple[str, float]] = []
    for variant in (enlarged, gray, thresholded):
        text, score = recognize_text_strip(variant)
        rank = _rank(text)
        if rank is not None:
            readings.append((rank, float(score)))
    if len(readings) == 3 and len({rank for rank, _ in readings}) == 1:
        return readings
    text, score = recognize_text_strip(cv2.bitwise_not(_white_ink_mask(enlarged)))
    rank = _rank(text)
    if rank is not None:
        readings.append((rank, float(score)))
    return readings


def _region_rank_readings(image: Any, top: int, bottom: int, left: int, right: int) -> list[tuple[str, float]]:
    """Read a card sub-region, preferring a crop tightened to the glyph itself."""

    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return []
    tight = _glyph_line_crop(crop)
    if tight is not None and tight.size:
        readings = _variant_rank_readings(tight)
        if readings:
            return readings
        # An occluded glyph can defeat the tight crop while the wider region still
        # yields something (a half-covered hero centre glyph) — fall back to it
        # rather than losing the reading.
    return _variant_rank_readings(crop)


def _cropped_rank(
    image: Any,
    card: _CardBox,
    corroborating_rank: str | None = None,
) -> tuple[str | None, float]:
    height_img, width_img = image.shape[:2]
    expected_height = max(card.height, int(card.width * 1.35))

    # Voter A: the top-left corner rank (as before).
    corner_readings = _region_rank_readings(
        image,
        card.top,
        min(height_img, card.top + int(expected_height * 0.40)),
        card.left,
        min(width_img, card.left + int(card.width * 0.48)),
    )
    # Fast path: a very strong corner colour-variant read is trusted immediately
    # only when broad-OCR corroborates the SAME rank. A lone strong corner glyph
    # (corroborating_rank is None) must still face the centre cross-check below,
    # so a confidently-misread corner (e.g. Q->A in frame_004) cannot bypass it.
    if (
        corner_readings
        and corner_readings[0][1] >= 0.985
        and corroborating_rank == corner_readings[0][0]
    ):
        return corner_readings[0][0], min(1.0, corner_readings[0][1] + 0.01)

    # Second fast path: EVERY corner transform agreed, two of them at >= 0.99, and
    # broad OCR does not dissent. Since the corner crop is tightened to the glyph
    # this is now the normal outcome for a clean card, and reading the centre as
    # well costs more OCR passes than the whole rest of the scan (~250 ms/card) —
    # for a card the corner is already unanimous about. Anything less than
    # unanimous, an inferred box, or a dissenting broad read still goes through the
    # centre cross-check below, which is where the Q->A class of misread is caught.
    # Capped at 0.98 so a two-region agreement still ranks strictly higher.
    corner_ranks = {rank for rank, _ in corner_readings}
    if (
        len(corner_readings) >= 3
        and len(corner_ranks) == 1
        and not getattr(card, "inferred", False)
        and corroborating_rank in (None, corner_readings[0][0])
        and sorted((score for _, score in corner_readings), reverse=True)[1] >= 0.99
    ):
        best_two = sorted((score for _, score in corner_readings), reverse=True)[:2]
        return corner_readings[0][0], min(0.98, sum(best_two) / 2.0 + 0.02)

    # Voter B: the large, clean rank glyph in the card centre. It cross-checks a
    # confident-but-wrong corner (Q->A, 9->6) that the corner alone cannot fix.
    centre_readings = _region_rank_readings(
        image,
        min(height_img, card.top + int(expected_height * 0.34)),
        min(height_img, card.top + int(expected_height * 0.95)),
        max(0, card.left + int(card.width * 0.12)),
        min(width_img, card.left + int(card.width * 0.88)),
    )

    combined = list(corner_readings)
    combined.extend(centre_readings)
    if corroborating_rank is not None:
        combined.append((corroborating_rank, 0.90))
    if not combined:
        return None, 0.0

    votes = Counter(rank for rank, _ in combined)
    winner, vote_count = max(
        votes.items(),
        key=lambda item: (
            item[1],
            max(score for rank, score in combined if rank == item[0]),
        ),
    )
    winner_in_corner = any(rank == winner for rank, _ in corner_readings)
    winner_in_centre = any(rank == winner for rank, _ in centre_readings)
    is_inferred = getattr(card, "inferred", False)
    uncorroborated = corroborating_rank != winner and not (
        winner_in_corner and winner_in_centre
    )
    # An INFERRED (guessed) partner box is placed where a second hero card *should*
    # be; its rank has no independent detection behind it. If that rank is not
    # confirmed by the centre glyph (or by broad OCR), the corner alone must not
    # drive a live-money decision -- frame_004 misreads Q->A on all three corner
    # variants with an EMPTY centre, so it slips past a competing-rank check.
    # Also gate a normal box when a distinct competing rank was read and the
    # winner was seen by only one region with no broad-OCR support.
    dissent = sum(count for rank, count in votes.items() if rank != winner)
    # ... unless that region was DECISIVE about it. A single stray variant (Otsu
    # merging Q's tail into an A) used to drag a 3:1 majority down to 0.45, which
    # lands at 0.77 overall — one hundredth above the watcher's 0.76 verify floor,
    # so genuinely clean hands sat on a knife edge. A >=3:1 majority is real
    # evidence: trust it, but cap it below a two-region agreement so it still
    # ranks under a fully corroborated read. An INFERRED box keeps the hard gate:
    # its rank has no detection behind it, and frame_004's Q->A misread was
    # unanimous across variants, so a majority cannot vouch for it.
    decisive = vote_count >= 3 and vote_count >= dissent * 3

    def winner_quality() -> float:
        """Mean of the winner's two BEST agreeing reads.

        Averaging over *every* agreeing read let one weak-but-correct variant
        (a marginal ink-mask pass on an already unanimous glyph) pull the score
        below what two strong reads had earned on their own — extra corroboration
        must never lower confidence. Two reads is the quality sample; the count
        beyond that is credited through ``consensus_bonus``.
        """

        scores = sorted(
            (score for rank, score in combined if rank == winner),
            reverse=True,
        )[:2]
        return sum(scores) / len(scores)

    if uncorroborated and (is_inferred or len(votes) > 1):
        if is_inferred or not decisive:
            return winner, 0.45
        return winner, min(0.88, winner_quality())
    if vote_count >= 2:
        consensus_bonus = 0.04 if vote_count >= 3 else 0.02
        return winner, min(1.0, winner_quality() + consensus_bonus)

    # No agreement across voters. Accept a lone reading only when exactly one
    # side saw a glyph and it is very strong; conflicting single reads are left
    # unrecognized so a poker decision is never based on an unstable card.
    corner_best = max(corner_readings, key=lambda item: item[1], default=None)
    centre_best = max(centre_readings, key=lambda item: item[1], default=None)
    if corner_best and not centre_readings and corner_best[1] >= 0.97:
        return corner_best[0], 0.78
    if centre_best and not corner_readings and centre_best[1] >= 0.97:
        return centre_best[0], 0.78
    # Dark-SPADE fallback (board or hero). A black card face OCRs its rank weakly
    # and usually reads no centre glyph, so the strict 0.97 lone-corner bar drops
    # it — e.g. a board J-spade between two colour cards, leaving the board a card
    # short and blocking the decision. When the card is a colour-confirmed dark
    # spade and EVERY corner reading agrees on ONE rank (garbage is already
    # filtered out by _rank, so a single entry is a clean rank), accept it at a
    # reduced threshold. The caller boosts a genuine dark spade's confidence and
    # 2-frame corroboration still guards a lone misread, so nothing acts on one
    # shaky frame.
    if (
        getattr(card, "is_dark", False)
        and not getattr(card, "uncertain", False)
        and corner_best is not None
        and not centre_readings
        and len({rank for rank, _ in corner_readings}) == 1
        and corner_best[1] >= 0.30
    ):
        return corner_best[0], corner_best[1]
    return None, 0.0


def _card_rank_cache_key(image: Any, card: _CardBox) -> tuple[str, bytes] | None:
    """Return a compact signature for an unchanged CoinPoker card corner."""

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    expected_height = max(card.height, int(card.width * 1.35))
    corner = image[
        card.top : min(image.shape[0], card.top + int(expected_height * 0.40)),
        card.left : min(image.shape[1], card.left + int(card.width * 0.48)),
    ]
    if corner.size == 0:
        return None
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    white = (
        (hsv[:, :, 1] < 125)
        & (hsv[:, :, 2] > 155)
    ).astype(np.uint8) * 255
    normalized = cv2.resize(white, (24, 24), interpolation=cv2.INTER_AREA)
    packed = np.packbits(normalized >= 96).tobytes()
    return card.suit, packed


def _recognize_cards(
    image: Any,
    lines: Sequence[Any],
    chat_right: float,
    hero_anchor: Any | None,
    width: int,
    height: int,
    rank_cache: dict[tuple[str, bytes], tuple[str, float]] | None = None,
    hero_card_boxes: Sequence[tuple[float, float, float, float]] | None = None,
    locate_cache: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], list[float], list[str], list]:
    warnings: list[str] = []
    localized = _locate_colored_cards(
        image, chat_right, width, height, locate_cache=locate_cache
    )
    localized_groups: list[list[_CardBox]] = []
    for card in localized:
        target = next(
            (
                group
                for group in localized_groups
                if abs(median(item.center_y for item in group) - card.center_y) <= 38.0
            ),
            None,
        )
        if target is None:
            localized_groups.append([card])
        else:
            target.append(card)
    for group in localized_groups:
        group.sort(key=lambda item: item.center_x)

    # The hero seat plate is always in the BOTTOM part of the table, so an anchor
    # resolved above mid-frame cannot be it: the hero name was mis-identified
    # (a partial Dealer Chat promoted the opponent's top seat to players[0]).
    # Trusting it puts both card-band limits ABOVE the cards and drops every card
    # in the frame — hero and board alike — so fall back to the geometric
    # defaults, which is what a missing anchor already does.
    anchor_top = (
        hero_anchor.box.top
        if hero_anchor is not None and hero_anchor.box.top > height * 0.55
        else None
    )
    board_limit = anchor_top - 70.0 if anchor_top is not None else height * 0.68
    localized_board = min(
        (
            group
            for group in localized_groups
            if 3 <= len(group) <= 5
            and median(card.center_y for card in group) < board_limit
        ),
        key=lambda group: median(card.center_y for card in group),
        default=[],
    )
    localized_board_y = (
        median(card.center_y for card in localized_board)
        if localized_board
        else height * 0.35
    )
    hero_limit = anchor_top if anchor_top is not None else height * 0.90
    if hero_card_boxes is not None:
        # The caller (live CoinPoker fast path) knows the table-window geometry
        # and supplies ONE small box per hole card (left card, right card).
        # Each card is taken as the largest detected contour whose CENTRE falls
        # in its own box, so the dealer-button "D", side chips, the board and the
        # action badge/glow can never be mistaken for a hole card, and the two
        # cards keep a stable left->right order. A box with no card contributes
        # nothing (that slot stays unresolved); the full-frame fallback below
        # still applies if neither box yields a card.
        picked: list[_CardBox] = []
        seen: set[tuple[int, int]] = set()
        for box_left, box_top, box_right, box_bottom in hero_card_boxes:
            # Each small rank box sits INSIDE its card, so match the card whose
            # detected contour COVERS the box centre — not by the contour's own
            # corner, which the hero glow can stretch well outside a tight rank
            # box. Among cards covering the point, take the one whose centre is
            # nearest (handles the fanned overlap where the top card wins).
            box_cx = (box_left + box_right) / 2.0
            box_cy = (box_top + box_bottom) / 2.0
            covering = [
                card
                for card in localized
                if card.left <= box_cx <= card.right
                and card.top <= box_cy <= card.bottom
            ]
            if not covering:
                continue
            best = min(
                covering,
                key=lambda card: abs(card.center_x - box_cx)
                + abs(card.center_y - box_cy),
            )
            key = (best.left, best.top)
            if key not in seen:
                seen.add(key)
                picked.append(best)
        localized_hero = sorted(picked, key=lambda card: card.center_x)
    else:
        # The hero pair is the topmost 2-card cluster below the board. Accept a
        # 2-3 box cluster: a small phantom (the red dealer-button "D", or a chip)
        # is often colour-detected between the two hole cards, giving a 3-box
        # cluster that the old exact-len==2 test rejected -> hero fell back to
        # full-frame OCR (capped 0.75, below the verify floor) even though both
        # cards were found.
        hero_groups = [
            group
            for group in localized_groups
            if 2 <= len(group) <= 3
            and median(card.center_y for card in group) > localized_board_y + 70.0
            and median(card.center_y for card in group) < hero_limit
        ]
        localized_hero = min(
            hero_groups,
            key=lambda group: median(card.center_y for card in group),
            default=[],
        )
    if len(localized_hero) > 2:
        # Keep the two most card-like (largest-area) boxes; drop the phantom.
        localized_hero = sorted(
            sorted(
                localized_hero,
                key=lambda card: card.width * card.height,
                reverse=True,
            )[:2],
            key=lambda card: card.center_x,
        )
    if not localized_hero:
        single_groups = [
            group
            for group in localized_groups
            if len(group) == 1
            and median(card.center_y for card in group) > localized_board_y + 70.0
            and median(card.center_y for card in group) < hero_limit
        ]
        if single_groups:
            table_center = (chat_right + width) / 2.0
            known = min(
                (group[0] for group in single_groups),
                key=lambda card: (
                    abs(card.center_x - table_center),
                    card.center_y,
                ),
            )
            card_width = known.width
            card_height = max(known.height, int(card_width * 1.35))
            offset = int(card_width * 0.86)
            missing_left = (
                known.left - offset
                if known.center_x >= table_center
                else known.left + offset
            )
            inferred = _CardBox(
                left=missing_left,
                top=known.top,
                right=missing_left + card_width,
                bottom=min(height, known.top + card_height),
                suit=SUITS["spades"],
                fill=0.65,
            )
            inferred_suit, inf_conf, inf_dark, inf_uncertain = _reclassify_suit(
                image, inferred
            )
            if inferred_suit is not None:
                inferred = _CardBox(
                    left=inferred.left,
                    top=inferred.top,
                    right=inferred.right,
                    bottom=inferred.bottom,
                    suit=inferred_suit,
                    fill=inferred.fill,
                    is_dark=inf_dark,
                    suit_confidence=inf_conf,
                    uncertain=inf_uncertain,
                    inferred=True,
                )
                # convert_localized rank-gates every card, so the inferred
                # partner survives only if a readable rank is found on it; a
                # card is never invented from seat glow.
                localized_hero = sorted(
                    [known, inferred],
                    key=lambda card: card.center_x,
                )

    def convert_localized(
        group: Sequence[_CardBox],
    ) -> tuple[list[str], list[float], list[_CardBox]]:
        cards: list[str] = []
        confidences: list[float] = []
        boxes: list[_CardBox] = []
        for card in group:
            overlapping_ranks = [
                (_rank(line.text), float(line.confidence))
                for line in lines
                if card.left - 3 <= line.box.center_x <= card.left + card.width * 0.68
                and card.top - 3 <= line.box.center_y <= card.top + max(
                    card.height, card.width * 1.35
                ) * 0.48
                and _rank(line.text) is not None
            ]
            broad_rank: str | None = None
            broad_confidence = 0.0
            if overlapping_ranks:
                broad_rank, broad_confidence = max(
                    overlapping_ranks,
                    key=lambda item: item[1],
                )
            cache_key = (
                _card_rank_cache_key(image, card)
                if rank_cache is not None
                else None
            )
            cached_rank = (
                rank_cache.get(cache_key)
                if rank_cache is not None and cache_key is not None
                else None
            )
            if cached_rank is not None:
                local_rank, local_confidence = cached_rank
            else:
                local_rank, local_confidence = _cropped_rank(
                    image,
                    card,
                    broad_rank,
                )
                # Cache the per-card OCR result keyed on the corner's PIXEL
                # signature, INCLUDING an unreadable (None) result. A stable card
                # region — a settled board card, or the opponent's face-down card
                # — is otherwise re-run through the 6-variant OCR every frame for
                # nothing. The signature changes when the pixels change, so a card
                # that later becomes readable (or a new card) is re-read.
                if rank_cache is not None and cache_key is not None:
                    if len(rank_cache) >= 128:
                        rank_cache.clear()
                    rank_cache[cache_key] = (local_rank, local_confidence)
            if local_rank is not None:
                rank = local_rank
                rank_confidence = local_confidence
                if broad_rank == local_rank:
                    rank_confidence = min(
                        1.0,
                        rank_confidence * 0.75 + broad_confidence * 0.25 + 0.03,
                    )
                elif broad_rank is not None:
                    warnings.append(
                        "A broad OCR card rank conflicted with the verified card-corner rank."
                    )
            elif broad_rank is not None and broad_confidence >= 0.97:
                rank = broad_rank
                rank_confidence = min(0.78, broad_confidence)
            else:
                rank = None
                rank_confidence = 0.0
            if rank is None:
                continue
            # Never emit a card whose suit is not one of the four real glyphs.
            # A malformed suit (e.g. a non-breaking space) otherwise leaks a
            # "6\xa0"-style card into the hero pair, which can never verify and
            # blocks every decision; dropping it leaves the slot unresolved so a
            # cleaner frame (or the cached pair) fills it — correct-or-silent.
            if card.suit not in _VALID_SUITS:
                continue
            cards.append(f"{rank}{card.suit}")
            confidence = min(1.0, 0.55 + rank_confidence * 0.35 + card.fill * 0.1)
            # An uncertain suit (muted colour face bucketed as spade by the
            # low-saturation fallback) is capped below the cards_verified floor
            # so the frame DECLINES rather than acting on a wrong suit.
            if getattr(card, "uncertain", False):
                confidence = min(confidence, 0.75)
            elif getattr(card, "is_dark", False):
                # A genuine dark SPADE (colour-confirmed black face, NOT the
                # uncertain fallback above) reads a correct rank, but its dark
                # corner OCRs low, dragging this score to ~0.77 and blocking
                # verification/decisions on every spade hand. The card is
                # confidently identified — rank read + suit from the black face —
                # so lift it to a decision-clearing floor. Two-frame corroboration
                # still guards a lone misread (this stays below the 0.90
                # single-frame trust bar), so nothing acts on one bad frame.
                confidence = max(confidence, 0.88)
            confidences.append(confidence)
            boxes.append(card)
        return cards, confidences, boxes

    localized_hero_cards, localized_hero_confidence, localized_hero_boxes = (
        convert_localized(localized_hero)
    )
    # Recover a partner hole card that never formed a colour contour. A dark
    # SPADE face, bluish under the Hero-seat glow, merges into the low-saturation
    # felt as one giant blob and is never a distinct card — so only its partner
    # (a colour card) reads. When exactly one hero card resolved, infer the
    # partner's box adjacent to it (offset by a card width toward the empty
    # side) and try to read it there; convert_localized rank-gates it, so a card
    # is only added if a real rank is found — never invented from glow.
    if len(localized_hero_cards) == 1 and localized_hero_boxes:
        known_box = localized_hero_boxes[0]
        table_center = (chat_right + width) / 2.0
        partner_width = known_box.width
        partner_height = max(known_box.height, int(partner_width * 1.35))
        offset = int(partner_width * 0.90)
        partner_left = (
            known_box.left - offset
            if known_box.center_x >= table_center
            else known_box.left + offset
        )
        partner_probe = _CardBox(
            left=partner_left,
            top=known_box.top,
            right=partner_left + partner_width,
            bottom=min(height, known_box.top + partner_height),
            suit=SUITS["spades"],
            fill=0.65,
        )
        partner_suit, ps_conf, ps_dark, ps_unc = _reclassify_suit(image, partner_probe)
        if partner_suit is not None:
            partner_probe = _CardBox(
                left=partner_probe.left,
                top=partner_probe.top,
                right=partner_probe.right,
                bottom=partner_probe.bottom,
                suit=partner_suit,
                fill=partner_probe.fill,
                is_dark=ps_dark,
                suit_confidence=ps_conf,
                uncertain=ps_unc,
                inferred=True,
            )
            partner_cards, partner_conf, partner_boxes = convert_localized(
                [partner_probe]
            )
            if partner_cards:
                combined = sorted(
                    [
                        (
                            known_box.center_x,
                            localized_hero_cards[0],
                            localized_hero_confidence[0],
                            known_box,
                        ),
                        (
                            partner_boxes[0].center_x,
                            partner_cards[0],
                            partner_conf[0],
                            partner_boxes[0],
                        ),
                    ],
                    key=lambda item: item[0],
                )
                localized_hero_cards = [item[1] for item in combined]
                localized_hero_confidence = [item[2] for item in combined]
                localized_hero_boxes = [item[3] for item in combined]
    localized_board_cards, localized_board_confidence, localized_board_boxes = (
        convert_localized(localized_board)
    )

    # Full-frame OCR is retained ONLY as a slot-filler for a hero or board slot
    # the localized four-colour pass could not resolve (non-standard themes, or a
    # merged colour blob the localizer drops). It never discards a localized card.
    def _full_frame_cards() -> tuple[list[str], list[float], list[str], list[float]]:
        table_width = max(1.0, width - chat_right)
        candidates = [
            line
            for line in lines
            if chat_right + table_width * 0.14 < line.box.center_x < width - table_width * 0.18
            and height * 0.20 < line.box.center_y < height * 0.82
            and _rank(line.text) is not None
        ]
        groups: list[list[Any]] = []
        for line in sorted(candidates, key=lambda item: item.box.center_y):
            tolerance = max(15.0, line.box.height * 0.75)
            target = next(
                (
                    group
                    for group in groups
                    if abs(median(item.box.center_y for item in group) - line.box.center_y) <= tolerance
                ),
                None,
            )
            if target is None:
                groups.append([line])
            else:
                target.append(line)
        for group in groups:
            group.sort(key=lambda item: item.box.center_x)

        board_group = min(
            (group for group in groups if 3 <= len(group) <= 5),
            key=lambda group: median(line.box.center_y for line in group),
            default=[],
        )
        board_y = median(line.box.center_y for line in board_group) if board_group else height * 0.35
        hero_limit_ff = anchor_top if anchor_top is not None else height * 0.90
        hero_group = min(
            (
                group
                for group in groups
                if len(group) == 2
                and median(line.box.center_y for line in group) > board_y + 55.0
                and median(line.box.center_y for line in group) < hero_limit_ff
            ),
            key=lambda group: median(line.box.center_y for line in group),
            default=[],
        )

        def convert(group: Sequence[Any]) -> tuple[list[str], list[float], list[Any]]:
            cards: list[str] = []
            confidences: list[float] = []
            boxes: list[Any] = []
            for line in group:
                rank = _rank(line.text)
                suit, suit_confidence = _classify_suit(image, line)
                if rank is None or suit not in _VALID_SUITS:
                    continue
                cards.append(f"{rank}{suit}")
                confidences.append(float(line.confidence) * suit_confidence)
                boxes.append(line.box)
            return cards, confidences, boxes

        ff_board, ff_board_conf, ff_board_boxes = convert(board_group)
        ff_hero, ff_hero_conf, ff_hero_boxes = convert(hero_group)
        return (
            ff_hero,
            ff_hero_conf,
            ff_hero_boxes,
            ff_board,
            ff_board_conf,
            ff_board_boxes,
        )

    hero_cards = list(localized_hero_cards)
    hero_confidences = list(localized_hero_confidence)
    hero_boxes: list[Any] = list(localized_hero_boxes)
    board_cards = list(localized_board_cards)
    board_confidences = list(localized_board_confidence)
    board_boxes: list[Any] = list(localized_board_boxes)

    if not hero_cards or not board_cards:
        (
            ff_hero,
            ff_hero_conf,
            ff_hero_boxes,
            ff_board,
            ff_board_conf,
            ff_board_boxes,
        ) = _full_frame_cards()
        if not hero_cards and len(ff_hero) == 2:
            hero_cards = list(ff_hero)
            hero_boxes = list(ff_hero_boxes)
            # Full-frame OCR is a last-resort slot-filler used only when the
            # localized four-colour pass could not resolve the hero pair at all.
            # These cards bypass the corner+centre rank cross-validation, so a
            # broad-OCR misread (e.g. Q->A) can score high on raw OCR confidence
            # alone. Crucially the raw confidence does NOT separate right from
            # wrong here (frame_001's correct Q scores 0.77 while frame_004's
            # wrong A scores 0.86), so an uncorroborated full-frame hero read is
            # capped below the 0.82 cards_verified floor: the card is still
            # EMITTED (recall preserved) but the frame DECLINES rather than
            # acting on a possibly-wrong, cross-unvalidated hand.
            hero_confidences = [min(conf, 0.75) for conf in ff_hero_conf]
        if not board_cards and ff_board:
            board_cards, board_confidences = list(ff_board), list(ff_board_conf)
            board_boxes = list(ff_board_boxes)

    # Snapshot the inspected card regions BEFORE dedup, so the optional live
    # overlay shows every area the detector actually read (dedup only trims the
    # rare same-card-in-two-places case and would leave a box unpaired).
    card_boxes: list[tuple[str, float, float, float, float, float]] = []
    for role, cards_group, confs_group, boxes_group in (
        ("hero", hero_cards, hero_confidences, hero_boxes),
        ("board", board_cards, board_confidences, board_boxes),
    ):
        for card, confidence, box in zip(cards_group, confs_group, boxes_group):
            card_boxes.append(
                (
                    f"{role} {card}",
                    float(box.left),
                    float(box.top),
                    float(box.right),
                    float(box.bottom),
                    float(confidence),
                )
            )

    # Resolve a card that appears in both hero and board by dropping the
    # lower-confidence occurrence rather than discarding the whole read.
    for duplicate in set(hero_cards) & set(board_cards):
        hero_best = max(
            (conf for card, conf in zip(hero_cards, hero_confidences) if card == duplicate),
            default=0.0,
        )
        board_best = max(
            (conf for card, conf in zip(board_cards, board_confidences) if card == duplicate),
            default=0.0,
        )
        if hero_best >= board_best:
            board_cards, board_confidences = _drop_card(board_cards, board_confidences, duplicate)
        else:
            hero_cards, hero_confidences = _drop_card(hero_cards, hero_confidences, duplicate)

    if len(hero_cards) != 2:
        warnings.append("CoinPoker hero cards were not recognized as one complete pair.")
    if localized_board and len(board_cards) != len(localized_board):
        warnings.append("At least one CoinPoker board-card suit color was ambiguous.")
    return (
        hero_cards,
        board_cards,
        hero_confidences + board_confidences,
        warnings,
        card_boxes,
    )


def _visible_stacks(
    lines: Sequence[Any],
    players: Sequence[str],
    anchors: dict[str, Any],
    chat_right: float,
) -> list[int | None]:
    stacks: list[int | None] = []
    for name in players:
        anchor = anchors.get(name)
        if anchor is None:
            stacks.append(None)
            continue
        amounts = [
            (abs(line.box.center_x - anchor.box.center_x), line.box.top, _decimal_units(line.text))
            for line in lines
            if line.box.left >= chat_right
            # Long player names can occupy a taller OCR box and overlap the
            # first pixels of the stack line below. Allow that small overlap
            # while still excluding the action/timer text above the plaque.
            and anchor.box.bottom - 8.0 <= line.box.top <= anchor.box.bottom + 75.0
            and abs(line.box.center_x - anchor.box.center_x) <= 90.0
            and _decimal_units(line.text) is not None
        ]
        stacks.append(min(amounts, default=(0.0, 0.0, None))[2])
    return stacks


def _visible_pot(lines: Sequence[Any], chat_right: float) -> int | None:
    candidates = [
        (line.confidence, _decimal_units(line.text))
        for line in lines
        if line.box.left >= chat_right
        and re.search(r"\bPOT\b", line.text, re.IGNORECASE)
        and "SPLASH" not in line.text.upper()
        and _decimal_units(line.text) is not None
    ]
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _contributions(
    button: int | None,
    blinds: Sequence[int],
    actions: Sequence[dict[str, Any]],
) -> list[int] | None:
    if button is None:
        return None
    contributions = [0, 0]
    round_bets = [0, 0]
    contributions[button] = round_bets[button] = int(blinds[0])
    contributions[1 - button] = round_bets[1 - button] = int(blinds[1])
    street = "preflop"
    for action in actions:
        if action["street"] != street:
            street = action["street"]
            round_bets = [0, 0]
        player = int(action["player"])
        kind = action["action"]
        if kind == "call":
            paid = max(round_bets) - round_bets[player]
            round_bets[player] += paid
            contributions[player] += paid
        elif kind == "raise":
            amount = action.get("amount")
            if amount is None or amount < round_bets[player]:
                return None
            paid = int(amount) - round_bets[player]
            round_bets[player] = int(amount)
            contributions[player] += paid
        elif kind == "all_in":
            return None
    return contributions


def extract_coinpoker_layout(
    image: Any,
    lines: Sequence[Any],
    width: int,
    height: int,
    starting_stacks_override: Sequence[int] | None = None,
    rank_cache: dict[tuple[str, bytes], tuple[str, float]] | None = None,
    hero_card_boxes: Sequence[tuple[float, float, float, float]] | None = None,
    locate_cache: dict[str, Any] | None = None,
) -> CoinPokerLayoutResult:
    headers = _street_headers(lines)
    chat_right = _chat_right(lines, headers, width)
    header_bottom = max((line.box.bottom for _, line in headers), default=height * 0.12)
    chat_lines = [
        line
        for line in lines
        if line.box.right < chat_right and line.box.top <= height * 0.88
    ]
    players, anchors, warnings = _identify_players(lines, chat_right, header_bottom, height)
    columns = _column_bounds(headers, chat_right)

    if len(players) == 2:
        button, blinds, actions, starts_at_hand, action_confidences, action_warnings = (
            _parse_actions(lines, players, columns, chat_right)
        )
    else:
        button, blinds, actions, starts_at_hand, action_confidences, action_warnings = (
            None,
            [2, 5],
            [],
            False,
            [],
            ["Dealer Chat actions require two identified player names."],
        )
    warnings.extend(action_warnings)

    # Back-fill blinds from the window title when the chat did not show both
    # blind rows. Match the SB/BB PAIR specifically (two numbers around the "/"),
    # so the table id and the ante are ignored, and accept INTEGER blinds:
    # "NLH 961165 - Rs1/Rs2 (0.4)" -> 1/2. The old regex required decimals
    # (X.XX/X.XX) and so silently fell back to the wrong default on 1/2 tables.
    title = next(
        (
            line.text
            for line in lines
            if "DEALER CHAT" in line.text.upper()
            and re.search(r"\d+(?:[.,]\d+)?\s*[/\\]\s*[^\d]{0,3}\d+(?:[.,]\d+)?", line.text)
        ),
        None,
    )
    # The title is the AUTHORITATIVE stake source, so it takes precedence over
    # row-parsed blinds whenever it is present: the tiny SB/BB badges and their
    # amounts OCR unreliably (a "1" small blind read as "d"), whereas the title
    # states the stake plainly. Apply it regardless of starts_at_hand.
    if title:
        pair = re.search(
            r"(\d+(?:[.,]\d+)?)\s*/\s*[^\d]{0,3}(\d+(?:[.,]\d+)?)", title
        )
        if pair:
            def _stake_units(text: str) -> int | None:
                # Blinds may be integers ("1/2") which _decimal_units rejects, so
                # parse directly to hundredth-units: "1"->100, "0.5"->50.
                try:
                    return int(round(float(text.replace(",", ".")) * 100))
                except ValueError:
                    return None

            small = _stake_units(pair.group(1))
            big = _stake_units(pair.group(2))
            # Sanity: small blind positive, big blind between 1x and 4x the small
            # (heads-up BB is ~2x SB). This also rejects a stray table-id/number
            # that happened to sit next to a slash.
            if small and big and 0 < small <= big <= small * 4:
                blinds = [small, big]

    hero_anchor = anchors.get(players[0]) if players else None
    hero_cards, board, card_confidences, card_warnings, card_boxes = _recognize_cards(
        image,
        lines,
        chat_right,
        hero_anchor,
        width,
        height,
        rank_cache=rank_cache,
        hero_card_boxes=hero_card_boxes,
        locate_cache=locate_cache,
    )
    warnings.extend(card_warnings)
    current_stacks = (
        _visible_stacks(lines, players, anchors, chat_right)
        if len(players) == 2
        else [None, None]
    )
    visible_pot = _visible_pot(lines, chat_right)
    inferred = None
    committed = _contributions(button, blinds, actions)
    if committed is not None and all(stack is not None for stack in current_stacks):
        inferred = [
            int(current_stacks[player] or 0) + committed[player]
            for player in range(2)
        ]
    if starting_stacks_override is not None:
        starting_stacks = [int(value) for value in starting_stacks_override]
        if inferred is not None and starting_stacks != inferred:
            warnings.append(
                "The supplied starting stacks differ from the stacks inferred from the table."
            )
    elif inferred is not None and all(stack > 0 for stack in inferred):
        starting_stacks = inferred
    else:
        starting_stacks = [2_000, 2_000]
        warnings.append(
            "Starting stacks could not be inferred; validation used the 20.00 fallback."
        )

    hand_number_match = next(
        (
            re.search(r"\bHAND\s*#?\s*(\d+)", line.text, re.IGNORECASE)
            for line in lines
            if re.search(r"\bHAND\s*#?\s*(\d+)", line.text, re.IGNORECASE)
        ),
        None,
    )
    return CoinPokerLayoutResult(
        chat_lines=chat_lines,
        players=players,
        hand_number=int(hand_number_match.group(1)) if hand_number_match else None,
        button=button,
        blinds=blinds,
        starting_stacks=starting_stacks,
        current_stacks=current_stacks,
        visible_pot=visible_pot,
        hero_cards=hero_cards,
        board=board,
        actions=actions,
        timeline_starts_at_hand=starts_at_hand,
        action_confidences=action_confidences,
        card_confidences=card_confidences,
        warnings=warnings,
        card_boxes=card_boxes,
    )
