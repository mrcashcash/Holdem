"""Optional desktop input: press the poker client's buttons for a decision.

Everything else in this package is read-only: it looks at pixels and writes
hand histories. This module is the single place that can act on the desktop, so
every safety rule lives here rather than being spread across the watcher.

The rules that matter:

- A button is never clicked at a remembered coordinate. The strip is captured
  again at click time, OCR'd, and the click lands on the box whose text matches
  the intended action. A "call" can therefore never press "Fold": if the label
  is not found the attempt is abandoned.
- The matched box must also sit on a filled, saturated button. CoinPoker shows
  pre-action check boxes ("Check/Fold", "Call Any") that contain the same words
  on a dark background; the colour test rejects them.
- A raise types its amount, re-reads the field, and only presses the button when
  the value that is actually on screen matches the intended one.
- One click per decision fingerprint, a cooldown, per-hand and per-session caps,
  a panic key, and a confirmation window: if the validated history does not show
  the action shortly after the click, auto-play disables itself instead of
  clicking blind a second time.

Dry run is the default. It resolves the real click point and logs it without
pressing anything.
"""

from __future__ import annotations

import ctypes
import difflib
import random
import re
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .capture import CaptureRect, WindowInfo, list_windows, window_outer_rect


# Normalized to the poker table window, not to the captured frame: the client
# can be moved or resized and these stay valid. Measured on the 1920x1080
# CoinPoker layout; override per profile with "action_controls".
DEFAULT_ACTION_CONTROLS: dict[str, tuple[float, float, float, float]] = {
    "button_strip": (0.55, 0.885, 1.0, 1.0),
    "amount_field": (0.79, 0.840, 0.91, 0.905),
}

ACTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "fold": re.compile(r"\bFOLD\b"),
    "check": re.compile(r"\bCHECK\b"),
    "call": re.compile(r"\bCALL\b"),
    "raise": re.compile(r"\b(?:RAISE|BET)\b"),
    "all_in": re.compile(r"\bALL[\s-]?IN\b"),
}

ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fold": ("FOLD",),
    "check": ("CHECK",),
    "call": ("CALL",),
    "raise": ("RAISE", "BET", "RAISETO"),
    "all_in": ("ALLIN",),
}

# Button text is small and antialiased: "Check" comes back as "Chec<" or "Ched"
# often enough that exact matching would simply refuse to check. Fuzzy matching
# is allowed, but only when one action wins clearly — a label that is nearly as
# close to a second action is treated as unreadable, not guessed.
MINIMUM_LABEL_SCORE = 0.62
MINIMUM_LABEL_MARGIN = 0.15

# A label containing any of these is a pre-action control or a preset, not the
# button that acts now. "CHECK/FOLD" and "CALL ANY" are the dangerous ones.
REJECTED_LABEL = re.compile(r"/|\bANY\b|\bAUTO\b|\bPRE\b|%")

# Warnings that mean "this recommendation is not solid enough to act on".
# Deliberately a list of the specific ones rather than "any warning at all":
# the champion server attaches an abstraction-mapping note to EVERY response
# (backend/agents/gpu_blueprint_agent.py), so treating any warning as a blocker
# silently refuses every click. These are the ones that actually matter — no
# trained node, a fallback strategy, an untrained blind structure, or an action
# that needed local repair.
BLOCKING_WARNINGS: tuple[str, ...] = (
    r"did not map to a trained node",
    r"serving fallback",
    r"no probability mass",
    r"heuristic fallback",
    r"[Nn]o trained blueprint",
    r"small blind differs",
    r"not legal locally",
    r"failed local validation",
)


def blocking_warnings(
    warnings: Sequence[str],
    patterns: Sequence[str] = BLOCKING_WARNINGS,
) -> list[str]:
    """The subset of a decision's warnings that should prevent a click."""

    return [
        warning
        for warning in warnings
        if any(re.search(pattern, str(warning)) for pattern in patterns)
    ]


def label_scores(text: str) -> dict[str, float]:
    """How strongly one OCR label suggests each action (1.0 = exact word)."""

    upper = text.upper()
    tokens = [token for token in re.findall(r"[A-Z]+", upper) if len(token) >= 2]
    if not tokens:
        return {action: 0.0 for action in ACTION_KEYWORDS}
    candidates = (*tokens, "".join(tokens))
    scores: dict[str, float] = {}
    for action, keywords in ACTION_KEYWORDS.items():
        if ACTION_PATTERNS[action].search(upper):
            scores[action] = 1.0
            continue
        scores[action] = max(
            difflib.SequenceMatcher(None, candidate, keyword).ratio()
            for keyword in keywords
            for candidate in candidates
        )
    return scores


def matched_action(text: str) -> tuple[str, float] | None:
    """The one action a label means, or None when it is ambiguous."""

    scores = label_scores(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_action, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score < MINIMUM_LABEL_SCORE or best_score - runner_up < MINIMUM_LABEL_MARGIN:
        return None
    return best_action, best_score

VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_CONTROL = 0x11
VK_A = 0x41
VK_BACK = 0x08
DEFAULT_PANIC_KEY = 0x7B  # F12

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
MK_LBUTTON = 0x0001
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


@dataclass(frozen=True)
class AutoPlaySettings:
    """Everything that decides whether, and how carefully, a click happens."""

    enabled: bool = False
    # Resolve and log the click without pressing. The safe default; turning it
    # off is the explicit "really press buttons" switch.
    dry_run: bool = True
    # Deliberately stricter than the display threshold: 85% is fine for showing
    # a suggestion, not for pressing a button with money behind it.
    minimum_confidence: float = 0.90
    allowed_actions: tuple[str, ...] = ("fold", "check", "call", "raise", "all_in")
    # A runaway guard, not a style limit: a single hand can legitimately need a
    # decision on every street plus raises, and hitting the cap silently stops
    # play mid-hand.
    maximum_clicks_per_hand: int = 12
    # A click is checked against the screen straight away — if the button is
    # still sitting there, it did not register and is pressed again.
    click_attempts: int = 3
    # How long the client is given to take the button away after a click.
    click_settle_seconds: float = 0.8
    maximum_clicks_per_session: int = 0  # 0 = unlimited
    # Off by default. Consecutive decisions are routinely a FRACTION of a second
    # apart — calling preflop and then acting first on the flop arrives within
    # ~0.1-0.3s — so any cooldown at all silently drops real actions, and it was
    # measured doing exactly that. Nothing depends on it: clicking one spot
    # twice is prevented by the decision fingerprint, runaway clicking by the
    # per-hand cap, the click verification and the unconfirmed-strike counter.
    cooldown_seconds: float = 0.0
    # Randomized pause before clicking, so acting does not look mechanically
    # instant. The table is watched throughout: if the spot moves on during the
    # pause the click is abandoned rather than sent late. Set both to 0 to act
    # as soon as the button is verified.
    minimum_delay_seconds: float = 0.8
    maximum_delay_seconds: float = 2.4
    # Decisions carrying a BLOCKING_WARNINGS warning (an untrained node, a
    # fallback strategy, the CoinPoker 0.4 BB opening-blind mismatch) are
    # advisory only unless this is set. Informational warnings never block.
    allow_warned_decisions: bool = False
    blocking_warning_patterns: tuple[str, ...] = BLOCKING_WARNINGS
    require_foreground: bool = True
    # Auto-play takes the pointer straight away rather than waiting for the
    # mouse to be idle. It only backs off if a hand actively fights it for the
    # pointer for this long — a brief brush is re-aimed through, sustained
    # resistance means a person wants the mouse and gets it. 0 never yields.
    contest_seconds: float = 1.0
    # "input"   synthesized input (SendInput) — indistinguishable from a real
    #           mouse to normal applications, but a client that filters
    #           injected input drops it.
    # "message" posted window messages — bypasses low-level mouse hooks, but
    #           only works if the client reads the mouse from its message queue.
    click_method: str = "input"
    # How long a click has to show up in the validated history. Measured
    # confirmations on a live table ranged from 1s to 13.5s: the action only
    # becomes visible once the client redraws and a frame survives the
    # recognizer's two-frame verification, which can wait on the opponent and
    # the deal animation. Short windows fail clicks that actually worked.
    confirm_seconds: float = 30.0
    # Consecutive clicks that never confirm before auto-play switches itself
    # off. One unconfirmed click is a bad frame or a slow read; several in a
    # row means the clicks are not landing.
    unconfirmed_limit: int = 3
    panic_virtual_key: int = DEFAULT_PANIC_KEY
    table_window_keywords: tuple[str, ...] = ("NLH", "COINPOKER")
    excluded_window_keywords: tuple[str, ...] = ("DEALER CHAT",)
    # Only used by the standalone search below, when the recognizer has not
    # handed over the window it is actually reading. CoinPoker titles its lobby
    # and its tables identically ("CoinPoker"), so a title that names a game is
    # a positive signal but its absence proves nothing.
    table_title_patterns: tuple[str, ...] = (
        r"\bNLH\b",
        r"\bPLO\b",
        r"\d+(?:[.,]\d+)?\s*/\s*\d+",
    )

    def validate(self) -> None:
        if not self.enabled:
            return
        if sys.platform != "win32":
            raise ValueError("Auto-play uses Windows desktop input and needs Windows.")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("Auto-play confidence must be between 0 and 1.")
        if not self.allowed_actions:
            raise ValueError("Auto-play needs at least one allowed action.")
        unknown = set(self.allowed_actions) - set(ACTION_PATTERNS)
        if unknown:
            raise ValueError(f"Unknown auto-play actions: {', '.join(sorted(unknown))}.")
        if self.maximum_clicks_per_hand <= 0:
            raise ValueError("Auto-play clicks per hand must be positive.")
        if self.maximum_clicks_per_session < 0:
            raise ValueError("Auto-play clicks per session must not be negative.")
        if self.cooldown_seconds < 0:
            raise ValueError("Auto-play cooldown must not be negative.")
        if not 0.0 <= self.minimum_delay_seconds <= self.maximum_delay_seconds:
            raise ValueError("Auto-play delay range must be ordered and non-negative.")
        if self.maximum_delay_seconds > 15.0:
            raise ValueError("Auto-play delay must not exceed 15 seconds.")
        if self.confirm_seconds <= 0:
            raise ValueError("Auto-play confirmation window must be positive.")
        if self.unconfirmed_limit <= 0:
            raise ValueError("Auto-play unconfirmed limit must be positive.")
        if self.contest_seconds < 0:
            raise ValueError("Auto-play contest window must not be negative.")
        if self.click_attempts <= 0:
            raise ValueError("Auto-play click attempts must be positive.")
        if self.click_settle_seconds < 0:
            raise ValueError("Auto-play click settle delay must not be negative.")
        if self.click_method not in {"input", "message"}:
            raise ValueError("Auto-play click method must be input or message.")


@dataclass(frozen=True)
class AutoPlayResult:
    """What auto-play did with one decision. ``clicked`` is the only press."""

    status: str  # clicked | dry_run | skipped | aborted | disabled | confirmed | unconfirmed
    message: str
    action: str | None = None
    label: str | None = None
    point: tuple[int, int] | None = None
    typed_amount: str | None = None
    decision_id: str | None = None

    @property
    def pressed(self) -> bool:
        return self.status == "clicked"

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "action": self.action,
            "label": self.label,
            "point": list(self.point) if self.point else None,
            "typed_amount": self.typed_amount,
            "decision_id": self.decision_id,
        }


@dataclass
class _PendingClick:
    decision_id: str
    action: str
    hand_number: int | None
    action_count: int
    deadline: float
    # Total actions by BOTH players when the click went in. If this grows while
    # Hero's own count does not, the table moved past the spot without the
    # history showing Hero acting — the click's fate is unknown rather than
    # known-bad, and unknown must not count against auto-play.
    total_count: int = 0


def _user32():
    if sys.platform != "win32":
        raise RuntimeError("Desktop input is available only on Windows.")
    return ctypes.windll.user32


_ULONG_PTR = wintypes.WPARAM


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("value", _INPUTUNION)]


def _send(*inputs: _INPUT) -> None:
    user32 = _user32()
    array = (_INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), array, ctypes.sizeof(_INPUT))
    if sent != len(inputs):
        raise RuntimeError("Windows rejected the synthesized input.")


def _absolute_point(x: int, y: int) -> tuple[int, int]:
    """Map a desktop pixel to SendInput's 0..65535 virtual-desktop space."""

    user32 = _user32()
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = max(1, user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
    height = max(1, user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
    return (
        int(round((x - left) * 65535 / width)),
        int(round((y - top) * 65535 / height)),
    )


def _move_mouse(x: int, y: int) -> None:
    absolute_x, absolute_y = _absolute_point(x, y)
    _send(
        _INPUT(
            type=INPUT_MOUSE,
            value=_INPUTUNION(
                mi=_MOUSEINPUT(
                    dx=absolute_x,
                    dy=absolute_y,
                    mouseData=0,
                    dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )
    )


def _cursor_position() -> tuple[int, int]:
    point = wintypes.POINT()
    _user32().GetCursorPos(ctypes.byref(point))
    return int(point.x), int(point.y)


class MouseInUse(RuntimeError):
    """The person at the keyboard is using the mouse; auto-play stands down.

    Auto-play borrows the physical pointer for about a second per click. If a
    human hand is on the mouse at that moment the two fight: their click lands
    somewhere they did not aim, or ours does. Losing a click is much better than
    misdirecting theirs, so any sign of human input abandons the attempt.
    """


def _buttons_down() -> bool:
    user32 = _user32()
    return bool(
        user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000
        or user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000
    )


def _glide_to(x: int, y: int, steps: int = 6, *, verify: bool = False) -> None:
    """Move in a few steps so the client sees hover states, not a teleport.

    With ``verify``, each step checks that the pointer actually went where it
    was put. Our own moves land within a pixel, so a larger discrepancy means a
    hand is on the mouse — and we let go rather than wrestle for it.
    """

    start_x, start_y = _cursor_position()
    for step in range(1, max(1, steps) + 1):
        progress = step / float(steps)
        eased = progress * progress * (3.0 - 2.0 * progress)
        target = (
            int(round(start_x + (x - start_x) * eased)),
            int(round(start_y + (y - start_y) * eased)),
        )
        _move_mouse(*target)
        time.sleep(random.uniform(0.008, 0.022))
        if verify:
            actual = _cursor_position()
            if max(abs(actual[0] - target[0]), abs(actual[1] - target[1])) > 6:
                raise MouseInUse("the mouse moved while auto-play was aiming")


def _make_lparam(x: int, y: int) -> int:
    return ((y & 0xFFFF) << 16) | (x & 0xFFFF)


def _post_click(handle: int, x: int, y: int) -> None:
    """Deliver a click as window messages instead of synthesized input.

    Posted messages go straight to the window's queue, so they bypass any
    low-level mouse hook — which is how a client that ignores injected button
    events usually implements that. Whether the client honours them depends on
    how it reads the mouse, so this is a fallback, not a better default.
    """

    user32 = _user32()
    point = wintypes.POINT(int(x), int(y))
    if not user32.ScreenToClient(wintypes.HWND(handle), ctypes.byref(point)):
        raise RuntimeError("Could not map the click into the window's client area.")
    lparam = _make_lparam(point.x, point.y)
    user32.PostMessageW(wintypes.HWND(handle), WM_MOUSEMOVE, 0, lparam)
    time.sleep(random.uniform(0.03, 0.06))
    user32.PostMessageW(wintypes.HWND(handle), WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(random.uniform(0.09, 0.16))
    user32.PostMessageW(wintypes.HWND(handle), WM_LBUTTONUP, 0, lparam)


def _post_text(handle: int, text: str) -> None:
    user32 = _user32()
    for _ in range(12):
        user32.PostMessageW(wintypes.HWND(handle), WM_KEYDOWN, VK_BACK, 0)
        user32.PostMessageW(wintypes.HWND(handle), WM_KEYUP, VK_BACK, 0)
        time.sleep(0.01)
    for character in text:
        user32.PostMessageW(wintypes.HWND(handle), WM_CHAR, ord(character), 0)
        time.sleep(random.uniform(0.03, 0.07))


def _click(x: int, y: int, *, clicks: int = 1, contest_seconds: float = 0.0) -> None:
    """Take the pointer and click, re-aiming through brief interference.

    With ``contest_seconds`` at 0 the pointer is simply taken. Otherwise a hand
    on the mouse makes us re-aim and try again, and only sustained resistance
    for that long hands the mouse back — a person who actually wants it keeps
    moving, a stray brush does not.
    """

    contested_since: float | None = None
    while True:
        try:
            if contest_seconds > 0 and _buttons_down():
                raise MouseInUse("a mouse button is being held down")
            _glide_to(x, y, verify=contest_seconds > 0)
            break
        except MouseInUse as exc:
            now = time.monotonic()
            if contested_since is None:
                contested_since = now
            if now - contested_since >= contest_seconds:
                raise MouseInUse(
                    f"{exc} for over {contest_seconds:g}s"
                ) from None
            time.sleep(0.04)
    # Settle on the control before pressing, and hold the button for a human
    # length of time. A very short press can fall between a client's input
    # samples, which looks identical to the click being ignored.
    time.sleep(random.uniform(0.15, 0.30))
    for index in range(clicks):
        if index:
            time.sleep(random.uniform(0.05, 0.09))
        _send(
            _INPUT(
                type=INPUT_MOUSE,
                value=_INPUTUNION(
                    mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, 0)
                ),
            )
        )
        time.sleep(random.uniform(0.09, 0.16))
        _send(
            _INPUT(
                type=INPUT_MOUSE,
                value=_INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, 0)),
            )
        )


def _key(virtual_key: int, *, up: bool = False) -> _INPUT:
    return _INPUT(
        type=INPUT_KEYBOARD,
        value=_INPUTUNION(
            ki=_KEYBDINPUT(
                wVk=virtual_key,
                wScan=0,
                dwFlags=KEYEVENTF_KEYUP if up else 0,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def _unicode_key(character: str, *, up: bool = False) -> _INPUT:
    return _INPUT(
        type=INPUT_KEYBOARD,
        value=_INPUTUNION(
            ki=_KEYBDINPUT(
                wVk=0,
                wScan=ord(character),
                dwFlags=KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0),
                time=0,
                dwExtraInfo=0,
            )
        ),
    )


def _clear_field() -> None:
    """Select-all then delete, with backspaces for inputs that ignore Ctrl+A."""

    _send(_key(VK_CONTROL), _key(VK_A))
    time.sleep(0.03)
    _send(_key(VK_A, up=True), _key(VK_CONTROL, up=True))
    time.sleep(0.05)
    for _ in range(12):
        _send(_key(VK_BACK), _key(VK_BACK, up=True))
        time.sleep(random.uniform(0.012, 0.03))


def _type_text(text: str) -> None:
    for character in text:
        _send(_unicode_key(character), _unicode_key(character, up=True))
        time.sleep(random.uniform(0.03, 0.08))


def live_window_rect(window: WindowInfo) -> CaptureRect:
    """Re-read a known window's current desktop rectangle.

    Always called immediately before clicking: the client may have been moved
    or resized since the recognizer found it.
    """

    if _user32().IsIconic(window.handle):
        raise RuntimeError(f"The poker window '{window.title}' is minimized.")
    rect = window_outer_rect(window)
    if rect.width < 600 or rect.height < 450:
        raise RuntimeError(
            f"The poker window '{window.title}' is too small to click reliably "
            f"({rect.width}x{rect.height})."
        )
    return rect


def list_table_candidates(
    settings: AutoPlaySettings,
) -> list[tuple[bool, int, WindowInfo, CaptureRect]]:
    """Every visible window that could be the poker table, best guess first.

    Returned as ``(title_names_a_game, area, window, rect)``. CoinPoker gives
    its lobby and its tables the same window title and draws the game name in
    its own custom title bar, so this list is genuinely ambiguous — which is
    why auto-play prefers the window the recognizer is reading.
    """

    user32 = _user32()
    candidates: list[tuple[bool, int, WindowInfo, CaptureRect]] = []
    for window in list_windows():
        upper = window.title.upper()
        if not any(keyword in upper for keyword in settings.table_window_keywords):
            continue
        if any(keyword in upper for keyword in settings.excluded_window_keywords):
            continue
        if user32.IsIconic(window.handle):
            continue
        try:
            rect = window_outer_rect(window)
        except RuntimeError:
            continue
        if rect.width < 600 or rect.height < 450:
            continue
        named = any(
            re.search(pattern, upper) for pattern in settings.table_title_patterns
        )
        candidates.append((named, rect.width * rect.height, window, rect))
    # A title that names a game is a table for certain; otherwise larger first,
    # which is only a guess — the lobby can win it.
    candidates.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return candidates


def find_table_window(settings: AutoPlaySettings) -> tuple[WindowInfo, CaptureRect]:
    """Search for the poker table window without the recognizer's help.

    A fallback for the calibration tool and for profiles whose recognizer does
    not localize the client. During a watch session, auto-play uses the window
    the recognizer is actually reading instead of this heuristic.
    """

    candidates = list_table_candidates(settings)
    if not candidates:
        raise RuntimeError(
            "No visible poker table window was found for auto-play "
            f"(looking for {', '.join(settings.table_window_keywords)})."
        )
    return candidates[0][2], candidates[0][3]


def region_rect(
    window_rect: CaptureRect,
    bounds: Sequence[float],
) -> CaptureRect:
    """A normalized table-window box as an absolute desktop rectangle."""

    left, top, right, bottom = (float(value) for value in bounds)
    absolute_left = int(round(window_rect.left + left * window_rect.width))
    absolute_top = int(round(window_rect.top + top * window_rect.height))
    absolute_right = int(round(window_rect.left + right * window_rect.width))
    absolute_bottom = int(round(window_rect.top + bottom * window_rect.height))
    return CaptureRect(
        left=absolute_left,
        top=absolute_top,
        width=max(1, absolute_right - absolute_left),
        height=max(1, absolute_bottom - absolute_top),
    )


def _median_saturation(image, box, padding: int = 10) -> tuple[float, float]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    height, width = image.shape[:2]
    left = max(0, int(box.left) - padding)
    top = max(0, int(box.top) - padding)
    right = min(width, int(box.right) + padding)
    bottom = min(height, int(box.bottom) + padding)
    patch = image[top:bottom, left:right]
    if patch.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 1])), float(np.median(hsv[:, :, 2]))


def ocr_controls(image, minimum_height: int = 220) -> list[Any]:
    """OCR a control strip, upscaled first, with boxes in the strip's own pixels.

    The action strip is a short band (about 100 px tall). RapidOCR's detector
    returns nothing at all on it when the controls are dark-on-dark, which is
    how CoinPoker draws its pre-action row — and a detector that silently finds
    nothing would look exactly like "auto-play never clicks". Upscaling first
    costs a millisecond and makes detection reliable at this size.
    """

    import cv2  # type: ignore

    from .recognition import Box, OcrLine, run_ocr

    height, width = image.shape[:2]
    scale = max(1.0, float(minimum_height) / float(max(1, height)))
    if scale <= 1.0:
        return run_ocr(image)
    scaled = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return [
        OcrLine(
            text=line.text,
            confidence=line.confidence,
            box=Box(
                line.box.left / scale,
                line.box.top / scale,
                line.box.right / scale,
                line.box.bottom / scale,
            ),
        )
        for line in run_ocr(scaled)
    ]


def slot_lines(image, slots: int = 3) -> list[Any]:
    """Read the action strip without running text detection.

    Detection is the expensive half of OCR: on this strip it costs three to five
    SECONDS, which is most of the delay between deciding and clicking. The
    client always lays the action buttons out as equal columns, so recognizing
    each column directly is ~80 ms — the same answer, 40x faster. Detection
    remains the fallback for a layout this does not fit.
    """

    import cv2  # type: ignore

    from .recognition import Box, OcrLine, recognize_text_strip

    height, width = image.shape[:2]
    lines: list[Any] = []
    for index in range(slots):
        left = int(width * index / slots)
        right = int(width * (index + 1) / slots)
        crop = image[:, left:right]
        if crop.size == 0:
            continue
        # Buttons that carry an amount are drawn on two lines ("Call" above
        # "0.04"). Recognition without detection treats a crop as ONE line, so
        # the whole face comes back as "Ca04" or "S A01" and matches nothing.
        # Reading the label band on its own gives a clean "Call". Single-line
        # buttons resolve on the first try, so this costs them nothing.
        text, score = "", 0.0
        band_height = crop.shape[0]
        for band in (
            crop,
            crop[: int(band_height * 0.55), :],
            crop[int(band_height * 0.45) :, :],
        ):
            if band.size == 0:
                continue
            scaled = cv2.resize(band, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            candidate, candidate_score = recognize_text_strip(scaled)
            if not str(candidate).strip():
                continue
            if not str(text).strip():
                text, score = candidate, candidate_score
            if matched_action(str(candidate)) is not None:
                text, score = candidate, candidate_score
                break
        if not str(text).strip():
            continue
        # Report the middle of the column: it is where the button face is, so
        # both the click point and the colour test land on the button itself
        # rather than on the gap between buttons.
        inset_x = (right - left) * 0.15
        inset_y = height * 0.15
        lines.append(
            OcrLine(
                text=text,
                confidence=score,
                box=Box(left + inset_x, inset_y, right - inset_x, height - inset_y),
            )
        )
    return lines


def _has_button_pixels(image, minimum_fraction: float = 0.05) -> bool:
    """Is anything button-like drawn here at all?

    Action buttons are large saturated blocks of colour and cover most of the
    strip; an empty strip between hands has none. Sub-millisecond, and it saves
    OCR from spending seconds confirming a blank image.
    """

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if image is None or getattr(image, "size", 0) == 0:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    solid = (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 60)
    return bool(np.count_nonzero(solid) >= minimum_fraction * solid.size)


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(r"\d+(?:[.,]\d+)?", text.replace(" ", "")):
        try:
            values.append(float(token.replace(",", ".")))
        except ValueError:
            continue
    return values


class AutoPlayer:
    """Turns validated decisions into verified clicks, or into refusals."""

    def __init__(
        self,
        settings: AutoPlaySettings,
        controls: dict[str, tuple[float, float, float, float]] | None = None,
        notify: Callable[[AutoPlayResult], None] | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.controls = {**DEFAULT_ACTION_CONTROLS, **(controls or {})}
        # Used only for events nobody is waiting on — the panic key, which can
        # fire while no decision is in flight.
        self._notify = notify
        self._lock = threading.RLock()
        self._panic = threading.Event()
        self._stop_panic_watch = threading.Event()
        self._panic_thread: threading.Thread | None = None
        self._capture: Any = None
        self._disabled_reason: str | None = None
        self._executed: set[str] = set()
        self._last_click_at = 0.0
        self._hand_key: tuple[Any, ...] | None = None
        self._hand_clicks = 0
        self._session_clicks = 0
        self._pending: _PendingClick | None = None
        self._strikes = 0

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if not self.settings.enabled or self._panic_thread is not None:
            return
        self._panic_thread = threading.Thread(
            target=self._watch_panic_key,
            name="screen-history-autoplay-panic",
            daemon=True,
        )
        self._panic_thread.start()

    def stop(self) -> None:
        self._stop_panic_watch.set()
        thread = self._panic_thread
        self._panic_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        if self._capture is not None:
            try:
                self._capture.close()
            except (RuntimeError, OSError):
                pass
            self._capture = None

    def _watch_panic_key(self) -> None:
        try:
            user32 = _user32()
        except RuntimeError:
            return
        key = self.settings.panic_virtual_key
        while not self._stop_panic_watch.wait(0.05):
            if user32.GetAsyncKeyState(key) & 0x8000:
                self._panic.set()
                message = "Auto-play stopped: the panic key was pressed."
                self.disable("the panic key was pressed.")
                if self._notify is not None:
                    self._notify(AutoPlayResult(status="disabled", message=message))
                return

    def disable(self, reason: str) -> None:
        with self._lock:
            if self._disabled_reason is None:
                self._disabled_reason = reason

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    @property
    def active(self) -> bool:
        return self.settings.enabled and self._disabled_reason is None

    def describe(self) -> str:
        mode = "DRY RUN (no clicks)" if self.settings.dry_run else "LIVE CLICKING"
        think = (
            "no think time"
            if self.settings.maximum_delay_seconds <= 0
            else (
                f"think {self.settings.minimum_delay_seconds:g}"
                f"-{self.settings.maximum_delay_seconds:g}s"
            )
        )
        return (
            f"Auto-play {mode} • {self.settings.click_method} clicks • {think} • "
            f"min confidence {self.settings.minimum_confidence:.0%} • "
            f"{', '.join(self.settings.allowed_actions)} • "
            f"max {self.settings.maximum_clicks_per_hand}/hand • "
            f"panic key 0x{self.settings.panic_virtual_key:02X}"
        )

    # ------------------------------------------------------------------ capture

    def _grab(self, rect: CaptureRect):
        from .capture import ScreenCapture

        if self._capture is None:
            self._capture = ScreenCapture(region=rect)
        return self._capture.grab(rect)

    # -------------------------------------------------------------- pressing

    def _press(self, handle: int, x: int, y: int) -> None:
        if self.settings.click_method == "message":
            # Posted messages never touch the physical pointer, so they cannot
            # interfere with the person using the mouse.
            _post_click(handle, x, y)
        else:
            _click(x, y, contest_seconds=self.settings.contest_seconds)

    def _button_gone(self, window_rect: CaptureRect, intent: str) -> bool:
        """Did the button we just pressed disappear? That is the click landing.

        The client removes the action buttons the instant Hero acts, seconds
        before that action can be read back out of the hand history — so this
        is the fast, direct check on whether a press registered.
        """

        try:
            self._find_button(window_rect, intent)
        except RuntimeError:
            return True
        return False

    def _press_and_verify(
        self,
        window: WindowInfo,
        window_rect: CaptureRect,
        intent: str,
        point: tuple[int, int],
        spot_changed: Callable[[], str | None],
    ) -> tuple[tuple[int, int], int]:
        """Press, check the screen, and press again if nothing happened."""

        attempts = max(1, self.settings.click_attempts)
        for attempt in range(1, attempts + 1):
            if self._panic.is_set():
                raise RuntimeError("The panic key was pressed.")
            self._press(window.handle, *point)
            if self.settings.click_settle_seconds > 0:
                time.sleep(self.settings.click_settle_seconds)
            if self._button_gone(window_rect, intent):
                return point, attempt
            # The button is still on screen. Only press again while the table
            # genuinely has not moved on — otherwise the strip we are looking
            # at belongs to a later spot and clicking it would act twice.
            changed = spot_changed()
            if changed is not None:
                return point, attempt
            if attempt == attempts:
                raise RuntimeError(
                    f"The '{intent}' button was still on screen after {attempts} "
                    "clicks; the client did not accept them."
                )
            # Re-locate before retrying: the strip may have shifted or
            # re-rendered around a staged amount.
            try:
                _strip, _image, point, _label, _box = self._find_button(
                    window_rect, intent
                )
            except RuntimeError:
                # It cleared between the check and the re-aim — the press did
                # land, just slowly. Nothing left to retry.
                return point, attempt
        raise RuntimeError(f"The '{intent}' click could not be verified.")

    def _write(self, handle: int, text: str) -> None:
        if self.settings.click_method == "message":
            _post_text(handle, text)
        else:
            _clear_field()
            _type_text(text)

    # ------------------------------------------------------------- verification

    def confirm_from_state(self, state: Any) -> AutoPlayResult | None:
        """Check a recognized state against the click we are waiting on.

        The validated history is the only proof a click landed. Without this,
        a click that silently missed would be followed by more blind clicks.
        """

        with self._lock:
            pending = self._pending
            if pending is None:
                return None
            hero_actions = [
                action
                for action in getattr(state, "visible_actions", ())
                if int(getattr(action, "player", -1)) == 0
            ]
            if (
                state.hand_number is not None
                and pending.hand_number is not None
                and state.hand_number != pending.hand_number
            ):
                self._pending = None
                return AutoPlayResult(
                    status="confirmed",
                    message=(
                        f"Auto-play {pending.action} accepted: the table moved on to "
                        f"hand #{state.hand_number}."
                    ),
                    action=pending.action,
                    decision_id=pending.decision_id,
                )
            if len(hero_actions) > pending.action_count:
                observed = hero_actions[-1]
                self._pending = None
                matched = str(observed.action) == pending.action or (
                    pending.action in {"raise", "all_in"}
                    and str(observed.action) == "raise"
                )
                if matched:
                    self._strikes = 0
                    return AutoPlayResult(
                        status="confirmed",
                        message=f"Auto-play {pending.action} confirmed in the hand history.",
                        action=pending.action,
                        decision_id=pending.decision_id,
                    )
                self.disable(
                    f"The table recorded '{observed.action}' after auto-play clicked "
                    f"'{pending.action}'."
                )
                return AutoPlayResult(
                    status="unconfirmed",
                    message=(
                        f"Auto-play disabled: clicked '{pending.action}' but the hand "
                        f"history recorded '{observed.action}'."
                    ),
                    action=pending.action,
                    decision_id=pending.decision_id,
                )
            if time.monotonic() >= pending.deadline:
                self._pending = None
                total = len(getattr(state, "visible_actions", ()) or ())
                if total > pending.total_count:
                    # Play moved on, but Hero's action never showed up in the
                    # read history — the Dealer Chat dropped the row. Seen live:
                    # the opponent's call of Hero's bet was recorded while the
                    # bet itself was not. The click plainly worked; the history
                    # is just incomplete, so this is no evidence against it.
                    return AutoPlayResult(
                        status="inconclusive",
                        message=(
                            f"Auto-play {pending.action}: the table moved on but the "
                            "hand history never recorded Hero's action, so the click "
                            "could not be confirmed either way."
                        ),
                        action=pending.action,
                        decision_id=pending.decision_id,
                    )
                return self._strike(
                    pending,
                    f"the '{pending.action}' click did not appear in the validated "
                    f"history within {self.settings.confirm_seconds:g}s",
                )
        return None

    def _strike(self, pending: _PendingClick, detail: str) -> AutoPlayResult:
        """Record a click that never showed up; switch off only on a streak."""

        self._strikes += 1
        limit = self.settings.unconfirmed_limit
        if self._strikes >= limit:
            self.disable(f"{self._strikes} clicks in a row were never confirmed.")
            return AutoPlayResult(
                status="unconfirmed",
                message=(
                    f"Auto-play disabled after {self._strikes} unconfirmed clicks: {detail}."
                ),
                action=pending.action,
                decision_id=pending.decision_id,
            )
        return AutoPlayResult(
            status="unconfirmed",
            message=(
                f"Auto-play {self._strikes}/{limit} unconfirmed: {detail}. "
                "Still enabled."
            ),
            action=pending.action,
            decision_id=pending.decision_id,
        )

    # -------------------------------------------------------------------- gates

    def _intent(self, decision: Any) -> str:
        action = str(decision.action)
        if action != "all_in":
            return action
        # The engine calls a shove "all_in" whether it is a raise to the stack
        # or a call that covers it (decision.py). Only the raise needs typing.
        try:
            if int(decision.to_call) >= int(decision.stacks[0]):
                return "call"
        except (TypeError, ValueError, IndexError):
            pass
        return "raise"

    def _refuse(self, decision: Any, message: str, status: str = "skipped") -> AutoPlayResult:
        return AutoPlayResult(
            status=status,
            message=message,
            action=str(decision.action),
            decision_id=getattr(decision, "decision_id", None),
        )

    def _pre_click_gates(self, decision: Any) -> AutoPlayResult | None:
        with self._lock:
            if self._disabled_reason is not None:
                return self._refuse(
                    decision, f"Auto-play is off: {self._disabled_reason}", "disabled"
                )
            fingerprint = str(getattr(decision, "state_fingerprint", "")) or str(
                getattr(decision, "decision_id", "")
            )
            if fingerprint and fingerprint in self._executed:
                return self._refuse(
                    decision, "This exact spot was already auto-played."
                )
            # Identify the hand by its hole cards as well as its number. The
            # CoinPoker recognizer assigns numbers itself and can keep calling a
            # new hand by the old number for a while; keying the per-hand cap on
            # the number alone then charges a fresh hand for the previous one's
            # clicks and stops play dead. New cards are unambiguously a new hand.
            hand_key = (
                decision.hand_number,
                tuple(getattr(decision, "hero_cards", ()) or ()),
            )
            if hand_key != self._hand_key:
                self._hand_key = hand_key
                self._hand_clicks = 0
            if self._hand_clicks >= self.settings.maximum_clicks_per_hand:
                return self._refuse(
                    decision,
                    "Auto-play reached its per-hand click limit "
                    f"({self.settings.maximum_clicks_per_hand}).",
                )
            if (
                self.settings.maximum_clicks_per_session
                and self._session_clicks >= self.settings.maximum_clicks_per_session
            ):
                self.disable("The auto-play session click limit was reached.")
                return self._refuse(
                    decision, "Auto-play reached its session click limit.", "disabled"
                )
            waited = time.monotonic() - self._last_click_at
            if waited < self.settings.cooldown_seconds:
                return self._refuse(
                    decision,
                    f"Auto-play cooldown has {self.settings.cooldown_seconds - waited:.1f}s left.",
                )
            if self._pending is not None:
                # A new decision means the table asked Hero to act again, which
                # only happens once the history has advanced — so if the earlier
                # click still has not been seen, it did not land. Resolve it now
                # rather than holding play up: re-clicking the SAME spot is
                # already impossible (the fingerprint is remembered), and a real
                # streak of failures still switches auto-play off.
                stale = self._pending
                self._pending = None
                self._strike(
                    stale,
                    f"the '{stale.action}' click was still unconfirmed when the "
                    "table asked for another decision",
                )
                if self._disabled_reason is not None:
                    return self._refuse(
                        decision,
                        f"Auto-play is off: {self._disabled_reason}",
                        "disabled",
                    )
        intent = self._intent(decision)
        if intent not in self.settings.allowed_actions and (
            str(decision.action) not in self.settings.allowed_actions
        ):
            return self._refuse(decision, f"Auto-play is not allowed to {intent}.")
        if decision.recognition_confidence < self.settings.minimum_confidence:
            return self._refuse(
                decision,
                f"Recognition confidence {decision.recognition_confidence:.0%} is below the "
                f"{self.settings.minimum_confidence:.0%} auto-play minimum.",
            )
        if not self.settings.allow_warned_decisions:
            blocking = blocking_warnings(
                decision.warnings or (), self.settings.blocking_warning_patterns
            )
            if blocking:
                return self._refuse(
                    decision,
                    f"The decision carries a blocking warning: {blocking[0]}",
                )
        return None

    def _activate(self, window: WindowInfo) -> str | None:
        user32 = _user32()
        if int(user32.GetForegroundWindow()) == int(window.handle):
            return None
        if not self.settings.require_foreground:
            return None
        user32.SetForegroundWindow(window.handle)
        time.sleep(0.15)
        if int(user32.GetForegroundWindow()) != int(window.handle):
            return (
                f"The poker window '{window.title}' could not be brought to the "
                "foreground; the click was not sent."
            )
        time.sleep(0.1)
        return None

    def _wait(self, spot_changed: Callable[[], str | None]) -> str | None:
        """Optional pause before clicking, abandoned if the table moves on."""

        delay = random.uniform(
            self.settings.minimum_delay_seconds, self.settings.maximum_delay_seconds
        )
        if delay <= 0:
            return None
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if self._panic.is_set():
                return "The panic key was pressed while waiting to click."
            changed = spot_changed()
            if changed is not None:
                return f"The table advanced while waiting to click: {changed}"
        return None

    # ------------------------------------------------------------------- reading

    def _strip_lines(self, window_rect: CaptureRect) -> tuple[CaptureRect, Any, list[Any]]:
        strip = region_rect(window_rect, self.controls["button_strip"])
        image = self._grab(strip)
        return strip, image, ocr_controls(image)

    def _find_button(
        self,
        window_rect: CaptureRect,
        intent: str,
    ) -> tuple[CaptureRect, Any, tuple[int, int], str, Any]:
        """Locate the button for ``intent``, cheapest reliable read first."""

        attempts: list[str] = []
        for retry in range(2):
            strip = region_rect(window_rect, self.controls["button_strip"])
            image = self._grab(strip)
            if not _has_button_pixels(image):
                # Nothing is drawn here — between hands, or the client has not
                # rendered the controls yet. Skip OCR entirely; it would spend
                # seconds confirming an empty picture.
                attempts.append("no action buttons are on screen")
                if retry == 0:
                    time.sleep(0.35)
                continue
            quick = slot_lines(image)
            if quick:
                try:
                    point, label, box = self._resolve_button(
                        intent, strip, image, quick
                    )
                    return strip, image, point, label, box
                except RuntimeError as exc:
                    attempts.append(str(exc))
            # Detection costs seconds, so only pay for it when the quick read
            # looks unreliable. Three clean labels that simply do not include
            # the one we want means the button is not on screen, and detection
            # would only confirm that slowly.
            if len(quick) < 2:
                try:
                    lines = ocr_controls(image)
                except (RuntimeError, ValueError) as exc:
                    attempts.append(str(exc))
                    lines = []
                if lines:
                    try:
                        point, label, box = self._resolve_button(
                            intent, strip, image, lines
                        )
                        return strip, image, point, label, box
                    except RuntimeError as exc:
                        attempts.append(str(exc))
            if retry == 0:
                # A frame caught mid-animation reads as noise ("FOL", "9144").
                # One re-read avoids throwing away a decision over one bad
                # capture.
                time.sleep(0.35)
        raise RuntimeError(attempts[-1] if attempts else "The action strip was unreadable.")

    def _resolve_button(
        self,
        intent: str,
        strip: CaptureRect,
        image: Any,
        lines: Sequence[Any],
    ) -> tuple[tuple[int, int], str, Any]:
        """Find the button that says what we intend to do. Never a guess."""

        matches: list[tuple[Any, str]] = []
        seen: list[str] = []
        for line in lines:
            text = str(line.text).upper().strip()
            if not text:
                continue
            seen.append(text)
            if REJECTED_LABEL.search(text):
                continue
            resolved = matched_action(text)
            if resolved is not None and resolved[0] == intent:
                matches.append((line, text))
        if not matches:
            visible = ", ".join(f"'{text}'" for text in seen) or "nothing"
            raise RuntimeError(
                f"No '{intent}' button was readable in the action strip (saw {visible})."
            )
        if len(matches) > 1:
            labels = ", ".join(sorted(f"'{text}'" for _line, text in matches))
            raise RuntimeError(
                f"The action strip showed more than one '{intent}' control ({labels})."
            )
        line, text = matches[0]
        saturation, value = _median_saturation(image, line.box)
        if saturation < 60.0 or value < 60.0:
            raise RuntimeError(
                f"'{text}' does not sit on a filled action button "
                f"(saturation {saturation:.0f}, brightness {value:.0f}); it looks like a "
                "pre-action control."
            )
        point = (
            strip.left + int(round(line.box.center_x)),
            strip.top + int(round(line.box.center_y)),
        )
        return point, text, line.box

    def _button_shows_amount(
        self,
        window_rect: CaptureRect,
        button_box: Any,
        target: float,
        tolerance: float,
    ) -> bool:
        """Second witness: CoinPoker mirrors the staged amount on the button.

        Only the amount printed on the button itself counts — the lower half of
        the button face, where the client draws it. A number lying loose on the
        felt (a side pot, a stack) must never be able to confirm a bet.
        """

        import cv2  # type: ignore

        from .recognition import recognize_text_strip

        strip = region_rect(window_rect, self.controls["button_strip"])
        image = self._grab(strip)
        height, width = image.shape[:2]
        left = max(0, int(button_box.left))
        right = min(width, int(button_box.right))
        top = min(height - 1, int(button_box.top + button_box.height * 0.45))
        crop = image[top:min(height, int(button_box.bottom)), left:right]
        if crop.size == 0:
            return False
        saturation, brightness = _median_saturation(image, button_box)
        if saturation < 60.0 or brightness < 60.0:
            return False
        scaled = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        observed, _score = recognize_text_strip(scaled)
        return any(
            abs(value - target) <= tolerance for value in _numbers(str(observed))
        )

    def _set_amount(
        self,
        window_rect: CaptureRect,
        amount: int,
        amount_scale: int,
        button_box: Any,
        handle: int,
    ) -> str:
        """Type the raise-to amount and prove it is on screen. Raises on doubt."""

        from .recognition import recognize_text_strip

        bounds = self.controls.get("amount_field")
        if not bounds:
            raise RuntimeError(
                "No amount field is calibrated for this profile; a raise cannot be typed."
            )
        target = float(amount) / float(max(1, amount_scale))
        text = f"{target:.2f}" if amount_scale > 1 else str(int(amount))
        field = region_rect(window_rect, bounds)
        tolerance = 0.5 / float(max(1, amount_scale))
        observed = ""
        # Typing can miss if the field did not take focus on the first click,
        # which leaves the client's own suggested amount sitting there. Retry
        # once; never press the button on an amount we have not seen.
        for attempt in range(2):
            self._press(
                handle,
                field.left + field.width // 2,
                field.top + field.height // 2,
            )
            time.sleep(0.12)
            self._write(handle, text)
            time.sleep(0.25)
            observed, _score = recognize_text_strip(self._grab(field))
            if any(abs(value - target) <= tolerance for value in _numbers(observed)):
                return text
            if self._button_shows_amount(window_rect, button_box, target, tolerance):
                return text
        raise RuntimeError(
            f"The bet amount on screen ('{observed.strip() or 'unreadable'}') does not "
            f"match the intended {text} after two attempts; nothing was pressed."
        )

    # ------------------------------------------------------------------ the act

    def _locate(self, table_window: WindowInfo | None) -> tuple[WindowInfo, CaptureRect]:
        """Prefer the window the recognizer is reading over any search."""

        if table_window is not None:
            return table_window, live_window_rect(table_window)
        return find_table_window(self.settings)

    def execute(
        self,
        decision: Any,
        *,
        amount_scale: int = 1,
        spot_changed: Callable[[], str | None] = lambda: None,
        table_window: WindowInfo | None = None,
    ) -> AutoPlayResult:
        """Click the client for one decision, or explain why it did not.

        ``spot_changed`` reports whether the poker spot this decision was made
        for is still the one on the table, returning a human-readable reason
        when it is not. It is asked repeatedly: before the click, after the
        humanization pause, and again after a bet amount is typed.
        """

        if not self.settings.enabled:
            return self._refuse(decision, "Auto-play is disabled.", "disabled")
        refusal = self._pre_click_gates(decision)
        if refusal is not None:
            return refusal
        intent = self._intent(decision)
        fingerprint = str(getattr(decision, "state_fingerprint", "")) or str(
            getattr(decision, "decision_id", "")
        )

        try:
            window, window_rect = self._locate(table_window)
            reason = self._wait(spot_changed)
            if reason is not None:
                return self._refuse(decision, reason, "aborted")
            changed = spot_changed()
            if changed is not None:
                return self._refuse(
                    decision, f"The table advanced before the click: {changed}", "aborted"
                )
            # Re-read geometry after the pause: the window may have been moved.
            window, window_rect = self._locate(table_window)
            strip, image, point, label, button_box = self._find_button(
                window_rect, intent
            )

            typed: str | None = None
            if self.settings.dry_run:
                if intent == "raise" and decision.amount is not None:
                    typed = (
                        f"{float(decision.amount) / float(max(1, amount_scale)):.2f}"
                        if amount_scale > 1
                        else str(int(decision.amount))
                    )
                return AutoPlayResult(
                    status="dry_run",
                    message=(
                        f"Auto-play (dry run) would press '{label}' at "
                        f"{point[0]},{point[1]}"
                        + (f" after typing {typed}" if typed else "")
                        + "."
                    ),
                    action=intent,
                    label=label,
                    point=point,
                    typed_amount=typed,
                    decision_id=getattr(decision, "decision_id", None),
                )

            blocked = self._activate(window)
            if blocked is not None:
                return self._refuse(decision, blocked, "aborted")

            cursor = _cursor_position()
            try:
                if intent == "raise":
                    if decision.amount is None:
                        raise RuntimeError("The decision is a raise without an amount.")
                    typed = self._set_amount(
                        window_rect,
                        int(decision.amount),
                        amount_scale,
                        button_box,
                        window.handle,
                    )
                    changed = spot_changed()
                    if changed is not None:
                        raise RuntimeError(
                            f"The table advanced after the amount was typed: {changed}"
                        )
                    # The strip re-renders around the staged amount, so find the
                    # button again rather than trusting the earlier point.
                    strip, image, point, label, button_box = self._find_button(
                        window_rect, intent
                    )
                point, attempts_used = self._press_and_verify(
                    window, window_rect, intent, point, spot_changed
                )
            finally:
                # Put the pointer back only if it is still ours. If a hand took
                # it while we were clicking, dragging it back to a stale spot is
                # exactly the interference we are trying to avoid.
                try:
                    if (
                        self.settings.click_method != "message"
                        and point is not None
                        and max(
                            abs(_cursor_position()[0] - point[0]),
                            abs(_cursor_position()[1] - point[1]),
                        )
                        <= 6
                    ):
                        _glide_to(*cursor, steps=3)
                except (RuntimeError, OSError):
                    pass
        except (RuntimeError, ValueError, OSError) as exc:
            return self._refuse(decision, str(exc), "aborted")

        with self._lock:
            if fingerprint:
                self._executed.add(fingerprint)
            self._last_click_at = time.monotonic()
            self._hand_clicks += 1
            self._session_clicks += 1
            self._pending = _PendingClick(
                decision_id=str(getattr(decision, "decision_id", "")),
                action=intent,
                hand_number=decision.hand_number,
                action_count=sum(
                    1
                    for entry in getattr(decision, "action_signature", ())
                    if int(entry[0]) == 0
                ),
                deadline=time.monotonic() + self.settings.confirm_seconds,
                total_count=len(getattr(decision, "action_signature", ()) or ()),
            )
        return AutoPlayResult(
            status="clicked",
            message=(
                f"Auto-play pressed '{label}' at {point[0]},{point[1]}"
                + (f" for {typed}" if typed else "")
                + (
                    f" (took {attempts_used} presses)"
                    if attempts_used > 1
                    else ""
                )
                + ", and the button cleared."
            ),
            action=intent,
            label=label,
            point=point,
            typed_amount=typed,
            decision_id=getattr(decision, "decision_id", None),
        )
