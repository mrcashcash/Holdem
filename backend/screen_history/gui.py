"""Tkinter control panel for pixel-only poker hand reconstruction."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import tkinter as tk
from ctypes import wintypes
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from .autoplay import AutoPlaySettings
from .capture import CaptureRect, list_windows, parse_region
from .runtime import RuntimeEvent, RuntimeSettings, WatchRuntime, capture_preview
from .watcher import (
    BACKEND_DIRECTORY,
    CUSTOM_PROFILE_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    PROFILE_DIRECTORY,
    VisibleTableState,
    calibrate_profile,
)


REPOSITORY_ROOT = BACKEND_DIRECTORY.parent
ASSET_DIRECTORY = REPOSITORY_ROOT / "frontend" / "public" / "assets" / "casino-cards"
GUI_SETTINGS_PATH = BACKEND_DIRECTORY / "data" / "screen_history_gui.json"
GUI_TITLE = "Screen Hand History"


class CaptureBorder:
    """Four non-activating windows framing, but not covering, captured pixels."""

    def __init__(self, root: tk.Tk, thickness: int = 4) -> None:
        self.thickness = thickness
        self._visible = False
        self._windows: list[tk.Toplevel] = []
        for _index in range(4):
            window = tk.Toplevel(root)
            window.withdraw()
            window.overrideredirect(True)
            window.configure(background="#ff2020")
            window.attributes("-topmost", True)
            self._windows.append(window)

    def show(self, rectangle: CaptureRect) -> None:
        thickness = self.thickness
        positions = (
            (
                rectangle.left - thickness,
                rectangle.top - thickness,
                rectangle.width + thickness * 2,
                thickness,
            ),
            (
                rectangle.left - thickness,
                rectangle.top + rectangle.height,
                rectangle.width + thickness * 2,
                thickness,
            ),
            (
                rectangle.left - thickness,
                rectangle.top,
                thickness,
                rectangle.height,
            ),
            (
                rectangle.left + rectangle.width,
                rectangle.top,
                thickness,
                rectangle.height,
            ),
        )
        for window, (left, top, width, height) in zip(self._windows, positions):
            # NOTE: build the offset as "+{left}+{top}" (explicit reference-edge
            # "+" followed by a possibly-negative value), NOT "{left:+d}". In Tk
            # geometry a leading "-" means "offset from the far edge", so a
            # monitor placed left of / above the primary (negative left/top, e.g.
            # top=-1081) would land on the wrong screen with "{top:+d}".
            window.geometry(f"{width}x{height}+{left}+{top}")
            if not self._visible:
                window.deiconify()
                window.update_idletasks()
                self._make_click_through(window)
            window.lift()
        self._visible = True

    def hide(self) -> None:
        if not self._visible:
            return
        for window in self._windows:
            window.withdraw()
        self._visible = False

    def destroy(self) -> None:
        for window in self._windows:
            try:
                window.destroy()
            except tk.TclError:
                pass
        self._windows.clear()
        self._visible = False

    @staticmethod
    def _make_click_through(window: tk.Toplevel) -> None:
        if sys.platform != "win32":
            return
        window.update_idletasks()
        user32 = ctypes.windll.user32
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        handle = int(window.winfo_id())
        parent_handle = user32.GetParent(handle)
        parent = int(parent_handle) if parent_handle else 0
        if parent:
            handle = parent
        if hasattr(user32, "GetWindowLongPtrW"):
            get_style = user32.GetWindowLongPtrW
            set_style = user32.SetWindowLongPtrW
            style_type = ctypes.c_ssize_t
        else:
            get_style = user32.GetWindowLongW
            set_style = user32.SetWindowLongW
            style_type = ctypes.c_long
        get_style.argtypes = [wintypes.HWND, ctypes.c_int]
        get_style.restype = style_type
        set_style.argtypes = [wintypes.HWND, ctypes.c_int, style_type]
        set_style.restype = style_type
        extended_style = int(get_style(handle, -20))
        # WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        set_style(handle, -20, extended_style | 0x20 | 0x08000000 | 0x80)
        # Exclude this window from screen capture (WDA_EXCLUDEFROMCAPTURE, 0x11).
        # The watcher captures the whole monitor with Windows Graphics Capture,
        # which composites overlay windows into the frame — so without this the
        # recognizer OCRs its own boxes/labels and, e.g., reads the "hero name"
        # caption as the hero's name, breaking Dealer-Chat row matching. The
        # window stays fully visible to the user; it just isn't captured.
        # Requires Windows 10 2004+ (present on Windows 11).
        try:
            user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
            user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
            user32.SetWindowDisplayAffinity(handle, 0x00000011)
        except (AttributeError, OSError):
            pass


class InspectionOverlay:
    """One click-through, always-on-top window covering the captured area that
    draws a red box + accuracy readout over every region the recognizer read.

    Driven by RuntimeSettings.show_inspection_boxes: the runtime emits inspected
    regions (desktop coordinates) on each "state" event and this overlay paints
    them. A single transparent-keyed window with a Canvas keeps the cost flat no
    matter how many boxes there are, and never steals clicks from the table."""

    # A rare colour used as the see-through key: any pixel painted this exact
    # colour is fully transparent AND click-through, so only the drawn boxes and
    # labels show over the live table.
    _TRANSPARENT = "#0b0b0b"

    def __init__(self, root: tk.Tk) -> None:
        self._window = tk.Toplevel(root)
        self._window.withdraw()
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        self._supported = True
        try:
            self._window.attributes("-transparentcolor", self._TRANSPARENT)
        except tk.TclError:
            # Non-Windows Tk has no transparent-colour key; the overlay would
            # paint an opaque block over the table, so disable it entirely.
            self._supported = False
        self._canvas = tk.Canvas(
            self._window,
            highlightthickness=0,
            borderwidth=0,
            background=self._TRANSPARENT,
        )
        self._canvas.pack(fill="both", expand=True)
        self._visible = False
        # Latest drawn state, kept so a decision (which arrives on its own event,
        # after the frame's regions) can be repainted without a fresh frame.
        self._last_rect: CaptureRect | None = None
        self._regions: tuple[Any, ...] = ()
        self._decision_text: str | None = None
        self._decision_color: str = "#8ef16b"
        # Auto-play badge: what auto-play is doing right now, drawn on the table
        # itself so it is readable while playing without watching the control
        # panel. None hides it entirely.
        self._auto_text: str | None = None
        self._auto_color: str = "#8ef16b"
        # Whether to paint the OCR detection/rank boxes. The recommended-action
        # banner is independent (controlled by whether a decision is set), so the
        # two overlays can be toggled separately.
        self.draw_boxes = True

    @staticmethod
    def _accuracy_color(confidence: float) -> str:
        if confidence >= 0.85:
            return "#8ef16b"
        if confidence >= 0.70:
            return "#f1c75b"
        return "#ff6b6b"

    def show(self, rectangle: CaptureRect, regions: tuple[Any, ...]) -> None:
        if not self._supported:
            return
        self._last_rect = rectangle
        self._regions = regions
        # "+{left}+{top}" (not "{left:+d}") so a monitor above/left of the
        # primary — negative left/top, e.g. top=-1081 — is positioned on its own
        # screen. A leading "-" in Tk geometry means "from the far edge", which
        # put the overlay on the wrong monitor.
        self._window.geometry(
            f"{rectangle.width}x{rectangle.height}"
            f"+{rectangle.left}+{rectangle.top}"
        )
        if not self._visible:
            self._window.deiconify()
            self._window.update_idletasks()
            CaptureBorder._make_click_through(self._window)
            self._visible = True
        self._render()
        self._window.lift()

    def set_decision(self, text: str | None, color: str = "#8ef16b") -> None:
        """Set (or clear with None) the recommended-action banner and repaint."""

        self._decision_text = text
        self._decision_color = color
        if self._supported and self._visible and self._last_rect is not None:
            self._render()
            self._window.lift()

    def set_auto_play(self, text: str | None, color: str = "#8ef16b") -> None:
        """Set (or clear with None) the auto-play status badge and repaint."""

        self._auto_text = text
        self._auto_color = color
        if self._supported and self._visible and self._last_rect is not None:
            self._render()
            self._window.lift()

    def _render(self) -> None:
        rectangle = self._last_rect
        if rectangle is None:
            return
        self._canvas.delete("all")
        hero_centres: list[float] = []
        hero_top: float | None = None
        decision_anchor: tuple[float, float] | None = None
        for label, left, top, right, bottom, confidence in self._regions:
            x0 = left - rectangle.left
            y0 = top - rectangle.top
            x1 = right - rectangle.left
            y1 = bottom - rectangle.top
            # Invisible banner anchor (left felt of the table): never drawn.
            if confidence <= -1.5 and label == "decision anchor":
                decision_anchor = (x0, y0)
                continue
            # Track the hero-card markers even when boxes aren't drawn, so the
            # decision banner can still anchor itself next to the cards.
            if confidence < 0 and label == "hero card":
                hero_centres.append((x0 + x1) / 2.0)
                hero_top = y0 if hero_top is None else min(hero_top, y0)
            if not self.draw_boxes:
                continue
            # A negative confidence marks a fixed search zone (e.g. the
            # hero-card ROI), not a recognized read: draw it in yellow with a
            # plain caption so it reads as a boundary, not a detection.
            if confidence < 0:
                self._canvas.create_rectangle(
                    x0, y0, x1, y1, outline="#ffd21e", width=3
                )
                caption = label
                text_fill = "#ffd21e"
            else:
                self._canvas.create_rectangle(
                    x0, y0, x1, y1, outline="#ff2020", width=2
                )
                caption = f"{label} {confidence:.0%}"
                text_fill = self._accuracy_color(confidence)
            # Prefer the label just above the box; fall back to just below it
            # when the box hugs the top edge of the capture area.
            text_y = y0 - 13 if y0 - 13 >= 0 else y1 + 2
            self._canvas.create_text(
                x0 + 2,
                text_y,
                anchor="nw",
                text=caption,
                fill=text_fill,
                font=("Segoe UI", 8, "bold"),
            )
        if self._decision_text:
            # Recommended-action banner, left-aligned. Preferred position: the
            # left-felt anchor from the watcher; otherwise fall back to above the
            # hero cards, else top-left. First line = chosen action (big); any
            # further lines = the strategy distribution (smaller).
            lines = self._decision_text.split("\n")
            head, subs = lines[0], lines[1:]
            longest = max(len(line) for line in lines)
            banner_w = max(120.0, longest * 7.0)
            banner_h = 24 + 16 * len(subs)
            if decision_anchor is not None:
                bx, by = decision_anchor
            elif hero_centres:
                centre_x = sum(hero_centres) / len(hero_centres)
                bx = centre_x - banner_w / 2.0
                by = (
                    hero_top - 10 - banner_h
                    if hero_top and hero_top > banner_h + 10
                    else 8.0
                )
            else:
                bx, by = 20.0, 20.0
            self._canvas.create_rectangle(
                bx,
                by,
                bx + banner_w,
                by + banner_h,
                fill="#0d1512",
                outline=self._decision_color,
                width=2,
            )
            self._canvas.create_text(
                bx + 8,
                by + 4,
                anchor="nw",
                text=head,
                fill=self._decision_color,
                font=("Segoe UI", 13, "bold"),
            )
            for index, line in enumerate(subs):
                self._canvas.create_text(
                    bx + 8,
                    by + 24 + index * 15,
                    anchor="nw",
                    text=line,
                    fill="#cbd8d1",
                    font=("Segoe UI", 8),
                )
        if self._auto_text:
            # Pinned to the top-left of the captured area rather than to the
            # cards: auto-play's state matters even on frames where no cards or
            # decision were recognized, so it must not move around or vanish.
            lines = self._auto_text.split("\n")
            width = max(150.0, max(len(line) for line in lines) * 6.6)
            height = 8 + 15 * len(lines)
            self._canvas.create_rectangle(
                10, 10, 10 + width, 10 + height,
                fill="#0d1512", outline=self._auto_color, width=2,
            )
            for index, line in enumerate(lines):
                self._canvas.create_text(
                    18,
                    14 + index * 15,
                    anchor="nw",
                    text=line,
                    fill=self._auto_color if index == 0 else "#cbd8d1",
                    font=("Segoe UI", 9, "bold" if index == 0 else "normal"),
                )

    def hide(self) -> None:
        if not self._visible:
            return
        self._window.withdraw()
        self._visible = False

    def destroy(self) -> None:
        try:
            self._window.destroy()
        except tk.TclError:
            pass
        self._visible = False


class ScreenHistoryGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(GUI_TITLE)
        self.root.geometry("1180x790")
        self.root.minsize(980, 680)
        self._events: queue.Queue[RuntimeEvent] = queue.Queue()
        self._runtime: WatchRuntime | None = None
        self._worker: threading.Thread | None = None
        self._preview_image: tk.PhotoImage | None = None
        self._capture_border = CaptureBorder(root)
        self._inspection_overlay = InspectionOverlay(root)
        self._closing = False
        self._capture_count = 0
        self._pending_frames = 0
        self._active_backend = ""
        self._recognition_path = ""
        self._slow_path_warned = False
        self._region_selecting = False
        self._control_widgets: list[tk.Widget] = []

        self.source_mode = tk.StringVar(value="window")
        self.window_title = tk.StringVar(value="Text Hold'em")
        self.monitor = tk.StringVar(value="1")
        self.region = tk.StringVar(value="0,0,1920,1080")
        self.profile = tk.StringVar(value="default")
        self.capture_backend = tk.StringVar(value="auto")
        self.capture_fps = tk.StringVar(value="15")
        self.stability_ms = tk.StringVar(value="300")
        self.small_blind = tk.StringVar(value="10")
        self.big_blind = tk.StringVar(value="20")
        self.max_actions = tk.StringVar(value="4")
        self.brain_decisions = tk.BooleanVar(value=False)
        # Two independent on-screen overlays over the captured table:
        #   show_inspection_boxes -> the red detection / yellow rank boxes (OCR),
        #   show_decision_overlay -> the recommended-action banner near the cards.
        self.show_inspection_boxes = tk.BooleanVar(value=True)
        self.show_decision_overlay = tk.BooleanVar(value=True)
        # Server source returns the full strategy distribution (probabilities);
        # local returns a single sampled action. Distribution needs the server.
        self.decision_source = tk.StringVar(value="server")
        self.minimum_decision_confidence = tk.StringVar(value="85")
        # Desktop input. Off by default, and even when on it stays a dry run
        # until "Live clicking" is ticked as well.
        self.auto_play = tk.BooleanVar(value=False)
        self.auto_play_live = tk.BooleanVar(value=False)
        self.auto_play_confidence = tk.StringVar(value="90")
        self.auto_play_max_per_hand = tk.StringVar(value="12")
        # Randomized think-time before each click, in seconds. "0 0" clicks as
        # soon as the button is verified.
        self.auto_play_delay_min = tk.StringVar(value="0.8")
        self.auto_play_delay_max = tk.StringVar(value="2.4")
        # "input" is a synthesized mouse; "message" posts window messages
        # instead, for clients that ignore injected input.
        self.auto_play_click_method = tk.StringVar(value="input")
        self.output_directory = tk.StringVar(value=str(DEFAULT_OUTPUT_DIRECTORY.resolve()))
        self.status = tk.StringVar(value="Ready")
        self.recognition = tk.StringVar(value="No frame recognized yet.")
        self.decision = tk.StringVar(value="Brain decisions are disabled.")
        self.auto_play_status = tk.StringVar(value="Auto-play is off.")

        self._configure_style()
        self._build_interface()
        self._load_settings()
        self._refresh_profiles()
        self._refresh_windows(quiet=True)
        self._source_changed()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _configure_style(self) -> None:
        self.root.configure(background="#101614")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background="#16201c", foreground="#e7f2ec", fieldbackground="#0e1713")
        style.configure("TFrame", background="#101614")
        style.configure("Panel.TFrame", background="#16201c")
        style.configure("TLabel", background="#16201c", foreground="#dcebe3")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#b8f36b")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground="#b8f36b")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"), foreground="#f1c75b")
        style.configure("TButton", padding=(10, 6))
        style.configure("Primary.TButton", foreground="#101614", background="#b8f36b")
        style.map("Primary.TButton", background=[("active", "#d0ff91"), ("disabled", "#52623f")])
        style.configure("TEntry", fieldbackground="#0e1713", foreground="#ffffff")
        style.configure("TCombobox", fieldbackground="#0e1713", foreground="#ffffff")
        style.configure("TLabelframe", background="#16201c", foreground="#b8f36b")
        style.configure("TLabelframe.Label", background="#16201c", foreground="#b8f36b")
        style.configure("TRadiobutton", background="#16201c", foreground="#e7f2ec")

    def _build_interface(self) -> None:
        shell = ttk.Frame(self.root, padding=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=0, minsize=390)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(1, weight=1)

        ttk.Label(shell, text="Screen Hand History", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        controls = ttk.Frame(shell, style="Panel.TFrame", padding=12)
        controls.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        viewer = ttk.Frame(shell, style="Panel.TFrame", padding=12)
        viewer.grid(row=1, column=1, sticky="nsew")

        self._build_capture_controls(controls)
        self._build_viewer(viewer)

    def _build_capture_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        source = ttk.LabelFrame(parent, text="Capture source", padding=10)
        source.grid(row=0, column=0, sticky="ew")
        source.columnconfigure(1, weight=1)

        modes = (("Window", "window"), ("Monitor", "monitor"), ("Region", "region"))
        for column, (label, value) in enumerate(modes):
            radio = ttk.Radiobutton(
                source,
                text=label,
                value=value,
                variable=self.source_mode,
                command=self._source_changed,
            )
            radio.grid(row=0, column=column, sticky="w", padx=(0, 12), pady=(0, 8))
            self._control_widgets.append(radio)

        ttk.Label(source, text="Window").grid(row=1, column=0, sticky="w", pady=3)
        self.window_combo = ttk.Combobox(source, textvariable=self.window_title)
        self.window_combo.grid(row=1, column=1, sticky="ew", pady=3)
        self.refresh_windows_button = ttk.Button(source, text="Refresh", command=self._refresh_windows)
        self.refresh_windows_button.grid(row=1, column=2, sticky="ew", padx=(6, 0), pady=3)

        ttk.Label(source, text="Monitor").grid(row=2, column=0, sticky="w", pady=3)
        self.monitor_combo = ttk.Combobox(
            source, textvariable=self.monitor, values=("1", "2", "3", "4"), state="readonly"
        )
        self.monitor_combo.grid(row=2, column=1, sticky="ew", pady=3)

        ttk.Label(source, text="Region coordinates").grid(row=3, column=0, sticky="w", pady=3)
        self.region_entry = ttk.Entry(source, textvariable=self.region)
        self.region_entry.grid(row=3, column=1, sticky="ew", pady=3)
        self.select_region_button = ttk.Button(
            source,
            text="Select with mouse",
            command=self._select_region,
        )
        self.select_region_button.grid(row=3, column=2, sticky="ew", padx=(6, 0), pady=3)

        profile_box = ttk.LabelFrame(parent, text="Recognition", padding=10)
        profile_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        profile_box.columnconfigure(1, weight=1)
        ttk.Label(profile_box, text="Profile").grid(row=0, column=0, sticky="w")
        self.profile_combo = ttk.Combobox(
            profile_box, textvariable=self.profile, state="readonly"
        )
        self.profile_combo.grid(row=0, column=1, sticky="ew", padx=(8, 6))
        self.calibrate_button = ttk.Button(
            profile_box, text="Calibrate…", command=self._calibrate_profile
        )
        self.calibrate_button.grid(row=0, column=2, sticky="ew")

        timing = ttk.LabelFrame(parent, text="Watcher settings", padding=10)
        timing.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        timing.columnconfigure(1, weight=1)
        ttk.Label(timing, text="Capture backend").grid(row=0, column=0, sticky="w", pady=3)
        self.backend_combo = ttk.Combobox(
            timing,
            textvariable=self.capture_backend,
            values=("auto", "windows", "mss"),
            state="readonly",
        )
        self.backend_combo.grid(row=0, column=1, sticky="ew", pady=3)
        self._control_widgets.append(self.backend_combo)
        self._entry_row(timing, 1, "Stream FPS", self.capture_fps)
        blinds = ttk.Frame(timing, style="Panel.TFrame")
        blinds.grid(row=2, column=1, sticky="ew", pady=3)
        blinds.columnconfigure(0, weight=1)
        blinds.columnconfigure(1, weight=1)
        small_entry = ttk.Entry(blinds, textvariable=self.small_blind, width=8)
        small_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        big_entry = ttk.Entry(blinds, textvariable=self.big_blind, width=8)
        big_entry.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(timing, text="Blinds (small / big)").grid(row=2, column=0, sticky="w", pady=3)
        self._control_widgets.extend((small_entry, big_entry))
        self._entry_row(timing, 3, "Stability delay (ms)", self.stability_ms)
        self._entry_row(timing, 4, "Max inferred actions", self.max_actions)
        brain_toggle = ttk.Checkbutton(
            timing,
            text="Enable brain decisions",
            variable=self.brain_decisions,
        )
        brain_toggle.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 3))
        self._control_widgets.append(brain_toggle)
        decision_overlay_toggle = ttk.Checkbutton(
            timing,
            text="Decision overlay (recommended action on table)",
            variable=self.show_decision_overlay,
        )
        decision_overlay_toggle.grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 3))
        self._control_widgets.append(decision_overlay_toggle)
        boxes_overlay_toggle = ttk.Checkbutton(
            timing,
            text="OCR boxes overlay (card / read boxes on table)",
            variable=self.show_inspection_boxes,
        )
        boxes_overlay_toggle.grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 3))
        self._control_widgets.append(boxes_overlay_toggle)
        self._entry_row(
            timing,
            8,
            "Decision confidence (%)",
            self.minimum_decision_confidence,
        )

        auto = ttk.LabelFrame(parent, text="Auto-play (clicks the poker client)", padding=10)
        auto.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        auto.columnconfigure(1, weight=1)
        auto_toggle = ttk.Checkbutton(
            auto,
            text="Enable auto-play for accepted decisions",
            variable=self.auto_play,
            command=self._auto_play_changed,
        )
        auto_toggle.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 3))
        self.auto_play_live_toggle = ttk.Checkbutton(
            auto,
            text="Live clicking (off = dry run, logs the click only)",
            variable=self.auto_play_live,
            command=self._auto_play_changed,
        )
        self.auto_play_live_toggle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 3))
        self._entry_row(auto, 2, "Auto-play confidence (%)", self.auto_play_confidence)
        self._entry_row(auto, 3, "Max clicks per hand", self.auto_play_max_per_hand)
        ttk.Label(auto, text="Think time (min / max s)").grid(
            row=7, column=0, sticky="w", pady=3
        )
        think = ttk.Frame(auto, style="Panel.TFrame")
        think.grid(row=7, column=1, sticky="ew", pady=3)
        think.columnconfigure(0, weight=1)
        think.columnconfigure(1, weight=1)
        delay_min = ttk.Entry(think, textvariable=self.auto_play_delay_min, width=8)
        delay_min.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        delay_max = ttk.Entry(think, textvariable=self.auto_play_delay_max, width=8)
        delay_max.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._control_widgets.extend((delay_min, delay_max))
        ttk.Label(auto, text="Click delivery").grid(row=4, column=0, sticky="w", pady=3)
        self.click_method_combo = ttk.Combobox(
            auto,
            textvariable=self.auto_play_click_method,
            values=("input", "message"),
            state="readonly",
        )
        self.click_method_combo.grid(row=4, column=1, sticky="ew", pady=3)
        self._control_widgets.append(self.click_method_combo)
        ttk.Label(auto, textvariable=self.auto_play_status, wraplength=330).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )
        ttk.Label(
            auto,
            text="Press F12 at any time to stop auto-play.",
            wraplength=330,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._control_widgets.extend((auto_toggle, self.auto_play_live_toggle))

        output = ttk.LabelFrame(parent, text="Output", padding=10)
        output.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        output.columnconfigure(0, weight=1)
        output_entry = ttk.Entry(output, textvariable=self.output_directory)
        output_entry.grid(row=0, column=0, sticky="ew")
        browse_button = ttk.Button(output, text="Browse…", command=self._browse_output)
        browse_button.grid(row=0, column=1, padx=(6, 0))
        open_button = ttk.Button(output, text="Open", command=self._open_output)
        open_button.grid(row=0, column=2, padx=(6, 0))
        self._control_widgets.extend((output_entry, browse_button))

        actions = ttk.Frame(parent, style="Panel.TFrame")
        actions.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        self.test_button = ttk.Button(actions, text="Test Capture", command=self._test_capture)
        self.test_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.start_button = ttk.Button(
            actions, text="Start", style="Primary.TButton", command=self._start
        )
        self.start_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

        reloads = ttk.Frame(parent, style="Panel.TFrame")
        reloads.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        reloads.columnconfigure(0, weight=1)
        reloads.columnconfigure(1, weight=1)
        # Reload the champion model on the decision server (after promoting a new
        # champion) — no server restart needed.
        self.reload_server_button = ttk.Button(
            reloads, text="Reload server model", command=self._reload_server
        )
        self.reload_server_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        # Restart this app so edited recognition/overlay code takes effect.
        self.reload_watcher_button = ttk.Button(
            reloads, text="Reload watcher (restart)", command=self._reload_watcher
        )
        self.reload_watcher_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self._control_widgets.extend(
            (
                self.window_combo,
                self.refresh_windows_button,
                self.monitor_combo,
                self.region_entry,
                self.select_region_button,
                self.profile_combo,
                self.calibrate_button,
                self.test_button,
            )
        )

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        self._control_widgets.append(entry)

    def _build_viewer(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        # minsize pins a floor for the preview so it never collapses; the fixed
        # recognized-table box below keeps its own height constant.
        parent.rowconfigure(1, weight=3, minsize=320)
        parent.rowconfigure(5, weight=2)
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Status", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status, style="Status.TLabel").grid(
            row=0, column=1, sticky="e"
        )

        self.preview_label = tk.Label(
            parent,
            text="Use Test Capture to verify the selected source.",
            background="#090e0c",
            foreground="#98aaa1",
            anchor="center",
            relief="sunken",
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(10, 8))

        ttk.Label(parent, text="Recognized table", style="Section.TLabel").grid(
            row=2, column=0, sticky="w", pady=(2, 4)
        )
        # Fixed-height holder: recognized-table text changes length every frame
        # (warnings appear/vanish), and without a fixed box that reflow squeezed
        # the weighted preview row, making the image jump size. grid_propagate
        # off pins the height so the preview stays constant.
        recognition_holder = tk.Frame(parent, background="#16201c", height=104)
        recognition_holder.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        recognition_holder.grid_propagate(False)
        recognition_holder.columnconfigure(0, weight=1)
        recognition_holder.rowconfigure(0, weight=1)
        ttk.Label(
            recognition_holder,
            textvariable=self.recognition,
            justify="left",
            anchor="nw",
            wraplength=700,
        ).grid(row=0, column=0, sticky="nsew")

        decision_frame = ttk.LabelFrame(parent, text="Brain recommendation", padding=8)
        decision_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            decision_frame,
            textvariable=self.decision,
            justify="left",
            anchor="w",
            wraplength=700,
        ).grid(row=0, column=0, sticky="ew")
        decision_frame.columnconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(parent, text="Activity log", padding=6)
        log_frame.grid(row=5, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            height=8,
            state="disabled",
            background="#090e0c",
            foreground="#cbd8d1",
            insertbackground="#ffffff",
            relief="flat",
            wrap="word",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def _settings(self) -> RuntimeSettings:
        source = self.source_mode.get()
        region = parse_region(self.region.get()) if source == "region" else None
        capture_fps = float(self.capture_fps.get())
        if capture_fps <= 0:
            raise ValueError("Stream FPS must be greater than zero.")
        settings = RuntimeSettings(
            asset_directory=ASSET_DIRECTORY,
            output_directory=Path(self.output_directory.get()).expanduser().resolve(),
            profile=self.profile.get() or "default",
            interval=1.0 / capture_fps,
            capture_fps=capture_fps,
            stability_seconds=float(self.stability_ms.get()) / 1000.0,
            capture_backend=self.capture_backend.get(),
            blinds=(int(self.small_blind.get()), int(self.big_blind.get())),
            maximum_transition_actions=int(self.max_actions.get()),
            brain_decisions=bool(self.brain_decisions.get()),
            minimum_decision_confidence=(
                float(self.minimum_decision_confidence.get()) / 100.0
            ),
            decision_source=self.decision_source.get() or "server",
            window_title=self.window_title.get().strip() if source == "window" else None,
            monitor=int(self.monitor.get()) if source == "monitor" else None,
            region=region,
            # Collect/emit inspection regions when EITHER overlay is on: the OCR
            # boxes need them to draw, and the decision banner uses the hero-card
            # markers to anchor itself next to the cards.
            show_inspection_boxes=bool(
                self.show_inspection_boxes.get() or self.show_decision_overlay.get()
            ),
            auto_play=AutoPlaySettings(
                enabled=bool(self.auto_play.get()),
                dry_run=not bool(self.auto_play_live.get()),
                minimum_confidence=float(self.auto_play_confidence.get()) / 100.0,
                maximum_clicks_per_hand=int(self.auto_play_max_per_hand.get()),
                minimum_delay_seconds=float(self.auto_play_delay_min.get()),
                maximum_delay_seconds=float(self.auto_play_delay_max.get()),
                click_method=self.auto_play_click_method.get() or "input",
            ),
        )
        settings.validate()
        return settings

    # Green = it acted, amber = it stood down for a reason, red = it stopped.
    _AUTO_PLAY_COLORS = {
        "clicked": "#8ef16b",
        "confirmed": "#8ef16b",
        "dry_run": "#7fd1ff",
        "skipped": "#f1c75b",
        "aborted": "#f1c75b",
        "inconclusive": "#f1c75b",
        "unconfirmed": "#ff9f43",
        "disabled": "#ff6b6b",
    }

    def _auto_play_badge(self, heading: str, message: str, color: str) -> None:
        """Show auto-play's state on the table overlay, wrapped to fit."""

        if not self.show_decision_overlay.get():
            return
        words = str(message).split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 46 and current:
                lines.append(current)
                current = word
            else:
                current = candidate
            if len(lines) == 3:
                break
        if current and len(lines) < 3:
            lines.append(current)
        self._inspection_overlay.set_auto_play(
            "\n".join([f"AUTO-PLAY: {heading}", *lines]), color
        )

    def _auto_play_changed(self) -> None:
        if not self.auto_play.get():
            self.auto_play_live.set(False)
            self.auto_play_status.set("Auto-play is off.")
            return
        if not self.auto_play_live.get():
            self.auto_play_status.set(
                "Dry run: the resolved button and click point are logged, "
                "nothing is pressed."
            )
            return
        confirmed = messagebox.askokcancel(
            "Enable live clicking?",
            "Auto-play will press Fold / Check / Call / Bet in the poker client "
            "for accepted decisions.\n\n"
            "Every click is verified against the button's own text first, and F12 "
            "stops it immediately.\n\n"
            "Automating a real-money client is against most poker sites' terms of "
            "service and can cost the account.",
            icon="warning",
        )
        if not confirmed:
            self.auto_play_live.set(False)
            self.auto_play_status.set("Live clicking cancelled; still a dry run.")
            return
        self.auto_play_status.set("LIVE: accepted decisions will be clicked.")

    def _source_changed(self) -> None:
        if self._runtime is not None and self._runtime.running:
            return
        mode = self.source_mode.get()
        self.window_combo.configure(state="normal" if mode == "window" else "disabled")
        self.refresh_windows_button.configure(state="normal" if mode == "window" else "disabled")
        self.monitor_combo.configure(state="readonly" if mode == "monitor" else "disabled")
        self.region_entry.configure(state="normal" if mode == "region" else "disabled")
        # Mouse selection is always available while stopped. Finishing a drag
        # automatically chooses Region as the active capture source.
        self.select_region_button.configure(state="normal")

    def _refresh_windows(self, quiet: bool = False) -> None:
        try:
            titles = [
                window.title
                for window in list_windows()
                if not window.minimized and window.title != GUI_TITLE
            ]
            self.window_combo.configure(values=titles)
            if not self.window_title.get() and titles:
                self.window_title.set(titles[0])
            if not quiet:
                self._append_log(f"Found {len(titles)} visible windows.")
        except RuntimeError as exc:
            if not quiet:
                messagebox.showerror(GUI_TITLE, str(exc), parent=self.root)

    def _refresh_profiles(self) -> None:
        profiles = [
            path.stem for path in sorted(PROFILE_DIRECTORY.glob("*.json"))
        ]
        if CUSTOM_PROFILE_DIRECTORY.is_dir():
            profiles.extend(
                path.stem for path in sorted(CUSTOM_PROFILE_DIRECTORY.glob("*.json"))
            )
        self.profile_combo.configure(values=tuple(dict.fromkeys(profiles)))
        if self.profile.get() not in profiles:
            self.profile.set("default")

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(
            title="Choose hand-history output directory",
            initialdir=self.output_directory.get(),
            parent=self.root,
        )
        if selected:
            self.output_directory.set(selected)

    def _open_output(self) -> None:
        try:
            destination = Path(self.output_directory.get()).expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            os.startfile(destination)  # type: ignore[attr-defined]
        except (OSError, ValueError) as exc:
            messagebox.showerror(GUI_TITLE, f"Could not open output directory:\n{exc}", parent=self.root)

    def _reload_server(self) -> None:
        """Tell the decision server to re-read the promoted champion from disk."""

        base = "http://127.0.0.1:8000".rstrip("/")
        url = f"{base}/api/champion/reload"
        self.status.set("Reloading server model…")
        self.reload_server_button.configure(state="disabled")

        def work() -> None:
            try:
                http_request = urllib.request.Request(
                    url,
                    data=b"",
                    headers={"Accept": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(http_request, timeout=15.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                message = (
                    f"Server model reloaded: iteration "
                    f"{payload.get('iteration', '?')} ({payload.get('source', '?')})"
                )
            except urllib.error.HTTPError as exc:
                try:
                    detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    detail = str(exc)
                message = f"Server reload failed: {detail}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                message = f"Server reload failed: {reason}"

            def finish() -> None:
                self._append_log(message)
                self.status.set(message[:70])
                self.reload_server_button.configure(state="normal")

            self.root.after(0, finish)

        threading.Thread(target=work, name="server-reload", daemon=True).start()

    def _reload_watcher(self) -> None:
        """Restart this app so edited recognition/overlay code takes effect."""

        if not messagebox.askyesno(
            GUI_TITLE,
            "Restart the watcher app now? It reopens with your saved settings, "
            "picking up any code changes.",
            parent=self.root,
        ):
            return
        try:
            self._save_settings()
        except OSError:
            pass
        self.status.set("Restarting…")
        # Release the capture before the fresh process starts one.
        if self._runtime is not None and self._runtime.running:
            self._runtime.stop()
            if self._worker is not None:
                self._worker.join(timeout=8.0)
        try:
            self._capture_border.destroy()
            self._inspection_overlay.destroy()
        except tk.TclError:
            pass
        script = os.path.abspath(sys.argv[0])
        subprocess.Popen([sys.executable, script, *sys.argv[1:]])
        os._exit(0)

    def _select_region(self) -> None:
        if self._region_selecting:
            return
        if sys.platform != "win32":
            messagebox.showerror(
                GUI_TITLE,
                "Mouse region selection currently requires Windows.",
                parent=self.root,
            )
            return

        self._region_selecting = True
        self.select_region_button.configure(state="disabled")
        self.status.set(
            "Select region: after this window minimizes, drag anywhere on the desktop"
        )
        self.root.update_idletasks()
        self.root.iconify()

        def select_from_global_mouse() -> None:
            try:
                user32 = ctypes.windll.user32
                user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
                user32.GetAsyncKeyState.restype = ctypes.c_short
                user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
                user32.GetCursorPos.restype = wintypes.BOOL

                def pressed(virtual_key: int) -> bool:
                    return bool(user32.GetAsyncKeyState(virtual_key) & 0x8000)

                def cursor_position() -> tuple[int, int]:
                    point = wintypes.POINT()
                    if not user32.GetCursorPos(ctypes.byref(point)):
                        raise RuntimeError("Windows could not read the mouse position.")
                    return int(point.x), int(point.y)

                deadline = time.monotonic() + 60.0
                # Ignore the button click that launched selection.
                while pressed(0x01) and time.monotonic() < deadline:
                    time.sleep(0.01)

                while time.monotonic() < deadline:
                    if pressed(0x1B) or pressed(0x02):  # Esc or right mouse button
                        self._events.put(
                            RuntimeEvent("region_cancelled", message="Region selection cancelled.")
                        )
                        return
                    if not pressed(0x01):
                        time.sleep(0.01)
                        continue

                    start_x, start_y = cursor_position()
                    end_x, end_y = start_x, start_y
                    while pressed(0x01) and time.monotonic() < deadline:
                        end_x, end_y = cursor_position()
                        if pressed(0x1B) or pressed(0x02):
                            self._events.put(
                                RuntimeEvent(
                                    "region_cancelled",
                                    message="Region selection cancelled.",
                                )
                            )
                            return
                        time.sleep(0.01)

                    left, right = sorted((start_x, end_x))
                    top, bottom = sorted((start_y, end_y))
                    width = right - left
                    height = bottom - top
                    if width < 20 or height < 20:
                        self._events.put(
                            RuntimeEvent(
                                "region_cancelled",
                                message="Selection was too small; drag a larger rectangle.",
                            )
                        )
                        return
                    self._events.put(
                        RuntimeEvent(
                            "region_selected",
                            message=f"Selected region {left},{top},{width},{height}.",
                            rect=CaptureRect(left, top, width, height),
                        )
                    )
                    return

                self._events.put(
                    RuntimeEvent(
                        "region_cancelled",
                        message="Region selection timed out after one minute.",
                    )
                )
            except Exception as exc:
                self._events.put(
                    RuntimeEvent(
                        "region_cancelled",
                        message=f"Mouse region selection failed: {exc}",
                    )
                )

        threading.Thread(
            target=select_from_global_mouse,
            name="screen-history-region-selection",
            daemon=True,
        ).start()

    @staticmethod
    def _virtual_screen_bounds() -> tuple[int, int, int, int]:
        user32 = ctypes.windll.user32
        return (
            int(user32.GetSystemMetrics(76)),
            int(user32.GetSystemMetrics(77)),
            int(user32.GetSystemMetrics(78)),
            int(user32.GetSystemMetrics(79)),
        )

    def _calibrate_profile(self) -> None:
        screenshot = filedialog.askopenfilename(
            title="Choose a simulator screenshot",
            filetypes=(("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")),
            parent=self.root,
        )
        if not screenshot:
            return
        requested = simpledialog.askstring(
            GUI_TITLE,
            "Profile name:",
            initialvalue="my-simulator",
            parent=self.root,
        )
        if not requested:
            return
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", requested.strip()).strip("-")
        if not name:
            messagebox.showerror(GUI_TITLE, "Profile name must contain letters or numbers.", parent=self.root)
            return
        destination = CUSTOM_PROFILE_DIRECTORY / f"{name}.json"
        self.status.set("Calibrating profile…")

        def work() -> None:
            try:
                profile = calibrate_profile(Path(screenshot), destination, name)
                self._events.put(
                    RuntimeEvent("calibrated", message=f"Profile '{profile.name}' saved.")
                )
            except (RuntimeError, ValueError, OSError) as exc:
                self._events.put(RuntimeEvent("error", message=f"Calibration failed: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _test_capture(self) -> None:
        try:
            settings = self._settings()
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            messagebox.showerror(GUI_TITLE, str(exc), parent=self.root)
            return
        self.status.set("Capturing and recognizing…")
        self.test_button.configure(state="disabled")

        def work() -> None:
            try:
                frame, state = capture_preview(settings)
                self._events.put(RuntimeEvent("preview", state=state, frame=frame))
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
                self._events.put(RuntimeEvent("error", message=f"Test capture failed: {exc}"))
            finally:
                self._events.put(RuntimeEvent("test_finished"))

        threading.Thread(target=work, daemon=True).start()

    def _start(self) -> None:
        try:
            settings = self._settings()
            self._save_settings()
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            messagebox.showerror(GUI_TITLE, str(exc), parent=self.root)
            return
        self._runtime = WatchRuntime(settings, self._events.put)
        self._worker = threading.Thread(target=self._runtime.run, daemon=True)
        self._set_running(True)
        self.status.set("Starting…")
        self._worker.start()

    def _stop(self) -> None:
        if self._runtime is not None:
            queue_status = (
                f" • processing {self._pending_frames} queued frame(s)"
                if self._pending_frames
                else ""
            )
            self.status.set(f"Stopping…{queue_status}")
            self._capture_border.hide()
            self._inspection_overlay.hide()
            self._runtime.stop()
            self.stop_button.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        for widget in self._control_widgets:
            try:
                widget.configure(state="disabled" if running else "normal")
            except tk.TclError:
                pass
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        if not running:
            self.profile_combo.configure(state="readonly")
            self.backend_combo.configure(state="readonly")
            self._source_changed()

    def _poll_events(self) -> None:
        try:
            while True:
                self._handle_event(self._events.get_nowait())
        except queue.Empty:
            pass
        if self._closing:
            if self._worker is None or not self._worker.is_alive():
                self._capture_border.destroy()
                self._inspection_overlay.destroy()
                self.root.destroy()
                return
        self.root.after(100, self._poll_events)

    def _handle_event(self, event: RuntimeEvent) -> None:
        if event.kind == "started":
            self._capture_count = 0
            self._pending_frames = 0
            self._active_backend = event.backend
            self._recognition_path = ""
            self._slow_path_warned = False
            self.status.set(f"Running • {self._active_backend}")
            self.decision.set(
                "Waiting for a validated Hero turn…"
                if self.brain_decisions.get()
                else "Brain decisions are disabled."
            )
            self._inspection_overlay.set_auto_play(
                None
                if not self.auto_play.get()
                else "AUTO-PLAY: STARTING\nwaiting for a validated Hero turn",
                "#7fd1ff",
            )
            self._append_log(event.message)
        elif event.kind == "fallback":
            self._append_log(event.message)
        elif event.kind == "capture":
            self._capture_count = event.capture_count
            self._pending_frames = event.pending_frames
            self._active_backend = event.backend or self._active_backend
            queue_status = (
                f" • {self._pending_frames} awaiting OCR"
                if self._pending_frames
                else ""
            )
            path_status = ""
            if self._recognition_path == "fast":
                path_status = " • FAST"
            elif self._recognition_path.startswith("slow"):
                path_status = " • SLOW (full-frame OCR)"
            self.status.set(
                f"Running • {self._active_backend} • {event.stream_fps:.0f} FPS • "
                f"{self._capture_count} frames{path_status}{queue_status}"
            )
            if event.rect is not None:
                self._capture_border.show(event.rect)
        elif event.kind in {"state", "preview"}:
            self._pending_frames = event.pending_frames
            if event.kind == "state":
                if event.recognition_path:
                    self._recognition_path = event.recognition_path
                    if event.recognition_path.startswith("slow"):
                        if not self._slow_path_warned:
                            self._slow_path_warned = True
                            self._append_log(
                                "SLOW path — " + event.recognition_path + ". "
                                "For fast recognition, put the CoinPoker TABLE and "
                                "DEALER CHAT windows on the captured monitor."
                            )
                    elif event.recognition_path == "fast":
                        self._slow_path_warned = False
                # The overlay window is shown when EITHER overlay is enabled; the
                # OCR-boxes checkbox only controls whether the boxes are painted,
                # while the decision banner is driven by the brain events below.
                overlays_on = bool(
                    self.show_inspection_boxes.get() or self.show_decision_overlay.get()
                )
                if overlays_on and event.rect is not None:
                    self._inspection_overlay.draw_boxes = bool(
                        self.show_inspection_boxes.get()
                    )
                    self._inspection_overlay.show(event.rect, event.regions)
                else:
                    self._inspection_overlay.hide()
                # Drop the recommendation the moment it is no longer Hero's turn.
                if event.state is not None and (
                    event.state.current_player != 0 or event.state.complete
                ):
                    self._inspection_overlay.set_decision(None)
            if event.frame is not None:
                self._show_frame(event.frame)
            if event.state is not None:
                self.recognition.set(self._format_state(event.state, event.transition))
                label = "Preview recognized." if event.kind == "preview" else self._state_log(event.state, event.transition)
                self._append_log(label)
            if event.kind == "preview":
                self.status.set("Test capture complete")
        elif event.kind in {"region_selected", "region_cancelled"}:
            self._region_selecting = False
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.select_region_button.configure(state="normal")
            if event.kind == "region_selected" and event.rect is not None:
                rectangle = event.rect
                self.region.set(
                    f"{rectangle.left},{rectangle.top},"
                    f"{rectangle.width},{rectangle.height}"
                )
                self.source_mode.set("region")
                self._source_changed()
                self.status.set("Region selected")
                self._capture_border.show(rectangle)
                self.root.after(1_500, self._capture_border.hide)
            else:
                self.status.set("Ready")
            self._append_log(event.message)
        elif event.kind == "brain_ready":
            self.decision.set(f"{event.message} Waiting for a validated Hero turn…")
            self._append_log(event.message)
        elif event.kind == "brain_thinking":
            self.decision.set("Brain thinking…")
            if self.show_decision_overlay.get():
                self._inspection_overlay.set_decision("Thinking…", "#f1c75b")
            self._append_log(event.message)
        elif event.kind == "brain_decision" and event.decision is not None:
            decision = event.decision
            action = decision.action.replace("_", " ").title()
            amount = f" {decision.amount:,}" if decision.amount is not None else ""
            depth = (
                f" • {decision.selected_depth_bb:g} BB blueprint"
                if decision.selected_depth_bb is not None
                else ""
            )
            self.decision.set(
                f"{action}{amount} • {decision.model}{depth} • "
                f"input confidence {decision.recognition_confidence:.0%}"
            )
            if self.show_decision_overlay.get():
                self._inspection_overlay.set_decision(
                    self._decision_banner_text(decision), "#8ef16b"
                )
            self._append_log(event.message)
        elif event.kind == "auto_play_ready":
            self.auto_play_status.set(event.message)
            self._auto_play_badge("READY", event.message, "#8ef16b")
            self._append_log(event.message)
        elif event.kind == "auto_play":
            status = event.auto_play.status if event.auto_play is not None else ""
            self.auto_play_status.set(event.message)
            self._auto_play_badge(
                status.replace("_", " ").upper() or "AUTO-PLAY",
                event.message,
                self._AUTO_PLAY_COLORS.get(status, "#cbd8d1"),
            )
            # A refusal is the normal, safe outcome; only surface the ones that
            # change what auto-play will do next.
            if status in {"clicked", "dry_run", "aborted", "unconfirmed", "disabled"}:
                self._append_log(f"Auto-play [{status}]: {event.message}")
        elif event.kind == "brain_skipped":
            self.decision.set(f"No decision: {event.message}")
            self._inspection_overlay.set_decision(None)
        elif event.kind == "brain_stale":
            self.decision.set(event.message)
            self._inspection_overlay.set_decision(None)
            self._append_log(event.message)
        elif event.kind == "brain_error":
            self.decision.set(f"Brain error: {event.message}")
            self._inspection_overlay.set_decision(None)
            self._append_log(f"Brain error: {event.message}")
        elif event.kind == "recognition_error":
            self.status.set("Running with recognition errors")
            self._append_log(f"Recognition error: {event.message}")
        elif event.kind == "error":
            self._capture_border.hide()
            self._inspection_overlay.hide()
            self.status.set("Error")
            self._append_log(event.message)
            if not self._closing:
                messagebox.showerror(GUI_TITLE, event.message, parent=self.root)
        elif event.kind == "finalized":
            self._append_log(event.message)
        elif event.kind == "calibrated":
            self.status.set("Profile ready")
            self._refresh_profiles()
            match = re.search(r"Profile '([^']+)'", event.message)
            if match:
                self.profile.set(match.group(1))
            self._append_log(event.message)
        elif event.kind == "test_finished":
            if self._runtime is None or not self._runtime.running:
                self.test_button.configure(state="normal")
        elif event.kind == "stopped":
            self._capture_border.hide()
            self._inspection_overlay.set_auto_play(None)
            self._inspection_overlay.hide()
            self._pending_frames = 0
            self.status.set(f"Stopped • {self._capture_count} frames")
            self._append_log(event.message)
            self._set_running(False)
            self._runtime = None
            self._worker = None

    def _show_frame(self, frame: Any) -> None:
        try:
            import cv2  # type: ignore

            available_width = max(500, self.preview_label.winfo_width() - 8)
            available_height = max(280, self.preview_label.winfo_height() - 8)
            height, width = frame.shape[:2]
            scale = min(available_width / width, available_height / height, 1.0)
            resized = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            success, encoded = cv2.imencode(".png", resized)
            if not success:
                raise RuntimeError("Could not encode preview image.")
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            self._preview_image = tk.PhotoImage(data=payload)
            self.preview_label.configure(image=self._preview_image, text="")
        except (RuntimeError, tk.TclError) as exc:
            self._append_log(f"Preview unavailable: {exc}")

    def _decision_banner_text(self, decision: Any) -> str:
        """Two lines for the on-screen banner: the chosen action (with amount),
        then the full strategy distribution when the server provided one."""

        scale = 100 if (self.profile.get() or "").lower() == "coinpoker" else 1

        def amount(value: Any) -> str:
            if value is None:
                return ""
            return f" {value / scale:.2f}" if scale != 1 else f" {value:,}"

        head = decision.action.replace("_", " ").title() + amount(decision.amount)
        # Compact distribution line: single-letter actions, probability only.
        abbrev = {
            "fold": "F",
            "check": "X",
            "call": "C",
            "raise": "R",
            "bet": "B",
            "all_in": "AI",
        }
        parts: list[str] = []
        for option in sorted(
            decision.strategy or (),
            key=lambda item: item.get("probability") or 0.0,
            reverse=True,
        ):
            probability = option.get("probability")
            if probability is None:
                continue
            code = abbrev.get(str(option.get("action", "")).lower(), "?")
            parts.append(f"{code} {float(probability):.0%}")
        return head + ("\n" + "  ".join(parts[:4]) if parts else "")

    @staticmethod
    def _format_state(state: VisibleTableState, transition: Any = None) -> str:
        values = (
            f"Hand: #{state.hand_number or '?'}    Street: {state.street or '?'}    "
            f"Pot: {state.pot if state.pot is not None else '?'}\n"
            f"Hero stack: {state.stacks[0] if state.stacks[0] is not None else '?'}    "
            f"Opponent stack: {state.stacks[1] if state.stacks[1] is not None else '?'}\n"
            f"Hero cards: {' '.join(state.hero_cards) or '?'}    "
            f"Board: {' '.join(state.board) or '-'}\n"
            f"Button: {'Hero' if state.button == 0 else 'Opponent' if state.button == 1 else '?'}    "
            f"Stable: {'yes' if state.stable else 'no'}    Confidence: {state.confidence:.0%}"
        )
        if transition is not None:
            values += f"    Transition: {transition.status}"
        if state.warnings:
            values += "\nWarnings: " + " | ".join(state.warnings)
        return values

    @staticmethod
    def _state_log(state: VisibleTableState, transition: Any) -> str:
        transition_name = transition.status if transition is not None else "recognized"
        return (
            f"Hand #{state.hand_number or '?'} • {state.street or '?'} • "
            f"pot {state.pot if state.pot is not None else '?'} • {transition_name}"
        )

    def _append_log(self, message: str) -> None:
        if not message:
            return
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _save_settings(self) -> None:
        payload = {
            "source_mode": self.source_mode.get(),
            "window_title": self.window_title.get(),
            "monitor": self.monitor.get(),
            "region": self.region.get(),
            "profile": self.profile.get(),
            "capture_backend": self.capture_backend.get(),
            "capture_fps": self.capture_fps.get(),
            "stability_ms": self.stability_ms.get(),
            "small_blind": self.small_blind.get(),
            "big_blind": self.big_blind.get(),
            "max_actions": self.max_actions.get(),
            "brain_decisions": bool(self.brain_decisions.get()),
            "show_inspection_boxes": bool(self.show_inspection_boxes.get()),
            "show_decision_overlay": bool(self.show_decision_overlay.get()),
            "decision_source": self.decision_source.get(),
            "minimum_decision_confidence": self.minimum_decision_confidence.get(),
            "auto_play": bool(self.auto_play.get()),
            "auto_play_confidence": self.auto_play_confidence.get(),
            "auto_play_max_per_hand": self.auto_play_max_per_hand.get(),
            "auto_play_click_method": self.auto_play_click_method.get(),
            "auto_play_delay_min": self.auto_play_delay_min.get(),
            "auto_play_delay_max": self.auto_play_delay_max.get(),
            "output_directory": self.output_directory.get(),
        }
        GUI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        GUI_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_settings(self) -> None:
        if not GUI_SETTINGS_PATH.is_file():
            return
        try:
            payload = json.loads(GUI_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        variables: dict[str, tk.StringVar] = {
            "source_mode": self.source_mode,
            "window_title": self.window_title,
            "monitor": self.monitor,
            "region": self.region,
            "profile": self.profile,
            "capture_backend": self.capture_backend,
            "capture_fps": self.capture_fps,
            "stability_ms": self.stability_ms,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "max_actions": self.max_actions,
            "decision_source": self.decision_source,
            "minimum_decision_confidence": self.minimum_decision_confidence,
            "auto_play_confidence": self.auto_play_confidence,
            "auto_play_max_per_hand": self.auto_play_max_per_hand,
            "auto_play_click_method": self.auto_play_click_method,
            "auto_play_delay_min": self.auto_play_delay_min,
            "auto_play_delay_max": self.auto_play_delay_max,
            "output_directory": self.output_directory,
        }
        for name, variable in variables.items():
            value = payload.get(name)
            if value is not None:
                variable.set(str(value))
        if payload.get("brain_decisions") is not None:
            self.brain_decisions.set(bool(payload["brain_decisions"]))
        # "auto_play_live" is deliberately never restored: live clicking has to
        # be armed by hand, with its confirmation, in every session.
        if payload.get("auto_play") is not None:
            self.auto_play.set(bool(payload["auto_play"]))
            self._auto_play_changed()
        if payload.get("show_inspection_boxes") is not None:
            self.show_inspection_boxes.set(bool(payload["show_inspection_boxes"]))
        if payload.get("show_decision_overlay") is not None:
            self.show_decision_overlay.set(bool(payload["show_decision_overlay"]))
        if "capture_fps" not in payload and payload.get("interval") is not None:
            try:
                legacy_interval = float(payload["interval"])
                if legacy_interval > 0:
                    self.capture_fps.set(
                        str(min(60.0, max(15.0, 1.0 / legacy_interval)))
                    )
            except (TypeError, ValueError):
                pass

    def _on_close(self) -> None:
        try:
            self._save_settings()
        except OSError:
            pass
        if self._runtime is not None and self._runtime.running:
            self._closing = True
            self.status.set("Stopping before exit…")
            self._capture_border.hide()
            self._inspection_overlay.hide()
            self._runtime.stop()
            return
        self._capture_border.destroy()
        self._inspection_overlay.destroy()
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    ScreenHistoryGui(root)
    root.mainloop()
