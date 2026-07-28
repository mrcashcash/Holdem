"""Window-independent screen capture sources for the poker watcher."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptureRect:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Capture width and height must be positive.")

    def as_mss(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    minimized: bool


def parse_region(value: str) -> CaptureRect:
    try:
        left, top, width, height = (int(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise ValueError("Region must be left,top,width,height.") from exc
    return CaptureRect(left, top, width, height)


def _windows_user32():
    if sys.platform != "win32":
        raise RuntimeError("Window-title capture is currently available only on Windows.")
    user32 = ctypes.windll.user32
    try:
        # Per-monitor V2 awareness keeps physical capture pixels aligned with
        # GetClientRect coordinates on mixed-DPI desktops.
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()
    return user32


def list_windows() -> list[WindowInfo]:
    user32 = _windows_user32()
    windows: list[WindowInfo] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def collect(handle, _parameter):
        if not user32.IsWindowVisible(handle):
            return True
        length = user32.GetWindowTextLengthW(handle)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        title = buffer.value.strip()
        if title:
            windows.append(
                WindowInfo(
                    handle=int(handle),
                    title=title,
                    minimized=bool(user32.IsIconic(handle)),
                )
            )
        return True

    user32.EnumWindows(collect, 0)
    return sorted(windows, key=lambda window: window.title.lower())


def find_window(title_fragment: str) -> WindowInfo:
    cleaned = title_fragment.strip().lower()
    if not cleaned:
        raise ValueError("Window title must not be empty.")
    matches = [window for window in list_windows() if cleaned in window.title.lower()]
    if not matches:
        raise RuntimeError(f"No visible window contains title: {title_fragment}")
    exact = [window for window in matches if window.title.lower() == cleaned]
    return (exact or sorted(matches, key=lambda window: len(window.title)))[0]


def window_client_rect(window: WindowInfo) -> CaptureRect:
    user32 = _windows_user32()
    if user32.IsIconic(window.handle):
        raise RuntimeError(f"Window is minimized: {window.title}")
    rectangle = wintypes.RECT()
    if not user32.GetClientRect(window.handle, ctypes.byref(rectangle)):
        raise RuntimeError(f"Could not read client area for: {window.title}")
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(window.handle, ctypes.byref(origin)):
        raise RuntimeError(f"Could not locate client area for: {window.title}")
    return CaptureRect(
        left=int(origin.x),
        top=int(origin.y),
        width=int(rectangle.right - rectangle.left),
        height=int(rectangle.bottom - rectangle.top),
    )


def window_outer_rect(window: WindowInfo) -> CaptureRect:
    user32 = _windows_user32()
    rectangle = wintypes.RECT()
    if not user32.GetWindowRect(window.handle, ctypes.byref(rectangle)):
        raise RuntimeError(f"Could not read window bounds for: {window.title}")
    return CaptureRect(
        left=int(rectangle.left),
        top=int(rectangle.top),
        width=int(rectangle.right - rectangle.left),
        height=int(rectangle.bottom - rectangle.top),
    )


def list_monitor_rects() -> list[CaptureRect]:
    try:
        from mss import mss  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Screen capture dependency is missing. Run "
            "`pip install -r backend/requirements.txt`."
        ) from exc
    with mss() as capture:
        return [
            CaptureRect(
                int(monitor["left"]),
                int(monitor["top"]),
                int(monitor["width"]),
                int(monitor["height"]),
            )
            for monitor in capture.monitors[1:]
        ]


def monitor_containing(region: CaptureRect) -> tuple[int, CaptureRect] | None:
    for index, monitor in enumerate(list_monitor_rects(), start=1):
        if (
            monitor.left <= region.left
            and monitor.top <= region.top
            and monitor.left + monitor.width >= region.left + region.width
            and monitor.top + monitor.height >= region.top + region.height
        ):
            return index, monitor
    return None


class ScreenCapture:
    """Capture a window client area, monitor, or explicit desktop rectangle."""

    def __init__(
        self,
        *,
        window_title: str | None = None,
        region: CaptureRect | None = None,
        monitor: int | None = None,
    ) -> None:
        selected = sum(value is not None for value in (window_title, region, monitor))
        if selected != 1:
            raise ValueError("Choose exactly one of window_title, region, or monitor.")
        try:
            from mss import mss  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Screen capture dependency is missing. Run "
                "`pip install -r backend/requirements.txt`."
            ) from exc
        self._capture = mss()
        self.window_title = window_title
        self.region = region
        self.monitor = monitor

    def close(self) -> None:
        self._capture.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def resolved_rect(self) -> CaptureRect:
        if self.window_title is not None:
            return window_client_rect(find_window(self.window_title))
        if self.region is not None:
            return self.region
        monitors = self._capture.monitors
        index = int(self.monitor or 1)
        if index <= 0 or index >= len(monitors):
            raise ValueError(f"Monitor {index} is unavailable; choose 1..{len(monitors) - 1}.")
        selected = monitors[index]
        return CaptureRect(
            int(selected["left"]),
            int(selected["top"]),
            int(selected["width"]),
            int(selected["height"]),
        )

    def grab(self, rectangle: CaptureRect | None = None):
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("NumPy is required for screen capture.") from exc
        rectangle = rectangle or self.resolved_rect()
        screenshot = self._capture.grab(rectangle.as_mss())
        # MSS supplies BGRA, which is already the channel order OpenCV expects.
        return np.asarray(screenshot, dtype=np.uint8)[:, :, :3].copy()

    def description(self) -> str:
        if self.window_title is not None:
            return f'window containing "{self.window_title}"'
        if self.region is not None:
            return (
                f"region {self.region.left},{self.region.top},"
                f"{self.region.width},{self.region.height}"
            )
        return f"monitor {self.monitor or 1}"
