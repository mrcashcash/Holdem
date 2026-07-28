"""Event-driven frame streams with Windows Graphics Capture and MSS fallback."""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .capture import (
    CaptureRect,
    ScreenCapture,
    WindowInfo,
    find_window,
    list_monitor_rects,
    monitor_containing,
    window_client_rect,
    window_outer_rect,
)


@dataclass(frozen=True)
class StreamFrame:
    image: Any
    captured_at: str
    captured_monotonic: float
    rect: CaptureRect


class FrameStream:
    backend_name = "unknown"

    def start(self) -> None:
        raise NotImplementedError

    def next_frame(self, timeout: float = 0.1) -> StreamFrame | None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def wait(self) -> None:
        raise NotImplementedError

    def description(self) -> str:
        raise NotImplementedError


class _QueuedFrameStream(FrameStream):
    def __init__(self, fps: float, buffer_seconds: float = 2.0) -> None:
        self.fps = fps
        self._frames: queue.Queue[StreamFrame] = queue.Queue(
            maxsize=max(4, int(round(fps * buffer_seconds)))
        )
        self._stop_event = threading.Event()
        self._closed_event = threading.Event()
        self._errors: list[str] = []

    def _push(self, frame: StreamFrame) -> None:
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)

    def next_frame(self, timeout: float = 0.1) -> StreamFrame | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            if self._errors:
                raise RuntimeError(self._errors[0])
            if self._closed_event.is_set() and not self._stop_event.is_set():
                raise RuntimeError("The capture stream closed unexpectedly.")
            return None


class MssFrameStream(_QueuedFrameStream):
    backend_name = "MSS fallback"

    def __init__(
        self,
        *,
        fps: float,
        window_title: str | None,
        region: CaptureRect | None,
        monitor: int | None,
    ) -> None:
        super().__init__(fps)
        self.window_title = window_title
        self.region = region
        self.monitor = monitor
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="screen-history-mss-stream",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            with ScreenCapture(
                window_title=self.window_title,
                region=self.region,
                monitor=self.monitor,
            ) as capture:
                while not self._stop_event.is_set():
                    started = time.monotonic()
                    rectangle = capture.resolved_rect()
                    image = capture.grab(rectangle)
                    self._push(
                        StreamFrame(
                            image=image,
                            captured_at=datetime.now(timezone.utc).isoformat(),
                            captured_monotonic=time.monotonic(),
                            rect=rectangle,
                        )
                    )
                    remaining = (1.0 / self.fps) - (time.monotonic() - started)
                    self._stop_event.wait(max(0.0, remaining))
        except Exception as exc:  # third-party capture errors surface on this worker
            self._errors.append(str(exc))
        finally:
            self._closed_event.set()

    def stop(self) -> None:
        self._stop_event.set()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def description(self) -> str:
        if self.window_title is not None:
            return f'window containing "{self.window_title}"'
        if self.region is not None:
            return (
                f"region {self.region.left},{self.region.top},"
                f"{self.region.width},{self.region.height}"
            )
        return f"monitor {self.monitor or 1}"


class WindowsGraphicsFrameStream(_QueuedFrameStream):
    backend_name = "Windows Graphics Capture"

    def __init__(
        self,
        *,
        fps: float,
        window_title: str | None,
        region: CaptureRect | None,
        monitor: int | None,
    ) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Windows Graphics Capture is available only on Windows.")
        super().__init__(fps)
        try:
            from windows_capture import WindowsCapture  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Windows Graphics Capture is not installed. Run "
                "`pip install -r backend/requirements.txt`."
            ) from exc

        self.window_title = window_title
        self.region = region
        self.monitor = monitor
        self._window: WindowInfo | None = None
        self._monitor_rect: CaptureRect | None = None
        self._region_offset: tuple[int, int, int, int] | None = None
        self._capture_control = None

        capture_arguments: dict[str, Any] = {
            "cursor_capture": False,
            "draw_border": False,
            "minimum_update_interval": max(1, int(round(1000.0 / fps))),
            "dirty_region": False,
        }
        if window_title is not None:
            self._window = find_window(window_title)
            capture_arguments["window_hwnd"] = self._window.handle
        elif region is not None:
            selected = monitor_containing(region)
            if selected is None:
                raise RuntimeError(
                    "The selected region crosses monitor boundaries; use MSS fallback."
                )
            monitor_index, monitor_rect = selected
            self._monitor_rect = monitor_rect
            self._region_offset = (
                region.left - monitor_rect.left,
                region.top - monitor_rect.top,
                region.width,
                region.height,
            )
            # windows-capture's monitor_index is 1-based (index 1 == first
            # monitor), matching monitor_containing()'s 1-based index and the MSS
            # backend. Passing monitor_index - 1 captured the PREVIOUS physical
            # monitor while _monitor_rect/_region_offset described the intended
            # one, so the crop (and any overlay drawn from the rect) landed on
            # the wrong screen.
            capture_arguments["monitor_index"] = monitor_index
        else:
            monitors = list_monitor_rects()
            selected_index = int(monitor or 1)
            if selected_index <= 0 or selected_index > len(monitors):
                raise ValueError(
                    f"Monitor {selected_index} is unavailable; choose 1..{len(monitors)}."
                )
            self._monitor_rect = monitors[selected_index - 1]
            # 1-based to match windows-capture's monitor_index and the MSS
            # backend (selection N == physical monitor N). Passing
            # selected_index - 1 captured monitor N-1 while reporting monitor N's
            # rect, putting the capture and the overlay on different screens.
            capture_arguments["monitor_index"] = selected_index

        try:
            self._capture = WindowsCapture(**capture_arguments)
        except Exception as exc:  # normalize native-extension exceptions
            raise RuntimeError(f"Could not initialize Windows capture: {exc}") from exc
        self._register_callbacks()

    def _register_callbacks(self) -> None:
        @self._capture.event
        def on_frame_arrived(frame, _capture_control) -> None:
            if self._stop_event.is_set():
                return
            try:
                image = frame.frame_buffer[:, :, :3]
                rectangle = self._resolved_rect()
                if self._window is not None:
                    image = self._crop_window_client(image, rectangle)
                elif self._region_offset is not None:
                    left, top, width, height = self._region_offset
                    image = image[top : top + height, left : left + width]
                if image.size == 0:
                    raise RuntimeError("Windows capture returned an empty frame.")
                self._push(
                    StreamFrame(
                        image=image.copy(),
                        captured_at=datetime.now(timezone.utc).isoformat(),
                        captured_monotonic=time.monotonic(),
                        rect=rectangle,
                    )
                )
            except Exception as exc:  # never let a callback failure disappear in native code
                self._errors.append(str(exc))

        @self._capture.event
        def on_closed() -> None:
            self._closed_event.set()

    def _resolved_rect(self) -> CaptureRect:
        if self._window is not None:
            return window_client_rect(self._window)
        if self.region is not None:
            return self.region
        assert self._monitor_rect is not None
        return self._monitor_rect

    def _crop_window_client(self, image, client: CaptureRect):
        height, width = image.shape[:2]
        if width == client.width and height == client.height:
            return image
        assert self._window is not None
        outer = window_outer_rect(self._window)
        left = max(0, client.left - outer.left)
        top = max(0, client.top - outer.top)
        right = min(width, left + client.width)
        bottom = min(height, top + client.height)
        if right - left == client.width and bottom - top == client.height:
            return image[top:bottom, left:right]
        return image

    def start(self) -> None:
        try:
            self._capture_control = self._capture.start_free_threaded()
        except Exception as exc:
            raise RuntimeError(f"Could not start Windows capture: {exc}") from exc

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_control is not None:
            try:
                self._capture_control.stop()
            except Exception as exc:
                self._errors.append(f"Could not stop Windows capture: {exc}")

    def wait(self) -> None:
        if self._capture_control is not None:
            try:
                self._capture_control.wait()
            except Exception as exc:
                self._errors.append(f"Windows capture ended with an error: {exc}")

    def description(self) -> str:
        if self._window is not None:
            return f'window "{self._window.title}"'
        if self.region is not None:
            return (
                f"region {self.region.left},{self.region.top},"
                f"{self.region.width},{self.region.height}"
            )
        return f"monitor {self.monitor or 1}"


def create_frame_stream(
    *,
    backend: str,
    fps: float,
    window_title: str | None,
    region: CaptureRect | None,
    monitor: int | None,
) -> FrameStream:
    cleaned = backend.strip().lower()
    if cleaned not in {"auto", "windows", "mss"}:
        raise ValueError("Capture backend must be auto, windows, or mss.")
    if cleaned in {"auto", "windows"}:
        try:
            return WindowsGraphicsFrameStream(
                fps=fps,
                window_title=window_title,
                region=region,
                monitor=monitor,
            )
        except (ImportError, RuntimeError, ValueError, OSError):
            if cleaned == "windows":
                raise
    return MssFrameStream(
        fps=fps,
        window_title=window_title,
        region=region,
        monitor=monitor,
    )
