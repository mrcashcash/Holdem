"""Tkinter control panel to launch and stop the project's runnable scripts.

Each script is defined once in ``SCRIPTS`` with a base command (what follows
``python -u``) and an editable arguments string that is pre-filled with the
CLI defaults. The user tweaks the arguments per script, then Starts/Stops it
individually or uses Start All / Stop All. Every process runs as a subprocess
rooted at the repository, its stdout/stderr is streamed live into a per-script
log tab, and the arguments are persisted between sessions.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GUI_SETTINGS_PATH = REPOSITORY_ROOT / "backend" / "data" / "script_control_gui.json"
GUI_TITLE = "Script Control Panel"


@dataclass(frozen=True)
class ScriptSpec:
    """A runnable project script.

    ``base`` is everything that follows ``python -u`` and is fixed; ``args`` is
    the editable, user-supplied remainder (parsed with :func:`shlex.split`).
    """

    key: str
    label: str
    base: tuple[str, ...]
    default_args: str
    description: str


SCRIPTS: tuple[ScriptSpec, ...] = (
    ScriptSpec(
        key="api",
        label="API server (FastAPI / uvicorn)",
        base=("-m", "uvicorn", "backend.main:app"),
        default_args="--reload --host 127.0.0.1 --port 8000",
        description="Serves the game + blueprint trainer endpoints.",
    ),
    ScriptSpec(
        key="gpu_train",
        label="GPU blueprint trainer",
        base=("-m", "backend.solver.gpu.train"),
        default_args="--iterations 2000 --device cuda --save-every 200 --stack-bb 100",
        description="Trains the dense GPU blueprint.",
    ),
    ScriptSpec(
        key="mccfr_train",
        label="MCCFR blueprint trainer",
        base=("-m", "backend.solver.blueprint"),
        default_args="--iterations 10000 --save-every 5000",
        description="Trains the hold'em blueprint with Linear MCCFR.",
    ),
    ScriptSpec(
        key="benchmark",
        label="Benchmark vs scripted styles",
        base=("-m", "backend.eval.benchmark"),
        default_args="--hands 1000",
        description="Benchmarks the blueprint against scripted styles.",
    ),
    ScriptSpec(
        key="duel",
        label="Duel challenger vs champion",
        base=("-m", "backend.eval.duel"),
        default_args="--data-dir backend/data/gpu_blueprint_100bb --stack-bb 100 --pairs 3000",
        description="Requires --data-dir and --stack-bb.",
    ),
    ScriptSpec(
        key="promote",
        label="Promotion gate",
        base=("-m", "backend.eval.promote"),
        default_args="--stack-bb 100",
        description="Promotes the latest checkpoint if it beats the champion.",
    ),
    ScriptSpec(
        key="slumbot",
        label="Play vs Slumbot",
        base=("-m", "backend.eval.slumbot"),
        default_args="--hands 500 --gpu",
        description="Plays the serving GPU champion against Slumbot (the external absolute benchmark).",
    ),
    ScriptSpec(
        key="slumbot_null",
        label="Slumbot harness NULL check",
        base=("-m", "backend.eval.slumbot"),
        default_args="--hands 200 --null always-fold --no-aivat",
        description="Instrument self-test: always-fold must read -50 bb/100 from the button.",
    ),
    ScriptSpec(
        key="lbr",
        label="LBR exploitability probe",
        base=("-m", "backend.eval.lbr"),
        default_args="--hands 500",
        description="Local best-response exploitability probe.",
    ),
    ScriptSpec(
        key="cfv_dataset",
        label="River CFV dataset generator",
        base=("tools/generate_river_cfv.py",),
        default_args="--samples 250000 --out backend/data/cfv/river",
        description="Generates exact river CFV training data (resumable).",
    ),
    ScriptSpec(
        key="cfv_train",
        label="River CFV net gate",
        base=("tools/river_net_gate.py",),
        default_args="--net backend/data/cfv/river_net.pt --situations 20",
        description="Action-agreement gate for the river CFV net.",
    ),
)


@dataclass
class ScriptState:
    """Live UI + process state for a single script."""

    spec: ScriptSpec
    args_var: tk.StringVar
    status_var: tk.StringVar
    dot: tk.Label
    start_button: ttk.Button
    stop_button: ttk.Button
    log: tk.Text
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    stopping: bool = False


# Log-message kinds pushed by reader threads onto the shared queue.
@dataclass
class LogEvent:
    key: str
    kind: str  # "line" | "started" | "exited"
    text: str = ""
    code: int | None = None


DOT_COLORS = {
    "idle": "#52623f",
    "running": "#b8f36b",
    "stopping": "#f1c75b",
    "exited_ok": "#7fd0ff",
    "exited_err": "#ff6b6b",
}


class ScriptControlGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(GUI_TITLE)
        self.root.geometry("1220x820")
        self.root.minsize(1040, 680)

        self._events: queue.Queue[LogEvent] = queue.Queue()
        self._states: dict[str, ScriptState] = {}
        self._closing = False
        self.status = tk.StringVar(value="Ready")

        self._configure_style()
        self._build_interface()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    # ---- styling -----------------------------------------------------------

    def _configure_style(self) -> None:
        self.root.configure(background="#101614")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background="#16201c", foreground="#e7f2ec", fieldbackground="#0e1713")
        style.configure("TFrame", background="#101614")
        style.configure("Panel.TFrame", background="#16201c")
        style.configure("Card.TFrame", background="#16201c")
        style.configure("TLabel", background="#16201c", foreground="#dcebe3")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#b8f36b")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground="#b8f36b")
        style.configure("Script.TLabel", font=("Segoe UI", 10, "bold"), foreground="#e7f2ec")
        style.configure("Hint.TLabel", font=("Segoe UI", 8), foreground="#8fa79a")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"), foreground="#f1c75b")
        style.configure("TButton", padding=(10, 5))
        style.configure("Primary.TButton", foreground="#101614", background="#b8f36b")
        style.map("Primary.TButton", background=[("active", "#d0ff91"), ("disabled", "#52623f")])
        style.configure("Danger.TButton", foreground="#101614", background="#f0906b")
        style.map("Danger.TButton", background=[("active", "#ffb094"), ("disabled", "#63483f")])
        style.configure("TEntry", fieldbackground="#0e1713", foreground="#ffffff")
        style.configure("TLabelframe", background="#16201c", foreground="#b8f36b")
        style.configure("TLabelframe.Label", background="#16201c", foreground="#b8f36b")
        style.configure("TNotebook", background="#101614", borderwidth=0)
        style.configure("TNotebook.Tab", background="#16201c", foreground="#dcebe3", padding=(10, 4))
        style.map("TNotebook.Tab", background=[("selected", "#22322b")], foreground=[("selected", "#b8f36b")])

    # ---- layout ------------------------------------------------------------

    def _build_interface(self) -> None:
        shell = ttk.Frame(self.root, padding=14)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=0, minsize=560)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Script Control Panel", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        buttons = ttk.Frame(header)
        buttons.grid(row=0, column=1, sticky="e")
        ttk.Button(buttons, text="Start All", style="Primary.TButton", command=self._start_all).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(buttons, text="Stop All", style="Danger.TButton", command=self._stop_all).grid(
            row=0, column=1
        )
        ttk.Label(header, textvariable=self.status, style="Status.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        # Log panel first: card construction registers a tab per script.
        self._build_log_panel(shell)
        self._build_script_list(shell)

    def _build_script_list(self, shell: ttk.Frame) -> None:
        container = ttk.Frame(shell, style="Panel.TFrame", padding=8)
        container.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, background="#16201c", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        inner = ttk.Frame(canvas, style="Panel.TFrame")
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.columnconfigure(0, weight=1)

        def _resize(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window, width=canvas.winfo_width())

        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", _resize)
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"),
        )

        for row, spec in enumerate(SCRIPTS):
            self._build_card(inner, spec, row)

    def _build_card(self, parent: ttk.Frame, spec: ScriptSpec, row: int) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 8), padx=2)
        card.columnconfigure(1, weight=1)

        dot = tk.Label(card, text="●", background="#16201c", foreground=DOT_COLORS["idle"])
        dot.grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Label(card, text=spec.label, style="Script.TLabel").grid(row=0, column=1, sticky="w")

        status_var = tk.StringVar(value="idle")
        ttk.Label(card, textvariable=status_var, style="Hint.TLabel").grid(
            row=0, column=2, sticky="e"
        )

        ttk.Label(card, text=f"python -u {' '.join(spec.base)}", style="Hint.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(2, 2)
        )

        args_var = tk.StringVar(value=spec.default_args)
        entry = ttk.Entry(card, textvariable=args_var)
        entry.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        controls = ttk.Frame(card, style="Card.TFrame")
        controls.grid(row=3, column=0, columnspan=3, sticky="ew")
        controls.columnconfigure(2, weight=1)
        start_button = ttk.Button(
            controls, text="Start", style="Primary.TButton",
            command=lambda k=spec.key: self._start(k),
        )
        start_button.grid(row=0, column=0, padx=(0, 6))
        stop_button = ttk.Button(
            controls, text="Stop", state="disabled",
            command=lambda k=spec.key: self._stop(k),
        )
        stop_button.grid(row=0, column=1)
        ttk.Label(controls, text=spec.description, style="Hint.TLabel").grid(
            row=0, column=2, sticky="e"
        )

        log = self._log_tabs.add_tab(spec.key, spec.label)

        self._states[spec.key] = ScriptState(
            spec=spec,
            args_var=args_var,
            status_var=status_var,
            dot=dot,
            start_button=start_button,
            stop_button=stop_button,
            log=log,
        )

    def _build_log_panel(self, shell: ttk.Frame) -> None:
        panel = ttk.Frame(shell, style="Panel.TFrame", padding=8)
        panel.grid(row=1, column=1, sticky="nsew")
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="Output", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self._log_tabs = LogTabs(panel)
        self._log_tabs.notebook.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

    # ---- process control ---------------------------------------------------

    def _command(self, state: ScriptState) -> list[str]:
        try:
            extra = shlex.split(state.args_var.get(), posix=False)
        except ValueError as exc:
            raise ValueError(f"Could not parse arguments: {exc}") from exc
        return [sys.executable, "-u", *state.spec.base, *extra]

    def _start(self, key: str) -> None:
        state = self._states[key]
        if state.process is not None and state.process.poll() is None:
            return
        try:
            command = self._command(state)
        except ValueError as exc:
            messagebox.showerror(GUI_TITLE, str(exc), parent=self.root)
            return

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            process = subprocess.Popen(
                command,
                cwd=str(REPOSITORY_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror(GUI_TITLE, f"Could not launch {state.spec.label}:\n{exc}", parent=self.root)
            return

        state.process = process
        state.stopping = False
        state.reader = threading.Thread(
            target=self._reader_loop, args=(key, process), daemon=True
        )
        state.reader.start()

        self._events.put(LogEvent(key, "started", text=" ".join(shlex.quote(c) for c in command)))
        self._log_tabs.select(key)

    def _reader_loop(self, key: str, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self._events.put(LogEvent(key, "line", text=line.rstrip("\n")))
        code = process.wait()
        self._events.put(LogEvent(key, "exited", code=code))

    def _stop(self, key: str) -> None:
        state = self._states[key]
        process = state.process
        if process is None or process.poll() is not None:
            return
        state.stopping = True
        state.status_var.set("stopping…")
        state.dot.configure(foreground=DOT_COLORS["stopping"])
        state.stop_button.configure(state="disabled")
        threading.Thread(target=self._terminate, args=(process,), daemon=True).start()

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                # Kill the whole tree; uvicorn --reload and torch spawn children.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.send_signal(signal.SIGINT)
        except OSError:
            pass
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass

    def _start_all(self) -> None:
        for key in self._states:
            self._start(key)

    def _stop_all(self) -> None:
        for key in self._states:
            self._stop(key)

    # ---- event pump --------------------------------------------------------

    def _poll_events(self) -> None:
        try:
            while True:
                self._handle_event(self._events.get_nowait())
        except queue.Empty:
            pass
        if self._closing:
            if not any(
                s.process is not None and s.process.poll() is None
                for s in self._states.values()
            ):
                self.root.destroy()
                return
        self.root.after(100, self._poll_events)
        self._refresh_status()

    def _handle_event(self, event: LogEvent) -> None:
        state = self._states[event.key]
        if event.kind == "started":
            state.status_var.set("running")
            state.dot.configure(foreground=DOT_COLORS["running"])
            state.start_button.configure(state="disabled")
            state.stop_button.configure(state="normal")
            self._append(state, f"$ {event.text}")
        elif event.kind == "line":
            self._append(state, event.text)
        elif event.kind == "exited":
            code = event.code
            was_stopping = state.stopping
            state.process = None
            state.reader = None
            state.stopping = False
            state.start_button.configure(state="normal")
            state.stop_button.configure(state="disabled")
            if was_stopping or code in (0, None):
                state.dot.configure(foreground=DOT_COLORS["exited_ok"])
                state.status_var.set("stopped" if was_stopping else "finished")
            else:
                state.dot.configure(foreground=DOT_COLORS["exited_err"])
                state.status_var.set(f"exited {code}")
            self._append(state, f"[process exited with code {code}]")

    def _refresh_status(self) -> None:
        running = sum(
            1 for s in self._states.values()
            if s.process is not None and s.process.poll() is None
        )
        if self._closing:
            self.status.set(f"Stopping… {running} still running")
        elif running:
            self.status.set(f"{running} script(s) running")
        else:
            self.status.set("Ready")

    def _append(self, state: ScriptState, message: str) -> None:
        log = state.log
        log.configure(state="normal")
        log.insert("end", message + "\n")
        log.see("end")
        # Cap the buffer so long-running trainers don't grow without bound.
        if int(log.index("end-1c").split(".")[0]) > 4000:
            log.delete("1.0", "1500.0")
        log.configure(state="disabled")

    # ---- settings ----------------------------------------------------------

    def _save_settings(self) -> None:
        payload = {key: state.args_var.get() for key, state in self._states.items()}
        try:
            GUI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            GUI_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_settings(self) -> None:
        if not GUI_SETTINGS_PATH.is_file():
            return
        try:
            payload = json.loads(GUI_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for key, value in payload.items():
            state = self._states.get(key)
            if state is not None and isinstance(value, str):
                state.args_var.set(value)

    def _on_close(self) -> None:
        self._save_settings()
        running = [
            s for s in self._states.values()
            if s.process is not None and s.process.poll() is None
        ]
        if running and not self._closing:
            if not messagebox.askyesno(
                GUI_TITLE,
                f"{len(running)} script(s) are still running. Stop them and exit?",
                parent=self.root,
            ):
                return
            self._closing = True
            self._stop_all()
            return
        self.root.destroy()


class LogTabs:
    """A notebook of per-script read-only log panes."""

    def __init__(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self._tabs: dict[str, tk.Text] = {}
        self._frames: dict[str, ttk.Frame] = {}

    def add_tab(self, key: str, label: str) -> tk.Text:
        frame = ttk.Frame(self.notebook, style="Panel.TFrame")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = tk.Text(
            frame,
            state="disabled",
            background="#090e0c",
            foreground="#cbd8d1",
            insertbackground="#ffffff",
            relief="flat",
            wrap="none",
            font=("Consolas", 9),
        )
        text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.notebook.add(frame, text=label)
        self._tabs[key] = text
        self._frames[key] = frame
        return text

    def select(self, key: str) -> None:
        frame = self._frames.get(key)
        if frame is not None:
            self.notebook.select(frame)


def run_gui() -> None:
    root = tk.Tk()
    ScriptControlGui(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
