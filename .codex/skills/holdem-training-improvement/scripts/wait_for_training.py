#!/usr/bin/env python3
"""Wait for a Holdem background training run to write its result report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def get_status(base_url: str, timeout_seconds: float) -> dict:
    with urlopen(f"{base_url.rstrip('/')}/api/training/status", timeout=timeout_seconds) as response:  # noqa: S310 - local user-supplied API
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--report-directory", default="backend/data/training_reports", type=Path)
    parser.add_argument("--timeout-seconds", default=7200, type=float)
    parser.add_argument("--poll-seconds", default=5, type=float)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout_seconds
    observed_running = False
    last_error: str | None = None

    while time.monotonic() < deadline:
        try:
            status = get_status(args.base_url, min(30.0, args.poll_seconds + 5.0))
            observed_running = observed_running or bool(status.get("running"))
            filename = status.get("report_filename")
            report_path = args.report_directory / filename if isinstance(filename, str) and filename else None
            if observed_running and not status.get("running") and status.get("report_status") == "written" and report_path and report_path.is_file():
                print(json.dumps({"completed": True, "status": status, "report_path": str(report_path)}, indent=2, sort_keys=True))
                return 0
            if observed_running and not status.get("running") and status.get("last_error"):
                print(json.dumps({"completed": False, "reason": "training reported an error", "status": status}, indent=2, sort_keys=True))
                return 3
            last_error = None
        except (OSError, URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(max(0.2, args.poll_seconds))

    print(json.dumps({"completed": False, "reason": "timed out waiting for a training result", "observed_running": observed_running, "last_error": last_error}, indent=2, sort_keys=True))
    return 4


if __name__ == "__main__":
    sys.exit(main())
