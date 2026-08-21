"""A record of exactly what produced a set of results."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        h = out.stdout.strip() or "unknown"
        return h + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def write(path: Path, cfg: dict[str, Any], extra: dict[str, Any]) -> Path:
    """Config, code version and environment alongside the numbers.

    Without this a results file is an assertion. With it, a reader can tell
    which parameters and which commit produced the figures, which is the
    difference between a reproducible result and a screenshot.
    """
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": cfg,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
