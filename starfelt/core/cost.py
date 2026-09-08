from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from starfelt.core.config import StarfeltConfig


def _runs_dir() -> Path:
    d = Path.cwd() / ".starfelt" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _history_path() -> Path:
    return Path.cwd() / ".starfelt" / "history.json"


class CostTracker:
    def __init__(self, cfg: StarfeltConfig, script: str):
        self.cfg = cfg
        self.script = script
        self.run_id = uuid.uuid4().hex
        self.t0 = time.time()

    def finish(self, exit_code: int) -> dict[str, Any]:
        duration_s = time.time() - self.t0
        hours = duration_s / 3600.0
        cost = hours * self.cfg.gpu_hour_usd
        row = {
            "run_id": self.run_id,
            "script": self.script,
            "duration_s": duration_s,
            "cost_usd": cost,
            "baseline_cost_usd": cost * self.cfg.baseline_multiplier,
            "exit_code": exit_code,
            "ts": time.time(),
        }
        path = _history_path()
        hist = load_history()
        hist.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
        ( _runs_dir() / f"{self.run_id}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        return row


def load_history() -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _active_path() -> Path:
    return Path.cwd() / ".starfelt" / "active.json"


def write_active_run(row: dict[str, Any]) -> None:
    path = _active_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")


def clear_active_run() -> None:
    path = _active_path()
    if path.exists():
        path.unlink()


def read_active_run() -> dict[str, Any] | None:
    path = _active_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
