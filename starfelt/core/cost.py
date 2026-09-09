from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from starfelt.core.config import StarfeltConfig
from starfelt.core.telemetry import TelemetryHints, build_run_telemetry


def _runs_dir() -> Path:
    d = Path.cwd() / ".starfelt" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _history_path() -> Path:
    return Path.cwd() / ".starfelt" / "history.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON without leaving a half-written file after an interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


class CostTracker:
    def __init__(self, cfg: StarfeltConfig, script: str):
        self.cfg = cfg
        self.script = script
        self.run_id = uuid.uuid4().hex
        self.t0 = time.time()
        self.extras: dict[str, Any] = {}

    def finish(
        self,
        exit_code: int,
        *,
        hints: TelemetryHints | None = None,
        model_param_count: int | None = None,
        gpu_name: str | None = None,
        gpu_util_avg: float | None = None,
        gpu_samples: int | None = None,
        gpu_power_avg_w: float | None = None,
        gpu_energy_j: float | None = None,
        interrupted: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Persist run record with real duration/cost, then full telemetry."""
        duration_s = time.time() - self.t0
        hours = duration_s / 3600.0
        cost = hours * self.cfg.gpu_hour_usd

        row = build_run_telemetry(
            run_id=self.run_id,
            script=self.script,
            duration_s=duration_s,
            cost_usd=cost,
            exit_code=exit_code,
            hints=hints,
            model_param_count=model_param_count,
            gpu_name=gpu_name,
            gpu_util_avg=gpu_util_avg,
            gpu_samples=gpu_samples,
            gpu_power_avg_w=gpu_power_avg_w,
            gpu_energy_j=gpu_energy_j,
            interrupted=interrupted,
            extra={**self.extras, **extra, "ts": time.time()},
        )

        path = _history_path()
        hist = load_history()
        hist.append(row)
        _atomic_write_json(path, hist)
        _atomic_write_json(_runs_dir() / f"{self.run_id}.json", row)
        return row


def load_history() -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not read {path}: invalid JSON. Restore the file or remove it "
            "after making a backup."
        ) from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Could not read {path}: expected a JSON list of runs.")
    return data


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


def write_interrupted_marker(
    run_id: str,
    *,
    script: str,
    elapsed_s: float,
    cost_usd: float,
    signal_name: str,
) -> Path:
    path = _runs_dir() / f"{run_id}_interrupted.json"
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "script": script,
                "elapsed_s": elapsed_s,
                "cost_usd": cost_usd,
                "signal": signal_name,
                "ts": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
