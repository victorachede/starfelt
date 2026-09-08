"""Run telemetry schema + workload fingerprint (Batch 3).

This is the shape that later feeds chip design: what people actually train.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

TELEMETRY_SCHEMA_VERSION = "1.0.0"

# Documented schema (also enforced loosely when building records)
TELEMETRY_SCHEMA: dict[str, Any] = {
    "schema_version": "string",
    "run_id": "string",
    "script": "string",
    "ts": "number",
    "duration_s": "number",
    "cost_usd": "number",
    "baseline_cost_usd": "number",
    "saved_usd": "number",
    "exit_code": "integer",
    "framework": "pytorch|jax|tensorflow|keras|unknown",
    "frameworks_detected": "string[]",
    "batch_size": "integer|null",
    "epochs": "integer|null",
    "learning_rate": "number|null",
    "dataset_size": "integer|null",
    "model_param_count": "integer|null",
    "model_size_bucket": "string|null",
    "dataset_size_bucket": "string|null",
    "gpu_name": "string|null",
    "gpu_util_avg": "number|null",
    "gpu_samples": "integer|null",
    "optimization_flags": "string[]",
    "workload_id": "string",
    "interrupted": "string|null",
}


@dataclass
class TelemetryHints:
    """Static analysis side of telemetry (pre-run)."""

    framework: str = "unknown"
    frameworks_detected: list[str] = field(default_factory=list)
    batch_size: int | None = None
    epochs: int | None = None
    learning_rate: float | None = None
    dataset_size: int | None = None
    optimization_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def size_bucket(n: int | None, *, kind: str) -> str | None:
    if n is None:
        return None
    if kind == "model":
        # parameters
        if n < 1_000_000:
            return "lt_1m"
        if n < 100_000_000:
            return "1m_100m"
        if n < 1_000_000_000:
            return "100m_1b"
        return "gte_1b"
    # dataset examples
    if n < 1_000:
        return "lt_1k"
    if n < 100_000:
        return "1k_100k"
    if n < 1_000_000:
        return "100k_1m"
    return "gte_1m"


def workload_id(
    *,
    framework: str,
    model_size_bucket: str | None,
    dataset_size_bucket: str | None,
    batch_size: int | None,
    epochs: int | None,
) -> str:
    """Stable fingerprint for grouping similar runs across users later."""
    payload = "|".join(
        [
            framework or "unknown",
            model_size_bucket or "na",
            dataset_size_bucket or "na",
            str(batch_size if batch_size is not None else "na"),
            str(epochs if epochs is not None else "na"),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"wl_{digest}"


def build_run_telemetry(
    *,
    run_id: str,
    script: str,
    duration_s: float,
    cost_usd: float,
    baseline_cost_usd: float,
    exit_code: int,
    hints: TelemetryHints | None = None,
    model_param_count: int | None = None,
    gpu_name: str | None = None,
    gpu_util_avg: float | None = None,
    gpu_samples: int | None = None,
    interrupted: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hints = hints or TelemetryHints()
    model_bucket = size_bucket(model_param_count, kind="model")
    data_bucket = size_bucket(hints.dataset_size, kind="dataset")
    wl = workload_id(
        framework=hints.framework,
        model_size_bucket=model_bucket,
        dataset_size_bucket=data_bucket,
        batch_size=hints.batch_size,
        epochs=hints.epochs,
    )
    row: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "run_id": run_id,
        "script": script,
        "duration_s": duration_s,
        "cost_usd": cost_usd,
        "baseline_cost_usd": baseline_cost_usd,
        "saved_usd": baseline_cost_usd - cost_usd,
        "exit_code": exit_code,
        "framework": hints.framework,
        "frameworks_detected": hints.frameworks_detected,
        "batch_size": hints.batch_size,
        "epochs": hints.epochs,
        "learning_rate": hints.learning_rate,
        "dataset_size": hints.dataset_size,
        "model_param_count": model_param_count,
        "model_size_bucket": model_bucket,
        "dataset_size_bucket": data_bucket,
        "gpu_name": gpu_name,
        "gpu_util_avg": gpu_util_avg,
        "gpu_samples": gpu_samples,
        "optimization_flags": hints.optimization_flags,
        "workload_id": wl,
        "interrupted": interrupted,
    }
    if extra:
        row.update(extra)
    return row


def schema_markdown() -> str:
    return json.dumps(TELEMETRY_SCHEMA, indent=2)
