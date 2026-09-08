from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

KNOWN_PROVIDERS = {"runpod", "lambda", "aws", "gcp"}

DEFAULT_CONFIG = """# Starfelt project config
project: my-training
budget_usd_per_run: 50
early_stop:
  enabled: true
  patience_steps: 500
checkpoint:
  every_steps: 200
providers:
  preferred:
    - runpod
    - lambda
    - aws
    - gcp
  allow_spot: true
cost:
  # rough $/GPU-hour defaults for local estimates
  gpu_hour_usd: 1.20
  baseline_multiplier: 1.35  # assumed waste without Starfelt
"""


@dataclass
class StarfeltConfig:
    project: str = "my-training"
    budget_usd_per_run: float = 50.0
    gpu_hour_usd: float = 1.20
    baseline_multiplier: float = 1.35
    early_stop_enabled: bool = True
    patience_steps: int = 500
    checkpoint_every_steps: int = 200
    preferred_providers: list[str] = field(
        default_factory=lambda: ["runpod", "lambda", "aws", "gcp"]
    )
    allow_spot: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        errors: list[str] = []
        if self.budget_usd_per_run < 0:
            errors.append(f"budget_usd_per_run must be >= 0 (got {self.budget_usd_per_run})")
        if self.gpu_hour_usd <= 0:
            errors.append(f"cost.gpu_hour_usd must be > 0 (got {self.gpu_hour_usd})")
        if self.baseline_multiplier < 1.0:
            errors.append(
                f"cost.baseline_multiplier should be >= 1.0 (got {self.baseline_multiplier})"
            )
        if self.patience_steps < 1:
            errors.append(f"early_stop.patience_steps must be >= 1")
        for p in self.preferred_providers:
            if p not in KNOWN_PROVIDERS:
                errors.append(
                    f"unknown provider '{p}' (known: {', '.join(sorted(KNOWN_PROVIDERS))})"
                )
        if errors:
            raise click_error("Invalid starfelt config:\n  - " + "\n  - ".join(errors))


def click_error(msg: str) -> Exception:
    import click

    return click.ClickException(msg)


def init_project(cwd: Path, force: bool = False) -> Path:
    path = cwd / "starfelt.yaml"
    if path.exists() and not force:
        raise click_error(f"{path} already exists (use --force)")
    path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    (cwd / ".starfelt").mkdir(exist_ok=True)
    (cwd / ".starfelt" / "runs").mkdir(exist_ok=True)
    return path


def validate_environment() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    major, minor = sys.version_info[:2]
    py_ok = (major, minor) >= (3, 10)
    rows.append(
        (
            "python",
            "ok" if py_ok else "fail",
            f"{major}.{minor}.{sys.version_info[2]}"
            + ("" if py_ok else " (need 3.10+)"),
        )
    )
    try:
        import psutil  # noqa: F401

        rows.append(("psutil", "ok", "importable"))
    except ImportError:
        rows.append(("psutil", "fail", "not installed — pip install psutil"))
    try:
        import yaml as _yaml  # noqa: F401

        rows.append(("pyyaml", "ok", "importable"))
    except ImportError:
        rows.append(("pyyaml", "fail", "not installed"))
    try:
        import rich  # noqa: F401

        rows.append(("rich", "ok", "importable"))
    except ImportError:
        rows.append(("rich", "fail", "not installed"))
    starfelt_dir = Path.cwd() / ".starfelt"
    try:
        starfelt_dir.mkdir(exist_ok=True)
        probe = starfelt_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        rows.append((".starfelt/", "ok", "writable"))
    except OSError as e:
        rows.append((".starfelt/", "fail", str(e)))
    return rows


def load_config(path: Path | None = None) -> StarfeltConfig:
    candidates = []
    if path:
        candidates.append(path)
    candidates.append(Path.cwd() / "starfelt.yaml")
    data: dict[str, Any] = {}
    for c in candidates:
        if c and c.exists():
            data = yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            break
    cost = data.get("cost") or {}
    early = data.get("early_stop") or {}
    ckpt = data.get("checkpoint") or {}
    providers = data.get("providers") or {}

    def _f(d, key, default):
        if key not in d or d[key] is None:
            return float(default)
        return float(d[key])

    cfg = StarfeltConfig(
        project=str(data.get("project") or "my-training"),
        budget_usd_per_run=_f(data, "budget_usd_per_run", 50),
        gpu_hour_usd=_f(cost, "gpu_hour_usd", 1.20),
        baseline_multiplier=_f(cost, "baseline_multiplier", 1.35),
        early_stop_enabled=bool(early.get("enabled", True)),
        patience_steps=int(early.get("patience_steps") or 500),
        checkpoint_every_steps=int(ckpt.get("every_steps") or 200),
        preferred_providers=list(
            providers.get("preferred") or ["runpod", "lambda", "aws", "gcp"]
        ),
        allow_spot=bool(providers.get("allow_spot", True)),
        raw=data,
    )
    cfg.validate()
    return cfg
