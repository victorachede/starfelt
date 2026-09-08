from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    """Return list of (check, status ok|fail, detail)."""
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
    return StarfeltConfig(
        project=str(data.get("project") or "my-training"),
        budget_usd_per_run=float(data.get("budget_usd_per_run") or 50),
        gpu_hour_usd=float(cost.get("gpu_hour_usd") or 1.20),
        baseline_multiplier=float(cost.get("baseline_multiplier") or 1.35),
        early_stop_enabled=bool(early.get("enabled", True)),
        patience_steps=int(early.get("patience_steps") or 500),
        checkpoint_every_steps=int(ckpt.get("every_steps") or 200),
        preferred_providers=list(
            providers.get("preferred") or ["runpod", "lambda", "aws", "gcp"]
        ),
        allow_spot=bool(providers.get("allow_spot", True)),
        raw=data,
    )
