"""Environment diagnostics for `starfelt doctor`."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: float = 3.0) -> str | None:
    try:
        out = subprocess.check_output(
            cmd, text=True, timeout=timeout, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None


def collect_doctor_rows() -> list[tuple[str, str, str]]:
    """Return (check, ok|warn|fail, detail)."""
    rows: list[tuple[str, str, str]] = []

    major, minor, patch = sys.version_info[:3]
    py_ok = (major, minor) >= (3, 10)
    rows.append(
        (
            "python",
            "ok" if py_ok else "fail",
            f"{major}.{minor}.{patch}" + ("" if py_ok else " (need >= 3.10)"),
        )
    )

    for pkg in ("click", "rich", "yaml", "psutil"):
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
            rows.append((pkg if pkg != "yaml" else "pyyaml", "ok", "importable"))
        except ImportError:
            rows.append((pkg if pkg != "yaml" else "pyyaml", "fail", "not installed"))

    smi = shutil.which("nvidia-smi")
    if not smi:
        rows.append(("nvidia-smi", "warn", "not found — GPU util disabled"))
    else:
        q = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        )
        rows.append(("nvidia-smi", "ok", q.split("\n")[0] if q else "present"))

    cuda = _run(["nvcc", "--version"])
    if cuda:
        line = next((ln for ln in cuda.splitlines() if "release" in ln.lower()), cuda.splitlines()[-1])
        rows.append(("cuda", "ok", line.strip()))
    else:
        rows.append(("cuda", "warn", "nvcc not found (optional)"))

    try:
        import torch

        rows.append(
            (
                "torch",
                "ok",
                f"{torch.__version__} · cuda={torch.cuda.is_available()}",
            )
        )
    except ImportError:
        rows.append(("torch", "warn", "not installed (ok for non-PyTorch scripts)"))

    yaml_path = Path.cwd() / "starfelt.yaml"
    if yaml_path.exists():
        rows.append(("starfelt.yaml", "ok", str(yaml_path)))
    else:
        rows.append(("starfelt.yaml", "warn", "missing — run starfelt init"))

    starfelt_dir = Path.cwd() / ".starfelt"
    try:
        starfelt_dir.mkdir(exist_ok=True)
        probe = starfelt_dir / ".doctor_write"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        rows.append((".starfelt/", "ok", "writable"))
    except OSError as e:
        rows.append((".starfelt/", "fail", str(e)))

    auth = Path.home() / ".starfelt" / "auth.json"
    if auth.exists():
        rows.append(("hosted auth", "ok", str(auth)))
    else:
        rows.append(("hosted auth", "warn", "not logged in (local history only)"))

    return rows
