"""GPU utilization sampling via nvidia-smi (optional)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GpuSample:
    util_pct: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None
    name: str | None = None


@dataclass
class GpuMonitor:
    samples: list[float] = field(default_factory=list)
    last: GpuSample | None = None
    available: bool = False

    def __post_init__(self) -> None:
        self.available = shutil.which("nvidia-smi") is not None

    def poll(self) -> GpuSample | None:
        if not self.available:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            self.available = False
            return None
        line = out.strip().splitlines()[0] if out.strip() else ""
        if not line:
            return None
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None
        try:
            name, util, mem_u, mem_t = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            return None
        sample = GpuSample(util_pct=util, mem_used_mb=mem_u, mem_total_mb=mem_t, name=name)
        self.last = sample
        if util is not None:
            self.samples.append(util)
        return sample

    def average_util(self) -> float | None:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)
