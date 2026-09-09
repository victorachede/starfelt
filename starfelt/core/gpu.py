"""GPU utilization sampling via nvidia-smi (optional)."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class GpuSample:
    util_pct: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None
    name: str | None = None
    power_w: float | None = None


@dataclass
class GpuMonitor:
    samples: list[float] = field(default_factory=list)
    power_samples: list[float] = field(default_factory=list)
    last: GpuSample | None = None
    available: bool = False
    energy_j: float = 0.0
    _last_sample_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.available = shutil.which("nvidia-smi") is not None

    def poll(self) -> GpuSample | None:
        if not self.available:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw",
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
        if len(parts) < 5:
            return None
        try:
            name = parts[0]
            util = float(parts[1])
            mem_u = float(parts[2])
            mem_t = float(parts[3])
        except ValueError:
            return None
        power: float | None
        try:
            power = float(parts[4])
        except ValueError:
            power = None
        if power is not None:
            now = time.monotonic()
            if self._last_sample_at is not None:
                self.energy_j += power * max(0.0, now - self._last_sample_at)
            self._last_sample_at = now
        sample = GpuSample(
            util_pct=util,
            mem_used_mb=mem_u,
            mem_total_mb=mem_t,
            name=name,
            power_w=power,
        )
        self.last = sample
        if util is not None:
            self.samples.append(util)
        if power is not None:
            self.power_samples.append(power)
        return sample

    def average_util(self) -> float | None:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)

    def average_power_w(self) -> float | None:
        if not self.power_samples:
            return None
        return sum(self.power_samples) / len(self.power_samples)
