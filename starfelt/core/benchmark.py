"""Repeatable microbenchmarks for comparing training efficiency.

The benchmark intentionally measures a fixed workload rather than claiming to
represent every model. Its output is designed to become the stable baseline
for later data-pipeline and kernel optimizations.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from starfelt.core.config import StarfeltConfig, load_config
from starfelt.core.gpu import GpuMonitor


@dataclass(frozen=True)
class BenchmarkConfig:
    steps: int = 50
    warmup_steps: int = 5
    repeats: int = 3
    batch_size: int = 32
    input_dim: int = 512
    hidden_dim: int = 512
    device: str = "auto"
    seed: int = 0

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.repeats <= 0:
            raise ValueError("repeats must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be > 0")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


@dataclass
class BenchmarkResult:
    repeat: int
    device: str
    steps: int
    batch_size: int
    elapsed_s: float
    steps_per_s: float
    samples_per_s: float
    final_loss: float
    cost_usd: float
    gpu_name: str | None = None
    gpu_util_avg: float | None = None
    gpu_power_avg_w: float | None = None
    gpu_energy_j: float | None = None
    peak_mem_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    config: BenchmarkConfig
    results: list[BenchmarkResult]

    @property
    def median_elapsed_s(self) -> float:
        return statistics.median(r.elapsed_s for r in self.results)

    @property
    def median_steps_per_s(self) -> float:
        return statistics.median(r.steps_per_s for r in self.results)

    @property
    def median_samples_per_s(self) -> float:
        return statistics.median(r.samples_per_s for r in self.results)

    @property
    def median_cost_usd(self) -> float:
        return statistics.median(r.cost_usd for r in self.results)

    @property
    def median_energy_j(self) -> float | None:
        values = [r.gpu_energy_j for r in self.results if r.gpu_energy_j is not None]
        return statistics.median(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": "synthetic_mlp_v1",
            "config": asdict(self.config),
            "summary": {
                "median_elapsed_s": self.median_elapsed_s,
                "median_steps_per_s": self.median_steps_per_s,
                "median_samples_per_s": self.median_samples_per_s,
                "median_cost_usd": self.median_cost_usd,
                "median_energy_j": self.median_energy_j,
            },
            "results": [result.to_dict() for result in self.results],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(requested)


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_mlp_benchmark(
    config: BenchmarkConfig,
    project_config: StarfeltConfig | None = None,
) -> BenchmarkReport:
    """Run a deterministic MLP benchmark repeatedly and return raw measurements."""
    config.validate()
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch required for benchmark. Install with: pip install torch") from exc

    device = _resolve_device(torch, config.device)
    cfg = project_config or load_config()
    results: list[BenchmarkResult] = []

    for repeat in range(config.repeats):
        torch.manual_seed(config.seed + repeat)
        model = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 10),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(config.batch_size, config.input_dim, device=device)
        y = torch.randint(0, 10, (config.batch_size,), device=device)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(config.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        _synchronize(torch, device)

        monitor = GpuMonitor()
        monitor.poll()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        _synchronize(torch, device)
        elapsed_s = time.perf_counter() - started
        monitor.poll()

        steps_per_s = config.steps / max(elapsed_s, 1e-9)
        peak_mem_mb = None
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        results.append(
            BenchmarkResult(
                repeat=repeat + 1,
                device=device.type,
                steps=config.steps,
                batch_size=config.batch_size,
                elapsed_s=elapsed_s,
                steps_per_s=steps_per_s,
                samples_per_s=steps_per_s * config.batch_size,
                final_loss=float(loss.detach().item()),
                cost_usd=(elapsed_s / 3600.0) * cfg.gpu_hour_usd,
                gpu_name=monitor.last.name if monitor.last else None,
                gpu_util_avg=monitor.average_util(),
                gpu_power_avg_w=monitor.average_power_w(),
                gpu_energy_j=monitor.energy_j if monitor.power_samples else None,
                peak_mem_mb=peak_mem_mb,
            )
        )

    return BenchmarkReport(config=config, results=results)
