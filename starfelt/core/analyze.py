from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from starfelt.core.config import StarfeltConfig


@dataclass
class Check:
    name: str
    level: str  # ok | warn | fail
    detail: str


@dataclass
class AnalysisReport:
    checks: list[Check]
    est_hours: float
    est_cost_usd: float
    est_cost_optimized_usd: float


def analyze_script(script: Path, cfg: StarfeltConfig) -> AnalysisReport:
    text = script.read_text(encoding="utf-8", errors="replace")
    checks: list[Check] = []

    batch = _find_number(text, r"batch_size\s*=\s*(\d+)")
    lr = _find_float(text, r"lr\s*=\s*([0-9.eE+-]+)") or _find_float(
        text, r"learning_rate\s*=\s*([0-9.eE+-]+)"
    )
    epochs = _find_number(text, r"epochs\s*=\s*(\d+)") or _find_number(
        text, r"num_epochs\s*=\s*(\d+)"
    )

    if batch is None:
        checks.append(
            Check("batch_size", "warn", "No batch_size found — confirm dataloader settings")
        )
    elif batch < 8:
        checks.append(
            Check(
                "batch_size",
                "warn",
                f"batch_size={batch} may under-utilize GPU; consider grad accumulation",
            )
        )
    elif batch > 512:
        checks.append(
            Check(
                "batch_size",
                "warn",
                f"batch_size={batch} is large — risk of OOM or noisy steps",
            )
        )
    else:
        checks.append(Check("batch_size", "ok", f"batch_size={batch}"))

    if lr is None:
        checks.append(Check("learning_rate", "warn", "No lr / learning_rate literal found"))
    elif lr > 0.1:
        checks.append(
            Check("learning_rate", "fail", f"lr={lr} is very high — likely wasted runs")
        )
    elif lr < 1e-6:
        checks.append(
            Check("learning_rate", "warn", f"lr={lr} is extremely small — slow convergence risk")
        )
    else:
        checks.append(Check("learning_rate", "ok", f"lr={lr}"))

    if "DataLoader" in text or "dataloader" in text.lower():
        if "num_workers" in text and re.search(r"num_workers\s*=\s*0", text):
            checks.append(
                Check(
                    "data_pipeline",
                    "warn",
                    "num_workers=0 — GPU may idle on data; enable workers + prefetch",
                )
            )
        else:
            checks.append(Check("data_pipeline", "ok", "DataLoader usage detected"))
    else:
        checks.append(
            Check("data_pipeline", "warn", "No DataLoader pattern detected in script")
        )

    try:
        ast.parse(text)
        checks.append(Check("syntax", "ok", "Python parses cleanly"))
    except SyntaxError as e:
        checks.append(Check("syntax", "fail", f"SyntaxError: {e.msg}"))

    # crude time/cost sketch
    ep = epochs or 10
    est_hours = max(0.1, ep * 0.15 * (1.0 if (batch or 32) >= 16 else 1.4))
    est_cost = est_hours * cfg.gpu_hour_usd
    est_opt = est_cost / cfg.baseline_multiplier

    checks.append(
        Check(
            "budget",
            "ok" if est_cost <= cfg.budget_usd_per_run else "warn",
            f"est ${est_cost:.2f} vs budget ${cfg.budget_usd_per_run:.2f}",
        )
    )

    return AnalysisReport(
        checks=checks,
        est_hours=est_hours,
        est_cost_usd=est_cost,
        est_cost_optimized_usd=est_opt,
    )


def _find_number(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _find_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
