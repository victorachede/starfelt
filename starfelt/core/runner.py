from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from starfelt.core.analyze import analyze_script
from starfelt.core.config import StarfeltConfig
from starfelt.core.cost import CostTracker


@dataclass
class RunResult:
    run_id: str
    exit_code: int
    duration_s: float
    cost_usd: float
    dry_run: bool = False


def run_wrapped(
    script: Path,
    script_args: list[str],
    cfg: StarfeltConfig,
    dry_run: bool = False,
) -> RunResult:
    """Execute training script as a child process with cost tracking.

    Stage 1: wrap + measure. Runtime auto-tune hooks land as SDK signals.
    """
    report = analyze_script(script, cfg)
    if dry_run:
        return RunResult(run_id="dry-run", exit_code=0, duration_s=0, cost_usd=0, dry_run=True)

    tracker = CostTracker(cfg, str(script))
    env = os.environ.copy()
    env["STARFELT_RUN_ID"] = tracker.run_id
    env["STARFELT_EARLY_STOP"] = "1" if cfg.early_stop_enabled else "0"
    env["STARFELT_CHECKPOINT_EVERY"] = str(cfg.checkpoint_every_steps)

    # Surface analysis hints to the process (optional for user scripts)
    env["STARFELT_EST_COST_USD"] = f"{report.est_cost_usd:.4f}"

    cmd = [sys.executable, str(script), *script_args]
    proc = subprocess.run(cmd, env=env)
    row = tracker.finish(proc.returncode)
    return RunResult(
        run_id=tracker.run_id,
        exit_code=proc.returncode,
        duration_s=row["duration_s"],
        cost_usd=row["cost_usd"],
        dry_run=False,
    )
