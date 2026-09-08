from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from starfelt.core.analyze import AnalysisReport, Check, analyze_script
from starfelt.core.config import StarfeltConfig
from starfelt.core.cost import CostTracker, write_active_run, clear_active_run

console = Console(stderr=True)


@dataclass
class RunResult:
    run_id: str
    exit_code: int
    duration_s: float
    cost_usd: float
    dry_run: bool = False
    checks: list[Check] = field(default_factory=list)


def _format_elapsed(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _print_preflight(report: AnalysisReport) -> None:
    warns = [c for c in report.checks if c.level == "warn"]
    fails = [c for c in report.checks if c.level == "fail"]
    if not warns and not fails:
        console.print("[dim]Pre-run analysis[/] · all checks ok")
        return
    console.print("[bold]Pre-run analysis[/]")
    for c in fails:
        console.print(f"  [red]✗ fail[/]  {c.name}: {c.detail}")
    for c in warns:
        console.print(f"  [yellow]⚠ warn[/]  {c.name}: {c.detail}")
    if fails:
        console.print(
            "[yellow]Launching anyway[/] — fix these when you can; "
            "failed runs will re-show them."
        )


def _print_failed_checks(checks: list[Check]) -> None:
    bad = [c for c in checks if c.level in ("warn", "fail")]
    if not bad:
        return
    console.print()
    console.print("[bold red]Analysis flags (revisit these)[/]")
    for c in bad:
        color = "red" if c.level == "fail" else "yellow"
        console.print(f"  [{color}]{c.level}[/] · {c.name}: {c.detail}")


def run_wrapped(
    script: Path,
    script_args: list[str],
    cfg: StarfeltConfig,
    dry_run: bool = False,
) -> RunResult:
    """Execute training script with streamed output + live cost ticker."""
    report = analyze_script(script, cfg)
    _print_preflight(report)

    if dry_run:
        return RunResult(
            run_id="dry-run",
            exit_code=0,
            duration_s=0,
            cost_usd=0,
            dry_run=True,
            checks=report.checks,
        )

    tracker = CostTracker(cfg, str(script))
    env = os.environ.copy()
    env["STARFELT_RUN_ID"] = tracker.run_id
    env["STARFELT_EARLY_STOP"] = "1" if cfg.early_stop_enabled else "0"
    env["STARFELT_CHECKPOINT_EVERY"] = str(cfg.checkpoint_every_steps)
    env["STARFELT_EST_COST_USD"] = f"{report.est_cost_usd:.4f}"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, "-u", str(script), *script_args]
    console.print(
        f"[bold cyan]▶[/] [cyan]starfelt run[/] {script.name}  "
        f"[dim]id={tracker.run_id[:8]}[/]"
    )
    console.print()

    write_active_run(
        {
            "run_id": tracker.run_id,
            "script": str(script),
            "started_at": time.time(),
            "gpu_hour_usd": cfg.gpu_hour_usd,
        }
    )

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    assert proc.stdout is not None
    t0 = time.time()
    last_tick = 0.0
    # binary read for select compatibility
    leftover = b""

    try:
        while True:
            # live ticker every ~0.25s even if no output
            now = time.time()
            if now - last_tick >= 0.25:
                elapsed = now - t0
                cost = (elapsed / 3600.0) * cfg.gpu_hour_usd
                tick = f"⏱ {_format_elapsed(elapsed)}  ·  ${cost:.4f}"
                console.print(f"\r[dim]{tick}[/]", end="", highlight=False)
                last_tick = now

            if proc.poll() is not None:
                # drain remaining
                rest = proc.stdout.read()
                if rest:
                    leftover += rest
                break

            ready, _, _ = select.select([proc.stdout], [], [], 0.25)
            if not ready:
                continue
            chunk = proc.stdout.read(4096)
            if not chunk:
                if proc.poll() is not None:
                    break
                continue
            leftover += chunk
            while b"\n" in leftover:
                line, leftover = leftover.split(b"\n", 1)
                text = line.decode("utf-8", errors="replace")
                # clear ticker line then print process output
                console.print("\r" + " " * 40 + "\r", end="")
                # child output to stdout (user-facing)
                sys.stdout.write(text + "\n")
                sys.stdout.flush()

        if leftover.strip():
            console.print("\r" + " " * 40 + "\r", end="")
            sys.stdout.write(leftover.decode("utf-8", errors="replace"))
            if not leftover.endswith(b"\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
    finally:
        clear_active_run()
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # final newline after ticker
    console.print("\r" + " " * 40 + "\r", end="")
    exit_code = proc.returncode if proc.returncode is not None else 1
    row = tracker.finish(exit_code)

    if exit_code != 0:
        _print_failed_checks(report.checks)

    return RunResult(
        run_id=tracker.run_id,
        exit_code=exit_code,
        duration_s=row["duration_s"],
        cost_usd=row["cost_usd"],
        dry_run=False,
        checks=report.checks,
    )
