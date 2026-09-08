from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from starfelt.core.analyze import AnalysisReport, Check, analyze_script
from starfelt.core.config import StarfeltConfig
from starfelt.core.cost import CostTracker, clear_active_run, write_active_run

console = Console(stderr=True)

# Sentinel for reader thread end
_STDOUT_DONE = object()


@dataclass
class RunResult:
    run_id: str
    exit_code: int
    duration_s: float
    cost_usd: float
    dry_run: bool = False
    checks: list[Check] = field(default_factory=list)
    aborted: bool = False


def _format_elapsed(seconds: float) -> str:
    s = int(max(0, seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_preflight_table(report: AnalysisReport) -> None:
    """Same shape as `starfelt analyze` — intentional preflight."""
    table = Table(title="Pre-run analysis", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for item in report.checks:
        style = {"ok": "green", "warn": "yellow", "fail": "red"}.get(item.level, "white")
        table.add_row(item.name, f"[{style}]{item.level}[/]", item.detail)
    console.print(table)


def _print_failed_checks(checks: list[Check]) -> None:
    bad = [c for c in checks if c.level in ("warn", "fail")]
    if not bad:
        return
    console.print()
    console.print("[bold red]Analysis flags (revisit these)[/]")
    for c in bad:
        color = "red" if c.level == "fail" else "yellow"
        console.print(f"  [{color}]{c.level}[/] · {c.name}: {c.detail}")


def _reader_thread(pipe, q: queue.Queue) -> None:
    """Read child stdout line-by-line; works on Windows (no select)."""
    try:
        # Text mode IO wrapper
        for line in iter(pipe.readline, ""):
            q.put(line)
    except Exception as e:  # noqa: BLE001
        q.put(f"[starfelt] stdout reader error: {e}\n")
    finally:
        try:
            pipe.close()
        except Exception:  # noqa: BLE001
            pass
        q.put(_STDOUT_DONE)


def run_wrapped(
    script: Path,
    script_args: list[str],
    cfg: StarfeltConfig,
    dry_run: bool = False,
    force: bool = False,
    confirm_fails: bool = True,
) -> RunResult:
    """Execute training script with streamed output + live cost ticker.

    Cross-platform: thread + queue for stdout (no select).
    """
    report = analyze_script(script, cfg)
    print_preflight_table(report)

    fails = [c for c in report.checks if c.level == "fail"]
    if fails and confirm_fails and not force and not dry_run:
        n = len(fails)
        label = "issue" if n == 1 else "issues"
        console.print()
        console.print(
            f"[bold red]{n} critical {label} found.[/] "
            "Run anyway? [y/N]",
            end=" ",
        )
        try:
            answer = input().strip().lower()
        except EOFError:
            answer = "n"
        if answer not in {"y", "yes"}:
            console.print("[yellow]Aborted[/] — fix fails or pass --force.")
            return RunResult(
                run_id="aborted",
                exit_code=2,
                duration_s=0,
                cost_usd=0,
                dry_run=False,
                checks=report.checks,
                aborted=True,
            )

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
    console.print()
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
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    q: queue.Queue = queue.Queue()
    assert proc.stdout is not None
    t = threading.Thread(target=_reader_thread, args=(proc.stdout, q), daemon=True)
    t.start()

    t0 = time.time()
    last_tick = 0.0

    try:
        done_reading = False
        while True:
            now = time.time()
            if now - last_tick >= 0.25:
                elapsed = now - t0
                cost = (elapsed / 3600.0) * cfg.gpu_hour_usd
                tick = f"⏱ {_format_elapsed(elapsed)}  ·  ${cost:.4f}"
                console.print(f"\r[dim]{tick}[/]", end="", highlight=False)
                last_tick = now

            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None and done_reading:
                    break
                if proc.poll() is not None and q.empty():
                    # process ended; wait a beat for trailing lines
                    try:
                        item = q.get(timeout=0.1)
                    except queue.Empty:
                        done_reading = True
                        if not t.is_alive():
                            break
                        continue
                else:
                    continue

            if item is _STDOUT_DONE:
                done_reading = True
                if proc.poll() is not None:
                    break
                continue

            line = item if isinstance(item, str) else str(item)
            console.print("\r" + " " * 48 + "\r", end="")
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
            sys.stdout.flush()

        t.join(timeout=2.0)
    finally:
        clear_active_run()
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    console.print("\r" + " " * 48 + "\r", end="")
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
        aborted=False,
    )
