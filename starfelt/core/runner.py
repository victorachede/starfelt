from __future__ import annotations

import os
import queue
import signal
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
from starfelt.core.cost import (
    CostTracker,
    clear_active_run,
    write_active_run,
    write_interrupted_marker,
)
from starfelt.core.gpu import GpuMonitor
from starfelt.core.model_size import estimate_model_params

console = Console(stderr=True)
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
    gpu_util_avg: float | None = None
    interrupted: bool = False
    budget_exceeded: bool = False
    workload_id: str | None = None
    framework: str | None = None


def _format_elapsed(seconds: float) -> str:
    s = int(max(0, seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_preflight_table(report: AnalysisReport) -> None:
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
    try:
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
    enforce_budget: bool = True,
) -> RunResult:
    report = analyze_script(script, cfg)
    print_preflight_table(report)

    fails = [c for c in report.checks if c.level == "fail"]
    if fails and confirm_fails and not force and not dry_run:
        n = len(fails)
        label = "issue" if n == 1 else "issues"
        console.print()
        console.print(
            f"[bold red]{n} critical {label} found.[/] Run anyway? [y/N]",
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
    env["STARFELT_PATIENCE_STEPS"] = str(cfg.patience_steps)
    env["STARFELT_EST_COST_USD"] = f"{report.est_cost_usd:.4f}"
    env["STARFELT_BUDGET_USD"] = f"{cfg.budget_usd_per_run:.4f}"
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
            "budget_usd": cfg.budget_usd_per_run if enforce_budget else None,
        }
    )

    gpu = GpuMonitor()
    if gpu.available:
        console.print("[dim]GPU monitoring via nvidia-smi[/]")
    else:
        console.print("[dim]No nvidia-smi — GPU util not tracked this run[/]")

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

    interrupted = {"sig": None}

    def _on_signal(signum: int, _frame) -> None:
        name = signal.Signals(signum).name
        interrupted["sig"] = name
        elapsed = time.time() - tracker.t0
        cost = (elapsed / 3600.0) * cfg.gpu_hour_usd
        write_interrupted_marker(
            tracker.run_id,
            script=str(script),
            elapsed_s=elapsed,
            cost_usd=cost,
            signal_name=name,
        )
        console.print(
            f"\n[yellow]Caught {name}[/] — marker "
            f".starfelt/runs/{tracker.run_id}_interrupted.json"
        )
        try:
            if proc.poll() is None:
                proc.send_signal(signum)
        except OSError:
            pass

    prev_int = signal.signal(signal.SIGINT, _on_signal)
    prev_term = signal.signal(signal.SIGTERM, _on_signal)

    q: queue.Queue = queue.Queue()
    assert proc.stdout is not None
    t = threading.Thread(target=_reader_thread, args=(proc.stdout, q), daemon=True)
    t.start()

    t0 = time.time()
    last_tick = 0.0
    last_gpu = 0.0
    budget_exceeded = False
    budget_stop_sent_at: float | None = None

    try:
        done_reading = False
        while True:
            now = time.time()
            elapsed = now - t0
            cost = (elapsed / 3600.0) * cfg.gpu_hour_usd
            if now - last_gpu >= 2.0:
                gpu.poll()
                last_gpu = now
            if now - last_tick >= 0.25:
                util = gpu.last.util_pct if gpu.last else None
                util_s = f"  ·  GPU {util:.0f}%" if util is not None else ""
                tick = f"⏱ {_format_elapsed(elapsed)}  ·  ${cost:.4f}{util_s}"
                console.print(f"\r[dim]{tick}[/]", end="", highlight=False)
                last_tick = now

            if (
                enforce_budget
                and cfg.budget_usd_per_run > 0
                and cost >= cfg.budget_usd_per_run
                and proc.poll() is None
            ):
                if budget_stop_sent_at is None:
                    budget_exceeded = True
                    budget_stop_sent_at = now
                    console.print(
                        f"\n[yellow]Budget limit reached (${cfg.budget_usd_per_run:.2f})[/] "
                        "— asking the training process to stop safely."
                    )
                    try:
                        proc.send_signal(signal.SIGTERM)
                    except OSError:
                        pass
                elif now - budget_stop_sent_at >= 10.0:
                    console.print(
                        "[yellow]Training did not stop after 10s; terminating it.[/]"
                    )
                    proc.kill()

            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None and done_reading:
                    break
                if proc.poll() is not None and q.empty():
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
            console.print("\r" + " " * 56 + "\r", end="")
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
            sys.stdout.flush()

        t.join(timeout=2.0)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        clear_active_run()
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    console.print("\r" + " " * 56 + "\r", end="")
    exit_code = proc.returncode if proc.returncode is not None else 1
    avg = gpu.average_util()
    model_params = estimate_model_params(script)
    # Costs computed inside tracker.finish, then build_run_telemetry with real values
    row = tracker.finish(
        exit_code,
        hints=report.telemetry,
        model_param_count=model_params,
        gpu_name=(gpu.last.name if gpu.last else None),
        gpu_util_avg=avg,
        gpu_samples=len(gpu.samples),
        gpu_power_avg_w=gpu.average_power_w(),
        gpu_energy_j=gpu.energy_j if gpu.power_samples else None,
        interrupted=interrupted["sig"],
        budget_exceeded=budget_exceeded,
        budget_usd=cfg.budget_usd_per_run if enforce_budget else None,
    )

    if exit_code != 0:
        _print_failed_checks(report.checks)

    return RunResult(
        run_id=tracker.run_id,
        exit_code=exit_code,
        duration_s=row["duration_s"],
        cost_usd=row["cost_usd"],
        checks=report.checks,
        gpu_util_avg=avg,
        interrupted=interrupted["sig"] is not None,
        budget_exceeded=budget_exceeded,
        workload_id=row.get("workload_id"),
        framework=row.get("framework"),
    )
