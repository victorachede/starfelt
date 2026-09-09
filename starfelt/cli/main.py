"""Starfelt CLI entrypoint."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from starfelt import __version__
from starfelt.core.analyze import analyze_script
from starfelt.core.benchmark import BenchmarkConfig, run_mlp_benchmark
from starfelt.core.config import init_project, load_config, validate_environment
from starfelt.core.cost import load_history, read_active_run
from starfelt.core.runner import run_wrapped

console = Console()


@click.group()
@click.version_option(__version__, prog_name="starfelt")
def cli() -> None:
    """Starfelt — cheaper, smarter training runs. No rewrite."""


@cli.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing starfelt.yaml")
def init_cmd(force: bool) -> None:
    """Create starfelt.yaml and validate the local environment."""
    path = init_project(Path.cwd(), force=force)
    console.print(f"[bold green]✓[/] Wrote {path}")

    rows = validate_environment()
    table = Table(title="Environment", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, status, detail in rows:
        style = "green" if status == "ok" else "red"
        table.add_row(name, f"[{style}]{status}[/]", detail)
    console.print(table)

    if any(s == "fail" for _, s, _ in rows):
        console.print(
            "[yellow]Fix failed checks before relying on cost/GPU metrics.[/]"
        )
    else:
        console.print("Next: [cyan]starfelt analyze examples/train_toy.py[/]")


@cli.command("analyze")
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_path", default=None, type=click.Path(dir_okay=False))
def analyze_cmd(script: str, config_path: str | None) -> None:
    """Pre-run analysis: batch size, LR risk, data bottlenecks, cost estimate."""
    cfg = load_config(Path(config_path) if config_path else None)
    report = analyze_script(Path(script), cfg)

    table = Table(title="Pre-run analysis", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for item in report.checks:
        style = {"ok": "green", "warn": "yellow", "fail": "red"}.get(item.level, "white")
        table.add_row(item.name, f"[{style}]{item.level}[/]", item.detail)
    console.print(table)

    console.print(
        Panel(
            f"Est. time: [bold]{report.est_hours:.1f}h[/]\n"
            f"Est. cost (current): [bold]${report.est_cost_usd:.2f}[/]\n"
            f"Est. cost (optimized): [bold]${report.est_cost_optimized_usd:.2f}[/]\n"
            f"Potential save: [bold green]${report.est_cost_usd - report.est_cost_optimized_usd:.2f}[/]",
            title="Cost sketch",
            border_style="cyan",
        )
    )


@cli.command("run")
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
@click.option("--config", "config_path", default=None, type=click.Path(dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Analyze + plan only; do not execute")
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmation when analysis has fail-level checks",
)
def run_cmd(
    script: str,
    script_args: tuple[str, ...],
    config_path: str | None,
    dry_run: bool,
    force: bool,
) -> None:
    """Wrap and run a training script with Starfelt monitoring."""
    cfg = load_config(Path(config_path) if config_path else None)
    result = run_wrapped(
        Path(script),
        list(script_args),
        cfg,
        dry_run=dry_run,
        force=force,
        confirm_fails=not force,
    )
    if result.aborted:
        raise SystemExit(2)
    if result.dry_run:
        console.print("[yellow]Dry run[/] — no process started.")
        return
    console.print()
    lines = [
        f"Exit code: {result.exit_code}",
        f"Duration: {result.duration_s:.1f}s",
        f"Tracked cost: ${result.cost_usd:.4f}",
    ]
    if result.gpu_util_avg is not None:
        lines.append(f"Avg GPU util: {result.gpu_util_avg:.0f}%")
    if result.interrupted:
        lines.append("Interrupted: marker written under .starfelt/runs/")
    if result.framework:
        lines.append(f"Framework: {result.framework}")
    if result.workload_id:
        lines.append(f"Workload: {result.workload_id}")
    lines.append(f"Run id: {result.run_id}")
    console.print(
        Panel(
            "\n".join(lines),
            title="Run complete",
            border_style="green" if result.exit_code == 0 else "red",
        )
    )


@cli.command("cost")
@click.option("--json", "as_json", is_flag=True)
def cost_cmd(as_json: bool) -> None:
    """Show local run cost history.

    For real savings between two runs, use: starfelt compare run_a run_b
    """
    history = load_history()
    if as_json:
        click.echo(json.dumps(history, indent=2))
        return
    if not history:
        console.print("No runs recorded yet. [cyan]starfelt run …[/] first.")
        return
    table = Table(title="Cost history", header_style="bold")
    table.add_column("Run")
    table.add_column("Script")
    table.add_column("Duration")
    table.add_column("Cost")
    for row in history[-20:]:
        cost = float(row.get("cost_usd", 0) or 0)
        table.add_row(
            str(row.get("run_id", "?"))[:8],
            str(row.get("script", "")),
            f"{row.get('duration_s', 0):.1f}s",
            f"${cost:.4f}",
        )
    console.print(table)
    total = sum(float(r.get("cost_usd", 0) or 0) for r in history)
    console.print(f"Total tracked: [bold]${total:.4f}[/]")
    console.print("[dim]Compare two runs: [cyan]starfelt compare run_a run_b[/][/]")


def _status_renderable():
    active = read_active_run()
    history = load_history()
    from rich.console import Group

    parts = []
    if active:
        started = float(active.get("started_at") or time.time())
        elapsed = max(0.0, time.time() - started)
        rate = float(active.get("gpu_hour_usd") or 1.2)
        cost = (elapsed / 3600.0) * rate
        parts.append(
            Panel(
                f"Run id: {active.get('run_id', '?')}\n"
                f"Script: {active.get('script', '?')}\n"
                f"Elapsed: {elapsed:.0f}s\n"
                f"Live cost: ${cost:.4f}",
                title="Active run",
                border_style="cyan",
            )
        )
    else:
        parts.append(Panel("No active run", border_style="dim"))

    table = Table(title="Last 5 runs", header_style="bold")
    table.add_column("Run")
    table.add_column("Script")
    table.add_column("Duration")
    table.add_column("Cost")
    table.add_column("Exit")
    if not history:
        table.add_row("—", "no runs yet", "—", "—", "—")
    else:
        for row in history[-5:]:
            table.add_row(
                str(row.get("run_id", "?"))[:8],
                str(row.get("script", "")),
                f"{row.get('duration_s', 0):.1f}s",
                f"${row.get('cost_usd', 0):.4f}",
                str(row.get("exit_code", "")),
            )
    parts.append(table)
    total = sum(r.get("cost_usd", 0) for r in history)
    parts.append(Panel(f"Total spend tracked: ${total:.4f}", border_style="green"))
    return Group(*parts)


@cli.command("status")
@click.option(
    "--watch",
    "-w",
    is_flag=True,
    help="Refresh every second (live elapsed + cost)",
)
def status_cmd(watch: bool) -> None:
    """Active run (if any), last 5 runs, total spend."""
    if not watch:
        console.print(_status_renderable())
        return

    console.print("[dim]Watching status · Ctrl+C to stop[/]")
    try:
        with Live(_status_renderable(), console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(1)
                live.update(_status_renderable())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching[/]")



@cli.group("providers")
def providers_group() -> None:
    """Compute price catalog (estimates)."""


@providers_group.command("list")
def providers_list() -> None:
    """Show static GPU catalog + price disclaimer."""
    from starfelt.providers.base import CATALOG, CATALOG_WARNING, PRICES_LAST_UPDATED

    console.print(f"[yellow]{CATALOG_WARNING}[/]")
    console.print(f"[dim]PRICES_LAST_UPDATED = {PRICES_LAST_UPDATED}[/]")
    table = Table(title="Providers (estimates)", header_style="bold")
    table.add_column("Provider")
    table.add_column("GPU")
    table.add_column("$/hr")
    table.add_column("Spot")
    for o in CATALOG:
        table.add_row(o.name, o.gpu, f"{o.usd_per_hour:.2f}", "yes" if o.spot else "no")
    console.print(table)



@cli.command("doctor")
def doctor_cmd() -> None:
    """Full environment check — first stop when something breaks."""
    from starfelt.core.doctor import collect_doctor_rows

    rows = collect_doctor_rows()
    table = Table(title="starfelt doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, status, detail in rows:
        style = {"ok": "green", "warn": "yellow", "fail": "red"}.get(status, "white")
        table.add_row(name, f"[{style}]{status}[/]", detail)
    console.print(table)
    fails = sum(1 for _, s, _ in rows if s == "fail")
    warns = sum(1 for _, s, _ in rows if s == "warn")
    if fails:
        console.print(f"[red]{fails} failed[/] · fix these before relying on runs")
        raise SystemExit(1)
    if warns:
        console.print(f"[yellow]{warns} warning(s)[/] · optional / non-blocking")
    else:
        console.print("[green]All checks passed[/]")


@cli.command("login")
@click.option("--url", prompt="Supabase URL", help="https://xxxx.supabase.co")
@click.option("--key", prompt=True, hide_input=True, help="Supabase anon or service key")
def login_cmd(url: str, key: str) -> None:
    """Opt-in hosted history — store credentials in ~/.starfelt/auth.json."""
    from starfelt.core.hosted import save_auth

    path = save_auth({"supabase_url": url.strip().rstrip("/"), "supabase_key": key.strip()})
    console.print(f"[green]✓[/] Saved credentials → {path}")
    console.print("Create table [cyan]starfelt_runs[/] (see docs/HOSTED.md), then: [cyan]starfelt sync[/]")


@cli.command("logout")
def logout_cmd() -> None:
    """Remove ~/.starfelt/auth.json."""
    from starfelt.core.hosted import auth_path, clear_auth

    clear_auth()
    console.print(f"[dim]Cleared[/] {auth_path()}")


@cli.command("sync")
def sync_cmd() -> None:
    """Push local .starfelt history to Supabase (requires login)."""
    from starfelt.core.cost import load_history
    from starfelt.core.hosted import sync_runs

    n, msg = sync_runs(load_history())
    if n:
        console.print(f"[green]{msg}[/]")
    else:
        console.print(f"[yellow]{msg}[/]")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Batch 5 — inspect / compare / resume / benchmark / config doctor
# ---------------------------------------------------------------------------


def _find_run(run_id: str) -> dict | None:
    """Resolve full or prefix run_id from history or individual run files."""
    history = load_history()
    for row in reversed(history):
        rid = str(row.get("run_id", ""))
        if rid == run_id or rid.startswith(run_id):
            return row
    runs_dir = Path.cwd() / ".starfelt" / "runs"
    if runs_dir.exists():
        for p in runs_dir.glob("*.json"):
            if p.stem == run_id or p.stem.startswith(run_id):
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
    return None


@cli.command("inspect")
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, help="Dump full telemetry JSON")
def inspect_cmd(run_id: str, as_json: bool) -> None:
    """Deep dive into a single run — telemetry, loss curve, cost, recommendations."""
    row = _find_run(run_id)
    if row is None:
        console.print(f"[red]No run found matching[/] {run_id}")
        raise SystemExit(1)

    if as_json:
        click.echo(json.dumps(row, indent=2))
        return

    rid = str(row.get("run_id", "?"))
    console.print(Panel(f"[bold]{rid}[/]", title="Run", border_style="cyan"))

    table = Table(show_header=False, box=None)
    table.add_column("k", style="dim")
    table.add_column("v")
    table.add_row("Script", str(row.get("script", "—")))
    table.add_row("Framework", str(row.get("framework", "—")))
    table.add_row("Duration", f"{row.get('duration_s', 0):.1f}s")
    table.add_row("Cost", f"${float(row.get('cost_usd') or 0):.4f}")
    table.add_row("Exit", str(row.get("exit_code", "—")))
    table.add_row("Workload", str(row.get("workload_id", "—")))
    if row.get("interrupted"):
        table.add_row("Interrupted", str(row["interrupted"]))
    if row.get("gpu_name"):
        util = row.get("gpu_util_avg")
        util_s = f" · avg util {util:.0f}%" if util is not None else ""
        table.add_row("GPU", f"{row['gpu_name']}{util_s}")
    if row.get("batch_size") is not None:
        table.add_row("Batch size", str(row["batch_size"]))
    if row.get("epochs") is not None:
        table.add_row("Epochs", str(row["epochs"]))
    if row.get("learning_rate") is not None:
        table.add_row("LR", f"{row['learning_rate']}")
    if row.get("optimization_flags"):
        table.add_row("Opts", ", ".join(row["optimization_flags"]))
    if row.get("source"):
        table.add_row("Source", str(row["source"]))
    console.print(table)

    history = row.get("epoch_history") or []
    if history:
        console.print()
        ep_table = Table(title="Per-epoch telemetry", header_style="bold")
        ep_table.add_column("Epoch")
        ep_table.add_column("Loss")
        ep_table.add_column("Val")
        ep_table.add_column("LR")
        ep_table.add_column("smp/s")
        ep_table.add_column("Duration")
        ep_table.add_column("Cost")
        ep_table.add_column("GPU %")
        ep_table.add_column("Mem MB")
        for e in history:
            ep_table.add_row(
                str(e.get("epoch", "")),
                f"{e.get('loss', 0):.4f}",
                f"{e.get('val_loss'):.4f}" if e.get("val_loss") is not None else "—",
                f"{e.get('lr'):.2e}" if e.get("lr") is not None else "—",
                f"{e.get('samples_per_sec'):.0f}" if e.get("samples_per_sec") is not None else "—",
                f"{e.get('duration_s', 0):.1f}s",
                f"${e.get('cost_usd', 0):.4f}",
                f"{e.get('gpu_util'):.0f}" if e.get("gpu_util") is not None else "—",
                f"{e.get('peak_mem_mb'):.0f}" if e.get("peak_mem_mb") is not None else "—",
            )
        console.print(ep_table)
    if row.get("best_val_loss") is not None:
        console.print(f"[dim]Best val loss: {row['best_val_loss']:.4f}[/]")

    console.print()
    flags = row.get("optimization_flags") or []
    recs = []
    if "amp" not in flags and row.get("framework") == "pytorch":
        recs.append("Enable mixed precision (AMP) — often 1.3–1.8× throughput")
    if (row.get("gpu_util_avg") or 100) < 40:
        recs.append("Low GPU util — check DataLoader num_workers / prefetch")
    if row.get("batch_size") and int(row["batch_size"]) < 16:
        recs.append("Small batch size — try larger if memory allows")
    if not recs:
        recs.append("No obvious extra wins from static signals")
    console.print(
        Panel(
            "\n".join(f"• {r}" for r in recs),
            title="What Starfelt would try next",
            border_style="yellow",
        )
    )


@cli.command("compare")
@click.argument("run_id_1")
@click.argument("run_id_2")
def compare_cmd(run_id_1: str, run_id_2: str) -> None:
    """Side-by-side comparison of two runs (cost, duration, efficiency)."""
    a = _find_run(run_id_1)
    b = _find_run(run_id_2)
    if a is None:
        console.print(f"[red]No run matching[/] {run_id_1}")
        raise SystemExit(1)
    if b is None:
        console.print(f"[red]No run matching[/] {run_id_2}")
        raise SystemExit(1)

    def _f(row: dict, key: str, default: float = 0.0) -> float:
        return float(row.get(key) or default)

    table = Table(title="Run comparison", header_style="bold")
    table.add_column("Metric")
    table.add_column(str(a.get("run_id", "?"))[:10], justify="right")
    table.add_column(str(b.get("run_id", "?"))[:10], justify="right")
    table.add_column("Winner")

    metrics = [
        ("Cost ($)", "cost_usd", True),
        ("Duration (s)", "duration_s", True),
        ("GPU util %", "gpu_util_avg", False),
        ("Exit code", "exit_code", True),
    ]
    for label, key, lower_better in metrics:
        va, vb = _f(a, key), _f(b, key)
        if key == "exit_code":
            winner = "A" if va == 0 and vb != 0 else ("B" if vb == 0 and va != 0 else "—")
        elif va == vb:
            winner = "tie"
        elif lower_better:
            winner = "A" if va < vb else "B"
        else:
            winner = "A" if va > vb else "B"
        fmt = (lambda v: str(int(v))) if key == "exit_code" else (lambda v: f"{v:.4f}")
        table.add_row(label, fmt(va), fmt(vb), winner)

    table.add_row("Framework", str(a.get("framework", "—")), str(b.get("framework", "—")), "—")
    table.add_row("Batch", str(a.get("batch_size", "—")), str(b.get("batch_size", "—")), "—")
    table.add_row("Epochs", str(a.get("epochs", "—")), str(b.get("epochs", "—")), "—")
    table.add_row("LR", str(a.get("learning_rate", "—")), str(b.get("learning_rate", "—")), "—")
    console.print(table)

    ca, cb = _f(a, "cost_usd"), _f(b, "cost_usd")
    if ca < cb:
        pct = ((cb - ca) / cb * 100) if cb else 0
        console.print(f"\n[green]Run A was cheaper by ${cb - ca:.4f} ({pct:.0f}%)[/]")
    elif cb < ca:
        pct = ((ca - cb) / ca * 100) if ca else 0
        console.print(f"\n[green]Run B was cheaper by ${ca - cb:.4f} ({pct:.0f}%)[/]")
    else:
        console.print("\n[dim]Same tracked cost[/]")


@cli.command("resume")
@click.argument("run_id")
@click.option(
    "--script",
    "script_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Override script path",
)
@click.option("--force", is_flag=True)
def resume_cmd(run_id: str, script_path: str | None, force: bool) -> None:
    """Resume an interrupted or checkpointed run.

    Finds the last checkpoint under .starfelt/checkpoints/{run_id}/ and
    re-launches the script with STARFELT_RESUME_FROM set.
    """
    import os as _os

    row = _find_run(run_id)
    if row is None:
        console.print(f"[red]No run found matching[/] {run_id}")
        raise SystemExit(1)

    full_id = str(row["run_id"])
    ckpt_dir = Path.cwd() / ".starfelt" / "checkpoints" / full_id
    latest = ckpt_dir / "latest.pt"
    if not latest.exists():
        candidates = sorted(ckpt_dir.glob("epoch_*.pt")) if ckpt_dir.exists() else []
        if not candidates:
            console.print(f"[red]No checkpoint found under[/] {ckpt_dir}")
            console.print("Tip: use the Trainer SDK or write checkpoints yourself.")
            raise SystemExit(1)
        latest = candidates[-1]

    script = Path(script_path) if script_path else Path(str(row.get("script", "")))
    if not script.exists() or str(script) in {"starfelt.Trainer", ""}:
        console.print(
            "[yellow]Original script path missing or was Trainer SDK.[/]\n"
            "Pass --script path/to/train.py explicitly."
        )
        raise SystemExit(1)

    console.print(f"[cyan]Resuming[/] {full_id[:12]}… from {latest}")
    cfg = load_config()
    backup = dict(_os.environ)
    try:
        _os.environ["STARFELT_RESUME_FROM"] = str(latest.resolve())
        _os.environ["STARFELT_RUN_ID"] = full_id
        result = run_wrapped(script, [], cfg, force=force, confirm_fails=not force)
    finally:
        _os.environ.clear()
        _os.environ.update(backup)

    if result.aborted:
        raise SystemExit(2)
    console.print(
        Panel(
            f"Exit: {result.exit_code}\nDuration (this segment): {result.duration_s:.1f}s\n"
            f"Cost (this segment): ${result.cost_usd:.4f}\nRun id: {result.run_id}",
            title="Resume complete",
            border_style="green" if result.exit_code == 0 else "red",
        )
    )


@cli.command("benchmark")
@click.option("--steps", default=50, show_default=True, help="Mini training steps")
@click.option("--warmup", default=5, show_default=True, help="Unmeasured warmup steps")
@click.option("--repeats", default=3, show_default=True, help="Measured repetitions")
@click.option("--batch-size", default=32, show_default=True)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
)
@click.option("--seed", default=0, show_default=True)
@click.option("--json-out", type=click.Path(dir_okay=False), default=None)
def benchmark_cmd(
    steps: int,
    warmup: int,
    repeats: int,
    batch_size: int,
    device: str,
    seed: int,
    json_out: str | None,
) -> None:
    """Run repeatable throughput, cost, power, and energy measurements."""
    try:
        report = run_mlp_benchmark(
            BenchmarkConfig(
                steps=steps,
                warmup_steps=warmup,
                repeats=repeats,
                batch_size=batch_size,
                device=device,
                seed=seed,
            )
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1)

    from starfelt.providers.base import CATALOG_WARNING

    summary = report.to_dict()["summary"]
    console.print(
        Panel(
            f"Workload: [bold]synthetic_mlp_v1[/]\n"
            f"Device: [bold]{report.results[0].device}[/] · repeats={repeats}\n"
            f"Median elapsed: {summary['median_elapsed_s']:.2f}s\n"
            f"Median throughput: [bold]{summary['median_steps_per_s']:.1f}[/] steps/s  ·  "
            f"[bold]{summary['median_samples_per_s']:.0f}[/] samples/s\n"
            f"Median cost: ${summary['median_cost_usd']:.6f}\n"
            f"Median energy: {summary['median_energy_j'] or 'n/a'} J",
            title="starfelt benchmark",
            border_style="cyan",
        )
    )
    console.print(f"[dim]{CATALOG_WARNING}[/]")
    if json_out:
        report.write_json(Path(json_out))
        console.print(f"[green]Wrote JSON benchmark report → {json_out}[/]")


@cli.command("config")
@click.argument("subcommand", type=click.Choice(["doctor"]))
def config_cmd(subcommand: str) -> None:
    """Config utilities. Currently: [cyan]starfelt config doctor[/]."""
    if subcommand == "doctor":
        _config_doctor()


def _config_doctor() -> None:
    """Validate starfelt.yaml against the live environment and price catalog."""
    from starfelt.providers.base import CATALOG, pick_cheapest

    try:
        cfg = load_config()
    except Exception as e:
        console.print(f"[red]Failed to load config:[/] {e}")
        raise SystemExit(1)

    table = Table(title="starfelt config doctor", header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    rates = [o.usd_per_hour for o in CATALOG]
    lo, hi = min(rates), max(rates)
    if lo * 0.5 <= cfg.gpu_hour_usd <= hi * 1.5:
        table.add_row(
            "gpu_hour_usd",
            "[green]ok[/]",
            f"${cfg.gpu_hour_usd:.2f} within catalog band ${lo:.2f}–${hi:.2f}",
        )
    else:
        table.add_row(
            "gpu_hour_usd",
            "[yellow]warn[/]",
            f"${cfg.gpu_hour_usd:.2f} outside typical catalog ${lo:.2f}–${hi:.2f} — "
            "run [cyan]starfelt benchmark[/]",
        )

    if cfg.budget_usd_per_run <= 0:
        table.add_row("budget_usd_per_run", "[red]fail[/]", "must be > 0")
    else:
        hours_affordable = cfg.budget_usd_per_run / cfg.gpu_hour_usd
        table.add_row(
            "budget_usd_per_run",
            "[green]ok[/]",
            f"${cfg.budget_usd_per_run:.0f} ≈ {hours_affordable:.1f} GPU-hours at current rate",
        )

    try:
        best = pick_cheapest(cfg.preferred_providers, allow_spot=cfg.allow_spot)
        table.add_row(
            "preferred providers",
            "[green]ok[/]",
            f"cheapest match right now: {best.name} {best.gpu} @ ${best.usd_per_hour:.2f}/hr",
        )
    except Exception as e:
        table.add_row("preferred providers", "[yellow]warn[/]", str(e))

    if cfg.patience_steps < 10:
        table.add_row("patience_steps", "[yellow]warn[/]", f"{cfg.patience_steps} is very low")
    else:
        table.add_row("patience_steps", "[green]ok[/]", str(cfg.patience_steps))

    console.print(table)
    console.print(
        "[dim]Tip: after changing hardware, re-run [cyan]starfelt benchmark[/] "
        "and update cost.gpu_hour_usd[/]"
    )


if __name__ == "__main__":
    cli()
