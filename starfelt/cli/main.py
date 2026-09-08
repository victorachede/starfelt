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
@click.option(
    "--compare",
    is_flag=True,
    help="Show baseline (without Starfelt) vs actual and savings",
)
def cost_cmd(as_json: bool, compare: bool) -> None:
    """Show local run cost history."""
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
    if compare:
        table.add_column("Baseline")
        table.add_column("Saved")
    for row in history[-20:]:
        cost = float(row.get("cost_usd", 0) or 0)
        baseline = float(row.get("baseline_cost_usd") or cost * 1.35)
        saved = float(row.get("saved_usd") if row.get("saved_usd") is not None else baseline - cost)
        cells = [
            str(row.get("run_id", "?"))[:8],
            str(row.get("script", "")),
            f"{row.get('duration_s', 0):.1f}s",
            f"${cost:.4f}",
        ]
        if compare:
            cells.extend([f"${baseline:.4f}", f"${saved:.4f}"])
        table.add_row(*cells)
    console.print(table)
    total = sum(float(r.get("cost_usd", 0) or 0) for r in history)
    if compare:
        total_base = sum(
            float(r.get("baseline_cost_usd") or float(r.get("cost_usd", 0) or 0) * 1.35)
            for r in history
        )
        console.print(
            f"Total tracked: [bold]${total:.4f}[/]  ·  "
            f"Baseline: [bold]${total_base:.4f}[/]  ·  "
            f"Saved: [bold green]${total_base - total:.4f}[/]"
        )
    else:
        console.print(f"Total tracked: [bold]${total:.4f}[/]")


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


if __name__ == "__main__":
    cli()
