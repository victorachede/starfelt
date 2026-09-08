"""Starfelt CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from starfelt import __version__
from starfelt.core.analyze import analyze_script
from starfelt.core.cost import CostTracker, load_history
from starfelt.core.config import init_project, load_config
from starfelt.core.runner import run_wrapped

console = Console()


@click.group()
@click.version_option(__version__, prog_name="starfelt")
def cli() -> None:
    """Starfelt — cheaper, smarter training runs. No rewrite."""


@cli.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing starfelt.yaml")
def init_cmd(force: bool) -> None:
    """Create starfelt.yaml in the current directory."""
    path = init_project(Path.cwd(), force=force)
    console.print(f"[bold green]✓[/] Wrote {path}")
    console.print("Edit providers / budget, then: [cyan]starfelt run train.py[/]")


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
def run_cmd(
    script: str,
    script_args: tuple[str, ...],
    config_path: str | None,
    dry_run: bool,
) -> None:
    """Wrap and run a training script with Starfelt monitoring."""
    cfg = load_config(Path(config_path) if config_path else None)
    result = run_wrapped(
        Path(script),
        list(script_args),
        cfg,
        dry_run=dry_run,
    )
    if result.dry_run:
        console.print("[yellow]Dry run[/] — no process started.")
        return
    console.print(
        Panel(
            f"Exit code: {result.exit_code}\n"
            f"Duration: {result.duration_s:.1f}s\n"
            f"Tracked cost: ${result.cost_usd:.4f}\n"
            f"Run id: {result.run_id}",
            title="Run complete",
            border_style="green" if result.exit_code == 0 else "red",
        )
    )


@cli.command("cost")
@click.option("--json", "as_json", is_flag=True)
def cost_cmd(as_json: bool) -> None:
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
    table.add_column("Cost USD")
    for row in history[-20:]:
        table.add_row(
            row.get("run_id", "?")[:8],
            row.get("script", ""),
            f"{row.get('duration_s', 0):.1f}s",
            f"${row.get('cost_usd', 0):.4f}",
        )
    console.print(table)
    total = sum(r.get("cost_usd", 0) for r in history)
    console.print(f"Total tracked: [bold]${total:.4f}[/]")


if __name__ == "__main__":
    cli()
