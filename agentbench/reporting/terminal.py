"""Terminal reporting using Rich for beautiful CLI output."""

from __future__ import annotations

from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich import box

from agentbench.core.models import CaseResult, EvalRun


console = Console()


def print_header(suite_name: str) -> None:
    """Print the AgentBench run header."""
    console.print()
    console.print("[bold cyan]AgentBench[/bold cyan]", justify="left")
    console.print(f"[dim]Evaluating:[/dim] [white]{suite_name}[/white]")
    console.print("[dim]" + "─" * 50 + "[/dim]")
    console.print()


def print_run_report(run: EvalRun, verbose: bool = False) -> None:
    """Print a complete evaluation run report."""
    _print_metrics_table(run)
    console.print()
    _print_summary_panel(run)

    if run.failed_cases > 0:
        console.print()
        _print_failures(run, verbose=verbose)


def _print_metrics_table(run: EvalRun) -> None:
    """Print the per-evaluator metrics as a table."""
    if not run.metrics:
        return

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Bar", width=20)

    for metric_name, score in sorted(run.metrics.items()):
        pct = score * 100
        bar_filled = int(pct / 5)  # 20 chars = 100%
        bar = "█" * bar_filled + "░" * (20 - bar_filled)

        color = "green" if pct >= 90 else "yellow" if pct >= 70 else "red"
        icon = "✓" if pct >= 70 else "✗"

        metric_display = metric_name.replace("_", " ").title()
        table.add_row(
            f"[{color}]{icon}[/{color}] {metric_display}",
            f"[{color}]{pct:.1f}%[/{color}]",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(table)


def _print_summary_panel(run: EvalRun) -> None:
    """Print the overall summary panel."""
    pct = run.overall_score * 100
    color = "green" if pct >= 90 else "yellow" if pct >= 70 else "red"
    status_icon = "✅" if run.failed_cases == 0 else "⚠️" if pct >= 70 else "❌"

    lines = [
        f"[bold {color}]Overall Score: {pct:.1f}%[/bold {color}]",
        "",
        f"[green]Passed:[/green] {run.passed_cases}    "
        f"[red]Failed:[/red] {run.failed_cases}    "
        f"[dim]Total:[/dim] {run.total_cases}",
        f"[dim]Duration:[/dim] {run.duration_ms / 1000:.1f}s    "
        f"[dim]Est. Cost:[/dim] ${run.total_cost_usd:.4f}    "
        f"[dim]Run ID:[/dim] {run.run_id}",
    ]

    panel = Panel(
        "\n".join(lines),
        title=f"{status_icon} Evaluation Complete",
        border_style=color,
        padding=(1, 2),
    )
    console.print(panel)


def _print_failures(run: EvalRun, verbose: bool = False) -> None:
    """Print details of failed test cases."""
    failed = [r for r in run.case_results if not r.passed]
    console.print(f"[bold red]Failed Cases ({len(failed)})[/bold red]")
    console.print("[dim]" + "─" * 50 + "[/dim]")

    for result in failed:
        _print_failure_detail(result, verbose=verbose)
        console.print()


def _print_failure_detail(result: CaseResult, verbose: bool = False) -> None:
    """Print a single failure's detail block."""
    console.print(f"[bold red]FAILED:[/bold red] [white]{result.case_id}[/white]")

    for score in result.failed_evaluators:
        console.print(f"  [dim]Evaluator:[/dim] [yellow]{score.evaluator_name}[/yellow]")
        console.print(f"  [dim]Score:[/dim]     {score.score:.2f}")
        if score.reason:
            console.print(f"  [dim]Reason:[/dim]    {score.reason}")

    if verbose:
        console.print(f"\n  [dim]Response:[/dim]")
        response_preview = result.agent_result.response[:300]
        if len(result.agent_result.response) > 300:
            response_preview += "..."
        console.print(f"  [italic]{response_preview}[/italic]")

    if result.agent_result.errors:
        for err in result.agent_result.errors:
            console.print(f"  [red]Error:[/red] {err}")


def print_comparison_table(
    run_a: EvalRun,
    run_b: EvalRun,
    label_a: str | None = None,
    label_b: str | None = None,
) -> None:
    """Print a side-by-side comparison of two evaluation runs."""
    label_a = label_a or run_a.run_id
    label_b = label_b or run_b.run_id

    console.print()
    console.print("[bold cyan]AgentBench Comparison[/bold cyan]")
    console.print("[dim]" + "─" * 60 + "[/dim]")
    console.print()

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Metric", style="dim", width=25)
    table.add_column(label_a, justify="right", width=12)
    table.add_column(label_b, justify="right", width=12)
    table.add_column("Change", justify="right", width=10)

    all_metrics = sorted(set(run_a.metrics) | set(run_b.metrics))
    for metric in all_metrics:
        val_a = run_a.metrics.get(metric, 0.0)
        val_b = run_b.metrics.get(metric, 0.0)
        delta = val_b - val_a
        delta_str = f"+{delta * 100:.1f}%" if delta >= 0 else f"{delta * 100:.1f}%"
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "dim"

        table.add_row(
            metric.replace("_", " ").title(),
            f"{val_a * 100:.1f}%",
            f"{val_b * 100:.1f}%",
            f"[{delta_color}]{delta_str}[/{delta_color}]",
        )

    # Add latency and cost rows
    lat_a = sum(r.agent_result.latency_ms for r in run_a.case_results) / max(len(run_a.case_results), 1)
    lat_b = sum(r.agent_result.latency_ms for r in run_b.case_results) / max(len(run_b.case_results), 1)
    lat_delta = lat_b - lat_a
    lat_color = "green" if lat_delta < 0 else "red"
    table.add_row(
        "Avg Latency",
        f"{lat_a:.0f}ms",
        f"{lat_b:.0f}ms",
        f"[{lat_color}]{lat_delta:+.0f}ms[/{lat_color}]",
    )

    cost_delta = run_b.total_cost_usd - run_a.total_cost_usd
    cost_color = "green" if cost_delta < 0 else "red"
    table.add_row(
        "Total Cost",
        f"${run_a.total_cost_usd:.4f}",
        f"${run_b.total_cost_usd:.4f}",
        f"[{cost_color}]{cost_delta:+.4f}[/{cost_color}]",
    )

    console.print(table)


def make_progress() -> Progress:
    """Create a Rich progress bar for live evaluation output."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
