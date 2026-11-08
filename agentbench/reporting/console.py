"""
Rich-powered console reporter for evaluation results.
"""
from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentbench.core.models import EvalResult, EvalRun

console = Console()


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "green"
    elif score >= 0.7:
        return "yellow"
    else:
        return "red"


def _pct(score: float) -> str:
    return f"{score * 100:.1f}%"


class ConsoleReporter:
    """Renders evaluation results to the terminal using Rich."""

    def __init__(self, show_failures: bool = True, verbose: bool = False) -> None:
        self.show_failures = show_failures
        self.verbose = verbose

    def print_run_summary(self, run: EvalRun) -> None:
        """Print the full evaluation run summary."""
        console.print()
        console.print(Panel.fit(
            "[bold cyan]AgentBench[/bold cyan]",
            border_style="cyan",
        ))
        console.print()

        # Per-evaluator scores
        scores_by_evaluator = run.avg_score_by_evaluator
        if scores_by_evaluator:
            for evaluator, avg in sorted(scores_by_evaluator.items()):
                color = _score_color(avg)
                icon = "✓" if avg >= 0.7 else "✗"
                console.print(
                    f"  [{color}]{icon}[/{color}] {evaluator:<28} [{color}]{_pct(avg)}[/{color}]"
                )

        console.print()
        console.rule(style="dim")

        overall = run.pass_rate
        overall_color = _score_color(overall)
        console.print(
            f"  [bold]Overall Score[/bold]           "
            f"[bold {overall_color}]{_pct(overall)}[/bold {overall_color}]"
        )
        console.print()

        # Summary stats
        stats_table = Table(box=None, show_header=False, padding=(0, 2))
        stats_table.add_column("", style="dim")
        stats_table.add_column("", style="bold")
        stats_table.add_row("Passed", f"[green]{run.passed_cases}[/green]")
        stats_table.add_row("Failed", f"[red]{run.failed_cases}[/red]")
        stats_table.add_row("Total", str(run.total_cases))
        if run.total_latency_ms > 0:
            stats_table.add_row("Duration", f"{run.total_latency_ms/1000:.1f}s")
        if run.total_cost_usd > 0:
            stats_table.add_row("Estimated Cost", f"${run.total_cost_usd:.4f}")

        console.print(stats_table)
        console.print()

        if self.show_failures:
            failed = [r for r in run.results if not r.passed]
            if failed:
                self._print_failures(failed)

    def _print_failures(self, failed: list[EvalResult]) -> None:
        console.print(f"[bold red]─── Failed Cases ({len(failed)}) ───[/bold red]")
        console.print()
        for result in failed[:10]:  # Cap at 10 to avoid overwhelming output
            self._print_single_failure(result)
            if failed.index(result) < len(failed) - 1:
                console.print()

        if len(failed) > 10:
            console.print(f"  [dim]... and {len(failed) - 10} more failures. Run with --verbose to see all.[/dim]")

    def _print_single_failure(self, result: EvalResult) -> None:
        console.print(f"  [bold red]FAILED[/bold red]: [bold]{result.case_id}[/bold]")

        if result.error:
            console.print(f"  [red]Error: {result.error}[/red]")
            return

        for score in result.scores:
            if not score.passed:
                console.print(f"  [dim]Evaluator:[/dim] {score.evaluator}")
                console.print(f"  [dim]Score:[/dim]     {score.score:.2f}")
                console.print(f"  [dim]Reason:[/dim]    {score.reason}")

        if result.agent_result:
            response_preview = result.agent_result.response[:200].replace("\n", " ")
            console.print(f"  [dim]Response:[/dim]  {response_preview}{'...' if len(result.agent_result.response) > 200 else ''}")

    def print_comparison(self, run_a: EvalRun, run_b: EvalRun, label_a: str = "A", label_b: str = "B") -> None:
        """Print a side-by-side comparison of two evaluation runs."""
        console.print()
        console.print(f"[bold]Comparing[/bold] [cyan]{label_a}[/cyan] vs [cyan]{label_b}[/cyan]")
        console.print()

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Metric", style="dim", min_width=24)
        table.add_column(label_a, justify="right")
        table.add_column(label_b, justify="right")
        table.add_column("Change", justify="right")

        def diff_str(a: float, b: float, percent: bool = True) -> str:
            diff = b - a
            if percent:
                diff_pct = diff * 100
                sign = "+" if diff_pct > 0 else ""
                color = "green" if diff_pct > 0 else ("red" if diff_pct < 0 else "dim")
                return f"[{color}]{sign}{diff_pct:.1f}%[/{color}]"
            else:
                sign = "+" if diff > 0 else ""
                color = "green" if diff > 0 else ("red" if diff < 0 else "dim")
                return f"[{color}]{sign}{diff:.2f}[/{color}]"

        # Overall
        table.add_row(
            "Pass Rate",
            _pct(run_a.pass_rate),
            _pct(run_b.pass_rate),
            diff_str(run_a.pass_rate, run_b.pass_rate),
        )

        # Per-evaluator
        all_evals = set(run_a.avg_score_by_evaluator) | set(run_b.avg_score_by_evaluator)
        for ev in sorted(all_evals):
            a_score = run_a.avg_score_by_evaluator.get(ev, 0.0)
            b_score = run_b.avg_score_by_evaluator.get(ev, 0.0)
            table.add_row(ev, _pct(a_score), _pct(b_score), diff_str(a_score, b_score))

        # Latency
        if run_a.total_latency_ms or run_b.total_latency_ms:
            a_lat = run_a.total_latency_ms / 1000
            b_lat = run_b.total_latency_ms / 1000
            lat_diff = b_lat - a_lat
            sign = "+" if lat_diff > 0 else ""
            color = "red" if lat_diff > 0 else "green"  # lower is better for latency
            table.add_row(
                "Duration (s)",
                f"{a_lat:.1f}s",
                f"{b_lat:.1f}s",
                f"[{color}]{sign}{lat_diff:.1f}s[/{color}]",
            )

        # Cost
        if run_a.total_cost_usd or run_b.total_cost_usd:
            cost_diff = run_b.total_cost_usd - run_a.total_cost_usd
            sign = "+" if cost_diff > 0 else ""
            color = "red" if cost_diff > 0 else "green"
            table.add_row(
                "Cost (USD)",
                f"${run_a.total_cost_usd:.4f}",
                f"${run_b.total_cost_usd:.4f}",
                f"[{color}]{sign}${cost_diff:.4f}[/{color}]",
            )

        console.print(table)
