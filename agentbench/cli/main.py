"""
AgentBench CLI — main entry point.

Commands:
  run      Run an evaluation dataset
  report   Show results from a previous run
  compare  Compare two evaluation runs
  validate Check if a run meets quality gate requirements
  list     List recent evaluation runs
  serve    Start the web report interface
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(package_name="agentbench")
def cli() -> None:
    """AgentBench — Test your AI agent before your users do."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--agent-type", "-a", default=None, help="Override the agent type (e.g. echo, openai, http)")
@click.option("--concurrency", "-c", default=5, show_default=True, help="Max concurrent agent calls")
@click.option("--no-failures", is_flag=True, help="Don't print failed cases in output")
@click.option("--output", "-o", default=None, help="Save run ID to file")
@click.option("--verbose", "-v", is_flag=True)
def run(
    path: str,
    agent_type: str | None,
    concurrency: int,
    no_failures: bool,
    output: str | None,
    verbose: bool,
) -> None:
    """Run evaluations from a YAML file or directory."""
    asyncio.run(_run(path, agent_type, concurrency, not no_failures, output, verbose))


async def _run(
    path: str,
    agent_type: str | None,
    concurrency: int,
    show_failures: bool,
    output: str | None,
    verbose: bool,
) -> None:
    from agentbench.datasets.loader import DatasetLoader
    from agentbench.reporting.console import ConsoleReporter
    from agentbench.runners.evaluation_runner import EvaluationRunner
    from agentbench.storage.sqlite_store import SQLiteStore

    p = Path(path)
    datasets = []

    if p.is_dir():
        datasets = DatasetLoader.from_directory(p)
        if not datasets:
            console.print(f"[red]No YAML files found in {path}[/red]")
            sys.exit(1)
    else:
        datasets = [DatasetLoader.from_yaml(p)]

    store = SQLiteStore()
    await store.init()
    try:
        reporter = ConsoleReporter(show_failures=show_failures, verbose=verbose)

        all_passed = True
        run_ids = []

        for dataset in datasets:
            if agent_type:
                dataset.agent["type"] = agent_type

            total = len(dataset.cases)
            completed = 0

            def progress(result, done, total_cases):
                nonlocal completed
                completed = done
                icon = "✓" if result.passed else "✗"
                color = "green" if result.passed else "red"
                console.print(
                    f"  [{color}]{icon}[/{color}] [{done}/{total_cases}] {result.case_id}",
                    end="\r",
                )

            console.print(f"\n[bold]Running[/bold] [cyan]{dataset.name}[/cyan] "
                          f"[dim]({total} cases)[/dim]\n")

            runner = EvaluationRunner(
                dataset=dataset,
                max_concurrency=concurrency,
                progress_callback=progress,
            )

            try:
                eval_run = await runner.run()
            except (ValueError, ImportError) as e:
                # Adapter misconfiguration (e.g. missing credentials or extras).
                console.print(f"[bold red]✗ {dataset.name}:[/bold red] {e}")
                all_passed = False
                continue
            eval_run.name = dataset.name

            # Clear progress line
            console.print(" " * 80, end="\r")

            await store.save_run(eval_run)
            run_ids.append(eval_run.id)

            reporter.print_run_summary(eval_run)

            if output:
                Path(output).write_text(eval_run.id)

            if eval_run.failed_cases > 0:
                all_passed = False

        if len(run_ids) > 1:
            console.print(f"\n[dim]Run IDs: {', '.join(run_ids)}[/dim]")

        sys.exit(0 if all_passed else 1)
    finally:
        await store.close()


@cli.command()
@click.option("--minimum-score", "-m", default=0.9, show_default=True, type=float,
              help="Minimum pass rate required (0.0-1.0)")
@click.option("--run-id", default=None, help="Specific run ID to validate")
def validate(minimum_score: float, run_id: str | None) -> None:
    """Validate that evaluation results meet quality gate requirements."""
    asyncio.run(_validate(minimum_score, run_id))


async def _validate(minimum_score: float, run_id: str | None) -> None:

    from agentbench.storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    await store.init()
    try:
        if run_id:
            record = await store.get_run(run_id)
            records = [record] if record else []
        else:
            records = await store.list_runs(limit=1)

        if not records:
            console.print("[red]No evaluation runs found. Run 'agentbench run' first.[/red]")
            sys.exit(1)

        record = records[0]
        pass_rate = record.pass_rate

        if pass_rate >= minimum_score:
            console.print(
                f"\n[green]✓ Quality gate passed[/green]\n\n"
                f"  Score:    {pass_rate*100:.1f}%\n"
                f"  Required: {minimum_score*100:.1f}%\n"
            )
            sys.exit(0)
        else:
            console.print(
                f"\n[bold red]✗ Quality gate failed[/bold red]\n\n"
                f"  Score:    {pass_rate*100:.1f}%\n"
                f"  Required: {minimum_score*100:.1f}%\n"
            )
            sys.exit(1)
    finally:
        await store.close()


@cli.command("list")
@click.option("--limit", "-n", default=20, show_default=True)
def list_runs(limit: int) -> None:
    """List recent evaluation runs."""
    asyncio.run(_list_runs(limit))


async def _list_runs(limit: int) -> None:
    from rich import box
    from rich.table import Table

    from agentbench.storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    await store.init()
    try:
        records = await store.list_runs(limit=limit)

        if not records:
            console.print("[dim]No evaluation runs found.[/dim]")
            return

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Run ID", style="cyan")
        table.add_column("Dataset")
        table.add_column("Agent")
        table.add_column("Cases", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Status")
        table.add_column("Started At", style="dim")

        for r in records:
            color = "green" if r.pass_rate >= 0.9 else ("yellow" if r.pass_rate >= 0.7 else "red")
            table.add_row(
                r.id,
                r.dataset_name or "-",
                r.agent_type or "-",
                str(r.total_cases),
                f"[{color}]{r.pass_rate*100:.1f}%[/{color}]",
                r.status,
                str(r.started_at)[:16] if r.started_at else "-",
            )

        console.print(table)
    finally:
        await store.close()


@cli.command()
@click.argument("run_a")
@click.argument("run_b")
def compare(run_a: str, run_b: str) -> None:
    """Compare two evaluation runs side by side."""
    asyncio.run(_compare(run_a, run_b))


async def _compare(run_a_id: str, run_b_id: str) -> None:
    import json

    from agentbench.core.models import EvalResult, EvalRun
    from agentbench.reporting.console import ConsoleReporter
    from agentbench.storage.sqlite_store import SQLiteStore

    store = SQLiteStore()
    await store.init()
    try:
        rec_a = await store.get_run(run_a_id)
        rec_b = await store.get_run(run_b_id)

        if not rec_a:
            console.print(f"[red]Run {run_a_id!r} not found.[/red]")
            sys.exit(1)
        if not rec_b:
            console.print(f"[red]Run {run_b_id!r} not found.[/red]")
            sys.exit(1)

        def record_to_run(rec) -> EvalRun:
            results_data = json.loads(rec.results_json or "[]")
            results = [EvalResult(**r) for r in results_data]
            run = EvalRun(
                id=rec.id,
                name=rec.name or "",
                dataset_name=rec.dataset_name or "",
                agent_type=rec.agent_type or "",
            )
            run.results = results
            return run

        run_a = record_to_run(rec_a)
        run_b = record_to_run(rec_b)

        reporter = ConsoleReporter()
        reporter.print_comparison(run_a, run_b, label_a=run_a_id, label_b=run_b_id)
    finally:
        await store.close()


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True)
def serve(host: str, port: int) -> None:
    """Start the AgentBench web interface."""
    import uvicorn

    from agentbench.api.app import create_app

    app = create_app()
    console.print(f"\n[cyan]AgentBench[/cyan] web interface running at http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)
