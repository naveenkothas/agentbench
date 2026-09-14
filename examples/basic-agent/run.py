"""
Run AgentBench evaluation with the basic customer support agent.

Usage:
    python run.py

This script shows how to wire a Python agent function into AgentBench
using the Python API rather than the CLI.
"""

import asyncio
import sys
from pathlib import Path

# Add the repo root to the path if running without installing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentbench.adapters.builtin import CustomAdapter
from agentbench.core.models import AgentConfig
from agentbench.datasets.loader import load_suite_from_yaml
from agentbench.reporting.terminal import print_header, print_run_report
from agentbench.runners.engine import EvaluationEngine
from agentbench.storage.sqlite import SQLiteStorage

from agent import customer_support_agent


async def main() -> None:
    # Load the evaluation suite
    suite = load_suite_from_yaml(Path(__file__).parent / "evaluation.yaml")

    # Override agent type to "custom" and attach our function
    suite.agent.type = "custom"
    adapter = CustomAdapter(AgentConfig(type="custom"))
    adapter.set_callable(customer_support_agent)

    # Run evaluation
    print_header(suite.name)
    print(f"Running {len(suite.cases)} test cases...\n")

    engine = EvaluationEngine(max_concurrency=3)
    run = await engine.run(suite, adapter=adapter)

    # Save results
    storage = SQLiteStorage()
    await storage.save_run(run)

    # Print report
    print_run_report(run, verbose=True)
    print(f"\nResults saved. Run ID: {run.run_id}")
    print("View full report: agentbench report")


if __name__ == "__main__":
    asyncio.run(main())
