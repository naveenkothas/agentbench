"""
Example: simple custom agent using AgentBench's Python API directly.

This shows how to use AgentBench programmatically, not just via the CLI.
"""
from __future__ import annotations

import asyncio

from agentbench import AgentAdapter, AgentResult
from agentbench.datasets.loader import DatasetLoader
from agentbench.runners.evaluation_runner import EvaluationRunner
from agentbench.reporting.console import ConsoleReporter


class SimpleCustomAgent(AgentAdapter):
    """A trivial agent that responds to common support questions."""

    adapter_type = "simple_custom"

    async def run(self, input: str, context: dict) -> AgentResult:
        # Your real agent logic goes here
        if "password" in input.lower():
            response = "To reset your password, click 'Forgot Password' on the login page."
        elif "cancel" in input.lower():
            response = "I can help you cancel your subscription. Please confirm your email address."
        elif "invoice" in input.lower():
            response = "Your invoices are available in Account Settings > Billing."
        else:
            response = f"Thank you for reaching out. I'm here to help with: {input}"

        return AgentResult(
            response=response,
            latency_ms=10.0,
        )


async def main():
    dataset = DatasetLoader.from_yaml("evaluation.yaml")

    # Override the adapter from the YAML with our custom one
    agent = SimpleCustomAgent()

    runner = EvaluationRunner(dataset=dataset, adapter=agent)
    run = await runner.run()

    reporter = ConsoleReporter()
    reporter.print_run_summary(run)


if __name__ == "__main__":
    asyncio.run(main())
