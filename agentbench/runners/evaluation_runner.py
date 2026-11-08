"""
Core evaluation runner.

Orchestrates the full pipeline:
Dataset → Agent Adapter → Evaluator Pipeline → EvalRun
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from agentbench.adapters.base import AgentAdapter
from agentbench.adapters.registry import AdapterRegistry
from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import EvalCase, EvalDataset, EvalResult, EvalRun, RunStatus
from agentbench.evaluators.registry import EvaluatorRegistry

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runs an evaluation dataset against an agent and returns an EvalRun.

    Args:
        dataset: The evaluation dataset to run.
        adapter: Optional pre-configured agent adapter. If not provided,
                 one will be created from dataset.agent config.
        evaluators: Optional list of evaluator instances. If not provided,
                    they will be created from dataset.evaluators config.
        max_concurrency: Maximum number of concurrent agent calls.
        progress_callback: Optional callback called after each case completes.
    """

    def __init__(
        self,
        dataset: EvalDataset,
        adapter: AgentAdapter | None = None,
        evaluators: list[BaseEvaluator] | None = None,
        max_concurrency: int = 5,
        progress_callback: Callable[[EvalResult, int, int], None] | None = None,
    ) -> None:
        self.dataset = dataset
        self._adapter = adapter
        self._evaluators = evaluators
        self.max_concurrency = max_concurrency
        self.progress_callback = progress_callback

    def _build_adapter(self) -> AgentAdapter:
        agent_config = self.dataset.agent
        adapter_type = agent_config.get("type", "echo")
        config = {k: v for k, v in agent_config.items() if k != "type"}
        return AdapterRegistry.create(adapter_type, config)

    def _build_evaluators(self) -> list[BaseEvaluator]:
        evaluators = []
        for name in self.dataset.evaluators:
            if isinstance(name, dict):
                ev_name = name.get("name", "")
                ev_config = {k: v for k, v in name.items() if k != "name"}
            else:
                ev_name = name
                ev_config = {}
            evaluators.append(EvaluatorRegistry.create(ev_name, ev_config))
        return evaluators

    async def run(self) -> EvalRun:
        """Execute the full evaluation and return results."""
        adapter = self._adapter or self._build_adapter()
        evaluators = self._evaluators or self._build_evaluators()

        run = EvalRun(
            dataset_name=self.dataset.name,
            agent_type=adapter.adapter_type,
        )
        run.status = RunStatus.RUNNING

        await adapter.setup()

        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0
        total = len(self.dataset.cases)

        async def run_case(case: EvalCase) -> EvalResult:
            nonlocal completed
            async with semaphore:
                result = await self._run_single_case(case, adapter, evaluators)
                completed += 1
                if self.progress_callback:
                    self.progress_callback(result, completed, total)
                return result

        tasks = [run_case(case) for case in self.dataset.cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                case_id = self.dataset.cases[i].id
                logger.error(f"Case {case_id} failed with exception: {result}")
                results[i] = EvalResult(
                    case_id=case_id,
                    passed=False,
                    error=str(result),
                )

        run.results = [r for r in results if isinstance(r, EvalResult)]
        run.status = RunStatus.COMPLETED

        from datetime import datetime
        run.completed_at = datetime.utcnow()

        await adapter.teardown()
        return run

    async def _run_single_case(
        self,
        case: EvalCase,
        adapter: AgentAdapter,
        evaluators: list[BaseEvaluator],
    ) -> EvalResult:
        """Run a single evaluation case through the agent and all evaluators."""
        try:
            agent_result = await adapter.run(case.input, case.context)
        except Exception as e:
            logger.error(f"Agent error on case {case.id}: {e}")
            return EvalResult(
                case_id=case.id,
                passed=False,
                error=f"Agent execution failed: {e}",
            )

        scores = []
        for evaluator in evaluators:
            try:
                score = await evaluator.evaluate(agent_result, case)
                scores.append(score)
            except Exception as e:
                logger.error(f"Evaluator {evaluator.name} failed on case {case.id}: {e}")

        passed = all(s.passed for s in scores) if scores else True

        return EvalResult(
            case_id=case.id,
            passed=passed,
            scores=scores,
            agent_result=agent_result,
        )
