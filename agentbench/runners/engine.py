"""Core evaluation engine."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, AsyncIterator

import structlog

from agentbench.adapters.base import AgentAdapterRegistry, BaseAgentAdapter
from agentbench.core.models import (
    AgentResult,
    CaseResult,
    EvalCase,
    EvalRun,
    EvalSuiteConfig,
    EvaluatorScore,
    RunStatus,
)
from agentbench.evaluators.base import BaseEvaluator, EvaluatorRegistry
from agentbench.evaluators.builtin import ContainsEvaluator, ExactMatchEvaluator

logger = structlog.get_logger(__name__)


class EvaluationEngine:
    """
    Orchestrates the execution of an evaluation suite.

    The engine:
    1. Loads the agent adapter from the suite config.
    2. Determines which evaluators to run (auto-detect or explicit list).
    3. Executes each test case concurrently (up to max_concurrency).
    4. Aggregates results into an EvalRun.
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        progress_callback: Callable[[CaseResult, int, int], None] | None = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.progress_callback = progress_callback

    async def run(
        self,
        suite: EvalSuiteConfig,
        adapter: BaseAgentAdapter | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> EvalRun:
        """
        Execute an evaluation suite.

        Args:
            suite: The parsed EvalSuiteConfig to run.
            adapter: Optional pre-configured adapter. If None, one is created from suite config.
            extra_context: Optional additional context passed to evaluators.

        Returns:
            A completed EvalRun with all case results and aggregated metrics.
        """
        run_id = str(uuid.uuid4())[:8]
        run = EvalRun(
            run_id=run_id,
            suite_name=suite.name,
            status=RunStatus.RUNNING,
            total_cases=len(suite.cases),
        )

        logger.info("evaluation_started", run_id=run_id, suite=suite.name, cases=len(suite.cases))

        # Build adapter
        if adapter is None:
            adapter = AgentAdapterRegistry.create(suite.agent)

        # Determine evaluators
        evaluators = self._resolve_evaluators(suite)

        context = {**(extra_context or {}), **suite.settings}

        # Setup adapter
        await adapter.setup()

        start_time = time.monotonic()
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_case(case: EvalCase, index: int) -> CaseResult:
            async with semaphore:
                return await self._run_single_case(case, adapter, evaluators, context, index)

        tasks = [
            asyncio.create_task(run_case(case, i))
            for i, case in enumerate(suite.cases)
        ]

        case_results: list[CaseResult] = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            case_results.append(result)
            if self.progress_callback:
                self.progress_callback(result, len(case_results), len(suite.cases))

        await adapter.teardown()

        run.case_results = case_results
        run.duration_ms = (time.monotonic() - start_time) * 1000
        run.status = RunStatus.COMPLETED
        run.compute_summary()

        logger.info(
            "evaluation_completed",
            run_id=run_id,
            overall_score=run.overall_score,
            passed=run.passed_cases,
            failed=run.failed_cases,
        )

        return run

    async def _run_single_case(
        self,
        case: EvalCase,
        adapter: BaseAgentAdapter,
        evaluators: list[BaseEvaluator],
        context: dict[str, Any],
        index: int,
    ) -> CaseResult:
        """Run a single test case through the adapter and evaluator pipeline."""
        case_start = time.monotonic()

        # Execute agent
        try:
            agent_result = await adapter.run(case.input, {**context, **case.context})
        except Exception as e:
            logger.error("agent_execution_failed", case_id=case.id, error=str(e))
            agent_result = AgentResult(
                response="",
                errors=[f"Agent execution failed: {e}"],
            )

        # Run evaluators
        evaluator_scores: list[EvaluatorScore] = []
        for evaluator in evaluators:
            try:
                score = await evaluator.evaluate(agent_result, case, context)
                evaluator_scores.append(score)
            except Exception as e:
                logger.warning("evaluator_failed", evaluator=evaluator.name, case_id=case.id, error=str(e))
                evaluator_scores.append(
                    EvaluatorScore(
                        evaluator_name=evaluator.name,
                        score=0.0,
                        passed=False,
                        reason=f"Evaluator error: {e}",
                    )
                )

        # Compute overall score for this case
        active_scores = [
            s for s in evaluator_scores
            if s.score < 1.0 or "skipped" not in (s.reason or "").lower()
        ]
        if active_scores:
            overall_score = sum(s.score for s in active_scores) / len(active_scores)
        else:
            overall_score = 1.0

        passed = all(s.passed for s in evaluator_scores)

        return CaseResult(
            case_id=case.id,
            passed=passed,
            agent_result=agent_result,
            evaluator_scores=evaluator_scores,
            overall_score=overall_score,
            duration_ms=(time.monotonic() - case_start) * 1000,
        )

    def _resolve_evaluators(self, suite: EvalSuiteConfig) -> list[BaseEvaluator]:
        """
        Determine which evaluators to run for a suite.

        If the suite specifies evaluators explicitly, use those.
        Otherwise, auto-detect based on the assertion types present in test cases.
        """
        if suite.evaluators:
            return [EvaluatorRegistry.create(name) for name in suite.evaluators]

        # Auto-detect from assertions
        evaluator_names: set[str] = set()
        for case in suite.cases:
            exp = case.expected
            if exp.exact is not None:
                evaluator_names.add("exact_match")
            if exp.contains or exp.not_contains:
                evaluator_names.add("contains")
            if exp.similar_to is not None:
                evaluator_names.add("semantic_similarity")
            if exp.tool_call is not None:
                evaluator_names.add("tool_call")
            if exp.json_schema is not None:
                evaluator_names.add("json_schema")
            if exp.max_latency_ms is not None:
                evaluator_names.add("latency")
            if exp.max_cost_usd is not None:
                evaluator_names.add("cost")
            if exp.llm_criteria:
                evaluator_names.add("instruction_following")
            if case.reference_documents:
                evaluator_names.add("hallucination")

        if not evaluator_names:
            # Fallback: run contains evaluator as a sanity check
            evaluator_names.add("contains")

        return [EvaluatorRegistry.create(name) for name in sorted(evaluator_names)]
