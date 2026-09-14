"""
Hallucination evaluator — detects claims not supported by context.

Uses simple keyword overlap as a baseline. When an LLM is configured,
uses LLM-as-judge for deeper analysis.
"""
from __future__ import annotations

import re

from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


def _simple_overlap_score(response: str, context: str) -> float:
    """
    Compute a rough groundedness score using token overlap.

    This is a lightweight heuristic — not as accurate as semantic similarity
    or LLM-as-judge, but requires no API calls.
    """
    def tokens(text: str) -> set[str]:
        return set(re.findall(r"\b\w{4,}\b", text.lower()))

    response_tokens = tokens(response)
    context_tokens = tokens(context)

    if not response_tokens:
        return 1.0

    overlap = response_tokens & context_tokens
    return len(overlap) / len(response_tokens)


class HallucinationEvaluator(BaseEvaluator):
    """
    Detects hallucinations by checking whether agent claims are
    grounded in the provided context.

    When ``context.documents`` is provided in the EvalCase, this
    evaluator assesses whether the response introduces claims not
    supported by those documents.

    Two modes:
    - ``simple`` (default): fast token-overlap heuristic, no API calls
    - ``llm`` : delegates to LLMJudgeEvaluator for deeper analysis
    """

    name = "hallucination"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        context_docs = case.context.get("documents", "")
        if isinstance(context_docs, list):
            context_docs = "\n".join(context_docs)

        if not context_docs:
            return EvaluatorScore(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason="No reference context provided — skipping hallucination check.",
            )

        mode = self.config.get("mode", "simple")
        threshold = self.config.get("threshold", 0.3)

        if mode == "llm":
            return await self._llm_evaluate(result, case, context_docs)

        score = _simple_overlap_score(result.response, context_docs)
        passed = score >= threshold

        return EvaluatorScore(
            evaluator=self.name,
            score=score,
            passed=passed,
            reason=(
                f"Groundedness score {score:.2f} ≥ threshold {threshold}"
                if passed
                else f"Low groundedness: {score:.2f} < {threshold}. "
                     f"Response may contain unsupported claims."
            ),
            details={"mode": "simple_overlap", "score": score, "threshold": threshold},
        )

    async def _llm_evaluate(
        self, result: AgentResult, case: EvalCase, context_docs: str
    ) -> EvaluatorScore:
        """Use LLM-as-judge for hallucination detection."""
        from agentbench.evaluators.llm_judge import LLMJudgeEvaluator

        judge = LLMJudgeEvaluator(config={
            **self.config,
            "criteria": "groundedness",
            "system_prompt": (
                "You are an expert fact-checker. Given a CONTEXT and an AGENT RESPONSE, "
                "determine whether the response introduces claims not supported by the context. "
                "Return JSON: {\"score\": 0.0-1.0, \"reason\": \"explanation\", \"passed\": true/false}"
            ),
        })
        return await judge.evaluate(result, case)
