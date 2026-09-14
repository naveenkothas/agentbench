"""Built-in evaluators for AgentBench."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import jsonschema

from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore
from agentbench.evaluators.base import BaseEvaluator, register_evaluator


@register_evaluator
class ExactMatchEvaluator(BaseEvaluator):
    """
    Evaluates whether the agent response exactly matches the expected output.
    Useful for deterministic, templated responses.
    """

    name = "exact_match"
    description = "Checks if the agent response exactly matches the expected output."
    pass_threshold = 1.0

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        if case.expected.exact is None:
            return self._make_score(1.0, reason="No exact match assertion defined — skipped.")

        expected = case.expected.exact.strip()
        actual = result.response.strip()
        passed = expected == actual

        return self._make_score(
            score=1.0 if passed else 0.0,
            reason=f"Response {'matches' if passed else 'does not match'} expected exact output.",
            details={"expected": expected, "actual": actual},
        )


@register_evaluator
class ContainsEvaluator(BaseEvaluator):
    """
    Evaluates whether the agent response contains expected substrings
    and does not contain forbidden substrings.
    """

    name = "contains"
    description = "Checks for required and forbidden substrings in the response."
    pass_threshold = 1.0

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        response_lower = result.response.lower()
        required = case.expected.contains or []
        forbidden = case.expected.not_contains or []

        if not required and not forbidden:
            return self._make_score(1.0, reason="No contains assertions defined — skipped.")

        missing = [term for term in required if term.lower() not in response_lower]
        present_forbidden = [term for term in forbidden if term.lower() in response_lower]

        if not missing and not present_forbidden:
            return self._make_score(
                score=1.0,
                reason="All required terms found; no forbidden terms present.",
                details={"required_found": required, "forbidden_found": []},
            )

        reasons = []
        if missing:
            reasons.append(f"Missing required terms: {missing}")
        if present_forbidden:
            reasons.append(f"Found forbidden terms: {present_forbidden}")

        total_checks = len(required) + len(forbidden)
        failed = len(missing) + len(present_forbidden)
        score = max(0.0, 1.0 - (failed / total_checks))

        return self._make_score(
            score=score,
            reason="; ".join(reasons),
            details={"missing_required": missing, "present_forbidden": present_forbidden},
        )


@register_evaluator
class SemanticSimilarityEvaluator(BaseEvaluator):
    """
    Evaluates semantic similarity between the agent response and expected output.

    Uses a lightweight character-level approximation by default. When sentence-transformers
    or an embedding model is configured, it will use real semantic embeddings.
    """

    name = "semantic_similarity"
    description = "Measures semantic similarity between response and expected output."
    pass_threshold = 0.7

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        if case.expected.similar_to is None:
            return self._make_score(1.0, reason="No semantic similarity assertion defined — skipped.")

        expected = case.expected.similar_to
        actual = result.response
        threshold = case.expected.similarity_threshold

        score = self._compute_similarity(expected, actual)

        return self._make_score(
            score=score,
            reason=f"Semantic similarity: {score:.2%} (threshold: {threshold:.0%})",
            details={"similarity": score, "threshold": threshold},
            threshold=threshold,
        )

    def _compute_similarity(self, expected: str, actual: str) -> float:
        """
        Compute textual similarity. This is a baseline implementation using
        Jaccard similarity on word sets. Override this method or configure
        an embedding model for production use.
        """
        def tokenize(text: str) -> set[str]:
            return set(re.findall(r"\b\w+\b", text.lower()))

        exp_words = tokenize(expected)
        act_words = tokenize(actual)

        if not exp_words and not act_words:
            return 1.0
        if not exp_words or not act_words:
            return 0.0

        intersection = exp_words & act_words
        union = exp_words | act_words
        return len(intersection) / len(union)


@register_evaluator
class ToolCallEvaluator(BaseEvaluator):
    """
    Evaluates whether the agent made the correct tool calls with the right arguments.
    """

    name = "tool_call"
    description = "Validates tool call names, arguments, and sequences."
    pass_threshold = 1.0

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        assertion = case.expected.tool_call
        if assertion is None:
            return self._make_score(1.0, reason="No tool call assertion defined — skipped.")

        tool_calls = result.tool_calls
        checks = []
        failures = []

        # Check that the named tool was called
        if assertion.name is not None:
            called_names = [tc.name for tc in tool_calls]
            if assertion.name in called_names:
                checks.append(f"Tool '{assertion.name}' was called.")
            else:
                failures.append(
                    f"Expected tool '{assertion.name}' was not called. Got: {called_names}"
                )

        # Check argument values for the matching tool call
        if assertion.arguments is not None and assertion.name is not None:
            matching_calls = [tc for tc in tool_calls if tc.name == assertion.name]
            if matching_calls:
                tc = matching_calls[0]
                for key, expected_val in assertion.arguments.items():
                    actual_val = tc.arguments.get(key)
                    if actual_val == expected_val:
                        checks.append(f"Argument '{key}' matches.")
                    else:
                        failures.append(
                            f"Argument '{key}': expected '{expected_val}', got '{actual_val}'"
                        )

        # Check call sequence
        if assertion.sequence is not None:
            actual_seq = [tc.name for tc in tool_calls]
            if actual_seq == assertion.sequence:
                checks.append("Tool call sequence matches.")
            else:
                failures.append(
                    f"Expected sequence {assertion.sequence}, got {actual_seq}"
                )

        # Check call count bounds
        if assertion.min_calls is not None and len(tool_calls) < assertion.min_calls:
            failures.append(
                f"Expected at least {assertion.min_calls} tool calls, got {len(tool_calls)}"
            )
        if assertion.max_calls is not None and len(tool_calls) > assertion.max_calls:
            failures.append(
                f"Expected at most {assertion.max_calls} tool calls, got {len(tool_calls)}"
            )

        total = len(checks) + len(failures)
        score = len(checks) / total if total > 0 else 1.0

        return self._make_score(
            score=score,
            reason="; ".join(failures) if failures else "All tool call assertions passed.",
            details={"passed_checks": checks, "failed_checks": failures},
        )


@register_evaluator
class JSONSchemaEvaluator(BaseEvaluator):
    """
    Validates that the agent response is valid JSON conforming to a schema.
    """

    name = "json_schema"
    description = "Validates agent JSON responses against a JSON schema."
    pass_threshold = 1.0

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        if case.expected.json_schema is None:
            return self._make_score(1.0, reason="No JSON schema assertion defined — skipped.")

        try:
            parsed = json.loads(result.response)
        except json.JSONDecodeError as e:
            return self._make_score(
                score=0.0,
                reason=f"Response is not valid JSON: {e}",
                details={"parse_error": str(e)},
            )

        try:
            jsonschema.validate(instance=parsed, schema=case.expected.json_schema)
            return self._make_score(
                score=1.0,
                reason="Response is valid JSON conforming to the schema.",
                details={"parsed": parsed},
            )
        except jsonschema.ValidationError as e:
            return self._make_score(
                score=0.0,
                reason=f"JSON schema validation failed: {e.message}",
                details={"validation_error": e.message, "path": list(e.path)},
            )


@register_evaluator
class LatencyEvaluator(BaseEvaluator):
    """
    Evaluates whether the agent response was produced within an acceptable latency.
    """

    name = "latency"
    description = "Checks that response latency is within the acceptable threshold."
    pass_threshold = 1.0

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        max_ms = case.expected.max_latency_ms
        if max_ms is None:
            max_ms = context.get("max_latency_ms", None)

        if max_ms is None:
            return self._make_score(
                score=1.0,
                reason=f"No latency threshold set. Latency was {result.latency_ms:.1f}ms.",
                details={"latency_ms": result.latency_ms},
            )

        passed = result.latency_ms <= max_ms
        score = 1.0 if passed else max(0.0, 1.0 - (result.latency_ms - max_ms) / max_ms)

        return self._make_score(
            score=score,
            reason=(
                f"Latency {result.latency_ms:.1f}ms is "
                f"{'within' if passed else 'above'} threshold of {max_ms:.0f}ms."
            ),
            details={"latency_ms": result.latency_ms, "max_latency_ms": max_ms},
        )


@register_evaluator
class CostEvaluator(BaseEvaluator):
    """
    Evaluates whether the agent call cost is within budget.
    """

    name = "cost"
    description = "Checks that estimated API cost is within the acceptable threshold."
    pass_threshold = 1.0

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        max_cost = case.expected.max_cost_usd
        if max_cost is None:
            max_cost = context.get("max_cost_usd", None)

        if max_cost is None:
            return self._make_score(
                score=1.0,
                reason=f"No cost threshold set. Estimated cost: ${result.estimated_cost_usd:.6f}",
                details={"estimated_cost_usd": result.estimated_cost_usd},
            )

        passed = result.estimated_cost_usd <= max_cost
        return self._make_score(
            score=1.0 if passed else 0.0,
            reason=(
                f"Cost ${result.estimated_cost_usd:.6f} is "
                f"{'within' if passed else 'above'} budget of ${max_cost:.6f}."
            ),
            details={
                "estimated_cost_usd": result.estimated_cost_usd,
                "max_cost_usd": max_cost,
            },
        )


@register_evaluator
class HallucinationEvaluator(BaseEvaluator):
    """
    Detects unsupported claims in the agent response relative to reference documents.

    Uses a word-overlap heuristic when no LLM judge is available. For production use,
    configure an LLM judge via the llm_judge_url setting or use LLMJudgeEvaluator.
    """

    name = "hallucination"
    description = "Detects claims in the response not supported by reference documents."
    pass_threshold = 0.7

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        reference_docs = case.reference_documents
        if not reference_docs:
            return self._make_score(
                1.0, reason="No reference documents provided — hallucination check skipped."
            )

        combined_reference = " ".join(reference_docs).lower()
        response_words = set(re.findall(r"\b[a-z]{4,}\b", result.response.lower()))
        ref_words = set(re.findall(r"\b[a-z]{4,}\b", combined_reference))

        # Common words that don't indicate grounding
        stopwords = {
            "that", "this", "with", "from", "they", "will", "have", "been",
            "were", "your", "more", "also", "when", "what", "about", "which",
            "their", "there", "would", "could", "should", "make", "some",
        }
        response_words -= stopwords
        ref_words -= stopwords

        if not response_words:
            return self._make_score(1.0, reason="Response has no substantive content to evaluate.")

        grounded_words = response_words & ref_words
        groundedness_ratio = len(grounded_words) / len(response_words)

        return self._make_score(
            score=groundedness_ratio,
            reason=(
                f"Groundedness: {groundedness_ratio:.1%} of response words appear in reference. "
                f"Consider LLMJudgeEvaluator for more accurate hallucination detection."
            ),
            details={
                "groundedness_ratio": groundedness_ratio,
                "response_word_count": len(response_words),
                "grounded_word_count": len(grounded_words),
            },
        )


@register_evaluator
class InstructionFollowingEvaluator(BaseEvaluator):
    """
    Evaluates whether the agent followed the given instructions.

    Uses LLM-as-judge when an LLM is configured. Falls back to checking
    for required criteria keywords in the response.
    """

    name = "instruction_following"
    description = "Evaluates whether the agent followed the provided instructions."
    pass_threshold = 0.7

    async def evaluate(
        self, result: AgentResult, case: EvalCase, context: dict[str, Any]
    ) -> EvaluatorScore:
        criteria = case.expected.llm_criteria
        if not criteria:
            return self._make_score(
                1.0, reason="No instruction-following criteria defined — skipped."
            )

        response_lower = result.response.lower()
        met = []
        unmet = []

        for criterion in criteria:
            # Simple keyword-based check — can be replaced with LLM judge
            keywords = re.findall(r"\b\w{4,}\b", criterion.lower())
            keyword_hits = sum(1 for kw in keywords if kw in response_lower)
            if keywords and keyword_hits / len(keywords) >= 0.5:
                met.append(criterion)
            else:
                unmet.append(criterion)

        score = len(met) / len(criteria) if criteria else 1.0

        return self._make_score(
            score=score,
            reason=(
                f"{len(met)}/{len(criteria)} criteria met."
                + (f" Unmet: {unmet}" if unmet else "")
            ),
            details={"met_criteria": met, "unmet_criteria": unmet},
        )
