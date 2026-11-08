"""Tool call evaluator — validates that the agent called the right tools."""
from __future__ import annotations

from typing import Any

from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore


def _match_args(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str]:
    """Check that all expected args appear in actual args with correct values."""
    for key, exp_val in expected.items():
        if key not in actual:
            return False, f"Missing argument: {key!r}"
        if exp_val is not None and actual[key] != exp_val:
            return False, f"Argument {key!r}: expected {exp_val!r}, got {actual[key]!r}"
    return True, ""


class ToolCallEvaluator(BaseEvaluator):
    """
    Validates agent tool/function calls.

    Checks:
    - Correct tool name was called
    - Required arguments were provided
    - Argument values match expectations
    - Tool call sequence order (if sequence defined)
    """

    name = "tool_call"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        # Single tool call check
        if case.expected.tool_call:
            return await self._evaluate_single(result, case.expected.tool_call)

        # Sequence check
        if case.expected.tool_call_sequence:
            return await self._evaluate_sequence(result, case.expected.tool_call_sequence)

        return EvaluatorScore(
            evaluator=self.name,
            score=1.0,
            passed=True,
            reason="No tool call expectations defined.",
        )

    async def _evaluate_single(
        self, result: AgentResult, expected: dict[str, Any]
    ) -> EvaluatorScore:
        expected_name = expected.get("name")
        expected_args = expected.get("arguments", {})

        if not result.tool_calls:
            return EvaluatorScore(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason=f"Agent made no tool calls. Expected: {expected_name!r}",
            )

        # Find a matching tool call
        for tc in result.tool_calls:
            if tc.name == expected_name:
                args_ok, args_reason = _match_args(expected_args, tc.arguments)
                if args_ok:
                    return EvaluatorScore(
                        evaluator=self.name,
                        score=1.0,
                        passed=True,
                        reason=f"Tool {expected_name!r} called with correct arguments.",
                    )
                return EvaluatorScore(
                    evaluator=self.name,
                    score=0.5,
                    passed=False,
                    reason=f"Tool {expected_name!r} called but wrong args: {args_reason}",
                )

        called = [tc.name for tc in result.tool_calls]
        return EvaluatorScore(
            evaluator=self.name,
            score=0.0,
            passed=False,
            reason=f"Expected tool {expected_name!r} not called. Called: {called}",
        )

    async def _evaluate_sequence(
        self, result: AgentResult, sequence: list[dict[str, Any]]
    ) -> EvaluatorScore:
        actual_calls = result.tool_calls
        if len(actual_calls) < len(sequence):
            return EvaluatorScore(
                evaluator=self.name,
                score=len(actual_calls) / len(sequence),
                passed=False,
                reason=f"Expected {len(sequence)} tool calls, got {len(actual_calls)}",
            )

        issues: list[str] = []
        for i, (expected, actual) in enumerate(zip(sequence, actual_calls)):
            if expected.get("name") and actual.name != expected["name"]:
                issues.append(
                    f"Step {i+1}: expected {expected['name']!r}, got {actual.name!r}"
                )
            elif expected.get("arguments"):
                ok, reason = _match_args(expected["arguments"], actual.arguments)
                if not ok:
                    issues.append(f"Step {i+1}: {reason}")

        score = 1.0 - (len(issues) / len(sequence))
        return EvaluatorScore(
            evaluator=self.name,
            score=score,
            passed=len(issues) == 0,
            reason="; ".join(issues) if issues else "Tool call sequence matched.",
        )
