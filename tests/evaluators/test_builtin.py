"""Unit tests for built-in evaluators."""

import pytest

from agentbench.core.models import (
    AgentResult,
    Assertion,
    EvalCase,
    ToolCall,
    ToolCallAssertion,
)
from agentbench.evaluators.builtin import (
    ContainsEvaluator,
    CostEvaluator,
    ExactMatchEvaluator,
    HallucinationEvaluator,
    InstructionFollowingEvaluator,
    JSONSchemaEvaluator,
    LatencyEvaluator,
    SemanticSimilarityEvaluator,
    ToolCallEvaluator,
)


def make_result(response: str = "", **kwargs) -> AgentResult:
    return AgentResult(response=response, **kwargs)


def make_case(case_id: str = "test-001", input: str = "test input", **kwargs) -> EvalCase:
    expected = kwargs.pop("expected", Assertion())
    return EvalCase(id=case_id, input=input, expected=expected, **kwargs)


# ────────────────────────────────────────────────────────────────
# ExactMatchEvaluator
# ────────────────────────────────────────────────────────────────

class TestExactMatchEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExactMatchEvaluator()

    @pytest.mark.asyncio
    async def test_passes_on_exact_match(self, evaluator):
        result = make_result("Hello, world!")
        case = make_case(expected=Assertion(exact="Hello, world!"))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True
        assert score.score == 1.0

    @pytest.mark.asyncio
    async def test_fails_on_mismatch(self, evaluator):
        result = make_result("Hello, earth!")
        case = make_case(expected=Assertion(exact="Hello, world!"))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False
        assert score.score == 0.0

    @pytest.mark.asyncio
    async def test_skipped_when_no_assertion(self, evaluator):
        result = make_result("anything")
        case = make_case(expected=Assertion())
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, evaluator):
        result = make_result("  hello  ")
        case = make_case(expected=Assertion(exact="  hello  "))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True


# ────────────────────────────────────────────────────────────────
# ContainsEvaluator
# ────────────────────────────────────────────────────────────────

class TestContainsEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ContainsEvaluator()

    @pytest.mark.asyncio
    async def test_passes_when_all_terms_present(self, evaluator):
        result = make_result("I can help you with your refund request today.")
        case = make_case(expected=Assertion(contains=["refund", "help"]))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_fails_when_term_missing(self, evaluator):
        result = make_result("Sure, I can help.")
        case = make_case(expected=Assertion(contains=["refund"]))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False

    @pytest.mark.asyncio
    async def test_fails_when_forbidden_term_present(self, evaluator):
        result = make_result("There was an error processing your request.")
        case = make_case(expected=Assertion(not_contains=["error"]))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False

    @pytest.mark.asyncio
    async def test_case_insensitive(self, evaluator):
        result = make_result("REFUND available now")
        case = make_case(expected=Assertion(contains=["refund"]))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_skipped_when_no_assertions(self, evaluator):
        result = make_result("anything")
        case = make_case(expected=Assertion())
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True


# ────────────────────────────────────────────────────────────────
# ToolCallEvaluator
# ────────────────────────────────────────────────────────────────

class TestToolCallEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ToolCallEvaluator()

    @pytest.mark.asyncio
    async def test_passes_when_correct_tool_called(self, evaluator):
        result = make_result(
            "Checking your order...",
            tool_calls=[ToolCall(name="get_order_status", arguments={"order_id": "123"})],
        )
        case = make_case(
            expected=Assertion(tool_call=ToolCallAssertion(name="get_order_status"))
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_fails_when_wrong_tool_called(self, evaluator):
        result = make_result(
            "...",
            tool_calls=[ToolCall(name="send_email", arguments={})],
        )
        case = make_case(
            expected=Assertion(tool_call=ToolCallAssertion(name="get_order_status"))
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False

    @pytest.mark.asyncio
    async def test_validates_arguments(self, evaluator):
        result = make_result(
            "...",
            tool_calls=[ToolCall(name="get_order_status", arguments={"order_id": "123"})],
        )
        case = make_case(
            expected=Assertion(
                tool_call=ToolCallAssertion(
                    name="get_order_status",
                    arguments={"order_id": "123"},
                )
            )
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_fails_on_wrong_argument_value(self, evaluator):
        result = make_result(
            "...",
            tool_calls=[ToolCall(name="get_order_status", arguments={"order_id": "456"})],
        )
        case = make_case(
            expected=Assertion(
                tool_call=ToolCallAssertion(
                    name="get_order_status",
                    arguments={"order_id": "123"},
                )
            )
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False

    @pytest.mark.asyncio
    async def test_validates_sequence(self, evaluator):
        result = make_result(
            "...",
            tool_calls=[
                ToolCall(name="search", arguments={}),
                ToolCall(name="get_order_status", arguments={}),
            ],
        )
        case = make_case(
            expected=Assertion(
                tool_call=ToolCallAssertion(sequence=["search", "get_order_status"])
            )
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_min_calls_enforced(self, evaluator):
        result = make_result("...", tool_calls=[])
        case = make_case(
            expected=Assertion(tool_call=ToolCallAssertion(min_calls=1))
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False


# ────────────────────────────────────────────────────────────────
# JSONSchemaEvaluator
# ────────────────────────────────────────────────────────────────

class TestJSONSchemaEvaluator:
    @pytest.fixture
    def evaluator(self):
        return JSONSchemaEvaluator()

    @pytest.mark.asyncio
    async def test_passes_valid_json(self, evaluator):
        result = make_result('{"status": "ok", "count": 5}')
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["status"],
        }
        case = make_case(expected=Assertion(json_schema=schema))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_fails_invalid_json(self, evaluator):
        result = make_result("not json at all")
        schema = {"type": "object"}
        case = make_case(expected=Assertion(json_schema=schema))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False

    @pytest.mark.asyncio
    async def test_fails_schema_violation(self, evaluator):
        result = make_result('{"count": "not-a-number"}')
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        case = make_case(expected=Assertion(json_schema=schema))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False


# ────────────────────────────────────────────────────────────────
# LatencyEvaluator
# ────────────────────────────────────────────────────────────────

class TestLatencyEvaluator:
    @pytest.fixture
    def evaluator(self):
        return LatencyEvaluator()

    @pytest.mark.asyncio
    async def test_passes_within_threshold(self, evaluator):
        result = make_result("fast response", latency_ms=150.0)
        case = make_case(expected=Assertion(max_latency_ms=500.0))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True

    @pytest.mark.asyncio
    async def test_fails_above_threshold(self, evaluator):
        result = make_result("slow response", latency_ms=2000.0)
        case = make_case(expected=Assertion(max_latency_ms=500.0))
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is False

    @pytest.mark.asyncio
    async def test_skipped_with_no_threshold(self, evaluator):
        result = make_result("response", latency_ms=9999.0)
        case = make_case(expected=Assertion())
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True


# ────────────────────────────────────────────────────────────────
# SemanticSimilarityEvaluator
# ────────────────────────────────────────────────────────────────

class TestSemanticSimilarityEvaluator:
    @pytest.fixture
    def evaluator(self):
        return SemanticSimilarityEvaluator()

    @pytest.mark.asyncio
    async def test_high_similarity(self, evaluator):
        result = make_result("The quick brown fox jumps over the lazy dog")
        case = make_case(
            expected=Assertion(
                similar_to="The quick brown fox jumps over the lazy dog",
                similarity_threshold=0.9,
            )
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.score > 0.9

    @pytest.mark.asyncio
    async def test_low_similarity(self, evaluator):
        result = make_result("apples and oranges")
        case = make_case(
            expected=Assertion(
                similar_to="The quick brown fox jumps over the lazy dog",
                similarity_threshold=0.8,
            )
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.score < 0.5


# ────────────────────────────────────────────────────────────────
# HallucinationEvaluator
# ────────────────────────────────────────────────────────────────

class TestHallucinationEvaluator:
    @pytest.fixture
    def evaluator(self):
        return HallucinationEvaluator()

    @pytest.mark.asyncio
    async def test_grounded_response(self, evaluator):
        result = make_result("You can return items within 30 days for a full refund.")
        case = make_case(
            expected=Assertion(),
            reference_documents=[
                "Items may be returned within 30 days for a full refund of the purchase price."
            ],
        )
        score = await evaluator.evaluate(result, case, {})
        assert score.score > 0.4  # Should have decent groundedness

    @pytest.mark.asyncio
    async def test_skipped_without_reference(self, evaluator):
        result = make_result("Some response without reference documents.")
        case = make_case(expected=Assertion())
        score = await evaluator.evaluate(result, case, {})
        assert score.passed is True
