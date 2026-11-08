"""
Core data models for AgentBench.

These Pydantic models form the spine of the evaluation pipeline.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a single tool/function call made by an agent."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None
    latency_ms: float | None = None


class AgentResult(BaseModel):
    """The raw output from running an agent on a single input."""

    response: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ExpectedOutput(BaseModel):
    """Defines what a correct agent response looks like for a test case."""

    response: str | None = None
    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    tool_call: dict[str, Any] | None = None
    tool_call_sequence: list[dict[str, Any]] = Field(default_factory=list)
    json_schema: dict[str, Any] | None = None
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None
    semantic_similarity_threshold: float | None = None


class EvalCase(BaseModel):
    """A single evaluation test case."""

    id: str
    input: str
    expected: ExpectedOutput = Field(default_factory=ExpectedOutput)
    context: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Runtime fields (populated after execution)
    result: AgentResult | None = None


class EvalDataset(BaseModel):
    """A collection of evaluation test cases with agent configuration."""

    name: str
    description: str = ""
    version: str = "1.0"
    agent: dict[str, Any] = Field(default_factory=dict)
    evaluators: list[str] = Field(default_factory=list)
    cases: list[EvalCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluatorScore(BaseModel):
    """The result from a single evaluator on a single test case."""

    evaluator: str
    score: float  # 0.0 to 1.0
    passed: bool
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Aggregated evaluation result for a single test case."""

    case_id: str
    passed: bool
    scores: list[EvaluatorScore] = Field(default_factory=list)
    agent_result: AgentResult | None = None
    error: str | None = None

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalRun(BaseModel):
    """Represents a complete evaluation run across all test cases."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    dataset_name: str = ""
    agent_type: str = ""
    status: RunStatus = RunStatus.PENDING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    results: list[EvalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_cases(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def total_latency_ms(self) -> float:
        return sum(
            r.agent_result.latency_ms for r in self.results if r.agent_result
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(
            r.agent_result.estimated_cost_usd for r in self.results if r.agent_result
        )

    @property
    def avg_score_by_evaluator(self) -> dict[str, float]:
        """Compute average score per evaluator across all results."""
        scores: dict[str, list[float]] = {}
        for result in self.results:
            for score in result.scores:
                scores.setdefault(score.evaluator, []).append(score.score)
        return {k: sum(v) / len(v) for k, v in scores.items()}
