"""Tests for the adapter and evaluator registries."""
import pytest

from agentbench.adapters.echo import EchoAdapter
from agentbench.adapters.registry import AdapterRegistry
from agentbench.evaluators.contains import ContainsEvaluator
from agentbench.evaluators.registry import EvaluatorRegistry


def test_builtin_adapters_registered():
    assert {"echo", "http", "openai"} <= set(AdapterRegistry.list())


def test_create_adapter_passes_config():
    adapter = AdapterRegistry.create("echo", {"latency_ms": 0})
    assert isinstance(adapter, EchoAdapter)
    assert adapter.config == {"latency_ms": 0}


def test_unknown_adapter_raises():
    with pytest.raises(ValueError, match="Unknown adapter type"):
        AdapterRegistry.get("does-not-exist")


def test_builtin_evaluators_registered():
    expected = {"exact_match", "contains", "tool_call", "latency", "cost",
                "json_schema", "hallucination", "llm_judge"}
    assert expected <= set(EvaluatorRegistry.list())


def test_create_evaluator_passes_config():
    evaluator = EvaluatorRegistry.create("contains", {"threshold": 0.5})
    assert isinstance(evaluator, ContainsEvaluator)
    assert evaluator.config == {"threshold": 0.5}


def test_unknown_evaluator_raises():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        EvaluatorRegistry.get("does-not-exist")
