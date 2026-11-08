"""Tests for DatasetLoader."""
import pytest

from agentbench.datasets.loader import DatasetLoader

SUITE_YAML = """
name: Support Suite
description: Customer support checks
version: 2
agent:
  type: echo
  response_template: "Reply: {input}"
evaluators:
  - contains
cases:
  - id: refund-001
    input: How do I get a refund?
    tags: [billing]
    expected:
      contains: [refund]
      not_contains: [error]
  - input: Where is my order?
    expected:
      tool_call:
        name: get_order_status
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_from_yaml_parses_suite(tmp_path):
    dataset = DatasetLoader.from_yaml(_write(tmp_path, "suite.yaml", SUITE_YAML))

    assert dataset.name == "Support Suite"
    assert dataset.description == "Customer support checks"
    assert dataset.version == "2"
    assert dataset.agent == {"type": "echo", "response_template": "Reply: {input}"}
    assert dataset.evaluators == ["contains"]
    assert len(dataset.cases) == 2

    first = dataset.cases[0]
    assert first.id == "refund-001"
    assert first.tags == ["billing"]
    assert first.expected.contains == ["refund"]
    assert first.expected.not_contains == ["error"]


def test_missing_case_id_gets_generated(tmp_path):
    dataset = DatasetLoader.from_yaml(_write(tmp_path, "suite.yaml", SUITE_YAML))
    assert dataset.cases[1].id == "case-2"
    assert dataset.cases[1].expected.tool_call == {"name": "get_order_status"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetLoader.from_yaml(tmp_path / "nope.yaml")


def test_name_defaults_to_file_stem(tmp_path):
    path = _write(tmp_path, "my-suite.yaml", "cases:\n  - id: a\n    input: hi\n")
    assert DatasetLoader.from_yaml(path).name == "my-suite"


@pytest.mark.parametrize(
    ("expected_yaml", "evaluator"),
    [
        ("response: hello", "exact_match"),
        ("contains: [x]", "contains"),
        ("not_contains: [x]", "contains"),
        ("tool_call: {name: lookup}", "tool_call"),
        ("max_latency_ms: 100", "latency"),
        ("max_cost_usd: 0.01", "cost"),
        ("json_schema: {type: object}", "json_schema"),
    ],
)
def test_auto_detects_evaluators(tmp_path, expected_yaml, evaluator):
    text = f"cases:\n  - id: a\n    input: hi\n    expected:\n      {expected_yaml}\n"
    dataset = DatasetLoader.from_yaml(_write(tmp_path, "s.yaml", text))
    assert evaluator in dataset.evaluators


def test_auto_detects_hallucination_from_documents(tmp_path):
    text = "cases:\n  - id: a\n    input: hi\n    context:\n      documents: [some doc]\n"
    dataset = DatasetLoader.from_yaml(_write(tmp_path, "s.yaml", text))
    assert "hallucination" in dataset.evaluators


def test_auto_detect_falls_back_to_contains(tmp_path):
    dataset = DatasetLoader.from_yaml(_write(tmp_path, "s.yaml", "cases:\n  - input: hi\n"))
    assert dataset.evaluators == ["contains"]


def test_from_directory_loads_yaml_and_yml(tmp_path):
    _write(tmp_path, "b.yaml", "name: B\ncases: []\n")
    _write(tmp_path, "a.yaml", "name: A\ncases: []\n")
    _write(tmp_path, "c.yml", "name: C\ncases: []\n")
    _write(tmp_path, "notes.txt", "ignored")

    names = [d.name for d in DatasetLoader.from_directory(tmp_path)]
    assert names == ["A", "B", "C"]


def test_repository_eval_suites_load():
    """Every suite shipped in evals/ must parse."""
    datasets = DatasetLoader.from_directory("evals")
    assert len(datasets) >= 1
    assert all(d.cases for d in datasets)
