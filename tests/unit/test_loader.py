"""Tests for the YAML dataset loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from agentbench.datasets.loader import DatasetLoadError, load_suite_from_yaml, load_suites_from_directory


def write_yaml(data: dict, path: Path) -> None:
    path.write_text(yaml.dump(data), encoding="utf-8")


class TestLoadSuiteFromYAML:
    def test_loads_valid_suite(self, tmp_path):
        data = {
            "name": "Test Suite",
            "agent": {"type": "echo", "response": "Hello"},
            "cases": [
                {
                    "id": "case-001",
                    "input": "hello",
                    "expected": {"contains": ["ello"]},
                }
            ],
        }
        path = tmp_path / "suite.yaml"
        write_yaml(data, path)
        suite = load_suite_from_yaml(path)
        assert suite.name == "Test Suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "case-001"

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(DatasetLoadError, match="not found"):
            load_suite_from_yaml(tmp_path / "missing.yaml")

    def test_raises_for_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("{ invalid yaml: [", encoding="utf-8")
        with pytest.raises(DatasetLoadError):
            load_suite_from_yaml(path)

    def test_default_agent_type_is_echo(self, tmp_path):
        data = {"name": "Minimal", "cases": [{"id": "c1", "input": "hi"}]}
        path = tmp_path / "minimal.yaml"
        write_yaml(data, path)
        suite = load_suite_from_yaml(path)
        assert suite.agent.type == "echo"

    def test_parses_tool_call_assertion(self, tmp_path):
        data = {
            "name": "Tool Suite",
            "agent": {"type": "echo"},
            "cases": [
                {
                    "id": "tc-001",
                    "input": "test",
                    "expected": {
                        "tool_call": {
                            "name": "my_tool",
                            "arguments": {"key": "value"},
                        }
                    },
                }
            ],
        }
        path = tmp_path / "tool.yaml"
        write_yaml(data, path)
        suite = load_suite_from_yaml(path)
        tc = suite.cases[0].expected.tool_call
        assert tc is not None
        assert tc.name == "my_tool"
        assert tc.arguments == {"key": "value"}

    def test_parses_reference_documents(self, tmp_path):
        data = {
            "name": "RAG Suite",
            "agent": {"type": "echo"},
            "cases": [
                {
                    "id": "rag-001",
                    "input": "test",
                    "reference_documents": ["doc 1", "doc 2"],
                }
            ],
        }
        path = tmp_path / "rag.yaml"
        write_yaml(data, path)
        suite = load_suite_from_yaml(path)
        assert len(suite.cases[0].reference_documents) == 2


class TestLoadSuitesFromDirectory:
    def test_loads_multiple_suites(self, tmp_path):
        for i in range(3):
            data = {
                "name": f"Suite {i}",
                "agent": {"type": "echo"},
                "cases": [{"id": f"c{i}", "input": "test"}],
            }
            (tmp_path / f"suite_{i}.yaml").write_text(yaml.dump(data), encoding="utf-8")

        suites = load_suites_from_directory(tmp_path)
        assert len(suites) == 3

    def test_raises_for_non_directory(self, tmp_path):
        file = tmp_path / "not_a_dir.yaml"
        file.write_text("name: x\nagent:\n  type: echo\ncases: []")
        with pytest.raises(DatasetLoadError, match="Not a directory"):
            load_suites_from_directory(file)

    def test_returns_empty_for_no_yaml_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# hello")
        suites = load_suites_from_directory(tmp_path)
        assert suites == []
