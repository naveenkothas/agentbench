"""Tests for the CLI run command."""

import pytest
import yaml
from click.testing import CliRunner

from agentbench.cli.main import cli


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Keep the CLI's agentbench.db out of the repository."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def basic_eval_yaml(tmp_path):
    data = {
        "name": "CLI Test",
        "agent": {"type": "echo", "response_template": "Echo: {input}"},
        "evaluators": ["contains"],
        "cases": [
            {
                "id": "cli-001",
                "input": "Hello",
                "expected": {"contains": ["Echo"]},
            }
        ],
    }
    yaml_file = tmp_path / "test_eval.yaml"
    yaml_file.write_text(yaml.dump(data))
    return str(yaml_file)


def test_cli_run_succeeds(basic_eval_yaml):
    runner = CliRunner()
    result = runner.invoke(cli, ["run", basic_eval_yaml])
    assert result.exit_code == 0, result.output


def test_cli_run_shows_score(basic_eval_yaml):
    runner = CliRunner()
    result = runner.invoke(cli, ["run", basic_eval_yaml])
    assert "100" in result.output or "Pass" in result.output or "Overall" in result.output


def test_cli_validate_passes_after_run(basic_eval_yaml):
    runner = CliRunner()
    assert runner.invoke(cli, ["run", basic_eval_yaml]).exit_code == 0

    result = runner.invoke(cli, ["validate", "--minimum-score", "0.9"])
    assert result.exit_code == 0, result.output
    assert "passed" in result.output


def test_cli_validate_fails_below_threshold(tmp_path):
    data = {
        "name": "Failing",
        "agent": {"type": "echo"},
        "evaluators": ["contains"],
        "cases": [{"id": "f-1", "input": "Hello", "expected": {"contains": ["absent"]}}],
    }
    path = tmp_path / "failing.yaml"
    path.write_text(yaml.dump(data))
    runner = CliRunner()

    assert runner.invoke(cli, ["run", str(path)]).exit_code == 1
    result = runner.invoke(cli, ["validate", "--minimum-score", "0.9"])
    assert result.exit_code == 1
    assert "failed" in result.output


def test_cli_run_reports_adapter_misconfiguration(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    data = {
        "name": "Azure Unconfigured",
        "agent": {"type": "azure_openai"},
        "cases": [{"id": "a-1", "input": "Hi", "expected": {"contains": ["x"]}}],
    }
    path = tmp_path / "azure.yaml"
    path.write_text(yaml.dump(data))

    result = CliRunner().invoke(cli, ["run", str(path)])
    assert result.exit_code == 1
    assert "AZURE_OPENAI_ENDPOINT" in result.output
    assert not isinstance(result.exception, ValueError)
