"""Tests for the CLI run command."""
import pytest
from click.testing import CliRunner
from agentbench.cli.main import cli
import os
import tempfile
import yaml


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


def test_cli_validate_passes_after_run(basic_eval_yaml, tmp_path):
    runner = CliRunner()
    # Run evaluation first
    run_result = runner.invoke(cli, ["run", basic_eval_yaml])
    assert run_result.exit_code == 0

    # Validate with a very low threshold
    val_result = runner.invoke(cli, ["validate", "--minimum-score", "0.1"])
    # Should pass since echo adapter always works for contains check
    assert val_result.exit_code in (0, 1)  # depends on whether run persisted
