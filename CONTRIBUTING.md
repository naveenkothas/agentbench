# Contributing to AgentBench

Thank you for your interest in contributing! This document explains how to get involved.

## Development Setup

```bash
git clone https://github.com/your-org/agentbench.git
cd agentbench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                          # all tests
pytest tests/test_evaluators/   # evaluator tests only
pytest -v --tb=short            # verbose output
pytest --cov=agentbench         # with coverage
```

## Code Style

We use Ruff for linting and Black for formatting:

```bash
ruff check agentbench/
black agentbench/ tests/
mypy agentbench/
```

## Adding an Evaluator

1. Create `agentbench/evaluators/my_evaluator.py`
2. Subclass `BaseEvaluator`
3. Set `name = "my_evaluator"`
4. Implement `async def evaluate(self, result, case) -> EvaluatorScore`
5. Register in `agentbench/evaluators/registry.py`
6. Add tests in `tests/test_evaluators/`
7. Document in `docs/evaluators.md`

```python
from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore

class MyEvaluator(BaseEvaluator):
    name = "my_evaluator"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        passed = True  # your logic here
        return EvaluatorScore(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="Passed" if passed else "Failed because...",
        )
```

## Adding an Agent Adapter

1. Create `agentbench/adapters/my_framework.py`
2. Subclass `AgentAdapter`
3. Set `adapter_type = "my_framework"`
4. Implement `async def run(self, input, context) -> AgentResult`
5. Register in `agentbench/adapters/registry.py`
6. Add an example in `examples/my-framework/`
7. Update the integration table in `README.md`

## Submitting a Pull Request

- Open an issue first for significant changes
- Write tests for new functionality
- Keep PRs focused — one feature or fix per PR
- Update documentation if needed
- All CI checks must pass

## Questions?

Open a Discussion on GitHub or file an issue.
