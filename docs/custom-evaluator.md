# Custom Evaluators

AgentBench is designed to make adding custom evaluators easy.

## Basic example

```python
# my_evaluators.py
from agentbench.core.base_evaluator import BaseEvaluator
from agentbench.core.models import AgentResult, EvalCase, EvaluatorScore
from agentbench.evaluators.registry import EvaluatorRegistry


class PolitenessEvaluator(BaseEvaluator):
    """Checks that the agent response includes a courtesy phrase."""

    name = "politeness"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        courtesy_phrases = ["please", "thank you", "happy to help", "glad to assist"]
        response_lower = result.response.lower()
        is_polite = any(phrase in response_lower for phrase in courtesy_phrases)

        return EvaluatorScore(
            evaluator=self.name,
            score=1.0 if is_polite else 0.5,
            passed=is_polite,
            reason="Contains courtesy phrase" if is_polite else "No courtesy phrase detected",
        )


# Register it
EvaluatorRegistry.register(PolitenessEvaluator)
```

## Using it

```python
# In your evaluation script
import my_evaluators  # This registers PolitenessEvaluator

from agentbench.datasets.loader import DatasetLoader
from agentbench.runners.evaluation_runner import EvaluationRunner

dataset = DatasetLoader.from_yaml("evals/my-eval.yaml")
runner = EvaluationRunner(dataset=dataset)
run = await runner.run()
```

## In YAML (after registration)

```yaml
evaluators:
  - politeness
  - contains
```

## Accessing configuration

```python
class ConfigurableEvaluator(BaseEvaluator):
    name = "configurable"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        threshold = self.config.get("threshold", 0.7)
        keywords = self.config.get("keywords", [])
        # ... your logic
```

```yaml
evaluators:
  - name: configurable
    threshold: 0.8
    keywords: [helpful, clear, concise]
```
