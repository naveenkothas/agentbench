---
name: New Evaluator
about: Add a new evaluation metric or check
labels: evaluator, good first issue
---

## Evaluator name

<!-- e.g., toxicity, semantic_similarity, readability -->

## What does it check?

<!-- Describe what the evaluator measures -->

## Example YAML

```yaml
evaluators:
  - name: my_evaluator
    threshold: 0.8
```

## Implementation sketch

```python
class MyEvaluator(BaseEvaluator):
    name = "my_evaluator"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        score = ...  # your logic
        return EvaluatorScore(evaluator=self.name, score=score, passed=score >= 0.7, reason="...")
```

## Are you willing to implement this?

- [ ] Yes, I will submit a PR
- [ ] I can help test it
