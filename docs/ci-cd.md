# CI/CD Integration

## GitHub Actions

### Basic setup

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install AgentBench
        run: pip install agentbench

      - name: Run evaluations
        run: agentbench run evals/

      - name: Quality gate (90% minimum)
        run: agentbench validate --minimum-score 0.90
```

### With LLM evaluators

```yaml
      - name: Run evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: agentbench run evals/
```

Store your API key as a GitHub Actions secret: Settings > Secrets > New repository secret.

### Quality gate failure output

```
✗ Quality gate failed

  Score:    87.4%
  Required: 90.0%
```

Exit code is non-zero — the CI job fails and blocks merge.

## GitLab CI

```yaml
agent-evaluation:
  image: python:3.11
  script:
    - pip install agentbench
    - agentbench run evals/
    - agentbench validate --minimum-score 0.90
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
```

## Docker

```bash
# Run evaluations in Docker
docker build -t agentbench .
docker run --env OPENAI_API_KEY=$OPENAI_API_KEY agentbench agentbench run evals/
```
