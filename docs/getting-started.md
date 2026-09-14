# Getting Started

## Installation

### From PyPI

```bash
pip install agentbench
```

### From source

```bash
git clone https://github.com/your-org/agentbench.git
cd agentbench
pip install -e ".[dev]"
```

## Your first evaluation

### 1. Create an evaluation file

```yaml
# evals/my-agent.yaml
name: My First Evaluation

agent:
  type: echo  # No API key needed

cases:
  - id: hello-001
    input: "Hello, how are you?"
    expected:
      contains:
        - hello
```

### 2. Run it

```bash
agentbench run evals/my-agent.yaml
```

### 3. See results

```
Running My First Evaluation (1 case)

  AgentBench

  ✓ contains          100.0%

  Overall Score       100.0%
  Passed    1
```

## Connecting your real agent

Replace `echo` with your agent type:

```yaml
agent:
  type: openai
  model: gpt-4o-mini
  system_prompt: "You are a helpful assistant."
```

```bash
export OPENAI_API_KEY=sk-...
agentbench run evals/my-agent.yaml
```

## Next steps

- [Architecture](architecture.md)
- [Evaluators reference](evaluators.md)
- [Custom evaluators](custom-evaluator.md)
- [CI/CD integration](ci-cd.md)
- [Integrations](integrations.md)
