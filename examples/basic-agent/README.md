# Basic Agent Example

This example demonstrates AgentBench with a simple echo agent.
**No API keys required.**

## Setup

```bash
pip install agentbench
```

## Run the evaluation

```bash
agentbench run evaluation.yaml
```

## What it does

- Runs 5 evaluation cases against a deterministic echo agent
- Checks that responses contain expected phrases
- Shows pass/fail results with scores
- Requires no external services or API keys

## Customize

Replace the `echo` adapter with your own agent:

```yaml
agent:
  type: http
  url: http://localhost:8000/agent/run
```

or

```yaml
agent:
  type: openai
  model: gpt-4o-mini
  system_prompt: "You are a helpful customer support agent."
```
