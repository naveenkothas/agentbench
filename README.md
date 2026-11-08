# AgentBench

**Open-source evaluation and testing framework for AI agents.**

> Test your AI agent before your users do.

[![CI](https://github.com/your-org/agentbench/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/agentbench/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/agentbench)](https://pypi.org/project/agentbench/)

<!-- A GIF demo can be added here: ![AgentBench Demo](docs/assets/demo.gif) -->

---

## Why AgentBench?

You test your APIs. You test your databases. You run CI on every commit.

**But are you testing your AI agent?**

Most teams discover agent failures in production — when a real user hits an edge case, the agent hallucinates a policy, calls the wrong tool, or produces a malformed response.

AgentBench gives you a systematic way to catch these problems **before deployment**, by running structured evaluation suites against your agent and integrating the results into your CI/CD pipeline.

---

## Features

- **YAML-first evaluation datasets** — define test cases in plain text, no Python required
- **7 built-in evaluators** — exact match, contains, tool-call accuracy, hallucination, JSON schema, latency, cost
- **LLM-as-Judge** — use GPT-4o or Claude to evaluate correctness, relevance, and groundedness
- **Beautiful CLI** — rich terminal output with pass/fail analysis
- **Comparison mode** — diff two agent versions side by side
- **CI/CD quality gates** — fail the build if agent quality drops below a threshold
- **Adapter pattern** — works with OpenAI, Anthropic, LangGraph, LangChain, HTTP agents, and custom Python
- **Web dashboard** — FastAPI-based report server
- **No mandatory cloud** — run your first evaluation in under 5 minutes, no API key required

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     agentbench run                       │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │      YAML Dataset       │
              │  (cases + expectations) │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │      Agent Adapter      │  ← OpenAI / Anthropic /
              │  AgentAdapter.run()     │    LangGraph / HTTP / Custom
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    AgentResult          │
              │  response, tool_calls,  │
              │  latency, tokens, cost  │
              └────────────┬────────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │         Evaluator Pipeline          │
         │                                     │
         │  ┌──────────┐  ┌────────────────┐  │
         │  │ Contains │  │  Tool Call     │  │
         │  └──────────┘  └────────────────┘  │
         │  ┌──────────┐  ┌────────────────┐  │
         │  │ Latency  │  │ Hallucination  │  │
         │  └──────────┘  └────────────────┘  │
         │  ┌────────────────────────────┐     │
         │  │      LLM-as-Judge          │     │
         │  └────────────────────────────┘     │
         └─────────────────┬──────────────────┘
                           │
              ┌────────────▼────────────┐
              │       EvalRun           │
              │  scores, pass rate,     │
              │  failed cases, report   │
              └─────────────────────────┘
```

---

## Quick Start

### Install

```bash
pip install agentbench
```

### Write your first evaluation

```yaml
# evals/my-agent.yaml
name: Customer Support Agent

agent:
  type: echo                           # No API key needed to start

cases:
  - id: refund-001
    input: "I want a refund for my order"
    expected:
      contains:
        - refund

  - id: greeting-001
    input: "Hello!"
    expected:
      contains:
        - hello
      not_contains:
        - error
```

### Run it

```bash
agentbench run evals/my-agent.yaml
```

```
Running Customer Support Agent (2 cases)

╭────────────╮
│ AgentBench │
╰────────────╯

  ✓ contains                     100.0%

────────────────────────────────────────
  Overall Score           100.0%

  Passed      2
  Failed      0
  Total       2
  Duration    0.1s
```

**Total time from install to first result: ~2 minutes.**

---

## Connect Your Agent

### OpenAI

```yaml
agent:
  type: openai
  model: gpt-4o-mini
  system_prompt: "You are a helpful customer support agent."
  temperature: 0
```

```bash
export OPENAI_API_KEY=sk-...
agentbench run evals/my-agent.yaml
```

### Azure OpenAI

```yaml
agent:
  type: azure_openai
  deployment: gpt-4o-mini       # your Azure deployment name
  system_prompt: "You are a helpful customer support agent."
  temperature: 0
```

```bash
pip install "agentbench[azure]"
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
export AZURE_OPENAI_API_KEY=...   # or set `auth: entra` to use Microsoft Entra ID
agentbench run evals/my-agent.yaml
```

See [Integrations](docs/integrations.md#azure-openai) for Entra ID, API versions and the Azure LLM judge.

### HTTP Agent (any deployed agent)

```yaml
agent:
  type: http
  url: "https://your-agent.example.com/run"
  response_field: "output"
```

### Custom Python Agent

```python
from agentbench import AgentAdapter, AgentResult

class MyAgent(AgentAdapter):
    adapter_type = "my_agent"

    async def run(self, input: str, context: dict) -> AgentResult:
        response = await my_llm_call(input)
        return AgentResult(response=response)
```

---

## Evaluators

| Evaluator | What it checks | API key needed? |
|-----------|----------------|-----------------|
| `contains` | Response contains/excludes required strings | No |
| `exact_match` | Response exactly matches expected | No |
| `tool_call` | Agent called the right tool with correct args | No |
| `latency` | Response time within threshold | No |
| `cost` | Estimated API cost within budget | No |
| `hallucination` | Response grounded in provided context | No (LLM mode: yes) |
| `json_schema` | Structured output matches schema | No |
| `llm_judge` | LLM-evaluated correctness / relevance / safety | Yes |

### Example: LLM-as-Judge

```yaml
evaluators:
  - name: llm_judge
    criteria: correctness
    threshold: 0.8
  - name: llm_judge
    criteria: groundedness

cases:
  - id: rag-001
    input: "What is the refund policy?"
    context:
      documents:
        - "Refunds are accepted within 30 days of purchase."
    expected:
      contains:
        - "30 days"
```

### Example: Tool Call Validation

```yaml
cases:
  - id: order-lookup
    input: "Where is my order #12345?"
    expected:
      tool_call:
        name: get_order_status
        arguments:
          order_id: "12345"
```

---

## CI/CD Integration

Add AgentBench to your pipeline to catch quality regressions before they reach production.

### GitHub Actions

```yaml
# .github/workflows/agent-eval.yml
name: Agent Evaluation

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install AgentBench
        run: pip install agentbench

      - name: Run Evaluations
        run: agentbench run evals/

      - name: Quality Gate (90% minimum)
        run: agentbench validate --minimum-score 0.90
```

When the quality gate fails:

```
✗ Quality gate failed

  Score:    87.4%
  Required: 90.0%
```

Exit code is non-zero — the build fails automatically.

---

## Compare Agent Versions

```bash
agentbench list           # see run IDs
agentbench compare abc123 def456
```

```
Comparing abc123 vs def456

Metric                    abc123   def456   Change
────────────────────────────────────────────────────
Pass Rate                 88.0%    94.0%    +6.0%
contains                  91.0%    97.0%    +6.0%
hallucination             84.0%    91.0%    +7.0%
latency                   89.0%    95.0%    +6.0%
Duration (s)              24.1s    18.3s    -5.8s
Cost (USD)                $0.0240  $0.0180  -$0.0060
```

---

## Custom Evaluator

Creating a custom evaluator takes less than 10 lines:

```python
from agentbench import BaseEvaluator, AgentResult, EvalCase, EvaluatorScore
from agentbench.evaluators.registry import EvaluatorRegistry

class ToneEvaluator(BaseEvaluator):
    name = "tone"

    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore:
        is_polite = any(word in result.response.lower()
                       for word in ["please", "thank you", "happy to help"])
        return EvaluatorScore(
            evaluator=self.name,
            score=1.0 if is_polite else 0.5,
            passed=is_polite,
            reason="Polite" if is_polite else "No courtesy phrase detected",
        )

EvaluatorRegistry.register(ToneEvaluator)
```

Then use it in YAML:

```yaml
evaluators:
  - tone
  - contains
```

---

## Web Dashboard

```bash
agentbench serve
# Open http://localhost:8080
```

View all runs, scores, failed cases, latency, and cost history in a clean browser interface.

---

## Project Layout

```
agentbench/
├── agentbench/
│   ├── core/           # Models, base classes, config
│   ├── evaluators/     # Built-in evaluators + registry
│   ├── adapters/       # Agent framework adapters
│   ├── datasets/       # YAML dataset loading
│   ├── runners/        # Evaluation execution engine
│   ├── storage/        # SQLite persistence
│   ├── reporting/      # Console output, comparison
│   ├── cli/            # Click CLI commands
│   └── api/            # FastAPI web interface
├── evals/              # Sample evaluation datasets
├── examples/           # Runnable example projects
├── tests/              # Unit and integration tests
└── docs/               # Documentation
```

---

## Integrations

| Framework | Adapter | Status |
|-----------|---------|--------|
| OpenAI (direct) | `openai` | ✅ Available |
| Any HTTP agent | `http` | ✅ Available |
| Echo (testing/demo) | `echo` | ✅ Available |
| Anthropic | `anthropic` | 🔜 [Contribute](CONTRIBUTING.md) |
| Azure OpenAI | `azure_openai` | ✅ Available |
| LangGraph | `langgraph` | 🔜 [Contribute](CONTRIBUTING.md) |
| LangChain | `langchain` | 🔜 [Contribute](CONTRIBUTING.md) |
| CrewAI | `crewai` | 🔜 [Contribute](CONTRIBUTING.md) |
| Semantic Kernel | `semantic_kernel` | 🔜 [Contribute](CONTRIBUTING.md) |

Adding a new adapter is a great **good-first-issue**. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Roadmap

```
v0.1  ✅ Core evaluation engine, CLI, 7 built-in evaluators, YAML datasets
      ✅ OpenAI and HTTP adapters, SQLite persistence, web dashboard

v0.2  ✅ Azure OpenAI adapter (API key and Microsoft Entra ID)
      □  Anthropic, LangGraph adapters
      □  Evaluation comparison improvements
      □  CSV and JSON dataset formats
      □  pytest plugin (agentbench.pytest)

v0.3  □  Distributed evaluation (async queue)
      □  Advanced tracing and span capture
      □  Dataset versioning and management
      □  Prometheus metrics export
      □  PostgreSQL storage backend

v1.0  □  Production evaluation platform
      □  Plugin ecosystem
      □  Advanced agent benchmarks
      □  Evaluation-as-a-Service (optional)
```

---

## Contributing

We'd love your contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

**Good first issues:**

- [ ] Add Anthropic adapter ([#2](https://github.com/your-org/agentbench/issues/2))
- [x] Add Azure OpenAI adapter ([#3](https://github.com/your-org/agentbench/issues/3))
- [ ] Add LangGraph adapter ([#4](https://github.com/your-org/agentbench/issues/4))
- [ ] Add CrewAI adapter ([#5](https://github.com/your-org/agentbench/issues/5))
- [ ] Add toxicity evaluator ([#6](https://github.com/your-org/agentbench/issues/6))
- [ ] Add semantic similarity evaluator ([#7](https://github.com/your-org/agentbench/issues/7))
- [ ] Add CSV dataset support ([#8](https://github.com/your-org/agentbench/issues/8))
- [ ] Add PostgreSQL storage backend ([#9](https://github.com/your-org/agentbench/issues/9))
- [ ] Improve CLI output formatting ([#10](https://github.com/your-org/agentbench/issues/10))
- [ ] Add evaluation export to JSON/CSV ([#11](https://github.com/your-org/agentbench/issues/11))
- [ ] Add Prometheus metrics ([#12](https://github.com/your-org/agentbench/issues/12))
- [ ] Add pytest plugin ([#13](https://github.com/your-org/agentbench/issues/13))
- [ ] Add LangChain adapter ([#14](https://github.com/your-org/agentbench/issues/14))
- [ ] Improve documentation ([#15](https://github.com/your-org/agentbench/issues/15))
- [ ] Add Semantic Kernel adapter ([#16](https://github.com/your-org/agentbench/issues/16))

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

⭐ **If AgentBench helps you test AI agents, consider starring the repository.**

It helps other developers find the project.

🍴 **Fork the examples** to create your own evaluation suite.

💡 **Have an evaluator idea?** [Open a feature request.](https://github.com/your-org/agentbench/issues/new?template=feature_request.md)

🤝 **Want to contribute?** Check out the [good-first-issue tickets.](https://github.com/your-org/agentbench/issues?q=label%3A%22good+first+issue%22)
