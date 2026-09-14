# Architecture

## Overview

AgentBench is structured around a clean pipeline pattern:

```
Dataset → Adapter → Agent → Evaluators → EvalRun → Report
```

## Core components

### Models (`agentbench.core.models`)

The central data structures:

- `EvalCase` — a single test case with input, expected output, and context
- `EvalDataset` — a collection of cases with agent and evaluator configuration
- `AgentResult` — what an agent returns: response, tool calls, latency, cost
- `EvaluatorScore` — what an evaluator returns: score [0,1], pass/fail, reason
- `EvalResult` — all evaluator scores for one case
- `EvalRun` — all results across an entire evaluation run

### Agent Adapters (`agentbench.adapters`)

Adapters bridge AgentBench with specific agent implementations.
Implement `AgentAdapter.run()` to connect any agent:

```python
class AgentAdapter(ABC):
    async def run(self, input: str, context: dict) -> AgentResult: ...
```

### Evaluators (`agentbench.evaluators`)

Each evaluator takes an `AgentResult` and `EvalCase` and returns an `EvaluatorScore`:

```python
class BaseEvaluator(ABC):
    async def evaluate(self, result: AgentResult, case: EvalCase) -> EvaluatorScore: ...
```

### Evaluation Runner (`agentbench.runners`)

Orchestrates the pipeline with async concurrency:

```python
runner = EvaluationRunner(dataset=dataset, max_concurrency=5)
run = await runner.run()
```

### Storage (`agentbench.storage`)

`SQLiteStore` persists `EvalRun` objects to a local SQLite database.
Swap for `PostgresStore` for production use (see roadmap).

### Reporting (`agentbench.reporting`)

`ConsoleReporter` renders results with Rich. The `FastAPI` app in `agentbench.api`
serves results as a web dashboard.
