# Changelog

All notable changes to AgentBench are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `azure_openai` adapter for agents on Azure OpenAI deployments, with API-key
  or Microsoft Entra ID (`auth: entra`) authentication, tool calling, and
  `AZURE_OPENAI_*` environment-variable configuration. Install with
  `pip install "agentbench[azure]"`.
- `azure_openai` provider for the `llm_judge` evaluator.
- `examples/azure-openai`.

### Fixed
- CI and Agent Evaluation workflows now install from source and pass on
  Python 3.11 and 3.12.
- Python 3.11 SyntaxError in the `llm_judge` evaluator.
- Missing `greenlet` dependency (via `sqlalchemy[asyncio]`) that broke result storage.
- `json_schema` evaluator silently skipped schema checks because `jsonschema`
  was not a dependency; it is now required.
- `SQLiteStore` now closes its database connections (`close()` and `async with`).
- The quality gate validates each evaluation suite, not only the last one run.
- `llm_judge` always failed on a clean install: `pydantic-settings` was not a
  dependency. Its `provider` and `model` options, and `AGENTBENCH_JUDGE_PROVIDER`,
  were also ignored.
- OpenAI cost estimates now price dated model versions (e.g.
  `gpt-4o-mini-2024-07-18`), and the OpenAI adapter closes its client on teardown.

### Removed
- Unused legacy modules that could not be imported: `adapters.builtin`,
  `evaluators.base`, `evaluators.builtin`, `runners.engine`, `storage.sqlite`,
  `reporting.terminal`, their tests, and `examples/basic-agent/run.py`.

### Added
- Tests for the dataset loader, evaluation runner, adapter and evaluator
  registries, SQLite store, and the cost and JSON-schema evaluators.

## [0.1.0] - 2024-01-01

### Added
- Core evaluation engine with async evaluation pipeline
- CLI: `agentbench run`, `validate`, `compare`, `list`, `serve`
- Built-in evaluators: exact_match, contains, tool_call, latency, cost,
  json_schema, hallucination, llm_judge
- Agent adapters: echo, http, openai
- YAML evaluation dataset format with auto-evaluator detection
- SQLite persistence for evaluation runs
- FastAPI web dashboard
- Rich terminal output with pass/fail analysis
- GitHub Actions CI workflow
- Docker and Docker Compose support
- Examples: basic-agent, openai, langgraph
