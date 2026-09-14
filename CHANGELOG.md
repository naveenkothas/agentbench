# Changelog

All notable changes to AgentBench are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
