"""
Dataset loader — parses YAML evaluation files into EvalDataset objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentbench.core.models import EvalCase, EvalDataset, ExpectedOutput


class DatasetLoader:
    """Loads evaluation datasets from YAML files or directories."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalDataset:
        """Load a single YAML evaluation file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._parse(data, source=str(path))

    @classmethod
    def from_directory(cls, directory: str | Path) -> list[EvalDataset]:
        """Load all YAML files from a directory."""
        directory = Path(directory)
        datasets = []
        for yaml_file in sorted(directory.glob("*.yaml")):
            datasets.append(cls.from_yaml(yaml_file))
        for yaml_file in sorted(directory.glob("*.yml")):
            datasets.append(cls.from_yaml(yaml_file))
        return datasets

    @classmethod
    def _parse(cls, data: dict[str, Any], source: str = "") -> EvalDataset:
        cases = []
        for raw_case in data.get("cases", []):
            expected_data = raw_case.get("expected", {})

            # Support both `tool_call` as a single object or under expected
            tool_call = expected_data.get("tool_call")
            if isinstance(tool_call, dict):
                tool_call_parsed = tool_call
            else:
                tool_call_parsed = None

            expected = ExpectedOutput(
                response=expected_data.get("response"),
                contains=expected_data.get("contains", []),
                not_contains=expected_data.get("not_contains", []),
                tool_call=tool_call_parsed,
                tool_call_sequence=expected_data.get("tool_call_sequence", []),
                json_schema=expected_data.get("json_schema"),
                max_latency_ms=expected_data.get("max_latency_ms"),
                max_cost_usd=expected_data.get("max_cost_usd"),
                semantic_similarity_threshold=expected_data.get(
                    "semantic_similarity_threshold"
                ),
            )

            case = EvalCase(
                id=str(raw_case.get("id", f"case-{len(cases)+1}")),
                input=str(raw_case.get("input", "")),
                expected=expected,
                context=raw_case.get("context", {}),
                tags=raw_case.get("tags", []),
                metadata=raw_case.get("metadata", {}),
            )
            cases.append(case)

        # Determine which evaluators to run
        # If none specified, auto-detect from case expectations
        evaluators = data.get("evaluators", [])
        if not evaluators:
            evaluators = cls._auto_detect_evaluators(cases)

        return EvalDataset(
            name=data.get("name", Path(source).stem if source else "unnamed"),
            description=data.get("description", ""),
            version=str(data.get("version", "1.0")),
            agent=data.get("agent", {}),
            evaluators=evaluators,
            cases=cases,
            metadata={"source": source},
        )

    @classmethod
    def _auto_detect_evaluators(cls, cases: list[EvalCase]) -> list[str]:
        """Infer which evaluators are needed based on case expectations."""
        needed = set()
        for case in cases:
            exp = case.expected
            if exp.response:
                needed.add("exact_match")
            if exp.contains or exp.not_contains:
                needed.add("contains")
            if exp.tool_call or exp.tool_call_sequence:
                needed.add("tool_call")
            if exp.max_latency_ms:
                needed.add("latency")
            if exp.max_cost_usd:
                needed.add("cost")
            if exp.json_schema:
                needed.add("json_schema")
            if case.context.get("documents"):
                needed.add("hallucination")

        # Always include at least contains if nothing else detected
        if not needed:
            needed.add("contains")

        return sorted(needed)
