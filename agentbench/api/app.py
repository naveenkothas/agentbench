"""
FastAPI application for the AgentBench web interface.

Provides a JSON API over evaluation run history, with an optional
simple HTML dashboard for viewing results in a browser.
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agentbench.core.models import EvalRun, EvalResult


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgentBench",
        description="AI Agent Evaluation Framework",
        version="0.1.0",
    )

    from agentbench.storage.sqlite_store import SQLiteStore
    store = SQLiteStore()

    @app.on_event("startup")
    async def startup():
        await store.init()

    @app.get("/api/runs")
    async def list_runs(limit: int = 20):
        records = await store.list_runs(limit=limit)
        return [
            {
                "id": r.id,
                "name": r.name,
                "dataset_name": r.dataset_name,
                "agent_type": r.agent_type,
                "status": r.status,
                "total_cases": r.total_cases,
                "passed_cases": r.passed_cases,
                "pass_rate": r.pass_rate,
                "total_cost_usd": r.total_cost_usd,
                "total_latency_ms": r.total_latency_ms,
                "started_at": str(r.started_at) if r.started_at else None,
                "completed_at": str(r.completed_at) if r.completed_at else None,
            }
            for r in records
        ]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        record = await store.get_run(run_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

        results = json.loads(record.results_json or "[]")
        return {
            "id": record.id,
            "name": record.name,
            "dataset_name": record.dataset_name,
            "agent_type": record.agent_type,
            "status": record.status,
            "total_cases": record.total_cases,
            "passed_cases": record.passed_cases,
            "pass_rate": record.pass_rate,
            "total_cost_usd": record.total_cost_usd,
            "total_latency_ms": record.total_latency_ms,
            "started_at": str(record.started_at) if record.started_at else None,
            "completed_at": str(record.completed_at) if record.completed_at else None,
            "results": results,
        }

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return _DASHBOARD_HTML

    return app


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgentBench Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
    header { background: #1a1f2e; border-bottom: 1px solid #2d3748; padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
    header h1 { font-size: 1.25rem; font-weight: 700; color: #63b3ed; }
    header span { color: #718096; font-size: 0.875rem; }
    main { padding: 2rem; max-width: 1200px; margin: 0 auto; }
    .card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; padding: 0.75rem 1rem; color: #718096; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; border-bottom: 1px solid #2d3748; }
    td { padding: 0.75rem 1rem; border-bottom: 1px solid #1e2433; font-size: 0.875rem; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #232938; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .badge-green { background: #1a3a2a; color: #48bb78; }
    .badge-yellow { background: #3a2e1a; color: #ed8936; }
    .badge-red { background: #3a1a1a; color: #fc8181; }
    .empty { color: #718096; text-align: center; padding: 3rem; }
    h2 { font-size: 1rem; font-weight: 600; margin-bottom: 1rem; color: #a0aec0; }
  </style>
</head>
<body>
  <header>
    <h1>AgentBench</h1>
    <span>Evaluation Dashboard</span>
  </header>
  <main>
    <div class="card">
      <h2>Recent Evaluation Runs</h2>
      <div id="runs-container"><p class="empty">Loading...</p></div>
    </div>
  </main>
  <script>
    async function loadRuns() {
      const resp = await fetch('/api/runs');
      const runs = await resp.json();
      const container = document.getElementById('runs-container');
      if (!runs.length) {
        container.innerHTML = '<p class="empty">No evaluation runs yet. Run <code>agentbench run evals/</code> to get started.</p>';
        return;
      }
      const rows = runs.map(r => {
        const pct = (r.pass_rate * 100).toFixed(1) + '%';
        const cls = r.pass_rate >= 0.9 ? 'badge-green' : r.pass_rate >= 0.7 ? 'badge-yellow' : 'badge-red';
        return `<tr>
          <td><a href="/api/runs/${r.id}" style="color:#63b3ed">${r.id}</a></td>
          <td>${r.dataset_name || '-'}</td>
          <td>${r.agent_type || '-'}</td>
          <td>${r.total_cases}</td>
          <td><span class="badge ${cls}">${pct}</span></td>
          <td>${r.passed_cases}/${r.total_cases}</td>
          <td>${r.total_cost_usd > 0 ? '$' + r.total_cost_usd.toFixed(4) : '-'}</td>
          <td>${r.started_at ? r.started_at.slice(0, 16) : '-'}</td>
        </tr>`;
      }).join('');
      container.innerHTML = `<table>
        <thead><tr>
          <th>Run ID</th><th>Dataset</th><th>Agent</th><th>Cases</th>
          <th>Pass Rate</th><th>Passed</th><th>Cost</th><th>Started</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
    }
    loadRuns();
  </script>
</body>
</html>
"""
