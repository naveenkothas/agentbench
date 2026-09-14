# Integrations

## OpenAI

```bash
pip install "agentbench[openai]"
export OPENAI_API_KEY=sk-...
```

```yaml
agent:
  type: openai
  model: gpt-4o-mini
  system_prompt: "You are a helpful assistant."
  temperature: 0
  max_tokens: 500
```

## Any HTTP agent

```yaml
agent:
  type: http
  url: "https://your-agent.com/run"
  method: POST
  response_field: "output"
  timeout: 30
```

The HTTP adapter sends:
```json
{"input": "user message", "context": {...}}
```
And reads the `response_field` from the JSON response.

## Custom Python agent

```python
from agentbench import AgentAdapter, AgentResult

class MyAgent(AgentAdapter):
    adapter_type = "my_agent"

    async def run(self, input: str, context: dict) -> AgentResult:
        response = await your_llm_call(input)
        return AgentResult(response=response)
```

Then use `adapter_type = "my_agent"` in YAML after registering it.

## LangGraph (via HTTP adapter)

Start your LangGraph server, then point AgentBench at it:

```yaml
agent:
  type: http
  url: "http://localhost:8100/invoke"
  response_field: "output"
```

## Anthropic (coming in v0.2)

```yaml
agent:
  type: anthropic
  model: claude-3-haiku-20240307
```

Want to add this? See the [CONTRIBUTING guide](../CONTRIBUTING.md) — this is a great first contribution.
