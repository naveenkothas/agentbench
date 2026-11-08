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

## Azure OpenAI

```bash
pip install "agentbench[azure]"
```

```yaml
agent:
  type: azure_openai
  endpoint: https://my-resource.openai.azure.com   # or AZURE_OPENAI_ENDPOINT
  deployment: support-gpt4o-mini                    # or AZURE_OPENAI_DEPLOYMENT
  api_version: "2024-10-21"                         # or AZURE_OPENAI_API_VERSION
  auth: api_key                                     # api_key (default) or entra
  model: gpt-4o-mini   # optional: underlying model, used only for cost estimates
  system_prompt: "You are a helpful assistant."
  tools: []            # OpenAI-format tool definitions
  temperature: 0
  max_tokens: 500
```

Every connection setting can come from the YAML file or from environment
variables (or a `.env` file). Keep secrets in environment variables, not YAML.

| Setting | Environment variable | Default |
|---------|----------------------|---------|
| `endpoint` | `AZURE_OPENAI_ENDPOINT` | required |
| `deployment` | `AZURE_OPENAI_DEPLOYMENT` | required |
| `api_key` | `AZURE_OPENAI_API_KEY` | required for `auth: api_key` |
| `api_version` | `AZURE_OPENAI_API_VERSION` or `OPENAI_API_VERSION` | `2024-10-21` |

Azure routes each request by **deployment name**, so `deployment` (not `model`)
selects what runs. Costs are estimated from `model` if set, otherwise from the
model name Azure reports in its response.

### Microsoft Entra ID (keyless)

Set `auth: entra` to sign in with `DefaultAzureCredential` instead of an API key.
It works with `az login` locally, managed identity on Azure, and workload
identity or service principal environment variables in CI. The identity needs
the **Cognitive Services OpenAI User** role on the Azure OpenAI resource.

```yaml
agent:
  type: azure_openai
  deployment: support-gpt4o-mini
  auth: entra
```

### Azure OpenAI as the LLM judge

```yaml
evaluators:
  - name: llm_judge
    provider: azure_openai
    deployment: judge-gpt4o      # default: AZURE_OPENAI_DEPLOYMENT
    auth: api_key                # or entra
    criteria: correctness
```

Or set `AGENTBENCH_JUDGE_PROVIDER=azure_openai` to make it the default judge.

### In GitHub Actions

```yaml
- name: Run evaluation on Azure OpenAI
  env:
    AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
    AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
    AZURE_OPENAI_DEPLOYMENT: ${{ vars.AZURE_OPENAI_DEPLOYMENT }}
  run: |
    pip install "agentbench[azure]"
    agentbench run evals/
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
