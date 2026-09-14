---
name: New Integration
about: Add support for a new agent framework or LLM provider
labels: integration, good first issue
---

## Framework / Provider

<!-- e.g., Anthropic, CrewAI, Semantic Kernel, Vertex AI -->

## What would the adapter look like?

```python
class MyFrameworkAdapter(AgentAdapter):
    adapter_type = "my_framework"

    async def run(self, input: str, context: dict) -> AgentResult:
        # Call my_framework here
        ...
```

## Dependencies

<!-- What pip packages would be needed? -->

## Are you willing to implement this?

- [ ] Yes, I will submit a PR
- [ ] I can help test it
