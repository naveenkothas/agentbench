# Azure OpenAI Agent Example

Evaluate a support agent running on an Azure OpenAI deployment.

## Setup

```bash
pip install "agentbench[azure]"

export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=my-gpt-4o-mini   # your deployment name
export AZURE_OPENAI_API_KEY=...
```

### Keyless: Microsoft Entra ID

Skip `AZURE_OPENAI_API_KEY`, sign in with `az login` (or use a managed
identity), and uncomment `auth: entra` in `evaluation.yaml`. Your identity
needs the **Cognitive Services OpenAI User** role on the resource.

## Run

```bash
agentbench run evaluation.yaml
```

See [docs/integrations.md](../../docs/integrations.md#azure-openai) for every option.
