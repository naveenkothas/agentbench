# Evaluators Reference

## contains

Checks that the response contains required phrases and excludes forbidden ones.

```yaml
evaluators:
  - contains

cases:
  - id: example
    input: "Help with my account"
    expected:
      contains:
        - "help"
        - "account"
      not_contains:
        - "error"
        - "sorry, I cannot"
```

## exact_match

Checks for exact string equality. Useful for deterministic factual answers.

```yaml
cases:
  - id: capital
    input: "What is the capital of France?"
    expected:
      response: "Paris"
```

## tool_call

Validates tool/function call name and arguments.

```yaml
cases:
  - id: order-lookup
    input: "Where is order #12345?"
    expected:
      tool_call:
        name: get_order_status
        arguments:
          order_id: "12345"
```

## latency

Checks that the agent responds within a time threshold.

```yaml
evaluators:
  - latency

cases:
  - id: fast-query
    input: "What is 2+2?"
    expected:
      max_latency_ms: 2000
```

## cost

Checks that estimated API cost stays within budget.

```yaml
cases:
  - id: budget-check
    input: "Write a haiku."
    expected:
      max_cost_usd: 0.001
```

## hallucination

Detects whether the response introduces claims not in the provided context.

```yaml
cases:
  - id: rag-check
    input: "What is the return policy?"
    context:
      documents:
        - "Returns accepted within 30 days with receipt."
    expected:
      contains:
        - "30 days"
```

## json_schema

Validates that the response is valid JSON conforming to a schema.

```yaml
cases:
  - id: structured-output
    input: "Return user info as JSON"
    expected:
      json_schema:
        type: object
        properties:
          name:
            type: string
          age:
            type: integer
        required: [name, age]
```

## llm_judge

Uses an LLM to evaluate response quality. Requires OPENAI_API_KEY or ANTHROPIC_API_KEY.

```yaml
evaluators:
  - name: llm_judge
    criteria: correctness
    threshold: 0.8
    provider: openai

  - name: llm_judge
    criteria: groundedness
```

Available criteria: `correctness`, `relevance`, `groundedness`, `instruction_following`, `safety`, `conciseness`
