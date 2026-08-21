---
name: multi-model-query
description: Query multiple LLM models for consensus using model-router-ai
---

# Multi-Model Consensus Query

When the user invokes this skill, send their query to multiple LLM models via model-router-ai, compare the responses, and synthesize a consensus answer.

## Steps

### 1. Call the MCP tool

Use the `model-router` MCP server's `multi_model_chat` tool:

- `prompt`: The user's query
- `num_models`: 3-5 (default 3)
- `system_prompt`: Optional context

The tool returns a JSON array of responses from different models, each with `content`, `model`, `provider`, `latency_ms`, `cost_usd`, and `usage`.

If the MCP server is not available, fall back to multiple `mr-chat` CLI calls with `--model` to target specific models.

### 2. Compare and synthesize

After collecting responses:

1. Identify points of agreement (themes/facts all or most models share).
2. Identify points of disagreement (where models diverge).
3. Synthesize a consensus answer, noting confidence level.
4. Present a summary table:

```
| Model              | Provider   | Latency | Cost     | Agrees? |
|--------------------|------------|---------|----------|---------|
| gemini-2.5-flash   | gemini     | 450ms   | $0.0000  | Yes     |
| llama-4-maverick   | openrouter | 820ms   | $0.0000  | Yes     |
| command-a-03-2025  | cohere     | 1200ms  | $0.0012  | Partial |
```

### 6. Report

Present to the user:
- The consensus answer
- The agreement/disagreement breakdown
- The per-model latency and cost table
- Total cost for the multi-model query
