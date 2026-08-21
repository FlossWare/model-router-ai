---
name: model-stats
description: Show Thompson Sampling bandit statistics for all routed models
---

# Model Performance Statistics

When the user invokes this skill, display Thompson Sampling bandit statistics and latency data for all models the router has interacted with.

## Steps

### 1. Call the MCP tool

Use the `model-router` MCP server's `model_performance` tool to get Thompson Sampling bandit statistics.

The tool returns JSON:
```json
{
  "gemini-2.5-flash": {"alpha": 45.0, "beta": 3.0, "mean": 0.9375, "trials": 46},
  "llama-4-maverick": {"alpha": 32.0, "beta": 5.0, "mean": 0.8649, "trials": 35}
}
```

### 2. Display Thompson Sampling statistics

Present a table sorted by mean success rate (descending):

```
Thompson Sampling Model Statistics
===================================

| Model              | Alpha | Beta | Mean  | Trials | Status      |
|--------------------|-------|------|-------|--------|-------------|
| gemini-2.5-flash   | 45.0  | 3.0  | 0.938 | 46     | Exploiting  |
| llama-4-maverick   | 32.0  | 5.0  | 0.865 | 35     | Exploiting  |
| command-a-03-2025  | 8.0   | 2.0  | 0.800 | 8      | Exploring   |
| mixtral-8x7b       | 3.0   | 4.0  | 0.429 | 5      | Exploring   |
| phi-4              | 1.0   | 1.0  | 0.500 | 0      | Untested    |
```

Status labels:
- **Exploiting**: trials >= 20 and mean >= 0.7 (proven model, used frequently)
- **Exploring**: trials > 0 but < 20 (still gathering data)
- **Untested**: trials == 0 (never selected)
- **Underperforming**: trials >= 10 and mean < 0.5 (consistently poor)

### 3. Identify notable models

Call out:
- **Best performer**: Highest mean with >= 10 trials
- **Most explored**: Highest trial count
- **Rising star**: Highest mean with < 10 trials (promising but unproven)
- **Needs attention**: Lowest mean with >= 5 trials

### 4. Display latency data (if available)

Check `chat` tool responses for `latency_ms` data to identify fastest models.

Present as:

```
Fastest Models (by average latency)
====================================
  1. groq/llama-3.3-70b          142ms
  2. cerebras/llama-3.3-70b      198ms
  3. gemini/gemini-2.5-flash     312ms
  4. openrouter/llama-4-maverick 845ms
```

### 5. Summarize

Present key takeaways: which models are proven, which need more exploration, and any underperformers to consider removing from the pool.
