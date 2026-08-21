# Model Router MCP Server

MCP server that exposes model-router-ai as tools for Claude Code.

## Tools

- **chat** -- Send a prompt through the router (auto-selects cheapest/fastest model)
- **multi_model_chat** -- Query N models concurrently for comparison/consensus
- **list_models** -- List all discovered model endpoints
- **budget_status** -- Current spend, remaining budget, call count
- **model_performance** -- Thompson Sampling bandit stats per model

## Setup

Add to your Claude Code MCP config (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "model-router": {
      "command": "python3",
      "args": ["/path/to/model-router-repo/integrations/mcp/server.py"],
      "env": {
        "GROQ_API_KEY": "gsk_...",
        "OPENROUTER_API_KEY": "sk-or-...",
        "GEMINI_API_KEY": "AIza...",
        "COHERE_API_KEY": "...",
        "CEREBRAS_API_KEY": "..."
      }
    }
  }
}
```

Or copy `claude_code_config.json` and fill in the keys.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | No | Groq API key |
| `OPENROUTER_API_KEY` | No | OpenRouter API key |
| `GEMINI_API_KEY` | No | Google Gemini API key |
| `COHERE_API_KEY` | No | Cohere API key |
| `CEREBRAS_API_KEY` | No | Cerebras API key |
| `BUDGET_MAX_MONTHLY` | No | Monthly budget cap in USD (default: 300) |
| `SECRET_ORCHESTRATOR_URL` | No | Secret endpoint URL (default: http://aio-01:5000/secrets) |

At least one API key must be provided. Keys not found in env vars are
fetched from the orchestrator endpoint (2s timeout, silent fallback).
