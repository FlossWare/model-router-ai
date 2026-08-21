# mr-chat CLI

Command-line interface for model-router-ai. Routes prompts through the
decorator stack: PolicyGuard > BudgetGuard > CostAware > LatencyOptimizer > ThompsonSamplingSelector > ProviderRouter.

## Setup

Export at least one API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="AIza..."
export COHERE_API_KEY="..."
export CEREBRAS_API_KEY="..."
```

## Usage

```bash
# Simple prompt
./mr-chat "Explain quicksort in two sentences"

# Pipe from stdin
echo "Summarize this" | ./mr-chat

# Pick a model and get JSON output
./mr-chat --model gemini-2.0-flash --json "What is 2+2?"

# Free models only, with budget cap
./mr-chat --free-only --budget 50 "Hello world"

# Use specific providers
./mr-chat --providers groq,gemini "Translate to French: hello"

# List discovered models
./mr-chat --list-models

# Show budget status
./mr-chat --status
```
