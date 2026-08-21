# model-router-ai Integrations

Three ways to integrate model-router-ai with AI coding assistants.

## Option 1: CLI Wrapper

Location: `integrations/cli/`

A command-line interface for model-router-ai. Run queries, check budgets, and view model stats from the terminal.

## Option 2: MCP Server

Location: `integrations/mcp/`

An MCP (Model Context Protocol) server exposing model-router-ai as tools. Works with any MCP-compatible client (Claude Code, Cursor, etc.).

## Option 3: Claude Code Skills + CLAUDE.md Snippet

Location: `integrations/skills/` and `integrations/claude-code/`

Drop-in skill files for Claude Code and a CLAUDE.md snippet for project-level integration.

**Skills:**
- `multi-model-query` -- Query 3-5 models for consensus answers
- `budget-check` -- Check spending against monthly budget
- `model-stats` -- View Thompson Sampling and latency statistics

**Setup:**
1. Copy skill files from `integrations/skills/` to your project's `.claude/skills/` directory
2. Paste the snippet from `integrations/claude-code/CLAUDE_SNIPPET.md` into your project or global `CLAUDE.md`
