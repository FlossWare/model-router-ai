---
name: budget-check
description: Check current LLM spending against monthly budget using model-router-ai
---

# Budget Status Check

When the user invokes this skill, inspect the BudgetGuard state and report current spending against the monthly budget.

## Steps

### 1. Call the MCP tool

Use the `model-router` MCP server's `budget_status` tool to get current spending data.

If the MCP server is not available, fall back to the CLI:
```bash
mr-chat --status
```

### 2. Parse and report

The `budget_status` tool returns JSON:
```json
{
  "spent_usd": 47.23,
  "remaining_usd": 252.77,
  "max_usd": 300.0,
  "calls_made": 1247,
  "percent_used": 15.74
}
```

Present a clear summary:

```
Budget Status
=============
Monthly cap:    $300.00
Spent so far:   $47.23
Remaining:      $252.77
Percent used:   15.7%
Calls made:     1,247

Status: OK (below 50% threshold)
```

### 3. Warnings

Apply threshold warnings based on `percent_used`:
- Below 50%: "OK"
- 50-74%: "Caution -- over half the monthly budget consumed"
- 75-89%: "Warning -- approaching budget limit"
- 90-99%: "Critical -- budget nearly exhausted"
- 100%: "Exhausted -- BudgetGuard will reject further calls"
