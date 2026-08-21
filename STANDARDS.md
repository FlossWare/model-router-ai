# FlossWare Engineering Standards Compliance

This document describes how **model-router-ai** adheres to the
[FlossWare Engineering Standards](https://github.com/FlossWare/engineering-standards).

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-0001](https://github.com/FlossWare/engineering-standards/blob/main/adr/ADR-0001-explicit-opt-in-cross-cutting-behavior.md) | Explicit Opt-In Cross-Cutting Behavior | Compliant |
| [ADR-0006](https://github.com/FlossWare/engineering-standards/blob/main/adr/ADR-0006-cross-cutting-decorators.md) | Cross-Cutting Decorators | Compliant |
| [ADR-0008](https://github.com/FlossWare/engineering-standards/blob/main/adr/ADR-0008-free-first-modular-platform.md) | Free-First Modular Platform | Compliant |
| [ADR-0009](https://github.com/FlossWare/engineering-standards/blob/main/adr/ADR-0009-core-architecture-principles.md) | Core Architecture Principles | Compliant |
| [ADR-0017](https://github.com/FlossWare/engineering-standards/blob/main/adr/ADR-0017-agent-neutral-architecture.md) | Agent-Neutral Architecture | Compliant |
| [ADR-0020](https://github.com/FlossWare/engineering-standards/blob/main/adr/ADR-0020-capability-protocol-separation.md) | Capability-Protocol Separation | Compliant |

## ADR-0001: Explicit Opt-In

- No side effects on `import model_router_ai`.
- All routing behavior requires explicit instantiation of router and decorator classes.
- Decorators (`CostAware`, `BudgetGuard`, `PolicyGuard`, `LatencyOptimizer`, `ThompsonSamplingSelector`) must be deliberately composed by the developer.

## ADR-0006: Cross-Cutting Decorators

Decorators in `decorators.py` add exactly one concern each without changing the `ModelRouter` interface:

| Decorator | Concern |
|-----------|---------|
| `CostAware` | Tracks per-request and cumulative token costs |
| `BudgetGuard` | Enforces monthly spend limits |
| `PolicyGuard` | Filters models by allow/deny glob patterns |
| `LatencyOptimizer` | Routes to lowest-latency provider |
| `ThompsonSamplingSelector` | Bandit-based model selection via Thompson Sampling |

Decorators are stackable and order-independent for orthogonal concerns.

## ADR-0008: Free-First Modular Platform

- Zero external runtime dependencies (stdlib only).
- Installable via `pip install git+https://github.com/FlossWare/model-router-ai.git`.
- No paid services required for core functionality.

## ADR-0009: Core Architecture Principles

- **Modular**: each module has a single responsibility (types, protocol, router, decorators, strategies, providers).
- **Composable**: decorators wrap any `ModelRouter` implementation transparently.
- **Contracts over implementations**: `ModelRouter` is a `typing.Protocol` with `@runtime_checkable` — no inheritance required.

## ADR-0017: Agent-Neutral Architecture

- No imports from or coupling to any agent framework (loom-ai, LangChain, CrewAI, etc.).
- Works with Claude Code, Crush, Codex, or any Python application.

## ADR-0020: Capability-Protocol Separation

- `protocol.py` defines the `ModelRouter` Protocol (capability contract).
- Transport and provider details are isolated in `providers.py`.
- Consumers depend on the protocol, not on concrete implementations.
