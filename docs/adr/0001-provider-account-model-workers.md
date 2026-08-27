# ADR-0001: Provider, Account, Model, and Worker Routing

- **Status:** Accepted
- **Date:** 2026-08-27

## Context

LLM access may span local runtimes, multiple APIs, multiple credentials for one
API, and multiple models. Provider-level failover is insufficient because one
account can exhaust quota while another account for the same provider remains
usable.

Routing algorithms such as Thompson Sampling should optimize selection without
being coupled to transport, credentials, or provider-specific failure handling.

## Decision

Represent each executable route as a **worker** composed of:

- provider transport
- provider account/credential identity
- model

Maintain workers in a `WorkerPool` and select them through an `Arbiter`.

Worker availability is independent. Common operational states include
`RATE_LIMITED`, `QUOTA_EXHAUSTED`, `AUTH_FAILED`, `MODEL_UNAVAILABLE`,
`TIMEOUT`, and `NETWORK_ERROR`.

When a provider supplies a retry or quota reset time, the worker is suppressed
until that time rather than being retried immediately.

The existing model selection strategy remains a policy concern. The worker
layer provides an injectable scoring boundary so Thompson Sampling, genetic
optimization, deterministic policies, or future strategies can operate above
it.

The existing OpenAI-compatible chat API remains the public interface.

## Consequences

### Positive

- Multiple providers, accounts, and models can coexist.
- One exhausted account does not disable an entire provider.
- Local and remote workers use the same routing abstraction.
- Provider-specific transport remains isolated in adapters.
- Routing algorithms can evolve independently of worker execution.
- Quota-aware suppression prevents retry storms.

### Negative

- There is additional state and lifecycle management compared with a simple
  provider loop.
- Worker health state is currently in-memory and therefore process-local.
- Provider-specific quota semantics still require adapter support.

## Non-goals

This ADR does not select or implement Thompson Sampling, genetic algorithms,
consensus, adversarial review, or paid-provider policy. Those belong in higher
routing/policy layers.

## Default behavior

No provider, account, model, or worker is enabled merely because its adapter or
class exists. An empty worker pool is valid.
