# ADR 0001: Provider, Account, Model, and Worker Fabric

- Status: Accepted
- Date: 2026-08-27

## Context

The model router already supports multiple provider accounts, but account
health and quota failures were not represented as first-class routing state.
A 429 quota exhaustion on one OpenRouter account could therefore cause wasted
retries and made the routing boundary too provider-centric.

## Decision

Represent each concrete route as a `ModelWorker` consisting of:

- provider transport
- account identity
- model endpoint
- credential
- health/quota state

`ProviderRouter` owns a worker pool and uses its existing selection strategy to
order available workers. A failed worker can be quarantined independently of
other accounts for the same provider.

Worker failures are classified as success, rate limited, quota exhausted,
authentication failure, model unavailable, or generic failure. When provider
error metadata exposes `X-RateLimit-Reset`, the worker remains unavailable until
that reset time. Without a reset, daily quota exhaustion uses a conservative
24-hour quarantine and ordinary rate limiting uses a short cooldown.

## Consequences

The public `ProviderRouter.add_provider()` and `chat()` APIs remain compatible.
Multiple accounts for one provider can fail independently, and local or other
providers can be added without changing worker semantics.

Thompson Sampling remains a selection policy rather than part of worker
execution. Consensus, adversarial review, and genetic optimization can be
layered above the worker pool later.
