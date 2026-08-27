# Provider, Account, Model, and Worker Architecture

`model-router-ai` treats an executable LLM route as a **worker**. A worker is the
combination of a provider transport, one provider account/credential, and one
model.

```text
Provider
  └── Account
        └── Model
              └── Worker
                    └── Arbiter / policy
```

## Why workers exist

Provider accounts are independent resources. If one OpenRouter account is rate
limited or has exhausted its daily quota, other accounts and providers must
remain eligible. The router therefore never treats a provider name as a single
availability unit.

A worker has a stable identity:

```text
provider/account/model
```

For example:

```text
openrouter/flossware/qwen/qwen3-coder
openrouter/ncrr/qwen/qwen3-coder
groq/default/llama-3.3-70b-versatile
```

## Worker lifecycle

Workers start `AVAILABLE`. Provider failures are classified into structured
states such as `RATE_LIMITED`, `QUOTA_EXHAUSTED`, `AUTH_FAILED`, and
`MODEL_UNAVAILABLE`.

When a provider reports a reset or retry time, the worker records it and is
excluded from arbitration until that time. This prevents retry storms against
an exhausted account while preserving failover to other workers.

## Arbiter

`Arbiter` selects only currently available workers. Its scoring function is
injectable, so routing policies can be layered without coupling the worker
transport to a selection algorithm.

The current router continues to use the existing selection strategy for its
score. Thompson Sampling and genetic optimization remain policy concerns, not
worker infrastructure.

## Configuration principles

- Credentials are supplied by the existing caller/configuration mechanism.
- Credentials are not persisted by the worker registry.
- Provider adapters are inert until explicitly configured.
- Zero configured workers is a valid state.
- Paid providers must never become enabled merely because their adapter exists.

## Extending the system

Any provider implementing the existing provider interface can be represented by
workers. OpenAI-compatible providers can share `OpenAICompatProvider`; a custom
transport only needs to implement the provider contract.

The public `/v1/chat/completions` interface remains unchanged. Clients address
models normally while the router performs worker selection and failover.
