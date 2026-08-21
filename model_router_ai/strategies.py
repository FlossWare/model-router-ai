"""Pluggable model selection strategies.

Each strategy scores candidate models. Higher score = more likely to
be selected. Strategies track their own state (latency, success/failure)
and expose a ``score()`` / ``record()`` interface.
"""

from __future__ import annotations

import random
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SelectionStrategy(Protocol):
    """Protocol for model selection strategies."""

    def score(self, *, successes: int, failures: int, **kwargs: Any) -> float: ...

    def record(self, *, success: bool, **kwargs: Any) -> None: ...


class ThompsonSamplingStrategy:
    """Bayesian exploration/exploitation via Beta-distributed sampling."""

    def score(self, *, successes: int, failures: int, **kwargs: Any) -> float:
        return random.betavariate(successes + 1, failures + 1)

    def record(self, *, success: bool, **kwargs: Any) -> None:
        pass


class RoundRobinStrategy:
    """Cycle through endpoints evenly to spread rate-limit pressure."""

    def __init__(self) -> None:
        self._counter = 0

    def score(self, *, successes: int, failures: int, **kwargs: Any) -> float:
        self._counter += 1
        return 1.0 / self._counter

    def record(self, *, success: bool, **kwargs: Any) -> None:
        pass


class LatencyWeightedStrategy:
    """Prefer endpoints with lower observed latency."""

    def __init__(self) -> None:
        self._latencies: dict[str, list[float]] = {}

    def score(self, *, successes: int, failures: int, **kwargs: Any) -> float:
        key = kwargs.get("endpoint_key", "")
        samples = self._latencies.get(key, [])
        if not samples:
            return random.betavariate(successes + 1, failures + 1)
        avg = sum(samples[-20:]) / len(samples[-20:])
        return 1.0 / (avg + 0.001)

    def record(self, *, success: bool, **kwargs: Any) -> None:
        key = kwargs.get("endpoint_key", "")
        latency = kwargs.get("latency_s", 0.0)
        if key and latency > 0:
            self._latencies.setdefault(key, []).append(latency)


class CascadeStrategy:
    """Try preferred models first, fall back to everything else."""

    def __init__(self, preferred: list[str] | None = None) -> None:
        self._preferred = preferred or []

    def score(self, *, successes: int, failures: int, **kwargs: Any) -> float:
        model_id = kwargs.get("model_id", "")
        bonus = 100.0 if any(p in model_id for p in self._preferred) else 0.0
        return bonus + random.betavariate(successes + 1, failures + 1)

    def record(self, *, success: bool, **kwargs: Any) -> None:
        pass


STRATEGIES: dict[str, type] = {
    "thompson": ThompsonSamplingStrategy,
    "round_robin": RoundRobinStrategy,
    "latency": LatencyWeightedStrategy,
    "cascade": CascadeStrategy,
}
