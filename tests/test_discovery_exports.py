"""Public discovery API surface."""

from __future__ import annotations


def test_discover_identities_exported_from_package_root() -> None:
    from model_router_ai import discover_identities, discover_identity
    from model_router_ai.discovery import (
        discover_identities as submodule_identities,
        discover_identity as submodule_identity,
    )

    assert discover_identities is submodule_identities
    assert discover_identity is submodule_identity
    assert callable(discover_identities)
    assert callable(discover_identity)


def test_discover_identities_returns_list_without_network() -> None:
    from model_router_ai import discover_identities

    # No provider credentials in the environment → empty or unverified list, never raises.
    result = discover_identities(timeout=0.1)
    assert isinstance(result, list)
