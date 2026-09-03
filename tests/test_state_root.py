"""Canonical FlossWare AI state-root behavior."""

from __future__ import annotations

import importlib


def test_default_state_root_is_canonical(monkeypatch):
    monkeypatch.delenv("FLOSSWARE_AI_HOME", raising=False)
    monkeypatch.delenv("FLOSSWARE_AI_ROOT", raising=False)

    import model_router_ai.discovery as discovery

    discovery = importlib.reload(discovery)
    assert discovery.ROOT.name == "ai"
    assert discovery.ROOT.parent.name == ".FlossWare"


def test_state_root_override_is_honored(monkeypatch, tmp_path):
    override = tmp_path / "custom-ai"
    monkeypatch.setenv("FLOSSWARE_AI_HOME", str(override))
    monkeypatch.delenv("FLOSSWARE_AI_ROOT", raising=False)

    import model_router_ai.discovery as discovery

    discovery = importlib.reload(discovery)
    assert discovery.ROOT == override
    assert discovery.ACCOUNTS_FILE == override / "config" / "accounts.toml"
