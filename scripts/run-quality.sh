#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== ruff check ==="
ruff check model_router_ai/ tests/

echo "=== ruff format check ==="
ruff format --check model_router_ai/ tests/

echo "=== mypy ==="
mypy model_router_ai/ --ignore-missing-imports

echo "=== bandit ==="
bandit -r model_router_ai/ -c pyproject.toml -q

echo "All quality checks passed."
