#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install

echo "model-router-ai dev environment ready."
