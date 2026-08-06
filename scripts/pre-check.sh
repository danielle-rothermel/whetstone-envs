#!/usr/bin/env bash

set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${repo_root}"

uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run actionlint -ignore 'label "depot-ubuntu-24.04" is unknown' \
    .github/workflows/*.yml
uvx tombi@1.2.5 lint --offline .defs/terms.toml .defs/contracts.toml
pytest_args=(-q)
if [[ "${CI:-}" == "true" ]]; then
    pytest_args+=(--run-integration)
fi
uv run pytest "${pytest_args[@]}"
uv build --clear --no-sources
uv run python scripts/check_distributions.py
