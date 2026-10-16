#!/usr/bin/env bash

set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${repo_root}"

if uv run --no-project python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)'; then
  uv sync --locked --extra c18 --extra optim
else
  uv sync --locked --extra c18
fi
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync actionlint .github/workflows/*.yml
uv run --no-sync python -m unittest discover -s tests/core -v
if uv run --no-sync python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)'; then
  uv run --no-sync python -m unittest discover -s tests/optim -v
fi
uvx tombi@1.2.5 lint --offline .defs/terms.toml .defs/contracts.toml
uv build --clear --no-sources
