from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from whetstone_envs.c19.generation import (
    DEFAULT_N_PER_STRATUM,
    DEFAULT_SEED_START,
    build_manifest,
    generate_pool,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from whetstone_envs.manifests import Manifest


def regenerate(
    manifest_path: Path,
    *,
    n_per_stratum: int = DEFAULT_N_PER_STRATUM,
    seed_start: int = DEFAULT_SEED_START,
) -> Manifest:
    """Regenerate C19 and atomically publish its canonical manifest."""
    pool = generate_pool(
        n_per_stratum=n_per_stratum,
        seed_start=seed_start,
    )
    manifest = build_manifest(pool)
    manifest.write(manifest_path)
    return manifest


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whetstone-envs-c19-regenerate",
        description="Regenerate the deterministic C19 pool manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    parser.add_argument(
        "--n-per-stratum",
        type=int,
        default=DEFAULT_N_PER_STRATUM,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
    )
    args = parser.parse_args(argv)
    regenerate(
        args.manifest,
        n_per_stratum=args.n_per_stratum,
        seed_start=args.seed_start,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
