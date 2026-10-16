from __future__ import annotations

import argparse
from pathlib import Path

from whetstone_envs.c18 import (
    DEFAULT_CONFIG,
    HARD_CONFIG,
    GenerationConfig,
    build_manifest,
    generate_pool,
)

_CONFIGS: dict[str, GenerationConfig] = {
    "default": DEFAULT_CONFIG,
    "hard": HARD_CONFIG,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate one canonical C18 pool manifest.",
    )
    parser.add_argument("--config", choices=tuple(_CONFIGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    config = _CONFIGS[arguments.config]
    pool = generate_pool(config)
    manifest = build_manifest(pool, config)
    manifest.write(arguments.output)
    print(
        f"wrote {arguments.output}: {config.generator_version} "
        f"{manifest.content_hash}",
    )


if __name__ == "__main__":
    main()
