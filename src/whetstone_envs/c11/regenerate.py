"""Regenerate C11's persisted default-pool manifest."""

from pathlib import Path

from whetstone_envs.c11.generation import build_manifest, generate_pool


def main() -> int:
    manifest_path = Path(__file__).with_name("manifest.json")
    build_manifest(generate_pool()).write(manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
