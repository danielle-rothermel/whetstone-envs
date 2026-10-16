from __future__ import annotations

PROVIDER_CONCURRENCY_FLAG = "--provider-concurrency"

PROVIDER_CONCURRENCY_FORCE_FLAG = "--force-provider-concurrency"

# Keep CLI bounds importable without the optional runtime.
DEFAULT_PROVIDER_CONCURRENCY = 5

MAX_UNFORCED_PROVIDER_CONCURRENCY = 64


def validate_provider_concurrency(value: int) -> int:
    if value < 1:
        raise ValueError(f"provider concurrency is at least 1; got {value}")
    return value


def resolve_provider_concurrency(value: int, *, force: bool) -> int:
    validate_provider_concurrency(value)
    if value > MAX_UNFORCED_PROVIDER_CONCURRENCY and not force:
        raise ValueError(
            f"provider concurrency {value} exceeds the sanity cap of "
            f"{MAX_UNFORCED_PROVIDER_CONCURRENCY}. OpenRouter publishes "
            "no per-account concurrency limit for paid models, so this "
            "cap is this runner's own prudence rather than a provider "
            "limit -- but a width this large opens that many billed "
            "connections at once, and an extra digit is far likelier "
            f"than a deliberate choice. Pass "
            f"{PROVIDER_CONCURRENCY_FORCE_FLAG} to mean it."
        )
    return value
