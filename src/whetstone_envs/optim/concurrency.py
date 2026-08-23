"""Provider concurrency bounds, owned apart from the transport.

These are the concurrency declarations a stage records and a CLI parses:
plain arithmetic over a flag value, with no provider client behind them.
They live here rather than beside :class:`
~whetstone_envs.optim.provider.RetryingTransport` because
``whetstone_envs.reporting.cli`` -- a base-install entry point, reachable
without the ``optim`` extra -- needs exactly these bounds and none of the
transport. Importing them from :mod:`whetstone_envs.optim.provider` would
drag ``dr_providers`` and ``whetstone`` into the base install through a
module-scope import, and the ``whetstone-eval`` console script would fail
at its entry point on any install that did not also take the extra.

:mod:`whetstone_envs.optim.provider` re-exports every name defined here,
so the optimizer stack keeps one import site for provider policy.

**This module must not import an optional dependency**, at module scope
or otherwise. ``tests/test_base_install.py`` imports every base-install
module in an extra-free subprocess and fails if one of them reaches a
package the base install does not have.
"""

from __future__ import annotations

#: The flag that sets the provider concurrency, spelled once.
#:
#: Named beside the bounds it is refused against, so the CLI declaration
#: and the sanity-cap refusal that quotes it cannot drift apart.
PROVIDER_CONCURRENCY_FLAG = "--provider-concurrency"

#: The operator's override of the concurrency sanity cap, spelled once.
PROVIDER_CONCURRENCY_FORCE_FLAG = "--force-provider-concurrency"

#: How many task evaluations a stage runs against the provider at once.
#:
#: This is whetstone's own ``DEFAULT_CONCURRENCY``, restated here as the
#: value this package records when an invocation names nothing. It is not
#: imported from whetstone because it is *persisted identity*: a stage
#: record written today must keep meaning what it said even if the
#: dependency's default moves, and a silently-tracking import would
#: rewrite the meaning of every historical record instead of surfacing the
#: change. ``test_recorded_default_matches_whetstones_own`` pins the two
#: together so the drift is caught rather than inferred.
DEFAULT_PROVIDER_CONCURRENCY = 5

#: The largest concurrency reachable without an explicit override.
#:
#: OpenRouter documents no per-account concurrency ceiling for paid models
#: -- only free-tier request-per-minute limits and Cloudflare's protection
#: against traffic that "dramatically exceeds reasonable usage" -- so
#: there is no provider-published number to encode here. This cap is
#: therefore this study's own prudence rather than a quoted limit: it is
#: high enough that a real stage is bounded by the provider's latency
#: rather than by this package, and low enough that a typo cannot open
#: hundreds of billed connections at once.
MAX_UNFORCED_PROVIDER_CONCURRENCY = 64


def validate_provider_concurrency(value: int) -> int:
    """Refuse a concurrency below one, and return it otherwise.

    Spelled once, here, because three surfaces refuse the same value for
    the same reason -- the CLI parses it, the stage environment binds it,
    and the stage record persists it -- and a bound that disagreed between
    them would let an invocation run at a width its record could not hold.
    """
    if value < 1:
        raise ValueError(f"provider concurrency is at least 1; got {value}")
    return value


def resolve_provider_concurrency(value: int, *, force: bool) -> int:
    """Refuse a concurrency below one or above the cap without ``force``.

    The lower bound is arithmetic and cannot be forced: a width below one
    names no run at all. The upper bound is prudence rather than a
    provider-published limit -- see
    :data:`MAX_UNFORCED_PROVIDER_CONCURRENCY` -- so it is overridable, and
    the override is a separate explicit flag rather than a larger number,
    because the number alone cannot distinguish a deliberate choice from a
    typo with an extra digit.
    """
    validate_provider_concurrency(value)
    if value > MAX_UNFORCED_PROVIDER_CONCURRENCY and not force:
        raise ValueError(
            f"provider concurrency {value} exceeds the sanity cap of "
            f"{MAX_UNFORCED_PROVIDER_CONCURRENCY}. OpenRouter publishes "
            "no per-account concurrency limit for paid models, so this "
            "cap is this study's own prudence rather than a provider "
            "limit -- but a width this large opens that many billed "
            "connections at once, and an extra digit is far likelier "
            f"than a deliberate choice. Pass "
            f"{PROVIDER_CONCURRENCY_FORCE_FLAG} to mean it."
        )
    return value
