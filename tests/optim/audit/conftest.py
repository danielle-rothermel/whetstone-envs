"""A real fake-transport COPRO run, produced once per test session.

The audit reads persisted evidence, so its tests must run against evidence
whetstone actually wrote. A hand-built artifact would drift from the real
persisted shape and let an invariant pass against a format nobody produces.

The run is session-scoped because it costs a few seconds; it uses the fake
transport, so it makes no provider calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.optim.run import RunSpec, run_optimizer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="session")
def copro_run_dir(tmp_path_factory) -> Path:
    """One completed fake-transport COPRO run directory."""
    output = tmp_path_factory.mktemp("audit-runs") / "copro"
    return run_optimizer(
        RunSpec(
            optimizer="copro",
            transport="fake",
            split_sizes=(2, 2, 0),
            output_dir=output,
            run_id="c19-copro-audit-fixture",
        )
    )


@pytest.fixture
def mutable_run_dir(copro_run_dir, tmp_path) -> Path:
    """A per-test copy of the run, safe to mutate into a negative fixture."""
    from whetstone_envs.optim.audit._mutate import copy_run

    return copy_run(copro_run_dir, tmp_path / "mutated")


# --- GEPA run shapes (wave 2c) --------------------------------------------
#
# GEPA's evidence takes four durable shapes, and an invariant that only ever
# sees one is untested for the others. Each is a real fake-transport run --
# no provider calls -- differing only in what the scripted reflection
# returns and how large the metric-call ceiling is.


def _gepa_run(
    output: Path,
    run_id: str,
    *,
    split_sizes: tuple[int, int, int] = (2, 2, 0),
    gepa_max_metric_calls: int | None = None,
) -> Path:
    return run_optimizer(
        RunSpec(
            optimizer="gepa",
            transport="fake",
            output_dir=output,
            run_id=run_id,
            split_sizes=split_sizes,
            gepa_max_metric_calls=gepa_max_metric_calls,
        )
    )


@pytest.fixture(scope="session")
def gepa_run_dir(tmp_path_factory) -> Path:
    """The ordinary case: the reflection is accepted and mutates the seed."""
    output = tmp_path_factory.mktemp("audit-runs") / "gepa"
    return _gepa_run(output, "c19-gepa-audit-fixture")


@pytest.fixture(scope="session")
def gepa_multistep_run_dir(tmp_path_factory) -> Path:
    """A larger ceiling, so the run takes four steps rather than two."""
    output = tmp_path_factory.mktemp("audit-runs") / "gepa-multistep"
    return _gepa_run(
        output,
        "c19-gepa-audit-multistep",
        split_sizes=(4, 2, 0),
        gepa_max_metric_calls=12,
    )


@pytest.fixture(scope="session")
def gepa_seed_retained_run_dir(tmp_path_factory, gepa_reflection_bodies):
    """The reflection returns the seed, so the search finds nothing better."""
    from whetstone_envs.c19 import PROBES

    output = tmp_path_factory.mktemp("audit-runs") / "gepa-seed-retained"
    with gepa_reflection_bodies((PROBES.naive_template,)):
        return _gepa_run(output, "c19-gepa-audit-seed-retained")


@pytest.fixture(scope="session")
def gepa_skipped_run_dir(tmp_path_factory, gepa_reflection_bodies):
    """The reflection omits required placeholders, so the format rejects it.

    ``TemplateRenderContract`` requires every c19 placeholder, so this draft
    is rejected twice and the run records both attempts, the second
    ``exhausted``. That is the only way to produce a real, durable
    ``GepaSkippedMutation`` on the fake transport.
    """
    from whetstone_envs.c19 import PROBES

    output = tmp_path_factory.mktemp("audit-runs") / "gepa-skipped"
    bodies = ("a draft with no placeholders at all", PROBES.ceiling_template)
    with gepa_reflection_bodies(bodies):
        return _gepa_run(output, "c19-gepa-audit-skipped")


@pytest.fixture(scope="session")
def gepa_reflection_bodies():
    """Script what the fake reflection transport returns, for one run.

    ``build_c19_gepa_adapter`` hardcodes the family's probe pair as the
    scripted reflection bodies, and the seed-retained and rejected-draft
    shapes need different ones. Overriding the module's own transport
    builder keeps the run otherwise identical -- same control, same splits,
    same runner -- so the resulting artifact differs only in what the
    reflection said.
    """
    from contextlib import contextmanager
    from unittest.mock import patch

    import whetstone_envs.optim.gepa as gepa_module

    original = gepa_module._gepa_transport

    @contextmanager
    def scripted(bodies: tuple[str, ...]):
        def build(
            *,
            engine,
            prompt_adapter,
            proposal_bodies,  # noqa: ARG001 - overridden by `bodies`
            proposer_transport,
        ):
            return original(
                engine=engine,
                prompt_adapter=prompt_adapter,
                proposal_bodies=bodies,
                proposer_transport=proposer_transport,
            )

        with patch.object(gepa_module, "_gepa_transport", build):
            yield

    return scripted
