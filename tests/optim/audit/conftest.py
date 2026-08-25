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


#: Three further scripted proposer bodies, each satisfying C19's render
#: contract. They are what lets a fake COPRO run at ``breadth`` 3 fill a
#: genuinely multi-draft round: the family scripts a ceiling draft and the
#: naive seed, and a draft filling the seed's slot is rejected as a no-op
#: mutation, so without these a wider round underfills and the run ends in
#: a proposal-cardinality failure.
_DRAFT_BODY_TAIL = "Grid:\n{grid}\n\nActions: {command}\n\n{question}\nAnswer."
MULTI_DRAFT_PROPOSAL_BODIES = (
    f"Alpha: reason step by step.\n\n{_DRAFT_BODY_TAIL}",
    f"Beta: trace each move.\n\n{_DRAFT_BODY_TAIL}",
    f"Gamma: simulate carefully.\n\n{_DRAFT_BODY_TAIL}",
)


@pytest.fixture(scope="session")
def copro_multi_draft_run_dir(tmp_path_factory) -> Path:
    """A fake COPRO run whose rounds really do propose several drafts.

    ``breadth=3`` with ``depth=2`` gives a seed round of two drafts and a
    history round of three, so an invariant over "proposals within a round"
    has real pairs to compare instead of passing vacuously.
    """
    output = tmp_path_factory.mktemp("audit-runs-multi") / "copro"
    return run_optimizer(
        RunSpec(
            optimizer="copro",
            transport="fake",
            split_sizes=(2, 2, 0),
            output_dir=output,
            run_id="c19-copro-audit-multi-draft",
            copro_breadth=3,
            copro_depth=2,
            extra_proposal_bodies=MULTI_DRAFT_PROPOSAL_BODIES,
        )
    )


# --- COPRO seed retention -------------------------------------------------
#
# whetstone-ai 0.1.16 lets COPRO keep its own seed when the seed ties or wins
# the terminal ranking, at both of its terminal emission points. The fake
# transport always drafts a usable proposal, so no scripted run reaches
# either branch; these are built by mutating a real run into the shape
# whetstone persists, and stay schema-valid ``OptimResult`` artifacts because
# ``OptimStepResult._validate`` enforces every structural clause of a
# retention.


@pytest.fixture(scope="session")
def copro_seed_retained_run_dir(copro_run_dir, tmp_path_factory) -> Path:
    """A retention at COPRO's ordinary finalizing step (``depth + 1``)."""
    from tests.optim.audit.copro_fixtures import (
        seed_retained_at_ordinary_finalize,
    )

    destination = tmp_path_factory.mktemp("audit-runs") / "copro-retained"
    return seed_retained_at_ordinary_finalize(copro_run_dir, destination)


@pytest.fixture(scope="session")
def copro_seed_retained_early_run_dir(copro_run_dir, tmp_path_factory) -> Path:
    """A retention from the early terminal, short of the configured depth."""
    from tests.optim.audit.copro_fixtures import (
        seed_retained_at_early_terminal,
    )

    destination = (
        tmp_path_factory.mktemp("audit-runs") / "copro-retained-early"
    )
    return seed_retained_at_early_terminal(copro_run_dir, destination)


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
    # An even split of whatever internal size this fixture asked for; the
    # control now requires an explicit disjoint partition.
    train = split_sizes[0] // 2
    return run_optimizer(
        RunSpec(
            optimizer="gepa",
            transport="fake",
            output_dir=output,
            run_id=run_id,
            split_sizes=split_sizes,
            gepa_max_metric_calls=gepa_max_metric_calls,
            train_size=train,
            val_size=split_sizes[0] - train,
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


@pytest.fixture(scope="session")
def codex_run_dir(tmp_path_factory) -> Path:
    """One completed Codex arm run, driven by the scripted fake CLI.

    The fake CLI is a real subprocess speaking real MCP to the real
    evaluation server, so this is genuine Codex evidence -- an admission
    ledger, tool evidence, and an output artifact -- rather than a
    hand-built approximation of one. Only the agent's decisions are
    scripted.

    Skipped off macOS: the Codex containment profile is ``sandbox-exec``.
    """
    import sys

    if sys.platform != "darwin":
        pytest.skip("the Codex sandbox is macOS sandbox-exec only")

    from tests.optim.codex_support import (
        CODEX_SPLIT_SIZES,
        codex_test_seam,
        codex_tool_steps,
    )
    from whetstone_envs.c19 import PROBES
    from whetstone_envs.optim.codex import CODEX_EVALUATE_CALL_CAP

    root = tmp_path_factory.mktemp("audit-runs")
    return run_optimizer(
        RunSpec(
            optimizer="codex",
            transport="fake",
            split_sizes=CODEX_SPLIT_SIZES,
            output_dir=root / "codex",
            run_id="c19-codex-audit-fixture",
            codex_capacity=CODEX_EVALUATE_CALL_CAP,
        ),
        codex_test_seam=codex_test_seam(
            steps=codex_tool_steps(
                templates=(PROBES.ceiling_template,),
                selected="c1",
                scratch=root,
            ),
            binary_dir=root / "codex-bin",
        ),
    )
