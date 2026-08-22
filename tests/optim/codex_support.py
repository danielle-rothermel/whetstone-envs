"""Driving one Codex arm run with the scripted fake CLI.

The fake CLI is a real subprocess that speaks real MCP over HTTP to the
whetstone-hosted evaluation server, so a run driven this way exercises the
production admission, lease, evaluation, and ledger path. Only the agent's
decisions are scripted.

This lives beside the tests rather than in the package because it is test
scaffolding, and it is shared rather than duplicated because the e2e suite
and the audit suite must drive *the same* path -- an audit that ran against
its own private construction would stop being evidence about the run the
e2e proves.

Every function here is macOS-only in effect: the Codex containment profile
is ``sandbox-exec``, so the caller gates on ``sys.platform``.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, cast

from dr_store.sync import open_sqlite
from whetstone.eval.reference_runtime import ReferenceEvalRuntimeConfig
from whetstone.experiment.candidate import candidate_reference
from whetstone.testing.fake_codex_cli import (
    FAKE_CODEX_TRANSCRIPT_ENV,
    install_fake_codex_binary,
)
from whetstone.testing.runtime import scripted_codex_preflight

from whetstone_envs.optim.codex import CodexTestSeam
from whetstone_envs.optim.families import family_spec

if TYPE_CHECKING:
    from pathlib import Path

    from dr_store import ObjectStore

#: The one tool the Codex MCP surface registers.
CODEX_EVAL_TOOL_NAME = "evaluate_candidate"

#: The splits every Codex fixture run uses. Small on purpose: the arm's
#: fidelity does not depend on split size, and each admitted call
#: evaluates the whole internal split.
CODEX_SPLIT_SIZES = (2, 2, 0)


def codex_tool_steps(
    *,
    templates: tuple[str, ...],
    selected: str | None,
    scratch: Path,
    family_id: str = "c19",
    split_sizes: tuple[int, int, int] = CODEX_SPLIT_SIZES,
) -> list[dict[str, Any]]:
    """Script one Codex session: evaluate each template, then select one.

    The tool arguments are rebuilt from the same deterministic experiment
    the run builds, because the fake CLI is told what to call rather than
    deciding for itself. ``base_ref`` and ``model_route`` are exactly what
    a real agent reads out of its prompt.

    ``selected`` names the call whose candidate the agent returns; ``None``
    is an honest "nothing beat the seed", which the run records as
    ``seed_retained``.
    """
    family = family_spec(family_id)
    pool = family.generate_pool(
        n_per_stratum=family.default_n_per_stratum,
        seed_start=family.default_pool_seed_start,
    )
    prepared = family.build_experiment(
        pool,
        split_sizes=split_sizes,
        num_seeds=1,
        provider_call_config=None,
    )
    experiment = prepared.experiment
    base_ref = candidate_reference(experiment.initial_candidate).record_ref
    # The engine is built only to read the model route a tool call must
    # carry; it evaluates nothing here.
    with open_sqlite(str(scratch / "codex-route.sqlite")) as store:
        engine = ReferenceEvalRuntimeConfig().build_engine(
            cast("ObjectStore", store),
            experiment=experiment,
            eval_runner=family.eval_runner(),
            mutation_field=family.mutation_field,
            render_contract=family.render_contract(),
        )
        model_route = engine.expected_model_route()
    steps: list[dict[str, Any]] = [
        {
            "tool": CODEX_EVAL_TOOL_NAME,
            "args": {
                "call_id": f"c{index}",
                "base_ref": {
                    "schema_name": base_ref.schema_name,
                    "content_hash": base_ref.content_hash,
                },
                "model_route": model_route,
                "template": template,
            },
        }
        for index, template in enumerate(templates, start=1)
    ]
    steps.append({"final": {"selected_call_id": selected}})
    return steps


def codex_test_seam(
    *, steps: list[dict[str, Any]], binary_dir: Path
) -> CodexTestSeam:
    """The seam that points one run at the scripted fake CLI.

    It names the scripted preflight, grants the fake CLI its transcript
    and ``PYTHONPATH``, and puts the shim first on the run's PATH.
    Nothing in ``RunSpec`` or the CLI can build one of these, so no
    production path reaches it.
    """
    install_fake_codex_binary(binary_dir)
    return CodexTestSeam(
        preflight=lambda **_kwargs: scripted_codex_preflight(),
        environment={
            "PATH": os.pathsep.join(
                [str(binary_dir), os.environ.get("PATH", "")]
            ),
            FAKE_CODEX_TRANSCRIPT_ENV: json.dumps(steps),
            # The real CLI needs no whetstone import; the scripted
            # stand-in is a Python module, so this grants it a path
            # explicitly rather than widening the runner's allowlist.
            "PYTHONPATH": _whetstone_source_root(),
            "OPENAI_API_KEY": "sk-fake",
        },
        extra_environment_keys=frozenset(
            {FAKE_CODEX_TRANSCRIPT_ENV, "PYTHONPATH"}
        ),
    )


def _whetstone_source_root() -> str:
    """The directory the fake CLI must have on its path to import whetstone."""
    from pathlib import Path

    import whetstone

    return str(Path(whetstone.__file__).resolve().parent.parent)


__all__ = [
    "CODEX_EVAL_TOOL_NAME",
    "CODEX_SPLIT_SIZES",
    "codex_test_seam",
    "codex_tool_steps",
]
