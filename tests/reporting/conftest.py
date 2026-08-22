from __future__ import annotations

import pytest

pytest.importorskip("whetstone.experiment.env")

from whetstone_envs.reporting.execution import C19EvalSpec, run_c19_evaluation


@pytest.fixture(scope="session")
def fake_eval_output(tmp_path_factory):
    return run_c19_evaluation(
        C19EvalSpec(
            transport="fake",
            role="internal",
            split_sizes=(2, 2, 0),
            run_id="reporting-tests",
            output_dir=tmp_path_factory.mktemp("reporting") / "eval",
        )
    )
