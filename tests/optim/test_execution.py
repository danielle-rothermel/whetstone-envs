from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from unittest import TestCase
from unittest.mock import patch

from dr_store.sync import open_sqlite

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dr_store import ObjectStore

from whetstone_envs.evaluation.runtime import (
    TransportKind,
    evaluation_reference_config,
    prepare_evaluation_runtime,
)
from whetstone_envs.evidence import load_trajectory_report
from whetstone_envs.optim.cli import build_parser, main
from whetstone_envs.optim.codex import (
    FORBID_REAL_CODEX_ENV,
    RealCodexRefusedError,
    build_codex_adapter,
    build_codex_control,
)
from whetstone_envs.optim.codex_runtime import (
    ENVS_CODEX_RUNTIME_CONFIG_CLASS,
    EnvsCodexRuntimeConfig,
)
from whetstone_envs.optim.problem import load_problem
from whetstone_envs.optim.run import run_optimizer
from whetstone_envs.optim.spec import (
    CodexSettings,
    CoproSettings,
    GepaSettings,
    Miprov2Settings,
    RunSpec,
    TrainValSizes,
)

PROBLEM_FILE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "optimization_problem.py"
)


class AcceptReplyRunner:
    def run_eval_node(
        self,
        *,
        node_id: str,
        node_inputs: Mapping[str, object],
        evaluation_procedure_config_hash: str,
        task: object,
    ) -> tuple[float | None, object | None, dict[str, object]]:
        _ = (node_id, evaluation_procedure_config_hash, task)
        text = node_inputs["provider_generation"]
        return 1.0, {"text": text}, {}


class ExecutionTests(TestCase):
    def test_generic_optimizer_paths_publish_complete_evidence(self) -> None:
        problem = load_problem(PROBLEM_FILE)
        partition = TrainValSizes(train_size=2, val_size=2)
        settings = (
            CoproSettings(),
            GepaSettings(partition=partition),
            Miprov2Settings(
                partition=partition, num_trials=2, num_candidates=2
            ),
        )
        with TemporaryDirectory() as directory:
            for optimizer in settings:
                with self.subTest(optimizer=optimizer.kind):
                    output = Path(directory) / optimizer.kind
                    run_optimizer(
                        RunSpec(
                            optimizer=optimizer,
                            transport=TransportKind.FAKE,
                            output_dir=output,
                            run_id=f"generic-{optimizer.kind}",
                        ),
                        problem=problem,
                    )
                    result = load_trajectory_report(output)
                    self.assertEqual(result.terminal_status, "complete")
                    self.assertEqual(result.mutation_field, "instruction")
                    self.assertTrue(result.resolutions)
                    evaluations = [
                        r.eval_report
                        for r in result.resolutions
                        if r.eval_report
                    ]
                    self.assertTrue(evaluations)
                    self.assertTrue(
                        all(
                            r.run.family == "example.echo" for r in evaluations
                        )
                    )
                    if optimizer.kind in ("copro", "gepa"):
                        self.assertTrue(
                            any(
                                getattr(r.results[0], "score", None) == 1.0
                                for r in evaluations
                            )
                        )

    def test_supplied_scorer_owns_scores_through_publication(self) -> None:
        problem = replace(
            load_problem(PROBLEM_FILE), eval_runner=AcceptReplyRunner()
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run_optimizer(
                RunSpec(
                    optimizer=CoproSettings(),
                    transport=TransportKind.FAKE,
                    output_dir=output,
                ),
                problem=problem,
            )
            result = load_trajectory_report(output)
            evaluations = [
                r.eval_report for r in result.resolutions if r.eval_report
            ]
            self.assertTrue(evaluations)
            self.assertTrue(
                all(
                    row.score == 1.0
                    for r in evaluations
                    for row in r.observations
                )
            )
            self.assertTrue(
                any(
                    row.output_text != task.gold
                    for r in evaluations
                    for row, task in zip(r.observations, r.tasks, strict=True)
                )
            )

    def test_invalid_inputs_fail_before_directory(
        self,
    ) -> None:
        problem = load_problem(PROBLEM_FILE)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            spec = RunSpec(
                optimizer=GepaSettings(
                    partition=TrainValSizes(train_size=1, val_size=1)
                ),
                transport=TransportKind.FAKE,
                output_dir=output,
            )
            with self.assertRaisesRegex(
                ValueError, "cover the internal split"
            ):
                run_optimizer(spec, problem=problem)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(
                ValueError, "explicit proposal_bodies"
            ):
                run_optimizer(
                    RunSpec(
                        optimizer=CoproSettings(),
                        transport=TransportKind.FAKE,
                        output_dir=output,
                    ),
                    problem=replace(problem, proposal_bodies=()),
                )
            self.assertFalse(output.exists())

    def test_cli_loads_python_file_and_has_no_task_selection_flags(
        self,
    ) -> None:
        flags = {
            flag
            for action in build_parser()._actions
            for flag in action.option_strings
        }
        self.assertIn("--problem", flags)
        self.assertTrue(
            flags.isdisjoint(
                {
                    "--family",
                    "--split-sizes",
                    "--n-per-category",
                    "--pool-seed-start",
                    "--model",
                    "--num-seeds",
                }
            )
        )
        with TemporaryDirectory() as directory:
            self.assertEqual(
                main(
                    [
                        "--problem",
                        str(PROBLEM_FILE),
                        "--optimizer",
                        "copro",
                        "--output",
                        str(Path(directory) / "cli"),
                    ]
                ),
                0,
            )
            invalid = Path(directory) / "invalid.py"
            invalid.write_text("problem = object()\n")
            with self.assertRaisesRegex(TypeError, "OptimizationProblem"):
                load_problem(invalid)

    def test_real_codex_authorization_remains_before_execution(self) -> None:
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, {FORBID_REAL_CODEX_ENV: "1"}),
        ):
            output = Path(directory) / "absent"
            with self.assertRaises(RealCodexRefusedError):
                run_optimizer(
                    RunSpec(
                        optimizer=CodexSettings(),
                        transport=TransportKind.FAKE,
                        output_dir=output,
                    ),
                    problem=load_problem(PROBLEM_FILE),
                    problem_file=PROBLEM_FILE,
                )
            self.assertFalse(output.exists())

    def test_codex_rejects_changed_problem_before_preflight(self) -> None:
        problem = load_problem(PROBLEM_FILE)
        runtime = prepare_evaluation_runtime(
            experiment=problem.prepared.experiment,
            eval_runner=problem.eval_runner,
            mutation_field=problem.contract.mutation_field,
            render_contract=problem.contract.render_contract(),
            fake_replies=problem.fake_replies,
            reference=evaluation_reference_config(TransportKind.FAKE),
            transport=TransportKind.FAKE,
            concurrency=3,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            changed = root / "changed.py"
            changed.write_text(
                PROBLEM_FILE.read_text().replace(
                    'namespace="example.echo"', 'namespace="example.changed"'
                )
            )
            with open_sqlite(str(root / "runtime.sqlite")) as store:
                objects = cast("ObjectStore", store)
                engine = runtime.build_engine(objects)
                control = build_codex_control(
                    engine=engine,
                    experiment=problem.prepared.experiment,
                    problem=problem,
                    model="unused-agent-model",
                )
                with patch(
                    "whetstone_envs.optim.codex.codex_auth_preflight"
                ) as preflight:
                    with self.assertRaisesRegex(
                        ValueError, "different Eval Config"
                    ):
                        build_codex_adapter(
                            store=objects,
                            control=control,
                            engine=engine,
                            runtime_config=EnvsCodexRuntimeConfig(
                                problem_file=str(changed),
                                transport="fake",
                                provider_concurrency=3,
                            ),
                            reward_policy=problem.prepared.experiment.reward_policy,
                            store_path=root / "runtime.sqlite",
                            run_root=root / "codex",
                        )
                    preflight.assert_not_called()

    def test_codex_reconstructs_same_runtime_in_a_fresh_process(self) -> None:
        self.assertEqual(
            ENVS_CODEX_RUNTIME_CONFIG_CLASS,
            "whetstone_envs.optim.codex_runtime:EnvsCodexRuntimeConfig",
        )
        problem = load_problem(PROBLEM_FILE)
        runtime = prepare_evaluation_runtime(
            experiment=problem.prepared.experiment,
            eval_runner=problem.eval_runner,
            mutation_field=problem.contract.mutation_field,
            render_contract=problem.contract.render_contract(),
            fake_replies=problem.fake_replies,
            reference=evaluation_reference_config(TransportKind.FAKE),
            transport=TransportKind.FAKE,
            concurrency=3,
        )
        config = EnvsCodexRuntimeConfig(
            problem_file=str(PROBLEM_FILE),
            transport="fake",
            provider_concurrency=3,
        )
        self.assertEqual(
            set(config.model_dump()),
            {
                "problem_file",
                "transport",
                "provider_concurrency",
                "partial_log_path",
                "prompt_cache_path",
                "row_job_entrypoint",
                "unit_deadline_seconds",
            },
        )
        code = """
import json, sys
from dr_store.sync import open_sqlite

from whetstone_envs.optim.codex_runtime import EnvsCodexRuntimeConfig
config=EnvsCodexRuntimeConfig.model_validate_json(sys.argv[1])
with open_sqlite(sys.argv[2]) as store:
    engine=config.build_engine(store)
    print(json.dumps([str(engine.eval_config_ref.record_ref.content_hash),str(engine.task_model_identity_hash()),list(engine.sampling.task_hashes),str(engine.execution_policy_identity_hash())]))
"""
        with TemporaryDirectory() as directory:
            with open_sqlite(str(Path(directory) / "parent.sqlite")) as store:
                engine = runtime.build_engine(cast("ObjectStore", store))
                expected = [
                    str(engine.eval_config_ref.record_ref.content_hash),
                    str(engine.task_model_identity_hash()),
                    list(engine.sampling.task_hashes),
                    str(engine.execution_policy_identity_hash()),
                ]
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    config.model_dump_json(),
                    str(Path(directory) / "child.sqlite"),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            self.assertEqual(json.loads(result.stdout), expected)
