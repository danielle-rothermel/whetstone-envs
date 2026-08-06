from types import ModuleType

import whetstone_envs
import whetstone_envs.instances
import whetstone_envs.manifests
import whetstone_envs.pools
import whetstone_envs.probes
import whetstone_envs.scoring


def assert_exports(module: ModuleType, expected: list[str]) -> None:
    assert module.__all__ == expected
    for name in expected:
        qualified_name = f"{module.__name__}.{name}"
        assert hasattr(module, name), f"missing export: {qualified_name}"


def test_root_namespace_is_empty() -> None:
    assert whetstone_envs.__all__ == []


def test_instances_public_exports() -> None:
    assert_exports(
        whetstone_envs.instances,
        ["Instance", "make_instance", "public_prompt_identity"],
    )


def test_pools_public_exports() -> None:
    assert_exports(whetstone_envs.pools, ["PoolSplit", "TaskPool"])


def test_probes_public_exports() -> None:
    assert_exports(
        whetstone_envs.probes,
        ["ProbePair", "normalize", "render_with_prompt_inputs"],
    )


def test_scoring_public_exports() -> None:
    assert_exports(
        whetstone_envs.scoring,
        [
            "Aggregate",
            "Observation",
            "Outcome",
            "aggregate",
            "aggregate_overall",
            "aggregate_stratum",
            "aggregate_task",
            "exact_match",
            "failed",
            "missing",
            "scored",
        ],
    )


def test_manifests_public_exports() -> None:
    assert_exports(
        whetstone_envs.manifests,
        ["MANIFEST_SCHEMA_VERSION", "Manifest", "content_hash"],
    )
