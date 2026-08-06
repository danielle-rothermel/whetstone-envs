import importlib

import pytest

import whetstone_envs
import whetstone_envs.core
import whetstone_envs.instances
import whetstone_envs.manifests
import whetstone_envs.pools
import whetstone_envs.probes
import whetstone_envs.scoring


def test_package_imports() -> None:
    assert whetstone_envs is not None


def test_functional_packages_import() -> None:
    assert whetstone_envs.instances.__all__
    assert whetstone_envs.pools.__all__
    assert whetstone_envs.probes.__all__
    assert whetstone_envs.scoring.__all__
    assert whetstone_envs.manifests.__all__
    assert whetstone_envs.core.__all__ == []


@pytest.mark.parametrize(
    "module_name",
    [
        "instance",
        "manifest",
        "pool",
        "probes",
        "scoring",
    ],
)
def test_old_core_modules_are_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"whetstone_envs.core.{module_name}")
