"""Vendor-patch verification for the c23 InductionBench vendor (checklist A).

These tests pin the named vendor modifications and prove the determinism
the canonical-order change buys:

1. the vendored module IMPORTS (the three upstream import-time crashes are
   fixed): missing config.py, the broken translate_fewshot import, and the
   sys.path.add-style hacks;
2. the seed parameter is really threaded (generate_rules / generate_data
   accept and honor it);
3. exactly the six named list(set(...)) sites became sorted(...), and the
   three apply_*_rule oracle transducers are UNTOUCHED versus upstream;
4. the determinism regenerate-twice proof under a randomized PYTHONHASHSEED
   (run in a subprocess so the hash seed actually varies), which is the
   whole reason the sorted() patches exist.
"""

from __future__ import annotations

import hashlib
import inspect
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from whetstone_envs.c23 import upstream
from whetstone_envs.c23._vendor.inductionbench import config
from whetstone_envs.c23._vendor.inductionbench import (
    synthetic_data_generation as sdg,
)

_VENDOR = Path(upstream.__file__).parent / "_vendor" / "inductionbench"
_SDG = _VENDOR / "synthetic_data_generation.py"
_UPSTREAM_SHA256 = {
    "synthetic_data_generation.py": (
        "a847e6317316af0ca6ffe1d521cc9d167dc6d437b87d41398d5a0a00faad44ac"
    ),
    "utils.py": (
        "b4ac3a724ac2aa184709aadb1cd6d2a8d4edf2e0386f8ddcebe21db51a308477"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_generator_imports_without_crashing() -> None:
    # Patches 1 & 2: the module imports (missing config.py added; the broken
    # translate_fewshot import / sys.path hack / tqdm dep removed). If any of
    # the three upstream import-time crashes remained, importing the boundary
    # (which imports the vendored module) would already have failed.
    assert hasattr(config, "vocab")
    assert hasattr(sdg, "apply_ISL_rule")
    assert hasattr(sdg, "apply_L_OSL_rule")
    assert hasattr(sdg, "apply_R_OSL_rule")
    assert hasattr(sdg, "generate_rules")
    assert hasattr(sdg, "generate_data")


def test_seed_parameter_is_threaded() -> None:
    # Patch 3: generate_rules / generate_data accept a seed kwarg.
    assert "seed" in inspect.signature(sdg.generate_rules).parameters
    assert "seed" in inspect.signature(sdg.generate_data).parameters


def test_exactly_six_sorted_patch_sites_and_no_stray_list_set() -> None:
    # Patch 4: the six named list(set(...)) sites are now sorted(set(...)).
    # The PLAN names 6 call sites (upstream lines 53/61/107/132/208/244).
    src = _SDG.read_text(encoding="utf-8")
    assert src.count("PATCH 4/4") == 6
    # No list(set(...)) over strings remains at the six patched forms. (The
    # source may retain unrelated set() usage, but not the patched idioms.)
    assert "list(set(all_k_strings))" not in src
    assert "list(set(config.vocab).difference([k_string[-1]]))" not in src
    assert "list(set(config.vocab).difference([new_rule[-1]]))" not in src
    assert "list(set(list(sample.values())))" not in src


def test_apply_rule_transducers_are_upstream_unmodified() -> None:
    # The three oracle transducers must be reused UNMODIFIED. Assert their
    # source bodies contain no PATCH marker (every vendor edit is tagged).
    for fn in (sdg.apply_ISL_rule, sdg.apply_L_OSL_rule, sdg.apply_R_OSL_rule):
        body = inspect.getsource(fn)
        assert "PATCH" not in body, fn.__name__


def test_vendored_diff_file_is_present_and_reviewable() -> None:
    # A vendored-diff file must exist so the complete delta versus upstream
    # is reviewable in one place.
    diff = (_VENDOR / "VENDORED_DIFF.patch").read_text(encoding="utf-8")
    assert "e0b8392" in diff  # the vendored upstream commit
    assert "sorted(set(" in diff
    assert "translate_fewshot_input_output_pairs" in diff  # the removed import
    assert "--- /dev/null\n+++ b/config.py" in diff
    assert "--- /dev/null\n+++ b/__init__.py" in diff


def test_vendored_diff_reverse_and_forward_applies(
    tmp_path: Path,
) -> None:
    patch_executable = shutil.which("patch")
    if patch_executable is None:
        pytest.skip("patch executable is unavailable")

    names = (
        "synthetic_data_generation.py",
        "utils.py",
        "config.py",
        "__init__.py",
    )
    for name in names:
        shutil.copyfile(_VENDOR / name, tmp_path / name)

    patch_path = _VENDOR / "VENDORED_DIFF.patch"
    subprocess.run(  # noqa: S603 - resolved system patch executable
        [
            patch_executable,
            "--batch",
            "-R",
            "-p1",
            "-i",
            str(patch_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not (tmp_path / "config.py").exists()
    assert not (tmp_path / "__init__.py").exists()
    for name, expected_hash in _UPSTREAM_SHA256.items():
        assert _sha256(tmp_path / name) == expected_hash

    subprocess.run(  # noqa: S603 - resolved system patch executable
        [
            patch_executable,
            "--batch",
            "-p1",
            "-i",
            str(patch_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    for name in names:
        assert (tmp_path / name).read_bytes() == (_VENDOR / name).read_bytes()


def test_provenance_records_the_vendored_commit_and_patches() -> None:
    prov = (_VENDOR / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "e0b839221a8509b351b324dfb247b35a434b7fd5" in prov
    assert "Apache-2.0" in prov
    for patch in (
        "config stub",
        "sorted",
        "seed parameter",
        "package-relative",
        "private RNG",
    ):
        assert patch in prov


def test_threaded_same_seed_generation_is_deterministic_and_rng_safe() -> None:
    random.seed(20_260_724)
    state = random.getstate()

    def generate() -> list[upstream.RawInstance]:
        return upstream.generate_raw(
            rule_type=upstream.ISL,
            k=2,
            vocab_size=4,
            seed=555_000_000,
            num_instances=2,
            sample_size_times=10,
            max_query_len=8,
            n_demos=6,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: generate(), range(8)))

    assert all(result == results[0] for result in results)
    assert random.getstate() == state


def test_package_import_ignores_unrelated_bare_modules() -> None:
    snippet = """
import sys
import types

foreign = {}
for name in ("config", "utils", "synthetic_data_generation"):
    module = types.ModuleType(name)
    foreign[name] = module
    sys.modules[name] = module

path_before = tuple(sys.path)
from whetstone_envs.c23 import upstream

assert tuple(sys.path) == path_before
assert upstream.config is not foreign["config"]
assert upstream._sdg is not foreign["synthetic_data_generation"]
assert sys.modules["config"] is foreign["config"]
assert sys.modules["utils"] is foreign["utils"]
assert (
    sys.modules["synthetic_data_generation"]
    is foreign["synthetic_data_generation"]
)
upstream.generate_raw(
    rule_type=upstream.ISL,
    k=2,
    vocab_size=4,
    seed=555_000_000,
    num_instances=1,
    sample_size_times=10,
    max_query_len=8,
    n_demos=6,
)
"""
    subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", snippet],
        check=True,
        env={"PATH": _path()},
    )


def test_non_positive_demo_and_query_configuration_bounds() -> None:
    common = {
        "rule_type": upstream.ISL,
        "k": 2,
        "vocab_size": 4,
        "seed": 555_000_000,
        "num_instances": 1,
        "sample_size_times": 10,
    }
    with pytest.raises(upstream.UpstreamError, match="n_demos"):
        upstream.generate_raw(**common, max_query_len=8, n_demos=-1)
    with pytest.raises(upstream.UpstreamError, match="max_query_len"):
        upstream.generate_raw(**common, max_query_len=1, n_demos=6)
    with pytest.raises(upstream.UpstreamError, match="exceeds"):
        upstream.generate_raw(**common, max_query_len=8, n_demos=10_000)


def test_query_length_must_leave_space_outside_full_demos() -> None:
    with pytest.raises(
        upstream.UpstreamError,
        match=r"max_query_len=2 leaves no held-out query",
    ):
        upstream.generate_raw(
            rule_type=upstream.ISL,
            k=2,
            vocab_size=4,
            seed=555_000_000,
            num_instances=1,
            sample_size_times=10,
            max_query_len=2,
            n_demos=6,
        )


def test_exhaustive_fallback_survives_32_random_candidate_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def colliding_candidates(
        _vocab: object,
        *,
        count: int,
        max_len: int,
        rng: object,
    ) -> list[list[str]]:
        assert max_len == 3
        assert rng is not None
        return [["aa"] * 32 for _ in range(count)]

    monkeypatch.setattr(
        upstream,
        "_draw_query_candidates",
        colliding_candidates,
    )
    generated = upstream.generate_raw(
        rule_type=upstream.ISL,
        k=2,
        vocab_size=4,
        seed=555_000_000,
        num_instances=2,
        sample_size_times=10,
        max_query_len=3,
        n_demos=6,
    )
    assert all(len(instance.query) == 3 for instance in generated)
    assert all(instance.query != "aa" for instance in generated)


def test_upstream_public_api_rejects_non_strict_integers() -> None:
    with pytest.raises(TypeError, match=r"vocab_size.*integer"):
        upstream.vocab_for(
            vocab_size=True,
        )
    with pytest.raises(TypeError, match=r"k.*integer"):
        upstream.apply_rule(
            upstream.ISL,
            2.0,  # ty: ignore[invalid-argument-type]
            {"aa": "b"},
            "aa",
        )
    with pytest.raises(TypeError, match=r"seed.*integer"):
        upstream.generate_raw(
            rule_type=upstream.ISL,
            k=2,
            vocab_size=4,
            seed=False,
            num_instances=1,
            sample_size_times=10,
            max_query_len=3,
            n_demos=6,
        )


# The determinism regenerate-twice proof. Run TWICE in fresh subprocesses
# with an explicit differing PYTHONHASHSEED so string-hash order genuinely
# varies between the two runs; a byte-identical content hash across them is
# the proof the sorted() patch killed the hash-order nondeterminism.
_DET_SNIPPET = (
    "from whetstone_envs.c23.generate import generate_pool;"
    "from whetstone_envs.core.manifest import content_hash;"
    "print(content_hash(generate_pool(n_per_stratum=6)))"
)


def _hash_with_hashseed(hashseed: str) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", _DET_SNIPPET],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": hashseed, "PATH": _path()},
    )
    return proc.stdout.strip()


def _path() -> str:
    import os

    return os.environ.get("PATH", "")


def test_regenerating_twice_is_byte_identical_across_hash_seeds() -> None:
    # Determinism (checklist A): two runs under DIFFERENT PYTHONHASHSEED
    # values must produce the identical pool content hash. This is the
    # regenerate-twice test the sorted() patch exists to satisfy.
    h1 = _hash_with_hashseed("0")
    h2 = _hash_with_hashseed("1")
    assert h1 == h2, (h1, h2)
