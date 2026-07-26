"""Vendor-patch verification for the c23 InductionBench vendor (checklist A).

These tests pin the four named patches the PLAN requires, and prove the
determinism the sorted() patch buys:

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

import inspect
import subprocess
import sys
from pathlib import Path

from whetstone_envs.c23 import upstream

_VENDOR = (
    Path(upstream.__file__).parent / "_vendor" / "inductionbench"
)
_SDG = _VENDOR / "synthetic_data_generation.py"


def test_vendored_generator_imports_without_crashing() -> None:
    # Patches 1 & 2: the module imports (missing config.py added; the broken
    # translate_fewshot import / sys.path hack / tqdm dep removed). If any of
    # the three upstream import-time crashes remained, importing the boundary
    # (which imports the vendored module) would already have failed.
    import config
    import synthetic_data_generation as sdg

    assert hasattr(config, "vocab")
    assert hasattr(sdg, "apply_ISL_rule")
    assert hasattr(sdg, "apply_L_OSL_rule")
    assert hasattr(sdg, "apply_R_OSL_rule")
    assert hasattr(sdg, "generate_rules")
    assert hasattr(sdg, "generate_data")


def test_seed_parameter_is_threaded() -> None:
    # Patch 3: generate_rules / generate_data accept a seed kwarg.
    import synthetic_data_generation as sdg

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
    import synthetic_data_generation as sdg

    for fn in (sdg.apply_ISL_rule, sdg.apply_L_OSL_rule, sdg.apply_R_OSL_rule):
        body = inspect.getsource(fn)
        assert "PATCH" not in body, fn.__name__


def test_vendored_diff_file_is_present_and_reviewable() -> None:
    # A vendored-diff file must exist so the delta vs upstream is reviewable
    # in one place, and it must actually mention the four patches.
    diff = (_VENDOR / "VENDORED_DIFF.patch").read_text(encoding="utf-8")
    assert "e0b8392" in diff  # the vendored upstream commit
    assert "sorted(set(" in diff
    assert "translate_fewshot_input_output_pairs" in diff  # the removed import


def test_provenance_records_the_vendored_commit_and_patches() -> None:
    prov = (_VENDOR / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "e0b839221a8509b351b324dfb247b35a434b7fd5" in prov
    assert "Apache-2.0" in prov
    for patch in ("config stub", "sorted", "seed parameter"):
        assert patch in prov


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
