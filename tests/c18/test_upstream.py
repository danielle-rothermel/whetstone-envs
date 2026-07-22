"""The vendored-generator subprocess boundary (checklist A determinism).

Exercises :mod:`whetstone_envs.c18.upstream` directly: the boundary must
drive the vendored PrOntoQA generator reproducibly, parse its JSON into
the four public fields, run isolated from the read-only vendored tree, and
reconstruct the output filename correctly across depths.
"""

from __future__ import annotations

from whetstone_envs.c18 import upstream
from whetstone_envs.c18.upstream import (
    UPSTREAM_DEFAULT_SEED,
    RawInstance,
    generate_raw,
)


def test_generate_raw_returns_requested_count_and_fields() -> None:
    rows = generate_raw(hops=2, seed=1_000_100_001, num_trials=3)
    assert len(rows) == 3
    for row in rows:
        assert isinstance(row, RawInstance)
        assert row.question
        assert row.query
        assert row.answer in {"True", "False"}
        assert row.hops == 2
        assert row.query.startswith("True or false:")


def test_same_seed_is_deterministic() -> None:
    a = generate_raw(hops=3, seed=1_000_100_002, num_trials=4)
    b = generate_raw(hops=3, seed=1_000_100_002, num_trials=4)
    assert a == b


def test_different_seed_differs() -> None:
    a = generate_raw(hops=3, seed=1_000_100_003, num_trials=4)
    c = generate_raw(hops=3, seed=1_000_100_004, num_trials=4)
    assert a != c


def test_all_spec_depths_generate() -> None:
    # Every spec Section 1 depth (D1, D2, D3, D5) drives the boundary.
    for hops in (1, 2, 3, 5):
        rows = generate_raw(hops=hops, seed=1_000_100_010 + hops, num_trials=2)
        assert len(rows) == 2
        assert all(r.hops == hops for r in rows)


def test_output_filename_reconstruction() -> None:
    # The boundary reads exactly the file the vendored __main__ writes; a
    # non-default seed always carries the _seed suffix.
    assert upstream._output_filename(2, 12345) == "2hop_seed12345.json"
    assert upstream._output_filename(5, 999) == "5hop_seed999.json"
    # At the (never-used) default seed there is no suffix.
    assert (
        upstream._output_filename(1, UPSTREAM_DEFAULT_SEED) == "1hop.json"
    )


def test_vendored_tree_is_not_written_to() -> None:
    # The boundary runs in a temp dir; the read-only vendored source must
    # gain no json/log output files from a generation.
    before = {p.name for p in upstream._VENDOR_DIR.iterdir()}
    generate_raw(hops=1, seed=1_000_100_050, num_trials=2)
    after = {p.name for p in upstream._VENDOR_DIR.iterdir()}
    new = after - before
    assert not any(
        n.endswith((".json", ".log")) for n in new
    ), f"vendored tree gained output files: {new}"
