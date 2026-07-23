r"""The subprocess/import boundary around the vendored PrOntoQA generator.

The c18 baseline spec (fixed-constraints callout) reseeds instances
directly from ``asaparov/prontoqa``'s ``run_experiment.py --model-name
json`` -- never reusing a published instance. This module is the thin
boundary that drives the *vendored* generator (see
``_vendor/prontoqa/PROVENANCE.md``) as a subprocess and parses its JSON
output into plain records, so nothing about the generator's internal
proof object leaks into the rest of c18 (the oracle re-derives the label
independently from the public text).

Two upstream integration facts from the repos review drive the design:

* ``run_experiment.py`` opens ``bad_patterns.txt`` with a **relative**
  path at import time and writes its output file to the process **cwd**,
  so the subprocess must run from a directory that holds both. We run in
  a fresh temp dir populated with symlinks to the vendored source, so the
  vendored tree stays read-only and concurrent generations never collide
  on the output filename.
* the output filename is derived only from the run config (e.g.
  ``2hop_seed12345.json``); we reconstruct it from the same rules the
  vendored ``__main__`` uses so we can read exactly the file it wrote.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_VENDOR_DIR = Path(__file__).parent / "_vendor" / "prontoqa"
_RUN_EXPERIMENT = _VENDOR_DIR / "run_experiment.py"

# The upstream default seed (``run_experiment.py`` argparse). The output
# filename only carries a ``_seed<N>`` suffix when the seed differs from
# this default, so we always pass a fresh non-default seed and rely on the
# suffix being present.
UPSTREAM_DEFAULT_SEED = 62471893

# Files the vendored generator needs colocated in its working directory.
# Every ``.py`` module (they import each other by bare name) plus the
# relative ``bad_patterns.txt`` the module opens at import time.
_VENDOR_FILES: tuple[str, ...] = (
    "run_experiment.py",
    "theory.py",
    "syntax.py",
    "proof.py",
    "prompt.py",
    "fol.py",
    "bad_patterns.txt",
)


class UpstreamError(RuntimeError):
    """Raised when the vendored generator fails or emits no output file."""


@dataclass(frozen=True, slots=True)
class RawInstance:
    """One parsed ``test_example`` from the generator's JSON output.

    Only the four public fields the c18 task uses are retained; the
    in-context demo examples and the optional ``chain_of_thought`` gold
    trace are dropped (the spec keeps the label only, and the oracle
    re-derives it from ``question`` + ``query`` alone).
    """

    question: str
    query: str
    answer: str
    hops: int


def _output_filename(
    hops: int, seed: int, distractors: str = "relevant"
) -> str:
    """Reconstruct the vendored json output filename for this run config.

    Mirrors the ``log_suffix`` assembly in the vendored ``__main__`` for
    the config this boundary uses: fictional ontology, ModusPonens, 8-shot,
    COT, ``--test-distractors`` pinned equal to ``--distractors`` -- every
    one of which contributes an *empty* suffix. The distractor mode is the
    one config axis this boundary varies (the c18 base + hard presets use
    ``relevant`` and ``none``), and the vendored suffix logic appends
    ``_nodistractor`` for ``none`` / ``_irrelevantdistractor`` for
    ``irrelevant`` while ``relevant`` (the upstream default) stays empty.
    Both ``--distractors`` and ``--test-distractors`` are set equal here, so
    the ``_testdistractor`` mismatch suffixes never appear. Kept in lockstep
    with that code so we read exactly the file the generator wrote.
    """
    suffix = f"{hops}hop"
    if distractors == "none":
        suffix += "_nodistractor"
    elif distractors == "irrelevant":
        suffix += "_irrelevantdistractor"
    if seed != UPSTREAM_DEFAULT_SEED:
        suffix += f"_seed{seed}"
    return suffix + ".json"


def generate_raw(
    *,
    hops: int,
    seed: int,
    num_trials: int,
    ontology: str = "fictional",
    distractors: str = "relevant",
    timeout_s: float = 300.0,
) -> list[RawInstance]:
    """Run the vendored generator once and return its parsed instances.

    Parameters
    ----------
    hops:
        The hop-depth loop bound (``--min-hops`` == ``--max-hops`` ==
        ``hops``); the vendored proof depth is ``1 + hops``.
    seed:
        A fresh non-default ``--seed`` (asserted fresh by the caller).
    num_trials:
        ``--num-trials``: how many test instances to emit at this depth.
    ontology / distractors:
        Held to the spec's fixed constraints by default (``fictional``
        nonce ontology; ``relevant`` distractors -- Open Decision O1's
        default). ``--test-distractors`` is pinned equal to
        ``--distractors`` so train and test conditions match (avoids the
        vendored default's train/test distractor mismatch).

    The subprocess runs in a throwaway temp dir symlinked to the vendored
    source, so the read-only vendored tree is never written to and
    parallel calls cannot collide on the fixed output filename.
    """
    with tempfile.TemporaryDirectory(prefix="c18-prontoqa-") as tmp:
        work = Path(tmp)
        for name in _VENDOR_FILES:
            (work / name).symlink_to(_VENDOR_DIR / name)
        cmd = [
            sys.executable,
            "run_experiment.py",
            "--model-name",
            "json",
            "--model-size",
            "1",
            "--num-trials",
            str(num_trials),
            "--min-hops",
            str(hops),
            "--max-hops",
            str(hops),
            "--ontology",
            ontology,
            "--distractors",
            distractors,
            "--test-distractors",
            distractors,
            "--seed",
            str(seed),
        ]
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell, vendored script
            cmd,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            msg = (
                f"vendored prontoqa exited {proc.returncode} for "
                f"hops={hops} seed={seed}: {proc.stderr[-500:]}"
            )
            raise UpstreamError(msg)
        out_path = work / _output_filename(hops, seed, distractors)
        if not out_path.exists():
            msg = (
                f"vendored prontoqa wrote no output file "
                f"{out_path.name!r} for hops={hops} seed={seed}"
            )
            raise UpstreamError(msg)
        data: object = json.loads(out_path.read_text(encoding="utf-8"))
    return _parse_examples(data, hops)


def _require(fields: dict[str, object], key: str) -> str:
    """Read ``key`` from ``fields`` as a string, or raise ``UpstreamError``."""
    if key not in fields:
        msg = f"generator output missing expected field {key!r}"
        raise UpstreamError(msg)
    return str(fields[key])


def _as_str_dict(value: object) -> dict[str, object] | None:
    """Return ``value`` as a ``{str: object}`` dict, or ``None``.

    ``json.loads`` yields ``dict`` objects keyed by ``str``; this narrows
    an untyped decoded value to that concrete shape (copying into a fresh
    dict so the static type is exact) or rejects a non-object.
    """
    if not isinstance(value, dict):
        return None
    return {str(k): v for k, v in value.items()}


def _parse_examples(data: object, hops: int) -> list[RawInstance]:
    """Project the generator's JSON into :class:`RawInstance` records.

    The top level maps ``example{i}`` to an object whose ``test_example``
    holds the scored instance; the in-context demos and the CoT trace are
    intentionally discarded.
    """
    top = _as_str_dict(data)
    if top is None:
        msg = "generator output is not a JSON object"
        raise UpstreamError(msg)
    out: list[RawInstance] = []
    for key in sorted(top, key=_example_index):
        block = _as_str_dict(top[key])
        if block is None:
            continue
        test_example = _as_str_dict(block.get("test_example"))
        if test_example is None:
            continue
        out.append(
            RawInstance(
                question=_require(test_example, "question"),
                query=_require(test_example, "query"),
                answer=_require(test_example, "answer"),
                hops=hops,
            ),
        )
    return out


def _example_index(key: str) -> int:
    """Sort key extracting the integer from an ``example{i}`` label."""
    digits = "".join(ch for ch in key if ch.isdigit())
    return int(digits) if digits else 0
