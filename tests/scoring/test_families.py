"""The family scorer registry is the one owner of every family's rule.

Two consumers apply a family's scoring rule: the eval-node runner that
produces a report, and :class:`~whetstone_envs.reporting.schema.EvalReport`'s
own check, which re-derives every scored observation and refuses the report
when a recorded score disagrees. If those two ever spell the rule
differently, a correct run becomes an unpublishable report -- which is
exactly what happened when the schema hard-coded c19's exact match and c18
started scoring terminal verdicts.

So the assertions here are:

* :func:`test_the_runner_scores_what_the_registry_scores` runs each family's
  actual eval-node runner over representative outputs and pins that its
  score equals :func:`whetstone_envs.scoring.families.family_score`. This is
  the anti-drift test: a runner that stopped routing through the registry
  fails here.
* :func:`test_the_registry_needs_no_optimizer_stack` pins that scoring a
  report's observation imports neither whetstone-ai nor the optimizer
  package, because a base install (Python 3.12, no ``optim`` extra) must be
  able to read a scored report.
* The per-family cases pin the rules themselves, so a change to either
  family's scoring is a deliberate edit to a golden expectation.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from whetstone.eval.eval_procedure import EvalProcedureRunner

from whetstone_envs.scoring.families import (
    family_score,
    scorable_family_ids,
)

#: Representative ``(output_text, gold, expected)`` triples per family,
#: chosen to exercise each family's rule where the two families *differ*:
#: c18's reasoned reply with a terminal verdict scores 1.0 while plain
#: exact match over the same text would score 0.0.
REPRESENTATIVE_OUTPUTS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "c19": (
        ("blue", "blue", 1.0),
        # Whitespace is normalized away; case is not.
        ("  blue  ", "blue", 1.0),
        ("Blue", "blue", 0.0),
        ("red", "blue", 0.0),
        ("", "blue", 0.0),
        ("1,2", "1,2", 1.0),
    ),
    "c18": (
        ("True", "True", 1.0),
        ("False", "False", 1.0),
        ("Every cat is a mammal.\nRex is a cat.\nTrue", "True", 1.0),
        ("Step one.\nStep two.\nfalse.", "False", 1.0),
        ("Every cat is a mammal.\nRex is a cat.\nTrue", "False", 0.0),
        ("I am not sure either way.", "True", 0.0),
    ),
}


def _runner_for(family: str) -> EvalProcedureRunner:
    """The eval-node runner the optimizer registry binds to ``family``.

    Imported inside the test because the optimizer family registry pulls in
    whetstone-ai, which is exactly the dependency the registry under test
    does not have.
    """
    from whetstone_envs.optim.families import family_spec

    return family_spec(family).eval_runner()


def test_every_scorable_family_has_representative_outputs() -> None:
    """A new family cannot be registered without golden cases here."""
    assert set(scorable_family_ids()) == set(REPRESENTATIVE_OUTPUTS)


@pytest.mark.parametrize("family", sorted(REPRESENTATIVE_OUTPUTS))
def test_the_registry_pins_each_family_rule(family: str) -> None:
    """Each family's rule is a golden expectation, not an inference."""
    for output_text, gold, expected in REPRESENTATIVE_OUTPUTS[family]:
        actual = family_score(
            family=family, output_text=output_text, gold=gold
        )
        assert actual == expected, (
            f"{family} scored {output_text!r} against {gold!r} as "
            f"{actual}, expected {expected}"
        )


@pytest.mark.parametrize("family", sorted(REPRESENTATIVE_OUTPUTS))
def test_the_runner_scores_what_the_registry_scores(family: str) -> None:
    """The run's own scorer and the report's re-derivation agree.

    Fails if a family's eval-node runner stops routing through the shared
    registry, which is the drift that makes a correct run unpublishable.
    """
    pytest.importorskip("whetstone.experiment.env")
    runner = _runner_for(family)
    for output_text, gold, _expected in REPRESENTATIVE_OUTPUTS[family]:
        score, _output, _metadata = runner.run_eval_node(
            node_id="drift-check",
            node_inputs={"provider_generation": output_text},
            evaluation_procedure_config_hash="",
            task=SimpleNamespace(gold=gold),
        )
        assert score == family_score(
            family=family, output_text=output_text, gold=gold
        ), (
            f"{family} runner and registry disagree on {output_text!r} "
            f"against {gold!r}"
        )


def test_unknown_family_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="unsupported family 'c99'"):
        family_score(family="c99", output_text="x", gold="x")


def test_the_registry_needs_no_optimizer_stack() -> None:
    """Reading a scored report must not require the stack that wrote it.

    A base install -- Python 3.12, no ``optim`` extra -- has no
    whetstone-ai, so importing the report schema or scoring an observation
    through it must not reach the optimizer package. Asserted in a
    subprocess because the test session itself has whetstone-ai installed
    and would report a false pass from an already-imported module.
    """
    probe = (
        "import sys\n"
        "import whetstone_envs.reporting.schema as schema\n"
        "assert schema._family_score("
        "family='c18',"
        " output_text='reasoning\\nTrue',"
        " gold='True') == 1.0\n"
        "leaked = sorted("
        "  name for name in sys.modules"
        "  if name == 'whetstone'"
        "  or name.startswith(('whetstone.', 'dr_providers',"
        " 'whetstone_envs.optim'))"
        ")\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"
