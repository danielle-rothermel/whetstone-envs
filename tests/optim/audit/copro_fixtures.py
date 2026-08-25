"""Build one negative fixture per COPRO invariant from a real run.

Section 3.2 of the Step 10 assignment makes a failing fixture a shipping
requirement: an invariant with no evidence that makes it FAIL is not yet an
invariant. Each builder here starts from an unmutated fake-transport COPRO
run and violates exactly one thing, so the resulting artifact is still in
the format whetstone actually persists.

Three kinds of mutation appear, in increasing depth:

- **A field of one step.** ``mutate_run`` handles it: rewrite the field and
  reseal the step chain.
- **A record the result points at.** Put a variant into the run's own store
  and repoint the ref -- the record must stay schema-valid, so a variant is
  derived from the real one rather than invented.
- **The run record itself.** Its optimizer control and seed candidate are
  embedded in every step request, so :func:`reseal_run_binding` re-derives
  the wrapper everywhere before resealing the chain.

``mutate_run``'s no-op guard does not reach the deeper two, so each builder
here asserts its own precondition instead: a builder that quietly changed
nothing would make its negative test pass for the wrong reason.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from dr_store.sync import open_sqlite
from whetstone.core.identity import TypedRef, typed_ref_for_record
from whetstone.experiment.binding import EVAL_CONFIG_RECORD_SCHEMA
from whetstone.experiment.candidate import Candidate, candidate_reference
from whetstone.experiment.reward import REWARD_SCHEMA
from whetstone.optim.copro.control import (
    COPRO_CONTROL_SCHEMA,
    CoproControl,
)

from whetstone_envs.optim.audit._evidence import (
    RESULT_FILENAME,
    RUNTIME_STORE_FILENAME,
)
from whetstone_envs.optim.audit._mutate import (
    MutationError,
    copy_run,
    mutate_run,
    put_record,
    reseal_run_binding,
    reseal_step_chain,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Where each mutation reaches inside ``result.json``. Named so a failing
#: negative test says which field it violated rather than showing a tuple.
FIRST_STEP_INTENTS = ("step_results", 0, "record", "resolved_intents")
FIRST_STEP_PROPOSALS = ("step_results", 0, "record", "proposed_candidates")
FIRST_STEP_SEARCH = ("step_results", 0, "record", "search_evidence")
FIRST_INTENT_EVAL_CONFIG = (
    *FIRST_STEP_INTENTS,
    0,
    "resolved_eval_config",
)
#: Intent 1 is the seed round's re-measurement of the initial candidate --
#: the occurrence COPRO did *not* select. Raising its reward is what makes
#: the recorded selection stop being the best-so-far one.
LOSING_INTENT_REWARD = (*FIRST_STEP_INTENTS, 1, "reward_ref")

#: A reward high enough that no honest COPRO run could have selected past
#: it. The exact value is arbitrary; being far outside ``[0, 1]`` makes a
#: failing detail obviously synthetic to whoever reads it.
_IMPLAUSIBLE_REWARD = 9.0


def _load(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RESULT_FILENAME).read_text(encoding="utf-8"))


def _save(run_dir: Path, document: dict[str, Any]) -> Path:
    (run_dir / RESULT_FILENAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    return run_dir


def _fresh_copy(source: Path, destination: Path) -> Path:
    shutil.rmtree(destination, ignore_errors=True)
    return copy_run(source, destination)


def _candidate_wrapper(record: dict[str, Any]) -> dict[str, Any]:
    """Seal a mutated candidate record into the wrapper the format uses."""
    reference = candidate_reference(Candidate.model_validate(record))
    return {
        "record": record,
        "record_ref": reference.record_ref.model_dump(mode="json"),
        "identity_hash": reference.identity_hash,
    }


# --- Control-level negatives ----------------------------------------------


def with_control_field(
    source: Path, destination: Path, **updates: Any
) -> Path:
    """Repoint the run at a COPRO control differing in ``updates``.

    Both invariants that read the configured search -- breadth and depth --
    are violable this way and only this way: a run that measured fewer
    occurrences than it was configured for is indistinguishable, in the
    persisted evidence, from one configured for fewer. Changing the control
    rather than deleting evidence keeps the artifact a valid ``OptimResult``
    that simply no longer did what it says it was set up to do.
    """
    run_dir = _fresh_copy(source, destination)
    document = _load(run_dir)
    ref = TypedRef.model_validate(
        document["run"]["record"]["optimizer_config"]["record_ref"]
    )
    with open_sqlite(str(run_dir / RUNTIME_STORE_FILENAME)) as store:
        control = CoproControl.model_validate(store.get(ref.reference))
    variant = control.model_copy(update=updates)
    if variant == control:
        raise MutationError(
            f"control update {updates!r} left the control unchanged; the "
            f"fixture would not be a negative"
        )
    document["run"]["record"]["optimizer_config"] = {
        "record_ref": put_record(
            run_dir, COPRO_CONTROL_SCHEMA, variant.model_dump(mode="json")
        ),
        "record_hash": variant.identity_hash(),
    }
    reseal_run_binding(document)
    return _save(run_dir, document)


def over_configured_breadth(source: Path, destination: Path) -> Path:
    """A run whose round measured more occurrences than breadth allows.

    Overfilling is the defect, not underfilling. A round that realized
    fewer drafts than it requested is a stochastic outcome the audit
    records and passes -- a proposer call can fail, a draft can be
    rejected -- but a round carrying *more* occurrences than the
    configured breadth measured candidates nobody budgeted for, which no
    infrastructure failure produces.

    The extra occurrence is added to the round rather than subtracted
    from the control, because lowering ``breadth`` would also shrink the
    proposal budget that ``COPRO_DEPTH_STEPS`` and ``COPRO_INTERNAL_ONLY``
    read -- and a fixture that trips three invariants cannot show that
    this one owns the defect.

    The copy gets its own request id: a Step Result refuses duplicate Eval
    Request IDs, so a verbatim duplicate would fail schema validation
    rather than reach the audit as the overfilled round it is meant to be.
    """

    def overfill(intents: list[Any]) -> list[Any]:
        extra = json.loads(json.dumps(intents[-1]))
        request = extra["optim_eval_request"]["eval_request"]
        request["request_id"] = f"{request['request_id']}:overfill"
        return [*intents, extra]

    return mutate_run(source, destination, FIRST_STEP_INTENTS, overfill)


def short_of_configured_depth(source: Path, destination: Path) -> Path:
    """A run that stopped early without declaring a terminal failure."""
    return with_control_field(source, destination, depth=3)


def seed_the_search_never_used(source: Path, destination: Path) -> Path:
    """A run declaring a seed no proposal descends from.

    The persisted format ties each proposal to its *request's* candidate,
    but nothing ties a step request's candidate to the run's declared
    ``initial_candidate_ref``. So a run can report improvement over a prompt
    it never optimized from, and stay perfectly schema-valid -- which is why
    this is the fixture ``COPRO_TERMINAL_PROVENANCE`` needs.
    """
    run_dir = _fresh_copy(source, destination)
    document = _load(run_dir)
    seed = document["run"]["record"]["initial_candidate_ref"]
    record = json.loads(json.dumps(seed["record"]))
    record["candidate_id"] = "declared-but-never-optimized-from"
    wrapper = _candidate_wrapper(record)
    if wrapper["record_ref"] == seed["record_ref"]:
        raise MutationError("the substituted seed hashes to the original")
    document["run"]["record"]["initial_candidate_ref"] = wrapper
    reseal_run_binding(document)
    return _save(run_dir, document)


# --- Step-level negatives -------------------------------------------------


def round_missing_an_occurrence(source: Path, destination: Path) -> Path:
    """A round that measured one fewer occurrence than it should have."""
    return mutate_run(
        source,
        destination,
        FIRST_STEP_INTENTS,
        lambda intents: intents[:1],
    )


def evaluation_off_the_internal_split(source: Path, destination: Path) -> Path:
    """An intent bound to an Eval Config that is not the control's.

    The substitute is derived from the real config with its sampling hash
    changed -- a different split, which is exactly the leak L1 forbids --
    rather than invented, so it stays a valid ``EvalConfigRef``.
    """

    def rebind(wrapper: dict[str, Any]) -> dict[str, Any]:
        record = json.loads(json.dumps(wrapper["record"]))
        record["sampling_config_hash"] = "c" * 64
        record["config_hash"] = "d" * 64
        return {
            "record": record,
            "record_ref": typed_ref_for_record(
                EVAL_CONFIG_RECORD_SCHEMA, record
            ).model_dump(mode="json"),
            "config_hash": record["config_hash"],
        }

    return mutate_run(source, destination, FIRST_INTENT_EVAL_CONFIG, rebind)


def unselected_candidate_scored_higher(
    source: Path, destination: Path
) -> Path:
    """A run whose selection passed over a better-scoring candidate."""

    def raise_reward(wrapper: dict[str, Any]) -> dict[str, Any]:
        record = json.loads(json.dumps(wrapper["record"]))
        record["value"] = _IMPLAUSIBLE_REWARD
        for citation in record.get("input_citations", []):
            citation["value"] = _IMPLAUSIBLE_REWARD
            citation["contributed"] = _IMPLAUSIBLE_REWARD
        return {
            "record": record,
            "record_ref": typed_ref_for_record(
                REWARD_SCHEMA, record
            ).model_dump(mode="json"),
        }

    return mutate_run(source, destination, LOSING_INTENT_REWARD, raise_reward)


def two_proposals_sharing_one_base(source: Path, destination: Path) -> Path:
    """A round whose drafts explored one direction while charging for two.

    The twin is byte-identical to the first proposal, so the two wrappers
    carry the same candidate identity: the round paid for two drafts and
    got one. Appending it rather than replacing an existing proposal keeps
    every resolved intent still citing a proposal it can name.
    """

    def add_twin(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        record = json.loads(json.dumps(proposals[0]["record"]))
        return [*proposals, _candidate_wrapper(record)]

    return mutate_run(source, destination, FIRST_STEP_PROPOSALS, add_twin)


def evaluation_recorded_as_search(source: Path, destination: Path) -> Path:
    """A COPRO step carrying a ``search_evidence`` entry.

    The entry mirrors an evaluation the run genuinely performed, so it is a
    well-formed ``SearchEvidence`` record: the violation is the channel it
    was recorded on, not the record's contents.
    """
    run_dir = _fresh_copy(source, destination)
    document = _load(run_dir)
    step = document["step_results"][0]["record"]
    if step["search_evidence"]:
        raise MutationError(
            "the source run already carries search evidence, so the fixture "
            "would not be a negative"
        )
    intent = step["resolved_intents"][0]
    step["search_evidence"] = [
        {
            "eval_request_id": (
                intent["optim_eval_request"]["eval_request"]["request_id"]
            ),
            "optim_run_id": intent["optim_eval_request"]["optim_run_id"],
            "optim_step_index": 0,
            "candidate": step["proposed_candidates"][0],
            "outcome": "completed",
            "eval_result_ref": intent["eval_result_ref"],
            "reward_ref": intent["reward_ref"],
            "reward_evidence_refs": intent["reward_evidence_refs"],
        }
    ]
    reseal_step_chain(document)
    return _save(run_dir, document)


__all__ = [
    "FIRST_INTENT_EVAL_CONFIG",
    "FIRST_STEP_INTENTS",
    "FIRST_STEP_PROPOSALS",
    "FIRST_STEP_SEARCH",
    "LOSING_INTENT_REWARD",
    "evaluation_off_the_internal_split",
    "evaluation_recorded_as_search",
    "over_configured_breadth",
    "round_missing_an_occurrence",
    "seed_the_search_never_used",
    "short_of_configured_depth",
    "two_proposals_sharing_one_base",
    "unselected_candidate_scored_higher",
    "with_control_field",
]
