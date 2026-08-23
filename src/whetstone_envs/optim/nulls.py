"""The study's two null optimizers, as proposer transports.

Both nulls reach the optimizer through the same ``ProposerTransport`` surface
a real proposer uses, so they run on the shared runner path with the same
budget, the same selection machinery, and the same evidence -- the only
difference is what comes back from ``draft``.

``NullRandomTransport`` (null-A) is the **selection-on-noise control**. It
perturbs the seed template with a seeded RNG and returns the perturbations as
ordinary drafts, so best-on-internal selection runs over candidates that carry
no information. A positive null-A delta whose CI excludes zero means selection
alone "improves" held-out accuracy and every efficacy claim in the study is
void. Because it is a control for *selection*, it must spend the same proposal
budget and fill the same slots as the optimizer it stands in for; a null that
skipped slots would be a control for a different thing.

``NullIdentityTransport`` (null-B) is the **pipeline-overhead control**. It
proposes the seed unchanged, so any measured movement is pipeline noise rather
than optimization.

Layout is the other hard constraint. A perturbation edits *wording*, so
the template's whitespace -- every newline, blank line, and indent -- is
carried through untouched. The real c19 seed is six newlines holding a
grid, an action list, and a question apart; a perturber that rejoined its
tokens on single spaces would hand the control a structurally degraded
prompt that no real arm ever ran, and a null-A delta would then measure
formatting damage rather than selection on noise.

Placeholders are the hard constraint on null-A. ``TemplateRenderContract
.validate_template`` rejects a template that drops a required field, and
``candidate_from_draft`` turns that rejection into a failed proposal, so a
perturber that chewed through ``{grid}`` would not produce a weaker candidate
-- it would produce a *failed* one, and null-A would silently stop being a
control. The perturber therefore treats every placeholder as an atomic,
immovable token and re-validates before returning.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

from whetstone.core.identity import compute_identity_hash, require_full_hash
from whetstone.optim.proposal.proposer import ProposalDraft

if TYPE_CHECKING:
    from whetstone.experiment.candidate import TemplateRenderContract
    from whetstone.optim.proposal.proposer import (
        ProposalRequest,
        ProposerRouteConfig,
    )

__all__ = [
    "NULL_IDENTITY_OPTIMIZER",
    "NULL_PERTURBATION_RATE",
    "NULL_RANDOM_OPTIMIZER",
    "NULL_TRANSPORT_DURABILITY_SCHEMA",
    "NULL_TRANSPORT_DURABILITY_SCHEMA_VERSION",
    "NullIdentityTransport",
    "NullRandomTransport",
    "perturb_template",
]

#: The ``--optimizer`` values these transports back. Wave 1a's CLI binds them;
#: they are named here because the transport is what defines each null.
NULL_RANDOM_OPTIMIZER = "null-random"
NULL_IDENTITY_OPTIMIZER = "null-identity"

#: Durability identity for a null transport. A null makes no provider call, so
#: it has no provider durability of its own; the schema still names the two
#: hashes the optimizer binds a transport by, so a null and a real proposer are
#: never mistaken for one another in recorded identity.
NULL_TRANSPORT_DURABILITY_SCHEMA = "whetstone_envs.optim.null_transport"
NULL_TRANSPORT_DURABILITY_SCHEMA_VERSION = 1

#: Fraction of eligible tokens null-A perturbs per draft (the protocol's 5%).
NULL_PERTURBATION_RATE = 0.05

#: Bounded retries for one perturbation slot before falling back to identity.
#: A retry costs nothing (no provider call), but an unbounded loop on a
#: template whose every draw is rejected would hang the run.
_MAX_PERTURBATION_ATTEMPTS = 8

#: A ``{field}`` placeholder. Null-A splits on this and never perturbs a
#: match, so a placeholder survives every operation intact.
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


#: A run of whitespace. The template is split on this so the *layout* --
#: every newline, blank line, and indent -- is carried through the
#: perturbation untouched rather than rebuilt from single spaces.
_WHITESPACE = re.compile(r"(\s+)")


def _split_layout(template: str) -> tuple[list[str], list[str]]:
    """Separate ``template`` into its words and the whitespace between them.

    Returns the word tokens and the whitespace runs that separate them,
    with ``len(gaps) == len(words) + 1``: a leading gap, one between each
    adjacent pair, and a trailing gap, any of which may be empty.

    Splitting layout away from content is what makes null-A a control for
    *wording*. The real c19 template is six newlines and two blank lines
    holding a grid, an action list, and a question apart, and a perturber
    that rejoined its tokens with single spaces would hand the control a
    structurally degraded prompt -- a different and much worse template
    than any real arm was ever given, which would make a null-A delta
    measure formatting damage rather than selection on noise.
    """
    parts = _WHITESPACE.split(template)
    # ``re.split`` with one capture group alternates text, separator, text,
    # ... and always starts and ends with a (possibly empty) text part.
    words = parts[0::2]
    gaps = parts[1::2]
    return list(words), ["", *gaps]


def _join_layout(words: list[str], gaps: list[str]) -> str:
    """Rebuild a template from its words and its original whitespace runs.

    The gaps are laid back down in order and are never invented: a word
    added or removed by a perturbation shifts which words the existing
    gaps separate, but the multiset of whitespace runs is exactly the one
    :func:`_split_layout` produced.
    """
    pieces = [gaps[0]]
    for index, word in enumerate(words):
        pieces.append(word)
        pieces.append(gaps[index + 1] if index + 1 < len(gaps) else "")
    return "".join(pieces)


def _frozen_words(words: list[str]) -> set[int]:
    """Which word positions carry a placeholder and are therefore atomic.

    Any word containing a placeholder is frozen, so a perturbation can
    neither drop it, duplicate it, nor swap it out of the template.
    """
    return {
        index for index, word in enumerate(words) if _PLACEHOLDER.search(word)
    }


def _perturb_once(rng: random.Random, template: str) -> str:
    """Apply one round of word swaps, deletions, and duplications.

    Every operation is drawn against the *eligible* words only -- those
    carrying no placeholder -- so the render contract's required fields are
    structurally preserved rather than checked for afterwards. The
    whitespace runs are held aside and laid back down unchanged, so the
    perturbation edits wording and never layout.
    """
    words, gaps = _split_layout(template)
    frozen = _frozen_words(words)
    eligible = [index for index in range(len(words)) if index not in frozen]
    if not eligible:
        return template
    count = max(1, round(len(eligible) * NULL_PERTURBATION_RATE))
    result = list(words)
    for _ in range(count):
        # Recompute eligibility each round: deletions and duplications shift
        # indices, and a stale index could otherwise land on a placeholder.
        frozen_now = _frozen_words(result)
        eligible_now = [
            index for index in range(len(result)) if index not in frozen_now
        ]
        if not eligible_now:
            break
        operation = rng.choice(("swap", "delete", "duplicate"))
        target = rng.choice(eligible_now)
        if operation == "swap" and len(eligible_now) > 1:
            other = rng.choice([i for i in eligible_now if i != target])
            result[target], result[other] = result[other], result[target]
        elif operation == "delete" and len(eligible_now) > 1:
            del result[target]
        else:
            result.insert(target, result[target])
    return _join_layout(result, gaps)


def perturb_template(
    template: str,
    *,
    seed: int,
    render_contract: TemplateRenderContract,
    excluded: frozenset[str] = frozenset(),
) -> str:
    """Return a placeholder-preserving perturbation of ``template``.

    Deterministic in ``seed``: the same seed and template always yield the
    same result, so a null-A run replays exactly.

    The result is validated against ``render_contract`` before it is returned.
    A draw the contract rejects is retried with a fresh draw from the same
    seeded stream, bounded by ``_MAX_PERTURBATION_ATTEMPTS``; if every attempt
    is rejected the seed is returned unchanged, which is a recorded no-op
    rather than a candidate the optimizer would reject downstream.

    ``excluded`` names templates an earlier slot in the same batch already
    produced. A short template has few eligible tokens and therefore a small
    perturbation space, so two independently seeded slots can easily draw the
    same result; COPRO requires pairwise-distinct proposals within a round, so
    a collision is retried rather than returned.
    """
    rng = random.Random(seed)  # noqa: S311 - study control, not cryptographic
    for _ in range(_MAX_PERTURBATION_ATTEMPTS):
        candidate = _perturb_once(rng, template)
        if not candidate or candidate == template or candidate in excluded:
            continue
        try:
            render_contract.validate_template(candidate)
        except ValueError:
            continue
        return candidate
    return template


def _null_durability_hash(
    *,
    null_kind: str,
    execution_policy_hash: str,
    prompt_adapter_identity_hash: str,
) -> str:
    return compute_identity_hash(
        schema=NULL_TRANSPORT_DURABILITY_SCHEMA,
        schema_version=NULL_TRANSPORT_DURABILITY_SCHEMA_VERSION,
        payload={
            "null_kind": null_kind,
            "execution_policy_hash": execution_policy_hash,
            "prompt_adapter_identity_hash": prompt_adapter_identity_hash,
        },
    )


class _NullTransport:
    """Shared identity surface for the two nulls.

    A null spends no provider call, so every draft it returns carries
    ``proposer_calls: 0``, no ``logical_call_id``, and no price. Run cost then
    records nothing for the proposer role rather than a zero-dollar phantom
    call, which keeps a null arm's reported spend honest.
    """

    _null_kind: str

    def __init__(
        self,
        *,
        execution_policy_hash: str,
        prompt_adapter_identity_hash: str,
    ) -> None:
        require_full_hash(
            execution_policy_hash,
            field="execution_policy_hash",
        )
        require_full_hash(
            prompt_adapter_identity_hash,
            field="prompt_adapter_identity_hash",
        )
        self._execution_policy_hash = execution_policy_hash
        self._prompt_adapter_identity_hash = prompt_adapter_identity_hash
        self.calls: list[tuple[str, ProposalRequest, int]] = []

    @property
    def execution_policy_hash(self) -> str:
        return self._execution_policy_hash

    @property
    def prompt_adapter_identity_hash(self) -> str:
        return self._prompt_adapter_identity_hash

    @property
    def durability_identity_hash(self) -> str:
        return _null_durability_hash(
            null_kind=self._null_kind,
            execution_policy_hash=self._execution_policy_hash,
            prompt_adapter_identity_hash=(self._prompt_adapter_identity_hash),
        )

    def _record(
        self,
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
    ) -> dict[str, Any]:
        if type(count) is not int or count < 0:
            raise ValueError("proposal draft count must be nonnegative")
        self.calls.append((config.identity_hash(), request, count))
        return {
            "null_kind": self._null_kind,
            "proposal_mode": request.proposal_mode,
            "request_ordinal": request.request_ordinal,
            "proposer_config": config.identity_payload(),
        }


class NullRandomTransport(_NullTransport):
    """null-A: seeded token perturbation of the seed candidate.

    Perturbs whichever candidate the optimizer asks it to mutate, so it tracks
    the optimizer's own search shape rather than always returning to the run
    seed. Each draft's RNG stream is keyed by the run seed, the request, and
    the batch slot, so a re-drive of one request reproduces its drafts exactly
    and two slots in one batch differ from each other.
    """

    _null_kind = "null_random"

    def __init__(
        self,
        *,
        seed: int,
        render_contract: TemplateRenderContract,
        execution_policy_hash: str,
        prompt_adapter_identity_hash: str,
    ) -> None:
        super().__init__(
            execution_policy_hash=execution_policy_hash,
            prompt_adapter_identity_hash=prompt_adapter_identity_hash,
        )
        self._seed = seed
        self._render_contract = render_contract

    def _slot_seed(self, request: ProposalRequest, slot: int) -> int:
        """A stable per-slot seed derived from the run seed and request.

        Derived from the request's identity hash rather than a call counter,
        so a replayed request draws the same stream no matter how many drafts
        preceded it in this process.
        """
        digest = compute_identity_hash(
            schema=NULL_TRANSPORT_DURABILITY_SCHEMA,
            schema_version=NULL_TRANSPORT_DURABILITY_SCHEMA_VERSION,
            payload={
                "seed": self._seed,
                "request": str(request.identity_hash()),
                "slot": slot,
            },
        )
        return int(digest[:16], 16)

    def draft(
        self,
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        evidence_base = self._record(config, request, count)
        base_template = request.base_template
        drafts: list[ProposalDraft] = []
        produced: set[str] = set()
        for slot in range(count):
            perturbed = perturb_template(
                base_template,
                seed=self._slot_seed(request, slot),
                render_contract=self._render_contract,
                excluded=frozenset(produced),
            )
            fell_back = perturbed == base_template
            produced.add(perturbed)
            drafts.append(
                ProposalDraft(
                    template=perturbed,
                    request_evidence={**evidence_base, "draft_index": slot},
                    response_evidence={
                        "draft_index": slot,
                        # Recorded so a run's evidence shows how often the
                        # perturber could not find a contract-valid draw; a
                        # high rate means null-A degenerated toward null-B.
                        "identity_fallback": fell_back,
                    },
                    usage={"proposer_calls": 0},
                    cost=None,
                )
            )
        return tuple(drafts)


class NullIdentityTransport(_NullTransport):
    """null-B: the proposer returns the seed unchanged.

    Every draft is a *failure* carrying the seed's own text as its detail
    rather than a successful draft repeating it. That is not a workaround: a
    successful draft whose template equals its base is rejected by
    ``whetstone.optim.proposal.mutation.diff_check`` ("proposal mutation must
    differ from its base"), because a no-op mutation is not a proposal. The
    honest transport-level statement of "I propose nothing" is therefore an
    unfilled slot, exactly as a real proposer that returned nothing would
    report it.

    What the optimizer does with that is the optimizer's contract, not the
    transport's, and it differs by optimizer -- see this module's note in
    ``CHANGELOG.md``. GEPA and MIPROv2 set ``terminal_proposal_count`` on
    their step contracts and so may terminalize ``seed_retained`` when their
    search accepts nothing over the seed. COPRO does not set it, so under
    COPRO's control shape an unfilled round is a ``copro_proposal_cardinality``
    terminal failure whose result still names the seed as the run's outcome.
    """

    _null_kind = "null_identity"

    def draft(
        self,
        config: ProposerRouteConfig,
        request: ProposalRequest,
        count: int,
    ) -> tuple[ProposalDraft, ...]:
        evidence_base = self._record(config, request, count)
        return tuple(
            ProposalDraft.failure(
                detail=(
                    "null-identity proposer returns the seed unchanged and "
                    "proposes no mutation"
                ),
                request_evidence={
                    **evidence_base,
                    "draft_index": slot,
                    "seed_template": request.base_template,
                },
                response_evidence={
                    "draft_index": slot,
                    "finish": "identity",
                },
                usage={"proposer_calls": 0},
                cost=None,
            )
            for slot in range(count)
        )
