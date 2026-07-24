"""Probe-prompt pairing and shared prediction normalization.

Each candidate exposes exactly two prompts per instance: a *naive*
prompt (the floor) and a *ceiling* prompt (the designer's known-good
prompt). :class:`ProbePair` bundles both templates plus a render
function so every candidate renders prompts the same way.

:func:`normalize` is the single normalization step shared by every
candidate's exact-match scoring: strip surrounding whitespace and all
complete wrapping code-fence pairs. Keeping it here means every oracle
extracts predictions identically, so scoring differences come from the
model, not from per-candidate string handling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone_envs.core.instance import Instance

_MIN_FENCED_LINES = 2


def _strip_code_fence(text: str) -> str:
    """Remove one matching pair of triple-backtick fences, if present.

    A leading fence line (```` ``` ```` optionally followed by a
    language tag) paired with a trailing fence line is removed. Text
    without a matched pair is returned unchanged, so a stray backtick
    inside a real answer is never eaten.
    """
    lines = text.split("\n")
    if len(lines) < _MIN_FENCED_LINES:
        return text
    first = lines[0].strip()
    last = lines[-1].strip()
    if first.startswith("```") and last == "```":
        return "\n".join(lines[1:-1])
    return text


def normalize(prediction: str) -> str:
    """Normalize a raw model prediction for exact-match comparison.

    Strips surrounding whitespace, then removes complete wrapping code
    fences to a fixed point, re-stripping whitespace after each pair.
    Unmatched fences and backticks within the remaining answer are
    preserved. No other content is changed, so exact-match semantics are
    deterministic and normalization is idempotent.
    """
    normalized = prediction.strip()
    while True:
        unfenced = _strip_code_fence(normalized)
        if unfenced == normalized:
            return normalized
        normalized = unfenced.strip()


def render_with_prompt_inputs(template: str, instance: Instance) -> str:
    """Render ``template`` against ``instance.prompt_inputs`` only.

    Formatting is restricted to the instance's public prompt inputs, so
    a template can never interpolate gold/oracle-only state even by
    accident. A template field with no matching input raises
    ``KeyError`` -- a loud template-drift signal rather than a silent
    empty substitution.
    """
    return template.format(**dict(instance.prompt_inputs))


@dataclass(frozen=True, slots=True)
class ProbePair:
    """A naive/ceiling prompt template pair with a shared renderer.

    Parameters
    ----------
    naive_template:
        The floor prompt. A ``str.format``-style template whose fields
        are drawn from an instance's ``prompt_inputs``.
    ceiling_template:
        The designer's known-good prompt, same field convention.
    render:
        A pure function mapping ``(template, instance)`` to a rendered
        prompt string. Defaults to :func:`render_with_prompt_inputs`,
        which formats the template against ``instance.prompt_inputs``
        only -- never against gold/oracle-only fields.
    """

    naive_template: str
    ceiling_template: str
    render: Callable[[str, Instance], str] = render_with_prompt_inputs

    def render_naive(self, instance: Instance) -> str:
        """Render the naive prompt for ``instance``."""
        return self.render(self.naive_template, instance)

    def render_ceiling(self, instance: Instance) -> str:
        """Render the ceiling prompt for ``instance``."""
        return self.render(self.ceiling_template, instance)
