"""Probe-prompt pairing and rendering against public instance inputs.

Each task environment exposes exactly two prompts per instance: a naive prompt
(the floor) and a ceiling prompt (the designer's known-good prompt).
:class:`ProbePair` bundles both templates plus a render function so every task
environment renders prompts the same way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone_envs.instances import Instance


def render_with_prompt_inputs(template: str, instance: Instance) -> str:
    """Render ``template`` against ``instance.prompt_inputs`` only.

    Formatting is restricted to the instance's public prompt inputs, so a
    template can never interpolate gold/oracle-only state even by accident. A
    template field with no matching input raises ``KeyError`` -- a loud
    template-drift signal rather than a silent empty substitution.
    """
    return template.format(**dict(instance.prompt_inputs))


@dataclass(frozen=True, slots=True)
class ProbePair:
    """A naive/ceiling prompt template pair with a shared renderer.

    Parameters
    ----------
    naive_template:
        The floor prompt. A ``str.format``-style template whose fields are
        drawn from an instance's ``prompt_inputs``.
    ceiling_template:
        The designer's known-good prompt, using the same field convention.
    render:
        A pure function mapping ``(template, instance)`` to a rendered prompt
        string. Defaults to :func:`render_with_prompt_inputs`, which formats
        against ``instance.prompt_inputs`` only -- never gold/oracle fields.
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
