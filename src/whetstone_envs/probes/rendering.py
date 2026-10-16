from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whetstone_envs.instances import Instance


def render_with_prompt_inputs(template: str, instance: Instance) -> str:
    """Format using only ``instance.prompt_inputs``.

    Missing fields raise ``KeyError``.
    """
    return template.format(**dict(instance.prompt_inputs))


@dataclass(frozen=True, slots=True)
class ProbePair:
    """Floor and known-good templates rendered by one callable."""

    naive_template: str
    ceiling_template: str
    render: Callable[[str, Instance], str] = render_with_prompt_inputs

    def render_naive(self, instance: Instance) -> str:
        return self.render(self.naive_template, instance)

    def render_ceiling(self, instance: Instance) -> str:
        return self.render(self.ceiling_template, instance)
