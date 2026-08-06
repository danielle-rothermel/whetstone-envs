from whetstone_envs.probes import ProbePair

NAIVE_TEMPLATE = """Satisfy every constraint below:
{constraints}

Answer:"""

CEILING_TEMPLATE = """Produce one response that satisfies EVERY constraint
below.
The response scores 1 only if all deterministic checks pass; there is no
partial credit.

{constraints}

Output only the answer text. Do not add a preamble or explanation. Before
finalizing, silently verify every required literal, forbidden token, format,
ending, and exact word count across the complete response.

Answer:"""

PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
)

__all__ = ["PROBES"]
