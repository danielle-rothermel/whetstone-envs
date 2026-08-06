from whetstone_envs.probes import ProbePair

NAIVE_TEMPLATE = """{question}

{query}

Answer True or False."""

CEILING_TEMPLATE = """You are a careful deductive reasoner. You are given a set of facts and if-then
rules, followed by a single query statement. Every predicate is fictional and
carries no real-world meaning — rely ONLY on the facts and rules given, never on
outside knowledge or surface plausibility.

Determine whether the query statement is entailed by the facts and rules under
the closed-world assumption: a statement is True if it can be derived by
chaining the given rules from the given facts, and False otherwise. Apply the
rules step by step, following each "every X is a Y" / "X are (not) Z" rule in
order, until you reach the queried property.

Facts and rules:
{question}

Query:
{query}

Reason step by step through the chain of rules, then end your reply with exactly
one word on its own final line: either
True
or
False"""

PROBES = ProbePair(
    naive_template=NAIVE_TEMPLATE,
    ceiling_template=CEILING_TEMPLATE,
)

__all__ = ["PROBES"]
