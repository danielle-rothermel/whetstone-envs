from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import UNIQUE as ENUM_UNIQUE
from enum import StrEnum, auto, verify

from whetstone_envs import c18, c22, c23, gsm8k, ifeval
from whetstone_envs.c18.oracle import extract_verdict
from whetstone_envs.c23._scoring import extract_last_output
from whetstone_envs.probes import normalize
from whetstone_envs.scoring import exact_match

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
MARKDOWN_EMPHASIS = re.compile(r"[*_`]")
VERDICT = re.compile(r"\b(true|false)\b", re.IGNORECASE)
CODE_FENCE = re.compile(r"```[A-Za-z]*")
JSON_START = re.compile(r"[\[{]")
C23_ANSWER = re.compile(r"[a-d]+")
C23_RUN = re.compile(r"(?<![A-Za-z])([a-d]+)(?![A-Za-z])")
C23_MARKED = re.compile(r"(?:(?i:output)\s*:|->)[ \t*_`]*([a-d]+)(?![A-Za-z])")
C23_FIRST_LINE_STRIP = re.compile(
    r"^[\s*_`\"'>]*(?:(?i:output|answer|final answer)\s*:)?[\s*_`\"']*"
    r"|[\s*_`\"'.]*$"
)
C18_VERDICTS = ("True", "False")


@verify(ENUM_UNIQUE)
class Family(StrEnum):
    C11 = auto()
    C18 = auto()
    C19 = auto()
    C22 = auto()
    C23 = auto()
    GSM8K = auto()
    IFEVAL = auto()


@verify(ENUM_UNIQUE)
class Reader(StrEnum):
    STRICT = auto()
    FORGIVING = auto()


@verify(ENUM_UNIQUE)
class C19Form(StrEnum):
    COORDINATE = auto()
    HEADING = auto()
    FRONT = auto()
    CARRYING = auto()


@dataclass(frozen=True, slots=True)
class Verdict:
    valid: bool
    correct: bool


@dataclass(frozen=True, slots=True)
class AnswerForm:
    gold: re.Pattern[str]
    found: re.Pattern[str]
    answer: Callable[[re.Match[str]], str]
    options: tuple[str, ...] | None


C19_FORMS: dict[C19Form, AnswerForm] = {
    C19Form.COORDINATE: AnswerForm(
        re.compile(r"\d+,\d+"),
        re.compile(r"(\d+)\s*,\s*(\d+)"),
        lambda match: f"{match[1]},{match[2]}",
        None,
    ),
    C19Form.HEADING: AnswerForm(
        re.compile(r"[NESW]"),
        re.compile(r"\b(?:[NESW]|(?i:north|east|south|west))\b"),
        lambda match: match[0][0].upper(),
        ("N", "S", "E", "W"),
    ),
    C19Form.FRONT: AnswerForm(
        re.compile(r"empty|wall|key|ball|door"),
        re.compile(r"\b(empty|wall|key|ball|door)\b", re.IGNORECASE),
        lambda match: match[1].lower(),
        ("empty", "wall", "key", "ball", "door"),
    ),
    C19Form.CARRYING: AnswerForm(
        re.compile(r"yes|no"),
        re.compile(r"\b(yes|no)\b", re.IGNORECASE),
        lambda match: match[1].lower(),
        ("yes", "no"),
    ),
}


def c19_form(gold: str) -> AnswerForm:
    for form in C19_FORMS.values():
        if form.gold.fullmatch(gold.strip()):
            return form
    msg = f"c19 gold has no known answer form: {gold!r}"
    raise ValueError(msg)


def options(family: Family, gold: str) -> tuple[str, ...] | None:
    if family is Family.C18:
        return C18_VERDICTS
    if family is Family.C19:
        return c19_form(gold).options
    return None


def strict_c18(response: str, gold: str) -> Verdict:
    verdict = extract_verdict(normalize(response))
    return Verdict(
        valid=verdict in C18_VERDICTS,
        correct=bool(c18.score_gold(response, gold)),
    )


def strict_c19(response: str, gold: str) -> Verdict:
    return Verdict(
        valid=c19_form(gold).gold.fullmatch(normalize(response)) is not None,
        correct=bool(exact_match(response, gold)),
    )


def strict_c11(response: str, gold: str) -> Verdict:
    try:
        json.loads(normalize(response))
    except ValueError:
        valid = False
    else:
        valid = True
    return Verdict(valid=valid, correct=bool(exact_match(response, gold)))


def strict_c22(response: str, gold: str) -> Verdict:
    return Verdict(
        valid=bool(normalize(response)),
        correct=bool(c22.score(gold, response)),
    )


def strict_c23(response: str, gold: str) -> Verdict:
    answer = normalize(extract_last_output(response))
    return Verdict(
        valid=C23_ANSWER.fullmatch(answer) is not None,
        correct=bool(c23.score_gold(response, gold)),
    )


def forgiving_c18(response: str, gold: str) -> Verdict:
    plain = MARKDOWN_EMPHASIS.sub("", THINK_BLOCK.sub("", response))
    verdicts = VERDICT.findall(plain)
    if not verdicts:
        return Verdict(valid=False, correct=False)
    return Verdict(valid=True, correct=verdicts[-1].capitalize() == gold)


def forgiving_c19(response: str, gold: str) -> Verdict:
    form = c19_form(gold)
    matches = list(form.found.finditer(THINK_BLOCK.sub("", response)))
    if not matches:
        return Verdict(valid=False, correct=False)
    return Verdict(
        valid=True, correct=form.answer(matches[-1]) == gold.strip()
    )


def first_json(response: str) -> tuple[bool, object]:
    plain = CODE_FENCE.sub("", THINK_BLOCK.sub("", response))
    decoder = json.JSONDecoder(parse_int=float)
    for match in JSON_START.finditer(plain):
        try:
            value, _ = decoder.raw_decode(plain, match.start())
        except json.JSONDecodeError:
            continue
        return True, value
    return False, None


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def forgiving_c11(response: str, gold: str) -> Verdict:
    found, value = first_json(response)
    if not found:
        return Verdict(valid=False, correct=False)
    expected = json.loads(gold, parse_int=float)
    return Verdict(
        valid=True, correct=canonical_json(value) == canonical_json(expected)
    )


def c23_last_run(response: str) -> str | None:
    plain = THINK_BLOCK.sub("", response)
    runs = C23_MARKED.findall(plain) or C23_RUN.findall(plain)
    return runs[-1] if runs else None


def c23_first_line(response: str) -> str | None:
    lines = [
        line
        for line in THINK_BLOCK.sub("", response).splitlines()
        if line.strip()
    ]
    if not lines:
        return None
    line = lines[0]
    if "->" in line:
        line = line.rsplit("->", 1)[1]
    answer = C23_FIRST_LINE_STRIP.sub("", line)
    return answer if C23_ANSWER.fullmatch(answer) else None


def forgiving_c23(response: str, gold: str) -> Verdict:
    first = c23_first_line(response)
    last = c23_last_run(response)
    return Verdict(
        valid=first is not None or last is not None,
        correct=gold in (first, last),
    )


def gsm8k_verdict(answer: str | None, gold: str) -> Verdict:
    return Verdict(valid=answer is not None, correct=answer == gold)


def strict_gsm8k(response: str, gold: str) -> Verdict:
    return gsm8k_verdict(gsm8k.strict_answer(response), gold)


def forgiving_gsm8k(response: str, gold: str) -> Verdict:
    return gsm8k_verdict(gsm8k.forgiving_answer(response), gold)


def strict_ifeval(response: str, gold: str) -> Verdict:
    return Verdict(
        valid=bool(normalize(response)),
        correct=ifeval.check_strict(response, gold).all_followed,
    )


def forgiving_ifeval(response: str, gold: str) -> Verdict:
    return Verdict(
        valid=bool(normalize(THINK_BLOCK.sub("", response))),
        correct=ifeval.check_loose(response, gold).all_followed,
    )


STRICT: dict[Family, Callable[[str, str], Verdict]] = {
    Family.C11: strict_c11,
    Family.C18: strict_c18,
    Family.C19: strict_c19,
    Family.C22: strict_c22,
    Family.C23: strict_c23,
    Family.GSM8K: strict_gsm8k,
    Family.IFEVAL: strict_ifeval,
}

FORGIVING: dict[Family, Callable[[str, str], Verdict]] = {
    Family.C11: forgiving_c11,
    Family.C18: forgiving_c18,
    Family.C19: forgiving_c19,
    Family.C22: strict_c22,
    Family.C23: forgiving_c23,
    Family.GSM8K: forgiving_gsm8k,
    Family.IFEVAL: forgiving_ifeval,
}

READERS: dict[Reader, dict[Family, Callable[[str, str], Verdict]]] = {
    Reader.STRICT: STRICT,
    Reader.FORGIVING: FORGIVING,
}


def strict(family: Family, response: str, gold: str) -> Verdict:
    return STRICT[family](response, gold)


def forgiving(family: Family, response: str, gold: str) -> Verdict:
    return FORGIVING[family](response, gold)


def read(reader: Reader, family: Family, response: str, gold: str) -> Verdict:
    return READERS[reader][family](response, gold)
