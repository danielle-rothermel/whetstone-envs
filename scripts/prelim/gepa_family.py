from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import UNIQUE as ENUM_UNIQUE
from enum import StrEnum, auto, verify
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from gepa import EvaluationBatch, optimize
from gepa.utils.stop_condition import TimeoutStopCondition
from readers import Family, Reader, options, read

from whetstone_envs import c11, c18, c19, c22, c23, gsm8k, ifeval
from whetstone_envs.c18.oracle import extract_verdict
from whetstone_envs.c22.constraints import ConstraintStack
from whetstone_envs.c22.generation import Preset as C22Preset
from whetstone_envs.c22.generation import _generate_pool as c22_generate
from whetstone_envs.c22.oracle import evaluate as c22_evaluate
from whetstone_envs.c23._scoring import extract_last_output
from whetstone_envs.instances import Instance
from whetstone_envs.probes import normalize

if TYPE_CHECKING:
    from collections.abc import Mapping

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
FORMAT_SENTENCES = Path(__file__).with_name("format_sentences.toml")
READERS_FILE = Path(__file__).with_name("readers.py")
SEED_INSTRUCTION = "Solve the task below."
SPLIT_SEED = 20261016
HTTP_TIMEOUT_S = 180.0
HTTP_RETRIES = 5
HTTP_RETRY_STATUS = 429
HTTP_SERVER_ERROR = 500
GOLD_FEEDBACK_CHARS = 800
REFLECT_INPUT_CHARS = 3000
REFLECT_OUTPUT_CHARS = 1500
STRICT_READING_CHARS = 200
TAIL_MARKER = "[... response head omitted ...]\n"
TOP_LOGPROBS = 5
PLAIN_ANSWER_PREFIX = "\nAnswer:"
PROBE_CHARS = 4000
CHAT_PATH = "/chat/completions"
COMPLETIONS_PATH = "/completions"
TOKENIZE_PATH = "/tokenize"
OPENAI_V1_SUFFIX = "/v1"

NULL_PROMPT = """\
An instruction is placed before every prompt of a task. The task prompts \
follow this template, where fields in braces are filled per item:
---
{template}
---
Current instruction:
---
{seed}
---
Instructions already written:
{previous}

Write one new paraphrase of the current instruction that keeps its meaning \
and differs from the instructions already written. You receive no examples, \
model outputs, answers, or scores. Reply with only the new instruction text."""

APE_PROPOSE_PROMPT = """\
I gave a friend an instruction and {count} inputs. The friend read the \
instruction and wrote an output for every one of the inputs. Here are the \
input-output pairs:

{demos}

Write the instruction I gave. Reply with only the instruction text."""

APE_RESAMPLE_PROMPT = """\
Generate a variation of the following instruction while keeping the semantic \
meaning.

Input: {instruction}

Reply with only the new instruction text."""

OPRO_PROMPT = """\
Your task is to generate the instruction <INS>. The instruction is placed \
before every problem. Below are some previous instructions with their \
scores. The score ranges from 0 to 100.

{history}

Below are some problems.

{exemplars}

Generate an instruction that is different from all the instructions <INS> \
above, and has a higher score than all the instructions <INS> above. The \
instruction should begin with <INS> and end with </INS>. The instruction \
should be concise, effective, and generally applicable to all problems \
above."""

PROPOSAL_ATTEMPTS = 5
APE_INITIAL = 8
APE_SURVIVORS = 3
APE_RESAMPLE = 2
APE_FINAL = 2
APE_DEMOS = 5
OPRO_HISTORY = 20
OPRO_EXEMPLARS = 3
OPRO_OPEN = "<INS>"
OPRO_CLOSE = "</INS>"
FENCE = "```"
SHUFFLE_ATTEMPTS = 20


@verify(ENUM_UNIQUE)
class Arm(StrEnum):
    NAIVE = auto()
    FORMAT_FIXED = auto()
    NULL = auto()
    BESTOFN = auto()
    APE = auto()
    OPRO = auto()
    GEPA = auto()
    GEPA_NOFEEDBACK = "gepa-nofeedback"
    GEPA_SHUFFLED = "gepa-shuffled"
    EVAL = auto()


GEPA_ARMS = frozenset({Arm.GEPA, Arm.GEPA_NOFEEDBACK, Arm.GEPA_SHUFFLED})
PARAPHRASE_ARMS = frozenset({Arm.NULL, Arm.BESTOFN})


@verify(ENUM_UNIQUE)
class Feedback(StrEnum):
    GOLD = auto()
    RICH = auto()


@verify(ENUM_UNIQUE)
class Status(StrEnum):
    RUNNING = auto()
    OUT_OF_BAND = auto()
    DONE = auto()
    FAILED = auto()


@verify(ENUM_UNIQUE)
class Scoring(StrEnum):
    GENERATE = auto()
    LIKELIHOOD = auto()


@verify(ENUM_UNIQUE)
class Logprobs(StrEnum):
    ON = auto()
    OFF = auto()
    AUTO = auto()


@verify(ENUM_UNIQUE)
class Thinking(StrEnum):
    ON = auto()
    OFF = auto()


@verify(ENUM_UNIQUE)
class PromptFormat(StrEnum):
    CHAT = auto()
    PLAIN = auto()


@verify(ENUM_UNIQUE)
class Serving(StrEnum):
    OPENROUTER = auto()
    VLLM = auto()


@dataclass(frozen=True, slots=True)
class FamilySpec:
    naive_template: str
    pool: Callable[[int | None], Sequence[Instance]]
    feedback: Callable[[str, str], str]


def gold_feedback(_response: str, gold: str) -> str:
    return f"Incorrect. Expected answer: {gold[:GOLD_FEEDBACK_CHARS]}"


def c22_feedback(response: str, gold: str) -> str:
    stack = ConstraintStack.from_gold(gold)
    verdicts = c22_evaluate(stack, response).per_constraint
    failed = ", ".join(name for name, passed in verdicts if not passed)
    return f"Incorrect. Failed constraints: {failed}"


def ifeval_feedback(response: str, gold: str) -> str:
    failed = ", ".join(ifeval.check_strict(response, gold).failed)
    return f"Incorrect. Failed instructions: {failed}"


def dataset_pool(
    generate: Callable[[], Any], n_per_category: int | None
) -> Sequence[Instance]:
    if n_per_category is not None:
        msg = "--n-per-category does not apply to dataset families"
        raise SystemExit(msg)
    return generate().instance_group.instances


def pool_of(
    generate: Callable[..., Any], n_per_category: int | None
) -> Sequence[Instance]:
    pool = (
        generate()
        if n_per_category is None
        else generate(n_per_category=n_per_category)
    )
    return pool.instance_group.instances


def c22_pool(n_per_category: int | None) -> Sequence[Instance]:
    if n_per_category is None:
        return c22.generate_pool().instance_group.instances
    pool = c22_generate(
        C22Preset.DEFAULT, instances_per_category=n_per_category
    )
    return pool.instance_group.instances


FAMILIES: dict[Family, FamilySpec] = {
    Family.C11: FamilySpec(
        c11.PROBES.naive_template,
        lambda n: pool_of(c11.generate_pool, n),
        gold_feedback,
    ),
    Family.C18: FamilySpec(
        c18.PROBES.naive_template,
        lambda n: pool_of(c18.generate_pool, n),
        gold_feedback,
    ),
    Family.C19: FamilySpec(
        c19.PROBES.naive_template,
        lambda n: pool_of(c19.generate_pool, n),
        gold_feedback,
    ),
    Family.C22: FamilySpec(
        c22.PROBES.naive_template,
        c22_pool,
        c22_feedback,
    ),
    Family.C23: FamilySpec(
        c23.PROBES.naive_template,
        lambda n: pool_of(c23.generate_pool, n),
        gold_feedback,
    ),
    Family.GSM8K: FamilySpec(
        gsm8k.NAIVE_TEMPLATE,
        lambda n: dataset_pool(gsm8k.generate_pool, n),
        gold_feedback,
    ),
    Family.IFEVAL: FamilySpec(
        ifeval.NAIVE_TEMPLATE,
        lambda n: dataset_pool(ifeval.generate_pool, n),
        ifeval_feedback,
    ),
}


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TokenLogprobs:
    tokens: list[str]
    token_logprobs: list[float | None]
    top: list[list[list[Any]]]


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    cost: float | None
    provider: str | None
    finish_reason: str | None
    logprobs: TokenLogprobs | None = None


@dataclass(frozen=True, slots=True)
class OptionScores:
    choice: str
    option_logprobs: dict[str, float]
    option_tokens: dict[str, int]


@dataclass(frozen=True, slots=True)
class Row:
    instance_id: str
    split: str
    prompt_inputs: dict[str, str]
    prompt: str
    response: str
    gold: str
    score: float
    strict_valid: bool
    strict_correct: bool
    forgiving_valid: bool
    forgiving_correct: bool
    option_logprobs: dict[str, float] | None
    error: str | None
    cost: float | None
    provider: str | None
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class Evaluated:
    row: Row
    logprobs: TokenLogprobs | None
    option_tokens: dict[str, int] | None


@dataclass(frozen=True, slots=True)
class Splits:
    train: list[Instance]
    val: list[Instance]
    heldout: list[Instance]


def render(
    template: str, instruction: str | None, inputs: Mapping[str, str]
) -> str:
    body = template
    for name, value in inputs.items():
        body = body.replace("{" + name + "}", value)
    return body if instruction is None else f"{instruction}\n\n{body}"


def mean(rows: Sequence[Row]) -> float:
    return sum(row.score for row in rows) / len(rows)


def sha256_of(path: str | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def git_sha(path: Path) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603
            ["git", "log", "-1", "--format=%H", "--", path.name],  # noqa: S607
            cwd=path.parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def make_splits(spec: FamilySpec, args: argparse.Namespace) -> Splits:
    instances = list(spec.pool(args.n_per_category))
    random.Random(args.split_seed).shuffle(instances)  # noqa: S311
    needed = args.train_n + args.val_n + args.heldout_n
    if needed > len(instances):
        msg = f"{args.family} pool has {len(instances)} items; need {needed}"
        raise SystemExit(msg)
    train_end = args.train_n
    val_end = train_end + args.val_n
    return Splits(
        train=instances[:train_end],
        val=instances[train_end:val_end],
        heldout=instances[val_end:needed],
    )


def likelihood_splits(
    family: Family, splits: Splits
) -> tuple[Splits, dict[str, int]]:
    def keep(part: list[Instance]) -> list[Instance]:
        return [i for i in part if options(family, i.gold) is not None]

    kept = Splits(
        train=keep(splits.train),
        val=keep(splits.val),
        heldout=keep(splits.heldout),
    )
    dropped = {
        "train": len(splits.train) - len(kept.train),
        "val": len(splits.val) - len(kept.val),
        "heldout": len(splits.heldout) - len(kept.heldout),
    }
    return kept, dropped


def with_shots(
    instruction: str, template: str, examples: Sequence[Instance]
) -> str:
    blocks = [instruction]
    for number, inst in enumerate(examples, start=1):
        body = render(template, None, inst.prompt_inputs)
        blocks.append(f"Example {number}:\n{body}\n{inst.gold}")
    return "\n\n".join(blocks)


def chat_logprobs(choice: Mapping[str, Any]) -> TokenLogprobs | None:
    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        return None
    return TokenLogprobs(
        tokens=[entry.get("token") for entry in content],
        token_logprobs=[entry.get("logprob") for entry in content],
        top=[
            [[t.get("token"), t.get("logprob")] for t in entry["top_logprobs"]]
            for entry in content
        ],
    )


def completion_logprobs(choice: Mapping[str, Any]) -> TokenLogprobs | None:
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens")
    if not tokens:
        return None
    return TokenLogprobs(
        tokens=tokens,
        token_logprobs=logprobs.get("token_logprobs") or [],
        top=[
            [[token, value] for token, value in (top or {}).items()]
            for top in logprobs.get("top_logprobs") or []
        ],
    )


def reader_summary(rows: Sequence[Row]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        Reader.STRICT: {
            "correct": sum(r.strict_correct for r in rows) / n,
            "valid": sum(r.strict_valid for r in rows) / n,
        },
        Reader.FORGIVING: {
            "correct": sum(r.forgiving_correct for r in rows) / n,
            "valid": sum(r.forgiving_valid for r in rows) / n,
        },
    }


class Run:
    def __init__(self, args: argparse.Namespace, out: Path | None) -> None:
        self.args = args
        self.out = out
        self.family = Family(args.family)
        self.spec = FAMILIES[self.family]
        self.objective = Reader(args.objective)
        self.feedback = Feedback(args.feedback)
        self.scoring = Scoring(args.scoring)
        self.prompt_format = PromptFormat(args.prompt_format)
        self.thinking = (
            None if args.thinking is None else Thinking(args.thinking)
        )
        self.task_base_url = args.base_url.rstrip("/")
        self.task_openrouter = self.task_base_url == OPENROUTER_BASE_URL
        self.serving = (
            Serving.OPENROUTER if self.task_openrouter else Serving.VLLM
        )
        mode = Logprobs(args.logprobs)
        self.logprobs = mode is Logprobs.ON or (
            mode is Logprobs.AUTO and not self.task_openrouter
        )
        self.task_key = os.environ[args.api_key_env]
        self.system_prompt = (
            None
            if args.system_prompt_file is None
            else Path(args.system_prompt_file).expanduser().read_text("utf-8")
        )
        splits = make_splits(self.spec, args)
        self.likelihood_dropped: dict[str, int] | None = None
        if self.scoring is Scoring.LIKELIHOOD:
            splits, self.likelihood_dropped = likelihood_splits(
                self.family, splits
            )
        self.splits = splits
        base = (
            None
            if args.seed_file is None
            else Path(args.seed_file).expanduser().read_text("utf-8").strip()
        )
        if args.shots:
            if args.shots > len(splits.train):
                msg = f"--shots {args.shots} exceeds the train split"
                raise SystemExit(msg)
            base = with_shots(
                base or SEED_INSTRUCTION,
                self.spec.naive_template,
                splits.train[: args.shots],
            )
        self.naive_instruction = base
        self.seed_instruction = base or SEED_INSTRUCTION
        self.split_of = {
            inst.id: name
            for name, part in (
                ("train", self.splits.train),
                ("val", self.splits.val),
                ("heldout", self.splits.heldout),
            )
            for inst in part
        }
        self.items = (
            None
            if out is None
            else (out / "items.jsonl").open("a", encoding="utf-8")
        )
        self.logprob_file = (
            None
            if out is None
            else (out / "logprobs.jsonl").open("a", encoding="utf-8")
        )
        self.candidate_ids: dict[str | None, int] = {}
        self.candidate_rows: list[dict[str, Any]] = []
        self.round = 0
        self.started = time.monotonic()
        self.task_calls = 0
        self.task_cost = 0.0
        self.reflection_calls = 0
        self.reflection_cost = 0.0
        self.uncosted_calls = 0
        self.heldout_rows: dict[str, list[Row]] = {}
        self.results: dict[str, Any] = {
            "family": args.family,
            "arm": args.arm,
            "task_model": args.task_model,
            "base_url": self.task_base_url,
            "serving": self.serving,
            "reasoning": args.reasoning,
            "reflection_model": args.reflection_model,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "n_per_category": args.n_per_category,
            "data_source": "generator",
            "sizes": {
                "train": args.train_n,
                "val": args.val_n,
                "heldout": args.heldout_n,
            },
            "likelihood_dropped": self.likelihood_dropped,
            "max_metric_calls": args.max_metric_calls,
            "minibatch": args.minibatch,
            "wall_cap_s": args.wall_cap_s,
            "band": [args.band_low, args.band_high],
            "objective": self.objective,
            "feedback": self.feedback,
            "scoring": self.scoring,
            "prompt_format": self.prompt_format,
            "logprobs": self.logprobs,
            "readers_git_sha": git_sha(READERS_FILE),
            "readers_sha256": sha256_of(str(READERS_FILE)),
            "thinking": args.thinking,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "shots": args.shots,
            "system_prompt_sha256": sha256_of(args.system_prompt_file),
            "seed_file_sha256": sha256_of(args.seed_file),
            "seed_instruction": self.seed_instruction,
            "code_tag": args.code_tag,
            "status": Status.RUNNING,
        }

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        if self.out is not None:
            with (self.out / "progress.log").open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def write_results(self) -> None:
        self.results.update(
            task_calls=self.task_calls,
            reflection_calls=self.reflection_calls,
            task_cost_usd=round(self.task_cost, 6),
            reflection_cost_usd=round(self.reflection_cost, 6),
            uncosted_calls=self.uncosted_calls,
            wall_s=round(time.monotonic() - self.started, 1),
            heldout_readers={
                phase: reader_summary(rows)
                for phase, rows in self.heldout_rows.items()
            },
        )
        assert self.out is not None
        (self.out / "results.json").write_text(
            json.dumps(self.results, indent=1) + "\n", encoding="utf-8"
        )
        path = self.out / "candidates.jsonl"
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(
            "".join(json.dumps(row) + "\n" for row in self.candidate_rows),
            encoding="utf-8",
        )
        tmp.replace(path)

    def total_cost(self) -> float:
        return round(self.task_cost + self.reflection_cost, 6)

    def candidate(self, instruction: str | None, source: str) -> int:
        if instruction not in self.candidate_ids:
            index = len(self.candidate_ids)
            self.candidate_ids[instruction] = index
            record = {
                "candidate": index,
                "source": source,
                "instruction": instruction,
                "arm": self.args.arm,
                "round": self.round,
                "scores": [],
            }
            self.candidate_rows.append(record)
            if self.out is not None:
                with (self.out / "candidates.jsonl").open(
                    "a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(record) + "\n")
        return self.candidate_ids[instruction]

    def body(
        self,
        model: str,
        prompt: str | list[dict[str, Any]],
        effort: str | None,
        *,
        openrouter: bool,
    ) -> dict:
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        body: dict[str, Any] = {"model": model, "messages": messages}
        if openrouter:
            body["usage"] = {"include": True}
            if effort:
                body["reasoning"] = {"effort": effort}
        elif effort:
            body["reasoning_effort"] = effort
        return body

    def messages(self, prompt: str) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def plain_text(self, prompt: str) -> str:
        if self.system_prompt is None:
            return prompt
        return f"{self.system_prompt}\n\n{prompt}"

    def template_kwargs(self) -> dict[str, Any]:
        if self.thinking is None or self.task_openrouter:
            return {}
        enabled = self.thinking is Thinking.ON
        return {"chat_template_kwargs": {"enable_thinking": enabled}}

    def task_request(self, prompt: str) -> tuple[str, dict[str, Any]]:
        args = self.args
        if self.prompt_format is PromptFormat.PLAIN:
            body: dict[str, Any] = {
                "model": args.task_model,
                "prompt": self.plain_text(prompt),
            }
            if self.logprobs:
                body["logprobs"] = TOP_LOGPROBS
            if args.stop:
                body["stop"] = args.stop
            path = COMPLETIONS_PATH
        else:
            body = self.body(
                args.task_model,
                self.messages(prompt),
                args.reasoning,
                openrouter=self.task_openrouter,
            )
            if self.logprobs:
                body["logprobs"] = True
                body["top_logprobs"] = TOP_LOGPROBS
            if self.thinking is not None and self.task_openrouter:
                reasoning = body.setdefault("reasoning", {})
                reasoning["enabled"] = self.thinking is Thinking.ON
            body.update(self.template_kwargs())
            path = CHAT_PATH
        body["temperature"] = args.temperature
        if args.max_tokens is not None:
            body["max_tokens"] = args.max_tokens
        if self.task_openrouter:
            body["usage"] = {"include": True}
        return path, body

    async def request(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: dict,
        key: str,
    ) -> dict[str, Any]:
        last = ""
        for attempt in range(HTTP_RETRIES):
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                )
            except httpx.HTTPError as exc:
                last = repr(exc)
            else:
                status = response.status_code
                if status == HTTP_RETRY_STATUS or status >= HTTP_SERVER_ERROR:
                    last = f"http {status}"
                elif status >= httpx.codes.BAD_REQUEST:
                    msg = f"http {status}: {response.text[:300]}"
                    raise ProviderError(msg)
                else:
                    return response.json()
            await asyncio.sleep(2**attempt)
        raise ProviderError(last)

    async def post(
        self,
        client: httpx.AsyncClient,
        body: dict,
        *,
        base_url: str,
        key: str,
        path: str = CHAT_PATH,
    ) -> Reply:
        last = ""
        for _ in range(HTTP_RETRIES):
            payload = await self.request(client, base_url + path, body, key)
            choices = payload.get("choices")
            if not choices:
                last = f"no choices: {str(payload)[:300]}"
                continue
            usage = payload.get("usage") or {}
            choice = choices[0]
            if path == COMPLETIONS_PATH:
                text = choice.get("text") or ""
                logprobs = completion_logprobs(choice)
            else:
                text = (choice.get("message") or {}).get("content") or ""
                logprobs = chat_logprobs(choice)
            return Reply(
                text=text,
                cost=usage.get("cost"),
                provider=payload.get("provider"),
                finish_reason=choice.get("finish_reason"),
                logprobs=logprobs,
            )
        raise ProviderError(last)

    def server_root(self) -> str:
        return self.task_base_url.removesuffix(OPENAI_V1_SUFFIX)

    async def tokenize(
        self, client: httpx.AsyncClient, body: dict[str, Any]
    ) -> list[int]:
        payload = await self.request(
            client,
            self.server_root() + TOKENIZE_PATH,
            {"model": self.args.task_model, **body},
            self.task_key,
        )
        return payload["tokens"]

    async def prefix_ids(
        self, client: httpx.AsyncClient, prompt: str
    ) -> list[int]:
        if self.prompt_format is PromptFormat.PLAIN:
            return await self.tokenize(
                client,
                {"prompt": self.plain_text(prompt) + PLAIN_ANSWER_PREFIX},
            )
        return await self.tokenize(
            client,
            {
                "messages": self.messages(prompt),
                "add_generation_prompt": True,
                **self.template_kwargs(),
            },
        )

    async def answer_ids(
        self, client: httpx.AsyncClient, option: str
    ) -> list[int]:
        text = (
            f" {option}"
            if self.prompt_format is PromptFormat.PLAIN
            else option
        )
        return await self.tokenize(
            client, {"prompt": text, "add_special_tokens": False}
        )

    async def option_logprob(
        self, client: httpx.AsyncClient, prefix: list[int], answer: list[int]
    ) -> tuple[float, dict[str, Any]]:
        payload = await self.request(
            client,
            self.task_base_url + COMPLETIONS_PATH,
            {
                "model": self.args.task_model,
                "prompt": prefix + answer,
                "max_tokens": 1,
                "temperature": 0,
                "prompt_logprobs": 0,
            },
            self.task_key,
        )
        positions = payload["choices"][0]["prompt_logprobs"]
        total = 0.0
        for index, token in enumerate(answer, start=len(prefix)):
            total += positions[index][str(token)]["logprob"]
        return total, payload

    async def score_options(
        self, client: httpx.AsyncClient, prompt: str, gold: str
    ) -> OptionScores:
        choices = options(self.family, gold)
        assert choices is not None
        prefix = await self.prefix_ids(client, prompt)
        option_logprobs: dict[str, float] = {}
        option_tokens: dict[str, int] = {}
        for option in choices:
            answer = await self.answer_ids(client, option)
            option_logprobs[option], _ = await self.option_logprob(
                client, prefix, answer
            )
            option_tokens[option] = len(answer)
        return OptionScores(
            choice=max(choices, key=lambda option: option_logprobs[option]),
            option_logprobs=option_logprobs,
            option_tokens=option_tokens,
        )

    def add_cost(self, cost: float | None, *, reflection: bool) -> None:
        if cost is None:
            self.uncosted_calls += 1
        elif reflection:
            self.reflection_cost += cost
        else:
            self.task_cost += cost

    def make_row(
        self,
        inst: Instance,
        prompt: str,
        reply: Reply,
        error: str | None,
        option_logprobs: dict[str, float] | None,
    ) -> Row:
        text = reply.text
        strict = read(Reader.STRICT, self.family, text, inst.gold)
        forgiving = read(Reader.FORGIVING, self.family, text, inst.gold)
        objective = strict if self.objective is Reader.STRICT else forgiving
        return Row(
            instance_id=inst.id,
            split=self.split_of[inst.id],
            prompt_inputs=dict(inst.prompt_inputs),
            prompt=prompt,
            response=text,
            gold=inst.gold,
            score=float(objective.correct),
            strict_valid=strict.valid,
            strict_correct=strict.correct,
            forgiving_valid=forgiving.valid,
            forgiving_correct=forgiving.correct,
            option_logprobs=option_logprobs,
            error=error,
            cost=reply.cost,
            provider=reply.provider,
            finish_reason=reply.finish_reason,
        )

    async def one(
        self,
        client: httpx.AsyncClient,
        gate: asyncio.Semaphore,
        instruction: str | None,
        inst: Instance,
    ) -> Evaluated:
        prompt = render(
            self.spec.naive_template, instruction, inst.prompt_inputs
        )
        error = None
        reply = Reply("", None, None, None)
        scores = None
        async with gate:
            try:
                if self.scoring is Scoring.LIKELIHOOD:
                    scores = await self.score_options(
                        client, prompt, inst.gold
                    )
                    reply = Reply(scores.choice, None, None, "likelihood")
                else:
                    path, body = self.task_request(prompt)
                    reply = await self.post(
                        client,
                        body,
                        base_url=self.task_base_url,
                        key=self.task_key,
                        path=path,
                    )
            except (ProviderError, KeyError, IndexError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        return Evaluated(
            row=self.make_row(
                inst,
                prompt,
                reply,
                error,
                None if scores is None else scores.option_logprobs,
            ),
            logprobs=reply.logprobs,
            option_tokens=None if scores is None else scores.option_tokens,
        )

    async def run_batch(
        self, instruction: str | None, batch: Sequence[Instance]
    ) -> list[Evaluated]:
        limits = httpx.Limits(max_connections=self.args.concurrency)
        gate = asyncio.Semaphore(self.args.concurrency)
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_S, limits=limits
        ) as client:
            return list(
                await asyncio.gather(
                    *(self.one(client, gate, instruction, i) for i in batch)
                )
            )

    def write_logprobs(
        self, phase: str, index: int, evaluated: Evaluated
    ) -> None:
        if self.logprob_file is None:
            return
        row = evaluated.row
        record: dict[str, Any] = {
            "phase": phase,
            "candidate": index,
            "instance_id": row.instance_id,
            "finish_reason": row.finish_reason,
        }
        if evaluated.option_tokens is not None:
            record.update(
                option_logprobs=row.option_logprobs,
                option_tokens=evaluated.option_tokens,
            )
        elif evaluated.logprobs is not None:
            record.update(
                tokens=evaluated.logprobs.tokens,
                token_logprobs=evaluated.logprobs.token_logprobs,
                top5=evaluated.logprobs.top,
            )
        else:
            return
        self.logprob_file.write(json.dumps(record) + "\n")

    def evaluate(
        self,
        instruction: str | None,
        batch: Sequence[Instance],
        phase: str,
        source: str,
    ) -> list[Row]:
        index = self.candidate(instruction, source)
        evaluated = asyncio.run(self.run_batch(instruction, batch))
        rows = [e.row for e in evaluated]
        for item in evaluated:
            self.task_calls += 1
            self.add_cost(item.row.cost, reflection=False)
            record = {"phase": phase, "candidate": index, **asdict(item.row)}
            if self.items is not None:
                self.items.write(json.dumps(record) + "\n")
            self.write_logprobs(phase, index, item)
        if self.items is not None:
            self.items.flush()
        if self.logprob_file is not None:
            self.logprob_file.flush()
        if phase in {"naive_heldout", "best_heldout"}:
            self.heldout_rows[phase] = rows
        self.candidate_rows[index]["scores"].append(
            {
                "phase": phase,
                "round": self.round,
                "n": len(rows),
                "mean": mean(rows),
            }
        )
        errors = sum(row.error is not None for row in rows)
        self.log(
            f"{phase} candidate={index} n={len(rows)} mean={mean(rows):.3f}"
            f" errors={errors} task_calls={self.task_calls}"
        )
        return rows

    def reflect(self, prompt: str | list[dict[str, Any]]) -> str:
        body = self.body(
            self.args.reflection_model, prompt, None, openrouter=True
        )
        key = os.environ[OPENROUTER_KEY_ENV]

        async def call() -> Reply:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                return await self.post(
                    client, body, base_url=OPENROUTER_BASE_URL, key=key
                )

        reply = asyncio.run(call())
        self.reflection_calls += 1
        self.add_cost(reply.cost, reflection=True)
        return reply.text

    def wall_exceeded(self, started: float) -> bool:
        return time.monotonic() - started > self.args.wall_cap_s


def strict_reading(family: Family, response: str) -> str | None:
    if family is Family.C18:
        return extract_verdict(normalize(response))
    if family in (Family.C11, Family.C19):
        return normalize(response)
    if family is Family.C23:
        return normalize(extract_last_output(response))
    if family is Family.GSM8K:
        return gsm8k.strict_answer(response)
    return None


class GepaAdapter:
    propose_new_texts = None

    def __init__(self, run: Run) -> None:
        self.run = run
        self.arm = Arm(run.args.arm)
        self.rng = random.Random(run.args.seed)  # noqa: S311
        self.answer_form = tomllib.loads(FORMAT_SENTENCES.read_text("utf-8"))[
            run.family
        ]

    def rich_suffix(self, row: Row) -> str:
        if row.strict_valid:
            reason = (
                "the answer was read in the required form but does not "
                "match the expected answer"
            )
        elif row.forgiving_correct:
            reason = (
                "the expected answer appears in the response, but not in "
                "the required form, so the strict reader cannot accept it"
            )
        else:
            reason = "no answer in the required form could be read"
        read_text = strict_reading(self.run.family, row.response)
        shown = (
            "nothing"
            if not read_text
            else repr(read_text[-STRICT_READING_CHARS:])
        )
        return (
            f"\nStrict reader read: {shown}"
            f"\nReason: {reason}."
            f"\nExpected answer form: {self.answer_form}"
        )

    def shown_output(self, response: str) -> str:
        if self.run.feedback is Feedback.GOLD:
            return response[:REFLECT_OUTPUT_CHARS]
        if len(response) <= REFLECT_OUTPUT_CHARS:
            return response
        return TAIL_MARKER + response[-REFLECT_OUTPUT_CHARS:]

    def evaluate(
        self,
        batch: list[Instance],
        candidate: dict[str, str],
        capture_traces: bool = False,  # noqa: FBT001, FBT002
    ) -> EvaluationBatch:
        self.run.round = self.run.reflection_calls
        rows = self.run.evaluate(
            candidate["instruction"], batch, "gepa", self.run.args.arm
        )
        return EvaluationBatch(
            outputs=[row.response for row in rows],
            scores=[row.score for row in rows],
            trajectories=rows if capture_traces else None,
        )

    def feedback_records(self, rows: Sequence[Row]) -> list[dict[str, Any]]:
        records = []
        for row in rows:
            body = render(
                self.run.spec.naive_template, None, row.prompt_inputs
            )
            feedback = (
                "Correct."
                if row.score >= 1
                else self.run.spec.feedback(row.response, row.gold)
            )
            if self.run.feedback is Feedback.RICH and row.score < 1:
                feedback += self.rich_suffix(row)
            records.append(
                {
                    "Inputs": body[:REFLECT_INPUT_CHARS],
                    "Generated Outputs": self.shown_output(row.response),
                    "Feedback": feedback,
                }
            )
        return records

    def derangement(self, n: int) -> list[int]:
        order = list(range(n))
        for _ in range(SHUFFLE_ATTEMPTS):
            self.rng.shuffle(order)
            if n < 2 or all(i != j for i, j in enumerate(order)):  # noqa: PLR2004
                break
        return order

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],  # noqa: ARG002
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        rows = list(eval_batch.trajectories or [])
        records: list[dict[str, Any]]
        if self.arm is Arm.GEPA_NOFEEDBACK:
            records = [{"Score": f"{row.score:g}"} for row in rows]
        else:
            records = self.feedback_records(rows)
        if self.arm is Arm.GEPA_SHUFFLED:
            order = self.derangement(len(records))
            feedback = [records[j]["Feedback"] for j in order]
            for record, text in zip(records, feedback, strict=True):
                record["Feedback"] = text
        return dict.fromkeys(components_to_update, records)


def clean_proposal(text: str) -> str:
    text = text.strip()
    if text.startswith(FENCE) and text.endswith(FENCE) and len(text) > 1:
        text = text[len(FENCE) : -len(FENCE)].strip()
    return text


def propose(
    run: Run, make_prompt: Callable[[], str], written: Sequence[str | None]
) -> str | None:
    for _ in range(PROPOSAL_ATTEMPTS):
        try:
            text = clean_proposal(run.reflect(make_prompt()))
        except ProviderError as exc:
            run.log(f"{run.args.arm} proposal failed: {exc}")
            continue
        if text and text not in written:
            return text
    run.log(f"{run.args.arm} proposals exhausted")
    return None


def demo_block(run: Run, items: Sequence[Instance]) -> str:
    blocks = []
    for number, inst in enumerate(items, start=1):
        body = render(run.spec.naive_template, None, inst.prompt_inputs)
        blocks.append(
            f"Input {number}:\n{body[:REFLECT_INPUT_CHARS]}\n"
            f"Output {number}:\n{inst.gold[:GOLD_FEEDBACK_CHARS]}"
        )
    return "\n\n".join(blocks)


def run_paraphrase(run: Run) -> tuple[str | None, float]:
    args = run.args
    phase = args.arm
    started = time.monotonic()
    val = run.splits.val
    seed = run.seed_instruction
    seed_rows = run.evaluate(seed, val, phase, "seed")
    best, best_val = seed, mean(seed_rows)
    written: list[str | None] = [seed]
    used = len(val)
    while used + len(val) <= args.max_metric_calls:
        if run.wall_exceeded(started):
            run.log(f"{phase} wall cap reached")
            break
        run.round = len(written)
        proposal = propose(
            run,
            lambda: NULL_PROMPT.format(
                template=run.spec.naive_template,
                seed=seed,
                previous="\n".join(f"- {text}" for text in written),
            ),
            written,
        )
        if proposal is None:
            break
        written.append(proposal)
        rows = run.evaluate(proposal, val, phase, phase)
        used += len(val)
        if mean(rows) > best_val:
            best, best_val = proposal, mean(rows)
            run.log(f"{phase} new best val={best_val:.3f}")
    run.results.update(
        metric_calls=used,
        candidates=len(written),
        proposals=len(written) - 1,
        hyperparameters={
            "selection": "full validation",
            "includes_seed": True,
            "proposal_attempts": PROPOSAL_ATTEMPTS,
        },
    )
    return best, best_val


def ranked(tally: dict[str, list[float]], order: Sequence[str]) -> list[str]:
    position = {text: index for index, text in enumerate(order)}
    return sorted(
        tally,
        key=lambda text: (
            -sum(tally[text]) / len(tally[text]),
            -len(tally[text]),
            position[text],
        ),
    )


def ape_initial(
    run: Run, rng: random.Random, count: int, order: list[str]
) -> list[str]:
    fresh: list[str] = []
    for _ in range(count):
        demos = rng.sample(
            run.splits.train, min(APE_DEMOS, len(run.splits.train))
        )
        text = propose(
            run,
            lambda demos=demos: APE_PROPOSE_PROMPT.format(
                count=len(demos), demos=demo_block(run, demos)
            ),
            order,
        )
        if text is not None:
            order.append(text)
            fresh.append(text)
    return fresh


def run_ape(run: Run) -> tuple[str | None, float]:
    args = run.args
    started = time.monotonic()
    rng = random.Random(args.seed)  # noqa: S311
    val = run.splits.val
    batch_n = min(args.minibatch, len(val))
    final_k = min(APE_FINAL, args.max_metric_calls // len(val))
    if final_k < 1:
        msg = "--max-metric-calls must be at least --val-n for ape"
        raise SystemExit(msg)
    remaining = args.max_metric_calls - final_k * len(val)
    tally: dict[str, list[float]] = {}
    order: list[str] = []
    proposals = 0

    def score(texts: Sequence[str], batch: Sequence[Instance]) -> None:
        nonlocal remaining
        for text in texts:
            rows = run.evaluate(text, batch, "ape", "ape")
            remaining -= len(rows)
            tally.setdefault(text, []).extend(row.score for row in rows)

    run.round = 0
    fresh = ape_initial(
        run, rng, min(APE_INITIAL, remaining // batch_n), order
    )
    proposals += len(fresh)
    score(fresh, rng.sample(val, batch_n))
    while tally and not run.wall_exceeded(started):
        survivors = ranked(tally, order)[:APE_SURVIVORS]
        room = remaining // batch_n - len(survivors)
        count = min(len(survivors) * APE_RESAMPLE, room)
        if count <= 0:
            break
        run.round += 1
        fresh = []
        for index in range(count):
            parent = survivors[index % len(survivors)]
            text = propose(
                run,
                lambda parent=parent: APE_RESAMPLE_PROMPT.format(
                    instruction=parent
                ),
                order,
            )
            if text is not None:
                order.append(text)
                fresh.append(text)
        if not fresh:
            break
        proposals += len(fresh)
        score([*survivors, *fresh], rng.sample(val, batch_n))
    if not tally:
        msg = "ape produced no proposals"
        raise ProviderError(msg)
    run.round += 1
    finalists = ranked(tally, order)[:final_k]
    scored = [
        (text, mean(run.evaluate(text, val, "ape_val", "ape")))
        for text in finalists
    ]
    best, best_val = max(scored, key=lambda pair: pair[1])
    run.results.update(
        metric_calls=args.max_metric_calls - remaining,
        candidates=len(order),
        proposals=proposals,
        rounds=run.round,
        hyperparameters={
            "initial": APE_INITIAL,
            "survivors": APE_SURVIVORS,
            "resample_per_survivor": APE_RESAMPLE,
            "finalists": final_k,
            "demos": APE_DEMOS,
            "minibatch": batch_n,
            "includes_seed": False,
            "selection": "full validation over finalists",
        },
    )
    return best, best_val


def opro_instruction(text: str) -> str:
    start = text.find(OPRO_OPEN)
    if start < 0:
        return text
    end = text.find(OPRO_CLOSE, start)
    stop = len(text) if end < 0 else end
    return text[start + len(OPRO_OPEN) : stop].strip()


def run_opro(run: Run) -> tuple[str | None, float]:
    args = run.args
    started = time.monotonic()
    rng = random.Random(args.seed)  # noqa: S311
    val = run.splits.val
    seed = run.seed_instruction
    history: list[tuple[str, float]] = [
        (seed, mean(run.evaluate(seed, val, "opro", "seed")))
    ]
    used = len(val)
    proposals = 0

    def prompt() -> str:
        shown = sorted(history, key=lambda pair: pair[1])[-OPRO_HISTORY:]
        lines = "\n\n".join(
            f"text:\n{text}\nscore:\n{round(100 * value)}"
            for text, value in shown
        )
        demos = rng.sample(
            run.splits.train, min(OPRO_EXEMPLARS, len(run.splits.train))
        )
        exemplars = "\n\n".join(
            f"Problem:\n{OPRO_OPEN}\n"
            f"{render(run.spec.naive_template, None, i.prompt_inputs)}"
            f"\nAnswer:\n{i.gold[:GOLD_FEEDBACK_CHARS]}"
            for i in demos
        )
        return OPRO_PROMPT.format(history=lines, exemplars=exemplars)

    while used + len(val) <= args.max_metric_calls:
        if run.wall_exceeded(started):
            run.log("opro wall cap reached")
            break
        run.round = len(history)
        written = [text for text, _ in history]
        text = None
        for _ in range(PROPOSAL_ATTEMPTS):
            raw = propose(run, prompt, written)
            if raw is None:
                break
            parsed = opro_instruction(raw)
            if parsed and parsed not in written:
                text = parsed
                break
        if text is None:
            break
        proposals += 1
        value = mean(run.evaluate(text, val, "opro", "opro"))
        used += len(val)
        history.append((text, value))
    best, best_val = max(history, key=lambda pair: pair[1])
    run.results.update(
        metric_calls=used,
        candidates=len(history),
        proposals=proposals,
        hyperparameters={
            "history": OPRO_HISTORY,
            "exemplars": OPRO_EXEMPLARS,
            "includes_seed": True,
            "selection": "full validation",
        },
    )
    return best, best_val


def run_gepa(run: Run) -> tuple[str | None, float]:
    args = run.args

    def reflect(prompt: str | list[dict[str, Any]]) -> str:
        return run.reflect(prompt)

    assert run.out is not None
    result = optimize(
        seed_candidate={"instruction": run.seed_instruction},
        trainset=run.splits.train,
        valset=run.splits.val,
        adapter=GepaAdapter(run),
        reflection_lm=reflect,
        reflection_minibatch_size=args.minibatch,
        max_metric_calls=args.max_metric_calls,
        stop_callbacks=[TimeoutStopCondition(args.wall_cap_s)],
        raise_on_exception=False,
        display_progress_bar=False,
        seed=args.seed,
        run_dir=str(run.out / "gepa"),
    )
    scores = result.val_aggregate_scores
    best = result.best_candidate
    assert isinstance(best, dict)
    run.results.update(
        metric_calls=result.total_metric_calls,
        candidates=len(result.candidates),
        candidate_val_scores=scores,
        proposals=run.reflection_calls,
        hyperparameters={
            "minibatch": args.minibatch,
            "reflective_dataset": {
                Arm.GEPA: "inputs, outputs, feedback with gold",
                Arm.GEPA_NOFEEDBACK: "score only",
                Arm.GEPA_SHUFFLED: "feedback deranged across items",
            }[Arm(args.arm)],
            "feedback": run.feedback,
        },
    )
    return best["instruction"], scores[result.best_idx]


def run_eval(run: Run) -> None:
    instruction = Path(run.args.instruction_file).read_text("utf-8")
    rows = run.evaluate(
        instruction, run.splits.heldout, "best_heldout", "given"
    )
    run.results.update(
        status=Status.DONE,
        best_instruction=instruction,
        best_candidate=run.candidate_ids[instruction],
        best_heldout=mean(rows),
    )
    run.write_results()
    run.log(
        f"done best_heldout={mean(rows):.3f} task_calls={run.task_calls}"
        f" cost=${run.task_cost:.4f}"
    )


def run_arm(run: Run) -> None:
    args = run.args
    if Arm(args.arm) is Arm.EVAL:
        run_eval(run)
        return
    naive = run.naive_instruction
    naive_val = mean(run.evaluate(naive, run.splits.val, "screen", "naive"))
    in_band = args.band_low <= naive_val <= args.band_high
    run.results.update(naive_val=naive_val, in_band=in_band)
    if not in_band:
        run.results["status"] = Status.OUT_OF_BAND
        run.write_results()
        run.log(f"naive val {naive_val:.3f} out of band; stopping")
        return
    heldout = run.splits.heldout
    naive_held = mean(run.evaluate(naive, heldout, "naive_heldout", "naive"))
    run.results["naive_heldout"] = naive_held
    run.write_results()

    arm = Arm(args.arm)
    best: str | None
    if arm is Arm.NAIVE:
        best, best_val = naive, naive_val
    elif arm is Arm.FORMAT_FIXED:
        sentences = tomllib.loads(FORMAT_SENTENCES.read_text("utf-8"))
        best = sentences[args.family]
        best_val = mean(run.evaluate(best, run.splits.val, "val", "format"))
    elif arm in PARAPHRASE_ARMS:
        best, best_val = run_paraphrase(run)
    elif arm is Arm.APE:
        best, best_val = run_ape(run)
    elif arm is Arm.OPRO:
        best, best_val = run_opro(run)
    else:
        best, best_val = run_gepa(run)

    if best == naive:
        best_held = naive_held
        run.heldout_rows["best_heldout"] = run.heldout_rows["naive_heldout"]
    else:
        rows = run.evaluate(best, heldout, "best_heldout", args.arm)
        best_held = mean(rows)
    run.results.update(
        status=Status.DONE,
        best_instruction=best,
        best_candidate=run.candidate_ids[best],
        best_val=best_val,
        best_heldout=best_held,
        delta_heldout=best_held - naive_held,
    )
    run.write_results()
    run.log(
        f"done naive_heldout={naive_held:.3f} best_heldout={best_held:.3f}"
        f" delta={best_held - naive_held:+.3f} task_calls={run.task_calls}"
        f" cost=${run.task_cost + run.reflection_cost:.4f}"
    )


def registry_line(  # noqa: PLR0913
    args: argparse.Namespace,
    out: Path,
    status: str,
    started: str,
    finished: str | None,
    cost: float | None,
) -> dict[str, Any]:
    return {
        "run_id": out.name,
        "workstream": args.workstream,
        "hypotheses": args.hypotheses,
        "model": args.task_model,
        "task": args.family,
        "arm": args.arm,
        "objective": args.objective,
        "scoring": args.scoring,
        "seed": args.seed,
        "serving": (
            Serving.OPENROUTER
            if args.base_url.rstrip("/") == OPENROUTER_BASE_URL
            else Serving.VLLM
        ),
        "code_tag": args.code_tag,
        "job_file": args.job_file,
        "out_dir": str(out),
        "status": status,
        "started": started,
        "finished": finished,
        "cost_usd": cost,
    }


def append_registry(args: argparse.Namespace, line: dict[str, Any]) -> None:
    if args.registry is None:
        return
    path = Path(args.registry).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def probe(run: Run) -> None:
    inst = run.splits.heldout[0]
    prompt = render(
        run.spec.naive_template, run.naive_instruction, inst.prompt_inputs
    )

    async def go() -> list[Any]:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            if run.scoring is Scoring.LIKELIHOOD:
                choices = options(run.family, inst.gold)
                assert choices is not None
                prefix = await run.prefix_ids(client, prompt)
                answer = await run.answer_ids(client, choices[0])
                _, payload = await run.option_logprob(client, prefix, answer)
                scores = await run.score_options(client, prompt, inst.gold)
                return [
                    {"prefix_ids_tail": prefix[-12:], "option_ids": answer},
                    payload,
                    asdict(scores),
                ]
            path, body = run.task_request(prompt)
            payload = await run.request(
                client, run.task_base_url + path, body, run.task_key
            )
            return [body, payload]

    print(json.dumps({"instance_id": inst.id, "gold": inst.gold}))
    for part in asyncio.run(go()):
        print(json.dumps(part, indent=1)[:PROBE_CHARS])


def validate(args: argparse.Namespace) -> None:
    if Arm(args.arm) is Arm.EVAL and args.instruction_file is None:
        msg = "--arm eval requires --instruction-file"
        raise SystemExit(msg)
    openrouter = args.base_url.rstrip("/") == OPENROUTER_BASE_URL
    if Scoring(args.scoring) is Scoring.LIKELIHOOD:
        if openrouter:
            msg = (
                "--scoring likelihood needs vLLM's /tokenize and "
                "prompt_logprobs; OpenRouter does not provide them"
            )
            raise SystemExit(msg)
        if Family(args.family) not in {Family.C18, Family.C19}:
            msg = (
                f"--scoring likelihood has no enumerable answer set for "
                f"{args.family}; only c18 and c19 are supported"
            )
            raise SystemExit(msg)
    if args.thinking == Thinking.OFF and args.reasoning:
        msg = "--thinking off conflicts with --reasoning"
        raise SystemExit(msg)
    if not args.probe and args.out is None:
        msg = "--out is required unless --probe is set"
        raise SystemExit(msg)


def main() -> None:
    args = parse_args()
    validate(args)
    if args.probe:
        probe(Run(args, None))
        return
    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        msg = f"output directory is not empty: {out}"
        raise SystemExit(msg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(
        json.dumps(vars(args), indent=1) + "\n", encoding="utf-8"
    )
    started = now()
    append_registry(
        args, registry_line(args, out, Status.RUNNING, started, None, None)
    )
    run: Run | None = None
    try:
        run = Run(args, out)
        run.log(
            f"start family={args.family} arm={args.arm}"
            f" model={args.task_model} objective={args.objective}"
            f" scoring={args.scoring} reasoning={args.reasoning}"
            f" thinking={args.thinking} reflection={args.reflection_model}"
        )
        run_arm(run)
    except BaseException:
        cost = None if run is None else run.total_cost()
        append_registry(
            args,
            registry_line(args, out, Status.FAILED, started, now(), cost),
        )
        raise
    append_registry(
        args,
        registry_line(
            args, out, run.results["status"], started, now(), run.total_cost()
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one prelim arm (naive, format_fixed, null, bestofn, ape, "
            "opro, gepa, gepa-nofeedback, gepa-shuffled, or eval) for one "
            "whetstone-envs task family. The task model is served "
            "by any OpenAI-compatible endpoint (--base-url, OpenRouter by "
            "default); reflection always uses OpenRouter. Items come from "
            "the family generator, shuffled by --split-seed into train, "
            "validation, and held-out slices. Every other arm screens the "
            "naive probe on validation and stops when it falls outside the "
            "band; in-band runs score naive and the arm's best instruction "
            "on held-out. Search arms share --max-metric-calls: null and "
            "bestofn score paraphrases of the seed on validation; ape "
            "proposes instructions from train demonstrations, screens them "
            "on validation minibatches, resamples the survivors, and picks "
            "among finalists on validation; opro proposes from the scored "
            "history; gepa reflects on feedback, gepa-nofeedback on scores "
            "only, and gepa-shuffled on feedback deranged across items. "
            "The eval arm skips the screen and the naive "
            "baseline and scores only --instruction-file on held-out. "
            "Every item is read by the strict and forgiving readers of "
            "readers.py; --objective picks the one the arms optimize. "
            "--scoring likelihood replaces generation by option "
            "log-likelihood on vLLM (c18 and c19 only). Writes items.jsonl, "
            "logprobs.jsonl, candidates.jsonl, results.json, and "
            "progress.log under --out. Requires the --api-key-env "
            "variable, and OPENROUTER_API_KEY for the search arms."
        )
    )
    parser.add_argument("--family", required=True, choices=list(Family))
    parser.add_argument("--arm", required=True, choices=list(Arm))
    parser.add_argument("--task-model", required=True)
    parser.add_argument(
        "--base-url",
        default=OPENROUTER_BASE_URL,
        help="OpenAI-compatible base URL serving the task model",
    )
    parser.add_argument(
        "--api-key-env",
        default=OPENROUTER_KEY_ENV,
        help="environment variable holding the task endpoint's API key",
    )
    parser.add_argument(
        "--reasoning",
        help=(
            "task-model reasoning effort, sent as OpenRouter's reasoning "
            "object or as reasoning_effort on other endpoints; omitted "
            "when unset"
        ),
    )
    parser.add_argument(
        "--thinking",
        choices=list(Thinking),
        help=(
            "chat_template_kwargs.enable_thinking on vLLM, "
            "reasoning.enabled on OpenRouter; omitted when unset"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=list(Reader),
        default=Reader.STRICT,
        help="reader whose correctness the arms optimize and report",
    )
    parser.add_argument(
        "--feedback",
        choices=list(Feedback),
        default=Feedback.GOLD,
        help=(
            "reflective feedback on failed items: gold gives the expected "
            "answer; rich adds the strict reader's validity verdict and the "
            "family's expected answer form from format_sentences.toml, "
            "what the strict reader read, and the reason for failure, and "
            "shows the response tail instead of its head"
        ),
    )
    parser.add_argument(
        "--scoring", choices=list(Scoring), default=Scoring.GENERATE
    )
    parser.add_argument(
        "--prompt-format",
        choices=list(PromptFormat),
        default=PromptFormat.CHAT,
        help="chat messages, or plain text on /completions for base models",
    )
    parser.add_argument(
        "--logprobs",
        choices=list(Logprobs),
        default=Logprobs.AUTO,
        help="request top-5 token logprobs; auto is on except OpenRouter",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--stop",
        action="append",
        help="stop sequence for --prompt-format plain; repeatable",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="system message (chat) or text prepended to the prompt (plain)",
    )
    parser.add_argument(
        "--seed-file",
        help=(
            "instruction replacing the seed of every arm: the naive "
            "baseline, and the starting candidate of every search arm but ape"
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="append this many train items with gold to the seed",
    )
    parser.add_argument(
        "--instruction-file",
        help="file whose exact text the eval arm scores on held-out",
    )
    parser.add_argument("--reflection-model", default="openai/gpt-5.4-nano")
    parser.add_argument("--train-n", type=int, default=24)
    parser.add_argument("--val-n", type=int, default=40)
    parser.add_argument("--heldout-n", type=int, default=48)
    parser.add_argument(
        "--seed", type=int, default=0, help="optimizer seed of the search arms"
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--n-per-category",
        type=int,
        help="generator items per category; the family default when unset",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=300,
        help="task calls available to the search arms",
    )
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument(
        "--wall-cap-s",
        type=float,
        default=3600.0,
        help="wall-clock cap on the search arms",
    )
    parser.add_argument("--band-low", type=float, default=0.10)
    parser.add_argument("--band-high", type=float, default=0.85)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="concurrent task requests (asyncio, single thread)",
    )
    parser.add_argument("--out")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="send the first held-out item and print the raw response",
    )
    parser.add_argument(
        "--registry", help="run registry JSONL to append start/end lines to"
    )
    parser.add_argument("--workstream")
    parser.add_argument(
        "--hypotheses",
        type=lambda text: [part for part in text.split(",") if part],
        default=[],
        help="comma-separated hypothesis ids",
    )
    parser.add_argument("--job-file")
    parser.add_argument("--code-tag")
    return parser.parse_args()


if __name__ == "__main__":
    main()
