from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
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

from whetstone_envs import c11, c18, c19, c22, c23
from whetstone_envs.c22.constraints import ConstraintStack
from whetstone_envs.c22.oracle import evaluate as c22_evaluate
from whetstone_envs.instances import Instance
from whetstone_envs.scoring import exact_match

if TYPE_CHECKING:
    from collections.abc import Mapping

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FORMAT_SENTENCES = Path(__file__).with_name("format_sentences.toml")
SEED_INSTRUCTION = "Solve the task below."
SPLIT_SEED = 20261016
HTTP_TIMEOUT_S = 180.0
HTTP_RETRIES = 5
HTTP_RETRY_STATUS = 429
HTTP_SERVER_ERROR = 500
GOLD_FEEDBACK_CHARS = 800
REFLECT_INPUT_CHARS = 3000
REFLECT_OUTPUT_CHARS = 1500
NULL_PROPOSAL_ATTEMPTS = 5

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


@verify(ENUM_UNIQUE)
class Family(StrEnum):
    C11 = auto()
    C18 = auto()
    C19 = auto()
    C22 = auto()
    C23 = auto()


@verify(ENUM_UNIQUE)
class Arm(StrEnum):
    NAIVE = auto()
    FORMAT_FIXED = auto()
    NULL = auto()
    GEPA = auto()


@verify(ENUM_UNIQUE)
class Status(StrEnum):
    RUNNING = auto()
    OUT_OF_BAND = auto()
    DONE = auto()


@dataclass(frozen=True, slots=True)
class FamilySpec:
    naive_template: str
    pool: Callable[[], Sequence[Instance]]
    score: Callable[[str, str], int]
    feedback: Callable[[str, str], str]


def gold_feedback(_response: str, gold: str) -> str:
    return f"Incorrect. Expected answer: {gold[:GOLD_FEEDBACK_CHARS]}"


def c22_feedback(response: str, gold: str) -> str:
    stack = ConstraintStack.from_gold(gold)
    verdicts = c22_evaluate(stack, response).per_constraint
    failed = ", ".join(name for name, passed in verdicts if not passed)
    return f"Incorrect. Failed constraints: {failed}"


FAMILIES: dict[Family, FamilySpec] = {
    Family.C11: FamilySpec(
        c11.PROBES.naive_template,
        lambda: c11.generate_pool().instance_group.instances,
        exact_match,
        gold_feedback,
    ),
    Family.C18: FamilySpec(
        c18.PROBES.naive_template,
        lambda: c18.generate_pool().instance_group.instances,
        c18.score_gold,
        gold_feedback,
    ),
    Family.C19: FamilySpec(
        c19.PROBES.naive_template,
        lambda: c19.generate_pool().instance_group.instances,
        exact_match,
        gold_feedback,
    ),
    Family.C22: FamilySpec(
        c22.PROBES.naive_template,
        lambda: c22.generate_pool().instance_group.instances,
        lambda response, gold: c22.score(gold, response),
        c22_feedback,
    ),
    Family.C23: FamilySpec(
        c23.PROBES.naive_template,
        lambda: c23.generate_pool().instance_group.instances,
        c23.score_gold,
        gold_feedback,
    ),
}


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    cost: float | None


@dataclass(frozen=True, slots=True)
class Row:
    instance_id: str
    split: str
    prompt_inputs: dict[str, str]
    prompt: str
    response: str
    gold: str
    score: float
    error: str | None
    cost: float | None


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


def make_splits(spec: FamilySpec, args: argparse.Namespace) -> Splits:
    instances = list(spec.pool())
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


class Run:
    def __init__(self, args: argparse.Namespace, out: Path) -> None:
        self.args = args
        self.out = out
        self.spec = FAMILIES[Family(args.family)]
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.splits = make_splits(self.spec, args)
        self.split_of = {
            inst.id: name
            for name, part in (
                ("train", self.splits.train),
                ("val", self.splits.val),
                ("heldout", self.splits.heldout),
            )
            for inst in part
        }
        self.items = (out / "items.jsonl").open("a", encoding="utf-8")
        self.candidates = (out / "candidates.jsonl").open(
            "a", encoding="utf-8"
        )
        self.candidate_ids: dict[str | None, int] = {}
        self.started = time.monotonic()
        self.task_calls = 0
        self.task_cost = 0.0
        self.reflection_calls = 0
        self.reflection_cost = 0.0
        self.uncosted_calls = 0
        self.results: dict[str, Any] = {
            "family": args.family,
            "arm": args.arm,
            "task_model": args.task_model,
            "reasoning": args.reasoning,
            "reflection_model": args.reflection_model,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "data_source": "generator",
            "sizes": {
                "train": args.train_n,
                "val": args.val_n,
                "heldout": args.heldout_n,
            },
            "max_metric_calls": args.max_metric_calls,
            "minibatch": args.minibatch,
            "wall_cap_s": args.wall_cap_s,
            "band": [args.band_low, args.band_high],
            "status": Status.RUNNING,
        }

    def log(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
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
        )
        (self.out / "results.json").write_text(
            json.dumps(self.results, indent=1) + "\n", encoding="utf-8"
        )

    def candidate(self, instruction: str | None, source: str) -> int:
        if instruction not in self.candidate_ids:
            index = len(self.candidate_ids)
            self.candidate_ids[instruction] = index
            record = {
                "candidate": index,
                "source": source,
                "instruction": instruction,
            }
            self.candidates.write(json.dumps(record) + "\n")
            self.candidates.flush()
        return self.candidate_ids[instruction]

    def body(
        self,
        model: str,
        prompt: str | list[dict[str, Any]],
        effort: str | None,
    ) -> dict:
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "usage": {"include": True},
        }
        if effort:
            body["reasoning"] = {"effort": effort}
        return body

    async def post(self, client: httpx.AsyncClient, body: dict) -> Reply:
        last = ""
        for attempt in range(HTTP_RETRIES):
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.key}"},
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
                    payload = response.json()
                    choices = payload.get("choices")
                    if not choices:
                        last = f"no choices: {str(payload)[:300]}"
                    else:
                        usage = payload.get("usage") or {}
                        return Reply(
                            text=choices[0]["message"].get("content") or "",
                            cost=usage.get("cost"),
                        )
            await asyncio.sleep(2**attempt)
        raise ProviderError(last)

    def add_cost(self, cost: float | None, *, reflection: bool) -> None:
        if cost is None:
            self.uncosted_calls += 1
        elif reflection:
            self.reflection_cost += cost
        else:
            self.task_cost += cost

    async def run_batch(
        self, instruction: str | None, batch: Sequence[Instance]
    ) -> list[Row]:
        limits = httpx.Limits(max_connections=self.args.concurrency)
        gate = asyncio.Semaphore(self.args.concurrency)
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_S, limits=limits
        ) as client:

            async def one(inst: Instance) -> Row:
                prompt = render(
                    self.spec.naive_template, instruction, inst.prompt_inputs
                )
                body = self.body(
                    self.args.task_model, prompt, self.args.reasoning
                )
                error = None
                cost = None
                async with gate:
                    try:
                        reply = await self.post(client, body)
                    except ProviderError as exc:
                        text, error = "", str(exc)
                    else:
                        text, cost = reply.text, reply.cost
                score = float(self.spec.score(text, inst.gold))
                return Row(
                    instance_id=inst.id,
                    split=self.split_of[inst.id],
                    prompt_inputs=dict(inst.prompt_inputs),
                    prompt=prompt,
                    response=text,
                    gold=inst.gold,
                    score=score,
                    error=error,
                    cost=cost,
                )

            return list(await asyncio.gather(*(one(i) for i in batch)))

    def evaluate(
        self,
        instruction: str | None,
        batch: Sequence[Instance],
        phase: str,
        source: str,
    ) -> list[Row]:
        index = self.candidate(instruction, source)
        rows = asyncio.run(self.run_batch(instruction, batch))
        for row in rows:
            self.task_calls += 1
            self.add_cost(row.cost, reflection=False)
            record = {"phase": phase, "candidate": index, **asdict(row)}
            self.items.write(json.dumps(record) + "\n")
        self.items.flush()
        errors = sum(row.error is not None for row in rows)
        self.log(
            f"{phase} candidate={index} n={len(rows)} mean={mean(rows):.3f}"
            f" errors={errors} task_calls={self.task_calls}"
        )
        return rows

    def reflect(self, prompt: str | list[dict[str, Any]]) -> str:
        body = self.body(self.args.reflection_model, prompt, None)

        async def call() -> Reply:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                return await self.post(client, body)

        reply = asyncio.run(call())
        self.reflection_calls += 1
        self.add_cost(reply.cost, reflection=True)
        return reply.text

    def wall_exceeded(self, started: float) -> bool:
        return time.monotonic() - started > self.args.wall_cap_s


class GepaAdapter:
    propose_new_texts = None

    def __init__(self, run: Run) -> None:
        self.run = run

    def evaluate(
        self,
        batch: list[Instance],
        candidate: dict[str, str],
        capture_traces: bool = False,  # noqa: FBT001, FBT002
    ) -> EvaluationBatch:
        rows = self.run.evaluate(
            candidate["instruction"], batch, "gepa", "gepa"
        )
        return EvaluationBatch(
            outputs=[row.response for row in rows],
            scores=[row.score for row in rows],
            trajectories=rows if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],  # noqa: ARG002
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records = []
        for row in eval_batch.trajectories or []:
            body = render(
                self.run.spec.naive_template, None, row.prompt_inputs
            )
            feedback = (
                "Correct."
                if row.score >= 1
                else self.run.spec.feedback(row.response, row.gold)
            )
            records.append(
                {
                    "Inputs": body[:REFLECT_INPUT_CHARS],
                    "Generated Outputs": row.response[:REFLECT_OUTPUT_CHARS],
                    "Feedback": feedback,
                }
            )
        return dict.fromkeys(components_to_update, records)


def run_null(run: Run) -> tuple[str | None, float]:
    args = run.args
    started = time.monotonic()
    val = run.splits.val
    seed_rows = run.evaluate(SEED_INSTRUCTION, val, "null", "seed")
    best, best_val = SEED_INSTRUCTION, mean(seed_rows)
    written = [SEED_INSTRUCTION]
    used = len(val)
    while used + len(val) <= args.max_metric_calls:
        if run.wall_exceeded(started):
            run.log("null wall cap reached")
            break
        proposal = None
        for _ in range(NULL_PROPOSAL_ATTEMPTS):
            prompt = NULL_PROMPT.format(
                template=run.spec.naive_template,
                seed=SEED_INSTRUCTION,
                previous="\n".join(f"- {text}" for text in written),
            )
            try:
                text = run.reflect(prompt).strip()
            except ProviderError as exc:
                run.log(f"null proposal failed: {exc}")
                continue
            if text and text not in written:
                proposal = text
                break
        if proposal is None:
            run.log("null proposals exhausted")
            break
        written.append(proposal)
        rows = run.evaluate(proposal, val, "null", "null")
        used += len(val)
        if mean(rows) > best_val:
            best, best_val = proposal, mean(rows)
            run.log(f"null new best val={best_val:.3f}")
    run.results.update(metric_calls=used, candidates=len(written))
    return best, best_val


def run_gepa(run: Run) -> tuple[str | None, float]:
    args = run.args

    def reflect(prompt: str | list[dict[str, Any]]) -> str:
        return run.reflect(prompt)

    result = optimize(
        seed_candidate={"instruction": SEED_INSTRUCTION},
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
    )
    return best["instruction"], scores[result.best_idx]


def main() -> None:
    args = parse_args()
    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        msg = f"output directory is not empty: {out}"
        raise SystemExit(msg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(
        json.dumps(vars(args), indent=1) + "\n", encoding="utf-8"
    )
    run = Run(args, out)
    run.log(
        f"start family={args.family} arm={args.arm} model={args.task_model}"
        f" reasoning={args.reasoning} reflection={args.reflection_model}"
    )
    naive_val = mean(run.evaluate(None, run.splits.val, "screen", "naive"))
    in_band = args.band_low <= naive_val <= args.band_high
    run.results.update(naive_val=naive_val, in_band=in_band)
    if not in_band:
        run.results["status"] = Status.OUT_OF_BAND
        run.write_results()
        run.log(f"naive val {naive_val:.3f} out of band; stopping")
        return
    heldout = run.splits.heldout
    naive_held = mean(run.evaluate(None, heldout, "naive_heldout", "naive"))
    run.results["naive_heldout"] = naive_held
    run.write_results()

    arm = Arm(args.arm)
    best: str | None
    if arm is Arm.NAIVE:
        best, best_val = None, naive_val
    elif arm is Arm.FORMAT_FIXED:
        sentences = tomllib.loads(FORMAT_SENTENCES.read_text("utf-8"))
        best = sentences[args.family]
        best_val = mean(run.evaluate(best, run.splits.val, "val", "format"))
    elif arm is Arm.NULL:
        best, best_val = run_null(run)
    else:
        best, best_val = run_gepa(run)

    if best is None:
        best_held = naive_held
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one prelim arm (naive, format_fixed, null, or gepa) for one "
            "whetstone-envs task family over OpenRouter. Items come from the "
            "family generator, shuffled by --split-seed into train, "
            "validation, and held-out slices. Every run screens the naive "
            "probe on validation and stops when it falls outside the band; "
            "in-band runs score naive and the arm's best instruction on "
            "held-out. Writes items.jsonl, candidates.jsonl, results.json, "
            "and progress.log under --out. Requires OPENROUTER_API_KEY."
        )
    )
    parser.add_argument("--family", required=True, choices=list(Family))
    parser.add_argument("--arm", required=True, choices=list(Arm))
    parser.add_argument("--task-model", required=True)
    parser.add_argument(
        "--reasoning", help="task-model reasoning effort; omitted when unset"
    )
    parser.add_argument("--reflection-model", default="openai/gpt-5.4-nano")
    parser.add_argument("--train-n", type=int, default=24)
    parser.add_argument("--val-n", type=int, default=40)
    parser.add_argument("--heldout-n", type=int, default=48)
    parser.add_argument(
        "--seed", type=int, default=0, help="optimizer seed (gepa)"
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=300,
        help="task calls available to the null and gepa searches",
    )
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument(
        "--wall-cap-s",
        type=float,
        default=3600.0,
        help="wall-clock cap on the null and gepa searches",
    )
    parser.add_argument("--band-low", type=float, default=0.10)
    parser.add_argument("--band-high", type=float, default=0.85)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="concurrent task requests (asyncio, single thread)",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
