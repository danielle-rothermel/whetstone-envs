from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

STATES = ("pending", "running", "done", "failed")
PORT_BASE = 18100
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_UTIL = 0.85
LARGE_GPU_UTIL = 0.92
LARGE_SLUGS = ("qwen3-30b-a3b-instruct-2507", "qwen3-32b-instruct")
LONG_TOKENS = 2048
RUNNER = [
    "uvx",
    "--from",
    "uv==0.12.3",
    "uv",
    "run",
    "--no-sync",
    "python",
    "scripts/prelim/gepa_family.py",
]
LIKELIHOOD_FAMILIES = ("c18", "c19")
SKILL_FAMILIES = ("c11", "c23")
N_PER_CATEGORY = {"c18": 90, "c23": 90}
HYPOTHESES = {
    ("strict", "generate"): "H3,H4,H10",
    ("forgiving", "generate"): "H1,H4",
    ("strict", "likelihood"): "H2,H4,H10",
}


@dataclass(frozen=True, slots=True)
class Spec:
    family: str
    arm: str
    objective: str
    scoring: str
    seed: int
    variant: tuple[str, ...] = ()


def specs(*, small: bool) -> list[Spec]:
    out: list[Spec] = []
    for fam in LIKELIHOOD_FAMILIES:
        out.append(Spec(fam, "naive", "strict", "generate", 0))
        out += [Spec(fam, "gepa", "strict", "generate", s) for s in range(3)]
        out += [
            Spec(fam, "gepa", "forgiving", "generate", s) for s in range(3)
        ]
        out.append(Spec(fam, "naive", "strict", "likelihood", 0))
        out += [Spec(fam, "gepa", "strict", "likelihood", s) for s in range(3)]
    if not small:
        for fam in SKILL_FAMILIES:
            out.append(Spec(fam, "naive", "strict", "generate", 0))
            out += [
                Spec(fam, "gepa", "strict", "generate", s) for s in range(3)
            ]
            out.append(Spec(fam, "gepa", "forgiving", "generate", 0))
    return out


def variant_specs(*, small: bool, base: bool, long_ok: bool) -> list[Spec]:
    families = (
        LIKELIHOOD_FAMILIES if small else LIKELIHOOD_FAMILIES + SKILL_FAMILIES
    )
    out: list[Spec] = []
    for fam in families:
        out.append(Spec(fam, "naive", "strict", "generate", 0, ("seedinstr",)))
        if long_ok:
            out.append(
                Spec(fam, "naive", "strict", "generate", 0, ("tok2048",))
            )
            out.append(
                Spec(
                    fam,
                    "naive",
                    "strict",
                    "generate",
                    0,
                    ("seedinstr", "tok2048"),
                )
            )
        if base:
            out.append(
                Spec(
                    fam,
                    "naive",
                    "strict",
                    "generate",
                    0,
                    ("seedinstr", "shots3ans"),
                )
            )
    for fam in LIKELIHOOD_FAMILIES:
        out.append(
            Spec(fam, "naive", "strict", "likelihood", 0, ("seedinstr",))
        )
        if base:
            out.append(
                Spec(
                    fam,
                    "naive",
                    "strict",
                    "likelihood",
                    0,
                    ("seedinstr", "shots3ans"),
                )
            )
    return out


def variant_args(variant: tuple[str, ...], seed_file: str) -> list[str]:
    out: list[str] = []
    if "seedinstr" in variant:
        out += ["--seed-file", seed_file]
    if "tok2048" in variant:
        out += ["--max-tokens", "2048"]
    if "shots3ans" in variant:
        out += ["--shots", "3"]
    return out


def choose_specs(m: dict, args: argparse.Namespace) -> list[Spec]:
    small = bool(m.get("small"))
    long_ok = m.get("max_model_len", DEFAULT_MAX_MODEL_LEN) > LONG_TOKENS
    if args.variants:
        return variant_specs(
            small=small, base=m["kind"] == "base", long_ok=long_ok
        )
    chosen = specs(small=small)
    if not args.gepa_seedinstr:
        return chosen
    variant = ("seedinstr", "tok2048") if long_ok else ("seedinstr",)
    return [
        Spec(sp.family, sp.arm, sp.objective, sp.scoring, sp.seed, variant)
        for sp in chosen
        if sp.arm == "gepa"
    ]


def write_models_tsv(models: list[dict], stage_dir: Path) -> None:
    rows = []
    for idx, m in enumerate(models):
        slug = m["slug"]
        rows.append(
            "\t".join(
                [
                    slug,
                    m["hf"],
                    m.get("revision", "-"),
                    m["kind"],
                    str(PORT_BASE + idx),
                    str(m.get("max_model_len", DEFAULT_MAX_MODEL_LEN)),
                    m.get("chat_template", "-"),
                    "1" if m.get("thinking_off") else "0",
                    str(
                        LARGE_GPU_UTIL
                        if slug in LARGE_SLUGS
                        else DEFAULT_GPU_UTIL
                    ),
                ]
            )
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "models.tsv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write model-major vLLM queue job files and models.tsv."
    )
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--cluster-qroot", required=True)
    parser.add_argument("--cluster-clone", required=True)
    parser.add_argument("--cluster-out-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--code-tag", required=True)
    parser.add_argument(
        "--scoring", choices=("generate", "likelihood", "all"), default="all"
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--gepa-seedinstr", action="store_true")
    parser.add_argument("--seed-file")
    args = parser.parse_args()

    models = tomllib.loads(args.models.read_text(encoding="utf-8"))["models"]
    write_models_tsv(models, args.stage_dir)
    total = 0
    for idx, m in enumerate(models):
        slug = m["slug"]
        if args.only is not None and slug not in args.only:
            continue
        port = PORT_BASE + idx
        qdir = args.stage_dir / slug
        for state in STATES:
            (qdir / state).mkdir(parents=True, exist_ok=True)
        chosen = choose_specs(m, args)
        for spec in chosen:
            if args.scoring not in ("all", spec.scoring):
                continue
            name = (
                f"{spec.family}-{slug}-{spec.arm}-{spec.objective}-"
                f"{spec.scoring}-s{spec.seed}"
                + "".join(f"-{v}" for v in spec.variant)
            )
            job_file = f"{args.cluster_qroot}/{slug}/pending/{name}.json"
            argv = [
                *RUNNER,
                "--family",
                spec.family,
                "--arm",
                spec.arm,
                "--task-model",
                slug,
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--api-key-env",
                "DUMMY_KEY",
                "--objective",
                spec.objective,
                "--scoring",
                spec.scoring,
                "--temperature",
                "0",
                "--max-tokens",
                "512",
                "--logprobs",
                "on",
                "--train-n",
                "48",
                "--val-n",
                "60",
                "--heldout-n",
                "240",
                "--split-seed",
                "0",
                "--band-low",
                "0",
                "--band-high",
                "1",
                "--concurrency",
                "16",
                "--registry",
                args.registry,
                "--workstream",
                "02",
                "--hypotheses",
                HYPOTHESES[(spec.objective, spec.scoring)],
                "--job-file",
                job_file,
                "--code-tag",
                args.code_tag,
            ]
            if spec.family in N_PER_CATEGORY:
                argv += ["--n-per-category", str(N_PER_CATEGORY[spec.family])]
            if m.get("thinking_off"):
                argv += ["--thinking", "off"]
            if m["kind"] == "base":
                argv += ["--prompt-format", "plain"]
            if spec.arm == "gepa":
                argv += [
                    "--seed",
                    str(spec.seed),
                    "--reflection-model",
                    "openai/gpt-5.4-nano",
                    "--max-metric-calls",
                    "300",
                    "--minibatch",
                    "4",
                    "--wall-cap-s",
                    "3600",
                ]
            if spec.variant:
                if args.seed_file is None:
                    msg = "--variants needs --seed-file"
                    raise SystemExit(msg)
                argv += variant_args(spec.variant, args.seed_file)
            argv += ["--out", f"{args.cluster_out_root}/{name}"]
            record = {"name": name, "cwd": args.cluster_clone, "argv": argv}
            (qdir / "pending" / f"{name}.json").write_text(
                json.dumps(record, indent=1) + "\n", encoding="utf-8"
            )
            total += 1
    print(f"wrote {total} jobs under {args.stage_dir}")


if __name__ == "__main__":
    main()
