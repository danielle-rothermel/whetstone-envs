from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

STATES = ("pending", "running", "done", "failed")
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a jobs TOML into one JSON file per job under "
            "QUEUE/pending/. The TOML has a [queue] table with `command` "
            "(argv prefix) and `out_root`, and [[jobs]] entries with a "
            "unique `name` and `args`. Each job runs `command + args + "
            "['--out', out_root/name]` from the repository root. Jobs whose "
            "name already exists in any queue state are skipped."
        )
    )
    parser.add_argument("jobs", type=Path, help="jobs TOML file")
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()

    spec = tomllib.loads(args.jobs.read_text(encoding="utf-8"))
    command = list(spec["queue"]["command"])
    out_root = Path(spec["queue"]["out_root"]).expanduser()
    names = [job["name"] for job in spec["jobs"]]
    if len(set(names)) != len(names):
        msg = "job names must be unique"
        raise SystemExit(msg)
    for state in STATES:
        (args.queue / state).mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    for job in spec["jobs"]:
        name = job["name"]
        filename = f"{name}.json"
        if any((args.queue / s / filename).exists() for s in STATES):
            skipped += 1
            continue
        record = {
            "name": name,
            "cwd": str(REPO_ROOT),
            "argv": [
                *command,
                *job["args"],
                "--out",
                str(out_root / name),
            ],
        }
        tmp = args.queue / f".{filename}.tmp"
        tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        tmp.rename(args.queue / "pending" / filename)
        added += 1
    print(
        f"added {added} jobs, skipped {skipped} existing, queue {args.queue}"
    )


if __name__ == "__main__":
    main()
