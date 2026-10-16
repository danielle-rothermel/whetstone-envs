from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def log(queue: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} {message}"
    print(line, flush=True)
    with (queue / "worker.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def claim(queue: Path) -> Path | None:
    for path in sorted((queue / "pending").glob("*.json")):
        target = queue / "running" / path.name
        try:
            path.rename(target)
        except FileNotFoundError:
            continue
        return target
    return None


def run_job(queue: Path, claimed: Path) -> None:
    job = json.loads(claimed.read_text(encoding="utf-8"))
    name = job["name"]
    log_path = queue / "logs" / f"{name}.log"
    log(queue, f"start {name}")
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as out:
        result = subprocess.run(  # noqa: S603
            job["argv"],
            cwd=job["cwd"],
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = round(time.monotonic() - started, 1)
    state = "done" if result.returncode == 0 else "failed"
    job.update(
        returncode=result.returncode, elapsed_s=elapsed, worker=os.getpid()
    )
    claimed.write_text(json.dumps(job, indent=1) + "\n", encoding="utf-8")
    claimed.rename(queue / state / claimed.name)
    log(queue, f"{state} {name} exit={result.returncode} elapsed={elapsed}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run queued jobs one at a time. The worker claims the first "
            "file in QUEUE/pending/ by atomic rename into running/, runs its "
            "argv with output in logs/<name>.log, then moves it to done/ or "
            "failed/ with the exit code. Start several workers for "
            "parallelism. A worker exits after --idle-exit-s with nothing "
            "pending. A job left in running/ by a dead worker is moved back "
            "to pending/ by hand."
        )
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--idle-exit-s", type=float, default=60.0)
    parser.add_argument("--poll-s", type=float, default=5.0)
    args = parser.parse_args()

    for state in ("pending", "running", "done", "failed", "logs"):
        (args.queue / state).mkdir(parents=True, exist_ok=True)
    log(args.queue, "worker up")
    idle_since = time.monotonic()
    while True:
        claimed = claim(args.queue)
        if claimed is not None:
            run_job(args.queue, claimed)
            idle_since = time.monotonic()
            continue
        if time.monotonic() - idle_since >= args.idle_exit_s:
            log(args.queue, "idle; exiting")
            return
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
