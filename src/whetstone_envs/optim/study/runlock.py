"""An exclusive, liveness-aware lock over one run directory.

**The defect this exists to prevent.** Run ids are deterministic on arm and
seed (``_run_id_for``), and that is deliberate: it is what makes a crashed
stage resumable without paying twice. The consequence is that two
``whetstone-study run`` invocations of the same stage compute the *same*
run directory, and the only interlock guarding it was directory existence
-- ``run_dir.exists()`` in :class:`
~whetstone_envs.optim.study.arms.StudyOptimizerRunner`, over a
``mkdir(exist_ok=True)``. Existence is a fact about the past, not about
whether anyone is writing right now: two processes launched seconds apart
both saw no directory, both proceeded, and both drove the same run. The
result was interleaved effects, an intent left permanently ``leased``, and
paid work written off.

**Why the effect-lease authority did not prevent it.** It is not a run-
directory lock. ``EffectLeaseAuthority.sqlite`` is scoped *per run
directory*, so two processes on one directory share one authority; it
correctly detected the foreign writer and refused, but only when a lease
renewal noticed its row had been taken -- at terminalization, after the
money was spent. This lock is the earlier, cheaper refusal: before the run
is dispatched, before anything is paid for.

**Liveness, not existence.** A lockfile that only records "someone held
this" reintroduces the original defect with an extra step: a crashed run
leaves residue that blocks every later invocation until someone deletes it
by hand, which trains operators to delete lockfiles reflexively. So the
holder is recorded with enough identity to be *checked* -- pid, the pid's
start time, and hostname -- and a conflict is resolved on what is true now:

* The holder is **live**: refuse, naming it. This is the real double-drive,
  and it is the one case that must never be resolved silently.
* The holder is **dead**: the lockfile is crash residue. Remove it and
  proceed, leaving the *directory's* own judgement to
  ``_is_reusable``/``--discard-stale-runs``, which reads the run's
  artifacts rather than this file.

Pid alone would not support that, because pids are reused: a dead run's pid
can be a live unrelated process, which would refuse forever, or -- worse --
a live run's pid could be judged dead. The start time disambiguates, and it
is read from ``ps``, so this module stays dependency-free (``psutil`` is not
a dependency) and works on macOS and Linux alike.

**Hostname is recorded but never used to judge.** A pid is only meaningful
on the machine that issued it, so a lockfile from another host cannot be
liveness-checked here at all. Rather than guess, a foreign-host lock is
treated as live and refused, and the hostname is in the message so the
operator knows where to look. Study directories on shared storage are the
case this protects.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

#: The suffix appended to a run directory's own name to spell its lockfile.
#:
#: The lockfile lives *beside* the run directory rather than inside it, so
#: that acquiring the lock is one atomic ``O_EXCL`` create that does not
#: require the directory to exist yet -- and so that the discard path
#: (``rmtree(run_dir)``) cannot delete the lock out from under its holder.
RUN_LOCK_SUFFIX = ".lock"

#: The persisted keys of a lockfile's holder record.
#:
#: Named constants because they are a persisted format: a lockfile written
#: by one version is read by another after a crash, and deriving these from
#: field names would let a rename silently orphan every existing lockfile.
LOCK_PID_KEY = "pid"
LOCK_STARTED_AT_KEY = "started_at"
LOCK_HOSTNAME_KEY = "hostname"


class RunDirectoryLockedError(RuntimeError):
    """Another live process is already driving this run directory.

    Raised rather than waited on: a study run is long and paid, so a second
    invocation blocking silently would look like a slow run rather than the
    operator mistake it is. The caller re-raises this as the refusal its
    own layer speaks (``StageError`` in the study stages).
    """


@dataclass(frozen=True, slots=True)
class LockHolder:
    """Who a lockfile says is driving the run directory.

    ``started_at`` is the holder pid's process start time exactly as the
    platform's ``ps`` prints it. It is compared as an opaque string and
    never parsed: the only question asked of it is whether the pid alive
    *now* is the same process that took the lock, and string equality of
    two reads from one ``ps`` answers that without a date format ever
    entering the trust boundary.
    """

    pid: int
    started_at: str
    hostname: str

    def describe(self) -> str:
        """The holder, as a refusal names it."""
        return (
            f"pid {self.pid} on {self.hostname!r} "
            f"started at {self.started_at!r}"
        )


def _process_start_time(pid: int) -> str | None:
    """The start time of ``pid``, or ``None`` if there is no such process.

    ``ps -o lstart=`` is the portable spelling across macOS and Linux, and
    it fails nonzero for a pid that does not exist -- which is exactly the
    liveness question, answered without a second syscall that could race
    against it.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable and argv
            ["ps", "-o", "lstart=", "-p", str(pid)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=_PS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        # ``ps`` is missing, unrunnable, or wedged. The safe answer is that
        # the holder may well be alive: refusing a live run costs an
        # operator a message, while declaring a live holder dead is the
        # double-drive this module exists to prevent.
        return _UNVERIFIABLE_START_TIME
    if completed.returncode != 0:
        return None
    start_time = completed.stdout.strip()
    # A zero exit with no output is not evidence of death, so it is not
    # treated as such.
    return start_time or _UNVERIFIABLE_START_TIME


#: How long to wait for ``ps`` before treating liveness as unverifiable.
_PS_TIMEOUT_SECONDS = 10.0

#: The stand-in start time recorded when liveness cannot be established.
#:
#: It never equals a real ``ps`` reading, so a holder whose liveness could
#: not be verified is treated as live and refused rather than cleared.
_UNVERIFIABLE_START_TIME = "\x00unverifiable"


def _holder_is_live(holder: LockHolder) -> bool:
    """Whether ``holder`` names a process running right now.

    A pid is only meaningful on the host that issued it, so a lockfile from
    elsewhere is treated as live: this process cannot see that machine's
    process table, and clearing a lock it cannot check would be a guess.
    """
    if holder.hostname != _hostname():
        return True
    start_time = _process_start_time(holder.pid)
    if start_time is None:
        return False
    # Pid reuse is the reason this is not simply "the pid exists": a dead
    # run's pid may since have been handed to an unrelated live process,
    # which would otherwise block this directory forever.
    return start_time == holder.started_at


def _hostname() -> str:
    return socket.gethostname()


def _self_holder() -> LockHolder:
    pid = os.getpid()
    start_time = _process_start_time(pid)
    return LockHolder(
        pid=pid,
        # This process is by definition alive, so a missing reading is a
        # ``ps`` failure rather than death. Recording the unverifiable
        # marker keeps the invariant that a live holder is never cleared.
        started_at=start_time or _UNVERIFIABLE_START_TIME,
        hostname=_hostname(),
    )


def run_lock_path(run_dir: Path) -> Path:
    """Where the lockfile for ``run_dir`` lives."""
    return run_dir.parent / f"{run_dir.name}{RUN_LOCK_SUFFIX}"


def _read_holder(lock_path: Path) -> LockHolder | None:
    """The holder a lockfile records, or ``None`` if it does not say.

    An unreadable, truncated, or malformed lockfile is a file whose holder
    cannot be checked for liveness. That is treated the same as a foreign
    host -- unverifiable, therefore live -- so a corrupt lockfile refuses
    loudly instead of being cleared into the double-drive it was written to
    prevent.
    """
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    pid = raw.get(LOCK_PID_KEY)
    started_at = raw.get(LOCK_STARTED_AT_KEY)
    hostname = raw.get(LOCK_HOSTNAME_KEY)
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(started_at, str)
        or not isinstance(hostname, str)
    ):
        return None
    return LockHolder(pid=pid, started_at=started_at, hostname=hostname)


def _write_lock(lock_path: Path, holder: LockHolder) -> bool:
    """Create the lockfile exclusively, returning whether we now hold it.

    ``O_CREAT | O_EXCL`` is the whole interlock: the kernel decides which
    of two racing processes creates the file, so there is no window between
    a check and a create for the other to slip through.
    """
    try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                LOCK_PID_KEY: holder.pid,
                LOCK_STARTED_AT_KEY: holder.started_at,
                LOCK_HOSTNAME_KEY: holder.hostname,
            },
            handle,
        )
    return True


@contextmanager
def run_directory_lock(run_dir: Path) -> Iterator[Path]:
    """Hold an exclusive lock over ``run_dir`` for the body's duration.

    Raises :class:`RunDirectoryLockedError` when a live process already
    holds it. A dead holder's lockfile is crash residue: it is removed and
    the lock retaken, because the question of whether the *directory* is
    reusable belongs to the run's own artifacts, not to a file its process
    never got to delete.

    The lock is released on the way out either way, so a failing run does
    not leave residue that a later invocation has to reason about.
    """
    lock_path = run_lock_path(run_dir)
    holder = _self_holder()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not _write_lock(lock_path, holder):
        existing = _read_holder(lock_path)
        if existing is None or _holder_is_live(existing):
            raise RunDirectoryLockedError(_refusal(lock_path, existing))
        # Crash residue from a process that is gone. Removing it can race
        # another recovering invocation, so the retake is the same atomic
        # create -- whoever wins, wins, and the loser refuses below rather
        # than proceeding alongside it.
        lock_path.unlink(missing_ok=True)
        if not _write_lock(lock_path, holder):
            raise RunDirectoryLockedError(
                _refusal(lock_path, _read_holder(lock_path))
            )
    try:
        yield lock_path
    finally:
        # Only ours to remove: we created this path exclusively and no
        # other holder can have taken it while we held it.
        lock_path.unlink(missing_ok=True)


def _refusal(lock_path: Path, holder: LockHolder | None) -> str:
    """The message a refused acquisition raises.

    It names the holder and the lock path because the recovery is the
    operator's: they are the one who knows whether the other process is a
    run they meant to start.
    """
    who = (
        holder.describe()
        if holder is not None
        else "a holder this lockfile does not readably record"
    )
    return (
        f"the run directory {lock_path.parent / lock_path.stem} is already "
        f"being driven by {who}. Two processes on one run directory "
        f"interleave their effects and corrupt the run's ledger, so this "
        f"invocation refuses rather than joining it. Wait for that run to "
        f"finish, or -- if it is gone and this lock is residue its process "
        f"never removed -- delete {lock_path}"
    )
