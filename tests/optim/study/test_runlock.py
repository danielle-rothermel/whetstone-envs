"""Two processes must not drive one run directory.

The defect: run ids are deterministic on arm and seed, so two
``whetstone-study run`` invocations of one stage compute the same run
directory. The only interlock was ``run_dir.exists()`` over a
``mkdir(exist_ok=True)``, and existence is a fact about the past -- two
processes launched seconds apart both saw no directory and both drove the
same run, interleaving effects and stranding an intent as ``leased``. The
effect-lease authority caught it only at terminalization, after the money
was spent.

Every check here synchronizes on state -- a lockfile that exists, a pid
that is or is not running -- and never on elapsed time. Conflict is
produced by *actually holding* the lock rather than by racing two
processes and hoping they overlap, so a refusal is evidence of the
interlock rather than of scheduling luck.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.optim.study.runlock import (
    LOCK_HOSTNAME_KEY,
    LOCK_PID_KEY,
    LOCK_STARTED_AT_KEY,
    RunDirectoryLockedError,
    run_directory_lock,
    run_lock_path,
)

if TYPE_CHECKING:
    from pathlib import Path


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "copro-seed1000"


def _write_holder(
    lock_path: Path, *, pid: int, started_at: str, hostname: str
) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                LOCK_PID_KEY: pid,
                LOCK_STARTED_AT_KEY: started_at,
                LOCK_HOSTNAME_KEY: hostname,
            }
        ),
        encoding="utf-8",
    )


def _acquire_and_release(run_dir: Path) -> None:
    """Take the lock and give it straight back.

    A single statement the refusal checks can wrap: entering at all is the
    behaviour under test, so there is nothing to do inside the body.
    """
    with run_directory_lock(run_dir):
        pass


def _this_process_start_time() -> str:
    """This pid's start time, read the same way the lock reads it."""
    completed = subprocess.run(  # noqa: S603
        ["ps", "-o", "lstart=", "-p", str(os.getpid())],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


# --------------------------------------------------------------------------
# A live holder is refused
# --------------------------------------------------------------------------


def test_a_second_acquisition_is_refused_while_the_first_holds(
    tmp_path: Path,
) -> None:
    """The incident, in one process: the second acquirer must not proceed.

    Holding the lock for real is what makes the conflict certain -- there
    is no window in which both bodies could run, so the refusal is the
    interlock rather than a scheduling coincidence.
    """
    run_dir = _run_dir(tmp_path)

    with (
        # The holder here is this very process, so it is live by
        # construction: no pid bookkeeping, and nothing to wait for.
        run_directory_lock(run_dir),
        pytest.raises(RunDirectoryLockedError) as refusal,
    ):
        _acquire_and_release(run_dir)

    message = str(refusal.value)
    # The refusal names the holder, because the operator is the one who
    # knows whether that process is a run they meant to start.
    assert str(os.getpid()) in message
    assert str(run_lock_path(run_dir)) in message
    assert str(run_dir) in message


def test_a_live_foreign_process_holding_the_lock_is_refused(
    tmp_path: Path,
) -> None:
    """A real second process, held live by state rather than by a timer.

    The child blocks on a pipe read until this test closes it, so the
    holder is provably alive at the moment of the refusal without any
    sleep standing in as evidence.
    """
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)
    # A child that reports its identity, then blocks until stdin closes.
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,sys;sys.stdout.write(str(os.getpid()));"
            "sys.stdout.flush();sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        # Reading the pid is the synchronization: the child has reached
        # its blocking read, so it is running now.
        child_pid = int(child.stdout.read(len(str(child.pid))))
        start_time = subprocess.run(  # noqa: S603
            ["ps", "-o", "lstart=", "-p", str(child_pid)],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        _write_holder(
            lock_path,
            pid=child_pid,
            started_at=start_time,
            hostname=os.uname().nodename,
        )

        with pytest.raises(RunDirectoryLockedError) as refusal:
            _acquire_and_release(run_dir)
    finally:
        if child.stdin is not None:
            child.stdin.close()
        child.wait()

    assert str(child_pid) in str(refusal.value)
    # And it refused rather than stealing: the holder's record is intact.
    assert (
        json.loads(lock_path.read_text(encoding="utf-8"))[LOCK_PID_KEY]
        == child_pid
    )


def test_a_lock_from_another_host_is_refused_rather_than_guessed(
    tmp_path: Path,
) -> None:
    """A pid means nothing off its own machine, so it is not cleared.

    Study directories can live on shared storage. Judging a foreign host's
    pid against this machine's process table would clear a live run's lock
    on a coin flip.
    """
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)
    # A pid that is certainly not running *here*, to prove the refusal
    # comes from the hostname rather than from local liveness.
    _write_holder(
        lock_path,
        pid=999_999,
        started_at="Mon Jan  1 00:00:00 2035",
        hostname="some-other-machine",
    )

    with pytest.raises(RunDirectoryLockedError) as refusal:
        _acquire_and_release(run_dir)

    assert "some-other-machine" in str(refusal.value)
    assert lock_path.is_file()


def test_an_unreadable_lockfile_is_refused_rather_than_cleared(
    tmp_path: Path,
) -> None:
    """A holder that cannot be checked is not thereby shown to be dead.

    A truncated or half-written lockfile is exactly what a crash *during*
    acquisition leaves, and it may sit beside a process that is still
    running. Clearing it would reintroduce the double-drive.
    """
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(RunDirectoryLockedError):
        _acquire_and_release(run_dir)

    assert lock_path.is_file()


# --------------------------------------------------------------------------
# A dead holder's lockfile is crash residue
# --------------------------------------------------------------------------


def test_a_dead_holders_lock_is_cleared_and_the_run_proceeds(
    tmp_path: Path,
) -> None:
    """Crash residue must not block every later invocation.

    A lock that only recorded "someone held this" would need deleting by
    hand after every crash, which trains operators to delete lockfiles
    reflexively -- and a reflexively-deleted lock protects nothing.
    """
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)
    # A real pid that has really exited: waiting on it is the
    # synchronization, so its death is a fact rather than a delay.
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait()
    _write_holder(
        lock_path,
        pid=child.pid,
        started_at="Mon Jan  1 00:00:00 2035",
        hostname=os.uname().nodename,
    )

    entered = False
    with run_directory_lock(run_dir):
        entered = True
        # The residue was replaced by this invocation's own record, not
        # merely deleted: the directory is held while the run proceeds.
        held = json.loads(lock_path.read_text(encoding="utf-8"))
        assert held[LOCK_PID_KEY] == os.getpid()

    assert entered
    assert not lock_path.exists()


def test_a_reused_pid_does_not_impersonate_the_dead_holder(
    tmp_path: Path,
) -> None:
    """Pid alone is not identity, so the start time is what decides.

    Pids are reused. If liveness were "the pid exists", a dead run whose
    pid was since handed to an unrelated live process would block its
    directory forever -- so the recorded start time must be what clears it.
    """
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)
    # This process is unambiguously alive, but the lock records a
    # *different* start time for its pid -- the signature of a dead holder
    # whose pid has since been reused.
    _write_holder(
        lock_path,
        pid=os.getpid(),
        started_at="Mon Jan  1 00:00:00 2035",
        hostname=os.uname().nodename,
    )

    with run_directory_lock(run_dir):
        held = json.loads(lock_path.read_text(encoding="utf-8"))
        assert held[LOCK_STARTED_AT_KEY] == _this_process_start_time()


# --------------------------------------------------------------------------
# The lock is released either way
# --------------------------------------------------------------------------


def test_the_lock_is_released_when_the_run_raises(tmp_path: Path) -> None:
    """A failed run must not leave residue the next invocation reasons about.

    The failure path is the common one -- a refused preflight, a provider
    error, an interrupted stage -- so if it leaked the lock, the interlock
    would degrade into the manual-deletion habit it exists to avoid.
    """
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)

    def failing_run() -> None:
        with run_directory_lock(run_dir):
            # Held while the body runs, so its absence afterwards is the
            # release rather than an acquisition that never happened.
            assert lock_path.is_file()
            raise RuntimeError("the run failed")

    with pytest.raises(RuntimeError, match="the run failed"):
        failing_run()

    assert not lock_path.exists()
    # And the directory is immediately acquirable again.
    with run_directory_lock(run_dir):
        assert lock_path.is_file()


def test_the_lock_is_released_on_normal_completion(tmp_path: Path) -> None:
    """The ordinary path leaves nothing behind either."""
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)

    with run_directory_lock(run_dir):
        assert lock_path.is_file()

    assert not lock_path.exists()


def test_the_lock_lives_beside_the_run_directory_not_inside_it(
    tmp_path: Path,
) -> None:
    """Placement is load-bearing, so it is pinned.

    Inside the directory, the discard path (``rmtree(run_dir)``) would
    delete the lock out from under its own holder, and the lock could not
    be taken before the directory existed.
    """
    run_dir = _run_dir(tmp_path)
    lock_path = run_lock_path(run_dir)

    assert lock_path.parent == run_dir.parent
    assert not lock_path.is_relative_to(run_dir)

    with run_directory_lock(run_dir):
        # Taken without the run directory existing yet: acquisition
        # precedes the run rather than depending on it.
        assert not run_dir.exists()
        assert lock_path.is_file()
