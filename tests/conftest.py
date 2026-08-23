from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from whetstone_envs.instances import make_instance
from whetstone_envs.pools import TaskPool

if TYPE_CHECKING:
    from collections.abc import Callable

    from whetstone_envs.instances import Instance


#: The real-Codex opt-in variable, spelled here rather than imported.
#: This conftest loads for every suite, including installs without the
#: optional ``optim`` extra, so it must not import
#: ``whetstone_envs.optim.codex``. The two spellings are pinned equal by
#: ``tests/optim/test_codex.py``.
ALLOW_REAL_CODEX_ENV = "WHETSTONE_ENVS_ALLOW_REAL_CODEX"

#: The tripwire the suite arms for its whole session, spelled here for the
#: same reason as the variable above. Pinned equal to the owning module's
#: constant by ``tests/optim/test_codex.py``.
FORBID_REAL_CODEX_ENV = "WHETSTONE_ENVS_FORBID_REAL_CODEX"

#: The real-CLI ladder's own opt-in, spelled here for the same reason.
#: Pinned equal to ``tests/real_codex/conftest.py``'s by
#: ``tests/optim/test_codex.py``.
REAL_CODEX_LADDER_ENV = "WHETSTONE_ENVS_REAL_CODEX"

#: The ``config`` stash key through which the ladder -- and only the
#: ladder -- claims its session. Set by ``tests/real_codex/conftest.py``
#: during collection, read by :func:`_no_real_codex_opt_in` afterwards.
#:
#: A stash key rather than an import: this conftest loads for every suite,
#: including installs without the optional ``optim`` extra, and
#: ``tests/real_codex/conftest.py`` imports the optimizer stack. Owning
#: the key here and letting the ladder write to it keeps the dependency
#: pointing the one direction that always resolves.
REAL_CODEX_LADDER_SESSION = pytest.StashKey[bool]()


@pytest.fixture(scope="session", autouse=True)
def _no_real_codex_opt_in(request: pytest.FixtureRequest) -> None:
    """No ordinary test may spawn the real, billed Codex CLI.

    Two mechanisms, because clearing the opt-in is not enough on its own.

    Clearing :data:`ALLOW_REAL_CODEX_ENV` means the suite does not inherit
    an opt-in from the developer's shell, and asserting it was unset first
    means a run that *would* have spent is reported rather than silently
    corrected.

    But the opt-in is process state, and ``monkeypatch.setenv`` is the
    ordinary way to test that a gate lifts once it is satisfied. Such a
    test lifts the real gate for its duration -- which is how an
    authorization test in this suite once reached the real CLI, by way of
    a session probe that runs after the opt-in is satisfied. So this also
    *sets* :data:`FORBID_REAL_CODEX_ENV`, which
    ``refuse_unauthorized_real_codex`` honours above every other input:
    for the rest of the session a test may monkeypatch the allow variable
    and still cannot reach a real session. Scripted runs through a
    ``CodexTestSeam`` are unaffected, since they reach no real CLI.

    Session-scoped and autouse rather than per-test: both variables are
    process state, and a per-test fixture would leave the gaps between
    tests unarmed.

    **The one exception is the real-CLI ladder**, whose whole purpose is
    to drive the paid CLI. This fixture does not decide that exception;
    it defers to ``tests/real_codex/conftest.py``, which claims the
    session through :data:`REAL_CODEX_LADDER_SESSION` only when the
    ladder's own opt-in (:data:`REAL_CODEX_LADDER_ENV`) is set *and* the
    session actually collected ladder items -- which the ``real_codex``
    marker deselects by default. The claim is made during collection,
    which pytest runs before any session fixture, so by the time this
    reads the stash the ladder has already had its say.

    Deferring rather than reading the variable here is what keeps the
    exception narrow. An exported ``WHETSTONE_ENVS_REAL_CODEX`` alone
    proves nothing about what the session is going to run; the ladder's
    own hook can additionally require that a ladder item was collected,
    and it is the file that owns the ladder. Every ordinary session --
    including one launched from a shell carrying a stray export -- takes
    the branch below and arms the tripwire exactly as before, and the
    forbid gate in ``whetstone_envs.optim.codex`` remains the single
    point of refusal either way.
    """
    if request.config.stash.get(REAL_CODEX_LADDER_SESSION, False):
        return
    present = os.environ.pop(ALLOW_REAL_CODEX_ENV, None)
    os.environ[FORBID_REAL_CODEX_ENV] = "1"
    if present is not None:
        message = (
            f"{ALLOW_REAL_CODEX_ENV}={present!r} is set in this process. "
            "The suite must never be able to opt in to the real, billed "
            "Codex CLI; unset it before running the tests. (The real-CLI "
            f"ladder is the one exception, and it sets "
            f"{REAL_CODEX_LADDER_ENV}=1 as well.)"
        )
        raise RuntimeError(message)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run opt-in integration tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="requires --run-integration outside CI",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def _synthetic_instance(
    index: int,
    stratum: str | tuple[str, ...],
    *,
    gold: str = "yes",
) -> Instance:
    stratum_label = stratum if isinstance(stratum, str) else "/".join(stratum)
    return make_instance(
        id=f"{stratum_label}-{index}",
        seed=1000 + index,
        strata=stratum,
        prompt_inputs={"question": f"q{index}", "hint": stratum_label},
        gold=gold,
    )


@pytest.fixture
def synthetic_instance() -> Callable[..., Instance]:
    return _synthetic_instance


@pytest.fixture
def two_stratum_pool() -> TaskPool:
    """A pool with two strata of three instances each (six total)."""
    instances = [_synthetic_instance(i, "easy") for i in range(3)] + [
        _synthetic_instance(i, "hard") for i in range(3, 6)
    ]
    return TaskPool(instances)
