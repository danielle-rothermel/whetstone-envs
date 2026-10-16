from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class DurableRunError(RuntimeError):
    """An operation failed after its durable run directory was created."""

    def __init__(self, directory: Path, cause: Exception) -> None:
        super().__init__(str(cause))
        self.directory = directory
        self.cause = cause


@contextmanager
def durable_run_boundary(directory: Path) -> Iterator[None]:
    try:
        yield
    except DurableRunError:
        raise
    except Exception as error:
        raise DurableRunError(directory, error) from error


def _git_root(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return None


def repository_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        root = _git_root(start)
        if root is not None and root not in roots:
            roots.append(root)
    return tuple(roots)


def validate_output_root(path: Path) -> Path:
    """Resolve and reject a run directory inside any detected repository."""
    resolved = path.resolve()
    target_repository = _git_root(resolved)
    if target_repository is not None or any(
        resolved.is_relative_to(root) for root in repository_roots()
    ):
        raise ValueError("run artifacts must not be written inside the repo")
    return resolved


def prepare_output_root(path: Path) -> Path:
    resolved = validate_output_root(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


__all__ = [
    "DurableRunError",
    "durable_run_boundary",
    "prepare_output_root",
    "repository_roots",
    "validate_output_root",
]
