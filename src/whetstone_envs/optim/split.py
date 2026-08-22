"""The train/val partition every optimizer with that concept must state.

MIPROv2 bootstraps demonstrations from a trainset and scores each search
trial on a valset; GEPA reflects over a trainset and selects its Pareto
frontier on a valset. In both cases an improvement measured on tasks the
optimizer already trained on cannot be distinguished from demonstration or
instruction memorization -- so while the setup is being debugged, both
optimizers take an explicit, disjoint partition of the internal split
rather than defaulting to one.

The partition is derived from two integers so it is reproducible from the
run spec alone: :func:`partition_internal_split` takes the first
``train_size`` task hashes as the trainset and the next ``val_size`` as the
valset. The internal split's task-hash tuple is already deterministic, so
the same spec always names the same two sets, and the run's durable record
carries both tuples for the audits to read.

COPRO, the null arms, and Codex-direct have no train/val concept and are
deliberately not routed through here.
"""

from __future__ import annotations

#: The optimizers that bootstrap or reflect on one set of tasks and score
#: on another, and so must state an explicit train/val partition. COPRO,
#: the nulls, and Codex-direct have no such concept, so supplying a split
#: for them is refused rather than ignored.
TRAIN_VAL_OPTIMIZERS = ("gepa", "miprov2")


def partition_internal_split(
    task_hashes: tuple[str, ...], *, train_size: int, val_size: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The ``(trainset, valset)`` this spec names over ``task_hashes``.

    A deterministic prefix/suffix cut: the trainset is the first
    ``train_size`` hashes and the valset the next ``val_size``. Taking
    adjacent slices of an already-deterministic tuple is what makes the
    partition reproducible from the two integers, with no seed or shuffle
    of its own to record.
    """
    if train_size < 1:
        raise ValueError("train_size must be at least 1")
    if val_size < 1:
        raise ValueError("val_size must be at least 1")
    if train_size + val_size > len(task_hashes):
        raise ValueError(
            f"train_size {train_size} + val_size {val_size} exceeds the "
            f"internal split of {len(task_hashes)}"
        )
    trainset = task_hashes[:train_size]
    valset = task_hashes[train_size : train_size + val_size]
    return trainset, valset


def require_disjoint_split(
    *,
    trainset_task_hashes: tuple[str, ...],
    valset_task_hashes: tuple[str, ...],
    task_hashes: tuple[str, ...],
    optimizer: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Refuse a partition the run could not honestly report on.

    Checked against the engine's own task hashes rather than trusted from
    the caller: the control builders are reached by the CLI, a study arm,
    and tests alike, so this is the one place that proves the sets the
    optimizer will actually train and score on are disjoint and inside the
    internal split. Refusing here keeps the failure outside the durable run
    boundary, so an unrunnable partition leaves no run directory behind.
    """
    if not trainset_task_hashes:
        raise ValueError(f"{optimizer} trainset must be non-empty")
    if not valset_task_hashes:
        raise ValueError(f"{optimizer} valset must be non-empty")
    internal = set(task_hashes)
    unknown_train = set(trainset_task_hashes) - internal
    if unknown_train:
        raise ValueError(
            f"{optimizer} trainset must be a subset of the internal split"
        )
    unknown_val = set(valset_task_hashes) - internal
    if unknown_val:
        raise ValueError(
            f"{optimizer} valset must be a subset of the internal split"
        )
    overlap = set(trainset_task_hashes) & set(valset_task_hashes)
    if overlap:
        raise ValueError(
            f"{optimizer} trainset and valset must be disjoint; "
            f"{len(overlap)} task(s) appear in both"
        )
    return trainset_task_hashes, valset_task_hashes


__all__ = [
    "TRAIN_VAL_OPTIMIZERS",
    "partition_internal_split",
    "require_disjoint_split",
]
