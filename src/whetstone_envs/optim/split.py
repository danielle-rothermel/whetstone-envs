from __future__ import annotations

#: The two optimizers with a train/val concept, named individually because
#: they do not share a rule: GEPA must partition the internal split
#: exactly, MIPROv2 need only fit inside it.
GEPA_OPTIMIZER = "gepa"
MIPROV2_OPTIMIZER = "miprov2"

#: The optimizers that bootstrap or reflect on one set of tasks and score
#: on another, and so must state an explicit train/val partition. COPRO,
#: the nulls, and Codex-direct have no such concept, so supplying a split
#: for them is refused rather than ignored.
TRAIN_VAL_OPTIMIZERS = (GEPA_OPTIMIZER, MIPROV2_OPTIMIZER)

#: COPRO itself, and the control whose search *is* COPRO's.


#: The smallest breadth upstream ``CoproControl`` accepts. A single draft
#: per step leaves nothing to select between, so upstream refuses it.
MIN_COPRO_BREADTH = 2

#: The smallest depth a search can have: one round of proposals.
MIN_COPRO_DEPTH = 1


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
    "GEPA_OPTIMIZER",
    "MIN_COPRO_BREADTH",
    "MIN_COPRO_DEPTH",
    "MIPROV2_OPTIMIZER",
    "TRAIN_VAL_OPTIMIZERS",
    "partition_internal_split",
    "require_disjoint_split",
]
