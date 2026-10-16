from __future__ import annotations

MIN_COPRO_BREADTH = 2

MIN_COPRO_DEPTH = 1


def partition_internal_split(
    task_hashes: tuple[str, ...], *, train_size: int, val_size: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
    "MIN_COPRO_BREADTH",
    "MIN_COPRO_DEPTH",
    "partition_internal_split",
    "require_disjoint_split",
]
