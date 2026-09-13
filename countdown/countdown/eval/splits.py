"""Leakage-safe splitting.

The trap this module exists for: ISCX captures hold thousands of correlated flows from one
session, one client and one time window.  A random flow-level split puts near-duplicates of
the same conversation on both sides, and the reported score measures memorisation.  Much of
the published work on these datasets splits that way, which is why the honest numbers here
read lower -- see ``phases/PHASE-3.md`` section 9.

So the default is group-aware: every sample sharing a group id lands on one side.

The subtler trap, and the reason for :func:`assert_groupable`: a group key can be *vacuous*.
Grouping CSTNET by ``source_file`` produces 46,372 groups for 46,372 samples, because one
pcap is one flow there -- a random split wearing the word "grouped".  It looks correct at
every call site.  ``group_split`` refuses it unless the caller says the dataset genuinely
has no grouping to offer.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from countdown.config import get_logger

log = get_logger(__name__)


def assert_groupable(groups: Sequence[Any], allow_vacuous: bool = False) -> int:
    """Check a grouping actually groups.  Returns the number of distinct groups."""
    groups = np.asarray(groups, dtype=object)
    n_groups = len(set(groups.tolist()))
    if n_groups == len(groups) and not allow_vacuous:
        raise ValueError(
            f"vacuous grouping: {n_groups} groups for {len(groups)} samples, so a "
            f"'group-aware' split is exactly a random split. Pick a group key that "
            f"actually groups (see group_keys: in configs/datasets.yaml), or pass "
            f"allow_vacuous=True if the dataset genuinely has one sample per capture."
        )
    if n_groups < 2:
        raise ValueError(f"cannot split on {n_groups} group(s): everything is one side")
    return n_groups


def group_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    test_size: float = 0.2,
    seed: int = 42,
    allow_vacuous: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split into ``(train_idx, test_idx)`` with no group on both sides.

    Uses ``StratifiedGroupKFold`` so the class balance is held as close as the grouping
    permits -- with whole captures moving at once, it cannot be exact.  ``X`` is accepted
    for signature symmetry with sklearn and is not otherwise used.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    if len(y) != len(groups):
        raise ValueError(f"y has {len(y)} rows but groups has {len(groups)}")
    assert_groupable(groups, allow_vacuous=allow_vacuous)

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    n_splits = max(2, int(round(1.0 / test_size)))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    leaked = set(groups[train_idx].tolist()) & set(groups[test_idx].tolist())
    if leaked:  # pragma: no cover - StratifiedGroupKFold guarantees this
        raise AssertionError(f"group leak across the split: {sorted(leaked)[:5]}")
    return train_idx, test_idx


def group_split_three(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    allow_vacuous: bool = False,
) -> dict[str, np.ndarray]:
    """``{"train", "val", "test"}`` indices, group-disjoint throughout.

    The val fold is carved out of train by the same rule, so early stopping never sees a
    capture that the test fold also contains.
    """
    train_idx, test_idx = group_split(
        X, y, groups, test_size=test_size, seed=seed, allow_vacuous=allow_vacuous
    )
    if val_size <= 0:
        return {"train": train_idx, "val": np.array([], dtype=np.int64), "test": test_idx}

    groups = np.asarray(groups, dtype=object)
    inner_frac = val_size / (1.0 - test_size)
    try:
        sub_train, sub_val = group_split(
            None, np.asarray(y)[train_idx], groups[train_idx],
            test_size=inner_frac, seed=seed + 1, allow_vacuous=allow_vacuous,
        )
    except ValueError as exc:  # too few groups left to carve a val fold
        log.warning("[splits] no val fold (%s); training without early stopping", exc)
        return {"train": train_idx, "val": np.array([], dtype=np.int64), "test": test_idx}
    return {
        "train": train_idx[sub_train],
        "val": train_idx[sub_val],
        "test": test_idx,
    }


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain stratified split -- **optimistic**, offered only for quick looks.

    On any dataset where one capture yields many flows this leaks correlated near-duplicates
    into the test set.  Never report a headline number from it.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    log.warning(
        "[splits] stratified_split is leakage-prone on multi-flow captures; its scores "
        "are optimistic and must not be reported as headline results"
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    return np.sort(train_idx), np.sort(test_idx)


def split_report(
    y: np.ndarray,
    groups: Sequence[Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """What the split actually did -- recorded in every run so it can be audited.

    ``classes_absent_from_test`` is the important field.  A group-aware split on a
    class with few groups can legitimately leave that class out of the test fold, and a
    macro-F1 that quietly averaged over a missing class is not a result.
    """
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    names = list(class_names) if class_names is not None else [str(i) for i in range(n_classes)]

    present_test = set(np.unique(y[test_idx]).tolist())
    present_train = set(np.unique(y[train_idx]).tolist())
    absent_test = [names[i] for i in range(n_classes) if i not in present_test]
    absent_train = [names[i] for i in range(n_classes) if i not in present_train]

    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_groups_train": len(set(groups[train_idx].tolist())),
        "n_groups_test": len(set(groups[test_idx].tolist())),
        "groups_disjoint": not (
            set(groups[train_idx].tolist()) & set(groups[test_idx].tolist())
        ),
        "classes_absent_from_test": absent_test,
        "classes_absent_from_train": absent_train,
        "train_class_counts": {
            names[i]: int(c) for i, c in zip(*np.unique(y[train_idx], return_counts=True))
        },
        "test_class_counts": {
            names[i]: int(c) for i, c in zip(*np.unique(y[test_idx], return_counts=True))
        },
    }
