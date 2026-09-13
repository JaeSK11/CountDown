"""Group-aware splitting, and the guard against a grouping that only looks group-aware."""

from __future__ import annotations

import numpy as np
import pytest

from qsentinel.eval.splits import (
    assert_groupable,
    group_split,
    group_split_three,
    split_report,
    stratified_split,
)


def _dataset(n_groups: int = 20, per_group: int = 30, n_classes: int = 4):
    """Flows clustered by capture, the way ISCX actually is."""
    y, groups = [], []
    for g in range(n_groups):
        cls = g % n_classes
        y.extend([cls] * per_group)
        groups.extend([f"capture_{g}.pcap"] * per_group)
    y = np.asarray(y)
    return np.zeros((len(y), 3)), y, np.asarray(groups, dtype=object)


# --- the core guarantee ------------------------------------------------------------------

def test_no_group_appears_on_both_sides():
    X, y, g = _dataset()
    tr, te = group_split(X, y, g, test_size=0.2, seed=42)
    assert set(g[tr]).isdisjoint(set(g[te]))
    assert len(tr) + len(te) == len(y)
    assert len(np.intersect1d(tr, te)) == 0


def test_split_is_deterministic_for_a_seed():
    X, y, g = _dataset()
    a = group_split(X, y, g, test_size=0.2, seed=7)
    b = group_split(X, y, g, test_size=0.2, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = group_split(X, y, g, test_size=0.2, seed=8)
    assert not np.array_equal(a[1], c[1])


def test_every_class_reaches_both_sides_when_groups_allow():
    X, y, g = _dataset(n_groups=40, per_group=10, n_classes=4)
    tr, te = group_split(X, y, g, test_size=0.25, seed=42)
    assert set(np.unique(y[tr])) == set(np.unique(y[te])) == {0, 1, 2, 3}


# --- the vacuous-grouping guard ----------------------------------------------------------

def test_vacuous_grouping_is_refused():
    """One group per sample is a random split; CSTNET by source_file is exactly this."""
    X, y, _ = _dataset()
    unique = np.arange(len(y), dtype=object)
    with pytest.raises(ValueError, match="vacuous grouping"):
        group_split(X, y, unique, test_size=0.2, seed=42)


def test_vacuous_grouping_is_allowed_when_declared():
    X, y, _ = _dataset(n_groups=8, per_group=20, n_classes=4)
    unique = np.arange(len(y), dtype=object)
    tr, te = group_split(X, y, unique, test_size=0.2, seed=42, allow_vacuous=True)
    assert len(tr) and len(te)


def test_assert_groupable_counts_groups_and_rejects_a_single_group():
    assert assert_groupable(np.asarray(["a", "a", "b", "b"], dtype=object)) == 2
    with pytest.raises(ValueError, match="one side"):
        assert_groupable(np.asarray(["a"] * 5, dtype=object))


# --- three-way split ---------------------------------------------------------------------

def test_val_fold_is_group_disjoint_from_both_train_and_test():
    X, y, g = _dataset(n_groups=40, per_group=10)
    parts = group_split_three(X, y, g, test_size=0.2, val_size=0.1, seed=42)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    assert len(va) > 0
    assert set(g[va]).isdisjoint(set(g[tr]))
    assert set(g[va]).isdisjoint(set(g[te]))
    assert len(tr) + len(va) + len(te) == len(y)


def test_val_fold_is_skipped_rather_than_crashing_when_groups_run_out():
    X, y, g = _dataset(n_groups=4, per_group=10, n_classes=2)
    parts = group_split_three(X, y, g, test_size=0.25, val_size=0.25, seed=42)
    assert len(parts["train"]) and len(parts["test"])


# --- audit -------------------------------------------------------------------------------

def test_split_report_names_classes_missing_from_test():
    X, y, g = _dataset(n_groups=20, per_group=10, n_classes=4)
    tr, te = group_split(X, y, g, test_size=0.2, seed=42)
    rep = split_report(y, g, tr, te, n_classes=6, class_names=list("abcdef"))
    assert rep["groups_disjoint"] is True
    assert rep["n_groups_train"] + rep["n_groups_test"] == 20
    # classes 4 and 5 exist in the space but not in the data at all
    assert set(rep["classes_absent_from_test"]) >= {"e", "f"}


def test_stratified_split_still_works_but_is_labelled_optimistic():
    X, y, _ = _dataset()
    tr, te = stratified_split(X, y, test_size=0.2, seed=42)
    assert len(np.intersect1d(tr, te)) == 0
    assert stratified_split.__doc__ and "optimistic" in stratified_split.__doc__.lower()
