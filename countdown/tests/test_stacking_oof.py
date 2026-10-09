"""Phase 5.3: no row's meta-feature comes from a base trained on that row (or its capture)."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.ensemble.stacking import group_folds, oof_predict


def test_no_row_is_predicted_by_a_model_that_saw_it_or_its_group():
    rng = np.random.default_rng(0)
    n, c = 300, 3
    rows = np.arange(1000, 1000 + n)                  # global ids, not positions
    groups = np.array([f"cap{i // 10}" for i in range(n)], dtype=object)
    y = rng.integers(0, c, n)
    seen: list[tuple[set, set]] = []

    def fit_predict(fit_rows, held_rows):
        fit, held = set(fit_rows.tolist()), set(held_rows.tolist())
        assert not fit & held
        gid = {r: groups[r - 1000] for r in rows}
        assert not {gid[r] for r in fit} & {gid[r] for r in held}
        seen.append((fit, held))
        return np.full((len(held_rows), c), 1.0 / c)

    oof = oof_predict(fit_predict, rows, y, groups, c, n_folds=5)
    assert len(seen) == 5
    held_all = [r for _, h in seen for r in h]
    assert sorted(held_all) == rows.tolist()            # every row exactly once
    assert (oof.fold >= 0).all() and not np.isnan(oof.proba).any()


def test_fewer_groups_than_folds_shrinks_k_and_one_group_refuses():
    y = np.array([0, 1, 0, 1, 0, 1])
    folds = group_folds(y, np.array(["a", "a", "b", "b", "c", "c"]), n_folds=5)
    assert len(folds) == 3
    with pytest.raises(ValueError):
        group_folds(y, np.array(["a"] * 6), n_folds=5)
    # MobileApp shape: 3 captures per class -> 3 folds, every class in every fold
    y3 = np.repeat(np.arange(4), 3)
    folds = group_folds(y3, np.arange(12).astype(object), n_folds=5)
    assert len(folds) == 3 and all(len(set(y3[b])) == 4 for _, b in folds)


def test_wrong_shape_from_a_member_is_an_error():
    y = np.array([0, 1] * 10)
    g = np.array([f"g{i}" for i in range(20)], dtype=object)
    with pytest.raises(ValueError, match="expected"):
        oof_predict(lambda a, b: np.zeros((len(b), 5)), np.arange(20), y, g, 2, n_folds=2)
