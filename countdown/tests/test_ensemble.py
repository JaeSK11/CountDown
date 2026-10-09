"""Phase 5.4-5.6: combiners, the Ensemble object, the window view, the ablation table."""

from __future__ import annotations

import numpy as np

from countdown.ensemble import ArrayView, Ensemble, ExpertData, WindowView
from countdown.ensemble.combiner import SoftVote, Stacker
from countdown.eval.ablation import run_ablation, to_markdown
from countdown.eval.splits import group_split
from countdown.schema import LabelSpace

FAST = {"n_estimators": 40, "num_leaves": 7}


def _toy(n=1500, seed=0):
    """Two members that each see half of the signal: A separates class 0, B separates 1 from 2."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    a = np.c_[(y == 0) + rng.normal(0, 0.6, n), rng.normal(size=(n, 3))].astype(np.float32)
    b = np.c_[(y == 2).astype(float) - (y == 1) + rng.normal(0, 0.6, n), rng.normal(size=(n, 3))].astype(np.float32)
    groups = np.array([f"g{i // 5}" for i in range(n)], dtype=object)
    space = LabelSpace.from_names("toy", ["x", "y", "z"])
    tr, te = group_split(None, y, groups, test_size=0.25, seed=0)
    data = ExpertData(expert="T", dataset="toy", space=space, y=y, groups=groups,
                      group_key="g", keys=np.arange(n).astype(object),
                      l7_hint=np.array([None] * n, dtype=object), train_rows=tr, test_rows=te)
    for name, X in (("a", a), ("b", b)):
        data.views[name] = ArrayView(X=X, name=name, model="flow_gbdt", space=space, y=y,
                                     groups=groups, params=FAST)
    return data


def _f1(p, y):
    from countdown.eval.metrics import compute_metrics
    return compute_metrics(y, p.argmax(1), p.shape[1])["macro_f1"]


def test_ensemble_beats_the_best_member_on_complementary_members(tmp_path):
    data = _toy()
    ens = Ensemble(data, combiner="stack_logreg", n_folds=4).fit(cache_dir=tmp_path)
    y_te = data.y[data.test_rows]
    singles = [_f1(ens.preds[m].test, y_te) for m in ("a", "b")]
    assert _f1(ens.test_proba(), y_te) > max(singles) + 0.05
    # cached predictions are reused, not recomputed
    again = Ensemble(data, combiner="vote", n_folds=4).fit(cache_dir=tmp_path)
    np.testing.assert_array_equal(again.preds["a"].test, ens.preds["a"].test)
    # explanation and persistence
    rows = data.test_rows[:3]
    exp = ens.explain(ens.predict_rows(rows))
    assert set(exp[0]["per_member"]) == {"a", "b"} and exp[0]["label"] in {"x", "y", "z"}
    assert ens.save(tmp_path / "ens.joblib").exists()


def test_ablation_table_has_every_variant_and_flags_the_winner():
    data = _toy(n=900, seed=1)
    ens = Ensemble(data, n_folds=3).fit()
    tr, te = data.train_rows, data.test_rows
    res = run_ablation({m: ens.preds[m].oof for m in ens.members},
                       {m: ens.preds[m].test for m in ens.members},
                       data.y[tr], data.y[te], 3, fold=ens.preds["a"].fold)
    kinds = {(r["kind"], r["combiner"], r["calibration"]) for r in res["rows"]}
    assert ("ensemble", "stack_lgbm", "temperature") in kinds
    assert ("single", "-", "none") in kinds
    assert res["ensemble_beats_best_single"] is True
    assert res["primary"]["combiner"] == "stack_logreg" and res["selected"]["cv_macro_f1"] > 0
    assert "| ensemble |" in to_markdown(res)


def test_combiners_keep_label_space_columns_and_handle_a_missing_member():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)                          # class 2 never occurs
    good = np.c_[np.eye(3)[y][:, :2] * 0.8 + 0.1, np.zeros(200)]
    absent = np.full_like(good, np.nan)
    absent[:100] = good[:100]
    for comb in (SoftVote(), Stacker("logreg"), Stacker("lgbm", n_estimators=20)):
        p = comb.fit([good, absent], y).predict_proba([good, absent])
        assert p.shape == (200, 3) and np.allclose(p.sum(1), 1.0)
        assert (p[:, 2] < 1e-6).all()


def test_window_view_averages_window_probabilities_per_flow():
    rng = np.random.default_rng(3)
    n = 200
    y = rng.integers(0, 2, n)
    parent = np.repeat(np.arange(n), 3)                  # three windows per flow
    Xw = (y[parent][:, None] + rng.normal(0, 0.3, (3 * n, 2))).astype(np.float32)
    space = LabelSpace.from_names("toy", ["p", "q"])
    view = WindowView(Xw=Xw, parent=parent, name="w", model="flow_gbdt", space=space, y=y,
                      groups=np.arange(n).astype(object), params=FAST)
    m = view.fit(np.arange(150))
    rows = np.arange(150, 200)
    p = view.predict(m, rows)
    direct = m.predict_proba(Xw[np.isin(parent, rows)]).reshape(50, 3, 2).mean(1)
    np.testing.assert_allclose(p, direct)
    assert _f1(p, y[rows]) > 0.9


def test_a_one_member_ensemble_is_its_calibrated_member():
    data = _toy(n=600, seed=4)
    ens = Ensemble(data, members=["a"], n_folds=3).fit()
    expected = ens.calibrators["a"].transform(ens.preds["a"].test)
    np.testing.assert_allclose(ens.test_proba(), expected)
