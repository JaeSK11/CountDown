import numpy as np
import pytest

from countdown.eval.tiers import load_tiers, tier_index, tier_metrics, to_tier_proba
from countdown.schema import TRAFFIC_TYPES


def test_shipped_tiers_cover_every_traffic_type_once():
    tiers = load_tiers()
    flat = [c for classes in tiers.values() for c in classes]
    assert sorted(flat) == sorted(TRAFFIC_TYPES)
    assert list(tiers) == ["migrate_first", "migrate_next", "low_priority"]


def test_tier_proba_sums_within_tier_not_argmax():
    tiers = {"high": ["a", "b"], "low": ["c"]}
    # argmax is "c", but "a" + "b" outweigh it
    proba = np.array([[0.35, 0.30, 0.35]])
    out = to_tier_proba(proba, ["a", "b", "c"], tiers)
    np.testing.assert_allclose(out, [[0.65, 0.35]])


def test_within_tier_confusion_is_free():
    tiers = {"high": ["a", "b"], "low": ["c"]}
    y = np.array([0, 1, 2])
    proba = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # a<->b swapped
    m = tier_metrics(y, proba, ["a", "b", "c"], tiers)
    assert m["macro_f1"] == 1.0
    assert m["class_names"] == ["high", "low"]


def test_unmapped_class_is_an_error():
    with pytest.raises(ValueError, match="no tier"):
        tier_index(["a", "z"], {"high": ["a"]})


def test_class_in_two_tiers_is_an_error(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("tiers:\n  x: [a]\n  y: [a]\n")
    with pytest.raises(ValueError, match="more than one tier"):
        load_tiers(p)
