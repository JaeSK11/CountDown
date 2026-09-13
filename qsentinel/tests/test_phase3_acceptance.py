"""Phase-3 acceptance: the end-to-end contract from phases/PHASE-3.md section 8.

The fast tests run the whole harness on a synthetic dataset so the contract is checked on
every commit; the slow one runs the documented exit criterion against the real corpus.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qsentinel.config import Config
from qsentinel.eval import evaluate, group_split, split_report
from qsentinel.models import ModelRegistry, load_model
from qsentinel.schema import FiveTuple, Flow, FlowDataset, LabelSpace
from qsentinel.training import ExperimentConfig, train


def _has_iscx() -> bool:
    cfg = Config.load()
    return all((cfg.data_root / cfg.dataset(n)["root"]).exists()
               for n in ("iscxvpn", "iscxtor"))


def _synthetic_dataset(n_groups=24, per_group=25):
    """Captures whose flows differ by class, so the model has something learnable."""
    rng = np.random.default_rng(0)
    classes = ["chat", "voip", "p2p", "browsing"]
    flows = []
    for g in range(n_groups):
        cls = classes[g % len(classes)]
        scale = 100 * (classes.index(cls) + 1)
        for i in range(per_group):
            n = 8
            flows.append(Flow(
                flow_id=f"g{g}f{i}",
                five_tuple=FiveTuple("10.0.0.1", 1000 + i, "1.1.1.1", 443, "tcp"),
                timestamps=np.cumsum(rng.exponential(0.01 * (classes.index(cls) + 1), n)),
                sizes=np.clip(rng.normal(scale, 15, n), 40, 1500).astype(np.int32),
                directions=np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8),
                label_fields={"traffic_type": cls},
                meta={"source_file": f"/caps/capture_{g}.pcap"},
            ))
    return FlowDataset(flows, "synthetic", "traffic_type", config=Config.load())


# --- the section-8 contract ---------------------------------------------------------------

def test_acceptance_contract_end_to_end():
    ds = _synthetic_dataset()
    space = LabelSpace.taxonomy("traffic_type")

    X = ds.features("flow_stats")
    y = ds.labels(space=space)
    g = ds.groups("source_file")

    tr, te = group_split(X, y, g, test_size=0.25, seed=42)
    assert set(g[tr]).isdisjoint(set(g[te]))

    m = ModelRegistry.create("flow_gbdt", label_space=space, n_estimators=60)
    m.fit(X[tr], y[tr], feature_names=ds.feature_names("flow_stats"))

    rep = evaluate(m, X[te], y[te], space=space, y_train=y[tr])
    assert rep["macro_f1"] > rep["baseline_macro_f1"]
    assert m.predict_proba(X[te]).shape == (len(te), len(space)) == (len(te), 8)
    assert m.classes_ == list(space.names)


def test_a_vacuous_group_key_is_refused_by_default():
    ds = _synthetic_dataset()
    space = LabelSpace.taxonomy("traffic_type")
    y = ds.labels(space=space)
    with pytest.raises(ValueError, match="vacuous grouping"):
        group_split(None, y, np.arange(len(y), dtype=object), test_size=0.2, seed=42)


def test_training_harness_writes_a_self_describing_run(tmp_path, monkeypatch):
    ds = _synthetic_dataset()
    monkeypatch.setattr("qsentinel.data.load", lambda *a, **k: ds, raising=False)
    monkeypatch.setattr("qsentinel.data.base.load", lambda *a, **k: ds)

    cfg = ExperimentConfig(
        dataset="synthetic", target="traffic_type", model="flow_gbdt",
        test_size=0.25, val_size=0.1, params={"n_estimators": 60},
    )
    out = train(cfg, runs_dir=tmp_path)
    run_dir = out["run_dir"]

    for artifact in ("metrics.json", "config.yaml", "split.json", "model.joblib",
                     "label_space.json"):
        assert (run_dir / artifact).exists(), artifact

    blob = json.loads((run_dir / "metrics.json").read_text())
    assert blob["metrics"]["target"] == "traffic_type"
    assert blob["metrics"]["n_classes"] == 8
    assert blob["split"]["groups_disjoint"] is True
    assert blob["split"]["group_key"] == "source_file"
    assert "classes_absent_from_test" in blob["split"]

    # the saved artifact is self-sufficient: it knows its own column meaning
    reloaded = load_model(run_dir / "model.joblib")
    assert reloaded.label_space == LabelSpace.taxonomy("traffic_type")
    assert LabelSpace.from_json((run_dir / "label_space.json").read_text()) \
        == reloaded.label_space


def test_harness_rejects_a_feature_model_mismatch(monkeypatch):
    ds = _synthetic_dataset()
    # train() resolves `from qsentinel.data import load`, so the re-exported name is the
    # one that has to be patched, not just the definition in data.base.
    monkeypatch.setattr("qsentinel.data.load", lambda *a, **k: ds, raising=False)
    monkeypatch.setattr("qsentinel.data.base.load", lambda *a, **k: ds)
    cfg = ExperimentConfig(dataset="synthetic", model="flow_gbdt", features="packet_seq")
    with pytest.raises(ValueError, match="needs a 2-D table|consumes"):
        train(cfg, write=False)


# --- the documented exit criterion ---------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _has_iscx(), reason="ISCX corpus not present")
def test_exit_criterion_on_the_real_pooled_dataset(tmp_path):
    cfg = ExperimentConfig.from_yaml("configs/experiments/iscx_pooled_traffic_type_gbdt.yaml")
    cfg.params = {"n_estimators": 120}
    out = train(cfg, runs_dir=tmp_path)

    assert out["metrics"]["macro_f1"] > out["metrics"]["baseline_macro_f1"]
    assert out["split"]["groups_disjoint"] is True
    assert out["metrics"]["n_classes"] == 8
