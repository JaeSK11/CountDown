"""``graph_gnn_baseline``: the TFE-GNN network behind the ``BaseModel`` contract.

Everything here runs on CPU with a handful of tiny samples.  The network itself and the
graph construction are covered by ``test_tfe_gnn.py``; this file checks the member
contract -- registry, rank, label-space-ordered probabilities, persistence -- which is what
the ensemble relies on and what the replication script never exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from countdown.models.base import ModelRegistry
from countdown.schema import LabelSpace

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from countdown.features.traffic_graph import PAD_TRUNC_DIGIT  # noqa: E402

HEADER, PAYLOAD, PACKETS = 8, 12, 3


def _byte_rows(n: int, seed: int, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """``(n, PACKETS, HEADER + PAYLOAD)`` int16 rows with a label-dependent byte pattern."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    X = rng.integers(0, 256, size=(n, PACKETS, HEADER + PAYLOAD)).astype(np.int16)
    for i, c in enumerate(y):
        X[i, :, :4] = c * 40                      # a header signature per class
        X[i, -1, HEADER + 6:] = PAD_TRUNC_DIGIT   # a short last payload, as byte_prep pads
    return X, y.astype(np.int64)


@pytest.fixture
def small_model():
    space = LabelSpace.from_names("traffic_type", ["a", "b", "c"])
    return ModelRegistry.create(
        "graph_gnn_baseline", label_space=space, header_len=HEADER,
        epochs=2, batch_size=4, accum=1, workers=0, device="cpu", seed=0,
        embedding_dim=8, hidden_dim=8, n_gnn_layers=2,
    )


def test_registered_with_the_contract_the_harness_checks():
    cls = ModelRegistry.get("graph_gnn_baseline")
    assert cls.input_type == "byte_matrix"
    assert cls.expects_ndim == 3


def test_module_imports_torch_lazily():
    """A missing heavy dep must not break the registry (PHASE-4 §8): no top-level torch import."""
    import ast
    import inspect

    import countdown.models.graph_gnn as mod

    tree = ast.parse(inspect.getsource(mod))
    top = {
        (n.names[0].name if isinstance(n, ast.Import) else n.module or "").split(".")[0]
        for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
    }
    assert not top & {"torch", "torch_geometric"}, top


def test_fit_predict_proba_is_label_space_shaped(small_model):
    X, y = _byte_rows(12, seed=1)
    small_model.fit(X, y)
    proba = small_model.predict_proba(X)
    assert proba.shape == (12, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert small_model.n_params() > 0
    assert len(small_model.history_) == 2


def test_rank_and_header_len_are_checked(small_model):
    X, y = _byte_rows(6, seed=2)
    with pytest.raises(ValueError, match=r"\(N, P, H \+ L\)"):
        small_model.fit(X.reshape(6, -1), y)
    bad = ModelRegistry.create(
        "graph_gnn_baseline", label_space=small_model.label_space,
        header_len=HEADER + PAYLOAD, epochs=1, workers=0, device="cpu",
    )
    with pytest.raises(ValueError, match="no payload bytes"):
        bad.fit(X, y)


def test_val_fold_and_early_stopping_record_history(small_model):
    X, y = _byte_rows(12, seed=3)
    Xv, yv = _byte_rows(6, seed=4)
    small_model.set_params(epochs=4, early_stop_patience=1)
    small_model.fit(X, y, val=(Xv, yv))
    assert all("val_macro_f1" in row for row in small_model.history_)
    assert 1 <= len(small_model.history_) <= 4


def test_balanced_class_weight_is_accepted_and_unknown_rejected(small_model):
    X, y = _byte_rows(12, seed=5)
    small_model.set_params(class_weight="balanced", epochs=1).fit(X, y)
    small_model.set_params(class_weight="sqrt")
    with pytest.raises(ValueError, match="class_weight"):
        small_model.fit(X, y)


def test_save_load_round_trip_preserves_probabilities(small_model, tmp_path):
    X, y = _byte_rows(10, seed=6)
    small_model.fit(X, y)
    before = small_model.predict_proba(X)
    path = small_model.save(tmp_path / "graph.joblib")

    from countdown.models import load_model

    restored = load_model(path)
    assert restored.name == "graph_gnn_baseline"
    assert restored.label_space == small_model.label_space
    assert restored._n_packets == PACKETS
    np.testing.assert_allclose(restored.predict_proba(X), before, atol=1e-6)


# -- the recommended variant: GIN through training/deep.py --------------------------------
def test_gin_member_contract_and_round_trip(tmp_path):
    from countdown.models import load_model
    from countdown.models.tfe_gnn import TFEGNNNet
    from torch_geometric.nn import GINConv, SAGEConv

    cls = ModelRegistry.get("graph_gnn")
    assert (cls.input_type, cls.expects_ndim) == ("byte_matrix", 3)
    assert isinstance(TFEGNNNet(3, 8, 8, 2, conv="gin").header_encoder.convs[0], GINConv)
    assert isinstance(TFEGNNNet(3, 8, 8, 2).header_encoder.convs[0], SAGEConv)   # paper default untouched

    space = LabelSpace.from_names("traffic_type", ["a", "b", "c"])
    m = ModelRegistry.create("graph_gnn", label_space=space, header_len=HEADER, epochs=2,
                             batch_size=4, workers=0, device="cpu", seed=0,
                             embedding_dim=8, hidden_dim=8, n_gnn_layers=2)
    X, y = _byte_rows(12, seed=7)
    Xv, yv = _byte_rows(6, seed=8)
    m.fit(X, y, val=(Xv, yv))
    proba = m.predict_proba(Xv)
    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert len(m.history_) == 2
    restored = load_model(m.save(tmp_path / "gin.joblib"))
    np.testing.assert_allclose(restored.predict_proba(Xv), proba, atol=1e-5)
