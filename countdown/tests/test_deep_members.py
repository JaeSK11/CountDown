"""The shared trainer and the recommended members that ride on it (decisions D1/D3/D5).

CPU only, toy data.  Covers the Phase-4 DoD line "``training/deep.py`` trains a toy net;
early stop + checkpoint", the harness's per-experiment ``feature_params``, and the member
contract for ``flow_image_cnn``, ``seq_cnn`` and ``seq_cnn_baseline``.
"""

from __future__ import annotations

import numpy as np
import pytest

from countdown.models.base import ModelRegistry
from countdown.schema import LabelSpace
from countdown.training.train import ExperimentConfig

torch = pytest.importorskip("torch")

SPACE = LabelSpace.from_names("toy", ["a", "b", "c"])
CPU = dict(device="cpu", workers=0, amp=False, seed=0)


def _separable(n: int, shape: tuple[int, ...], seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    X = rng.normal(0, 0.1, size=(n, *shape)).astype(np.float32)
    return X, y


# -- training/deep.py -------------------------------------------------------------------
def test_deep_trainer_learns_a_toy_problem_stops_early_and_checkpoints(tmp_path):
    from torch import nn

    from countdown.models.deep_member import ArrayDataset
    from countdown.training.deep import DeepConfig, DeepTrainer

    X, y = _separable(240, (6,), seed=1)
    X[np.arange(240), y] += 2.0                       # class c lights up feature c
    ckpt = tmp_path / "best.ckpt"
    cfg = DeepConfig(epochs=40, batch_size=32, learning_rate=5e-2, weight_decay=0.0, patience=3,
                     device="cpu", amp=False, num_workers=0, checkpoint_path=str(ckpt))
    net = nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 3))
    fwd = lambda m, b: m(b["x"])                       # noqa: E731
    res = DeepTrainer(cfg).fit(net, ArrayDataset(X[:180], y[:180]), n_classes=3,
                               val_ds=ArrayDataset(X[180:], y[180:]), forward_fn=fwd)
    assert res.best_val_macro_f1 > 0.95
    assert len(res.history) < 40, "patience=3 should stop well before the 40-epoch budget"
    assert ckpt.exists()
    proba = DeepTrainer(cfg).predict_proba(net, ArrayDataset(X[180:]), forward_fn=fwd)
    assert proba.shape == (60, 3) and (proba.argmax(1) == y[180:]).mean() > 0.95


def test_class_weights_default_off_for_every_member():
    """Decision D5: one convention -- no class weights -- for every registered member."""
    for name in ModelRegistry.list():
        if name.startswith("_"):
            continue
        m = ModelRegistry.create(name, label_space=SPACE)
        assert m.params.get("class_weight") in (None, "none"), name


# -- harness ----------------------------------------------------------------------------
def test_experiment_config_carries_feature_params():
    cfg = ExperimentConfig(dataset="x", features="flow_image",
                           feature_params={"construction": "flowpic", "channels": 3})
    assert cfg.to_dict()["feature_params"] == {"construction": "flowpic", "channels": 3}


@pytest.mark.parametrize("path", [
    "configs/experiments/iscx_pooled_traffic_type_seq_baseline.yaml",
    "configs/experiments/iscx_pooled_traffic_type_seq.yaml",
    "configs/experiments/mobileapp_activity_seq_baseline.yaml",
    "configs/experiments/mobileapp_activity_seq.yaml",
    "configs/experiments/mobileapp_activity_image.yaml",
])
def test_shipped_phase4_configs_match_their_model_contract(path):
    from countdown.features import get_extractor

    cfg = ExperimentConfig.from_yaml(path)
    model_cls = ModelRegistry.get(cfg.model)
    assert cfg.features == model_cls.input_type
    get_extractor(cfg.features, **cfg.feature_params)     # the params are ones the extractor takes
    rec = getattr(model_cls, "recommended_feature_params", None)
    if rec:
        assert {k: cfg.feature_params.get(k) for k in rec} == rec


# -- members ----------------------------------------------------------------------------
def _round_trip(model, X, tmp_path):
    from countdown.models import load_model

    before = model.predict_proba(X)
    restored = load_model(model.save(tmp_path / "m.joblib"))
    np.testing.assert_allclose(restored.predict_proba(X), before, atol=1e-5)
    return before


def test_flow_image_cnn_member_contract(tmp_path):
    cls = ModelRegistry.get("flow_image_cnn")
    assert (cls.input_type, cls.expects_ndim) == ("flow_image", 4)
    X, y = _separable(36, (3, 64, 64), seed=2)
    m = ModelRegistry.create("flow_image_cnn", label_space=SPACE, epochs=2, batch_size=12, **CPU)
    m.fit(X[:24], y[:24], val=(X[24:], y[24:]))
    proba = _round_trip(m, X[24:], tmp_path)
    assert proba.shape == (12, 3)
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-5)


def test_seq_cnn_learns_a_size_signature_and_round_trips(tmp_path):
    cls = ModelRegistry.get("seq_cnn")
    assert (cls.input_type, cls.expects_ndim) == ("packet_seq", 3)
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=150).astype(np.int64)
    X = np.zeros((150, 32, 2), dtype=np.float32)
    X[..., 0] = rng.choice([-1.0, 1.0], size=(150, 32)) * (200 + 500 * y[:, None])   # size encodes class
    X[..., 1] = rng.random((150, 32)) * 0.01
    m = ModelRegistry.create("seq_cnn", label_space=SPACE, epochs=8, batch_size=16, lr=5e-3, **CPU)
    m.fit(X[:120], y[:120], val=(X[120:], y[120:]))
    proba = _round_trip(m, X[120:], tmp_path)
    assert (proba.argmax(1) == y[120:]).mean() > 0.9
    with pytest.raises(ValueError, match="channels=2"):
        m.predict_proba(X[:4, :, 0])


def test_df_baseline_contract_and_architecture(tmp_path):
    from countdown.models.seq_cnn import build_df

    cls = ModelRegistry.get("seq_cnn_baseline")
    assert (cls.input_type, cls.expects_ndim) == ("dir_seq", 2)
    # 5000 -> 1250 -> 313 -> 79 -> 20 under Keras 'same' pooling: a 256 x 20 flatten.
    assert build_df(95, 5000)[1][0].in_features == 256 * 20

    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, size=90).astype(np.int64)
    X = np.zeros((90, 128), dtype=np.float32)
    for i, c in enumerate(y):                         # class = share of outgoing packets
        X[i, :100] = np.where(rng.random(100) < 0.2 + 0.3 * c, 1.0, -1.0)
    m = ModelRegistry.create("seq_cnn_baseline", label_space=SPACE, epochs=8, batch_size=16,
                             device="cpu", seed=0)
    m.fit(X[:72], y[:72], val=(X[72:], y[72:]))
    assert len(m.history_) == 8 and "val_macro_f1" in m.history_[0]
    proba = _round_trip(m, X[72:], tmp_path)
    assert proba.shape == (18, 3)


def test_dir_seq_extractor_pads_and_truncates():
    from countdown.features import get_extractor
    from countdown.schema import Flow

    from countdown.schema import FiveTuple

    def flow(dirs):
        n = len(dirs)
        return Flow(
            flow_id="t", five_tuple=FiveTuple("10.0.0.1", 1234, "10.0.0.2", 443, "tcp"),
            timestamps=np.arange(n, dtype=np.float64), sizes=np.full(n, 100, dtype=np.int32),
            directions=np.asarray(dirs, dtype=np.int8),
            label_fields={"traffic_type": "chat"}, meta={"source_file": "cap.pcap"},
        )

    X = get_extractor("dir_seq", n=6).transform([flow([1, -1, -1, 1]), flow([1] * 9)])
    assert X.shape == (2, 6) and X.dtype == np.float32
    np.testing.assert_array_equal(X[0], [1, -1, -1, 1, 0, 0])      # zero-padded
    np.testing.assert_array_equal(X[1], [1] * 6)                   # truncated


# -- bytes: field randomisation (decision D1b) -------------------------------------------
def test_field_randomiser_rewrites_only_the_seq_ack_bytes():
    from countdown.data.etbert_corpus import _tokens_to_bytes
    from countdown.features.payload_bytes import DEFAULT_VOCAB, ETBertTokenizer
    from countdown.models.byte_net import FieldRandomiser

    if not DEFAULT_VOCAB.exists():
        pytest.skip("ET-BERT vocabulary not fetched")
    tok = ETBertTokenizer()
    fr = FieldRandomiser(tok)
    rng = np.random.default_rng(0)
    hit = 0
    for trial in range(40):                            # random packets, incl. WordPiece-split words
        raw = bytes(rng.integers(0, 256, size=30).tolist())
        text = " ".join(f"{raw[i]:02x}{raw[i + 1]:02x}" for i in range(len(raw) - 1))
        src = np.asarray(tok.encode(text, 64)[0])
        before = src.copy()
        out, seg = fr(src, rng)
        assert (src == before).all()                   # input not modified
        words, _ = fr._words(out, len(raw) - 1)
        assert words is not None
        back = _tokens_to_bytes(" ".join(words))
        assert back[:2] == raw[:2] and back[10:] == raw[10:], trial   # bytes outside 2..9 intact
        hit += back[2:10] != raw[2:10]
        assert (seg == (out != tok.pad_id)).all()
    assert hit >= 39                                   # seq/ack replaced essentially always
    assert ModelRegistry.get("byte_net").input_type == "bytes"
