"""Draper-Gil (ICISSP 2016) replication components: time-only features, segmentation, models."""

from __future__ import annotations

import numpy as np
import pytest

from qsentinel.features import get_extractor
from qsentinel.features.timeonly import DraperGilTimeFeatures, active_idle_spans
from qsentinel.flows.segment import segment_flow, segment_flows
from qsentinel.models import ModelRegistry
from qsentinel.schema import FiveTuple, Flow, LabelSpace


def _flow(ts: list[float], dirs: list[int] | None = None, size: int = 100) -> Flow:
    n = len(ts)
    return Flow(
        flow_id="t",
        five_tuple=FiveTuple("10.0.0.1", 1234, "10.0.0.2", 443, "tcp"),
        timestamps=np.asarray(ts, dtype=np.float64),
        sizes=np.full(n, size, dtype=np.int32),
        directions=np.asarray(dirs if dirs else [1, -1] * n, dtype=np.int8)[:n],
        label_fields={"traffic_type": "chat", "is_vpn": False},
        meta={"source_file": "cap.pcap"},
    )


# --- time-only features ---------------------------------------------------------------


def test_timeonly_has_exactly_the_papers_23_features():
    ex = DraperGilTimeFeatures()
    assert len(ex.feature_names) == 23
    assert ex.feature_names[0] == "duration"
    assert list(ex.feature_names[-2:]) == ["fb_psec", "fp_psec"]


def test_timeonly_carries_no_packet_size_statistic():
    """The whole point of the baseline: rates only, never a size distribution."""
    names = set(DraperGilTimeFeatures().feature_names)
    assert not {n for n in names if "size" in n}
    assert "fb_psec" in names  # bytes enter only as a rate


def test_timeonly_registered_and_configurable():
    ex = get_extractor("timeonly", activity_timeout=1.0)
    assert isinstance(ex, DraperGilTimeFeatures)
    assert ex.activity_timeout == 1.0


def test_timeonly_is_finite_for_degenerate_flows(synthetic_flows):
    X = DraperGilTimeFeatures().transform(synthetic_flows)
    assert X.shape == (len(synthetic_flows), 23)
    assert np.isfinite(X).all()


def test_duration_and_rates_are_what_the_paper_defines():
    f = _flow([0.0, 1.0, 2.0, 3.0], size=100)  # 4 packets, 400 B, 3 s
    ex = DraperGilTimeFeatures()
    v = dict(zip(ex.feature_names, ex.transform_one(f)))
    assert v["duration"] == pytest.approx(3.0)
    assert v["fb_psec"] == pytest.approx(400 / 3.0)
    assert v["fp_psec"] == pytest.approx(4 / 3.0)


def test_active_idle_split_on_the_activity_threshold():
    # two bursts 10 s apart; with a 5 s threshold that is one idle gap of 10 s.
    active, idle = active_idle_spans(np.array([0.0, 1.0, 2.0, 12.0, 13.0]), activity_timeout=5.0)
    assert idle == pytest.approx([10.0])
    assert active == pytest.approx([2.0, 1.0])


def test_active_idle_threshold_is_respected():
    ts = np.array([0.0, 1.0, 2.0, 12.0, 13.0])
    # a 20 s threshold sees no gap at all -> one active span, no idle
    active, idle = active_idle_spans(ts, activity_timeout=20.0)
    assert idle.size == 0
    assert active == pytest.approx([13.0])


# --- segmentation ---------------------------------------------------------------------


def test_segment_cuts_on_fixed_duration_not_idleness():
    f = _flow([0.0, 1.0, 2.0, 16.0, 17.0, 31.0, 32.0])
    windows = segment_flow(f, window=15.0, min_packets=2)
    # [0,15) -> 3 packets, [15,30) -> 2, [30,45) -> 2
    assert [w.n_packets for w in windows] == [3, 2, 2]
    assert windows[0].flow_id.endswith("@w0")


def test_segment_inherits_labels_and_group_key():
    f = _flow([0.0, 1.0, 20.0, 21.0])
    for w in segment_flow(f, window=15.0):
        assert w.label_fields["traffic_type"] == "chat"
        assert w.meta["source_file"] == "cap.pcap"


def test_segment_label_fields_are_copied_not_shared():
    f = _flow([0.0, 1.0, 20.0, 21.0])
    a, b = segment_flow(f, window=15.0)
    a.label_fields["traffic_type"] = "mutated"
    assert b.label_fields["traffic_type"] == "chat"


def test_segment_drops_are_counted_not_silent():
    f = _flow([0.0, 1.0, 20.0])  # window 1 holds a single packet
    windows, stats = segment_flows([f], window=15.0, min_packets=2)
    assert len(windows) == 1
    assert stats["windows_dropped_min_packets"] == 1


def test_shorter_timeout_yields_more_windows():
    f = _flow(list(np.arange(0, 120, 1.0)))
    short, _ = segment_flows([f], window=15.0)
    long, _ = segment_flows([f], window=120.0)
    assert len(short) > len(long)


# --- paper models ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["flow_c45_paper", "flow_knn_paper"])
def test_paper_models_consume_the_paper_features(name):
    model = ModelRegistry.get(name)
    assert model.input_type == "timeonly"


@pytest.mark.parametrize("name", ["flow_c45_paper", "flow_knn_paper"])
def test_paper_models_emit_label_space_columns(name):
    space = LabelSpace.from_names("traffic_type", ["browsing", "chat", "voip"])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 23))
    y = np.repeat([1, 2], 30)  # class 0 ("browsing") never seen, as on ISCXVPN

    model = ModelRegistry.create(name, label_space=space)
    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (60, 3)
    assert proba.sum(axis=1) == pytest.approx(np.ones(60))
    assert (proba[:, 0] == 0).all()  # unseen class keeps a zero column


@pytest.mark.parametrize("name", ["flow_c45_paper", "flow_knn_paper"])
def test_paper_models_survive_a_single_class_fold(name):
    space = LabelSpace.from_names("traffic_type", ["browsing", "chat"])
    X = np.random.default_rng(1).normal(size=(10, 23))
    model = ModelRegistry.create(name, label_space=space)
    model.fit(X, np.ones(10, dtype=np.int64))
    proba = model.predict_proba(X)
    assert (proba[:, 1] == 1.0).all()


@pytest.mark.parametrize("name", ["flow_c45_paper", "flow_knn_paper"])
def test_paper_models_round_trip(tmp_path, name):
    from qsentinel.models import load_model

    space = LabelSpace.from_names("traffic_type", ["chat", "voip"])
    X = np.random.default_rng(2).normal(size=(40, 23))
    y = np.repeat([0, 1], 20)

    model = ModelRegistry.create(name, label_space=space)
    model.fit(X, y)
    before = model.predict_proba(X)

    path = model.save(tmp_path / f"{name}.joblib")
    restored = load_model(path)
    assert restored.predict_proba(X) == pytest.approx(before)


def test_knn_scales_features_before_distance():
    """Unscaled, fb_psec (~1e6) would swamp flowiat (~1e-5) in Euclidean distance."""
    space = LabelSpace.from_names("t", ["a", "b"])
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 23))
    X[:, -2] *= 1e6                      # fb_psec on a wild scale
    X[:20, 1] += 5.0                     # the real signal lives in fiat_mean
    y = np.repeat([0, 1], 20)

    model = ModelRegistry.create("flow_knn_paper", label_space=space)
    model.fit(X, y)
    assert (model.predict(X) == y).mean() > 0.9
