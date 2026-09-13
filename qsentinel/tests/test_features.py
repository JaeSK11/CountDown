"""Feature-extractor unit tests (task 0.10 DoD)."""

from __future__ import annotations

import numpy as np
import pytest

from qsentinel.features import FlowStats, PacketSeq, available_extractors, get_extractor
from qsentinel.schema import FiveTuple, Flow


def test_registry_exposes_both_phase0_extractors():
    # "timeonly" is the Draper-Gil replication baseline, added alongside the Phase-0 pair.
    assert {"flow_stats", "packet_seq"} <= set(available_extractors())
    assert isinstance(get_extractor("flow_stats"), FlowStats)
    assert isinstance(get_extractor("packet_seq"), PacketSeq)


def test_get_extractor_rejects_unknown_name():
    with pytest.raises(KeyError):
        get_extractor("does_not_exist")


# --- flow_stats -----------------------------------------------------------------------


def test_flow_stats_shape_matches_feature_names(synthetic_flows):
    ex = FlowStats(first_n_packets=20)
    X = ex.transform(synthetic_flows)
    assert X.shape == (len(synthetic_flows), len(ex.feature_names))
    assert X.shape[1] == 56
    assert X.dtype == np.float32


def test_flow_stats_first_n_is_configurable(synthetic_flows):
    assert FlowStats(first_n_packets=5).transform(synthetic_flows).shape[1] == 41
    assert FlowStats(first_n_packets=40).transform(synthetic_flows).shape[1] == 76


def test_flow_stats_is_finite_even_for_degenerate_flows(synthetic_flows):
    # synthetic_flows includes a 1-packet flow: zero duration, no IATs, no backward packets.
    X = FlowStats().transform(synthetic_flows)
    assert np.isfinite(X).all()


def test_flow_stats_caps_sizes_at_mtu(synthetic_flows):
    ex = FlowStats()
    names = list(ex.feature_names)
    X = ex.transform(synthetic_flows)
    big = X[2]  # the 9000-byte flow
    assert big[names.index("size_max")] == 1500


def test_flow_stats_counts_are_correct():
    f = Flow(
        flow_id="t", five_tuple=FiveTuple("a", 1, "b", 2, "tcp"),
        timestamps=np.array([0.0, 1.0, 2.0, 3.0]),
        sizes=np.array([100, 200, 300, 400], dtype=np.int32),
        directions=np.array([1, -1, 1, -1], dtype=np.int8),
    )
    ex = FlowStats()
    names = list(ex.feature_names)
    row = ex.transform([f])[0]
    assert row[names.index("n_packets")] == 4
    assert row[names.index("n_fwd_packets")] == 2
    assert row[names.index("n_bwd_packets")] == 2
    assert row[names.index("fwd_bytes")] == 400   # 100 + 300
    assert row[names.index("bwd_bytes")] == 600   # 200 + 400
    assert row[names.index("total_bytes")] == 1000
    assert row[names.index("duration")] == pytest.approx(3.0)
    assert row[names.index("iat_mean")] == pytest.approx(1.0)


def test_flow_stats_first_n_columns_are_signed_sizes():
    f = Flow(
        flow_id="t", five_tuple=FiveTuple("a", 1, "b", 2, "tcp"),
        timestamps=np.arange(3, dtype=np.float64),
        sizes=np.array([100, 200, 300], dtype=np.int32),
        directions=np.array([1, -1, 1], dtype=np.int8),
    )
    ex = FlowStats(first_n_packets=5)
    names = list(ex.feature_names)
    row = ex.transform([f])[0]
    start = names.index("pkt0_signed_size")
    assert row[start:start + 5].tolist() == [100.0, -200.0, 300.0, 0.0, 0.0]  # zero-padded


def test_flow_stats_deterministic(synthetic_flows):
    a = FlowStats().transform(synthetic_flows)
    b = FlowStats().transform(synthetic_flows)
    assert np.array_equal(a, b)


# --- packet_seq -----------------------------------------------------------------------


def test_packet_seq_shapes(synthetic_flows):
    assert PacketSeq(n=32, channels=2).transform(synthetic_flows).shape == (4, 32, 2)
    assert PacketSeq(n=16, channels=1).transform(synthetic_flows).shape == (4, 16)


def test_packet_seq_rejects_bad_channel_count():
    with pytest.raises(ValueError):
        PacketSeq(channels=3)


def test_packet_seq_pads_and_truncates(synthetic_flows):
    ex = PacketSeq(n=8, channels=1)
    X = ex.transform(synthetic_flows)
    short = X[1]   # 3-packet flow
    long_ = X[2]   # 100-packet flow
    assert (short[3:] == 0).all()          # tail zero-padded
    assert (long_ != 0).all()              # fully occupied after truncation
    assert X.shape[1] == 8


def test_packet_seq_mask_marks_real_packets(synthetic_flows):
    ex = PacketSeq(n=8)
    X, mask = ex.transform_with_mask(synthetic_flows)
    assert mask.shape == (4, 8)
    assert mask.sum(axis=1).tolist() == [8, 3, 8, 1]
    assert mask.dtype == bool


def test_packet_seq_signs_by_direction_and_caps_at_mtu(synthetic_flows):
    X = PacketSeq(n=4, channels=1).transform(synthetic_flows)
    assert X[0].tolist() == [100.0, -100.0, 100.0, -100.0]
    assert X[2].tolist() == [1500.0, -1500.0, 1500.0, -1500.0]  # 9000 B capped


def test_packet_seq_iat_channel_starts_at_zero(synthetic_flows):
    X = PacketSeq(n=4, channels=2).transform(synthetic_flows)
    assert X[0, 0, 1] == 0.0                       # no IAT before the first packet
    assert X[0, 1:, 1].tolist() == [0.5, 0.5, 0.5]  # fixture spaces packets 0.5 s apart


def test_packet_seq_is_finite_and_deterministic(synthetic_flows):
    a = PacketSeq().transform(synthetic_flows)
    b = PacketSeq().transform(synthetic_flows)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_empty_input_returns_empty_arrays():
    assert FlowStats().transform([]).shape == (0, 56)
    assert PacketSeq(n=32, channels=2).transform([]).shape == (0, 32, 2)
    assert PacketSeq(n=32, channels=1).transform([]).shape == (0, 32)
