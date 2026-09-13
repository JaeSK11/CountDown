"""Flow-image extractor, capture windowing, and the Okonkwo CNN."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.data.windows import expand_to_windows
from countdown.features.flow_image import FlowImage
from countdown.schema import FiveTuple, Flow

FT = FiveTuple("10.0.0.1", 1234, "10.0.0.2", 443, "tcp")


def _flow(ts, sizes, dirs=None, source_file="/cap/a.pcap", flow_id="f0"):
    n = len(ts)
    return Flow(
        flow_id=flow_id, five_tuple=FT,
        timestamps=np.asarray(ts, dtype=np.float64),
        sizes=np.asarray(sizes, dtype=np.int32),
        directions=np.asarray(dirs if dirs is not None else [1] * n, dtype=np.int8),
        label_fields={"traffic_type": "chat"}, meta={"source_file": source_file},
    )


# -- extractor ---------------------------------------------------------------------------


def test_scatter_is_binary_and_marks_the_right_cell():
    ex = FlowImage(size=16, construction="scatter", window=10.0, marker=1)
    img = ex.transform_one(_flow([0.0, 10.0], [0, 1500]))
    assert set(np.unique(img)) <= {0.0, 1.0}
    # t=0,size=0 -> bottom-left; t=window,size=mtu -> top-right.
    assert img[0, 0, 0] == 1.0
    assert img[0, 15, 15] == 1.0
    assert img.sum() == 2


def test_oversize_packets_are_dropped_not_clipped():
    """The paper discards packets above the MTU (section 4.2)."""
    dropped = FlowImage(size=16, window=10.0, marker=1, drop_oversize=True)
    clipped = FlowImage(size=16, window=10.0, marker=1, drop_oversize=False)
    flow = _flow([0.0, 5.0], [100, 9000])
    assert dropped.transform_one(flow).sum() == 1
    assert clipped.transform_one(flow).sum() == 2


def test_marker_widens_the_mark():
    """marker=k spreads a packet over k*k pixels, approximating matplotlib's blob.

    Times are relative to the flow's first packet, so packet 0 always lands on column 0
    and its marker clips against the edge.  The assertion targets the interior mark.
    """
    flow = _flow([0.0, 5.0], [750, 750])          # -> columns 0 and 15, row 15
    one = FlowImage(size=32, window=10.0, marker=1).transform_one(flow)
    three = FlowImage(size=32, window=10.0, marker=3).transform_one(flow)
    assert one[0, 13:18, 13:18].sum() == 1
    assert three[0, 13:18, 13:18].sum() == 9
    assert one[0, :, 0].sum() == 1 and three[0, :, 0].sum() == 3   # clipped at the edge


def test_flowpic_is_a_density_not_presence():
    """A busy bin must outweigh a quiet one -- the information a scatter throws away."""
    pic = FlowImage(size=8, construction="flowpic", window=10.0, marker=1,
                    log_density=False)
    scat = FlowImage(size=8, construction="scatter", window=10.0, marker=1)
    # one packet in bin A, three in bin B
    flow = _flow([0.0, 5.0, 5.0, 5.0], [100, 900, 900, 900])
    p, s_ = pic.transform_one(flow)[0], scat.transform_one(flow)[0]
    busy, quiet = p.max(), p[p > 0].min()
    assert busy > quiet                       # density distinguishes them
    assert set(np.unique(s_)) == {0.0, 1.0}   # the scatter cannot
    assert s_.sum() == 2


def test_direction_channel_separates_upstream_from_downstream():
    ex = FlowImage(size=8, channels=3, window=10.0, marker=1)
    up = ex.transform_one(_flow([1.0], [500], dirs=[1]))
    down = ex.transform_one(_flow([1.0], [500], dirs=[-1]))
    assert up[1].max() == pytest.approx(1.0)
    assert down[1].min() == pytest.approx(-1.0)
    # channel 0 (presence) cannot tell them apart -- which is the point.
    assert np.array_equal(up[0], down[0])


def test_empty_and_degenerate_flows_do_not_explode():
    ex = FlowImage(size=8, window=None, marker=1)
    assert ex.transform_one(_flow([], [])).sum() == 0
    # all packets at one timestamp -> zero span, must not divide by zero
    assert np.isfinite(ex.transform_one(_flow([3.0, 3.0], [10, 20]))).all()


def test_blank_fraction_counts_empty_windows():
    ex = FlowImage(size=8, window=10.0)
    assert ex.blank_fraction([_flow([], []), _flow([1.0], [100])]) == 0.5


# -- windowing ---------------------------------------------------------------------------


def test_windows_tile_a_capture_timeline():
    flow = _flow(np.arange(0.0, 60.0, 1.0), [100] * 60)
    assert len(expand_to_windows([flow], window=60.0)) == 1
    assert len(expand_to_windows([flow], window=30.0)) == 2
    assert len(expand_to_windows([flow], window=15.0)) == 4


def test_windows_merge_flows_of_one_capture():
    """Okonkwo's unit is the capture-window: concurrent flows share an image."""
    a = _flow([0.0, 1.0], [100, 200], flow_id="a", source_file="/cap/x.pcap")
    b = _flow([0.5, 1.5], [300, 400], flow_id="b", source_file="/cap/x.pcap")
    win = expand_to_windows([a, b], window=60.0)
    assert len(win) == 1
    assert win[0].n_packets == 4
    assert list(win[0].timestamps) == [0.0, 0.5, 1.0, 1.5]   # merged onto one clock


def test_per_flow_windowing_keeps_flows_apart():
    a = _flow([0.0], [100], flow_id="a", source_file="/cap/x.pcap")
    b = _flow([0.5], [300], flow_id="b", source_file="/cap/x.pcap")
    assert len(expand_to_windows([a, b], window=60.0, per_flow=True)) == 2


def test_windows_inherit_the_group_key_so_splits_stay_safe():
    """Every window of a capture must keep source_file, or the split leaks."""
    flow = _flow(np.arange(0.0, 60.0, 1.0), [100] * 60, source_file="/cap/y.pcap")
    win = expand_to_windows([flow], window=15.0)
    assert {w.meta["source_file"] for w in win} == {"/cap/y.pcap"}
    assert [w.meta["window_index"] for w in win] == [0, 1, 2, 3]


def test_blank_windows_are_dropped_by_default_and_kept_on_request():
    """A gap with no packets is the Sensors arrival-time-disparity artifact."""
    flow = _flow([0.0, 100.0], [100, 100])          # 60s window 1 is empty
    assert len(expand_to_windows([flow], window=60.0)) == 2
    assert len(expand_to_windows([flow], window=60.0, keep_blank=True)) == 2


def test_windowing_rejects_a_flow_with_no_group_key():
    flow = _flow([0.0], [100])
    flow.meta = {}
    with pytest.raises(ValueError, match="source_file"):
        expand_to_windows([flow], window=60.0)


# -- model -------------------------------------------------------------------------------


def test_network_matches_figure_2():
    """224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 3, flatten 64*3*3 = 576 as labelled."""
    torch = pytest.importorskip("torch")
    from countdown.models.flow_image_cnn import build_network

    net = build_network(n_classes=10, in_channels=1, size=224)
    assert net.n_flat == 576
    assert tuple(net(torch.zeros(2, 1, 224, 224)).shape) == (2, 10)


def test_cnn_fits_and_reports_label_space_columns():
    pytest.importorskip("torch")
    from countdown.models import ModelRegistry
    from countdown.schema import LabelSpace

    space = LabelSpace.from_names("traffic_type", ["chat", "voip", "p2p"])
    rng = np.random.default_rng(0)
    X = rng.random((24, 1, 64, 64)).astype(np.float32)
    y = np.array([0, 1] * 12)          # class 2 never appears in training
    model = ModelRegistry.create("flow_image_cnn_baseline", label_space=space,
                                 epochs=2, batch_size=8, device="cpu")
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (24, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


# -- per-sample axis scaling -------------------------------------------------------------


def test_auto_window_scales_each_sample_to_its_own_length():
    """Paper 4.2 caps the x-axis at the window size, so pooled sizes keep their scale.

    The same packet 7.5 s in must land mid-image in a 15 s window and quarter-way into a
    30 s one.  A fixed axis would put both in the same column and quietly turn the
    multi-window pooling into a distortion.
    """
    ex = FlowImage(size=32, window="auto", marker=1)
    made = {}
    for w in (15.0, 30.0):
        f = _flow([0.0, 7.5], [750, 750])
        f.meta = {**f.meta, "window_seconds": w}
        img = ex.transform_one(f)
        made[w] = sorted(np.nonzero(img[0])[1].tolist())
    assert made[15.0] == [0, 15]      # 7.5/15 -> halfway
    assert made[30.0] == [0, 7]       # 7.5/30 -> a quarter in
    # a fixed axis collapses the distinction
    fixed = FlowImage(size=32, window=30.0, marker=1)
    f = _flow([0.0, 7.5], [750, 750])
    f.meta = {**f.meta, "window_seconds": 15.0}
    assert sorted(np.nonzero(fixed.transform_one(f)[0])[1].tolist()) == [0, 7]


def test_auto_window_falls_back_to_flow_duration_when_unwindowed():
    ex = FlowImage(size=32, window="auto", marker=1)
    img = ex.transform_one(_flow([0.0, 4.0], [750, 750]))   # no window_seconds in meta
    assert sorted(np.nonzero(img[0])[1].tolist()) == [0, 31]


def test_auto_window_survives_the_windowing_round_trip():
    """expand_to_windows stamps window_seconds; the extractor must read it back."""
    flow = _flow(np.arange(0.0, 30.0, 0.5), [500] * 60)
    ex = FlowImage(size=32, window="auto", marker=1)
    for w in (15.0, 30.0):
        for win in expand_to_windows([flow], window=w):
            assert win.meta["window_seconds"] == w
            assert np.isfinite(ex.transform_one(win)).all()


def test_bad_window_string_is_rejected():
    with pytest.raises(ValueError, match="'auto'"):
        FlowImage(window="whole-flow")
