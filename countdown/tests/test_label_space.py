"""LabelSpace: the ensemble seam. Column meaning must not depend on the dataset."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.schema import (
    TRAFFIC_TYPES,
    TUNNEL_TYPES,
    FiveTuple,
    Flow,
    FlowDataset,
    LabelSpace,
)


def _mk_flow(i: int, traffic_type: str = "chat", **meta) -> Flow:
    return Flow(
        flow_id=f"f{i}",
        five_tuple=FiveTuple("10.0.0.1", 1000 + i, "1.1.1.1", 443, "tcp"),
        timestamps=np.arange(4, dtype=np.float64),
        sizes=np.full(4, 100, dtype=np.int32),
        directions=np.array([1, -1, 1, -1], dtype=np.int8),
        label_fields={"traffic_type": traffic_type, "app": f"app{i % 3}"},
        meta=meta or {"source_file": f"/caps/{i}.pcap"},
    )


# --- construction ------------------------------------------------------------------------

def test_taxonomy_space_is_the_full_taxonomy_not_the_observed_subset():
    space = LabelSpace.taxonomy("traffic_type")
    assert space.names == TRAFFIC_TYPES
    assert len(space) == 8
    assert LabelSpace.taxonomy("tunnel_type").names == TUNNEL_TYPES


def test_taxonomy_rejects_open_ended_targets():
    with pytest.raises(ValueError, match="not taxonomy-backed"):
        LabelSpace.taxonomy("app")


def test_from_dataset_picks_taxonomy_for_traffic_type_even_when_classes_are_missing():
    """The whole point: a 2-class ISCX-like dataset still yields 8 canonical columns."""
    ds = FlowDataset([_mk_flow(0, "chat"), _mk_flow(1, "voip")], "unit", "traffic_type")
    assert ds.label_names() == ["chat", "voip"]        # dataset-local: 2
    assert len(LabelSpace.from_dataset(ds)) == 8       # canonical: 8


def test_from_dataset_uses_observed_classes_for_open_ended_targets():
    ds = FlowDataset([_mk_flow(i) for i in range(4)], "unit", "app")
    space = LabelSpace.from_dataset(ds, "app")
    assert space.target == "app"
    assert space.names == ("app0", "app1", "app2")


def test_duplicate_or_empty_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        LabelSpace("t", ("a", "b", "a"))
    with pytest.raises(ValueError, match="at least one"):
        LabelSpace("t", ())


# --- encoding ----------------------------------------------------------------------------

def test_encode_decode_round_trip():
    space = LabelSpace.taxonomy("traffic_type")
    ids = space.encode(["voip", "browsing", "p2p"])
    assert ids.tolist() == [space.index("voip"), 0, 7]
    assert space.decode(ids) == ["voip", "browsing", "p2p"]


def test_encoding_an_unknown_class_is_a_hard_error():
    space = LabelSpace.taxonomy("traffic_type")
    with pytest.raises(KeyError, match="not in the 'traffic_type' LabelSpace"):
        space.encode(["voip", "gopher"])


def test_labels_with_space_are_canonical_and_without_space_are_local():
    """The ISCXVPN-vs-ISCXTor bug in miniature: same class, different local id."""
    narrow = FlowDataset([_mk_flow(0, "voip"), _mk_flow(1, "p2p")], "narrow", "traffic_type")
    wide = FlowDataset(
        [_mk_flow(0, "browsing"), _mk_flow(1, "voip"), _mk_flow(2, "p2p")],
        "wide", "traffic_type",
    )
    # dataset-local ids disagree about what "voip" is
    assert narrow.labels()[0] != wide.labels()[1]

    space = LabelSpace.taxonomy("traffic_type")
    assert narrow.labels(space=space)[0] == wide.labels(space=space)[1] == space.index("voip")


def test_labels_rejects_a_space_for_a_different_target():
    ds = FlowDataset([_mk_flow(0)], "unit", "traffic_type")
    with pytest.raises(ValueError, match="LabelSpace is for target"):
        ds.labels(space=LabelSpace.taxonomy("tunnel_type"))


def test_labels_without_space_is_unchanged_by_phase3():
    """Phase-0 behaviour is pinned: observed subset, taxonomy ordering."""
    ds = FlowDataset([_mk_flow(0, "chat"), _mk_flow(1, "browsing")], "unit", "traffic_type")
    assert ds.label_names() == ["browsing", "chat"]
    assert ds.labels().tolist() == [1, 0]


# --- the seam ----------------------------------------------------------------------------

def test_align_zero_fills_classes_the_source_never_saw():
    full = LabelSpace.taxonomy("traffic_type")
    partial = LabelSpace("traffic_type", ("chat", "voip"))
    proba = np.array([[0.3, 0.7], [0.9, 0.1]])

    out = full.align(proba, partial)
    assert out.shape == (2, 8)
    assert out[:, full.index("chat")].tolist() == [0.3, 0.9]
    assert out[:, full.index("voip")].tolist() == [0.7, 0.1]
    assert out[:, full.index("browsing")].tolist() == [0.0, 0.0]
    # nothing invented: row mass is preserved
    assert np.allclose(out.sum(axis=1), proba.sum(axis=1))


def test_align_checks_the_source_width():
    full = LabelSpace.taxonomy("traffic_type")
    with pytest.raises(ValueError, match="expected"):
        full.align(np.zeros((2, 3)), LabelSpace("traffic_type", ("chat", "voip")))


# --- persistence -------------------------------------------------------------------------

def test_json_round_trip():
    space = LabelSpace.taxonomy("traffic_type")
    assert LabelSpace.from_json(space.to_json()) == space


def test_num_classes_reports_the_dataset_not_the_space():
    ds = FlowDataset([_mk_flow(0, "chat"), _mk_flow(1, "voip")], "unit", "traffic_type")
    assert ds.num_classes == 2
    assert len(LabelSpace.from_dataset(ds)) == 8
