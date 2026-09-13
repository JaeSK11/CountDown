"""Pooled datasets, and the two invariants pooling must not silently violate."""

from __future__ import annotations

import numpy as np
import pytest

from qsentinel.config import Config
from qsentinel.data import pooled as pooled_mod
from qsentinel.data.pooled import available_pooled, load_pooled
from qsentinel.schema import FiveTuple, Flow, FlowDataset, LabelSpace

DATA_ROOT = Config.load().data_root


def _has(*names: str) -> bool:
    cfg = Config.load()
    return all((cfg.data_root / cfg.dataset(n)["root"]).exists() for n in names)


def _mk_flow(i: int, source: str, traffic_type: str = "chat", **fields) -> Flow:
    return Flow(
        flow_id=f"f{i}",
        five_tuple=FiveTuple("10.0.0.1", 1000 + i, "1.1.1.1", 443, "tcp"),
        timestamps=np.arange(4, dtype=np.float64),
        sizes=np.full(4, 100, dtype=np.int32),
        directions=np.array([1, -1, 1, -1], dtype=np.int8),
        label_fields={"traffic_type": traffic_type, **fields},
        meta={"source_file": source},
    )


def _fake_loader(monkeypatch, mapping: dict[str, FlowDataset]):
    """Stand in for the real loaders so the invariants can be tested without the corpus."""
    def fake_load(name, target=None, config=None, **kw):
        return mapping[name]

    monkeypatch.setattr("qsentinel.data.base.load", fake_load)


# --- config surface ----------------------------------------------------------------------

def test_iscx_pooled_is_configured():
    assert "iscx_pooled" in available_pooled()
    cfg = Config.load()
    spec = cfg.pooled_spec("iscx_pooled")
    assert spec["members"] == ["iscxvpn", "iscxtor"]
    assert cfg.group_key("iscx_pooled") == "source_file"


def test_group_keys_are_declared_outside_the_dataset_specs():
    """They must not enter flow_key, or every parsed cache is orphaned."""
    cfg = Config.load()
    assert cfg.group_key("cstnet") == "capture_day"
    assert "group_key" not in cfg.dataset("cstnet")
    assert cfg.allows_vacuous_groups("mobileapp") is True
    assert cfg.allows_vacuous_groups("cstnet") is False


# --- invariants --------------------------------------------------------------------------

def test_pooling_unions_the_classes(monkeypatch):
    a = FlowDataset([_mk_flow(0, "/a/1.pcap", "voip")], "a", "traffic_type")
    b = FlowDataset([_mk_flow(1, "/b/1.pcap", "browsing")], "b", "traffic_type")
    _fake_loader(monkeypatch, {"a": a, "b": b})

    ds = load_pooled(["a", "b"], target="traffic_type")
    assert len(ds) == 2
    assert ds.name == "a+b"
    assert sorted(ds.label_names()) == ["browsing", "voip"]
    # each constituent alone is 1 class; the canonical space is 8 regardless
    assert len(LabelSpace.from_dataset(ds)) == 8


def test_a_constituent_missing_the_target_is_refused(monkeypatch):
    a = FlowDataset([_mk_flow(0, "/a/1.pcap", "voip")], "a", "traffic_type")
    bad = Flow(
        flow_id="x", five_tuple=FiveTuple("1.1.1.1", 1, "2.2.2.2", 2, "tcp"),
        timestamps=np.zeros(2), sizes=np.zeros(2, dtype=np.int32),
        directions=np.ones(2, dtype=np.int8),
        label_fields={"app": "only_app"}, meta={"source_file": "/b/1.pcap"},
    )
    b = FlowDataset([bad], "b", "app")
    _fake_loader(monkeypatch, {"a": a, "b": b})

    with pytest.raises(ValueError, match="have no 'traffic_type' label field"):
        load_pooled(["a", "b"], target="traffic_type")


def test_colliding_source_files_are_refused(monkeypatch):
    """Two constituents reusing a capture id would merge unrelated captures into one group."""
    a = FlowDataset([_mk_flow(0, "/shared/1.pcap", "voip")], "a", "traffic_type")
    b = FlowDataset([_mk_flow(1, "/shared/1.pcap", "chat")], "b", "traffic_type")
    _fake_loader(monkeypatch, {"a": a, "b": b})

    with pytest.raises(ValueError, match="capture groups would collide"):
        load_pooled(["a", "b"], target="traffic_type")


def test_invalid_targets_survive_pooling(monkeypatch):
    a = FlowDataset([_mk_flow(0, "/a/1.pcap")], "a", "traffic_type", invalid_targets=["app"])
    b = FlowDataset([_mk_flow(1, "/b/1.pcap")], "b", "traffic_type")
    _fake_loader(monkeypatch, {"a": a, "b": b})

    ds = load_pooled(["a", "b"], target="traffic_type")
    with pytest.raises(ValueError, match="not a valid classification target"):
        ds.labels("app")


def test_nested_pools_are_refused():
    with pytest.raises(ValueError, match="themselves pooled aliases"):
        load_pooled(["iscx_pooled", "iscxvpn"])


def test_empty_pool_is_refused():
    with pytest.raises(ValueError, match="at least one dataset"):
        load_pooled([])


def test_filter_preserves_invalid_targets():
    """Regression: filter() used to drop the guard, so any filtered copy lost it."""
    flows = [_mk_flow(i, f"/a/{i}.pcap", "chat" if i < 8 else "voip") for i in range(10)]
    ds = FlowDataset(flows, "unit", "traffic_type", invalid_targets=["app"])
    assert ds.filter(min_samples_per_class=5).invalid_targets == frozenset({"app"})


# --- against the real corpus -------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _has("iscxvpn", "iscxtor"), reason="ISCX corpus not present")
def test_real_pooled_iscx_matches_the_documented_shape():
    from qsentinel.data import load

    ds = load("iscx_pooled", target="traffic_type")
    assert len(ds) == 70_523
    assert ds.num_classes == 8
    g = ds.groups("source_file")
    assert len(set(g)) == 235

    space = LabelSpace.taxonomy("traffic_type")
    y = ds.labels(space=space)
    # p2p is the thinnest class by capture count and the reason pooling is mandatory
    p2p_groups = set(g[y == space.index("p2p")])
    assert len(p2p_groups) == 10
