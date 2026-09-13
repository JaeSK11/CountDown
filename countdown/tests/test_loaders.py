"""Loader, label-mapping and cache tests (tasks 0.5-0.9, 0.11 DoD).

Discovery is exercised against the real corpus (it only stats filenames, no parsing), so
these tests are the guard that every one of the 235 ISCXVPN/ISCXTor captures maps to a
label.  Anything that would require parsing gigabytes is kept to mobileapp or synthetic data.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from countdown.config import Config, default_config
from countdown.data import available_datasets, get_loader, load
from countdown.data.base import _dict_to_flow, _flow_to_dict
from countdown.schema import TRAFFIC_TYPES, TUNNEL_TYPES, FiveTuple, Flow, FlowDataset


def _has_data(name: str) -> bool:
    try:
        return default_config().dataset_path(name).exists()
    except Exception:
        return False


needs_corpus = pytest.mark.skipif  # readability alias


# --- registry --------------------------------------------------------------------------


def test_all_five_phase0_loaders_are_registered():
    assert set(available_datasets()) == {
        "cstnet", "iscxtor", "iscxvpn", "mobileapp", "postquantumtls"
    }


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        get_loader("nope")


# --- label mapping completeness ---------------------------------------------------------


@pytest.mark.skipif(not _has_data("iscxvpn"), reason="ISCXVPN2016 not present")
def test_iscxvpn_every_capture_maps_to_a_label():
    loader = get_loader("iscxvpn")
    items = loader.discover()
    assert loader.unmapped == []          # the "fail loudly" rule from the phase spec
    assert len(items) == 140              # 102 .pcap + 38 .pcapng
    for it in items:
        assert it.label_fields["traffic_type"] in TRAFFIC_TYPES
        assert it.label_fields["tunnel_type"] in TUNNEL_TYPES
        assert it.label_fields["is_tor"] is False


@pytest.mark.skipif(not _has_data("iscxvpn"), reason="ISCXVPN2016 not present")
def test_iscxvpn_vpn_prefix_sets_tunnel_type():
    items = get_loader("iscxvpn").discover()
    vpn = [i for i in items if i.label_fields["is_vpn"]]
    plain = [i for i in items if not i.label_fields["is_vpn"]]
    assert len(vpn) == 31 and len(plain) == 109
    assert all(i.path.stem.lower().startswith("vpn_") for i in vpn)
    assert all(i.label_fields["tunnel_type"] == "vpn" for i in vpn)
    assert all(i.label_fields["tunnel_type"] == "none" for i in plain)


@pytest.mark.skipif(not _has_data("iscxvpn"), reason="ISCXVPN2016 not present")
@pytest.mark.parametrize(
    "stem,expected",
    [
        ("gmailchat1", "chat"),          # must not fall through to the "mail" rule
        ("email1a", "email"),
        ("skype_file3", "file_transfer"),  # must beat the "audio"/"chat" rules
        ("facebook_audio1a", "voip"),
        ("facebook_video1a", "video_streaming"),
        ("spotify1", "audio_streaming"),
        ("youtubeHTML5_1", "video_streaming"),
        ("scpDown4", "file_transfer"),
        ("bittorrent", "p2p"),
        ("voipbuster_4a", "voip"),
    ],
)
def test_iscxvpn_tricky_stems(stem, expected):
    loader = get_loader("iscxvpn")
    assert loader.map_traffic_type(stem.lower(), loader.spec["label_rules"]) == expected


@pytest.mark.skipif(not _has_data("iscxtor"), reason="ISCXTor2016 not present")
def test_iscxtor_every_capture_maps_to_a_label():
    loader = get_loader("iscxtor")
    items = loader.discover()
    assert loader.unmapped == []
    assert len(items) == 95
    tor = [i for i in items if i.label_fields["is_tor"]]
    assert len(tor) == 51
    assert all(i.label_fields["tunnel_type"] == "tor" for i in tor)
    for it in items:
        assert it.label_fields["traffic_type"] in TRAFFIC_TYPES


@pytest.mark.skipif(not _has_data("iscxtor"), reason="ISCXTor2016 not present")
@pytest.mark.parametrize(
    "stem,expected",
    [
        ("mail_gate_email_imap_filetransfer", "email"),   # prefix beats "filetransfer"
        ("file-transfer_gate_sftp_filetransfer", "file_transfer"),
        ("pop_filetransfer", "email"),
        ("sftp_filetransfer", "file_transfer"),
        ("workstation_thunderbird_imap", "email"),
        ("torgoogle", "browsing"),
        ("torrent01", "p2p"),
        ("tor_spotify2-1", "audio_streaming"),
        ("toryoutube1", "video_streaming"),
        ("skype_voice_workstation", "voip"),
        ("ssl", "browsing"),
    ],
)
def test_iscxtor_tricky_stems(stem, expected):
    loader = get_loader("iscxtor")
    assert loader.map_traffic_type(stem, loader.spec["label_rules"]) == expected


def test_unmapped_stem_fails_loudly():
    """A capture matching no rule must raise, never be silently dropped."""
    cfg = Config.load()
    cfg.datasets = copy.deepcopy(cfg.datasets)
    cfg.datasets["iscxvpn"]["label_rules"] = [{"pattern": "^nothing$", "traffic_type": "chat"}]
    loader = get_loader("iscxvpn", cfg)
    with pytest.raises(ValueError, match="matched no label rule"):
        loader.discover()


def test_unmapped_can_be_downgraded_to_a_warning():
    cfg = Config.load()
    cfg.datasets = copy.deepcopy(cfg.datasets)
    cfg.datasets["iscxvpn"]["label_rules"] = [{"pattern": "^nothing$", "traffic_type": "chat"}]
    cfg.datasets["iscxvpn"]["on_unmapped"] = "warn"
    loader = get_loader("iscxvpn", cfg)
    assert loader.discover() == []
    assert len(loader.unmapped) == 140


# --- discovery for the remaining datasets ------------------------------------------------


@pytest.mark.skipif(not _has_data("cstnet"), reason="CSTNET-TLS1.3 not present")
def test_cstnet_discovery_labels_by_directory_and_honours_subsampling():
    cfg = Config.load()
    cfg.datasets = copy.deepcopy(cfg.datasets)
    cfg.datasets["cstnet"]["filters"] = {"top_k_apps": 3, "max_pcaps_per_app": 4}
    items = get_loader("cstnet", cfg).discover()
    assert len(items) == 12
    assert len({i.label_fields["app"] for i in items}) == 3
    for it in items:
        assert it.label_fields["app"] == it.path.parent.name


@pytest.mark.skipif(not _has_data("cstnet"), reason="CSTNET-TLS1.3 not present")
def test_cstnet_etbert_mode_is_an_unimplemented_phase4_hook():
    cfg = Config.load()
    cfg.datasets = copy.deepcopy(cfg.datasets)
    cfg.datasets["cstnet"]["ingest_mode"] = "etbert"
    with pytest.raises(NotImplementedError, match="Phase-4"):
        get_loader("cstnet", cfg).discover()


@pytest.mark.skipif(not _has_data("postquantumtls"), reason="PostQuantumTLS not present")
def test_postquantumtls_discovery_labels_by_package():
    items = get_loader("postquantumtls").discover()
    assert len(items) == 90
    assert all(i.label_fields["app"] == i.path.stem for i in items)


# --- mobileapp end to end (small enough to parse in a test) ------------------------------


@pytest.mark.skipif(not _has_data("mobileapp"), reason="MobileApp dataset not present")
def test_mobileapp_loads_to_xy(tmp_path):
    cfg = Config.load()
    cfg.cache_dir = tmp_path
    ds = load("mobileapp", target="activity", config=cfg)
    assert len(ds) == 368

    X = ds.features("flow_stats")
    y = ds.labels()
    assert X.shape[0] == y.shape[0] == 368
    assert X.shape[1] == 56
    assert np.isfinite(X).all()
    assert len(ds.label_names()) == 92          # App/Activity classes
    assert len(ds.label_names("app")) == 8      # apps

    seq = ds.features("packet_seq")
    assert seq.shape == (368, 32, 2)


@pytest.mark.skipif(not _has_data("mobileapp"), reason="MobileApp dataset not present")
def test_mobileapp_direction_is_inferred_per_capture(tmp_path):
    cfg = Config.load()
    cfg.cache_dir = tmp_path
    ds = load("mobileapp", config=cfg)
    for f in ds.flows[:20]:
        assert set(np.unique(f.directions)) <= {1, -1}
        assert (f.directions > 0).any() and (f.directions < 0).any()
        assert f.five_tuple.src_ip == f.meta["client_mac"]
        assert f.five_tuple.l4_proto == "wlan"


# --- cache ------------------------------------------------------------------------------


def _mk_flow(i: int) -> Flow:
    n = 5 + i
    return Flow(
        flow_id=f"f{i}",
        five_tuple=FiveTuple("10.0.0.1", 1000 + i, "1.2.3.4", 443, "tcp"),
        timestamps=np.arange(n, dtype=np.float64) * 0.25,
        sizes=np.full(n, 100 + i, dtype=np.int32),
        directions=np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8),
        dataset="unit", label="chat",
        label_fields={"traffic_type": "chat", "is_vpn": False},
        meta={"source_file": f"/tmp/{i}.pcap"},
    )


def test_flow_survives_a_dict_round_trip():
    original = _mk_flow(3)
    restored = _dict_to_flow(_flow_to_dict(original))
    assert restored.flow_id == original.flow_id
    assert restored.five_tuple == original.five_tuple
    assert np.array_equal(restored.sizes, original.sizes)
    assert np.array_equal(restored.directions, original.directions)
    assert np.allclose(restored.timestamps, original.timestamps)
    assert restored.label_fields == original.label_fields
    assert restored.meta == original.meta


@pytest.mark.skipif(not _has_data("mobileapp"), reason="MobileApp dataset not present")
def test_second_load_reads_the_cache_instead_of_reparsing(tmp_path):
    cfg = Config.load()
    cfg.cache_dir = tmp_path
    loader = get_loader("mobileapp", cfg)
    first = loader.load()
    cache_file = loader._cache_file()
    assert cache_file.exists()

    reloaded = get_loader("mobileapp", cfg)
    reloaded.discover = lambda: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("discover() must not run when a valid cache exists")
    )
    second = reloaded.load()

    assert len(second) == len(first)
    assert np.array_equal(second.features("flow_stats"), first.features("flow_stats"))


@pytest.mark.skipif(not _has_data("mobileapp"), reason="MobileApp dataset not present")
def test_changing_flow_params_invalidates_the_cache_key(tmp_path):
    cfg = Config.load()
    cfg.cache_dir = tmp_path
    a = get_loader("mobileapp", cfg)._cache_file()
    cfg2 = Config.load(overrides={"flow.min_packets": 99})
    cfg2.cache_dir = tmp_path
    b = get_loader("mobileapp", cfg2)._cache_file()
    assert a != b


# --- FlowDataset semantics ---------------------------------------------------------------


def test_labels_use_the_canonical_taxonomy_ordering():
    flows = [_mk_flow(0), _mk_flow(1)]
    flows[1].label_fields["traffic_type"] = "browsing"
    ds = FlowDataset(flows, "unit", "traffic_type")
    # browsing precedes chat in TRAFFIC_TYPES, regardless of insertion order
    assert ds.label_names() == ["browsing", "chat"]
    assert ds.labels().tolist() == [1, 0]


def test_boolean_targets_encode_as_false_true():
    flows = [_mk_flow(0), _mk_flow(1)]
    flows[1].label_fields["is_vpn"] = True
    ds = FlowDataset(flows, "unit", "traffic_type")
    assert ds.labels("is_vpn").tolist() == [0, 1]
    assert ds.label_names("is_vpn") == ["false", "true"]


def test_missing_target_field_raises():
    ds = FlowDataset([_mk_flow(0)], "unit", "traffic_type")
    with pytest.raises(KeyError, match="no label field"):
        ds.labels("activity")


def test_split_is_stratified_disjoint_and_reproducible():
    flows = []
    for i in range(60):
        f = _mk_flow(i)
        f.label_fields["traffic_type"] = ["chat", "email", "voip"][i % 3]
        flows.append(f)
    ds = FlowDataset(flows, "unit", "traffic_type")
    s = ds.split(test_size=0.2, val_size=0.2, seed=7)

    all_idx = np.concatenate([s["train"], s["val"], s["test"]])
    assert len(np.unique(all_idx)) == 60          # disjoint and complete
    y = ds.labels()
    for part in ("train", "val", "test"):
        assert len(np.unique(y[s[part]])) == 3    # every class represented
    again = ds.split(test_size=0.2, val_size=0.2, seed=7)
    assert np.array_equal(s["test"], again["test"])


def test_filter_drops_rare_classes():
    flows = []
    for i in range(10):
        f = _mk_flow(i)
        f.label_fields["traffic_type"] = "chat" if i < 8 else "email"
        flows.append(f)
    ds = FlowDataset(flows, "unit", "traffic_type")
    assert len(ds.filter(min_samples_per_class=5)) == 8
    assert len(ds.filter(top_k_classes=1)) == 8
    assert len(ds.filter(min_samples_per_class=1)) == 10
