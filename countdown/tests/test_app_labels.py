"""ISCX capture-stem -> application normalisation."""

from __future__ import annotations

import pytest

from countdown.config import default_config
from countdown.data.app_labels import AppLabelRules, normalize_app_labels
from countdown.schema import FiveTuple, Flow

import numpy as np

FT = FiveTuple("10.0.0.1", 1, "10.0.0.2", 443, "tcp")


@pytest.fixture(scope="module")
def rules():
    return AppLabelRules.from_config(default_config())


def _flow(app, traffic_type="chat"):
    return Flow("f", FT, np.zeros(1), np.ones(1, dtype=np.int32),
                np.ones(1, dtype=np.int8),
                label_fields={"app": app, "traffic_type": traffic_type},
                meta={"source_file": "/a.pcap"})


@pytest.mark.parametrize("stem,expected", [
    # capture-instance suffixes collapse
    ("facebook_audio1a", "facebook"), ("facebook_audio2b", "facebook"),
    ("facebook_audio4", "facebook"),
    # Tor gateway spellings collapse onto the same service
    ("chat_gate_skype_chat", "skype"), ("voip_gate_skype_audio", "skype"),
    ("skypechat", "skype"), ("chat_skypechatgateway", "skype"),
    ("torvimeo1", "vimeo"), ("toryoutube2", "youtube"), ("torfacebook", "facebook"),
    ("tor_spotify2-1", "spotify"), ("audio_spotifygateway", "spotify"),
    ("hangout_chat", "hangouts"), ("hangouts_audio1a", "hangouts"),
])
def test_service_normalisation(rules, stem, expected):
    assert rules.service(stem) == expected


@pytest.mark.parametrize("stem,expected", [
    # ordering traps: each of these would map wrong under a naive rule order
    ("gmailchat1", "gmail"),                      # 'mail' must not win
    ("workstation_thunderbird_imap", "thunderbird"),   # before the imap/mail rule
    ("mail_gateway_thunderbird_pop", "thunderbird"),
    ("ftps_down_1a", "ftps"),                     # ftps before sftp before ftp
    ("sftp_filetransfer", "sftp"),
    ("ftp_filetransfer", "ftp"),
    ("email_imap_filetransfer", "email"),
    ("p2p_tor_p2p_vuze", "vuze"),                 # before the generic torrent rule
    ("browsing_gate_ssl_browsing", "browsing"),
])
def test_rule_order_is_load_bearing(rules, stem, expected):
    assert rules.service(stem) == expected


def test_activity_is_service_plus_modality(rules):
    assert rules.activity("facebook_audio1a", "voip") == "facebook_audio"
    assert rules.activity("facebook_video2b", "video_streaming") == "facebook_video"
    assert rules.activity("skype_file8", "file_transfer") == "skype_file"


def test_normalize_adds_both_fields_without_touching_app(rules):
    f = _flow("facebook_audio1a", "voip")
    normalize_app_labels([f], rules)
    assert f.label_fields["app"] == "facebook_audio1a"     # raw stem preserved
    assert f.label_fields["app_norm"] == "facebook"
    assert f.label_fields["app_activity"] == "facebook_audio"


def test_unmapped_stem_is_a_hard_error_reported_by_stem(rules):
    with pytest.raises(ValueError, match="matched no app rule"):
        normalize_app_labels([_flow("some_unknown_capture")], rules)
    # ...and is downgradeable, like the traffic_type rules
    normalize_app_labels([_flow("some_unknown_capture")], rules, on_unmapped="warn")


@pytest.mark.slow
def test_real_corpus_normalises_completely_and_recovers_the_paper_classes():
    """Every ISCX stem maps, and all 20 Okonkwo application classes come back."""
    from countdown.data import load

    vpn, tor = load("iscxvpn"), load("iscxtor")     # raises if any stem is unmapped

    acts = {f.label_fields["app_activity"] for f in vpn.flows if not f.label_fields["is_vpn"]}
    assert {"facebook_audio", "facebook_video", "hangouts_audio", "hangouts_video",
            "netflix_video", "skype_audio", "skype_video", "vimeo_video",
            "voipbuster_audio", "youtube_video"} <= acts

    vpn_apps = {f.label_fields["app_norm"] for f in vpn.flows if f.label_fields["is_vpn"]}
    assert {"email", "hangouts", "skype", "spotify", "voipbuster", "youtube"} <= vpn_apps

    tor_apps = {f.label_fields["app_norm"] for f in tor.flows if f.label_fields["is_tor"]}
    assert {"facebook", "skype", "spotify", "youtube"} <= tor_apps

    # the whole point: 135 stems collapse to a usable class set
    assert len({f.label_fields["app"] for f in vpn.flows}) > 100
    assert len({f.label_fields["app_norm"] for f in vpn.flows}) < 25


@pytest.mark.slow
def test_app_stays_invalid_as_a_target():
    """Normalisation adds fields; it must not quietly unlock the raw stem target."""
    from countdown.data import load

    ds = load("iscxvpn")
    with pytest.raises(ValueError, match="not a valid classification target"):
        ds.labels("app")
    ds.labels("app_norm")      # the normalised one works
