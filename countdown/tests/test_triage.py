"""Phase 6.5: triage buckets, shares by flows and bytes, tiers, breakdown error."""

from __future__ import annotations

import numpy as np

from countdown.eval.triage import (
    AT_RISK, AT_RISK_CEILING, SAFE, _bucket, breakdown_error, to_markdown, triage_records,
)


def test_buckets_respect_the_strength_of_evidence():
    assert _bucket("hybrid", "negotiated") == SAFE
    assert _bucket("hybrid", "offered") == "pqc_offered_undecided"
    assert _bucket("classical", "negotiated") == AT_RISK
    # Client key-shared only classical groups: certain only if no PQ group is even listed.
    assert _bucket("classical", "offered", any_pqc_supported=False) == AT_RISK_CEILING
    assert _bucket("classical", "offered", any_pqc_supported=True) == "pqc_offered_undecided"
    assert _bucket("classical", "offered", None) == "pqc_offered_undecided"
    assert _bucket("not_observable", "none") == "not_observable"   # never counted as exposed
    assert _bucket("unknown", "negotiated") == "unknown"


def _rec(verdict, basis, cls, target, nbytes, listed=False):
    return {"pqc_verdict": verdict, "pqc_basis": basis, "traffic_class": cls, "target": target,
            "bytes": nbytes, "kem_group": "x25519", "tls_version": "1.3", "pqc_listed": listed}


def test_records_triage_counts_flows_and_bytes_and_tiers_traffic_types():
    tiers = {"migrate_first": ["file_transfer"], "low_priority": ["video_streaming"]}
    recs = [_rec("classical", "negotiated", "file_transfer", "traffic_type", 900),
            _rec("classical", "negotiated", "video_streaming", "traffic_type", 50),
            _rec("classical", "negotiated", "video_streaming", "traffic_type", 50),
            _rec("hybrid", "negotiated", None, None, 1000),
            _rec("not_observable", "none", "chat", "traffic_type", 0)]
    t = triage_records(recs, tiers=tiers)
    assert t["flows"] == 5 and t["verdict_by_flows"]["at_risk"]["n"] == 3
    assert t["exposed_share_flows"] == 0.6 and t["exposed_share_bytes"] == 0.5
    tt = t["at_risk_classes_by_target"]["traffic_type"]
    assert tt["at_risk_tiers_by_flows"]["low_priority"]["share"] == round(2 / 3, 4)
    assert tt["at_risk_tiers_by_bytes"]["migrate_first"]["share"] == 0.9   # volume flips it
    assert "| at_risk | 3 |" in to_markdown("x", t)


def test_breakdown_error_is_total_variation_of_the_class_mix():
    same = breakdown_error(["a", "b", "b"], ["b", "a", "b"])         # every flow wrong, mix right
    assert same["total_variation"] == 0.0
    off = breakdown_error(["a", "a", "b", "b"], ["a", "a", "a", "a"])
    assert off["total_variation"] == 0.5 and off["worst_gap"] in (0.5, -0.5)
