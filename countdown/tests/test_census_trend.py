"""Phase 9: the census trend (per-run summary, churn, panel check) and the scheduler's due
time.  Synthetic run directories; no network."""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


trend = _load("census_trend")
sched = _load("census_scheduler")

FIELDS = ["site", "host", "profile", "client_ok", "client_protocol", "client_group",
          "client_leaf_key", "pqc_verdict", "pqc_basis", "kem_group", "quantum_safe"]


def _run(root: Path, day: str, rows: dict[str, dict[str, bool | None]], tranco="L1") -> Path:
    """rows: site -> {profile: True (safe) / False (classical) / None (no handshake)}."""
    d = root / day
    d.mkdir(parents=True)
    profiles = sorted({p for r in rows.values() for p in r})
    prof = {}
    for p in profiles:
        cap = [r[p] for r in rows.values() if r.get(p) is not None]
        prof[p] = {"handshake_ok": len(cap), "captured": len(cap), "quantum_safe": sum(cap),
                   "quantum_safe_share": round(sum(cap) / len(cap), 4) if cap else 0.0,
                   "passive_group_vs_client": {"wrong": 0}}
    (d / "census.json").write_text(json.dumps(
        {"meta": {"date": day, "tranco_list": tranco, "sites": len(rows)}, "profiles": prof}))
    with open(d / "sites.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        for site, r in rows.items():
            for p in profiles:
                v = r.get(p)
                w.writerow({"site": site, "host": site, "profile": p, "client_ok": v is not None,
                            "pqc_verdict": "" if v is None else ("hybrid" if v else "classical"),
                            "quantum_safe": "" if v is None else v})
    return d


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(trend, "RUN_DIRS", [])
    monkeypatch.setattr(trend, "SCHEDULED", tmp_path)
    monkeypatch.setattr(trend, "ROOT", tmp_path)
    return tmp_path


def test_either_share_and_churn(runs):
    _run(runs, "2026-01-01", {"a": {"tls13": True}, "b": {"tls13": False}, "c": {"tls13": False}})
    _run(runs, "2026-01-08", {"a": {"tls13": False, "quic": None},
                              "b": {"tls13": True, "quic": True},
                              "c": {"tls13": False, "quic": True}})
    res = trend.build(trend.find_runs())
    p0, p1 = res["points"]
    assert p0["quic"] is None and p0["either"] is None
    assert p1["tcp"]["safe"] == 1 and p1["quic"]["share"] == 1.0
    assert p1["either"] == {"sites": 3, "safe": 2, "share": pytest.approx(2 / 3, abs=1e-6)}
    (c,) = res["churn"]
    assert c["profile"] == "tls13" and c["became_safe"] == ["b"] and c["became_classical"] == ["a"]
    md = trend.markdown(res)
    assert "66.7% (2/3)" in md and "| 2026-01-01 → 2026-01-08 | tls13 | 3 |" in md


def test_partial_runs_are_not_points(runs):
    _run(runs, "2026-01-01", {"a": {"tls13": True}})
    _run(runs, "2026-01-08.partial", {"a": {"tls13": False}})
    assert [p["date"] for p in trend.build(trend.find_runs())["points"]] == ["2026-01-01"]


def test_only_one_tranco_panel_is_compared(runs):
    _run(runs, "2026-01-01", {"a": {"tls13": True}}, tranco="L1")
    _run(runs, "2026-01-08", {"a": {"tls13": True}}, tranco="L1")
    _run(runs, "2026-01-15", {"a": {"tls13": False}}, tranco="L2")
    res = trend.build(trend.find_runs())
    assert res["tranco_list"] == "L1" and len(res["points"]) == 2


def test_figure_renders(runs, tmp_path):
    _run(runs, "2026-01-01", {"a": {"tls13": True, "quic": True}, "b": {"tls13": False, "quic": None}})
    paths = trend.figure(trend.build(trend.find_runs()), tmp_path / "fig")
    assert all(p.stat().st_size > 0 for p in paths)


def test_next_due():
    assert sched.next_due(None, 7, 3) == datetime.min.replace(tzinfo=timezone.utc)
    assert sched.next_due(date(2026, 11, 9), 7, 3) == datetime(2026, 11, 16, 3, tzinfo=timezone.utc)
    assert sched.next_due(date(2026, 12, 28), 7, 0) == datetime(2027, 1, 4, 0, tzinfo=timezone.utc)
