"""scripts/ensemble_seeds.py: which runs count per seed, and the paired statistics.
Synthetic run directories; no training."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_spec = importlib.util.spec_from_file_location("ensemble_seeds", _SCRIPTS / "ensemble_seeds.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

DEPLOY = ["flow_gbdt", "seq_cnn"]


def _run(root: Path, run_id: str, seed: int, gbdt: float, vote: float, **args) -> None:
    d = root / "E3" / run_id
    d.mkdir(parents=True)
    (d / "ensemble.json").write_text(json.dumps(
        {"args": {"seed": seed, "members": None, "max_flows": None, **args}, "data": {"n_test": 92}}))
    rows = [{"members": ["flow_gbdt"], "combiner": "-", "calibration": "none", "macro_f1": gbdt},
            {"members": ["seq_cnn", "flow_gbdt"], "combiner": "vote", "calibration": "temperature",
             "macro_f1": vote}]
    (d / "ablation.json").write_text(json.dumps({
        "rows": rows,
        "best_single": {"members": ["flow_gbdt"], "macro_f1": gbdt},
        "selected": {"combiner": "vote", "calibration": "temperature", "macro_f1": vote}}))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "ROOT", tmp_path)
    return tmp_path


def test_latest_run_per_seed_wins_and_smoke_runs_are_skipped(root):
    _run(root, "20260101-000000", 1, 0.40, 0.45)
    _run(root, "20260102-000000", 1, 0.41, 0.50)          # rerun of seed 1: this one counts
    _run(root, "20260103-000000", 2, 0.42, 0.47)
    _run(root, "20260104-000000", 3, 0.90, 0.10, max_flows=500)        # smoke run
    _run(root, "20260105-000000", 4, 0.90, 0.10, members=["flow_gbdt"])  # member subset
    seeds = es.collect("E3", root, DEPLOY)
    assert [s["seed"] for s in seeds] == [1, 2]
    assert seeds[0]["gbdt"] == 0.41 and seeds[0]["deployed"] == 0.50
    assert seeds[0]["run"] == "E3/20260102-000000"


def test_paired_statistics():
    p = es.paired([0.5, 0.6, 0.7, 0.4], [0.4, 0.5, 0.7, 0.5])
    assert p["n"] == 4 and (p["wins"], p["ties"], p["losses"]) == (2, 1, 1)
    assert p["mean"] == pytest.approx(0.025)
    assert p["ci95"][0] < p["mean"] < p["ci95"][1]
    assert es._sign_p(10, 0) == pytest.approx(2 / 1024)
    assert es._sign_p(0, 0) == 1.0


def test_summary_and_markdown(root):
    for i, (g, v) in enumerate([(0.40, 0.50), (0.45, 0.52), (0.50, 0.55)], start=1):
        _run(root, f"2026010{i}-000000", i, g, v)
    res = es.summarise(es.collect("E3", root, DEPLOY))
    assert res["n_seeds"] == 3
    assert res["deployed_minus_gbdt"]["wins"] == 3 and res["stats"]["deployed"]["max"] == 0.55
    md = es.markdown("E3", DEPLOY, res)
    assert "| deployed − flow_gbdt | +0.073 |" in md and "3 / 0 / 0" in md
