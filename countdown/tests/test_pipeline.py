"""Phase 6.1/6.2: the pipeline's branch, routing and fallback, and the analyze CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from countdown.crypto.pqc_verdict import Verdict
from countdown.ensemble.calibration import ProbaCalibrator
from countdown.ensemble.combiner import SoftVote
from countdown.models.base import ModelRegistry
from countdown.pipeline import DeployedExpert, MemberRecipe, Pipeline, PipelineConfig
from countdown.schema import FiveTuple, Flow, FlowDataset, LabelSpace


def _flow(i: int, hint: str, big: bool) -> Flow:
    rng = np.random.default_rng(i)
    n = 12
    sizes = (rng.integers(900, 1400, n) if big else rng.integers(40, 200, n)).astype(np.int32)
    return Flow(flow_id=f"f{i}", five_tuple=FiveTuple("10.0.0.1", 40000 + i, "10.0.0.2", 443, "tcp"),
                timestamps=np.cumsum(rng.uniform(0.001, 0.05, n)), sizes=sizes,
                directions=np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8), l7_hint=hint)


def _expert(name: str, target: str, classes: list[str]) -> DeployedExpert:
    flows = [_flow(i, "tls", big=bool(i % 2)) for i in range(120)]
    y = np.array([i % 2 for i in range(120)])
    X = FlowDataset(flows, name="t", default_target=target).features("flow_stats")
    space = LabelSpace.from_names(target, classes)
    m = ModelRegistry.create("flow_gbdt", label_space=space, n_estimators=20, num_leaves=4)
    m.fit(X, y)
    p = m.predict_proba(X)
    cal = ProbaCalibrator("temperature").fit(p, y)
    return DeployedExpert(name=name, target=target, label_space=space, members={"flow_gbdt": m},
                          recipes={"flow_gbdt": MemberRecipe("flow_stats")},
                          calibrators={"flow_gbdt": cal}, combiner=SoftVote().fit([cal.transform(p)], y),
                          fallback_threshold={"full": 0.5, "fast": 0.5})


@pytest.fixture
def experts():
    return {"E1": _expert("E1", "traffic_type", ["chat", "file_transfer"]),
            "E2": _expert("E2", "app", ["small.example", "big.example"]),
            "E3": _expert("E3", "activity", ["scroll", "upload"])}


@pytest.fixture
def verdicts(monkeypatch):
    """Stub stages 2-3: the flow id says which verdict it gets."""
    def fake(flow):
        label = "hybrid" if flow.flow_id.startswith("h") else "classical"
        return Verdict(label=label, basis="negotiated", confidence=1.0)
    monkeypatch.setattr("countdown.crypto.enrich_flow", fake)


def test_quantum_safe_is_skipped_classical_tls_goes_to_e2(experts, verdicts):
    flows = [_flow(1, "tls", True), _flow(2, "tor", False)]
    flows.append(_flow(3, "tls", True)); flows[-1].flow_id = "h3"     # hybrid
    recs = Pipeline(PipelineConfig(fallback=False), experts=experts).analyze_flows(flows)
    assert recs[0]["expert"] == "E2" and recs[0]["target"] == "app"
    assert recs[0]["traffic_class"] == "big.example" and recs[0]["route_reason"] == "hint"
    assert recs[1]["expert"] == "E1" and recs[1]["target"] == "traffic_type"
    assert recs[2]["quantum_safe"] and recs[2]["expert"] is None
    assert recs[2]["route_reason"] == "skipped:quantum_safe" and recs[2]["traffic_class"] is None
    tagged = Pipeline(PipelineConfig(classify_quantum_safe=True, fallback=False),
                      experts=experts).analyze_flows(flows)
    assert tagged[2]["traffic_class"] is not None


def test_low_confidence_e2_falls_back_to_e1_and_keeps_e2s_guess(experts, verdicts):
    flows = [_flow(1, "tls", True), _flow(2, "tls", False)]
    recs = Pipeline(PipelineConfig(fallback_threshold=1.01), experts=experts).analyze_flows(flows)
    for r in recs:
        assert r["expert"] == "E1" and r["route_reason"] == "fallback" and r["target"] == "traffic_type"
        assert r["fallback_from"]["expert"] == "E2" and r["fallback_from"]["label"].endswith(".example")
    kept = Pipeline(PipelineConfig(fallback_threshold=0.0), experts=experts).analyze_flows(flows)
    assert all(r["expert"] == "E2" and "fallback_from" not in r for r in kept)


def test_context_overrides_the_hint_and_is_never_second_guessed(experts, verdicts):
    flows = [_flow(1, "tls", True), _flow(2, "tor", False)]
    recs = Pipeline(PipelineConfig(context="mobile", fallback_threshold=1.01),
                    experts=experts).analyze_flows(flows)
    assert [r["expert"] for r in recs] == ["E3", "E3"]
    assert all(r["route_reason"] == "context" and "fallback_from" not in r for r in recs)


def test_redact_and_member_breakdown(experts, verdicts):
    recs = Pipeline(PipelineConfig(redact=True, include_members=True, fallback=False),
                    experts=experts).analyze_flows([_flow(1, "tls", True)])
    assert recs[0]["five_tuple"] is None and recs[0]["sni"] is None
    assert set(recs[0]["members"]) == {"flow_gbdt"}


def test_deployed_expert_round_trips_and_cli_writes_csv(experts, tmp_path, monkeypatch):
    for n, e in experts.items():
        e.save(tmp_path / "store" / f"{n}.joblib")
    back = DeployedExpert.load(tmp_path / "store" / "E2.joblib")
    assert back.label_space.names == experts["E2"].label_space.names
    from countdown import cli

    monkeypatch.setattr("countdown.pipeline.Pipeline.analyze_pcap",
                        lambda self, p: self.analyze_flows([_flow(1, "tls", True)]))
    monkeypatch.setattr("countdown.crypto.enrich_flow",
                        lambda f: Verdict(label="classical", basis="negotiated", confidence=1.0))
    out = tmp_path / "v.csv"
    assert cli.main(["analyze", "x.pcap", "--out", str(out), "--store", str(tmp_path / "store"),
                     "--no-fallback"]) == 0
    head, row = out.read_text().splitlines()[:2]
    assert head.startswith("flow_id,five_tuple") and "big.example" in row
    jl = tmp_path / "v.jsonl"
    cli.main(["analyze", "x.pcap", "--out", str(jl), "--store", str(tmp_path / "store"), "--redact"])
    assert json.loads(jl.read_text().splitlines()[0])["source"] is None


def test_byte_matrices_join_flows_on_the_canonical_five_tuple(two_flow_pcap):
    from countdown.flows.reassembly import Reassembler
    from countdown.pipeline import byte_matrix_for_flows
    from countdown.sources.pcap import PcapSource

    with PcapSource(two_flow_pcap) as src:
        flows = list(Reassembler(min_packets=1).run(src))
    rows = byte_matrix_for_flows(two_flow_pcap, flows)
    assert rows.shape == (len(flows), 50, 190)
    has = rows[:, 0, 0] >= 0
    assert has.any()                                    # payload-bearing flows got their bytes
