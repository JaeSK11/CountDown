#!/usr/bin/env python
"""PQC-triage statistics over the labelled datasets (Phase 6, task 6.5).

    python scripts/run_triage.py           # -> runs/triage/<id>/{triage.json,triage.md}

Three parts:

1. **Ground truth, per dataset** -- every flow's Phase-2 verdict (in flows and bytes) and the
   class mix of the exposed flows, with readiness tiers for traffic types.
2. **The self-generated PQC captures** (``data/_pqc_synth``) through the real pipeline: the
   only quantum-safe handshakes we have (none of the public datasets contains one).
3. **How faithful the deployed tool's breakdown is**: on E1's and E2's test folds, the class
   mix of exposed flows as the deployed expert predicts it vs as labelled (total variation),
   from the Phase-5 cached test predictions -- the error bar on what ``countdown triage``
   reports for an operator's capture.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from countdown.data import load  # noqa: E402
from countdown.eval.tiers import load_tiers  # noqa: E402
from countdown.eval.triage import (  # noqa: E402
    EXPOSED, _bucket, breakdown_error, to_markdown, triage_flows, triage_records,
)

DATASETS = [("iscxvpn", "traffic_type"), ("iscxtor", "traffic_type"),
            ("cstnet", "app"), ("postquantumtls", "app")]


def exposed_mask(flows) -> np.ndarray:
    from countdown.crypto import enrich_flow, kem_registry, parse_handshake

    out = []
    for f in flows:
        v = enrich_flow(f)
        listed = (any(kem_registry.is_quantum_resistant(g) for g in parse_handshake(f).offered_groups)
                  if v.basis == "offered" else None)
        out.append(_bucket(v.label, v.basis, listed) in EXPOSED)
    return np.array(out)


def fidelity(name: str, tiers) -> dict:
    """Predicted vs true class mix of the exposed test flows of one expert."""
    from countdown.ensemble.views import PER_FLOW_OVERRIDES, build_expert_data, _graph_rows
    from countdown.experts import get_expert
    from countdown.pipeline.artifacts import DeployedExpert

    exp = DeployedExpert.load(ROOT / "models_store" / f"{name}.joblib")
    spec = get_expert(name)
    data = build_expert_data(spec, members=list(exp.members), seed=42,
                             overrides=PER_FLOW_OVERRIDES[name])
    flows = load(spec.dataset).flows
    if "graph_gnn" in exp.members:
        _, keep = _graph_rows(flows, ROOT / "cache/tfegnn/cstnet")
        flows = [flows[i] for i in keep]
    te = data.test_rows
    preds = ROOT / "runs/ensemble" / name / "preds-s42"
    member = {m: np.load(preds / f"{m}.npz")["test"] for m in exp.members}
    proba = exp.combine(member)
    names = exp.label_space.names
    mask = exposed_mask([flows[r] for r in te])
    true = [names[i] for i in data.y[te][mask]]
    pred = [names[i] for i in proba.argmax(1)[mask]]
    out = {"expert": name, "test_flows": int(len(te)), "exposed_test_flows": int(mask.sum()),
           "class_mix": breakdown_error(true, pred),
           "accuracy_on_exposed": round(float(np.mean(np.array(true) == np.array(pred))), 4)}
    if spec.target == "traffic_type":
        lookup = {c: t for t, cs in tiers.items() for c in cs}
        out["tier_mix"] = breakdown_error([lookup[c] for c in true], [lookup[c] for c in pred])
    return out


def main() -> int:
    t0 = time.time()
    tiers = load_tiers()
    report: dict = {"datasets": {}, "synthetic_pqc": None, "fidelity": {}}
    md = ["# PQC triage", ""]
    for ds_name, field in DATASETS:
        ds = load(ds_name)
        t = triage_flows(ds.flows, label_field=field, tiers=tiers if field == "traffic_type" else None)
        report["datasets"][ds_name] = t
        md.append(to_markdown(ds_name, t))
        print(f"{ds_name}: exposed {t['exposed_share_flows']:.1%} flows / "
              f"{t['exposed_share_bytes']:.1%} bytes; observable {t['observable_share']:.1%}")

    synth = sorted((ROOT.parent / "data" / "_pqc_synth").glob("*.pcap"))
    if synth:
        from countdown.pipeline import Pipeline, PipelineConfig

        pipe = Pipeline(PipelineConfig(mode="fast"))
        recs = []
        for p in synth:
            for r in pipe.analyze_pcap(p):
                r["source"] = p.name
                recs.append(r)
        report["synthetic_pqc"] = {
            "files": [p.name for p in synth],
            "per_file": {p.name: [{k: r[k] for k in ("pqc_verdict", "pqc_basis", "kem_group",
                                                     "quantum_safe", "route_reason")}
                                  for r in recs if r["source"] == p.name] for p in synth},
            "triage": triage_records(recs, tiers=tiers)}
        md.append(to_markdown("_pqc_synth (self-generated, through the pipeline)",
                              report["synthetic_pqc"]["triage"]))

    for name in ("E1", "E2"):
        report["fidelity"][name] = f = fidelity(name, tiers)
        print(f"fidelity {name}: TV {f['class_mix']['total_variation']} over "
              f"{f['exposed_test_flows']} exposed test flows")
    md += ["### Fidelity of the predicted breakdown (exposed test flows)", "",
           "| expert | exposed test flows | accuracy | class-mix TV | worst class (gap) | tier-mix TV |",
           "| --- | --- | --- | --- | --- | --- |"]
    for n, f in report["fidelity"].items():
        cm = f["class_mix"]
        md.append(f"| {n} | {f['exposed_test_flows']:,} | {f['accuracy_on_exposed']:.3f} | "
                  f"{cm['total_variation']:.3f} | {cm['worst_class']} ({cm['worst_gap']:+.3f}) | "
                  f"{f.get('tier_mix', {}).get('total_variation', '—')} |")

    out = ROOT / "runs/triage" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "triage.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "triage.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
