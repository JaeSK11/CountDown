#!/usr/bin/env python
"""Train/serve skew check: does ``countdown analyze`` reproduce the Phase-5 evaluation?

    python scripts/check_pipeline_skew.py          # 60 random E2 test-fold pcaps

Runs the real pipeline (pcap -> reassembly -> features / byte graphs -> deployed E2) on
CSTNET test-fold captures and compares each member's top label and confidence with the
predictions the same model made during the Phase-5 run.  Agreement must be exact: any
difference means the pipeline builds a member's input differently from training.
Note: CSTNET flow_ids repeat across pcaps (601 distinct ids for 46,372 flows), so rows are
mapped to pcaps by position, never by flow_id.
"""
import sys, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countdown.experts import get_expert
from countdown.ensemble.views import build_expert_data, PER_FLOW_OVERRIDES
from countdown.pipeline import Pipeline, PipelineConfig

d = build_expert_data(get_expert("E2"), members=["flow_gbdt", "graph_gnn"], seed=42,
                      overrides=PER_FLOW_OVERRIDES["E2"])
R = ROOT / "runs/ensemble/E2/preds-s42"
cached = {m: np.load(R / f"{m}.npz")["test"] for m in ("flow_gbdt", "graph_gnn")}
names = d.space.names
rng = np.random.default_rng(0)
pick = rng.choice(len(d.test_rows), 60, replace=False)
pipe = Pipeline(PipelineConfig(include_members=True, fallback=False))
agree = {"flow_gbdt": 0, "graph_gnn": 0}; dconf = {"flow_gbdt": [], "graph_gnn": []}
right = 0; n = 0; nflows = []
from countdown.data import load
from countdown.ensemble.views import _graph_rows
flows = load("cstnet").flows
_, keep = _graph_rows(flows, ROOT / "cache/tfegnn/cstnet")
row_src = [flows[i].meta["source_file"] for i in keep]     # same row order as d
for k in pick:
    row = d.test_rows[k]
    path = row_src[row]
    recs = pipe.analyze_pcap(path)
    nflows.append(len(recs))
    if len(recs) != 1:
        continue
    r = recs[0]; n += 1
    right += r["traffic_class"] == names[d.y[row]]
    for m in agree:
        c = cached[m][k]; pm = r["members"].get(m)
        if pm is None: continue
        agree[m] += pm["label"] == names[int(c.argmax())]
        dconf[m].append(abs(pm["confidence"] - float(c.max())))
out = ({"pcaps": len(pick), "single_flow": n, "flows_per_pcap": sorted(set(nflows)),
                  "pipeline_E2_acc_no_fallback": round(right / max(n,1), 3),
                  "member_top_label_agrees_with_cached": {m: f"{agree[m]}/{n}" for m in agree},
                  "member_conf_abs_diff_median": {m: round(float(np.median(v)), 4) if v else None for m, v in dconf.items()}})
import time
dest = ROOT / "runs/pipeline-skew" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2)); print("wrote", dest)
