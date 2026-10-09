#!/usr/bin/env python
"""Routing options for ISCX `tls` flows: how separable are they by E2's confidence, and what
does E1 score on them?  E2 is represented by its GBDT member (GIN needs per-flow byte graphs
that ISCX flows do not have), calibrated with the E2 run's temperature.

    python scripts/eval_routing.py      # -> runs/routing-eval/<id>.json
"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
from countdown.data import load
from countdown.models.base import BaseModel
from countdown.eval.metrics import compute_metrics
from countdown.experts import get_expert
from countdown.ensemble.views import build_expert_data

R = str(Path(__file__).resolve().parents[1] / "runs/ensemble")
e2 = joblib.load(f"{R}/E2/20261008-195912/ensemble.joblib")
gb2 = BaseModel.from_blob(e2["models"]["flow_gbdt"]); cal2 = e2["calibrators"]["flow_gbdt"]

# In-distribution: E2's own test rows (same table/split as the E2 run).
d2 = build_expert_data(get_expert("E2"), members=["flow_gbdt"], seed=42)
X2 = d2.views["flow_gbdt"].X
p_in = cal2.transform(gb2.predict_proba(X2[d2.test_rows]))
acc_in = float((p_in.argmax(1) == d2.y[d2.test_rows]).mean())

# Out-of-distribution: ISCX flows hinted tls.
iscx = load("iscx_pooled", target="traffic_type")
tls = np.array([f.l7_hint == "tls" for f in iscx.flows])
Xi = iscx.features("flow_stats")
p_out = cal2.transform(gb2.predict_proba(Xi[tls]))
c_in, c_out = p_in.max(1), p_out.max(1)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(np.r_[np.ones(len(c_in)), np.zeros(len(c_out))], np.r_[c_in, c_out])
rows = []
for keep in (0.99, 0.95, 0.90, 0.80):
    t = float(np.quantile(c_in, 1 - keep))
    kept_in = c_in >= t
    rows.append({"keep_cstnet": keep, "threshold": round(t, 4),
                 "cstnet_acc_on_kept": round(float((p_in.argmax(1) == d2.y[d2.test_rows])[kept_in].mean()), 4),
                 "iscx_tls_caught_by_fallback": round(float((c_out < t).mean()), 4)})
top = {}
names = gb2.label_space.names
for i in p_out.argmax(1): top[names[i]] = top.get(names[i], 0) + 1

# E1 (GBDT) on its own test rows, tls vs the rest -- what E1 does with these flows.
d1 = build_expert_data(get_expert("E1"), members=["flow_gbdt"], seed=42)
gb1 = BaseModel.load(f"{R}/E1/preds-s42/flow_gbdt.joblib")
te = d1.test_rows
p1 = gb1.predict_proba(d1.views["flow_gbdt"].X[te]); y1 = d1.y[te]
is_tls = np.array([h == "tls" for h in d1.l7_hint[te]])
def sc(m): r = compute_metrics(y1[m], p1[m].argmax(1), len(d1.space)); return {"n": int(m.sum()), "macro_f1": round(r["macro_f1"], 4), "accuracy": round(r["accuracy"], 4)}
cls = {}
for n, t in zip(d1.space.names, np.bincount(y1[is_tls], minlength=len(d1.space))): cls[n] = int(t)

out = {"e2_gbdt_cstnet_test_acc": round(acc_in, 4), "n_cstnet_test": int(len(c_in)), "n_iscx_tls": int(tls.sum()),
       "conf_median_cstnet": round(float(np.median(c_in)), 4), "conf_median_iscx_tls": round(float(np.median(c_out)), 4),
       "auroc_confidence_separates": round(float(auc), 4), "fallback_thresholds": rows,
       "iscx_tls_top_e2_labels": dict(sorted(top.items(), key=lambda kv: -kv[1])[:8]),
       "e1_gbdt_on_e1_test": {"tls": sc(is_tls), "non_tls": sc(~is_tls), "all": sc(np.ones_like(is_tls))},
       "e1_test_tls_class_counts": cls}
import time
dest = Path(R).parent / "routing-eval" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print("wrote", dest)
