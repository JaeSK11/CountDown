#!/usr/bin/env python
"""Render ``runs/replicate_drapergil.json`` as the replication comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SHORT = {"flow_c45_paper": "C4.5", "flow_knn_paper": "k-NN"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("runs/replicate_drapergil.json"))
    args = ap.parse_args()

    data = json.loads(args.path.read_text())
    runs = data["runs"]

    # index (scenario, model, ftm) -> {protocol: avg_precision}
    table: dict[tuple, dict] = {}
    for r in runs:
        key = (r["scenario"], r["model"], r["ftm"])
        table.setdefault(key, {"published": r.get("published_avg_precision"),
                               "n_classes": r["n_classes"]})
        table[key][r["protocol"]] = r.get("avg_precision")

    print(f"| Scenario | Model | ftm | classes | published | ours (paper protocol) | "
          f"ours (grouped) | leak |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for (scenario, model, ftm), v in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        pub = v.get("published")
        paper = v.get("paper")
        grouped = v.get("grouped")
        leak = (paper - grouped) if (paper is not None and grouped is not None) else None
        print(
            f"| {scenario} | {SHORT.get(model, model)} | {ftm:g}s | {v['n_classes']} | "
            f"{'—' if pub is None else f'{pub:.3f}'} | "
            f"{'—' if paper is None else f'{paper:.4f}'} | "
            f"{'—' if grouped is None else f'{grouped:.4f}'} | "
            f"{'—' if leak is None else f'+{leak:.4f}'} |"
        )


if __name__ == "__main__":
    main()
