#!/usr/bin/env python
"""Copy the scorecard evidence out of the gitignored ``runs/`` into the tracked ``results/``.

    python scripts/collect_results.py            # refresh results/ and results/INDEX.md

Decision D4 (2026-09-19): ``runs/`` stays gitignored -- it holds checkpoints and models of
hundreds of MB -- so the Phase-4 exit criterion's "scorecards committed under runs/" is met
by this folder instead.  Only the small, reviewable files are copied: ``metrics.json``,
``config.yaml`` and ``split.json`` of harness runs, and the result JSON of the replication
drivers.  Never a model, never a checkpoint.  ``MANIFEST`` is the curated list; a run that is
not named here is an experiment, not evidence.
"""

from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "results"
KEEP = ("metrics.json", "config.yaml", "split.json")
MAX_BYTES = 2_000_000

#: group -> list of globs under runs/.  A glob may match run directories or single JSON files.
MANIFEST: dict[str, list[str]] = {
    "flow-stats/phase3": ["2026*-flow_gbdt"],
    "flow-stats/replication-drapergil": ["replicate_drapergil.json", "ablate_features_vs_model.json",
                                         "ablate_gbdt_unweighted.json"],
    "flow-stats/classweight-retest": ["classweight-retest/*/*"],
    "flow-stats/gbdt-bakeoff": ["gbdt-bakeoff/*/*"],
    "image/replication-okonkwo": ["okonkwo/replication.json", "okonkwo/variant_*.json"],
    "image/seeds": ["okonkwo/seeds/*.json"],
    "image/mobileapp": ["mobileapp-image/*", "mobileapp-image-d5/*/*"],
    "sequence": ["seq/*/*"],
    "bytes/replication-etbert": ["etbert-repro/flow.json", "etbert-repro/packet.json",
                                 "etbert-repro-nopretrain/packet_scratch.json"],
    "bytes/scorecard": ["byte-net/scorecard.json"],
    "graph/replication-tfegnn": ["tfegnn*/tor.json"],
    "graph/tor": ["graph-gnn/*", "graph-gnn-noval/*", "graph-gin-tor/*"],
    "graph/vpn": ["graph-gnn-vpn/*"],
    "graph/nontor": ["graph-gnn-nontor/*"],
    "graph/nonvpn": ["graph-gnn-nonvpn/*"],
    "graph/cstnet": ["graph-cstnet/*/*"],
}


def _headline(path: Path) -> dict:
    """Best-effort one-line summary of a result file, for the index."""
    try:
        d = json.loads(path.read_text())
    except Exception:
        return {}
    m = d.get("metrics", d)
    out = {k: m[k] for k in ("macro_f1", "accuracy", "n") if isinstance(m.get(k), (int, float))}
    if "ours" in d and isinstance(d["ours"], dict):
        out = {k: d["ours"].get(k) for k in ("macro_f1", "accuracy", "n")}
    if "summary" in d and isinstance(d["summary"], dict) and "F1" in d["summary"]:
        out = {"macro_f1": d["summary"]["F1"].get("mean"), "accuracy": d["summary"]["AC"].get("mean")}
    return {k: v for k, v in out.items() if v is not None}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    rows, skipped = [], []
    for group, patterns in MANIFEST.items():
        for pat in patterns:
            for src in sorted(glob.glob(str(RUNS / pat))):
                src = Path(src)
                rel = src.relative_to(RUNS)
                if src.is_dir():
                    if not (src / "metrics.json").exists():
                        continue
                    files = [src / k for k in KEEP if (src / k).exists()]
                    dest_dir, head = OUT / group / rel, src / "metrics.json"
                elif src.suffix == ".json":
                    files, dest_dir, head = [src], OUT / group / rel.parent, src
                else:
                    continue
                for f in files:
                    if f.stat().st_size > MAX_BYTES:
                        skipped.append(str(f.relative_to(RUNS)))
                        continue
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest_dir / f.name)
                rows.append((group, str(rel), _headline(head)))

    lines = ["# Results index", "",
             "Copied from the gitignored `runs/` by `scripts/collect_results.py` (decision D4). "
             "Metrics, configs and split audits only -- no models, no checkpoints. The tables that "
             "interpret these live in `phases/phase4-models/`.", "",
             "| group | run | macro-F1 | accuracy | n |", "| --- | --- | --- | --- | --- |"]
    for group, rel, h in rows:
        f = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
        lines.append(f"| {group} | `{rel}` | {f(h.get('macro_f1'))} | {f(h.get('accuracy'))} | {f(h.get('n'))} |")
    if skipped:
        lines += ["", "Skipped for size: " + ", ".join(f"`{s}`" for s in skipped)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEX.md").write_text("\n".join(lines) + "\n")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"results/: {len(rows)} runs, {total / 1e6:.1f} MB, {len(skipped)} file(s) skipped for size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
