#!/usr/bin/env python
"""PQC-triage statistics over the labelled datasets (Phase 6, task 6.5).

    python scripts/run_triage.py           # -> runs/triage/<id>/{triage.json,triage.md}

The report itself is ``countdown.eval.triage.build_report`` (also reachable as
``countdown experiments run triage``): ground-truth triage per dataset, the self-generated PQC
captures through the real pipeline, and how faithful the deployed tool's predicted breakdown
is on E1's and E2's test folds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from countdown.eval.triage import build_report  # noqa: E402


def main() -> int:
    t0 = time.time()
    report, md = build_report()
    out = ROOT / "runs/triage" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "triage.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "triage.md").write_text(md)
    for n, t in report["datasets"].items():
        print(f"{n}: exposed {t['exposed_share_flows']:.1%} flows / {t['exposed_share_bytes']:.1%} bytes")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
