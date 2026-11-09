#!/usr/bin/env python
"""Run the PQC census on a schedule and keep the trend current (Phase 9).

    docker compose -f docker/compose.yaml up -d census      # the daemon (restart: unless-stopped)
    python scripts/census_scheduler.py --status            # when is the next run due?

Every ``--every-days`` (default 7), at ``--hour`` UTC (default 03), runs all three census
profiles on the fixed Tranco list (``../data/_census/top-1m.csv``, id in ``LIST_ID``) into
``results/census/runs/<date>/``, then rebuilds ``results/census/trend.*``.

* **Due** = the latest run's date + ``every_days``, at ``hour``:00 UTC.  A run missed while the
  machine was off happens at the next check after it falls due (checked every 15 min).
* **A run only counts once it finished**: it writes to ``<date>.partial`` and is renamed on
  success, so an interrupted run is redone, not mistaken for a data point.
* **A failed run** is retried after ``--retry-hours`` (default 6), not in a tight loop.
* **Nothing is committed**: the results land in the working tree for a person to review.

Log: ``runs/census/scheduler.log`` (one line per event) and ``runs/census/<date>.log``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "results/census/runs"
LOGS = ROOT / "runs/census"
SITES = ROOT.parent / "data/_census/top-1m.csv"
LIST_ID = ROOT.parent / "data/_census/LIST_ID"


def log(msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}"
    with open(LOGS / "scheduler.log", "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def last_run_date() -> date | None:
    """Latest finished census run: scheduled runs, plus the Phase 7/8 runs it continues."""
    from scripts.census_trend import find_runs, summarise_run

    dates = [date.fromisoformat(summarise_run(r)["date"]) for r in find_runs()]
    return max(dates) if dates else None


def next_due(last: date | None, every_days: int, hour: int) -> datetime:
    if last is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    d = last + timedelta(days=every_days)
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)


def run_census(args) -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    final, partial = RUNS / today, RUNS / f"{today}.partial"
    if final.exists():
        log(f"SKIP {today}: already have {final.relative_to(ROOT)}")
        return True
    shutil.rmtree(partial, ignore_errors=True)
    list_id = LIST_ID.read_text().strip() if LIST_ID.exists() else "unknown"
    cmd = [args.python, "scripts/pqc_census.py", "--sites", str(SITES), "--tranco-list", list_id,
           "--top", str(args.top), "--profiles", "tls13,tls12,quic", "--out", str(partial)]
    log(f"START {today} ({' '.join(cmd[1:])})")
    t0 = time.time()
    with open(LOGS / f"{today}.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not (partial / "census.json").exists():
        log(f"FAIL {today} rc={rc} after {time.time() - t0:.0f}s (see runs/census/{today}.log)")
        return False
    partial.rename(final)
    log(f"DONE {today} in {time.time() - t0:.0f}s")
    rc = subprocess.run([sys.executable, "scripts/census_trend.py"], cwd=ROOT,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    log(f"TREND rc={rc}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every-days", type=int, default=int(os.environ.get("CENSUS_EVERY_DAYS", 7)))
    ap.add_argument("--hour", type=int, default=int(os.environ.get("CENSUS_HOUR_UTC", 3)))
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--retry-hours", type=float, default=6.0)
    ap.add_argument("--check-minutes", type=float, default=15.0)
    ap.add_argument("--python", default=os.environ.get("CENSUS_PY", "python-netraw"),
                    help="interpreter with CAP_NET_RAW (the image's python-netraw)")
    ap.add_argument("--status", action="store_true", help="print the next due time and exit")
    args = ap.parse_args()

    last = last_run_date()
    if args.status:
        print(f"last run {last}, next due {next_due(last, args.every_days, args.hour):%Y-%m-%d %H:%M}Z")
        return 0

    log(f"SCHEDULER START every {args.every_days} d at {args.hour:02d}:00Z; last run {last}, "
        f"next due {next_due(last, args.every_days, args.hour):%Y-%m-%d %H:%M}Z")
    retry_after = None
    while True:
        now = datetime.now(timezone.utc)
        due = next_due(last_run_date(), args.every_days, args.hour)
        if now >= due and (retry_after is None or now >= retry_after):
            if run_census(args):
                retry_after = None
                log(f"next due {next_due(last_run_date(), args.every_days, args.hour):%Y-%m-%d %H:%M}Z")
            else:
                retry_after = now + timedelta(hours=args.retry_hours)
                log(f"retry after {retry_after:%Y-%m-%d %H:%M}Z")
        time.sleep(args.check_minutes * 60)


if __name__ == "__main__":
    raise SystemExit(main())
