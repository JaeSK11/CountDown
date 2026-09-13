"""Shared evaluation: leakage-safe splits, metrics, and reproducible run directories."""

from countdown.eval.metrics import (
    baseline_metrics,
    compute_metrics,
    evaluate,
    metrics_excluding_tail,
)
from countdown.eval.report import run_id, summarize, write_confusion_png, write_run
from countdown.eval.splits import (
    assert_groupable,
    group_split,
    group_split_three,
    split_report,
    stratified_split,
)

__all__ = [
    "assert_groupable",
    "baseline_metrics",
    "compute_metrics",
    "evaluate",
    "group_split",
    "group_split_three",
    "metrics_excluding_tail",
    "run_id",
    "split_report",
    "stratified_split",
    "summarize",
    "write_confusion_png",
    "write_run",
]
