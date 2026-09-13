"""Reproducible run directories.

A run is only a result if someone else can re-derive it, so ``runs/<id>/`` carries the
frozen config, the metrics, the split audit, the model artifact and the label space --
enough to reload the model and know what its probability columns mean without consulting
anything outside the directory.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from qsentinel.config import REPO_ROOT, get_logger

log = get_logger(__name__)

RUNS_DIR = REPO_ROOT / "runs"


def run_id(prefix: str = "") -> str:
    """Timestamped, sortable, collision-free enough for one machine."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{prefix}" if prefix else stamp


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "sklearn", "lightgbm", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover - optional at report time
            out[mod] = "absent"
    try:
        out["git"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "not-a-repo"
    except Exception:  # pragma: no cover
        out["git"] = "unknown"
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_confusion_png(
    confusion: Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: Path,
    normalize: bool = True,
) -> Path | None:
    """Confusion-matrix plot, or ``None`` when matplotlib is not installed.

    Optional on purpose: the run is defined by ``metrics.json``, which always contains the
    matrix.  A missing plotting library must not fail a training run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("[report] matplotlib absent; skipping confusion.png (pip install '.[viz]')")
        return None

    cm = np.asarray(confusion, dtype=np.float64)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)

    n = len(class_names)
    size = max(6.0, min(24.0, 0.42 * n + 3))
    fig, ax = plt.subplots(figsize=(size, size * 0.85), dpi=130)
    im = ax.imshow(cm, cmap="magma", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.045)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    fontsize = 8 if n <= 20 else max(3.5, 90 / n)
    ax.set_xticklabels(class_names, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("row-normalised confusion" if normalize else "confusion")

    if n <= 15:  # cell annotations only where they stay legible
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] < 0.6 else "black")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_run(
    metrics: dict[str, Any],
    config: dict[str, Any],
    model: Any = None,
    split: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    runs_dir: Path | None = None,
    rid: str | None = None,
) -> Path:
    """Write ``runs/<id>/`` and return the directory.

    Contents: ``metrics.json`` (headline + per-class + confusion + baseline),
    ``config.yaml`` (frozen), ``split.json`` (the audit, including any class missing from
    the test fold), ``model.joblib`` and ``label_space.json``.
    """
    import yaml

    runs_dir = Path(runs_dir or RUNS_DIR)
    rid = rid or run_id(str(config.get("model", "run")))
    out = runs_dir / rid
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": rid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=_json_safe))
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if split is not None:
        (out / "split.json").write_text(json.dumps(split, indent=2, default=_json_safe))

    if model is not None:
        model.save(out / "model.joblib")
        if getattr(model, "label_space", None) is not None:
            (out / "label_space.json").write_text(model.label_space.to_json())

    if metrics.get("confusion") and metrics.get("class_names"):
        write_confusion_png(metrics["confusion"], metrics["class_names"], out / "confusion.png")

    log.info("[report] wrote run -> %s", out)
    return out


def summarize(metrics: dict[str, Any], split: dict[str, Any] | None = None) -> str:
    """One human-readable block, printed at the end of a training run."""
    lines = [
        f"  target        {metrics.get('target')}  ({metrics.get('n_classes')} classes)",
        f"  test samples  {metrics.get('n')}",
        f"  macro-F1      {metrics.get('macro_f1', float('nan')):.4f}"
        f"   (baseline {metrics.get('baseline_macro_f1', float('nan')):.4f})",
        f"  weighted-F1   {metrics.get('weighted_f1', float('nan')):.4f}",
        f"  accuracy      {metrics.get('accuracy', float('nan')):.4f}"
        f"   (baseline {metrics.get('baseline_accuracy', float('nan')):.4f})",
    ]
    if split:
        lines.append(
            f"  groups        {split.get('n_groups_train')} train / "
            f"{split.get('n_groups_test')} test, disjoint={split.get('groups_disjoint')}"
        )
        absent = split.get("classes_absent_from_test") or []
        if absent:
            lines.append(f"  ABSENT FROM TEST  {absent}")
    verdict = "BEATS" if metrics.get("beats_baseline") else "DOES NOT BEAT"
    lines.append(f"  -> {verdict} the majority-class baseline on macro-F1")
    return "\n".join(lines)
