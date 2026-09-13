"""Cut flows into fixed-duration windows -- the Okonkwo / Sensors sample unit.

Why capture-level and not per-flow
----------------------------------
Okonkwo's pipeline never splits by 5-tuple.  It reads ``frame.len`` and
``frame.time_relative`` for every packet in a pcap and cuts that single timeline into
15/30/60 s windows (section 4.2), so one *image* is one *capture-window*, not one flow.
Table 2 confirms it: 2693 / 5386 / 10772 images at 60/30/15 s is exactly 1x / 2x / 4x,
which only holds if the windows tile a fixed total duration.  Reproducing the paper
therefore means merging a capture's flows back onto one clock before cutting.

``per_flow=True`` cuts each flow's own timeline instead, which is what a deployed
per-flow classifier would see.  Both are offered because the difference is a result, not
a detail: the capture-level unit lets one image mix several concurrent conversations.

Leakage
-------
Every window inherits ``meta["source_file"]`` from its capture, so the existing
``group_keys: iscxvpn: source_file`` grouping keeps all windows of a capture on one side
of a split automatically.  That is the exact leak the paper's random 80/10/10 over pooled
window images has -- a 60 s window contains the same packets as four 15 s windows, and
its near-duplicates land on both sides.  Reproducing *that* is a deliberate opt-in via
``countdown.eval.splits.stratified_split``, never the default.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from countdown.config import get_logger
from countdown.schema import FiveTuple, Flow, FlowDataset

log = get_logger(__name__)

#: Windows with fewer than this many packets are dropped as unusable.  1 keeps everything
#: except the truly empty windows; the paper kept blank windows and showed one (Fig. 3c).
DEFAULT_MIN_PACKETS = 1


def _merge_capture(flows: Sequence[Flow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All packets of a capture on one clock, sorted by absolute timestamp."""
    ts = np.concatenate([f.timestamps for f in flows])
    sz = np.concatenate([f.sizes for f in flows])
    dr = np.concatenate([f.directions for f in flows])
    order = np.argsort(ts, kind="stable")
    return ts[order], sz[order], dr[order]


def expand_to_windows(
    flows: Iterable[Flow],
    window: float = 60.0,
    per_flow: bool = False,
    group_key: str = "source_file",
    min_packets: int = DEFAULT_MIN_PACKETS,
    keep_blank: bool = False,
) -> list[Flow]:
    """Return one synthetic :class:`Flow` per non-empty window.

    Each result keeps its parent's ``label_fields`` and ``meta`` (so grouping and labels
    still resolve) and adds ``meta["window_index"]``, ``meta["window_seconds"]`` and
    ``meta["parent"]``.

    ``keep_blank=True`` emits empty windows too, reproducing the blank-image artifact the
    Sensors paper calls traffic arrival-time disparity.  They carry no signal, so they are
    dropped by default and counted in the log line either way.
    """
    flows = list(flows)
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    buckets: dict[Any, list[Flow]] = defaultdict(list)
    for f in flows:
        key = f.flow_id if per_flow else f.meta.get(group_key)
        if key is None:
            raise ValueError(
                f"flow {f.flow_id!r} has no meta[{group_key!r}] to window by; pass "
                f"per_flow=True to cut each flow's own timeline instead"
            )
        buckets[key].append(f)

    out: list[Flow] = []
    n_blank = 0
    for key, members in buckets.items():
        ts, sz, dr = _merge_capture(members)
        if ts.size == 0:
            continue
        t0 = float(ts[0])
        idx = ((ts - t0) / window).astype(np.int64)
        n_windows = int(idx.max()) + 1
        proto = members[0]

        for w in range(n_windows):
            sel = idx == w
            n = int(sel.sum())
            if n == 0:
                n_blank += 1
                if not keep_blank:
                    continue
            elif n < min_packets:
                continue
            out.append(
                Flow(
                    flow_id=f"{proto.flow_id}|w{window:g}s#{w}",
                    five_tuple=proto.five_tuple,
                    timestamps=ts[sel],
                    sizes=sz[sel],
                    directions=dr[sel],
                    dataset=proto.dataset,
                    label=proto.label,
                    label_fields=dict(proto.label_fields),
                    meta={
                        **proto.meta,
                        "parent": str(key),
                        "window_index": w,
                        "window_seconds": float(window),
                    },
                )
            )

    log.info(
        "[windows] %d flows / %d %s -> %d windows of %gs (%d blank %s)",
        len(flows), len(buckets), "flows" if per_flow else "captures",
        len(out), window, n_blank, "kept" if keep_blank else "dropped",
    )
    return out


def windowed_dataset(
    ds: FlowDataset,
    window: float = 60.0,
    windows: Sequence[float] | None = None,
    **kwargs: Any,
) -> FlowDataset:
    """A :class:`FlowDataset` of windows, preserving name/target/config and the guards.

    ``windows=[15, 30, 60]`` pools several window sizes into one dataset, which is what
    Okonkwo used both as augmentation and to counter arrival-time disparity.  Note what
    that does to independence: the same packets then appear in a 15 s, a 30 s and a 60 s
    image, so only a group-aware split keeps the copies together.
    """
    sizes = list(windows) if windows else [window]
    flows: list[Flow] = []
    for w in sizes:
        flows.extend(expand_to_windows(ds.flows, window=float(w), **kwargs))
    return FlowDataset(
        flows,
        name=f"{ds.name}_w{'+'.join(f'{w:g}' for w in sizes)}",
        default_target=ds.default_target,
        config=ds.config,
        invalid_targets=ds.invalid_targets,
    )
