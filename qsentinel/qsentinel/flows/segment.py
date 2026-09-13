"""Re-segment flows into fixed-duration windows -- Draper-Gil's "flow timeout".

The two projects mean different things by "flow timeout" and conflating them would
quietly invalidate the replication:

* **This pipeline** (``configs/features.yaml: flow.flow_timeout = 64``) uses an *idle*
  timeout -- a flow ends after 64 s with no packets.  Flows therefore run as long as the
  conversation does; on ISCXVPN the median is 24 s and the longest is 98 minutes.
* **The paper** (Section 4.1) fixes flow *duration*: "we set the duration of flows to
  15, 30, 60 and 120 seconds", then recomputes features per window.  A 10-minute
  conversation becomes 40 samples at ftm=15, not one.

So the paper's ftm sweep is a *segmentation* of the packet timeline, not a different idle
threshold, and it is what makes their "15 s delay" latency claim meaningful.

Segmenting cached flows rather than re-reading 28 GB of pcaps at four thresholds is an
equivalence worth stating: our 64 s idle timeout only ever *joins* packets that the paper
would also have joined (any gap it splits on is > 64 s, hence larger than every ftm we
sweep), so cutting our flows at ftm reproduces the same partition of the timeline.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from qsentinel.schema import Flow


def segment_flow(flow: Flow, window: float, min_packets: int = 2) -> list[Flow]:
    """Cut ``flow`` into consecutive ``window``-second slices.

    Windows are measured from the flow's own first packet, so a slice boundary is a
    property of the flow rather than of wall-clock time.

    ``min_packets`` defaults to 2 because every feature in the paper's Table 2 that is not
    a rate is an inter-arrival statistic, and a 1-packet window has no inter-arrival at
    all -- it would contribute a row of structural zeros.  Dropped windows are counted by
    :func:`segment_flows` rather than silently discarded.
    """
    ts = flow.timestamps
    if ts.size == 0:
        return []

    # Slice index of each packet; floor() so window k covers [k*w, (k+1)*w).
    idx = np.floor((ts - ts[0]) / float(window)).astype(np.int64)

    out: list[Flow] = []
    for w in np.unique(idx):
        mask = idx == w
        if int(mask.sum()) < min_packets:
            continue
        out.append(
            Flow(
                flow_id=f"{flow.flow_id}@w{int(w)}",
                five_tuple=flow.five_tuple,
                timestamps=ts[mask],
                sizes=flow.sizes[mask],
                directions=flow.directions[mask],
                dataset=flow.dataset,
                label=flow.label,
                # Copied, not shared: a downstream filter that mutates one window's labels
                # must not reach into its siblings.
                label_fields=dict(flow.label_fields),
                meta=dict(flow.meta),
            )
        )
    return out


def segment_flows(
    flows: Sequence[Flow], window: float, min_packets: int = 2
) -> tuple[list[Flow], dict[str, int]]:
    """:func:`segment_flow` over a collection, plus a drop report.

    Returns ``(windows, stats)`` where ``stats`` records how much of the timeline the
    ``min_packets`` floor removed -- a replication that quietly drops most of its data is
    not a replication, so the number is reported alongside the accuracy.
    """
    out: list[Flow] = []
    kept = dropped = 0

    for f in flows:
        ts = f.timestamps
        if ts.size == 0:
            continue
        n_windows = int(np.unique(np.floor((ts - ts[0]) / float(window))).size)
        segments = segment_flow(f, window, min_packets=min_packets)
        kept += len(segments)
        dropped += n_windows - len(segments)
        out.extend(segments)

    return out, {
        "source_flows": len(flows),
        "windows_kept": kept,
        "windows_dropped_min_packets": dropped,
    }
