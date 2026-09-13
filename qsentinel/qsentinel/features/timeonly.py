"""Draper-Gil et al. (ICISSP 2016) Table 2 -- time-related features only.

This is the *paper baseline* representation, kept deliberately separate from
``flow_stats``.  The difference is the whole point of the comparison:

* ``flow_stats``  56 columns -- packet-size statistics, ratios, first-N signed sizes.
* ``timeonly``    23 columns -- **no size statistics at all**, only durations, rates and
  inter-arrival times, exactly as Table 2 of the paper lists them.

Reproducing the paper's number with our own richer features would not be a replication,
so the two representations are separate extractors and the run reports both.

One parameter the paper does not state: the **activity threshold** that separates an
"active" span from an "idle" one.  ISCXFlowMeter inherits NetMate's 5 s default and
CICFlowMeter still ships 5 s, so that is the default here -- and it is a constructor
argument so the sensitivity can be measured rather than assumed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qsentinel.features.base import FeatureExtractor, register_extractor
from qsentinel.schema import BACKWARD, FORWARD, Flow

_EPS = 1e-9

#: Order used for every ``(mean, min, max, std)`` group, matching the paper's wording.
_STAT_SUFFIXES = ("mean", "min", "max", "std")


def _stats(values: np.ndarray) -> list[float]:
    """``(mean, min, max, std)``; all-zero when there is nothing to summarise."""
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(values.mean()),
        float(values.min()),
        float(values.max()),
        float(values.std()),
    ]


def active_idle_spans(
    timestamps: np.ndarray, activity_timeout: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Split a flow's timeline into active spans and the idle gaps between them.

    The NetMate / CICFlowMeter convention: walking the packets in order, a gap longer than
    ``activity_timeout`` closes the current active span and opens an idle one.  A span
    holding a single packet has duration 0, which is intentional -- it records that the
    flow was briefly alive, and dropping it would bias ``active_min`` upward.
    """
    if timestamps.size < 2:
        return np.zeros(0), np.zeros(0)

    active: list[float] = []
    idle: list[float] = []
    span_start = last = float(timestamps[0])

    for t in timestamps[1:]:
        t = float(t)
        if t - last > activity_timeout:
            active.append(last - span_start)
            idle.append(t - last)
            span_start = t
        last = t
    active.append(last - span_start)

    return np.asarray(active), np.asarray(idle)


@register_extractor("timeonly")
class DraperGilTimeFeatures(FeatureExtractor):
    """The paper's 23 time-related features, in Table 2 order.

    Layout::

        duration    (1)  total flow duration
        fiat        (4)  forward inter-arrival time   (mean, min, max, std)
        biat        (4)  backward inter-arrival time  (mean, min, max, std)
        flowiat     (4)  bidirectional inter-arrival  (mean, min, max, std)
        active      (4)  active-span duration         (mean, min, max, std)
        idle        (4)  idle-gap duration            (mean, min, max, std)
        fb_psec     (1)  flow bytes per second
        fp_psec     (1)  flow packets per second

    Sizes are used only to form the bytes-per-second *rate*; no size statistic reaches the
    feature vector.  They are also left **uncapped** -- the MTU cap is a ``flow_stats``
    modelling choice, and applying it here would silently change the paper's ``fb_psec``.
    """

    version = 1

    def __init__(self, activity_timeout: float = 5.0, **_ignored) -> None:
        self.activity_timeout = float(activity_timeout)
        self._names: list[str] | None = None

    @property
    def feature_names(self) -> Sequence[str]:
        if self._names is None:
            names = ["duration"]
            for group in ("fiat", "biat", "flowiat", "active", "idle"):
                names += [f"{group}_{s}" for s in _STAT_SUFFIXES]
            names += ["fb_psec", "fp_psec"]
            self._names = names
        return self._names

    def transform_one(self, flow: Flow) -> np.ndarray:
        ts = flow.timestamps.astype(np.float64)
        dirs = flow.directions
        sizes = flow.sizes.astype(np.float64)

        duration = float(ts[-1] - ts[0]) if ts.size else 0.0

        fwd_ts = ts[dirs == FORWARD]
        bwd_ts = ts[dirs == BACKWARD]

        # np.diff on a 0- or 1-element array yields an empty array, which _stats handles.
        fiat = np.diff(fwd_ts)
        biat = np.diff(bwd_ts)
        flowiat = np.diff(ts)

        active, idle = active_idle_spans(ts, self.activity_timeout)

        # Rates are per *second of flow*; a zero-duration window would divide by zero.
        denom = duration if duration > _EPS else _EPS
        fb_psec = float(sizes.sum()) / denom
        fp_psec = float(ts.size) / denom

        out: list[float] = [duration]
        for values in (fiat, biat, flowiat, active, idle):
            out += _stats(values)
        out += [fb_psec, fp_psec]

        return np.asarray(out, dtype=np.float64)
