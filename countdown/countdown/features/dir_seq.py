"""Direction-only packet sequence -- the Deep Fingerprinting (Sirinam et al., CCS'18) input.

DF classifies a trace from nothing but the order of packet directions: ``+1`` outgoing,
``-1`` incoming, zero-padded to a fixed 5,000.  Sizes and timing are deliberately absent --
on Tor, where DF was designed, cells are a constant 512 bytes, so size carries nothing.

This is a separate extractor from ``packet_seq`` for the reason ``timeonly`` is separate
from ``flow_stats``: running the paper's *model* on our richer *features* would not be a
replication (``MODEL-DECISIONS.md``).  Expect it to be a weak representation on ISCX: the
median pooled flow is 9 packets long, so a 5,000-wide input is >99 % padding there.  That
gap between what the paper's input assumes and what flow-level ETC data looks like is part
of what the baseline is for.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from countdown.features.base import FeatureExtractor, register_extractor
from countdown.schema import Flow


@register_extractor("dir_seq")
class DirectionSeq(FeatureExtractor):
    """``(N, n)`` float32 of ``+1 / -1`` per packet, zero-padded / truncated to ``n``."""

    version = 1

    def __init__(self, n: int = 5000, **_ignored) -> None:
        self.n = int(n)

    @property
    def feature_names(self) -> Sequence[str]:
        return [f"dir_{i}" for i in range(self.n)]

    def transform_one(self, flow: Flow) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        k = min(flow.n_packets, self.n)
        out[:k] = np.sign(flow.directions[:k]).astype(np.float32)
        return out

    def transform(self, flows: Sequence[Flow]) -> np.ndarray:
        out = np.zeros((len(flows), self.n), dtype=np.float32)
        for i, f in enumerate(flows):
            k = min(f.n_packets, self.n)
            out[i, :k] = np.sign(f.directions[:k])
        return out
