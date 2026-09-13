"""Packet-sequence tensors (signed sizes + IATs) -> CNN / SSM models in Phase 4."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qsentinel.features.base import FeatureExtractor, register_extractor
from qsentinel.schema import Flow


@register_extractor("packet_seq")
class PacketSeq(FeatureExtractor):
    """First ``n`` packets as a fixed-length tensor.

    ``channels=1`` -> ``(N, n)`` signed MTU-capped sizes.
    ``channels=2`` -> ``(N, n, 2)`` signed sizes and inter-arrival times.

    Shorter flows are zero-padded at the tail; longer flows are truncated.  ``masks``
    returns the matching ``(N, n)`` boolean array of real (non-padded) positions.
    """

    version = 1

    def __init__(self, n: int = 32, mtu_cap: int = 1500, channels: int = 2, **_ignored) -> None:
        self.n = int(n)
        self.mtu_cap = int(mtu_cap)
        self.channels = int(channels)
        if self.channels not in (1, 2):
            raise ValueError(f"channels must be 1 or 2, got {self.channels}")

    @property
    def feature_names(self) -> Sequence[str]:
        return ["signed_size", "iat"][: self.channels]

    def transform_one(self, flow: Flow) -> np.ndarray:
        n = self.n
        k = min(flow.n_packets, n)
        sizes = np.minimum(flow.sizes[:k].astype(np.float64), self.mtu_cap)
        signed = sizes * np.sign(flow.directions[:k])

        size_ch = np.zeros(n, dtype=np.float64)
        size_ch[:k] = signed
        if self.channels == 1:
            return size_ch

        iat_ch = np.zeros(n, dtype=np.float64)
        if k > 1:
            rel = flow.timestamps[:k] - flow.timestamps[0]
            iat_ch[1:k] = np.diff(rel)
        return np.stack([size_ch, iat_ch], axis=-1)

    def transform(self, flows: Sequence[Flow]) -> np.ndarray:
        if len(flows) == 0:
            shape = (0, self.n) if self.channels == 1 else (0, self.n, 2)
            return np.zeros(shape, dtype=np.float32)
        out = np.stack([self.transform_one(f) for f in flows]).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def masks(self, flows: Sequence[Flow]) -> np.ndarray:
        """``(N, n)`` boolean: True where a real packet occupies the slot."""
        m = np.zeros((len(flows), self.n), dtype=bool)
        for i, f in enumerate(flows):
            m[i, : min(f.n_packets, self.n)] = True
        return m

    def transform_with_mask(self, flows: Sequence[Flow]) -> tuple[np.ndarray, np.ndarray]:
        return self.transform(flows), self.masks(flows)
