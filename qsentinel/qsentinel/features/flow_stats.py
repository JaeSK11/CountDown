"""Tabular per-flow statistics (CICFlowMeter-inspired subset) -> LightGBM in Phase 3."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qsentinel.features.base import FeatureExtractor, register_extractor
from qsentinel.schema import Flow

_EPS = 1e-9


def _stats(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    names = [f"{prefix}_min", f"{prefix}_max", f"{prefix}_mean", f"{prefix}_std"]
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0], names
    return [
        float(values.min()),
        float(values.max()),
        float(values.mean()),
        float(values.std()),
    ], names


@register_extractor("flow_stats")
class FlowStats(FeatureExtractor):
    """Fixed-length numeric vector per flow.

    Layout (``first_n_packets=20`` by default -> 56 columns):

    * counts        (6)  packet / byte totals, overall and per direction
    * ratios        (3)  forward-backward packet, byte and down/up ratios
    * packet sizes (12)  min/max/mean/std overall, forward, backward
    * IATs         (12)  min/max/mean/std overall, forward, backward
    * flow rates    (3)  duration, packets/sec, bytes/sec
    * first-N sizes (N)  signed, MTU-capped sizes of the first N packets

    Sizes are capped at ``mtu_cap`` so that segmentation-offload super-frames (present in
    the ISCXTor captures, which reach ~10 kB) do not dominate the scale.
    """

    version = 1

    def __init__(self, first_n_packets: int = 20, mtu_cap: int = 1500, **_ignored) -> None:
        self.first_n_packets = int(first_n_packets)
        self.mtu_cap = int(mtu_cap)
        self._names: list[str] | None = None

    @property
    def feature_names(self) -> Sequence[str]:
        if self._names is None:
            self._names = self._build_names()
        return self._names

    def _build_names(self) -> list[str]:
        names = [
            "n_packets", "n_fwd_packets", "n_bwd_packets",
            "total_bytes", "fwd_bytes", "bwd_bytes",
            "fwd_bwd_packet_ratio", "fwd_bwd_byte_ratio", "down_up_ratio",
        ]
        for prefix in ("size", "fwd_size", "bwd_size"):
            names += [f"{prefix}_min", f"{prefix}_max", f"{prefix}_mean", f"{prefix}_std"]
        for prefix in ("iat", "fwd_iat", "bwd_iat"):
            names += [f"{prefix}_min", f"{prefix}_max", f"{prefix}_mean", f"{prefix}_std"]
        names += ["duration", "packets_per_sec", "bytes_per_sec"]
        names += [f"pkt{i}_signed_size" for i in range(self.first_n_packets)]
        return names

    def transform_one(self, flow: Flow) -> np.ndarray:
        sizes = np.minimum(flow.sizes.astype(np.float64), self.mtu_cap)
        dirs = flow.directions
        ts = flow.timestamps

        fwd = sizes[dirs > 0]
        bwd = sizes[dirs < 0]
        n, n_f, n_b = sizes.size, fwd.size, bwd.size
        total, t_f, t_b = sizes.sum(), fwd.sum(), bwd.sum()

        row: list[float] = [
            float(n), float(n_f), float(n_b),
            float(total), float(t_f), float(t_b),
            float(n_f / (n_b + _EPS)),
            float(t_f / (t_b + _EPS)),
            float(t_b / (t_f + _EPS)),  # down/up: server->client over client->server
        ]

        for values in (sizes, fwd, bwd):
            row += _stats(values, "x")[0]

        iat = np.diff(ts) if n > 1 else np.array([])
        iat_f = np.diff(ts[dirs > 0]) if n_f > 1 else np.array([])
        iat_b = np.diff(ts[dirs < 0]) if n_b > 1 else np.array([])
        for values in (iat, iat_f, iat_b):
            row += _stats(values, "x")[0]

        duration = flow.duration
        row += [
            float(duration),
            float(n / (duration + _EPS)),
            float(total / (duration + _EPS)),
        ]

        signed = (sizes * np.sign(dirs))[: self.first_n_packets]
        padded = np.zeros(self.first_n_packets, dtype=np.float64)
        padded[: signed.size] = signed
        row += padded.tolist()

        return np.asarray(row, dtype=np.float64)
