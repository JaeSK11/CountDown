"""Flow-as-image tensors: the Okonkwo scatter and the FlowPic histogram.

Two constructions of the same (packet size x arrival time) plane, selected by
``construction``:

``scatter`` (Okonkwo et al., AISC 2022)
    A binary scatter: one mark per packet at ``(time_relative, frame.len)``.  Sizes are
    capped at the MTU and packets above it are *discarded*, per the paper (section 4.2).
    This is the baseline representation and the default.

``flowpic`` (Shapira & Shavitt)
    A 2D histogram of the same plane -- packet *density* per bin rather than presence.
    Denser and far less aliasing-prone at stride-2 downsampling; the recommended default
    for new work (see ``phases/phase4-models/MODEL-image.md``).

Marker size
-----------
The one place a faithful reproduction cannot be exact.  Okonkwo rendered scatter plots
with matplotlib, saved them as JPEG and resized to 224x224, so each packet became an
anti-aliased blob several pixels across, not one pixel.  Rasterising a single pixel per
packet would hand the baseline a materially sparser image than the paper's model saw and
understate it.  ``marker`` (default 3, i.e. a 3x3 square) approximates matplotlib's
default marker at this resolution.  Set ``marker=1`` for a true one-pixel-per-packet
scatter.  It is a parameter rather than a constant because it is an assumption, and an
assumption that moves the number should be visible in the run config.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qsentinel.features.base import FeatureExtractor, register_extractor
from qsentinel.schema import Flow

CONSTRUCTIONS = ("scatter", "flowpic")


@register_extractor("flow_image")
class FlowImage(FeatureExtractor):
    """Per-flow ``(C, S, S)`` float32 image of the size-vs-time plane.

    Parameters
    ----------
    size
        Output edge length in pixels.  224 reproduces the paper; 64 is what the FlowPic
        construction actually needs.
    construction
        ``"scatter"`` (Okonkwo, binary presence) or ``"flowpic"`` (histogram density).
    window
        Seconds of flow time the x-axis spans.

        ``"auto"`` (the default) reads ``meta["window_seconds"]`` from each sample, so a
        window carries its own axis scale.  This is what the paper does -- section 4.2
        caps the x-axis "based on the time window size used (15, 30, 60secs)" -- and it
        matters whenever several window sizes are pooled: a fixed 60 s axis would squeeze
        every 15 s window into the leftmost quarter of its image, a different and much
        sparser representation than the one the paper trained on.
        A number fixes the axis for every sample.  ``None`` uses each flow's own duration,
        which normalises timing scale away entirely.
    channels
        ``1`` -> presence/density only (the paper).
        ``3`` -> ``(count, direction-signed, byte-volume)``, the recommended construction.
        Direction is the signal Okonkwo's two-field extraction has no way to encode.
    mtu_cap
        Sizes are capped here; with ``drop_oversize`` packets above it are dropped
        instead, which is what the paper did.
    """

    version = 1

    def __init__(
        self,
        size: int = 224,
        construction: str = "scatter",
        window: float | str | None = "auto",
        channels: int = 1,
        mtu_cap: int = 1500,
        marker: int = 3,
        drop_oversize: bool = True,
        log_density: bool = True,
        **_ignored,
    ) -> None:
        if construction not in CONSTRUCTIONS:
            raise ValueError(f"construction must be one of {CONSTRUCTIONS}, got {construction!r}")
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {channels}")
        if marker < 1 or marker % 2 == 0:
            raise ValueError(f"marker must be a positive odd integer, got {marker}")
        self.size = int(size)
        self.construction = construction
        if isinstance(window, str):
            if window != "auto":
                raise ValueError(f"window must be a number, None, or 'auto', got {window!r}")
            self.window = "auto"
        else:
            self.window = None if window is None else float(window)
        self.channels = int(channels)
        self.mtu_cap = int(mtu_cap)
        self.marker = int(marker)
        self.drop_oversize = bool(drop_oversize)
        self.log_density = bool(log_density)

    @property
    def feature_names(self) -> Sequence[str]:
        return ["count", "direction", "bytes"][: self.channels]

    # -- binning -----------------------------------------------------------------------
    def _bins(self, flow: Flow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(col, row, signed_size)`` integer pixel coordinates for each kept packet."""
        sizes = flow.sizes.astype(np.float64)
        times = flow.relative_timestamps()
        dirs = flow.directions.astype(np.float64)

        if self.drop_oversize:
            keep = sizes <= self.mtu_cap
        else:
            keep = np.ones(sizes.shape, dtype=bool)
            sizes = np.minimum(sizes, self.mtu_cap)
        sizes, times, dirs = sizes[keep], times[keep], dirs[keep]

        empty = np.zeros(0, dtype=np.int64)
        if sizes.size == 0:
            return empty, empty, np.zeros(0, dtype=np.float64)

        if self.window == "auto":
            # A windowed sample knows its own axis; an unwindowed flow falls back to its
            # own duration rather than silently borrowing another sample's scale.
            span = float(flow.meta.get("window_seconds") or times[-1])
        elif self.window is not None:
            span = self.window
        else:
            span = float(times[-1])
        # A zero span (single packet, or all packets at one timestamp) would divide by
        # zero; put those in column 0 rather than dropping a real flow.
        col = (
            np.zeros(times.shape, dtype=np.int64)
            if span <= 0
            else np.clip((times / span) * (self.size - 1), 0, self.size - 1).astype(np.int64)
        )
        # Row 0 is the bottom of the plot (small packets), matching the paper's y-axis.
        row = np.clip((sizes / self.mtu_cap) * (self.size - 1), 0, self.size - 1).astype(np.int64)
        return col, row, sizes * np.sign(dirs)

    def _stamp(self, plane: np.ndarray, row: np.ndarray, col: np.ndarray, value) -> None:
        """Accumulate ``value`` at each (row, col), spread over a ``marker``-square."""
        half = self.marker // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                r = np.clip(row + dr, 0, self.size - 1)
                c = np.clip(col + dc, 0, self.size - 1)
                np.add.at(plane, (r, c), value)

    def transform_one(self, flow: Flow) -> np.ndarray:
        s = self.size
        out = np.zeros((self.channels, s, s), dtype=np.float32)
        col, row, signed = self._bins(flow)
        if col.size == 0:
            return out

        counts = np.zeros((s, s), dtype=np.float64)
        self._stamp(counts, row, col, 1.0)

        if self.construction == "scatter":
            # Binary presence: the paper's plot marks a pixel or it does not.
            out[0] = (counts > 0).astype(np.float32)
        else:
            dens = np.log1p(counts) if self.log_density else counts
            peak = dens.max()
            out[0] = (dens / peak).astype(np.float32) if peak > 0 else dens.astype(np.float32)

        if self.channels == 3:
            direction = np.zeros((s, s), dtype=np.float64)
            volume = np.zeros((s, s), dtype=np.float64)
            self._stamp(direction, row, col, np.sign(signed))
            self._stamp(volume, row, col, np.abs(signed))
            # Direction is a ratio in [-1, 1]: +1 all-forward, -1 all-backward.  Dividing
            # by the count keeps a busy bin from dominating a sparse one.
            with np.errstate(invalid="ignore", divide="ignore"):
                out[1] = np.nan_to_num(direction / np.maximum(counts, 1.0)).astype(np.float32)
            vpeak = volume.max()
            out[2] = (volume / vpeak).astype(np.float32) if vpeak > 0 else volume.astype(np.float32)
        return out

    def transform(self, flows: Sequence[Flow]) -> np.ndarray:
        if len(flows) == 0:
            return np.zeros((0, self.channels, self.size, self.size), dtype=np.float32)
        out = np.stack([self.transform_one(f) for f in flows]).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def blank_fraction(self, flows: Sequence[Flow]) -> float:
        """Fraction of flows whose image has no marked pixel at all.

        The paper's Figure 3c artifact -- a window in which no packet arrived renders as a
        blank plot and carries no signal.  Reported per run rather than silently trained on.
        """
        if not flows:
            return 0.0
        blank = sum(1 for f in flows if self._bins(f)[0].size == 0)
        return blank / len(flows)
