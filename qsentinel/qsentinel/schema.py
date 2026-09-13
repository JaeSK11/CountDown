"""Core data model shared by every dataset loader, feature extractor and model.

The central objects are:

``Packet``      one packet: timestamp, size, direction.
``Flow``        one bidirectional conversation (or, for some datasets, one capture).
``FlowDataset`` a labelled collection of flows that can emit ``(X, y)``.

Memory note: a ``Flow`` stores its packets as three parallel numpy arrays rather than
a list of ``Packet`` objects -- the datasets here reach tens of millions of packets and
per-packet Python objects do not fit.  ``Flow.packets`` materialises ``Packet`` objects
on demand so that the readable API still exists for small flows and tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterator, NamedTuple, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# Label taxonomy
# --------------------------------------------------------------------------------------


class TrafficType(str, Enum):
    """Unified ``traffic_type`` taxonomy across ISCXVPN2016 / ISCXTor2016."""

    BROWSING = "browsing"
    EMAIL = "email"
    CHAT = "chat"
    AUDIO_STREAMING = "audio_streaming"
    VIDEO_STREAMING = "video_streaming"
    FILE_TRANSFER = "file_transfer"
    VOIP = "voip"
    P2P = "p2p"


class TunnelType(str, Enum):
    """How the traffic is encapsulated -- drives PQC observability in Phase 2."""

    NONE = "none"
    VPN = "vpn"
    TOR = "tor"


#: Canonical ordering used whenever traffic-type labels are integer-encoded.
TRAFFIC_TYPES: tuple[str, ...] = tuple(t.value for t in TrafficType)
TUNNEL_TYPES: tuple[str, ...] = tuple(t.value for t in TunnelType)

#: Every label field a loader may populate.  ``target=`` selects one of these as ``y``.
LABEL_FIELDS: tuple[str, ...] = (
    "traffic_type",  # ISCXVPN, ISCXTor
    "tunnel_type",  # ISCXVPN, ISCXTor
    "is_vpn",  # ISCXVPN
    "is_tor",  # ISCXTor
    "app",  # CSTNET, PostQuantumTLS, MobileApp
    "activity",  # MobileApp
    # Phase 4: the ISCX capture stem normalised into a real class set.  `app` stays the
    # raw stem (and stays in `invalid_targets`); these are what a model may train on.
    "app_norm",  # ISCXVPN, ISCXTor -- the service: facebook, skype, youtube, ...
    "app_activity",  # ISCXVPN, ISCXTor -- service + modality: facebook_audio, ...
)

#: Direction convention.
FORWARD = 1  # client -> server
BACKWARD = -1  # server -> client

#: Packet sizes are capped here before they reach any feature extractor.
MTU_CAP = 1500

#: Targets whose class set is fixed by the taxonomy above rather than by whatever a
#: dataset happens to contain.  For these, a model's ``predict_proba`` columns are
#: dataset-independent; see ``LabelSpace``.
TAXONOMY_TARGETS: dict[str, tuple[str, ...]] = {
    "traffic_type": TRAFFIC_TYPES,
    "tunnel_type": TUNNEL_TYPES,
}


@dataclass(frozen=True)
class LabelSpace:
    """The canonical column ordering a model's ``predict_proba`` is written against.

    This is the ensemble seam.  A model binds to a ``LabelSpace``, *not* to whatever
    classes the dataset in front of it happened to realise, so that Phase 5 can stack a
    LightGBM, a CNN and a transformer by simply concatenating their probability blocks.

    The distinction that makes this necessary: ``FlowDataset.labels()`` encodes over the
    **observed** classes, so ``traffic_type`` is 7 columns wide on ISCXVPN (no ``browsing``
    capture exists) and 8 on ISCXTor -- column 5 means ``voip`` on one and ``file_transfer``
    on the other.  Stacking those two would be silently wrong.  Binding both models to
    ``LabelSpace.taxonomy("traffic_type")`` gives every member the same 8 columns, with a
    zero column for a class its training data never contained.

    Which constructor applies is a property of the *target*, not of the dataset:

    * taxonomy-backed (``traffic_type``, ``tunnel_type``) -> :meth:`taxonomy`, fixed width.
    * open-ended (``app``, ``activity``) -> :meth:`from_dataset`, and the result must be
      persisted next to the model artifact or its columns cannot be read back.
    """

    target: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError(f"LabelSpace({self.target!r}) needs at least one class")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"LabelSpace({self.target!r}) has duplicate class names")

    # -- construction ------------------------------------------------------------------
    @classmethod
    def taxonomy(cls, target: str) -> "LabelSpace":
        """The full, dataset-independent class set for a taxonomy-backed target."""
        if target not in TAXONOMY_TARGETS:
            raise ValueError(
                f"{target!r} is not taxonomy-backed (those are "
                f"{sorted(TAXONOMY_TARGETS)}); use LabelSpace.from_dataset() and persist "
                f"the result with the model artifact"
            )
        return cls(target=target, names=tuple(TAXONOMY_TARGETS[target]))

    @classmethod
    def from_dataset(cls, ds: "FlowDataset", target: str | None = None) -> "LabelSpace":
        """The right space for ``target`` on ``ds`` -- taxonomy if there is one, else observed."""
        target = target or ds.default_target
        if target in TAXONOMY_TARGETS:
            return cls.taxonomy(target)
        return cls(target=target, names=tuple(ds.label_names(target)))

    @classmethod
    def from_names(cls, target: str, names: Sequence[str]) -> "LabelSpace":
        return cls(target=target, names=tuple(str(n) for n in names))

    # -- basics ------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return str(name) in self.names

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LabelSpace({self.target!r}, {len(self.names)} classes)"

    def index(self, name: str) -> int:
        try:
            return self.names.index(str(name))
        except ValueError:
            raise KeyError(
                f"class {name!r} is not in the {self.target!r} LabelSpace {list(self.names)}"
            ) from None

    def name(self, i: int) -> str:
        return self.names[int(i)]

    # -- encoding ----------------------------------------------------------------------
    def encode(self, values: Sequence[Any]) -> np.ndarray:
        """String class labels -> canonical integer ids."""
        index = {n: i for i, n in enumerate(self.names)}
        out = np.empty(len(values), dtype=np.int64)
        for i, v in enumerate(values):
            key = str(v)
            if key not in index:
                raise KeyError(
                    f"class {key!r} is not in the {self.target!r} LabelSpace "
                    f"{list(self.names)}; the model and the data disagree about the "
                    f"class set"
                )
            out[i] = index[key]
        return out

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.names[int(i)] for i in ids]

    # -- the seam ----------------------------------------------------------------------
    def align(self, proba: np.ndarray, source: "LabelSpace") -> np.ndarray:
        """Re-express ``proba`` (whose columns follow ``source``) in *this* space.

        Classes this space has but ``source`` lacks become zero columns -- a model that
        never saw ``browsing`` reports zero probability for it rather than shifting every
        other column left by one.
        """
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape[1] != len(source):
            raise ValueError(
                f"proba has shape {proba.shape}, expected (N, {len(source)}) to match "
                f"the source space"
            )
        out = np.zeros((proba.shape[0], len(self)), dtype=proba.dtype)
        for j, nm in enumerate(source.names):
            if nm in self.names:
                out[:, self.index(nm)] = proba[:, j]
        return out

    # -- persistence -------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "names": list(self.names)}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelSpace":
        return cls(target=d["target"], names=tuple(d["names"]))

    @classmethod
    def from_json(cls, blob: str) -> "LabelSpace":
        return cls.from_dict(json.loads(blob))


# --------------------------------------------------------------------------------------
# Packets and flows
# --------------------------------------------------------------------------------------


class FiveTuple(NamedTuple):
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    l4_proto: str  # "tcp" | "udp" | "wlan" (MobileApp frame-level captures)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}/{self.l4_proto}"


@dataclass(slots=True)
class Packet:
    """A single observed packet, oriented relative to the flow's client."""

    ts: float
    size: int
    direction: int  # +1 client->server, -1 server->client


@dataclass(slots=True)
class Flow:
    """One bidirectional flow.

    ``timestamps`` / ``sizes`` / ``directions`` are parallel arrays in capture order.
    ``sizes`` are raw byte counts (uncapped); the MTU cap is applied by feature extractors
    so that the cap stays a feature-level parameter.
    """

    flow_id: str
    five_tuple: FiveTuple
    timestamps: np.ndarray  # float64, seconds, absolute
    sizes: np.ndarray  # int32, bytes
    directions: np.ndarray  # int8, +1 / -1
    dataset: str = ""
    label: str = ""  # value of the loader's default target
    label_fields: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- reserved for Phase 2 (crypto detection); left None in Phase 0 -----------------
    tls_version: str | None = None
    cipher_suite: str | None = None
    kem_group: str | None = None
    pqc_verdict: str | None = None

    # -- Phase 1: ingest & flow reassembly ---------------------------------------------
    #: Coarse, transport-level protocol guess from ``flows/handshake.py``.  Advisory:
    #: Phase 2 does the authoritative parsing and may disagree.
    l7_hint: str | None = None
    #: Reassembled opening bytes of each direction (the handshake window).  These are what
    #: Phase 2 parses into TLS version / cipher suite / KEM group.  ``None`` when the
    #: direction carried no payload, or when handshake capture was disabled.
    handshake_client_bytes: bytes | None = None
    handshake_server_bytes: bytes | None = None
    #: True when the captured window cannot be trusted as the true start of the stream --
    #: a mid-stream capture (no SYN), or an unfilled gap in the head reassembly.  Phase 2
    #: should treat a parse failure on such a flow as "not observable", not "not PQC".
    handshake_incomplete: bool = False
    #: Why the reassembler closed this flow: fin / rst / idle_timeout / max_duration /
    #: end_of_capture / table_full.
    expiry_reason: str | None = None

    # -- reserved for Phase 4 (byte-level models) --------------------------------------
    # Every byte-level representation -- ET-BERT, YaTC, nPrint-style bit matrices --
    # needs the raw packet bytes, which Phase 0 deliberately does not retain (keeping
    # them for 90 GB of pcaps is what the flow_stats/packet_seq reduction avoids).  The
    # slot exists so that Phase 1's extraction has somewhere to put them and Phase 4 does
    # not force a second pass over the corpus.
    raw_bytes: list[bytes] | None = None

    # -- derived -----------------------------------------------------------------------
    @property
    def client_ip(self) -> str:
        """The initiator's address.  ``five_tuple`` is canonicalised client-first."""
        return self.five_tuple.src_ip

    @property
    def client_port(self) -> int:
        return self.five_tuple.src_port

    @property
    def server_ip(self) -> str:
        return self.five_tuple.dst_ip

    @property
    def server_port(self) -> int:
        return self.five_tuple.dst_port

    @property
    def l4_proto(self) -> str:
        return self.five_tuple.l4_proto

    @property
    def source_file(self) -> str | None:
        """Originating capture, when the flow came from a file-backed source."""
        return self.meta.get("source_file")

    @property
    def has_handshake(self) -> bool:
        """True when at least one direction yielded reassembled opening bytes."""
        return bool(self.handshake_client_bytes or self.handshake_server_bytes)

    @property
    def n_packets(self) -> int:
        return int(self.sizes.shape[0])

    @property
    def start_ts(self) -> float:
        return float(self.timestamps[0]) if self.n_packets else 0.0

    @property
    def end_ts(self) -> float:
        return float(self.timestamps[-1]) if self.n_packets else 0.0

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def packets(self) -> list[Packet]:
        """Materialise ``Packet`` objects.  Convenience only -- O(n) allocations."""
        return [
            Packet(float(t), int(s), int(d))
            for t, s, d in zip(self.timestamps, self.sizes, self.directions)
        ]

    def relative_timestamps(self) -> np.ndarray:
        if self.n_packets == 0:
            return self.timestamps
        return self.timestamps - self.timestamps[0]

    @staticmethod
    def from_packets(
        flow_id: str,
        five_tuple: FiveTuple,
        packets: Sequence[Packet],
        **kwargs: Any,
    ) -> "Flow":
        ts = np.array([p.ts for p in packets], dtype=np.float64)
        sz = np.array([p.size for p in packets], dtype=np.int32)
        dr = np.array([p.direction for p in packets], dtype=np.int8)
        return Flow(flow_id=flow_id, five_tuple=five_tuple, timestamps=ts, sizes=sz,
                    directions=dr, **kwargs)


# --------------------------------------------------------------------------------------
# Group keys (leakage-safe splitting)
# --------------------------------------------------------------------------------------

def _capture_day(f: Flow) -> str | None:
    """UTC calendar date of the flow's first packet."""
    if f.n_packets == 0:
        return None
    return datetime.fromtimestamp(f.start_ts, timezone.utc).strftime("%Y-%m-%d")


#: Named grouping keys for ``FlowDataset.groups()``.  A split must keep every sample that
#: shares a group on one side, or correlated near-duplicates leak into the test set.
#:
#: Which key is *right* is per-dataset and lives in ``configs/datasets.yaml`` under
#: ``group_keys:`` -- ``source_file`` is correct where one capture yields many flows
#: (ISCX) and meaningless where it yields exactly one (CSTNET, MobileApp).  This module
#: only resolves the key; ``qsentinel.eval.splits`` is what refuses a grouping that has
#: degenerated into a random split.
GROUP_KEYS: dict[str, Callable[[Flow], str | None]] = {
    "source_file": lambda f: f.meta.get("source_file"),
    "capture_day": _capture_day,
    "server_ip": lambda f: f.five_tuple.dst_ip,
    "stem": lambda f: f.meta.get("stem"),
}


# --------------------------------------------------------------------------------------
# Dataset container
# --------------------------------------------------------------------------------------


class FlowDataset:
    """A labelled collection of flows plus the ``(X, y)`` accessors Phase 3 needs."""

    def __init__(
        self,
        flows: list[Flow],
        name: str,
        default_target: str,
        config: Any = None,
        invalid_targets: Any = None,
    ) -> None:
        self.flows = flows
        self.name = name
        self.default_target = default_target
        self.config = config
        #: Targets that exist as label fields but are NOT valid classification targets for
        #: this dataset (e.g. ISCX "app" = capture-filename stems, ~1 class per source file).
        self.invalid_targets = frozenset(invalid_targets or ())
        self._feature_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._label_names: dict[str, list[str]] = {}

    # -- basics ------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.flows)

    def __iter__(self) -> Iterator[Flow]:
        return iter(self.flows)

    def __getitem__(self, i: int) -> Flow:
        return self.flows[i]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FlowDataset(name={self.name!r}, flows={len(self.flows)}, "
                f"default_target={self.default_target!r})")

    # -- labels ------------------------------------------------------------------------
    def _check_target(self, target: str) -> None:
        """Reject a target that exists as a label field but isn't a real class set here."""
        if target in self.invalid_targets:
            raise ValueError(
                f"target {target!r} is not a valid classification target for dataset "
                f"{self.name!r}: its {target!r} labels are capture-filename stems (~1 class "
                f"per source file), so a group-aware split is undefined and cross-dataset "
                f"comparison is meaningless. Use {self.default_target!r} or pool datasets. "
                f"See phases/PHASE-3.md."
            )

    def raw_labels(self, target: str | None = None) -> list[Any]:
        target = target or self.default_target
        self._check_target(target)
        out = []
        for f in self.flows:
            if target not in f.label_fields:
                raise KeyError(
                    f"flow {f.flow_id!r} in dataset {self.name!r} has no label field "
                    f"{target!r}; available: {sorted(f.label_fields)}"
                )
            out.append(f.label_fields[target])
        return out

    def label_names(self, target: str | None = None) -> list[str]:
        target = target or self.default_target
        if target not in self._label_names:
            self.labels(target)
        return self._label_names[target]

    def _string_labels(self, target: str) -> list[str]:
        """Raw label values normalised to the strings every encoder here keys on."""
        raw = self.raw_labels(target)
        if raw and isinstance(raw[0], (bool, np.bool_)):
            return ["true" if bool(v) else "false" for v in raw]
        return [str(v) for v in raw]

    def labels(
        self, target: str | None = None, space: "LabelSpace | None" = None
    ) -> np.ndarray:
        """Integer-encoded labels for ``target`` (default: the loader's target).

        With no ``space``, ids are **dataset-local**: they index the classes this dataset
        actually contains, in taxonomy order for ``traffic_type`` / ``tunnel_type`` and in
        sorted order otherwise.  That is the right encoding for describing a dataset, and
        it is what ``label_names()`` / ``class_counts()`` report.

        Pass ``space=`` to encode into a canonical :class:`LabelSpace` instead.  That is
        what a model trains on, because dataset-local ids mean different things on
        different datasets (``traffic_type`` is 7 wide on ISCXVPN, 8 on ISCXTor) and the
        Phase-5 combiner needs one fixed column meaning.
        """
        target = target or self.default_target
        values = self._string_labels(target)

        if space is not None:
            if space.target != target:
                raise ValueError(
                    f"LabelSpace is for target {space.target!r}, but labels were "
                    f"requested for {target!r}"
                )
            return space.encode(values)

        if target == "traffic_type":
            names = [t for t in TRAFFIC_TYPES if t in set(values)]
        elif target == "tunnel_type":
            names = [t for t in TUNNEL_TYPES if t in set(values)]
        elif self.raw_labels(target) and isinstance(
            self.raw_labels(target)[0], (bool, np.bool_)
        ):
            names = ["false", "true"]
        else:
            names = sorted(set(values))

        index = {n: i for i, n in enumerate(names)}
        self._label_names[target] = names
        return np.array([index[v] for v in values], dtype=np.int64)

    @property
    def num_classes(self) -> int:
        """Number of classes this dataset realises for its default target.

        A *model* reports ``len(model.label_space)`` instead, which can be wider -- see
        :class:`LabelSpace`.
        """
        return len(self.label_names())

    # -- grouping ----------------------------------------------------------------------
    def groups(self, key: str = "source_file") -> np.ndarray:
        """``(N,)`` group ids for a leakage-safe split, resolved by name.

        Raises rather than falling back when a flow cannot supply the key: a silent
        fallback would produce one group per sample, which looks exactly like a valid
        grouping and is in fact a random split.
        """
        if key not in GROUP_KEYS:
            raise KeyError(f"unknown group key {key!r}; available: {sorted(GROUP_KEYS)}")
        resolve = GROUP_KEYS[key]
        out: list[str] = []
        for f in self.flows:
            value = resolve(f)
            if value is None or value == "":
                raise ValueError(
                    f"flow {f.flow_id!r} in dataset {self.name!r} cannot supply group key "
                    f"{key!r}; available meta: {sorted(f.meta)}"
                )
            out.append(str(value))
        return np.asarray(out, dtype=object)

    def class_counts(self, target: str | None = None) -> dict[str, int]:
        target = target or self.default_target
        y = self.labels(target)
        names = self._label_names[target]
        counts = np.bincount(y, minlength=len(names))
        return {n: int(c) for n, c in zip(names, counts)}

    # -- features ----------------------------------------------------------------------
    def features(self, extractor: str = "flow_stats", **kwargs: Any) -> np.ndarray:
        """``(N, F)`` float32 matrix (``flow_stats``) or ``(N, ...)`` tensor (others)."""
        X, names = self._extract(extractor, **kwargs)
        return X

    def feature_names(self, extractor: str = "flow_stats", **kwargs: Any) -> list[str]:
        _, names = self._extract(extractor, **kwargs)
        return names

    def _extract(self, extractor: str, **kwargs: Any) -> tuple[np.ndarray, list[str]]:
        from qsentinel.features import get_extractor  # local import: avoids a cycle

        key = extractor + "|" + json.dumps(kwargs, sort_keys=True, default=str)
        if key not in self._feature_cache:
            ex = get_extractor(extractor, config=self.config, **kwargs)
            self._feature_cache[key] = (ex.transform(self.flows), list(ex.feature_names))
        return self._feature_cache[key]

    # -- splitting ---------------------------------------------------------------------
    def split(
        self,
        target: str | None = None,
        test_size: float = 0.2,
        val_size: float = 0.0,
        seed: int = 42,
        stratify: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return index arrays ``{"train": ..., "val": ..., "test": ...}``.

        Indices (not sub-datasets) are returned so that callers can slice whichever
        feature representation they built.
        """
        y = self.labels(target)
        n = len(y)
        rng = np.random.default_rng(seed)
        idx = np.arange(n)

        def _take(pool: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
            if frac <= 0:
                return np.array([], dtype=np.int64), pool
            if not stratify:
                shuffled = rng.permutation(pool)
                k = int(round(frac * len(pool)))
                return np.sort(shuffled[:k]), np.sort(shuffled[k:])
            picked = []
            for cls in np.unique(y[pool]):
                members = rng.permutation(pool[y[pool] == cls])
                k = int(round(frac * len(members)))
                # keep at least one sample per class on both sides when possible
                k = min(max(k, 1), max(len(members) - 1, 0)) if len(members) > 1 else 0
                picked.append(members[:k])
            hold = np.sort(np.concatenate(picked)) if picked else np.array([], dtype=np.int64)
            rest = np.sort(np.setdiff1d(pool, hold))
            return hold, rest

        test, rest = _take(idx, test_size)
        val, train = _take(rest, val_size / (1 - test_size) if test_size < 1 else 0.0)
        return {"train": train, "val": val, "test": test}

    # -- filtering ---------------------------------------------------------------------
    def filter(
        self,
        target: str | None = None,
        min_samples_per_class: int = 0,
        top_k_classes: int | None = None,
    ) -> "FlowDataset":
        """New dataset with rare / non-top-k classes dropped."""
        target = target or self.default_target
        raw = [str(v) for v in self.raw_labels(target)]
        counts: dict[str, int] = {}
        for v in raw:
            counts[v] = counts.get(v, 0) + 1

        keep = {k for k, c in counts.items() if c >= min_samples_per_class}
        if top_k_classes is not None:
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            keep &= {k for k, _ in ranked[:top_k_classes]}

        flows = [f for f, v in zip(self.flows, raw) if v in keep]
        # invalid_targets must survive: it is the guard that stops "app" being trained on
        # ISCX capture stems, and every filtered dataset is still that same dataset.
        return FlowDataset(
            flows, self.name, self.default_target, self.config, self.invalid_targets
        )
