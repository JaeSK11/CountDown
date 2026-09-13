"""Dataset loader interface, registry, and the on-disk flow cache.

Every loader:

1. ``discover()`` -- enumerate source files and attach label fields (no parsing).
2. ``build_flows()`` -- parse those files into ``Flow`` objects (parallel by default).
3. ``load()`` -- wrap them in a ``FlowDataset``, reading/writing the parquet cache.

Cache keys hash the dataset spec + flow params + schema version, so changing a label rule or
``flow_timeout`` invalidates the cache automatically rather than serving stale flows.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from qsentinel.config import Config, default_config, get_logger
from qsentinel.flows.extract import FlowExtractor, ParseStats
from qsentinel.schema import FiveTuple, Flow, FlowDataset

log = get_logger(__name__)

_REGISTRY: dict[str, type["DatasetLoader"]] = {}

DEFAULT_WORKERS = min(16, (os.cpu_count() or 4))


def register_loader(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_datasets() -> list[str]:
    return sorted(_REGISTRY)


def get_loader(name: str, config: Config | None = None) -> "DatasetLoader":
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {available_datasets()}")
    config = config or default_config()
    return _REGISTRY[name](config)


@dataclass
class Item:
    """One source file plus the labels its path implies."""

    path: Path
    label_fields: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Parallel parse worker (module level so that it is picklable)
# --------------------------------------------------------------------------------------


def _parse_item(payload: tuple) -> tuple[list[dict], dict, str]:
    """Worker: parse one file into serialisable flow dicts."""
    path, label_fields, meta, flow_params, dataset, granularity, default_target = payload
    extractor = FlowExtractor(**flow_params)
    stats = ParseStats()
    try:
        flows = extractor.extract(
            path,
            dataset=dataset,
            label=str(label_fields.get(default_target, "")),
            label_fields=label_fields,
            meta=meta,
            stats=stats,
        )
    except Exception as exc:  # a corrupt capture must not kill the whole load
        log.warning("failed to parse %s: %s: %s", path, type(exc).__name__, exc)
        return [], ParseStats().as_dict(), str(path)

    if granularity == "session" and flows:
        flows = [max(flows, key=lambda f: f.n_packets)]
    return [_flow_to_dict(f) for f in flows], stats.as_dict(), str(path)


def _flow_to_dict(f: Flow) -> dict:
    return {
        "flow_id": f.flow_id,
        "src_ip": f.five_tuple.src_ip,
        "src_port": int(f.five_tuple.src_port),
        "dst_ip": f.five_tuple.dst_ip,
        "dst_port": int(f.five_tuple.dst_port),
        "l4_proto": f.five_tuple.l4_proto,
        "dataset": f.dataset,
        "label": f.label,
        "label_fields": json.dumps(f.label_fields, default=str),
        "meta": json.dumps(f.meta, default=str),
        "timestamps": f.timestamps.tolist(),
        "sizes": f.sizes.astype(np.int32).tolist(),
        "directions": f.directions.astype(np.int8).tolist(),
        "tls_version": f.tls_version,
        "cipher_suite": f.cipher_suite,
        "kem_group": f.kem_group,
        "pqc_verdict": f.pqc_verdict,
        # Phase 1: what Phase 2 parses into TLS version / cipher / KEM.
        "l7_hint": f.l7_hint,
        "handshake_client_bytes": f.handshake_client_bytes,
        "handshake_server_bytes": f.handshake_server_bytes,
        "handshake_incomplete": bool(f.handshake_incomplete),
        "expiry_reason": f.expiry_reason,
    }


def _dict_to_flow(d: dict) -> Flow:
    return Flow(
        flow_id=d["flow_id"],
        five_tuple=FiveTuple(
            d["src_ip"], int(d["src_port"]), d["dst_ip"], int(d["dst_port"]), d["l4_proto"]
        ),
        timestamps=np.asarray(d["timestamps"], dtype=np.float64),
        sizes=np.asarray(d["sizes"], dtype=np.int32),
        directions=np.asarray(d["directions"], dtype=np.int8),
        dataset=d["dataset"],
        label=d["label"],
        label_fields=json.loads(d["label_fields"]),
        meta=json.loads(d["meta"]),
        tls_version=d.get("tls_version"),
        cipher_suite=d.get("cipher_suite"),
        kem_group=d.get("kem_group"),
        pqc_verdict=d.get("pqc_verdict"),
        # .get(): a cache written before Phase 1 simply has no handshake columns.
        l7_hint=d.get("l7_hint"),
        handshake_client_bytes=d.get("handshake_client_bytes"),
        handshake_server_bytes=d.get("handshake_server_bytes"),
        handshake_incomplete=bool(d.get("handshake_incomplete") or False),
        expiry_reason=d.get("expiry_reason"),
    )


_ARROW_SCHEMA = pa.schema([
    ("flow_id", pa.string()),
    ("src_ip", pa.string()),
    ("src_port", pa.int32()),
    ("dst_ip", pa.string()),
    ("dst_port", pa.int32()),
    ("l4_proto", pa.string()),
    ("dataset", pa.string()),
    ("label", pa.string()),
    ("label_fields", pa.string()),
    ("meta", pa.string()),
    ("timestamps", pa.list_(pa.float64())),
    ("sizes", pa.list_(pa.int32())),
    ("directions", pa.list_(pa.int8())),
    ("tls_version", pa.string()),
    ("cipher_suite", pa.string()),
    ("kem_group", pa.string()),
    ("pqc_verdict", pa.string()),
    ("l7_hint", pa.string()),
    ("handshake_client_bytes", pa.binary()),
    ("handshake_server_bytes", pa.binary()),
    ("handshake_incomplete", pa.bool_()),
    ("expiry_reason", pa.string()),
])

#: Columns added in Phase 1.  A cache written before then lacks them; loading such a file
#: is legal (the flows are still correct for Phase 0/1 features) but Phase 2 needs the
#: handshake bytes, so the loader says so out loud instead of serving silent ``None``s.
_PHASE1_COLUMNS = ("l7_hint", "handshake_client_bytes", "handshake_server_bytes")


# --------------------------------------------------------------------------------------
# Loader base class
# --------------------------------------------------------------------------------------


class DatasetLoader(ABC):
    """Base class for every dataset loader."""

    name: str = "base"

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or default_config()
        self.spec = self.config.dataset(self.name)
        self.root = self.config.dataset_path(self.name)
        self.default_target = self.spec.get("default_target", "traffic_type")
        self.granularity = self.spec.get("granularity", "split")
        self.stats = ParseStats()
        self.unmapped: list[str] = []

    # -- to implement ------------------------------------------------------------------
    @abstractmethod
    def discover(self) -> list[Item]:
        """Enumerate source files with their label fields.  No parsing."""

    # -- file discovery helpers --------------------------------------------------------
    def _glob(self, base: Path | None = None) -> list[Path]:
        base = base or self.root
        patterns = self.spec.get("glob") or ["**/*.pcap"]
        if isinstance(patterns, str):
            patterns = [patterns]
        seen: dict[Path, None] = {}
        for pattern in patterns:
            for p in sorted(base.glob(pattern)):
                if p.is_file():
                    seen[p] = None
        return list(seen)

    # -- parsing -----------------------------------------------------------------------
    def build_flows(self, items: Sequence[Item], workers: int | None = None) -> list[Flow]:
        """Parse items into flows, in parallel across processes."""
        workers = DEFAULT_WORKERS if workers is None else workers
        # `reassembly` rides along so the loader path captures handshake bytes too; it is
        # not part of flow_key (see Config.flow_key) because it cannot change (X, y).
        flow_params = {**dict(self.config.flow), **dict(self.config.reassembly)}
        payloads = [
            (
                str(it.path),
                it.label_fields,
                it.meta,
                flow_params,
                self.name,
                self.granularity,
                self.default_target,
            )
            for it in items
        ]

        records: list[dict] = []
        t0 = time.time()
        if workers <= 1 or len(payloads) == 1:
            results: Iterable = (_parse_item(p) for p in payloads)
            for i, (recs, st, _path) in enumerate(results, 1):
                records.extend(recs)
                self.stats.merge(_stats_from_dict(st))
                self._progress(i, len(payloads), len(records), t0)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_parse_item, p) for p in payloads]
                for i, fut in enumerate(as_completed(futures), 1):
                    recs, st, _path = fut.result()
                    records.extend(recs)
                    self.stats.merge(_stats_from_dict(st))
                    self._progress(i, len(payloads), len(records), t0)

        records.sort(key=lambda r: (r["meta"], r["flow_id"]))
        return [_dict_to_flow(r) for r in records]

    def _progress(self, done: int, total: int, n_flows: int, t0: float) -> None:
        step = max(1, total // 20)
        if done % step == 0 or done == total:
            log.info(
                "[%s] parsed %d/%d files -> %d flows (%.1fs)",
                self.name, done, total, n_flows, time.time() - t0,
            )

    # -- cache -------------------------------------------------------------------------
    def _cache_file(self) -> Path:
        return self.config.cache_path(
            self.name, f"flows_{self.config.flow_key(self.name)}.parquet"
        )

    def _write_cache(self, flows: Sequence[Flow], path: Path) -> None:
        table = pa.Table.from_pylist([_flow_to_dict(f) for f in flows], schema=_ARROW_SCHEMA)
        pq.write_table(table, path, compression="zstd")
        path.with_suffix(".stats.json").write_text(
            json.dumps({"stats": self.stats.as_dict(), "n_flows": len(flows)}, indent=2)
        )
        log.info("[%s] cached %d flows -> %s (%.1f MB)",
                 self.name, len(flows), path.name, path.stat().st_size / 1e6)

    def _read_cache(self, path: Path) -> list[Flow]:
        table = pq.read_table(path)
        if any(c not in table.column_names for c in _PHASE1_COLUMNS):
            log.warning(
                "[%s] cache %s predates Phase 1: no handshake bytes or l7_hint. Flows are "
                "valid for Phase 0/1 features; re-parse with refresh=True before Phase 2 "
                "needs the handshake window.", self.name, path.name,
            )
        flows = [_dict_to_flow(d) for d in table.to_pylist()]
        side = path.with_suffix(".stats.json")
        if side.exists():
            self.stats = _stats_from_dict(json.loads(side.read_text()).get("stats", {}))
        log.info("[%s] loaded %d flows from cache %s", self.name, len(flows), path.name)
        return flows

    # -- main entry point --------------------------------------------------------------
    def load(
        self,
        target: str | None = None,
        use_cache: bool = True,
        refresh: bool = False,
        workers: int | None = None,
        limit: int | None = None,
    ) -> FlowDataset:
        cache_file = self._cache_file()
        flows: list[Flow] | None = None

        if use_cache and not refresh and cache_file.exists() and limit is None:
            try:
                flows = self._read_cache(cache_file)
            except Exception as exc:
                log.warning("[%s] cache unreadable (%s); re-parsing", self.name, exc)
                flows = None

        if flows is None:
            items = self.discover()
            if limit is not None:
                items = items[:limit]
            log.info("[%s] discovered %d source files", self.name, len(items))
            flows = self.build_flows(items, workers=workers)
            if use_cache and limit is None:
                self._write_cache(flows, cache_file)

        flows = self.postprocess(flows)
        ds = FlowDataset(
            flows, name=self.name,
            default_target=target or self.default_target,
            config=self.config,
            invalid_targets=self.spec.get("invalid_targets"),
        )
        return self.apply_filters(ds)

    def postprocess(self, flows: list[Flow]) -> list[Flow]:
        """Hook for loaders that need to adjust flows after parsing/caching.

        The base implementation adds the normalised ``app_norm`` / ``app_activity`` label
        fields for any dataset listed in ``app_rules_datasets:``.  It runs *here* rather
        than in ``discover()`` because ``discover`` output is what gets cached, and
        ``Config.flow_key`` hashes the dataset spec -- normalising there would orphan
        every parsed-flow cache to add a label that cannot change ``X``.
        """
        if self.config.normalizes_app_labels(self.name):
            from qsentinel.data.app_labels import AppLabelRules, normalize_app_labels

            flows = normalize_app_labels(
                flows,
                AppLabelRules.from_config(self.config),
                on_unmapped=self.spec.get("on_unmapped", "error"),
                dataset=self.name,
            )
        return flows

    def apply_filters(self, ds: FlowDataset) -> FlowDataset:
        filters = self.spec.get("filters") or {}
        min_samples = filters.get("min_samples_per_class") or 0
        top_k = filters.get("top_k_apps") or filters.get("top_k_classes")
        if min_samples or top_k:
            before = len(ds)
            ds = ds.filter(min_samples_per_class=min_samples, top_k_classes=top_k)
            log.info("[%s] filters kept %d/%d flows", self.name, len(ds), before)
        return ds

    # -- label mapping helpers ---------------------------------------------------------
    def map_traffic_type(self, stem: str, rules: Sequence[dict]) -> str | None:
        import re

        for rule in rules:
            if re.search(rule["pattern"], stem):
                return rule["traffic_type"]
        return None

    def report_unmapped(self) -> None:
        """Fail loudly (or warn) when a filename matched no label rule."""
        if not self.unmapped:
            return
        mode = self.spec.get("on_unmapped", "error")
        msg = (
            f"[{self.name}] {len(self.unmapped)} source files matched no label rule: "
            f"{sorted(set(self.unmapped))}"
        )
        if mode == "error":
            raise ValueError(msg + " -- add a rule to configs/datasets.yaml")
        log.warning(msg)


def _stats_from_dict(d: dict) -> ParseStats:
    st = ParseStats()
    for k, v in d.items():
        if hasattr(st, k):
            setattr(st, k, v)
    return st


def load(
    name: str,
    target: str | None = None,
    config: Config | None = None,
    **kwargs: Any,
) -> FlowDataset:
    """Load a dataset -- or a pooled alias -- by name.  The Phase 0 entry point.

    >>> ds = load("iscxvpn", target="traffic_type")
    >>> ds = load("iscx_pooled", target="traffic_type")   # ISCXVPN + ISCXTor
    """
    config = config or default_config()
    if config.pooled_spec(name) is not None:
        from qsentinel.data.pooled import load_alias  # local import: avoids a cycle

        return load_alias(name, target=target, config=config, **kwargs)
    if name not in _REGISTRY:
        from qsentinel.data.pooled import available_pooled

        raise KeyError(
            f"unknown dataset {name!r}; available: {available_datasets()}, "
            f"pooled: {available_pooled(config)}"
        )
    loader = get_loader(name, config)
    return loader.load(target=target, **kwargs)
