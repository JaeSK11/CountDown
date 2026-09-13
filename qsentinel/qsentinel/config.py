"""Configuration loading.

Everything downstream (loaders, flow extraction, feature extractors, caching) reads its
parameters from a single ``Config`` object so that a run is reproducible from the two
YAML files in ``configs/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Repo root (the directory containing ``configs/`` and the ``data`` symlink).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

_LOG_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> logging.Logger:
    """Configure root logging once; honours ``QSENTINEL_LOG_LEVEL``."""
    global _LOG_CONFIGURED
    if level is None:
        level = os.environ.get("QSENTINEL_LOG_LEVEL", "INFO")
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    logging.getLogger("qsentinel").setLevel(level)
    return logging.getLogger("qsentinel")


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


@dataclass
class Config:
    """Merged view of ``configs/datasets.yaml`` and ``configs/features.yaml``."""

    datasets: dict[str, Any] = field(default_factory=dict)
    #: Per-dataset grouping key for leakage-safe splitting (Phase 3).  Kept out of the
    #: dataset specs on purpose -- see ``flow_key``.
    group_keys: dict[str, str] = field(default_factory=dict)
    #: Datasets whose grouping is vacuous by construction (1 capture == 1 sample).
    allow_vacuous_groups: list[str] = field(default_factory=list)
    #: Ordered capture-stem -> application rules (Phase 4).  Top-level for the same reason
    #: as ``group_keys``: they must not re-hash ``flow_key`` and orphan the flow caches.
    app_rules: list[dict[str, str]] = field(default_factory=list)
    #: traffic_type -> the short modality token used to build ``app_activity``.
    app_modality: dict[str, str] = field(default_factory=dict)
    #: Datasets whose ``app`` stem the rules above should normalise.
    app_rules_datasets: list[str] = field(default_factory=list)
    #: Named pooled datasets: ``{alias: {members: [...], default_target: ...}}``.
    pooled: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    flow: dict[str, Any] = field(default_factory=dict)
    reassembly: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    repo_root: Path = REPO_ROOT
    data_root: Path = REPO_ROOT / "data"
    cache_dir: Path = REPO_ROOT / "cache"

    # -- construction ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        datasets_yaml: str | Path | None = None,
        features_yaml: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> "Config":
        datasets_yaml = Path(datasets_yaml or CONFIG_DIR / "datasets.yaml")
        features_yaml = Path(features_yaml or CONFIG_DIR / "features.yaml")

        with open(datasets_yaml) as fh:
            d_raw = yaml.safe_load(fh) or {}
        with open(features_yaml) as fh:
            f_raw = yaml.safe_load(fh) or {}

        cfg = cls(
            datasets=d_raw.get("datasets", {}),
            group_keys=d_raw.get("group_keys", {}),
            allow_vacuous_groups=list(d_raw.get("allow_vacuous_groups", []) or []),
            app_rules=list(d_raw.get("app_rules", []) or []),
            app_modality=dict(d_raw.get("app_modality", {}) or {}),
            app_rules_datasets=list(d_raw.get("app_rules_datasets", []) or []),
            pooled=d_raw.get("pooled", {}),
            features=f_raw.get("features", {}),
            flow=f_raw.get("flow", {}),
            reassembly=f_raw.get("reassembly", {}),
            seed=int(f_raw.get("seed", 42)),
        )
        data_root = d_raw.get("data_root")
        if data_root:
            cfg.data_root = (REPO_ROOT / data_root).resolve()
        cache_dir = f_raw.get("cache_dir")
        if cache_dir:
            cfg.cache_dir = (REPO_ROOT / cache_dir).resolve()

        for key, value in (overrides or {}).items():
            _deep_set(cfg, key, value)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    # -- accessors ---------------------------------------------------------------------
    def dataset(self, name: str) -> dict[str, Any]:
        if name not in self.datasets:
            raise KeyError(
                f"unknown dataset {name!r}; configured: {sorted(self.datasets)}"
            )
        spec = dict(self.datasets[name])
        spec.setdefault("name", name)
        return spec

    def dataset_path(self, name: str) -> Path:
        spec = self.dataset(name)
        return (self.data_root / spec["root"]).resolve()

    def pooled_spec(self, name: str) -> dict[str, Any] | None:
        """The spec for a pooled alias, or ``None`` if ``name`` is a plain dataset."""
        if name not in self.pooled:
            return None
        spec = dict(self.pooled[name])
        spec.setdefault("name", name)
        if not spec.get("members"):
            raise KeyError(f"pooled dataset {name!r} declares no members")
        return spec

    def group_key(self, name: str) -> str:
        """The grouping key a leakage-safe split should use for ``name``.

        Resolved for pooled aliases too; falls back to ``source_file`` only when the
        dataset is unknown to both blocks, which a caller has already had to opt into.
        """
        pooled = self.pooled_spec(name)
        if pooled is not None:
            return str(pooled.get("group_key", "source_file"))
        return str(self.group_keys.get(name, "source_file"))

    def allows_vacuous_groups(self, name: str) -> bool:
        pooled = self.pooled_spec(name)
        if pooled is not None:
            members = pooled["members"]
            return any(m in self.allow_vacuous_groups for m in members)
        return name in self.allow_vacuous_groups

    def normalizes_app_labels(self, name: str) -> bool:
        """Whether ``name``'s ``app`` stem should be normalised into ``app_norm``.

        True for a pooled alias when any constituent opts in, so ``iscx_pooled`` inherits
        the rules from ISCXVPN and ISCXTor rather than needing its own entry.
        """
        if not self.app_rules:
            return False
        pooled = self.pooled_spec(name)
        if pooled is not None:
            return any(m in self.app_rules_datasets for m in pooled["members"])
        return name in self.app_rules_datasets

    def feature_params(self, extractor: str) -> dict[str, Any]:
        params = dict(self.features.get(extractor, {}))
        params.setdefault("mtu_cap", self.flow.get("mtu_cap", 1500))
        return params

    # -- cache keys --------------------------------------------------------------------
    def flow_key(self, name: str) -> str:
        """Hash of everything that affects the parsed-flow output for one dataset.

        ``reassembly`` is deliberately **not** hashed in.  Those parameters only add
        handshake bytes and an L7 hint to each flow; they cannot change the
        sizes/timings/directions that ``(X, y)`` is built from (proven byte-for-byte by
        ``tests/test_phase0_regression.py``).  Hashing them would invalidate every
        existing cache and force a re-parse of the whole corpus for no feature change.
        The cost is that a cache written before Phase 1 has no handshake columns --
        ``DatasetLoader._read_cache`` detects that and warns, so Phase 2 gets a loud
        signal rather than silent ``None``s.

        Likewise ``group_keys`` / ``pooled`` (Phase 3) are separate top-level blocks in
        ``datasets.yaml`` rather than keys inside a dataset spec, for exactly this reason:
        anything added to the spec dict below re-hashes and orphans every parsed cache.
        """
        payload = {
            "dataset": self.dataset(name),
            "flow": self.flow,
            "schema_version": SCHEMA_VERSION,
        }
        return _hash(payload)

    def feature_key(self, name: str, extractor: str, **kwargs: Any) -> str:
        payload = {
            "flow_key": self.flow_key(name),
            "extractor": extractor,
            "params": {**self.feature_params(extractor), **kwargs},
            "schema_version": SCHEMA_VERSION,
        }
        return _hash(payload)

    def cache_path(self, name: str, filename: str) -> Path:
        p = self.cache_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p / filename


#: Bump when the on-disk cache format or flow semantics change (invalidates caches).
SCHEMA_VERSION = 1


def _hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _deep_set(cfg: Config, dotted: str, value: Any) -> None:
    """``_deep_set(cfg, "flow.min_packets", 2)``."""
    head, _, tail = dotted.partition(".")
    if not hasattr(cfg, head):
        raise KeyError(f"no config section {head!r}")
    target = getattr(cfg, head)
    if not tail:
        setattr(cfg, head, value)
        return
    parts = tail.split(".")
    for p in parts[:-1]:
        target = target.setdefault(p, {})
    target[parts[-1]] = value


_DEFAULT: Config | None = None


def default_config() -> Config:
    """Process-wide default config (lazily loaded from ``configs/``)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Config.load()
    return _DEFAULT
