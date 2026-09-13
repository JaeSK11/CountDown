"""Feature-extractor interface and registry.

An extractor turns ``list[Flow]`` into an ``(N, ...)`` array.  Extractors are registered by
name so that ``FlowDataset.features("flow_stats")`` resolves without imports at the call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import numpy as np

from countdown.schema import Flow

_REGISTRY: dict[str, type["FeatureExtractor"]] = {}


def register_extractor(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def available_extractors() -> list[str]:
    return sorted(_REGISTRY)


def get_extractor(name: str, config: Any = None, **kwargs: Any) -> "FeatureExtractor":
    """Instantiate a registered extractor, defaulting its params from ``config``."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown extractor {name!r}; available: {available_extractors()}")
    params: dict[str, Any] = {}
    if config is not None:
        params.update(config.feature_params(name))
    params.update(kwargs)
    return _REGISTRY[name](**params)


class FeatureExtractor(ABC):
    """Base class: ``transform`` must be deterministic and NaN/inf-free."""

    name: str = "base"
    version: int = 1

    @property
    @abstractmethod
    def feature_names(self) -> Sequence[str]:
        """Stable, versioned column names (last axis of the output)."""

    @abstractmethod
    def transform_one(self, flow: Flow) -> np.ndarray:
        """Features for a single flow."""

    def transform(self, flows: Sequence[Flow]) -> np.ndarray:
        if len(flows) == 0:
            return np.zeros((0, len(self.feature_names)), dtype=np.float32)
        out = np.stack([self.transform_one(f) for f in flows]).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(v{self.version}, {len(self.feature_names)} features)"
