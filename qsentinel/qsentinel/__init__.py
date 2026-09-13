"""QSentinel -- encrypted-traffic crypto detection, PQC verdict, and ML classification."""

from qsentinel.config import Config, default_config, get_logger, setup_logging
from qsentinel.schema import (
    GROUP_KEYS,
    TRAFFIC_TYPES,
    TUNNEL_TYPES,
    FiveTuple,
    Flow,
    FlowDataset,
    LabelSpace,
    Packet,
    TrafficType,
    TunnelType,
)

__version__ = "0.1.0"

__all__ = [
    "Config",
    "FiveTuple",
    "Flow",
    "FlowDataset",
    "GROUP_KEYS",
    "LabelSpace",
    "Packet",
    "TRAFFIC_TYPES",
    "TUNNEL_TYPES",
    "TrafficType",
    "TunnelType",
    "default_config",
    "get_logger",
    "setup_logging",
    "__version__",
]


def load(name: str, target: str | None = None, **kwargs):
    """Load a dataset by name -- the Phase 0 entry point.

    >>> import qsentinel
    >>> ds = qsentinel.load("iscxvpn", target="traffic_type")
    """
    from qsentinel.data import load as _load

    return _load(name, target=target, **kwargs)
