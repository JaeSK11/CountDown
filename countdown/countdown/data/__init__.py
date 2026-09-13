"""Dataset loaders.  Importing this module registers every loader by name."""

from countdown.data.base import (
    DatasetLoader,
    Item,
    available_datasets,
    get_loader,
    load,
    register_loader,
)
from countdown.data.cstnet import CSTNETLoader
from countdown.data.iscxtor import ISCXTorLoader
from countdown.data.iscxvpn import ISCXVPNLoader
from countdown.data.mobileapp import MobileAppLoader
from countdown.data.pooled import available_pooled, load_alias, load_pooled
from countdown.data.postquantumtls import PostQuantumTLSLoader

__all__ = [
    "available_pooled",
    "load_alias",
    "load_pooled",
    "CSTNETLoader",
    "DatasetLoader",
    "ISCXTorLoader",
    "ISCXVPNLoader",
    "Item",
    "MobileAppLoader",
    "PostQuantumTLSLoader",
    "available_datasets",
    "get_loader",
    "load",
    "register_loader",
]
