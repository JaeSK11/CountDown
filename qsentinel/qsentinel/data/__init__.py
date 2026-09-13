"""Dataset loaders.  Importing this module registers every loader by name."""

from qsentinel.data.base import (
    DatasetLoader,
    Item,
    available_datasets,
    get_loader,
    load,
    register_loader,
)
from qsentinel.data.cstnet import CSTNETLoader
from qsentinel.data.iscxtor import ISCXTorLoader
from qsentinel.data.iscxvpn import ISCXVPNLoader
from qsentinel.data.mobileapp import MobileAppLoader
from qsentinel.data.pooled import available_pooled, load_alias, load_pooled
from qsentinel.data.postquantumtls import PostQuantumTLSLoader

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
