"""Ensemble members.  Importing this module registers every model by name."""

from qsentinel.models.base import (
    BaseModel,
    ModelRegistry,
    load_model,
    register,
)
from qsentinel.models.flow_gbdt import FlowStatsGBDT
from qsentinel.models.flow_image_cnn import FlowImageCNN
from qsentinel.models.flow_image_resnet import FlowImageResNet18
from qsentinel.models.byte_net import ETBertBaseline
from qsentinel.models.paper_baselines import PaperC45, PaperKNN

__all__ = [
    "BaseModel",
    "FlowImageCNN",
    "FlowImageResNet18",
    "FlowStatsGBDT",
    "PaperC45",
    "PaperKNN",
    "ModelRegistry",
    "load_model",
    "register",
]
