"""Ensemble members.  Importing this module registers every model by name."""

from countdown.models.base import (
    BaseModel,
    ModelRegistry,
    load_model,
    register,
)
from countdown.models.flow_gbdt import FlowStatsGBDT
from countdown.models.flow_image_cnn import FlowImageCNN
from countdown.models.flow_image_resnet import FlowImageResNet18
from countdown.models.byte_net import ETBertBaseline
from countdown.models.paper_baselines import PaperC45, PaperKNN

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
