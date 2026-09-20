"""Ensemble members.  Importing this module registers every model by name."""

from countdown.models.base import (
    BaseModel,
    ModelRegistry,
    load_model,
    register,
)
from countdown.models.flow_catboost import FlowStatsCatBoost
from countdown.models.flow_gbdt import FlowStatsGBDT
from countdown.models.flow_image_cnn import FlowImageCNN, FlowPicCNN
from countdown.models.flow_image_resnet import FlowImageResNet18
from countdown.models.byte_net import ByteNetRecommended, ETBertBaseline
from countdown.models.graph_gnn import GINGraphMember, TFEGNNBaseline
from countdown.models.paper_baselines import PaperC45, PaperKNN
from countdown.models.seq_cnn import DFBaseline, DilatedResSeqCNN

__all__ = [
    "BaseModel",
    "ByteNetRecommended",
    "DFBaseline",
    "DilatedResSeqCNN",
    "FlowImageCNN",
    "FlowImageResNet18",
    "FlowPicCNN",
    "FlowStatsCatBoost",
    "FlowStatsGBDT",
    "GINGraphMember",
    "PaperC45",
    "PaperKNN",
    "TFEGNNBaseline",
    "ModelRegistry",
    "load_model",
    "register",
]
