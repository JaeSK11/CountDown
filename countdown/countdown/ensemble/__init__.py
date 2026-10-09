"""Phase 5: calibration, out-of-fold stacking, the router, and the ``Ensemble`` object."""

from countdown.ensemble.calibration import (
    CalibratedModel,
    ProbaCalibrator,
    check_convention,
    ece,
    reliability,
)
from countdown.ensemble.combiner import SoftVote, Stacker, make_combiner
from countdown.ensemble.ensemble import Ensemble, MemberPreds, member_predictions
from countdown.ensemble.router import Router
from countdown.ensemble.stacking import OOF, group_folds, oof_predict
from countdown.ensemble.views import ArrayView, ExpertData, WindowView, build_expert_data

__all__ = [
    "ArrayView", "CalibratedModel", "Ensemble", "ExpertData", "MemberPreds", "OOF",
    "ProbaCalibrator", "Router", "SoftVote", "Stacker", "WindowView", "build_expert_data",
    "check_convention", "ece", "group_folds", "make_combiner", "member_predictions",
    "oof_predict", "reliability",
]
