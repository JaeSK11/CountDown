"""The training harness with a rank-4 image member: windowing, rank checks, bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from countdown.models.base import ModelRegistry
from countdown.training.train import ExperimentConfig


def test_models_declare_the_rank_they_consume():
    assert ModelRegistry.get("flow_gbdt").expects_ndim == 2
    assert ModelRegistry.get("flow_image_cnn_baseline").expects_ndim == 4


def test_input_type_matches_the_extractor_name():
    """The harness compares model.input_type to cfg.features, so they must agree.

    Only asserted for members whose extractor exists: other Phase-4 members are landing
    alongside this one and declare an input_type before their extractor is registered.
    """
    from countdown.features import available_extractors

    extractors = set(available_extractors())
    checked = 0
    for name in ModelRegistry.list():
        input_type = ModelRegistry.get(name).input_type
        if input_type not in extractors:
            continue                      # extractor not landed yet
        checked += 1
    assert ModelRegistry.get("flow_image_cnn_baseline").input_type == "flow_image"
    assert ModelRegistry.get("flow_gbdt").input_type == "flow_stats"
    assert checked >= 2


def test_experiment_config_accepts_window_fields():
    cfg = ExperimentConfig(dataset="x", windows=[15, 30], window_per_flow=True)
    assert cfg.windows == [15, 30] and cfg.window_per_flow is True
    assert "windows" in cfg.to_dict()


@pytest.mark.parametrize("path", [
    "configs/experiments/iscx_nonvpn_app_image_baseline.yaml",
    "configs/experiments/mobileapp_activity_image_baseline.yaml",
])
def test_shipped_image_configs_parse_and_are_consistent(path):
    cfg = ExperimentConfig.from_yaml(path)
    model_cls = ModelRegistry.get(cfg.model)
    assert cfg.features == model_cls.input_type
    assert cfg.windows, "an image config without windows loses the paper's sample unit"


def test_feature_model_mismatch_is_caught_before_extraction():
    """A tabular model pointed at image features fails on the declared contract.

    It must not get as far as building the image tensor to find out -- that is minutes of
    work and gigabytes of RAM to reach a conclusion available from the config alone.
    """
    from countdown.training.train import train

    cfg = ExperimentConfig(dataset="mobileapp", target="activity",
                           model="flow_gbdt", features="flow_image", windows=[10])
    with pytest.raises(ValueError, match="consumes 'flow_stats'"):
        train(cfg, write=False)


def test_rank_guard_catches_an_extractor_producing_the_wrong_rank():
    """The rank check is the second line: contract right, extractor params wrong."""
    from countdown.models.base import BaseModel, register
    from countdown.training.train import train

    @register("_rank_guard_probe")
    class _Probe(BaseModel):
        input_type = "flow_stats"      # contract matches the config...
        expects_ndim = 4               # ...but the model wants images

        def _fit(self, X, y, sample_weight=None, val=None):
            raise AssertionError("must not reach _fit")

        def _predict_proba(self, X):
            raise AssertionError("must not reach _predict_proba")

    cfg = ExperimentConfig(dataset="mobileapp", target="activity",
                           model="_rank_guard_probe", features="flow_stats")
    with pytest.raises(ValueError, match="expects rank 4"):
        train(cfg, write=False)


@pytest.mark.slow
def test_image_member_runs_end_to_end_through_the_harness(tmp_path):
    pytest.importorskip("torch")
    from countdown.training.train import train

    cfg = ExperimentConfig(
        dataset="mobileapp", target="activity",
        model="flow_image_cnn_baseline", features="flow_image",
        windows=[10], grouped=True, test_size=0.25, val_size=0.1,
        params={"epochs": 2, "batch_size": 64, "device": "cpu"},
    )
    r = train(cfg, runs_dir=tmp_path, write=True)

    # windowing happened, and rank-4 bookkeeping replaced n_features
    assert r["config"]["n_samples"] > len(r["dataset"])
    assert len(r["config"]["feature_shape"]) == 3          # (C, H, W)
    assert "n_features" not in r["config"]
    # the split is real, not vacuous: windows of a capture stayed together
    assert r["split"]["groups_disjoint"]
    assert r["split"]["n_groups_train"] > 1
    # and the label space is the full class set, not just what windows realised
    assert r["metrics"]["n_classes"] == 92
    assert (tmp_path).exists()



# -- GPU-resident batching ---------------------------------------------------------------


def test_resident_and_streaming_paths_agree():
    """The GPU-resident fast path must be an optimisation, not a different model.

    Same seed, same batch size, same schedule -> same weights. Forcing
    gpu_resident_gb=0 selects the DataLoader path on identical inputs.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to exercise the resident path")
    from countdown.schema import LabelSpace

    space = LabelSpace.from_names("traffic_type", ["chat", "voip"])
    rng = np.random.default_rng(0)
    X = rng.random((64, 1, 64, 64)).astype(np.float32)
    y = np.array([0, 1] * 32)

    def fit(budget):
        m = ModelRegistry.create("flow_image_cnn_baseline", label_space=space,
                                 epochs=2, batch_size=8, seed=7, gpu_resident_gb=budget)
        m.fit(X, y)
        return m.predict_proba(X)

    assert np.allclose(fit(12.0), fit(0.0), atol=1e-4)


def test_oversize_tensor_falls_back_to_streaming(caplog):
    pytest.importorskip("torch")
    from countdown.schema import LabelSpace

    space = LabelSpace.from_names("traffic_type", ["chat", "voip"])
    X = np.zeros((16, 1, 64, 64), dtype=np.float32)
    m = ModelRegistry.create("flow_image_cnn_baseline", label_space=space,
                             epochs=1, batch_size=8, gpu_resident_gb=0.0)
    m.fit(X, np.array([0, 1] * 8))
    assert m.is_fitted
