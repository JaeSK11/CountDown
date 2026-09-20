"""The three expert bindings (decision D2)."""

from __future__ import annotations

import pytest

from countdown.experts import EXPERTS, get_expert
from countdown.models.base import ModelRegistry


def test_three_experts_with_consistent_member_contracts():
    assert sorted(EXPERTS) == ["E1", "E2", "E3"]
    for e in EXPERTS.values():
        e.check()
        assert e.default_member.model == "flow_gbdt"          # the universal floor
        for m in e.members:
            rec = getattr(ModelRegistry.get(m.model), "recommended_feature_params", None)
            if rec and m.driver == "harness":
                assert {k: m.feature_params.get(k) for k in rec} == rec


def test_configs_are_built_for_harness_members_only():
    e1 = get_expert("E1")
    cfg = e1.config("flow_image_cnn")
    assert (cfg.dataset, cfg.target, cfg.features) == ("iscx_pooled", "traffic_type", "flow_image")
    assert cfg.windows == [15.0, 30.0, 60.0] and cfg.feature_params["channels"] == 3
    assert get_expert("E3").config().test_size == 0.25
    with pytest.raises(ValueError, match="trained by a script"):
        get_expert("E2").config("byte_net")
    with pytest.raises(KeyError):
        get_expert("E9")


@pytest.mark.slow
def test_activity_expert_trains_its_default_member():
    out = get_expert("E3").train(write=False)
    assert out["metrics"]["macro_f1"] > 0.2
