"""Training orchestration: config in, reproducible run directory out.

One code path for every member and every target, so that "same scaffolding for each model"
is structural rather than a convention people remember.  Phase 4 adds model *types* behind
the same function.

The flow, and why it is in this order:

1. Load the dataset (or pooled alias) from the Phase-0 cache.
2. Bind a :class:`~countdown.schema.LabelSpace` -- before any splitting, because it fixes
   what the class ids mean, and before any windowing, so the columns do not move when the
   window configuration changes.
3. Optionally cut flows into fixed-duration windows (``windows:``), the sample unit the
   image and sequence members use.  Windows inherit their capture's ``source_file``, so
   step 4 still keeps every window of one capture on one side.
4. Split **group-aware by the dataset's declared key**, val fold carved from train so
   early stopping never peeks at test.
5. Fit, evaluate against a majority baseline, and persist everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from countdown.config import Config, default_config, get_logger
from countdown.eval.metrics import evaluate, metrics_excluding_tail
from countdown.eval.report import summarize, write_run
from countdown.eval.splits import group_split_three, split_report, stratified_split
from countdown.models.base import ModelRegistry
from countdown.schema import LabelSpace

log = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Everything that defines a run.  Frozen into the run directory verbatim."""

    dataset: str
    target: str | None = None
    model: str = "flow_gbdt"
    features: str = "flow_stats"
    #: Per-experiment overrides of the extractor's ``configs/features.yaml`` parameters,
    #: e.g. ``{construction: flowpic, channels: 3}`` for ``flow_image`` or ``{n: 256}`` for
    #: ``packet_seq``.  Without this a recommended member could only differ from its
    #: baseline's representation by editing the global feature config.
    feature_params: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    group_key: str | None = None          # None -> the dataset's declared key
    grouped: bool = True                  # False -> stratified (optimistic), for quick looks
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int | None = None               # None -> config seed
    min_samples_per_class: int = 0
    top_k_classes: int | None = None
    #: Cut each capture's timeline into fixed-duration windows before extracting features.
    #: ``[15, 30, 60]`` pools three window sizes, which is what Okonkwo used as both
    #: augmentation and a countermeasure to arrival-time disparity.  None -> whole flows.
    windows: list[float] | None = None
    #: Window each flow's own timeline instead of merging a capture's flows onto one clock.
    #: False reproduces the paper; True is what a deployed per-flow classifier would see.
    window_per_flow: bool = False
    tail_min_support: int = 50
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise KeyError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def train(
    cfg: ExperimentConfig,
    config: Config | None = None,
    runs_dir: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run one experiment.  Returns ``{metrics, split, model, run_dir, ...}``."""
    from countdown.data import load

    config = config or default_config()
    seed = cfg.seed if cfg.seed is not None else config.seed

    # -- 1. data -----------------------------------------------------------------------
    ds = load(cfg.dataset, target=cfg.target, config=config)
    target = cfg.target or ds.default_target
    if cfg.min_samples_per_class or cfg.top_k_classes:
        before = len(ds)
        ds = ds.filter(
            target=target,
            min_samples_per_class=cfg.min_samples_per_class,
            top_k_classes=cfg.top_k_classes,
        )
        log.info("[train] filters kept %d/%d flows", len(ds), before)
    if len(ds) == 0:
        raise ValueError(f"dataset {cfg.dataset!r} is empty after filtering")

    # -- 2. label space ----------------------------------------------------------------
    # Bound to the *unwindowed* dataset on purpose: the class set is a property of the
    # data, not of the window size, so `predict_proba` columns stay comparable across
    # window configurations (and across members) for the Phase-5 stacker.  A class whose
    # every window turns out blank keeps a zero column rather than shifting the rest.
    space = LabelSpace.from_dataset(ds, target)

    # -- 3. windowing ------------------------------------------------------------------
    feat_ds = ds
    if cfg.windows:
        from countdown.data.windows import windowed_dataset

        feat_ds = windowed_dataset(
            ds, windows=cfg.windows, per_flow=cfg.window_per_flow
        )
        if len(feat_ds) == 0:
            raise ValueError(
                f"windowing {cfg.dataset!r} at {cfg.windows} produced no windows"
            )
        log.info("[train] windows %s -> %d samples from %d flows",
                 cfg.windows, len(feat_ds), len(ds))

    # -- 4. features -------------------------------------------------------------------
    # The declared contract is checked first, before anything is extracted: it is the
    # precise diagnosis (the config names the wrong representation) and it costs nothing,
    # where the rank check below would first materialise a multi-GB image tensor to reach
    # the same conclusion less clearly.
    model_cls = ModelRegistry.get(cfg.model)
    if model_cls.input_type != cfg.features:
        raise ValueError(
            f"model {cfg.model!r} consumes {model_cls.input_type!r} but the config asks "
            f"for features={cfg.features!r}"
        )
    X = feat_ds.features(cfg.features, **cfg.feature_params)
    # Rank is then an internal-consistency assertion: the extractor is the one named in
    # the contract, so a surprise here means the extractor's own params are wrong (e.g.
    # packet_seq channels=1 gives rank 2, channels=2 gives rank 3).
    if X.ndim != model_cls.expects_ndim:
        raise ValueError(
            f"feature set {cfg.features!r} produced shape {X.shape} (rank {X.ndim}), but "
            f"model {cfg.model!r} expects rank {model_cls.expects_ndim}. Check the "
            f"{cfg.features!r} parameters in configs/features.yaml."
        )
    y = feat_ds.labels(target, space=space)
    feature_names = feat_ds.feature_names(cfg.features, **cfg.feature_params)
    log.info(
        "[train] %s: X=%s, %d classes in space (%d realised)",
        feat_ds.name, X.shape, len(space), len(np.unique(y)),
    )

    # -- 5. split ----------------------------------------------------------------------
    # Windowing can *fix* a vacuous grouping: MobileApp is one CSV per sample, so
    # source_file groups nothing, but windowing that CSV yields many samples sharing it.
    group_key = cfg.group_key or config.group_key(cfg.dataset)
    if cfg.grouped:
        groups = feat_ds.groups(group_key)
        parts = group_split_three(
            X, y, groups,
            test_size=cfg.test_size, val_size=cfg.val_size, seed=seed,
            allow_vacuous=(
                config.allows_vacuous_groups(cfg.dataset) and not cfg.windows
            ),
        )
        tr, va, te = parts["train"], parts["val"], parts["test"]
    else:
        groups = np.arange(len(y), dtype=object)
        tr, te = stratified_split(X, y, test_size=cfg.test_size, seed=seed)
        va = np.array([], dtype=np.int64)

    audit = split_report(y, groups, tr, te, len(space), space.names)
    audit["group_key"] = group_key if cfg.grouped else "(stratified, OPTIMISTIC)"
    audit["n_val"] = int(len(va))
    if audit["classes_absent_from_test"]:
        log.warning(
            "[train] %d class(es) absent from the test fold: %s",
            len(audit["classes_absent_from_test"]), audit["classes_absent_from_test"],
        )

    # -- 6. fit ------------------------------------------------------------------------
    # seed comes from the resolved config unless the experiment pins its own, so that
    # `params: {seed: ...}` does not collide with the harness-level seed.
    model_params = {"seed": seed, **cfg.params}
    model = ModelRegistry.create(cfg.model, label_space=space, **model_params)
    model._check_target(target)   # input_type/rank already checked in step 4
    model.feature_names = list(feature_names)
    model.fit(
        X[tr], y[tr],
        val=(X[va], y[va]) if len(va) else None,
        feature_names=feature_names,
    )

    # -- 7. evaluate + persist ---------------------------------------------------------
    metrics = evaluate(model, X[te], y[te], space=space, y_train=y[tr], seed=seed)
    metrics["tail_excluded"] = metrics_excluding_tail(metrics, cfg.tail_min_support)
    if hasattr(model, "feature_importance"):
        metrics["top_features"] = [
            {"feature": f, "gain": float(g)} for f, g in model.feature_importance(top_k=20)
        ]

    frozen = cfg.to_dict()
    frozen.update({
        "dataset_resolved": ds.name, "target_resolved": target,
        "seed_resolved": seed, "group_key_resolved": group_key,
        "n_flows": len(ds), "n_samples": int(X.shape[0]),
        # rank-agnostic: (F,) for a table, (C, H, W) for an image
        "feature_shape": [int(d) for d in X.shape[1:]],
        "label_space": space.to_dict(),
    })

    run_dir = None
    if write:
        run_dir = write_run(
            metrics, frozen, model=model, split=audit, runs_dir=runs_dir,
            extra={"split": audit},
        )

    print(f"\n{cfg.model} / {ds.name} / {target}")
    print(summarize(metrics, audit))
    if run_dir:
        print(f"  run           {run_dir}")
    return {
        "metrics": metrics, "split": audit, "model": model,
        "run_dir": run_dir, "config": frozen, "dataset": ds, "space": space,
    }
