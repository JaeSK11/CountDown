"""One aligned row set per expert, and a *view* per member that can fit and predict any subset.

Stacking needs every member's prediction for the **same** rows, but the Phase-4 members do
not share a sample unit: ``flow_gbdt`` and ``seq_cnn`` see one flow per row, the image member
sees time *windows*, the graph member reads its own byte cache.  Phase 5 fixes the unit at
the **flow** -- what the Phase-6 pipeline will be asked to classify -- and gives each member
a view that maps flow rows to its own input:

``ArrayView``   one input row per flow (flow_stats, packet_seq, the graph byte matrix).
``WindowView``  several windows per flow, cut on the flow's **own** timeline
                (``per_flow=True``); fit on every window of the fit rows, predict by
                averaging a flow's window probabilities.  This is the deployed unit, not the
                capture-merged windows the image scorecard used, so its single-member number
                here is new, not a re-run of ``MODEL-image.md``.

Rows a member cannot score are the expert's problem to avoid: E2 keeps only the CSTNET
flows that have a graph sample, so every member covers every row.

Not stackable yet, recorded rather than hidden: ``byte_net`` scores ET-BERT's released
corpus, whose samples carry no flow identity to join on; it joins E2 once byte extraction
from our own pcaps exists (``PHASE-4.md`` task 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.models.base import BaseModel, ModelRegistry
from countdown.schema import LabelSpace

log = get_logger(__name__)


#: Per-flow input overrides.  The image member's 224-px default is ~600 KB per window; per
#: flow on ISCX that is >100 GB, and a FlowPic of a 9-packet flow carries no detail a
#: 64-px histogram loses (64 is the smallest the network's pooling stack accepts).  One
#: 60 s window: the median ISCX flow is far shorter than 15 s, so 15/30/60 s windows would
#: be three copies of one image.  Larger batches / fewer epochs because there are ~10^5
#: images here, not the ~10^3 the scorecard trained on.  E3 keeps the scorecard's
#: construction (MobileApp is one flow per capture, so per-flow = per-capture windows).
PER_FLOW_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "E1": {"flow_image_cnn": {"feature_params": {"size": 64}, "windows": [60.0],
                              "params": {"batch_size": 256, "epochs": 15}}},
    "E2": {"graph_gnn": {"params": {"epochs": 20, "workers": 24}}},
    "E3": {},
}


class MemberView:
    """Fits a fresh member on rows and predicts rows.  Rows index the expert's flow table."""

    def __init__(self, name: str, model: str, space: LabelSpace, y: np.ndarray,
                 groups: np.ndarray, params: dict[str, Any] | None = None,
                 val_size: float = 0.0, seed: int = 42) -> None:
        self.name = name
        self.model = model
        self.space = space
        self.y = y
        self.groups = groups
        self.params = dict(params or {})
        #: Grouped share of the fit rows held out for the member's own early stopping
        #: (the GBDT); 0 for the deep members, which report their final epoch (D3).
        self.val_size = float(val_size)
        self.seed = seed

    def new_model(self) -> BaseModel:
        return ModelRegistry.create(self.model, label_space=self.space,
                                    **{"seed": self.seed, **self.params})

    def _split_val(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.val_size <= 0:
            return rows, rows[:0]
        from countdown.eval.splits import group_split

        try:
            a, b = group_split(None, self.y[rows], self.groups[rows],
                               test_size=self.val_size, seed=self.seed + 1)
        except ValueError:
            return rows, rows[:0]
        return rows[a], rows[b]

    def fit(self, rows: np.ndarray) -> BaseModel:
        raise NotImplementedError

    def predict(self, model: BaseModel, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ArrayView(MemberView):
    def __init__(self, X: np.ndarray, **kw: Any) -> None:
        super().__init__(**kw)
        self.X = X

    def fit(self, rows: np.ndarray) -> BaseModel:
        tr, va = self._split_val(np.asarray(rows))
        m = self.new_model()
        m.fit(self.X[tr], self.y[tr], val=(self.X[va], self.y[va]) if len(va) else None)
        return m

    def predict(self, model: BaseModel, rows: np.ndarray) -> np.ndarray:
        return model.predict_proba(self.X[np.asarray(rows)])


class WindowView(MemberView):
    def __init__(self, Xw: np.ndarray, parent: np.ndarray, **kw: Any) -> None:
        super().__init__(**kw)
        self.Xw = Xw
        self.parent = np.asarray(parent, dtype=np.int64)     # window -> flow row

    def _windows(self, rows: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(self.parent, rows))

    def fit(self, rows: np.ndarray) -> BaseModel:
        tr, va = self._split_val(np.asarray(rows))
        wtr, wva = self._windows(tr), self._windows(va)
        m = self.new_model()
        m.fit(self.Xw[wtr], self.y[self.parent[wtr]],
              val=(self.Xw[wva], self.y[self.parent[wva]]) if len(wva) else None)
        return m

    def predict(self, model: BaseModel, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        w = self._windows(rows)
        pw = model.predict_proba(self.Xw[w])
        pos = {int(r): i for i, r in enumerate(rows)}
        idx = np.array([pos[int(p)] for p in self.parent[w]], dtype=np.int64)
        out = np.zeros((len(rows), pw.shape[1]))
        np.add.at(out, idx, pw)
        cnt = np.bincount(idx, minlength=len(rows)).astype(np.float64)
        if (cnt == 0).any():
            # A flow with no window: no opinion, which the combiner reads as missing.
            out[cnt == 0] = np.nan
        cnt[cnt == 0] = 1.0
        return out / cnt[:, None]


@dataclass
class ExpertData:
    """An expert's flow table: labels, groups, Stage-1 hints, the outer split, member views."""

    expert: str
    dataset: str
    space: LabelSpace
    y: np.ndarray
    groups: np.ndarray
    group_key: str
    keys: np.ndarray
    l7_hint: np.ndarray
    train_rows: np.ndarray
    test_rows: np.ndarray
    views: dict[str, MemberView] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "expert": self.expert, "dataset": self.dataset, "target": self.space.target,
            "n_classes": len(self.space), "n_rows": int(len(self.y)),
            "n_train": int(len(self.train_rows)), "n_test": int(len(self.test_rows)),
            "group_key": self.group_key,
            "n_groups_train": len(set(self.groups[self.train_rows].tolist())),
            "n_groups_test": len(set(self.groups[self.test_rows].tolist())),
            "members": {n: {"model": v.model, "view": type(v).__name__, "params": v.params}
                        for n, v in self.views.items()},
            "skipped": self.skipped,
        }


def _cache_key(source_file: Any) -> str:
    return "/".join(Path(str(source_file)).parts[-2:])


def build_expert_data(expert: Any, members: list[str] | None = None, seed: int = 42,
                      graph_cache: str | Path = "cache/tfegnn/cstnet",
                      overrides: dict[str, dict[str, Any]] | None = None,
                      max_flows: int | None = None, outer_fold: int = 0,
                      all_rows: bool = False) -> ExpertData:
    """Load the expert's dataset, align every member to one flow table, draw the outer split.

    ``overrides`` maps a member name to ``{"params": ..., "feature_params": ...,
    "windows": ...}`` replacements -- the image member's 224-px default is far too large to
    materialise per flow, for instance.  ``max_flows`` subsamples whole groups (tests).
    ``outer_fold`` picks which fold of the outer split is test (``group_split(fold=)``).
    ``all_rows`` puts every row in train and none in test: the refit-on-all release
    (``models-v2``), whose only score is out-of-fold.
    """
    from countdown.config import default_config
    from countdown.data import load
    from countdown.eval.splits import group_split

    config = default_config()
    overrides = overrides or {}
    ds = load(expert.dataset, target=expert.target, config=config)
    space = LabelSpace.from_dataset(ds, expert.target)
    group_key = config.group_key(expert.dataset)
    flows = list(ds.flows)
    skipped: dict[str, str] = {}
    specs = [m for m in expert.members if members is None or m.model in members]

    graph_X = None
    if any(m.model.startswith("graph_gnn") for m in specs):
        graph_X, keep = _graph_rows(flows, Path(graph_cache))
        log.info("[views] %s: %d/%d flows have a graph sample", expert.name, len(keep), len(flows))
        flows = [flows[i] for i in keep]
    if max_flows is not None and len(flows) > max_flows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(flows), max_flows, replace=False))
        flows = [flows[i] for i in idx]
        graph_X = None if graph_X is None else graph_X[idx]

    from countdown.schema import FlowDataset

    fds = FlowDataset(flows, name=ds.name, default_target=ds.default_target,
                      config=ds.config, invalid_targets=ds.invalid_targets)
    y = fds.labels(expert.target, space=space)
    groups = np.asarray(fds.groups(group_key), dtype=object)
    keys = np.array([f.flow_id for f in flows], dtype=object)
    hints = np.array([f.l7_hint for f in flows], dtype=object)
    if all_rows:
        tr, te = np.arange(len(y)), np.array([], dtype=np.int64)
    else:
        tr, te = group_split(None, y, groups, test_size=expert.test_size, seed=seed,
                             allow_vacuous=config.allows_vacuous_groups(expert.dataset),
                             fold=outer_fold)
    data = ExpertData(expert=expert.name, dataset=ds.name, space=space, y=y, groups=groups,
                      group_key=group_key, keys=keys, l7_hint=hints, train_rows=tr,
                      test_rows=te, skipped=skipped)

    for spec in specs:
        ov = overrides.get(spec.model, {})
        params = {**spec.params, **ov.get("params", {})}
        fparams = {**spec.feature_params, **ov.get("feature_params", {})}
        windows = ov.get("windows", spec.windows)
        common = dict(name=spec.model, model=spec.model, space=space, y=y, groups=groups,
                      params=params, seed=seed,
                      val_size=0.1 if spec.model == "flow_gbdt" else 0.0)
        if spec.model.startswith("graph_gnn"):
            common["params"] = {"header_len": 40, **params}
            data.views[spec.model] = ArrayView(X=graph_X, **common)
        elif spec.driver != "harness":
            skipped[spec.model] = (f"no flow-aligned input: trained by `{spec.script}` on a "
                                   f"corpus without flow identity")
        else:
            X, parent = member_input(fds, spec.features, fparams, windows)
            if parent is None:
                data.views[spec.model] = ArrayView(X=X, **common)
            else:
                log.info("[views] %s: %d windows over %d flows, X=%s", spec.model, len(parent),
                         len(np.unique(parent)), X.shape)
                data.views[spec.model] = WindowView(Xw=X, parent=parent, **common)
    for name, why in skipped.items():
        log.warning("[views] %s: skipping %s -- %s", expert.name, name, why)
    return data


def member_input(fds: Any, features: str, feature_params: dict[str, Any],
                 windows: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
    """A member's input for every flow of ``fds``: ``(X, None)`` for one row per flow, or
    ``(X_windows, parent_row)`` when the member reads per-flow windows.

    Shared by training (``build_expert_data``) and inference (``pipeline``), so a deployed
    member sees its input built exactly the way it was trained on.
    """
    from countdown.schema import FlowDataset

    if not windows:
        return fds.features(features, **feature_params), None
    from countdown.data.windows import windowed_dataset

    # Windowing keys on flow_id, which is not unique (14 ISCX flows share one with another
    # flow), so window row-tagged copies and map parents back by row.
    tagged = FlowDataset([replace(f, flow_id=f"row{i}") for i, f in enumerate(fds.flows)],
                         name=fds.name, default_target=fds.default_target,
                         config=fds.config, invalid_targets=fds.invalid_targets)
    wds = windowed_dataset(tagged, windows=list(windows), per_flow=True)
    parent = np.array([int(w.meta["parent"][3:]) for w in wds.flows], dtype=np.int64)
    return wds.features(features, **feature_params), parent


def _graph_rows(flows: list, root: Path) -> tuple[np.ndarray, np.ndarray]:
    """The byte-matrix row for each flow (joined on ``app/file``) and the flows that have one."""
    import glob

    files = sorted(f for f in glob.glob(str(root / "*.npz")) if not f.endswith(".tmp.npz"))
    if not files:
        raise FileNotFoundError(f"no graph cache shards in {root} -- run build_tfegnn_cache.py")
    H, P, S = [], [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            H.append(d["header"]); P.append(d["payload"]); S.append(d["source_file"])
    X = np.concatenate([np.concatenate(H), np.concatenate(P)], axis=2).astype(np.int16)
    pos: dict[str, int] = {}
    for i, sf in enumerate(np.concatenate(S)):
        pos.setdefault(_cache_key(sf), i)       # one sample per CSTNET pcap; keep the first
    keep, rows = [], []
    for j, fl in enumerate(flows):
        i = pos.get(_cache_key(fl.meta.get("source_file")))
        if i is not None:
            keep.append(j); rows.append(i)
    return X[np.asarray(rows)], np.asarray(keep, dtype=np.int64)
