# Phase 3 — ML Ensemble Foundation (Pipeline Stage 4, part 1)

**Roadmap ref:** `../PLAN.md` → Phase 3. **Depends on:** Phase 0 (`FlowDataset`, `flow_stats`
features, label taxonomy). **★ First ensemble member + shared eval infra.**
**Goal:** stand up the `BaseModel` contract + registry that every future ensemble member implements,
build the shared **train/split/evaluate** harness, and land the **first working model** —
**LightGBM on flow-stats** — trained on `traffic_type` (ISCXVPN/Tor) with honest, leakage-free
metrics.

**One-line exit criterion:**
`scripts/train.py --model flow_gbdt --dataset iscx_pooled --target traffic_type` trains, evaluates
with a **group-aware split (no capture in both train and test)**, beats the majority-class baseline
on macro-F1, and writes a reproducible run to `runs/<id>/` (metrics.json + model + confusion matrix).

> **Amended (A4).** The criterion runs on the **pooled** ISCXVPN+ISCXTor dataset, not `iscxvpn`
> alone. `iscxvpn` realises 7 of 8 classes and its entire `p2p` class comes from **one capture**
> (`vpn_bittorrent.pcap`, 255 flows), so a group-aware split cannot put `p2p` on both sides — the
> criterion as originally written was unreachable and contradicted §6. `--dataset iscxvpn` stays
> valid as a **smoke test**, never as a reportable result. See §11.

---

## 1. Why the interface matters now (with one model)

Only one model exists in Phase 3, but the **`BaseModel` contract is the linchpin of the whole
ensemble** (Phases 4–5). Getting the seam right now — especially `predict_proba` with a **canonical
class ordering** — is what lets a LightGBM, a CNN, a transformer, and a GNN be stacked later without
rewrites. So Phase 3 = infrastructure first, one concrete model second.

**The seam:** every model outputs `predict_proba(X) -> (N, C)` where columns follow the **shared
`LabelSpace`** from the Phase 0 taxonomy. Aligned columns = Phase 5 can calibrate + stack any subset.

---

## 2. Scope

**In scope**
- **Phase-0 surface additions (task 3.0, specified in §11):** `LabelSpace`,
  `FlowDataset.groups()` / `.num_classes`, `data/pooled.py`, and `group_key` + the `iscx_pooled`
  alias in `configs/datasets.yaml`. Phase 3 cannot be built without these; they are listed here so
  they are not mistaken for Phase-0 rework.
- `models/base.py` — `BaseModel` ABC + `ModelRegistry` (self-registration).
- `models/flow_gbdt.py` — LightGBM member (consumes `flow_stats`), class-imbalance-aware.
- `eval/` — `metrics.py`, `splits.py` (**group-aware, leakage-safe**), `report.py`.
- `training/train.py` + `scripts/train.py` CLI — load → split → fit → evaluate → persist.
- Experiment config + reproducible run directories; a majority/Dummy baseline for context.

**Out of scope (later)**
- Deep models (CNN/SSM/transformer/GNN) → Phase 4.
- Domain experts (per-protocol) → Phase 4.
- Calibration + stacking combiner + router → Phase 5 (but `predict_proba` seam is reserved now).
- Byte/image/graph features → Phase 4.

---

## 3. Dependencies
`lightgbm`, `scikit-learn` (splits/metrics/Dummy), `joblib` (persistence), `pandas`, optional
`matplotlib` (confusion-matrix PNG). No `torch` yet (Phase 4).

`joblib` is currently only a transitive dependency of scikit-learn and `matplotlib` is not
installed at all — task 3.0 adds `joblib` to `pyproject.toml` `dependencies` and `matplotlib` to a
new `[viz]` extra, so `confusion.png` degrades to a skipped artifact rather than an ImportError.

---

## 4. Directory additions

```
qsentinel/
├── schema.py              # MODIFIED (3.0): + LabelSpace, FlowDataset.groups()/.num_classes
├── data/
│   └── pooled.py          # NEW (3.0): load_pooled([...]) -> one FlowDataset
├── models/
│   ├── base.py            # BaseModel ABC + ModelRegistry + @register
│   └── flow_gbdt.py       # LightGBM on flow_stats
├── eval/
│   ├── splits.py          # group_split + StratifiedGroupKFold (by the dataset's group_key)
│   ├── metrics.py         # accuracy, macro/weighted F1, per-class P/R, confusion
│   └── report.py          # write runs/<id>/metrics.json (+ confusion.png)
├── training/
│   └── train.py           # orchestration
├── configs/
│   ├── datasets.yaml      # MODIFIED (3.0): + group_key per dataset, + iscx_pooled alias
│   └── experiments/
│       ├── iscx_pooled_traffic_type_gbdt.yaml     # the exit-criterion run
│       ├── iscx_pooled_tunnel_type_gbdt.yaml
│       ├── cstnet_app_gbdt.yaml
│       └── mobileapp_activity_gbdt.yaml
├── runs/                  # per-run artifacts (gitignored)
└── models_store/          # saved model artifacts
scripts/
└── train.py              # thin CLI wrapper
tests/
├── test_label_space.py    # taxonomy vs observed space; align/zero-fill; json round-trip
├── test_pooled.py         # pooled load: 8 classes, 235 groups, label fields reconcile
├── test_base_model.py
├── test_splits.py         # asserts NO group in both train & test; refuses vacuous grouping
├── test_metrics.py
└── test_flow_gbdt.py      # tiny synthetic (X,y): fit + predict_proba shape/sum-to-1
```

A run directory gains one file over the original spec: `label_space.json`, without which a saved
model's `predict_proba` columns cannot be interpreted at load time.

---

## 5. Components

### 5.1 `BaseModel` + registry (`models/base.py`)
```
class BaseModel(ABC):
    name: str                      # "flow_gbdt"
    input_type: str                # "flow_stats" | "packet_seq" | "bytes" | "image" | "graph"
    supports: set[str] | "*"       # targets it can train
    def fit(self, X, y, sample_weight=None, val=None): ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...      # (N, C) in LabelSpace order  ← ensemble seam
    @property
    def classes_(self) -> list: ...                    # canonical label ids
    def save(self, path): ...
    @classmethod
    def load(cls, path) -> "BaseModel": ...
    def get_params(self)/set_params(self,**kw): ...
```
- `@register("flow_gbdt")` decorator → `ModelRegistry.create(name, **params)`, `.list()`.
- **Canonical class ordering (`LabelSpace`) — resolved, A2.** Models bind to a `LabelSpace`, *not*
  to whatever classes a dataset happened to realise, so every `predict_proba` column means the same
  thing across members (required for Phase-5 stacking). Missing classes → zero columns.

  `LabelSpace` is a new frozen dataclass in `qsentinel/schema.py`, where the taxonomy already lives:
  `{target, names: tuple[str, ...]}` plus `index() / name() / encode() / decode() / align() /
  to_json() / from_json()`. Which constructor applies is a property of the **target**, not of the
  dataset that happens to be loaded:

  | Target kind | Space | Size |
  | --- | --- | --- |
  | taxonomy-backed — `traffic_type`, `tunnel_type` | `LabelSpace.taxonomy(target)` → the **full** `TRAFFIC_TYPES` / `TUNNEL_TYPES` | 8 / 3, dataset-independent |
  | open-ended — `app`, `activity` | `LabelSpace.from_dataset(ds, target)` → sorted observed classes, **persisted to `label_space.json`** beside the artifact | dataset-defined |

  **`FlowDataset.labels()` keeps its current behaviour.** It still returns compact dataset-local ids
  over the *observed* subset (`iscxvpn` → 7 names, `iscxtor` → 8) — that is what makes the ISCX
  column-meaning mismatch visible in the first place, and 331 existing tests pin it (e.g.
  `test_loaders.py::label_names() == ["browsing", "chat"]`). Widening is opt-in via a new keyword:

  ```python
  y_local     = ds.labels()                                    # unchanged: observed subset
  y_canonical = ds.labels(space=LabelSpace.taxonomy("traffic_type"))   # canonical ids, 8-wide
  ```

  So models train on canonical ids and always emit `(N, len(space))`. This buys the ensemble seam
  without a Phase-0 behaviour change.

**DoD:** registry creates a model by name; a stub model round-trips `save`/`load`; `predict_proba`
columns match `LabelSpace`.

### 5.2 `FlowStatsGBDT` (`models/flow_gbdt.py`)
- LightGBM multiclass on `flow_stats`. `input_type="flow_stats"`, `supports="*"`.
- **Imbalance handling:** `class_weight="balanced"` (or per-class `sample_weight`); report macro-F1
  as headline (not accuracy — ISCX categories are skewed).
- Early stopping on a validation fold; sane defaults (num_leaves, lr, n_estimators) in config;
  feature-name passthrough for importance.
- `save/load` via `joblib` (+ the `LabelSpace` and feature-name list).

**DoD:** trains on pooled-ISCX `traffic_type`; `predict_proba` sums to 1 and is 8 columns wide even
though no single constituent dataset realises all 8; feature importances available.

### 5.3 Splits (`eval/splits.py`) — leakage-safe
- **The trap:** ISCXVPN/Tor pcaps hold many correlated flows from the *same* session; a random
  flow-level split leaks near-duplicates into test → inflated scores (a common flaw in the
  literature).
- Provide `StratifiedGroupKFold` / `group_split(X, y, groups, test_size, seed)` so **all samples
  sharing a group stay on one side**. Default splitting is group-aware; plain stratified offered but
  flagged "optimistic."
- Deterministic via `SEED`.

**The group key is per-dataset — resolved, A5.** `source_file` is only the right key where one
capture yields many flows, which is *not* true of the primary dataset. Declared as `group_key:` in
`configs/datasets.yaml`; all numbers below measured from the built caches on 2026-08-23:

| Dataset | `group_key` | Why |
| --- | --- | --- |
| `iscxvpn` / `iscxtor` / `iscx_pooled` | `source_file` | 16,094 flows over 140 captures, and 54,429 over 95 — real grouping, real leakage to prevent |
| **`cstnet`** | **`capture_day`** | `source_file` is **vacuous here**: 46,372 flows from 46,372 pcaps, 1:1 — grouping by it *is* a random split. The real correlation is the crawl batch, so group by UTC capture date: 235 distinct days spanning 2020-10-16 → 2021-07-31, **every one of the 120 apps spans ≥7 days** (median 95), and no app puts more than 45% of its flows on a single day |
| `mobileapp` | `source_file` (vacuous, accepted) | 1 CSV = 1 sample by construction; no grouping exists to be had. At 4 samples/class this is a few-shot problem anyway (P4) |
| `postquantumtls` | — | not a classifier target; Phase-2 KEM ground truth |

`groups(key)` resolves a **named** key through a resolver in `schema.py`, so one dataset can offer
several: `source_file` (`meta["source_file"]`), `capture_day` (UTC date of `start_ts`), `server_ip`
(`five_tuple.dst_ip`), `stem` (`meta["stem"]`).

**Vacuous-grouping guard.** `group_split` raises when `n_groups == n_samples` unless the caller
passes `allow_vacuous=True` (which `mobileapp` must). A group key that silently degrades into a
random split — exactly what `source_file` does on CSTNET — can then never be reported as
"leakage-safe" again.

**Thin-class reporting.** Under `capture_day`, a 120-class stratified group split can still leave a
day-poor app with zero test samples (`amap.com` has only 7 days). `group_split` returns, and
`report.py` records, `classes_absent_from_test` — a macro-F1 that silently averaged over a missing
class is not a result.

**DoD:** `test_splits.py` asserts the group intersection across train/test is empty, **and** that
`group_split` refuses a vacuous grouping by default.

### 5.4 Metrics & report (`eval/metrics.py`, `report.py`)
- `evaluate(model, X, y)` → `{accuracy, macro_f1, weighted_f1, per_class:{p,r,f1,support},
  confusion, n}`. Headline = **macro-F1**.
- Majority-class / `DummyClassifier` baseline computed alongside for context.
- `report.py` writes `runs/<id>/metrics.json`, `config.yaml` (frozen), `confusion.png` (optional),
  and the model artifact → fully reproducible.

**DoD:** metrics match sklearn references on a toy set; run dir is self-describing.

### 5.5 Training harness (`training/train.py` + `scripts/train.py`)
- Config-driven: `{dataset, target, features, model, params, split, seed}`.
- Flow: load `FlowDataset` (cached from P0) → features → group-aware split → fit (with val for early
  stop) → evaluate on held-out test → persist run.
- CLI: `python scripts/train.py --model flow_gbdt --dataset iscx_pooled --target traffic_type`
  (or `--config configs/experiments/iscx_pooled_traffic_type_gbdt.yaml`). `--dataset` accepts a
  pooled alias from `configs/datasets.yaml`; `--group-key` defaults to the dataset's declared
  `group_key` and never has to be passed by hand.

**DoD:** one command produces a run dir with metrics beating the majority baseline.

---

## 6. Phase-3 experiment matrix (targets to run)

> Updated after verifying label distributions from the built caches (see
> `DEPLOYMENT-AND-PRIMARY-DATASET.md`). **CSTNET-TLS1.3 is the primary (deployment-distribution)
> dataset; ISCX is auxiliary** for the behavioral head only.

| `--dataset` | Target | `group_key` | Notes / expectation |
| --- | --- | --- | --- |
| **`iscx_pooled`** | traffic_type (8) | `source_file` | **Must pool** — ISCXVPN realizes only **7** (no `browsing` captures exist) and its `p2p` is a **single file** (`vpn_bittorrent.pcap`) → unlearnable under a group-aware split; ISCXTor supplies both. Pooled = 70,523 flows / 235 capture groups / 8 classes, min 10 groups/class (`p2p`) → group split well-defined. Headline macro-F1 with GBDT (+CNN in P4). **This is the exit-criterion run** |
| `iscx_pooled` | tunnel_type (none/vpn/tor) | `source_file` | coarse sanity. Pooled gives all 3 values (13,474+53,732 none / 2,620 vpn / 697 tor); `is_tor` is ~1% (77:1) → treat Tor as a flag, not a balanced class |
| **`cstnet`** | app (120, top-k) | **`capture_day`** | **primary** app/website-ID; GBDT = the **baseline floor** the P4 deep models must beat. Severe tail (`chia.net`=16) → report macro-F1 **with and without** the sub-50 tail, plus `classes_absent_from_test` |
| `mobileapp` | activity (92) | `source_file` (vacuous → `allow_vacuous=True`) | few-shot (**4/class**) — GBDT is a weak baseline only; the real member is few-shot (P4). App(8)→Activity is the one clean within-dataset hierarchy |
| `iscxvpn` | traffic_type (7) | `source_file` | **smoke test only, not reportable** — `p2p` is one capture, so it lands entirely on one side of any group split |

**Do-not-run (now guarded in code):** `target="app"` on ISCXVPN/ISCXTor — labels are
capture-filename stems (~1 class/file), so a group-aware split is undefined. `FlowDataset` raises
via `invalid_targets: [app]` in `configs/datasets.yaml`. PostQuantumTLS `app` (1/class) is Phase-2
KEM ground truth, not a classifier target.

---

## 7. Work breakdown & Definition of Done

All tasks below are **complete**; see §12 for the measured results.

| # | Task | DoD | ✓ |
| --- | --- | --- | --- |
| **3.0** | **Phase-0 surface (§11):** `LabelSpace`; `FlowDataset.groups()` / `.num_classes` / `labels(space=)`; `data/pooled.py`; `group_key` + `iscx_pooled` in `configs/datasets.yaml`; `joblib` dep + `[viz]` extra; carry `invalid_targets` through `filter()`; README status | the existing 331 tests still pass unchanged; `load_pooled(["iscxvpn","iscxtor"])` yields 70,523 flows / 235 groups / 8 classes | ✅ |
| 3.1 | `BaseModel` ABC + `ModelRegistry` + `@register` | create-by-name; save/load round-trip | ✅ |
| 3.2 | `LabelSpace` binding + canonical proba columns | proba cols align to taxonomy | ✅ |
| 3.3 | `eval/splits.py` group-aware splitter | no group leak (test asserts) + vacuous grouping refused | ✅ |
| 3.4 | `eval/metrics.py` + Dummy baseline | macro/weighted F1, per-class, confusion | ✅ |
| 3.5 | `models/flow_gbdt.py` (LightGBM, imbalance) | trains; proba sums to 1; importances | ✅ |
| 3.6 | `training/train.py` + `scripts/train.py` CLI | one-command run → `runs/<id>/` | ✅ |
| 3.7 | `report.py` reproducible run dir | metrics.json + frozen config + artifact + `label_space.json` | ✅ |
| 3.8 | Experiment configs (the 4 reportable rows above) | each runs; results logged via `scripts/summarize_runs.py` | ✅ |

---

## 8. Acceptance test (end of Phase 3)

Rewritten against the APIs task 3.0 actually adds (the original called `ds.groups()`,
`ds.num_classes` and a bare `LabelSpace`, none of which existed — see §11 A1/A2).

```python
from qsentinel.data import load_pooled
from qsentinel.models import ModelRegistry
from qsentinel.eval import group_split, evaluate
from qsentinel.schema import LabelSpace

ds = load_pooled(["iscxvpn", "iscxtor"], target="traffic_type")   # == load("iscx_pooled", ...)
assert len(ds) == 70_523 and ds.num_classes == 8

space = LabelSpace.taxonomy("traffic_type")          # 8 cols, dataset-independent  ← the seam
X = ds.features("flow_stats")                        # (N, 56) float32
y = ds.labels(space=space)                           # canonical ids, NOT the observed subset
g = ds.groups("source_file")                         # (N,) capture ids; 235 distinct

tr, te = group_split(X, y, g, test_size=0.2, seed=SEED)   # no capture on both sides
assert set(g[tr]).isdisjoint(set(g[te]))

m = ModelRegistry.create("flow_gbdt", label_space=space)
m.fit(X[tr], y[tr])
rep = evaluate(m, X[te], y[te], space=space)
assert rep["macro_f1"] > rep["baseline_macro_f1"]
assert m.predict_proba(X[te]).shape == (len(te), len(space)) == (len(te), 8)
assert m.classes_ == list(space.names)               # column meaning is stated, not implied

# a group key that is secretly a random split is refused, not silently accepted
with pytest.raises(ValueError):
    group_split(X, y, groups=np.arange(len(y)), test_size=0.2, seed=SEED)

# CLI writes a reproducible run
# $ python scripts/train.py --model flow_gbdt --dataset iscx_pooled --target traffic_type
#   → runs/<id>/{metrics.json, config.yaml, model.joblib, label_space.json, confusion.png}
```
**Then Phase 4 adds more model *types* (CNN/SSM/transformer/GNN) behind the same `BaseModel`, and
Phase 5 calibrates + stacks them.**

---

## 9. Risks & open questions
- **Leakage via correlated flows** (biggest one) → group-by-`source_file` splitting is the default;
  plain stratified only for quick looks, labeled "optimistic."
- **Class imbalance** (BitTorrent/Google/Twitter sparse) → balanced weights + macro-F1 headline;
  fold in Phase-0 `min_samples_per_class`.
- **Comparability to papers:** many report flow-level random-split accuracy (inflated); our
  group-aware macro-F1 will read *lower but honest* — document this explicitly so results aren't
  misread as "worse." **Quantified in `phase4-models/REPLICATION-DRAPERGIL.md`:** the same features,
  model and flow timeout score +0.10 (2-class) to +0.35 (7-class) higher under random 10-fold CV
  than under `source_file` grouping.
- **CSTNET 121-class with GBDT** will underperform deep models — that's the point; it sets the
  baseline P4 must beat.
- **`predict_proba` column alignment** across future models must be enforced now via `LabelSpace`,
  or Phase-5 stacking breaks silently. *Concrete instance found while resolving A2:* today
  `labels("traffic_type")` returns **7** names on `iscxvpn` and **8** on `iscxtor`, so column 5
  means `voip` on one and `file_transfer` on the other. Stacking those two would have been silently
  wrong. Fixed by binding models to `LabelSpace.taxonomy(...)` rather than to observed classes.
- **A vacuous group key reads exactly like a safe one** (resolved, A5). `source_file` on CSTNET
  yields 46,372 groups for 46,372 samples — a random split wearing the word "grouped". The guard in
  `group_split` exists so this class of error is loud rather than flattering.

## 10. Not doing yet (explicit)
Deep models (P4), domain experts (P4), calibration + stacking + router (P5), byte/image/graph
features (P4), live capture (P6).

---

## 11. Amendments — resolved gaps (2026-08-23)

Five gaps between this spec and the Phase-0 code it builds on, resolved before implementation.
Every number below was measured from the built caches, not assumed. Together they define **task
3.0**, which lands before 3.1.

### A1 — `FlowDataset.groups()` and `.num_classes` did not exist
§8 called both; `schema.py` had neither (only the `Flow.source_file` property). **Resolution:**
add to `FlowDataset` in `qsentinel/schema.py`.

```python
def groups(self, key: str = "source_file") -> np.ndarray:   # (N,) str, via GROUP_KEYS resolver
def num_classes(self, target=None) -> int                    # property-like; len(label_names())
```

`GROUP_KEYS: dict[str, Callable[[Flow], str]]` = `source_file` · `capture_day` (UTC date of
`start_ts`) · `server_ip` (`five_tuple.dst_ip`) · `stem`. Keeping the resolver a plain dict means a
dataset can offer several keys and `configs/datasets.yaml` picks the default — no per-dataset
branching in `eval/`. The *judgement* about whether a grouping is meaningful lives in
`eval/splits.py` (see A5), not here; `schema.py` stays dumb.

### A2 — `LabelSpace` did not exist
The doc treats it as a Phase-0 given. In reality canonical ordering was implicit in
`TRAFFIC_TYPES` / `TUNNEL_TYPES` plus `label_names()`, and `labels()` narrows to the **observed**
subset — so the same target has different column meanings on different datasets (7 vs 8 on
ISCXVPN vs ISCXTor). That is precisely the silent Phase-5 breakage §9 warns about, already present.

**Resolution:** new frozen dataclass in `qsentinel/schema.py`; full spec in §5.1. The load-bearing
decision is that **`labels()` is not changed** — widening is opt-in through `labels(space=...)`.
The alternative (making `labels()` always return the full taxonomy) would break existing pinned
behaviour such as `test_loaders.py`'s `label_names() == ["browsing", "chat"]` for no gain, since
the model is the thing that needs the wide space, not the dataset.

### A3 — no way to pool datasets
§6 and `CASCADE-AND-HNDL-ORDERING.md` both call pooling ISCXVPN+ISCXTor **mandatory** for
`traffic_type`, but `data/` could only load one dataset at a time. **Resolution:** new
`qsentinel/data/pooled.py` exposing `load_pooled(names, target=..., **kw) -> FlowDataset`, plus a
`pooled:` alias block in `configs/datasets.yaml` so `load("iscx_pooled")` and `--dataset
iscx_pooled` work.

Pooling is safe here, checked rather than hoped: **both ISCX loaders already populate the full
label-field set** (`traffic_type`, `is_vpn`, `is_tor`, `tunnel_type`, `app`), so no flow is missing
a field and `raw_labels` cannot `KeyError`; and `meta["source_file"]` is an absolute path, so
capture groups stay globally unique across constituents. `load_pooled` still validates both
invariants and raises on violation, since a future constituent may not be so tidy.

Pooled result: **70,523 flows · 235 capture groups · all 8 `traffic_type` classes**, minimum 10
groups for the thinnest class (`p2p`: 1 VPN + 9 Tor captures) — enough for a group-aware split.

### A4 — the exit criterion contradicted §6
Stated as `--dataset iscxvpn`, which §6 simultaneously declares unusable: 7 of 8 classes, and
`p2p` from a single capture, so a group-aware split cannot place it on both sides. **Resolution:**
the criterion moves to `--dataset iscx_pooled` (patched at the top of this document and in §5.5,
§6, §8). `iscxvpn` alone is retained as a labelled smoke test.

### A5 — group-splitting is a no-op on the primary dataset
`source_file` grouping protects ISCX, but CSTNET — the *primary* dataset per
`DEPLOYMENT-AND-PRIMARY-DATASET.md` — is `granularity: session`, one flow per pcap. Grouping by
`source_file` there gives 46,372 groups for 46,372 samples: a random split relabelled as
leakage-safe. MobileApp is the same by construction (1 CSV = 1 sample). The doc never addressed it.

**Resolution:** per-dataset `group_key` (table in §5.3) + a `group_split` guard that refuses a
vacuous grouping unless `allow_vacuous=True`. CSTNET's key becomes `capture_day`, which the data
supports comfortably:

| Measure | Value |
| --- | --- |
| capture window | 2020-10-16 → 2021-07-31 |
| distinct capture days | 235 |
| days per app | min **7**, 5th pct 30, median 95, max 193 — **no app below 7** |
| app's flows on its busiest single day | median 6%, max 45% — no app concentrated in one day |
| apps with ≤2 distinct server IPs | 4 / 120 (why `server_ip` was rejected as the key) |

So a day-grouped split keeps every one of the 120 classes on both sides while removing same-crawl
correlation. A *chronological* day split (train early, test late) is the stronger deployment-transfer
test and should be offered as a secondary report, but not as the default — at 7 days, `amap.com`
would be fragile. Thin-class dropout is reported explicitly rather than averaged away (§5.3).

### A6 — housekeeping folded into 3.0
- `joblib` is used by §5.2 but only present transitively via scikit-learn → add to
  `pyproject.toml` `dependencies`. `matplotlib` is absent → add a `[viz]` extra and make
  `confusion.png` a skipped artifact, not an ImportError.
- `FlowDataset.filter()` returns `FlowDataset(flows, self.name, self.default_target, self.config)`
  — dropping `invalid_targets`, so the ISCX `app` guard would evaporate on any filtered copy.
  Latent today (no filtered dataset sets it) but exactly the guard §6 relies on. Pass it through.
- `README.md` still says "Status: Phase 0 complete" though P1 and P2 shipped and 331 tests pass.

---

## 12. Results (Phase 3 complete, 2026-08-23)

All four reportable experiments run, logged, and reproducible from `runs/<id>/`. Regenerate
this table with `scripts/summarize_runs.py`.

| `--dataset` | target | classes | test n | **macro-F1** | baseline | accuracy | group key | groups tr/te |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `iscx_pooled` | traffic_type | 8 | 17,918 | **0.6317** | 0.0498 | 0.8837 | `source_file` | 137 / 75 |
| `iscx_pooled` | tunnel_type | 3 | 14,045 | **0.8983** | 0.3260 | 0.9946 | `source_file` | 169 / 48 |
| `cstnet` | app | 120 | 9,384 | **0.8110** | 0.0001 | 0.8376 | `capture_day` | 161 / 49 |
| `mobileapp` | activity | 92 | 92 | **0.4409** | 0.0002 | 0.4891 | `source_file`† | 276 / 92 |

†vacuous by construction (1 CSV = 1 sample), declared in `allow_vacuous_groups`.

**Every split was group-disjoint and no class was absent from any test fold** — including
CSTNET's 120 classes under day-grouping, which is what the ≥7-days-per-app measurement in
§11 A5 predicted.

### What the numbers say

**The exit criterion is met.** `iscx_pooled / traffic_type` reaches macro-F1 0.6317 against a
0.0498 majority baseline under a split where no capture appears on both sides. Per-class F1
splits sharply by *how much* behaviour a class has: `p2p` 0.99 and `browsing` 0.97 (bulk,
distinctive), while the interactive classes confuse with one another — `chat` 0.41,
`file_transfer` 0.45, `video_streaming` 0.47. That is the honest picture; the literature's
random-split numbers on these datasets are higher because they leak — and that is now **measured,
not asserted**: re-running the Draper-Gil (ICISSP 2016) baseline under both protocols, changing
*only* the fold assignment, inflates results by **+0.10** on binary VPN detection and up to
**+0.35** on 7-class characterisation. The inflation grows with class count. See
`phase4-models/REPLICATION-DRAPERGIL.md`.

**`tunnel_type` is the clearest argument for the macro-F1 headline in this whole project.**
Accuracy moves 0.9570 → 0.9946, which looks like a rounding error, while macro-F1 moves
0.3260 → 0.8983. Reporting accuracy here would have hidden the entire result. (It also
confirms `CASCADE-AND-HNDL-ORDERING.md`: tunnelling is so visible that Stage-1 should read
it deterministically rather than spend a model on it.)

**CSTNET sets a high floor for Phase 4.** 0.8110 macro-F1 over 120 classes — 0.8443 over the
91 classes with ≥50 test samples, so the tail costs about 3 points. Flow statistics alone
already separate most of the 120 services; the P4 byte-transformer and GNN have to beat
**0.81**, not the ~0.5 a "GBDT is just the baseline" framing would have assumed. Worst
classes are corporate/CDN-fronted sites that look alike on the wire (`vmware.com` 0.33,
`nvidia.com` 0.33, `ibm.com` 0.35); best are single-origin services (`digitaloceanspaces.com`
1.00, `amap.com` 0.99).

**MobileApp is a floor, not a result.** 92 test samples across 92 classes is *one sample per
class*, so 0.4409 carries enormous variance. It clears the baseline and that is all it is
being asked to do; the real member is the Phase-4 few-shot / App→Activity cascade.

### Cost note
The CSTNET run takes ~67 minutes (120 classes x 600 boosting rounds over 46,372 flows). The
other three finish in 1-4 minutes each.

### Handover to Phase 4
The seam is in place and exercised: every member is constructed with a `LabelSpace`, trains
on canonical ids, and emits `(N, len(space))` with zero columns for classes it never saw.
Phase 5's `CalibratedModel` can subclass `BaseModel` and override `_predict_proba` without
touching any member. One gap Phase 5 will need to fill: `eval/splits.py` exposes a single
split, not the K-fold generator that out-of-fold stacking requires.

