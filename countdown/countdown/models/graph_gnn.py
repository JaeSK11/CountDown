"""The graph member behind the ``BaseModel`` seam.

``graph_gnn_baseline`` wraps :class:`~countdown.models.tfe_gnn.TFEGNNNet` (Zhang et al.,
WWW'23) so it has the same ``fit / predict_proba / save / load`` surface as every other
member and returns probabilities in :class:`~countdown.schema.LabelSpace` order.  The
network, the graph construction and their fidelity notes stay where they are
(``models/tfe_gnn.py``, ``features/traffic_graph.py``); this module is the harness-facing
skin and the training loop.

**Input.** ``X`` is a rank-3 ``(N, P, H + L)`` int16 array: ``P`` packets per sample, each
row the ``H`` header bytes followed by the ``L`` payload bytes, already padded with
:data:`~countdown.features.traffic_graph.PAD_TRUNC_DIGIT` exactly as
``features/byte_prep.py`` writes them (``H = 40``, ``L = 150``, ``P = 50`` by default).
Byte graphs are built from those rows on the fly inside the data loader, the same way
``scripts/replicate_tfegnn.py`` does, because materialising ~2 M graph objects per epoch
costs far more than rebuilding them (~0.09 ms each).

``input_type`` is ``"byte_matrix"``.  There is **no registered extractor of that name
yet**: Phase-0 ``Flow`` objects keep ``(ts, size, direction)`` only, so the bytes come from
the separate pcap pass in ``features/byte_prep.py`` (``scripts/build_tfegnn_cache.py``).
Until byte retention lands in the flow layer (task 4.2), this member is driven from that
cache by ``scripts/train_graph_gnn.py`` rather than through ``scripts/train.py``.  The
name is declared now so the harness contract check has something to compare to when the
extractor exists.

**Training.** Defaults reproduce the authors' ``config.py``: Adam at 1e-2 with 10 % warmup
and a linear decay to ``lr_min``, batch 102 with 5-step gradient accumulation, 20 epochs
(their base setting -- their per-dataset overrides are 20 / 120 / 100 / 120 for
VPN / non-VPN / Tor / non-Tor, so pass ``epochs`` explicitly for a replication), no
validation set, final model reported.  Two project-side options are off by default so the
baseline stays the paper's baseline: ``class_weight="balanced"`` and early stopping on
validation macro-F1 (``early_stop_patience > 0``, needs ``val``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.models.base import BaseModel, register

log = get_logger(__name__)


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("graph_gnn_baseline needs PyTorch: pip install torch") from exc


def _pyg():
    try:
        import torch_geometric  # noqa: F401
        from torch_geometric.data import Batch, Data
        return Batch, Data
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "graph_gnn_baseline needs torch-geometric: pip install torch-geometric"
        ) from exc


class ByteMatrixGraphDataset:
    """``(N, P, H + L)`` byte rows -> per-packet header/payload byte graphs, on access.

    A plain class rather than a ``torch.utils.data.Dataset`` subclass so the module
    imports without torch; the loader only needs ``__len__`` / ``__getitem__``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray | None, header_len: int) -> None:
        self.X = X
        self.y = y
        self.header_len = int(header_len)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        from countdown.features.traffic_graph import build_byte_graph

        torch = _torch()
        _, Data = _pyg()
        sample = self.X[idx]
        parts = (sample[:, : self.header_len], sample[:, self.header_len:])
        graphs: list[list] = [[], []]
        for rows, acc in zip(parts, graphs):
            for row in rows:
                g = build_byte_graph(row)
                acc.append(Data(
                    x=torch.from_numpy(g.node_values.astype(np.int64)),
                    edge_index=torch.from_numpy(g.edge_index),
                    num_nodes=g.n_nodes,
                ))
        label = -1 if self.y is None else int(self.y[idx])
        return graphs[0], graphs[1], label


def _collate(items):
    torch = _torch()
    Batch, _ = _pyg()
    h = [g for it in items for g in it[0]]
    p = [g for it in items for g in it[1]]
    y = torch.tensor([it[2] for it in items], dtype=torch.long)
    return Batch.from_data_list(h), Batch.from_data_list(p), y


@register("graph_gnn_baseline")
class TFEGNNBaseline(BaseModel):
    """TFE-GNN as an ensemble member.  ``X`` is ``(N, P, H + L)`` int16 byte rows."""

    input_type = "byte_matrix"
    expects_ndim = 3
    supports = "*"

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        p = self.params
        # -- the authors' config.py ------------------------------------------------------
        p.setdefault("header_len", 40)
        p.setdefault("epochs", 20)
        p.setdefault("batch_size", 102)
        p.setdefault("accum", 5)
        p.setdefault("lr", 1e-2)
        p.setdefault("lr_min", 1e-4)
        p.setdefault("warmup", 0.1)
        p.setdefault("embedding_dim", 64)
        p.setdefault("hidden_dim", 128)
        p.setdefault("n_gnn_layers", 4)
        p.setdefault("dropout", 0.2)
        p.setdefault("readout", "last_state")
        # -- ours, off by default so the baseline is the paper's ---------------------------
        p.setdefault("class_weight", None)        # None = paper; "balanced" = ours
        p.setdefault("early_stop_patience", 0)    # 0 = full schedule, final model
        p.setdefault("device", "auto")
        p.setdefault("seed", 42)
        p.setdefault("workers", 8)
        p.setdefault("eval_batch_size", 102)
        self.net = None
        self.history_: list[dict[str, float]] = []
        self._n_packets: int | None = None

    # -- helpers -----------------------------------------------------------------------
    def _resolve_device(self):
        torch = _torch()
        want = self.params["device"]
        if want == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(want)

    def _build(self, n_classes: int):
        from countdown.models.tfe_gnn import TFEGNNNet

        p = self.params
        return TFEGNNNet(
            n_classes=n_classes,
            embedding_dim=int(p["embedding_dim"]),
            hidden_dim=int(p["hidden_dim"]),
            n_gnn_layers=int(p["n_gnn_layers"]),
            dropout=float(p["dropout"]),
            readout=str(p["readout"]),
        )

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(
                f"graph_gnn_baseline expects (N, P, H + L) byte rows, got {X.shape}. "
                f"Build X from the byte_prep cache (scripts/build_tfegnn_cache.py)."
            )
        if X.shape[2] <= int(self.params["header_len"]):
            raise ValueError(
                f"last axis is {X.shape[2]} but header_len is {self.params['header_len']}: "
                f"no payload bytes would remain"
            )
        return X.astype(np.int16, copy=False)

    def _loader(self, X: np.ndarray, y: np.ndarray | None, shuffle: bool, batch_size: int):
        torch = _torch()
        from torch.utils.data import DataLoader

        workers = int(self.params["workers"])
        ds = ByteMatrixGraphDataset(X, y, header_len=int(self.params["header_len"]))
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate,
            num_workers=workers, persistent_workers=workers > 0, drop_last=False,
        )

    def _logits(self, hb, pb, n_samples: int, device):
        hb, pb = hb.to(device), pb.to(device)
        z = self.net.encode_packets(
            hb.x, hb.edge_index, hb.batch, pb.x, pb.edge_index, pb.batch,
            n_packets=n_samples * self._n_packets,
        )
        return self.net(z.view(n_samples, self._n_packets, -1))

    def _predict_logits(self, X: np.ndarray, device):
        torch = _torch()
        self.net.eval()
        outs = []
        with torch.no_grad():
            for hb, pb, yb in self._loader(X, None, shuffle=False,
                                           batch_size=int(self.params["eval_batch_size"])):
                outs.append(self._logits(hb, pb, int(yb.shape[0]), device).cpu())
        return torch.cat(outs) if outs else torch.zeros(0, self.n_classes_)

    # -- the contract ------------------------------------------------------------------
    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        torch = _torch()
        from torch import nn
        from sklearn.metrics import f1_score

        p = self.params
        X = self._check_X(X)
        y = np.asarray(y, dtype=np.int64)
        torch.manual_seed(int(p["seed"]))
        np.random.seed(int(p["seed"]))
        device = self._resolve_device()
        self._n_packets = int(X.shape[1])

        self.net = self._build(self.n_classes_).to(device)

        weight = None
        if p["class_weight"] == "balanced":
            counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
            w = np.where(counts > 0, counts.sum() / (np.count_nonzero(counts) * np.maximum(counts, 1)), 0.0)
            weight = torch.tensor(w, dtype=torch.float32, device=device)
        elif p["class_weight"] not in (None, "none"):
            raise ValueError(f"class_weight must be None or 'balanced', got {p['class_weight']!r}")
        criterion = nn.CrossEntropyLoss(weight=weight)

        loader = self._loader(X, y, shuffle=True, batch_size=int(p["batch_size"]))
        accum = max(1, int(p["accum"]))
        epochs = int(p["epochs"])
        opt = torch.optim.Adam(self.net.parameters(), lr=float(p["lr"]))
        steps = max(1, (len(loader) // accum) * epochs)
        warmup = max(1, int(steps * float(p["warmup"])))
        floor = float(p["lr_min"]) / float(p["lr"])

        def lr_at(step: int) -> float:
            if step < warmup:
                return step / warmup
            prog = (step - warmup) / max(1, steps - warmup)
            return max(floor, 1.0 - prog * (1.0 - floor))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

        patience = int(p["early_stop_patience"])
        if patience > 0 and val is None:
            log.warning("[graph_gnn] early_stop_patience=%d but no val fold; training the "
                        "full schedule", patience)
            patience = 0
        best, stale, best_state = -1.0, 0, None
        self.history_ = []
        log.info("[graph_gnn] %d samples x %d packets, %d classes, %d epochs, device=%s",
                 len(y), self._n_packets, self.n_classes_, epochs, device)

        for epoch in range(epochs):
            self.net.train()
            t0, total, nb = time.time(), 0.0, 0
            opt.zero_grad(set_to_none=True)
            for i, (hb, pb, yb) in enumerate(loader):
                yb = yb.to(device)
                loss = criterion(self._logits(hb, pb, int(yb.shape[0]), device), yb) / accum
                loss.backward()
                total += float(loss.detach()) * accum
                nb += 1
                if (i + 1) % accum == 0:
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            if nb % accum:                       # flush a trailing partial accumulation
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            row = {"epoch": epoch, "loss": total / max(nb, 1),
                   "lr": float(sched.get_last_lr()[0]), "seconds": time.time() - t0}
            if val is not None:
                Xv, yv = self._check_X(val[0]), np.asarray(val[1], dtype=np.int64)
                pred = self._predict_logits(Xv, device).argmax(1).numpy()
                row["val_macro_f1"] = float(f1_score(yv, pred, average="macro", zero_division=0))
                row["val_acc"] = float((pred == yv).mean())
                if patience > 0:
                    if row["val_macro_f1"] > best:
                        best, stale = row["val_macro_f1"], 0
                        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    else:
                        stale += 1
                        if stale >= patience:
                            self.history_.append(row)
                            log.info("[graph_gnn] early stop at epoch %d (best val macro-F1 %.4f)",
                                     epoch + 1, best)
                            break
            self.history_.append(row)
            log.info("[graph_gnn] epoch %d/%d %s", epoch + 1, epochs,
                     " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _predict_proba(self, X) -> np.ndarray:
        torch = _torch()
        X = self._check_X(X)
        device = self._resolve_device()
        self.net.to(device)
        logits = self._predict_logits(X, device)
        return torch.softmax(logits, dim=1).numpy()

    # -- persistence -------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        if self.net is None:
            return {}
        return {
            "state_dict": {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
            "n_packets": self._n_packets,
            "history": self.history_,
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        torch = _torch()
        self._n_packets = int(state["n_packets"])
        self.net = self._build(self.n_classes_)
        self.net.load_state_dict({k: torch.from_numpy(v) for k, v in state["state_dict"].items()})
        self.history_ = state.get("history", [])

    def n_params(self) -> int:
        return 0 if self.net is None else sum(p.numel() for p in self.net.parameters())


# ---------------------------------------------------------------------------------------
# Recommended variant (decision D1c, 2026-09-19)
# ---------------------------------------------------------------------------------------
class _DictGraphDataset(ByteMatrixGraphDataset):
    """The same on-access byte graphs, shaped for ``training/deep.py`` (dict batches)."""

    def labels(self) -> np.ndarray:
        return self.y if self.y is not None else np.zeros(len(self), dtype=np.int64)


def _collate_dict(items):
    h, p, y = _collate(items)
    return {"h": h, "p": p, "y": y.clamp(min=0)}


@register("graph_gnn")
class GINGraphMember(BaseModel):
    """The recommended graph member: **GIN** in the TFE-GNN towers, trained by ``deep.py``.

    Same input (``byte_matrix``), same byte-graph construction, same dual towers, fusion and
    BiLSTM as ``graph_gnn_baseline``; the message-passing rule is GIN instead of GraphSAGE
    (decision D1c) and the optimisation is the shared trainer (D3): AdamW, warmup + linear
    decay, gradient clipping, no class weights (D5).  The authors' 5-step accumulation of
    102-sample batches becomes one 256-sample batch.  Pass no ``val`` on data with no
    honest validation fold (ISCX-Tor) and the final epoch is reported.
    """

    input_type = "byte_matrix"
    expects_ndim = 3
    supports = "*"

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        p = self.params
        p.setdefault("header_len", 40)
        p.setdefault("conv", "gin")
        p.setdefault("epochs", 20)
        p.setdefault("batch_size", 256)
        p.setdefault("lr", 5e-3)
        p.setdefault("weight_decay", 0.01)
        p.setdefault("embedding_dim", 64)
        p.setdefault("hidden_dim", 128)
        p.setdefault("n_gnn_layers", 4)
        p.setdefault("dropout", 0.2)
        p.setdefault("class_weight", None)
        p.setdefault("early_stop_patience", 0)
        p.setdefault("device", "auto")
        p.setdefault("seed", 42)
        p.setdefault("workers", 8)
        self.net = None
        self.fit_result_ = None
        self._n_packets: int | None = None

    _check_X = TFEGNNBaseline._check_X

    def _build(self, n_classes: int):
        from countdown.models.tfe_gnn import TFEGNNNet

        p = self.params
        return TFEGNNNet(n_classes=n_classes, embedding_dim=int(p["embedding_dim"]),
                         hidden_dim=int(p["hidden_dim"]), n_gnn_layers=int(p["n_gnn_layers"]),
                         dropout=float(p["dropout"]), conv=str(p["conv"]))

    def _deep_config(self):
        from countdown.training.deep import DeepConfig

        p = self.params
        if p["class_weight"] not in (None, "none", "balanced"):
            raise ValueError(f"class_weight must be None or 'balanced', got {p['class_weight']!r}")
        return DeepConfig(
            epochs=int(p["epochs"]), batch_size=int(p["batch_size"]), eval_batch_size=int(p["batch_size"]),
            learning_rate=float(p["lr"]), weight_decay=float(p["weight_decay"]), warmup_ratio=0.1,
            patience=int(p["early_stop_patience"]), seed=int(p["seed"]),
            device=None if p["device"] == "auto" else str(p["device"]),
            amp=False, num_workers=int(p["workers"]), log_every=0,
            restore_best=int(p["early_stop_patience"]) > 0,
            class_weighted_loss=p["class_weight"] == "balanced",
        )

    def _forward(self, model, batch):
        n = int(batch["y"].shape[0])
        h, pl = batch["h"], batch["p"]
        z = model.encode_packets(h.x, h.edge_index, h.batch, pl.x, pl.edge_index, pl.batch,
                                 n_packets=n * self._n_packets)
        return model(z.view(n, self._n_packets, -1))

    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        from countdown.training.deep import DeepTrainer

        _torch(); _pyg()
        X = self._check_X(X)
        y = np.asarray(y, dtype=np.int64)
        self._n_packets = int(X.shape[1])
        self.net = self._build(self.n_classes_)
        cfg = self._deep_config()
        hl = int(self.params["header_len"])
        class_weights = None
        if cfg.class_weighted_loss:
            c = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
            class_weights = np.where(c > 0, len(y) / (np.count_nonzero(c) * np.maximum(c, 1)), 0.0)
        val_ds = None
        if val is not None and len(val[1]):
            val_ds = _DictGraphDataset(self._check_X(val[0]), np.asarray(val[1], dtype=np.int64), hl)
        self.fit_result_ = DeepTrainer(cfg).fit(
            self.net, _DictGraphDataset(X, y, hl), n_classes=self.n_classes_, val_ds=val_ds,
            forward_fn=self._forward, class_weights=class_weights, collate_fn=_collate_dict,
        )

    def _predict_proba(self, X) -> np.ndarray:
        from countdown.training.deep import DeepTrainer

        X = self._check_X(X)
        ds = _DictGraphDataset(X, np.zeros(len(X), dtype=np.int64), int(self.params["header_len"]))
        return DeepTrainer(self._deep_config()).predict_proba(
            self.net, ds, forward_fn=self._forward, collate_fn=_collate_dict)

    @property
    def history_(self) -> list:
        return [] if self.fit_result_ is None else self.fit_result_.history

    def n_params(self) -> int:
        return 0 if self.net is None else sum(p.numel() for p in self.net.parameters())

    def _state(self) -> dict[str, Any]:
        if self.net is None:
            return {}
        return {"state_dict": {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()},
                "n_packets": self._n_packets, "history": self.history_}

    def _load_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        torch = _torch()
        self._n_packets = int(state["n_packets"])
        self.net = self._build(self.n_classes_)
        self.net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state["state_dict"].items()})
