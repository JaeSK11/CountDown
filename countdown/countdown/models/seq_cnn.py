"""The sequence member: DF as the paper baseline, a dilated-residual CNN as ours.

``seq_cnn_baseline`` -- Deep Fingerprinting (Sirinam et al., CCS 2018)
    Input: the ``dir_seq`` extractor, ``(N, 5000)`` packet directions, nothing else.
    Network: four VGG-style blocks of 2 x (Conv1D k=8 -> BatchNorm -> act) -> MaxPool(8,
    stride 4) -> Dropout(0.1), filters 32/64/128/256, ELU in block 1 and ReLU after; then
    FC(512) -> BN -> ReLU -> Dropout(0.7) -> FC(512) -> BN -> ReLU -> Dropout(0.5) -> softmax.
    Training, as published: Adamax lr 2e-3, batch 128, 30 epochs, categorical CE, final
    model.  Its own loop, per decision D3 -- a baseline reproduces the paper's protocol.

``seq_cnn`` -- dilated-residual 1D CNN (Var-CNN-style; decision D1d, 2026-09-19)
    Input: ``packet_seq`` with two channels, signed MTU-capped size and inter-arrival
    time, first ``n`` packets (``feature_params: {n: 256, channels: 2}``).
    Network: a k=7 stem, then four residual stages with dilations 1/2/4/8 (two blocks
    each, widths 64/128/192/256, stride-2 between stages), mean + max pooling over
    the sequence, one hidden FC.  ~1.6 M parameters against DF's ~3.9 M at n=5000,
    and a receptive field of ~250 packets from 256 inputs.
    Training: ``training/deep.py`` (D3), no class weights (D5).

The Mamba / S4 stage in the original plan is deferred: ``mamba-ssm`` needs a CUDA toolkit
this machine does not have, and the measured question first is whether sizes + timing over
a few hundred packets beat direction-only over 5,000.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from countdown.config import get_logger
from countdown.models.base import BaseModel, register
from countdown.models.deep_member import DeepArrayMember

log = get_logger(__name__)


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("the sequence member needs PyTorch: pip install torch") from exc


# ---------------------------------------------------------------------------------------
# DF
# ---------------------------------------------------------------------------------------
def build_df(n_classes: int, length: int = 5000):
    """The DF network.  ``ceil_mode`` pooling reproduces Keras ``padding='same'`` lengths."""
    torch = _torch()
    from torch import nn

    def block(c_in: int, c_out: int, act) -> list:
        return [
            nn.Conv1d(c_in, c_out, 8, padding="same"), nn.BatchNorm1d(c_out), act(),
            nn.Conv1d(c_out, c_out, 8, padding="same"), nn.BatchNorm1d(c_out), act(),
            nn.MaxPool1d(8, stride=4, padding=2, ceil_mode=True), nn.Dropout(0.1),
        ]

    body = nn.Sequential(
        *block(1, 32, nn.ELU), *block(32, 64, nn.ReLU),
        *block(64, 128, nn.ReLU), *block(128, 256, nn.ReLU), nn.Flatten(),
    )
    with torch.no_grad():
        n_flat = body.eval()(torch.zeros(1, 1, length)).shape[1]
    body.train()
    head = nn.Sequential(
        nn.Linear(n_flat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.7),
        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, n_classes),
    )
    return nn.Sequential(body, head)


@register("seq_cnn_baseline")
class DFBaseline(BaseModel):
    """Deep Fingerprinting.  ``X`` is ``(N, L)`` packet directions from ``dir_seq``."""

    input_type = "dir_seq"
    expects_ndim = 2
    supports = "*"

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        p = self.params
        p.setdefault("epochs", 30)
        p.setdefault("batch_size", 128)
        p.setdefault("lr", 2e-3)                 # Adamax, as published
        p.setdefault("class_weight", None)
        p.setdefault("early_stop_patience", 0)   # the paper trains a fixed schedule
        p.setdefault("device", "auto")
        p.setdefault("seed", 42)
        self.net = None
        self.history_: list[dict[str, float]] = []
        self._length: int | None = None

    def _device(self):
        torch = _torch()
        want = self.params["device"]
        return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if want == "auto" else want)

    def _check(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"seq_cnn_baseline expects (N, L) directions from 'dir_seq', got {X.shape}")
        return X

    def _logits(self, Xt, device, batch: int = 1024):
        torch = _torch()
        outs = []
        for i in range(0, len(Xt), batch):
            outs.append(self.net(Xt[i:i + batch].to(device).float().unsqueeze(1)))
        return torch.cat(outs) if outs else torch.zeros(0, self.n_classes_, device=device)

    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        torch = _torch()
        from torch import nn
        from sklearn.metrics import f1_score

        p = self.params
        X = self._check(X)
        torch.manual_seed(int(p["seed"])); np.random.seed(int(p["seed"]))
        device = self._device()
        self._length = int(X.shape[1])
        self.net = build_df(self.n_classes_, self._length).to(device)

        weight = None
        if p["class_weight"] == "balanced":
            c = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
            weight = torch.tensor(np.where(c > 0, c.sum() / (np.count_nonzero(c) * np.maximum(c, 1)), 0.0),
                                  dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.Adamax(self.net.parameters(), lr=float(p["lr"]))

        # int8 on the device: 70k x 5000 directions is 350 MB, against 1.4 GB as float32.
        Xt = torch.from_numpy(X.astype(np.int8)).to(device)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
        bs, n = int(p["batch_size"]), len(yt)
        patience, best, stale, best_state = int(p["early_stop_patience"]), -1.0, 0, None
        self.history_ = []
        for epoch in range(int(p["epochs"])):
            self.net.train()
            t0, total = time.time(), 0.0
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                if len(idx) < 2:                      # BatchNorm cannot take a batch of one
                    continue
                opt.zero_grad(set_to_none=True)
                loss = criterion(self.net(Xt[idx].float().unsqueeze(1)), yt[idx])
                loss.backward(); opt.step()
                total += float(loss.detach()) * len(idx)
            row = {"epoch": epoch, "loss": total / n, "seconds": time.time() - t0}
            if val is not None and len(val[1]):
                self.net.eval()
                with torch.no_grad():
                    pred = self._logits(torch.from_numpy(self._check(val[0]).astype(np.int8)), device).argmax(1).cpu().numpy()
                row["val_macro_f1"] = float(f1_score(val[1], pred, average="macro", zero_division=0))
                if patience > 0:
                    if row["val_macro_f1"] > best:
                        best, stale = row["val_macro_f1"], 0
                        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                    else:
                        stale += 1
            self.history_.append(row)
            if epoch % 5 == 0 or epoch == int(p["epochs"]) - 1:
                log.info("[seq_cnn_baseline] epoch %d/%d %s", epoch + 1, p["epochs"],
                         " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))
            if patience > 0 and stale >= patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _predict_proba(self, X) -> np.ndarray:
        torch = _torch()
        device = self._device()
        self.net.to(device).eval()
        with torch.no_grad():
            logits = self._logits(torch.from_numpy(self._check(X).astype(np.int8)), device)
            return torch.softmax(logits.float(), dim=1).cpu().numpy()

    def n_params(self) -> int:
        return 0 if self.net is None else sum(p.numel() for p in self.net.parameters())

    def _state(self) -> dict[str, Any]:
        if self.net is None:
            return {}
        return {"state_dict": {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
                "length": self._length, "history": self.history_}

    def _load_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        torch = _torch()
        self._length = int(state["length"])
        self.net = build_df(self.n_classes_, self._length)
        self.net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state["state_dict"].items()})
        self.history_ = state.get("history", [])


# ---------------------------------------------------------------------------------------
# dilated-residual CNN
# ---------------------------------------------------------------------------------------
def build_dilated_resnet(n_classes: int, in_channels: int = 2,
                         widths=(64, 128, 192, 256), dilations=(1, 2, 4, 8), dropout: float = 0.3):
    torch = _torch()
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c_in: int, c_out: int, dilation: int, stride: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
            self.bn1 = nn.BatchNorm1d(c_out)
            self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=dilation, dilation=dilation, bias=False)
            self.bn2 = nn.BatchNorm1d(c_out)
            self.skip = None
            if stride != 1 or c_in != c_out:
                self.skip = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))

        def forward(self, x):
            h = torch.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return torch.relu(h + (x if self.skip is None else self.skip(x)))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(in_channels, widths[0], 7, padding=3, bias=False),
                                      nn.BatchNorm1d(widths[0]), nn.ReLU())
            blocks, c = [], widths[0]
            for i, (w, d) in enumerate(zip(widths, dilations)):
                blocks += [Block(c, w, d, stride=1 if i == 0 else 2), Block(w, w, d, stride=1)]
                c = w
            self.stages = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.Linear(2 * c, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, n_classes))

        def forward(self, x):                     # x: (B, T, C) as the extractor emits it
            x = x.transpose(1, 2)
            h = self.stages(self.stem(x))
            return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))

    return Net()


@register("seq_cnn")
class DilatedResSeqCNN(DeepArrayMember):
    """Dilated-residual CNN over signed sizes + inter-arrival times (``packet_seq``)."""

    input_type = "packet_seq"
    expects_ndim = 3
    supports = "*"
    recommended_feature_params = {"n": 256, "channels": 2}
    deep_defaults = {
        "epochs": 40, "batch_size": 128, "eval_batch_size": 1024, "learning_rate": 2e-3,
        "weight_decay": 0.01, "warmup_ratio": 0.1, "patience": 0, "amp": True, "num_workers": 4,
    }

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        self.params.setdefault("mtu", 1500.0)
        self.params.setdefault("dropout", 0.3)

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[2] != 2:
            raise ValueError(
                f"seq_cnn expects (N, T, 2) from 'packet_seq' with channels=2, got {X.shape}")
        out = np.empty_like(X)
        out[..., 0] = X[..., 0] / float(self.params["mtu"])            # signed size in [-1, 1]
        out[..., 1] = np.log1p(np.clip(X[..., 1], 0.0, None) * 1e3) / 10.0   # IAT: ms, log scale
        return out

    def _build(self, n_classes: int, sample_shape: tuple[int, ...]):
        return build_dilated_resnet(n_classes, in_channels=sample_shape[1], dropout=float(self.params["dropout"]))
