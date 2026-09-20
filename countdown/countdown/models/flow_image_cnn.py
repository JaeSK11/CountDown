"""The Okonkwo flow-image CNN -- Phase 4's image baseline.

Architecture, verbatim from Okonkwo et al., "A CNN Based Encrypted Network Traffic
Classifier" (AISC 2022), section 3 and Figure 2.  Eleven layers:

    conv(32, 3x3, s2) -> conv(32, 3x3, s2) -> maxpool(2, s2)
    -> conv(64, 3x3, s2) -> conv(64, 3x3, s2) -> dropout(0.25) -> maxpool(2, s2)
    -> flatten -> dense(64) -> dropout(0.5) -> dense(N, softmax)

224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 3, so the flatten is 64*3*3 = 576.  The paper labels
that tensor 576 in Figure 2, which is the check that this reading of the figure is right:
stride 2 is on every convolution, *in addition to* the two pooling layers.

That aggressive stride is also the architecture's weakness on this input.  A stride-2 3x3
convolution discards half the spatial positions at every step, and on a binary scatter
where each packet is an isolated mark, the discarded positions are the signal.  The
FlowPic histogram construction is dense and does not lose information the same way -- see
``phases/phase4-models/MODEL-image.md``.

Defaults reproduce the paper's training setup (section 4.3): batch 10, 60 epochs, Adam at
1e-4 with beta = (0.9, 0.999) and eps = 1e-8, categorical cross-entropy.  Batch 10 is
small because the authors were storage-limited on an i5 CPU; it is a parameter here, but
the default stays 10 so the baseline is the paper's baseline.
"""

from __future__ import annotations

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
        raise ImportError(
            "flow_image_cnn needs PyTorch: pip install torch"
        ) from exc


def build_network(n_classes: int, in_channels: int = 1, dropout: tuple = (0.25, 0.5),
                  size: int = 224):
    """The eleven-layer stack as an ``nn.Module``."""
    torch = _torch()
    from torch import nn

    body = nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.Dropout(dropout[0]),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten(),
    )
    with torch.no_grad():
        n_flat = body(torch.zeros(1, in_channels, size, size)).shape[1]
    head = nn.Sequential(
        nn.Linear(n_flat, 64), nn.ReLU(inplace=True),
        nn.Dropout(dropout[1]),
        nn.Linear(64, n_classes),
    )
    net = nn.Sequential(body, head)
    net.n_flat = n_flat
    return net


@register("flow_image_cnn_baseline")
class FlowImageCNN(BaseModel):
    """CNN over flow images.  ``X`` is ``(N, C, S, S)`` from the ``flow_image`` extractor."""

    input_type = "flow_image"
    expects_ndim = 4
    supports = "*"

    def __init__(self, label_space=None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        p = self.params
        p.setdefault("epochs", 60)
        p.setdefault("batch_size", 10)
        p.setdefault("lr", 1e-4)
        p.setdefault("betas", (0.9, 0.999))
        p.setdefault("eps", 1e-8)
        p.setdefault("dropout", (0.25, 0.5))
        p.setdefault("device", "auto")
        p.setdefault("seed", 42)
        p.setdefault("class_weight", None)   # None reproduces the paper; "balanced" is ours
        p.setdefault("early_stop_patience", 0)  # 0 = train the full schedule, as the paper did
        #: Upload the whole training tensor to the GPU when it is at most this many GB.
        #: At the paper's batch size of 10 this is worth ~60x in wall-clock; see _fit.
        p.setdefault("gpu_resident_gb", 12.0)
        self.net = None
        self.history_: list[dict[str, float]] = []

    # -- helpers -----------------------------------------------------------------------
    def _resolve_device(self):
        torch = _torch()
        want = self.params["device"]
        if want == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(want)

    def _build(self, n_classes: int, in_channels: int, size: int):
        """The backbone.  Overridden by subclasses so that only the network differs.

        Everything around it -- optimiser, batching, seeding, early stopping,
        persistence -- is inherited, which is what makes a scorecard comparison against
        this baseline attributable to the architecture rather than to the harness.
        """
        return build_network(
            n_classes, in_channels=in_channels,
            dropout=tuple(self.params["dropout"]), size=size,
        )

    @staticmethod
    def _as_nchw(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:      # (N, S, S) -> (N, 1, S, S)
            X = X[:, None, :, :]
        if X.ndim != 4:
            raise ValueError(
                f"flow_image_cnn expects (N, C, S, S) or (N, S, S) images, got {X.shape}. "
                f"Build X with the 'flow_image' extractor."
            )
        return X

    # -- the contract ------------------------------------------------------------------
    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        torch = _torch()
        from torch import nn
        p = self.params
        torch.manual_seed(int(p["seed"]))
        np.random.seed(int(p["seed"]))

        X = self._as_nchw(X)
        device = self._resolve_device()
        self._in_channels, self._size = int(X.shape[1]), int(X.shape[-1])
        self.net = self._build(self.n_classes_, self._in_channels, self._size).to(device)

        weight = None
        if p["class_weight"] == "balanced":
            counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
            inv = np.divide(len(y), self.n_classes_ * counts,
                            out=np.zeros_like(counts), where=counts > 0)
            weight = torch.tensor(inv, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=float(p["lr"]),
            betas=tuple(p["betas"]), eps=float(p["eps"]),
        )

        # Batching.  The paper's batch size is 10, and at that size the per-batch
        # host->device copy dominates: measured 168 ms/step against 2.8 ms for the same
        # work with the tensor already on the GPU -- a 60x difference that is pure
        # transfer overhead, not computation.  So when the training tensor fits in
        # `gpu_resident_gb` it is uploaded once and batches are gathered on-device.
        #
        # Both paths use the *same* permutation logic and differ only in where the tensor
        # lives, so a given seed produces the same batch composition either way.  (A
        # DataLoader would be the obvious alternative, but its RandomSampler draws
        # differently from torch.randperm, which would make results depend on which path
        # the available GPU memory happened to select -- and it buys nothing here, since
        # indexing an in-memory tensor involves no I/O to overlap.)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        Xt = torch.from_numpy(X)
        nbytes = Xt.element_size() * Xt.nelement()
        resident = device.type == "cuda" and nbytes <= float(p["gpu_resident_gb"]) * 1e9
        batch_size = int(p["batch_size"])

        if resident:
            Xt, yt = Xt.to(device), yt.to(device)
            log.info("[flow_image_cnn] %d images (%.2f GB) resident on %s",
                     len(Xt), nbytes / 1e9, device)
        elif device.type == "cuda":
            log.info("[flow_image_cnn] %.2f GB exceeds gpu_resident_gb=%.1f; "
                     "streaming batches from host", nbytes / 1e9, p["gpu_resident_gb"])

        def epoch_batches():
            perm = torch.randperm(len(Xt))
            if resident:
                perm = perm.to(device)
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = Xt[idx], yt[idx]
                if not resident:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                yield xb, yb

        Xv = yv = None
        if val is not None and len(val[0]) > 0:
            # Kept on the host and moved in chunks by _forward_batched, so a large val
            # fold cannot push the resident training tensor out of GPU memory.
            Xv = torch.from_numpy(self._as_nchw(val[0]))
            yv = torch.from_numpy(np.asarray(val[1], dtype=np.int64)).to(device)

        best, best_state, stale = -1.0, None, 0
        patience = int(p["early_stop_patience"])

        for epoch in range(int(p["epochs"])):
            self.net.train()
            total, correct, running = 0, 0, 0.0
            for xb, yb in epoch_batches():
                optimizer.zero_grad(set_to_none=True)
                out = self.net(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * len(yb)
                correct += int((out.argmax(1) == yb).sum().item())
                total += len(yb)

            row = {"epoch": epoch, "loss": running / max(total, 1),
                   "acc": correct / max(total, 1)}
            if Xv is not None:
                self.net.eval()
                with torch.no_grad():
                    vout = self._forward_batched(Xv, device)
                    row["val_acc"] = float((vout.argmax(1) == yv).float().mean().item())
                    row["val_loss"] = float(criterion(vout, yv).item())
                if patience > 0:
                    if row["val_acc"] > best:
                        best, stale = row["val_acc"], 0
                        best_state = {k: v.detach().clone()
                                      for k, v in self.net.state_dict().items()}
                    else:
                        stale += 1
                        if stale >= patience:
                            log.info("[flow_image_cnn] early stop at epoch %d "
                                     "(best val_acc %.4f)", epoch, best)
                            break
            self.history_.append(row)
            if epoch % 10 == 0 or epoch == int(p["epochs"]) - 1:
                log.info("[flow_image_cnn] epoch %d/%d %s", epoch + 1, p["epochs"],
                         " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _forward_batched(self, X, device, batch: int = 256):
        torch = _torch()
        outs = []
        for i in range(0, len(X), batch):
            chunk = X[i:i + batch]
            if not chunk.is_cuda and device.type == "cuda":
                chunk = chunk.to(device)
            outs.append(self.net(chunk))
        return torch.cat(outs) if outs else torch.zeros(0, self.n_classes_, device=device)

    def _predict_proba(self, X) -> np.ndarray:
        torch = _torch()
        device = self._resolve_device()
        self.net.eval().to(device)
        Xt = torch.from_numpy(self._as_nchw(X))
        with torch.no_grad():
            logits = self._forward_batched(Xt.to(device), device)
            return torch.softmax(logits, dim=1).cpu().numpy()

    # -- persistence -------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        if self.net is None:
            return {}
        return {
            "state_dict": {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
            "in_channels": self._in_channels,
            "size": self._size,
            "history": self.history_,
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        torch = _torch()
        self._in_channels = state["in_channels"]
        self._size = state["size"]
        self.net = self._build(self.n_classes_, state["in_channels"], state["size"])
        self.net.load_state_dict(
            {k: torch.from_numpy(v) for k, v in state["state_dict"].items()}
        )
        self.history_ = state.get("history", [])

    _in_channels: int = 1
    _size: int = 224


# ---------------------------------------------------------------------------------------
# Recommended variant (decision D1a, 2026-09-19)
# ---------------------------------------------------------------------------------------
from countdown.models.deep_member import DeepArrayMember  # noqa: E402


@register("flow_image_cnn")
class FlowPicCNN(DeepArrayMember):
    """The recommended image member: **FlowPic 3-channel input, the same small CNN**.

    What changed against ``flow_image_cnn_baseline`` is the *representation*, not the
    backbone: a log-density FlowPic histogram with direction and byte-volume channels
    instead of a binary scatter.  That is where the measured gain is -- +0.066 macro-F1
    on VPN traffic type and +0.045 on Tor traffic type over five seeds
    (``phases/phase4-models/MODEL-image.md``) -- while ResNet-18 at 100x the parameters
    was never verified across seeds.  The representation is an extractor setting, so an
    experiment selects it with ``feature_params: {construction: flowpic, channels: 3}``;
    :attr:`recommended_feature_params` records that pairing in code.

    Training goes through ``training/deep.py`` (decision D3): AdamW, warmup + linear
    decay, best-validation-epoch restore when a val fold is given, no class weights (D5).
    """

    input_type = "flow_image"
    expects_ndim = 4
    supports = "*"
    recommended_feature_params = {"construction": "flowpic", "channels": 3}
    deep_defaults = {
        "epochs": 60, "batch_size": 32, "eval_batch_size": 256, "learning_rate": 1e-3,
        "weight_decay": 0.01, "warmup_ratio": 0.1, "patience": 0, "amp": True, "num_workers": 4,
    }

    def __init__(self, label_space=None, **params: Any) -> None:
        params.pop("gpu_resident_gb", None)      # a baseline-loop knob the drivers pass
        super().__init__(label_space=label_space, **params)
        self.params.setdefault("dropout", (0.25, 0.5))

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = FlowImageCNN._as_nchw(X)
        if X.shape[1] == 1:
            log.warning("[flow_image_cnn] got a 1-channel image; the recommended input is "
                        "%s", self.recommended_feature_params)
        return X

    def _build(self, n_classes: int, sample_shape: tuple[int, ...]):
        return build_network(n_classes, in_channels=sample_shape[0],
                             dropout=tuple(self.params["dropout"]), size=sample_shape[-1])
