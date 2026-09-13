"""Shared torch training loop for every Phase-4 deep member.

One loop, so the byte transformer, the sequence CNN, the flow-image CNN and the GNN are
compared on identical optimisation rather than on whose training script was tuned harder.
The members differ in :meth:`forward` and nothing else.

Deliberately small: the ensemble contract is ``predict_proba``, so this owns epochs, the
LR schedule, early stopping and checkpointing, and leaves batching shape to the member via
a plain ``torch.utils.data.Dataset``.

Early stopping selects on **validation macro-F1**, not loss, because every downstream table
in this project is macro-F1 and the two disagree badly on a 120-class corpus with a long
tail -- loss keeps improving on the head classes after macro-F1 has peaked.
"""

from __future__ import annotations

import copy
import signal
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from countdown.config import get_logger

log = get_logger(__name__)


@dataclass
class DeepConfig:
    """Optimisation settings.  Defaults are the ones a member should override, not adopt."""

    epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    #: Names whose parameters are excluded from weight decay.  ``gamma``/``beta`` are the
    #: UER LayerNorm parameter names; ``weight``/``bias`` cover torch-native modules.
    no_decay: tuple[str, ...] = ("bias", "gamma", "beta", "LayerNorm.weight")
    class_weighted_loss: bool = False
    #: Stop after this many epochs with no val macro-F1 improvement.  ``0`` disables.
    patience: int = 0
    seed: int = 42
    device: str | None = None
    #: Where to mirror the best-so-far weights.  A multi-hour fine-tune that only keeps
    #: its best epoch in RAM loses everything to one OOM or one stray Ctrl-C.
    checkpoint_path: str | None = None
    #: Where to mirror the *full* training state -- weights, optimiser, LR schedule, AMP
    #: scaler and RNG -- so an interrupted run continues instead of restarting.  Distinct
    #: from :attr:`checkpoint_path`, which keeps only the best-so-far weights for the
    #: downstream tables.  ``None`` with ``checkpoint_path`` set derives
    #: ``<checkpoint_path>.resume``.
    resume_path: str | None = None
    #: Load that state at the start of :meth:`DeepTrainer.fit` and continue from it.  A
    #: missing file is not an error, so one unchanged command serves both the first launch
    #: and every restart after it.
    resume: bool = False
    amp: bool = True
    num_workers: int = 4
    log_every: int = 200
    #: Bias-correction in AdamW.  UER fine-tunes with ``correct_bias=False`` (BertAdam
    #: behaviour); torch's AdamW has no such switch, so a faithful replication needs the
    #: custom step in :func:`_build_optimizer`.
    correct_bias: bool = True


def resolve_device(requested: str | None = None) -> "Any":
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _no_bias_correction_adamw(groups, lr: float, betas=(0.9, 0.999), eps: float = 1e-6):
    """``torch.optim.AdamW`` with UER's ``correct_bias=False`` behaviour.

    UER fine-tunes with BertAdam semantics: the update is ``lr * m / (sqrt(v) + eps)`` with
    no bias-correction term.  torch always applies one, scaling each step by
    ``sqrt(1 - b2^t) / (1 - b1^t)``.  Rather than reimplement the update (and its decoupled
    weight decay, and its foreach/fused kernels), this divides the group learning rate by
    exactly that factor for the duration of the step, so the two cancel.

    One residual difference, too small to chase: torch forms the denominator as
    ``sqrt(v)/sqrt(bc2) + eps`` where UER uses ``sqrt(v) + eps``, which leaves the epsilon
    scaled by ``sqrt(bc2)``.  With ``eps=1e-6`` and ``bc2 -> 1`` within a few hundred steps
    this is far below the noise floor of a fine-tuning run.
    """
    import torch

    class _NoBiasCorrectionAdamW(torch.optim.AdamW):
        def _next_step(self, group) -> int:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "step" in st:
                    v = st["step"]
                    return int(v.item() if hasattr(v, "item") else v) + 1
            return 1

        @torch.no_grad()
        def step(self, closure=None):
            saved = [g["lr"] for g in self.param_groups]
            for g in self.param_groups:
                t = self._next_step(g)
                b1, b2 = g["betas"]
                g["lr"] = g["lr"] * (1.0 - b1**t) / (1.0 - b2**t) ** 0.5
            try:
                return super().step(closure)
            finally:
                for g, lr0 in zip(self.param_groups, saved):
                    g["lr"] = lr0

    return _NoBiasCorrectionAdamW(groups, lr=lr, betas=betas, eps=eps)


def _build_optimizer(model, cfg: DeepConfig):
    import torch

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(nd in n for nd in cfg.no_decay) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.correct_bias:
        return torch.optim.AdamW(groups, lr=cfg.learning_rate, eps=1e-6)
    return _no_bias_correction_adamw(groups, lr=cfg.learning_rate)


def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class FitResult:
    """What the loop learned, for the run record."""

    best_epoch: int
    best_val_macro_f1: float
    history: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0
    n_params: int = 0
    n_trainable: int = 0


class DeepTrainer:
    """Fit a ``nn.Module`` that maps a batch dict to logits.

    ``forward_fn(model, batch) -> logits`` keeps the loop ignorant of how a member names
    its tensors; the byte transformer needs ``src``/``seg``, a CNN needs one ``x``.
    """

    def __init__(self, cfg: DeepConfig | None = None) -> None:
        self.cfg = cfg or DeepConfig()

    def fit(
        self,
        model,
        train_ds,
        n_classes: int,
        val_ds=None,
        forward_fn: Callable[[Any, dict], Any] | None = None,
        class_weights: np.ndarray | None = None,
    ) -> FitResult:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        cfg = self.cfg
        set_seed(cfg.seed)
        device = resolve_device(cfg.device)
        model.to(device)
        forward_fn = forward_fn or (lambda m, b: m(**{k: v for k, v in b.items() if k != "y"}))

        train_dl = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=cfg.num_workers > 0,
        )
        total_steps = max(1, len(train_dl) * cfg.epochs)
        optimizer = _build_optimizer(model, cfg)
        scheduler = _linear_warmup_decay(optimizer, int(total_steps * cfg.warmup_ratio), total_steps)

        weight = None
        if cfg.class_weighted_loss and class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        use_amp = cfg.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        n_params = sum(p.numel() for p in model.parameters())
        result = FitResult(
            best_epoch=-1, best_val_macro_f1=-1.0, n_params=n_params,
            n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        best_state, stale = None, 0
        start_epoch, prior_seconds = 0, 0.0

        resume_file = _resume_path(cfg)
        fingerprint = _fingerprint(cfg, total_steps, n_params)
        if cfg.resume:
            state = _load_resume(resume_file, fingerprint)
            if state is not None:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                scaler.load_state_dict(state["scaler"])
                _set_rng_state(state["rng"])
                best_state = state["best_state"]
                stale = state["stale"]
                start_epoch = state["next_epoch"]
                prior_seconds = state["train_seconds"]
                result.history = list(state["history"])
                result.best_epoch = state["best_epoch"]
                result.best_val_macro_f1 = state["best_val_macro_f1"]
                log.info(
                    "resumed %s at epoch %d/%d (best val macro-F1 %.4f so far)",
                    resume_file, start_epoch + 1, cfg.epochs, result.best_val_macro_f1,
                )

        # Continues the clock across restarts, so train_seconds stays the cost of the whole
        # run rather than of its last segment.
        t0 = time.time() - prior_seconds

        # Stop at an epoch boundary rather than wherever the signal happens to land: what is
        # on disk is then always a coherent {weights, optimiser, schedule, scaler, RNG}
        # tuple.  A host power guard sends SIGTERM minutes before poweroff precisely so this
        # can happen; SIGINT gives Ctrl-C the same behaviour.
        interrupted: dict[str, int] = {}

        def _on_signal(signum, _frame) -> None:
            if not interrupted:
                interrupted["signum"] = signum
                log.warning(
                    "caught %s -- stopping at the next epoch boundary",
                    signal.Signals(signum).name,
                )

        # signal.signal is main-thread only; a member fitted from a worker thread simply
        # keeps the old non-interruptible behaviour rather than raising here.
        prev_handlers: dict[Any, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                prev_handlers[sig] = signal.signal(sig, _on_signal)

        try:
            for epoch in range(start_epoch, cfg.epochs):
                model.train()
                running, seen = 0.0, 0
                for step, batch in enumerate(train_dl):
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    y = batch["y"]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        loss = loss_fn(forward_fn(model, batch), y)
                    scaler.scale(loss).backward()
                    if cfg.max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    running += loss.item() * len(y)
                    seen += len(y)
                    if cfg.log_every and step and step % cfg.log_every == 0:
                        log.info(
                            "epoch %d/%d step %d/%d loss %.4f lr %.2e",
                            epoch + 1, cfg.epochs, step, len(train_dl),
                            running / max(1, seen), scheduler.get_last_lr()[0],
                        )
                    if interrupted:
                        break

                if interrupted:
                    have = resume_file is not None and resume_file.exists()
                    log.warning(
                        "stopped inside epoch %d; the partial epoch is discarded and a rerun "
                        "with resume=True %s",
                        epoch + 1,
                        f"replays it from batch 0 ({resume_file})" if have
                        else "starts from epoch 1 (no resume state written yet)",
                    )
                    raise SystemExit(128 + interrupted["signum"])

                row: dict[str, float] = {"epoch": epoch + 1, "train_loss": running / max(1, seen)}
                if val_ds is not None:
                    from countdown.eval.metrics import compute_metrics

                    proba = self.predict_proba(model, val_ds, forward_fn=forward_fn)
                    y_true = _labels_of(val_ds)
                    m = compute_metrics(y_true, proba.argmax(1), n_classes)
                    row["val_macro_f1"] = m["macro_f1"]
                    row["val_accuracy"] = m["accuracy"]
                    if m["macro_f1"] > result.best_val_macro_f1:
                        result.best_val_macro_f1 = m["macro_f1"]
                        result.best_epoch = epoch + 1
                        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                        if cfg.checkpoint_path:
                            _save_checkpoint(cfg.checkpoint_path, best_state, epoch + 1, m["macro_f1"])
                        stale = 0
                    else:
                        stale += 1
                result.history.append(row)
                log.info("epoch %d done: %s", epoch + 1, {k: round(v, 4) for k, v in row.items()})

                if resume_file is not None:
                    _save_resume(resume_file, {
                        "fingerprint": fingerprint,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "rng": _rng_state(),
                        "best_state": best_state,
                        "best_epoch": result.best_epoch,
                        "best_val_macro_f1": result.best_val_macro_f1,
                        "stale": stale,
                        "history": result.history,
                        "next_epoch": epoch + 1,
                        "train_seconds": time.time() - t0,
                    })

                # Signalled during validation or the save: that epoch is safely on disk, so
                # this is the cheapest possible place to stop.
                if interrupted:
                    log.warning(
                        "stopped cleanly after epoch %d; rerun with resume=True to continue "
                        "from epoch %d", epoch + 1, epoch + 2,
                    )
                    raise SystemExit(128 + interrupted["signum"])

                if cfg.patience and stale >= cfg.patience:
                    log.info("early stop: %d epochs without val macro-F1 improvement", stale)
                    break
        finally:
            for sig, handler in prev_handlers.items():
                signal.signal(sig, handler)

        # Reaching here means the run finished on its own terms, so the resume state is dead
        # weight (it is the largest artefact the loop writes).  The best weights, which are
        # what downstream actually consumes, stay in checkpoint_path and in the returned model.
        if resume_file is not None and resume_file.exists():
            resume_file.unlink()
            log.info("run complete; removed resume state %s", resume_file)

        if best_state is not None:
            model.load_state_dict(best_state)
            log.info("restored epoch %d (val macro-F1 %.4f)", result.best_epoch, result.best_val_macro_f1)
        result.train_seconds = time.time() - t0
        return result

    def predict_proba(self, model, ds, forward_fn: Callable | None = None) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        cfg = self.cfg
        device = resolve_device(cfg.device)
        model.to(device).eval()
        forward_fn = forward_fn or (lambda m, b: m(**{k: v for k, v in b.items() if k != "y"}))
        dl = DataLoader(
            ds, batch_size=cfg.eval_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        )
        out = []
        use_amp = cfg.amp and device.type == "cuda"
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = forward_fn(model, batch)
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)


def _save_checkpoint(path: str, state: dict, epoch: int, val_macro_f1: float) -> None:
    """Atomic write, so a crash mid-save cannot leave a truncated checkpoint behind."""
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save({"state_dict": state, "epoch": epoch, "val_macro_f1": val_macro_f1}, tmp)
    tmp.replace(p)
    log.info("checkpointed epoch %d (val macro-F1 %.4f) -> %s", epoch, val_macro_f1, p)


def _resume_path(cfg: DeepConfig) -> Path | None:
    """Where the full training state lives.

    Derived from ``checkpoint_path`` when not given explicitly, so a caller that already
    asked for checkpointing gets resume without a second setting to remember.
    """
    if cfg.resume_path:
        return Path(cfg.resume_path)
    if cfg.checkpoint_path:
        q = Path(cfg.checkpoint_path)
        return q.with_name(q.name + ".resume")
    return None


def _fingerprint(cfg: DeepConfig, total_steps: int, n_params: int) -> dict[str, Any]:
    """The settings that change what a saved step count or moment estimate *means*.

    Pouring one configuration's optimiser state into another's is silent corruption: the
    LR schedule would be at the wrong point and Adam's moments would describe a different
    loss surface.  Cheaper to refuse.
    """
    return {
        "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio, "seed": cfg.seed,
        "correct_bias": cfg.correct_bias, "class_weighted_loss": cfg.class_weighted_loss,
        "total_steps": total_steps, "n_params": n_params,
    }


def _rng_state() -> dict[str, Any]:
    """Capture every generator the loop draws from, so a resumed run sees the same shuffle
    order and dropout masks it would have seen uninterrupted."""
    import random

    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    import random

    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_resume(path: Path, payload: dict[str, Any]) -> None:
    """Atomic, for the same reason as :func:`_save_checkpoint`, and more so: a power cut
    during this write must not also destroy the previous epoch's good state."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    log.info("resume state saved through epoch %d -> %s", payload["next_epoch"], path)


def _load_resume(path: Path | None, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    """``None`` when there is nothing to resume, so one unchanged command serves both the
    first launch and every restart after it.  A fingerprint mismatch raises instead: a
    silent restart would throw away hours without saying so.
    """
    import torch

    if path is None:
        log.warning(
            "resume requested but neither resume_path nor checkpoint_path is set; "
            "starting from epoch 1"
        )
        return None
    if not path.exists():
        log.info("no resume state at %s; starting from epoch 1", path)
        return None
    # weights_only=False: the payload carries RNG state (tuples, numpy arrays), not only
    # tensors.  This file is written by this loop into its own run directory.
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved = state.get("fingerprint", {})
    if saved != fingerprint:
        diff = {
            k: (saved.get(k), fingerprint.get(k))
            for k in set(saved) | set(fingerprint)
            if saved.get(k) != fingerprint.get(k)
        }
        raise ValueError(
            f"resume state at {path} was written by a different configuration "
            f"(saved vs current: {diff}); delete it to start fresh"
        )
    return state


def _labels_of(ds) -> np.ndarray:
    """Labels without a full pass over the loader, when the dataset can offer them."""
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels())
    if hasattr(ds, "y"):
        return np.asarray(ds.y)
    return np.asarray([int(ds[i]["y"]) for i in range(len(ds))])
