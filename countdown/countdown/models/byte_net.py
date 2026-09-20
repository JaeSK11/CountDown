"""The raw-byte ensemble member.  Baseline: ET-BERT (Lin et al., WWW 2022).

The baseline is a *replication*, so the architecture is transcribed from the authors'
released UER-py code (``reference/ET-BERT/uer/``) rather than assembled from the paper's
prose or swapped for a HuggingFace ``BertModel``.  Three details make that necessary --
each one silently degrades the released checkpoint if you get it wrong, and none of them
is visible in the paper:

1. **UER's LayerNorm is not torch's.**  It normalises by the *sample* standard deviation
   and divides by ``std + eps``, where ``nn.LayerNorm`` uses the biased variance and
   ``sqrt(var + eps)``.  With 768 features the gap is small per layer and compounds over
   12 of them.  :class:`UERLayerNorm` reproduces theirs, parameter names (``gamma`` /
   ``beta``) included, so the checkpoint loads by name.
2. **No trailing ``[SEP]``.**  UER's single-sentence classification path builds
   ``[CLS] + tokens`` and stops.  Appending ``[SEP]`` -- which nearly every other BERT
   pipeline does -- shifts every position embedding by one relative to pre-training.
3. **The attention mask comes from the segment ids**, via ``seg > 0``, not from a separate
   mask tensor.  Padding must therefore carry ``seg = 0``.

Named deviations from the authors' run, in the spirit of ``paper_baselines.py``:

* **Optimiser.**  They use UER's own ``AdamW(correct_bias=False)``, i.e. BertAdam without
  bias correction.  ``DeepConfig.correct_bias=False`` reproduces it by folding the
  correction factor out of the step size; set ``correct_bias=True`` for torch-standard
  behaviour.  Both are offered because the difference is a real fidelity axis.
* **Mixed precision.**  Their runs are fp32 on V100S.  We default to AMP on an RTX 3090
  because fp32 would put the 10-epoch packet-level run out of reach in wall-clock terms.
  ``DeepConfig.amp=False`` restores fp32 for a fidelity check.
* **Model selection.**  They train a fixed 10 epochs and report the final model.  We keep
  10 epochs but restore the best validation macro-F1 checkpoint, which is standard and can
  only help the baseline.

Recommended upgrade (``byte_net``) is deferred to its own change -- see
``phases/phase4-models/MODEL-bytes.md``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from countdown.config import REPO_ROOT, get_logger
from countdown.models.base import BaseModel, register
from countdown.training.deep import DeepConfig, DeepTrainer

log = get_logger(__name__)

DEFAULT_PRETRAINED = REPO_ROOT / "reference/pretrained/et-bert/pretrained_model.bin"
DEFAULT_VOCAB = REPO_ROOT / "reference/pretrained/et-bert/encryptd_vocab.txt"

#: ``reference/ET-BERT/bert_base_config.json`` verbatim.
BERT_BASE = {
    "emb_size": 768,
    "hidden_size": 768,
    "feedforward_size": 3072,
    "heads_num": 12,
    "layers_num": 12,
    "max_seq_length": 512,
    "vocab_size": 60005,
}


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "byte_net_baseline needs torch. Install it with the cu121 wheel: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from e
    import torch

    return torch


# ----------------------------------------------------------------------------------------
# Architecture (transcribed from reference/ET-BERT/uer/)
# ----------------------------------------------------------------------------------------


def _build_modules():
    """Define the nn.Modules lazily so importing the registry never imports torch."""
    torch = _torch()
    import torch.nn as nn

    class UERLayerNorm(nn.Module):
        """``uer/layers/layer_norm.py``: sample std, and ``/(std + eps)`` not ``/sqrt(var+eps)``."""

        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta

    class MultiHeadedAttention(nn.Module):
        def __init__(self, hidden_size: int, heads_num: int, dropout: float) -> None:
            super().__init__()
            self.heads_num = heads_num
            self.per_head_size = hidden_size // heads_num
            # [0] query, [1] key, [2] value -- the order UER zips them in.
            self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(3)]
            )
            self.dropout = nn.Dropout(dropout)
            self.final_linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, hidden, mask):
            b, t, _ = hidden.size()
            h, d = self.heads_num, self.per_head_size
            q, k, v = (
                lin(hidden).view(b, -1, h, d).transpose(1, 2) for lin in self.linear_layers
            )
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(d))
            scores = scores + mask
            probs = self.dropout(torch.softmax(scores, dim=-1))
            out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, t, h * d)
            return self.final_linear(out)

    class PositionwiseFeedForward(nn.Module):
        def __init__(self, hidden_size: int, feedforward_size: int) -> None:
            super().__init__()
            self.linear_1 = nn.Linear(hidden_size, feedforward_size)
            self.linear_2 = nn.Linear(feedforward_size, hidden_size)

        def forward(self, x):
            # UER's gelu is the exact erf form, which is torch's default F.gelu.
            return self.linear_2(torch.nn.functional.gelu(self.linear_1(x)))

    class TransformerLayer(nn.Module):
        """Post-LayerNorm block, matching ``layernorm_positioning="post"``."""

        def __init__(self, cfg: dict, dropout: float) -> None:
            super().__init__()
            self.self_attn = MultiHeadedAttention(cfg["hidden_size"], cfg["heads_num"], dropout)
            self.dropout_1 = nn.Dropout(dropout)
            self.feed_forward = PositionwiseFeedForward(
                cfg["hidden_size"], cfg["feedforward_size"]
            )
            self.dropout_2 = nn.Dropout(dropout)
            self.layer_norm_1 = UERLayerNorm(cfg["hidden_size"])
            self.layer_norm_2 = UERLayerNorm(cfg["hidden_size"])

        def forward(self, hidden, mask):
            inter = self.dropout_1(self.self_attn(hidden, mask))
            inter = self.layer_norm_1(inter + hidden)
            out = self.dropout_2(self.feed_forward(inter))
            return self.layer_norm_2(out + inter)

    class ETBertEncoder(nn.Module):
        """Embedding + 12 transformer blocks + the UER classification head.

        Parameter names mirror the checkpoint (``embedding.*``, ``encoder.transformer.N.*``)
        so ``load_state_dict`` needs no key remapping and a mismatch is loud.
        """

        def __init__(self, cfg: dict, n_classes: int, dropout: float = 0.5) -> None:
            super().__init__()
            hs = cfg["hidden_size"]
            self.embedding = nn.Module()
            self.embedding.word_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_size"])
            self.embedding.position_embedding = nn.Embedding(cfg["max_seq_length"], cfg["emb_size"])
            self.embedding.segment_embedding = nn.Embedding(3, cfg["emb_size"])
            self.embedding.layer_norm = UERLayerNorm(cfg["emb_size"])
            self.embedding_dropout = nn.Dropout(dropout)

            self.encoder = nn.Module()
            self.encoder.transformer = nn.ModuleList(
                [TransformerLayer(cfg, dropout) for _ in range(cfg["layers_num"])]
            )
            # Randomly initialised at fine-tune; not present in the pre-trained checkpoint.
            self.output_layer_1 = nn.Linear(hs, hs)
            self.output_layer_2 = nn.Linear(hs, n_classes)

        def forward(self, src, seg):
            emb = (
                self.embedding.word_embedding(src)
                + self.embedding.position_embedding(
                    torch.arange(src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
                )
                + self.embedding.segment_embedding(seg)
            )
            hidden = self.embedding_dropout(self.embedding.layer_norm(emb))

            # "fully_visible": every real token attends to every real token.
            mask = (seg > 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1).to(hidden.dtype)
            mask = (1.0 - mask) * -10000.0

            for layer in self.encoder.transformer:
                hidden = layer(hidden, mask)
            pooled = torch.tanh(self.output_layer_1(hidden[:, 0, :]))
            return self.output_layer_2(pooled)

    return ETBertEncoder


class TokenDataset:
    """``(src, seg, y)`` triples straight from pre-tokenised int arrays."""

    def __init__(self, src: np.ndarray, seg: np.ndarray, y: np.ndarray | None = None) -> None:
        self.src = np.ascontiguousarray(src, dtype=np.int64)
        self.seg = np.ascontiguousarray(seg, dtype=np.int64)
        self.y = None if y is None else np.ascontiguousarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.src)

    def labels(self) -> np.ndarray:
        return self.y if self.y is not None else np.zeros(len(self), dtype=np.int64)

    def __getitem__(self, i: int) -> dict:
        torch = _torch()
        item = {
            "src": torch.from_numpy(self.src[i]),
            "seg": torch.from_numpy(self.seg[i]),
        }
        item["y"] = torch.tensor(0 if self.y is None else int(self.y[i]), dtype=torch.long)
        return item


@register("byte_net_baseline")
class ETBertBaseline(BaseModel):
    """ET-BERT: pre-trained BERT over bi-gram datagram tokens, fine-tuned per task.

    ``X`` is ``(N, seq_length)`` of token ids as produced by
    :class:`~countdown.features.payload_bytes.ETBertTokenizer`, or ``(N, 2, seq_length)``
    stacking ``src`` and ``seg``.  When only ``src`` is given, ``seg`` is derived as
    ``src != pad_id``, which is exact for right-padded sequences.

    ``pretrained_path=None`` skips checkpoint loading and trains from scratch -- that is
    the paper's own "w/o pre-training" ablation row, not a broken configuration.
    """

    input_type = "bytes"
    supports = "*"
    #: ``(N, 2, L)`` stacking token ids and segment ids.  ``(N, L)`` ids alone are also
    #: accepted by ``_split_src_seg`` -- the harness rank check is the stricter contract.
    expects_ndim = 3

    def __init__(
        self,
        label_space=None,
        seq_length: int = 128,
        dropout: float = 0.5,
        pretrained_path: str | Path | None = DEFAULT_PRETRAINED,
        pad_id: int = 0,
        deep: dict[str, Any] | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            label_space=label_space, seq_length=seq_length, dropout=dropout,
            pretrained_path=str(pretrained_path) if pretrained_path else None,
            pad_id=pad_id, deep=deep or {}, **params,
        )
        self.seq_length = int(seq_length)
        self.dropout = float(dropout)
        self.pretrained_path = Path(pretrained_path) if pretrained_path else None
        self.pad_id = int(pad_id)
        self.deep_cfg = DeepConfig(**(deep or {}))
        self._net = None
        self.fit_result_ = None

    # -- input plumbing ----------------------------------------------------------------
    def _split_src_seg(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        if X.ndim == 3 and X.shape[1] == 2:
            return X[:, 0, :], X[:, 1, :]
        if X.ndim != 2:
            raise ValueError(f"expected (N, L) token ids or (N, 2, L) src/seg, got {X.shape}")
        return X, (X != self.pad_id).astype(np.int64)

    def _build(self, n_classes: int):
        encoder_cls = _build_modules()
        cfg = dict(BERT_BASE)
        if self.seq_length > cfg["max_seq_length"]:
            raise ValueError(
                f"seq_length={self.seq_length} exceeds the pre-trained position table "
                f"({cfg['max_seq_length']})"
            )
        net = encoder_cls(cfg, n_classes=n_classes, dropout=self.dropout)
        if self.pretrained_path is not None:
            self._load_pretrained(net)
        return net

    def _load_pretrained(self, net) -> None:
        torch = _torch()

        if not self.pretrained_path.exists():
            raise FileNotFoundError(
                f"pre-trained ET-BERT checkpoint not found at {self.pretrained_path}. "
                f"Fetch it with scripts/fetch_etbert_assets.sh, or pass "
                f"pretrained_path=None to train from scratch (the paper's ablation row)."
            )
        sd = torch.load(self.pretrained_path, map_location="cpu", weights_only=True)
        # ``target.*`` is the MLM + same-origin pre-training head; it has no fine-tune role.
        sd = {k: v for k, v in sd.items() if not k.startswith("target.")}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        expected_missing = {"output_layer_1.weight", "output_layer_1.bias",
                            "output_layer_2.weight", "output_layer_2.bias"}
        surprising = set(missing) - expected_missing
        if surprising or unexpected:
            raise RuntimeError(
                f"ET-BERT checkpoint did not match the transcribed architecture. "
                f"unexpectedly missing={sorted(surprising)}, unexpected={sorted(unexpected)}"
            )
        log.info(
            "loaded ET-BERT pre-trained weights from %s (%d tensors, classifier head fresh)",
            self.pretrained_path, len(sd),
        )

    # -- BaseModel contract ------------------------------------------------------------
    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        src, seg = self._split_src_seg(X)
        self._net = self._build(self.n_classes_)

        val_ds = None
        if val is not None:
            vX, vy = val
            vsrc, vseg = self._split_src_seg(vX)
            val_ds = TokenDataset(vsrc, vseg, vy)

        class_weights = None
        if self.deep_cfg.class_weighted_loss:
            counts = np.bincount(y, minlength=self.n_classes_).astype(np.float64)
            class_weights = np.where(counts > 0, len(y) / (self.n_classes_ * np.maximum(counts, 1)), 0.0)

        trainer = DeepTrainer(self.deep_cfg)
        self.fit_result_ = trainer.fit(
            self._net, TokenDataset(src, seg, y), n_classes=self.n_classes_,
            val_ds=val_ds, forward_fn=lambda m, b: m(b["src"], b["seg"]),
            class_weights=class_weights,
        )
        log.info(
            "fit done in %.1fs, best epoch %d, val macro-F1 %.4f, %.1fM params",
            self.fit_result_.train_seconds, self.fit_result_.best_epoch,
            self.fit_result_.best_val_macro_f1, self.fit_result_.n_params / 1e6,
        )

    def _predict_proba(self, X) -> np.ndarray:
        src, seg = self._split_src_seg(X)
        trainer = DeepTrainer(self.deep_cfg)
        return trainer.predict_proba(
            self._net, TokenDataset(src, seg), forward_fn=lambda m, b: m(b["src"], b["seg"])
        )

    # -- persistence -------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        torch = _torch()
        import io

        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in self._net.state_dict().items()}, buf)
        return {"weights": buf.getvalue(), "fit_result": self.fit_result_}

    def _load_state(self, state: dict[str, Any]) -> None:
        torch = _torch()
        import io

        self._net = self._build(self.n_classes_)
        self._net.load_state_dict(torch.load(io.BytesIO(state["weights"]), map_location="cpu", weights_only=True))
        self.fit_result_ = state.get("fit_result")


# ---------------------------------------------------------------------------------------
# Recommended variant (decision D1b, 2026-09-19)
# ---------------------------------------------------------------------------------------
class FieldRandomiser:
    """Overwrite header fields that carry no semantics with random bytes.

    TrafficFormer (Zhou et al., IEEE S&P 2025) fine-tunes with *random-initialisation field
    augmentation*: TCP sequence / acknowledgement numbers start from a random value, so
    their absolute bytes say nothing about the application -- but within one capture they
    are near-constant across a flow's packets, which makes them a capture fingerprint a
    byte model will happily memorise (the SoK's "contextual overfitting").  Randomising
    them at fine-tune time removes the shortcut without touching the payload.

    In the released ``packet_5000`` corpus the TCP header starts at byte 2 with the ports
    already stripped, so seq + ack are datagram **bytes 2..9**.  ET-BERT words are
    overlapping bi-grams (word ``t`` covers bytes ``t, t+1``), so those bytes live in words
    1..9 and the replacement chain has to agree with the untouched neighbours, bytes 1 and
    10.  The work is done on *words*, not token ids, because ~12 % of bi-grams are outside
    the vocabulary and WordPiece splits them in two (``c3`` + ``##de``): two thirds of real
    packets have at least one such split inside the span, and random bytes produce them at
    the same rate.  So the prefix is decoded to words, rewritten, re-tokenised, and the
    untouched tail is re-attached, truncated or padded back to the row length.
    """

    _HEX = set("0123456789abcdef")

    def __init__(self, tokenizer, fields: tuple[tuple[int, int], ...] = ((2, 10),)) -> None:
        self.tok = tokenizer
        self.fields = tuple((int(a), int(b)) for a, b in fields)
        self.inv = {i: t for t, i in tokenizer.vocab.items()}
        self.pad_id = tokenizer.pad_id

    def _words(self, row: np.ndarray, n_words: int):
        """First ``n_words`` bi-gram words of a row and the index where the rest starts."""
        words, cur, j = [], "", 1                       # row[0] is [CLS]
        while len(words) < n_words and j < len(row):
            t = self.inv.get(int(row[j]), "")
            piece = t[2:] if t.startswith("##") else t
            if not piece or not set(piece) <= self._HEX:
                return None, j                          # [PAD] / [UNK] / not a byte word
            cur = cur + piece if t.startswith("##") else piece
            j += 1
            if len(cur) == 4:
                words.append(cur); cur = ""
            elif len(cur) > 4:
                return None, j
        return (words, j) if len(words) == n_words and not cur else (None, j)

    def __call__(self, src: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """A randomised ``(src, seg)`` copy of one token row (the input is not modified)."""
        row = np.array(src, dtype=np.int64, copy=True)
        for a, b in self.fields:
            words, j = self._words(row, b)              # words 0 .. b-1 cover bytes 0 .. b
            if words is None or a < 1:
                continue
            left, right = words[a - 1][:2], words[b - 1][2:]
            chain = [left] + [f"{v:02x}" for v in rng.integers(0, 256, size=b - a)] + [right]
            new_words = words[:a - 1] + [chain[i] + chain[i + 1] for i in range(len(chain) - 1)]
            ids: list[int] = []
            for w in new_words:
                ids.extend(self.tok.vocab.get(t, self.tok.unk_id) for t in self.tok._wordpiece(w))
            tail = row[j:]
            merged = np.concatenate(([row[0]], np.asarray(ids, dtype=np.int64), tail[tail != self.pad_id]))
            row = np.full(len(src), self.pad_id, dtype=np.int64)
            row[:min(len(merged), len(src))] = merged[:len(src)]
        return row, (row != self.pad_id).astype(np.int64)

    def apply(self, src: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        rows = [self(r, rng) for r in np.asarray(src)]
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


class AugmentedTokenDataset(TokenDataset):
    """``TokenDataset`` whose rows are field-randomised afresh on every access."""

    def __init__(self, src, seg, y, randomiser: FieldRandomiser, prob: float = 1.0) -> None:
        super().__init__(src, seg, y)
        self.randomiser, self.prob = randomiser, float(prob)

    def __getitem__(self, i: int) -> dict:
        torch = _torch()
        item = super().__getitem__(i)
        # torch's generator is seeded per worker from the run seed, so this is reproducible.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        if rng.random() < self.prob:
            src, seg = self.randomiser(self.src[i], rng)
            item["src"], item["seg"] = torch.from_numpy(src), torch.from_numpy(seg)
        return item


@register("byte_net")
class ByteNetRecommended(ETBertBaseline):
    """The recommended bytes member: the ET-BERT checkpoint + field-randomising fine-tuning.

    Decision D1b.  The measured case for keeping the checkpoint is the no-pretrain ablation
    (0.688 vs 0.908 macro-F1 on the clean packet corpus); what changes is that the fine-tune
    no longer lets the model read TCP seq/ack.  Evaluate it twice -- on the released test
    split and on the same split with seq/ack randomised (:meth:`randomise_fields`) -- because
    the second column is the one that says how much of a byte model's score was the shortcut.
    """

    def __init__(self, label_space=None, augment_fields=((2, 10),), augment_prob: float = 1.0,
                 vocab_path: str | Path | None = None, **params: Any) -> None:
        super().__init__(label_space=label_space, **params)
        self.params.update(augment_fields=[list(f) for f in augment_fields],
                           augment_prob=float(augment_prob),
                           vocab_path=str(vocab_path) if vocab_path else None)
        self._randomiser: FieldRandomiser | None = None

    @property
    def randomiser(self) -> FieldRandomiser:
        if self._randomiser is None:
            from countdown.features.payload_bytes import ETBertTokenizer

            vp = self.params.get("vocab_path")
            tok = ETBertTokenizer(vp) if vp else ETBertTokenizer()
            self._randomiser = FieldRandomiser(tok, tuple(tuple(f) for f in self.params["augment_fields"]))
        return self._randomiser

    def randomise_fields(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        """``X`` with the augmented fields randomised -- the shortcut-free evaluation input."""
        src, _ = self._split_src_seg(X)
        return np.stack(self.randomiser.apply(src, seed=seed), axis=1)

    def _fit(self, X, y, sample_weight=None, val=None) -> None:
        src, seg = self._split_src_seg(X)
        self._net = self._build(self.n_classes_)
        val_ds = None
        if val is not None:
            vsrc, vseg = self._split_src_seg(val[0])
            val_ds = TokenDataset(vsrc, vseg, val[1])
        train_ds = AugmentedTokenDataset(src, seg, y, self.randomiser, self.params["augment_prob"])
        self.fit_result_ = DeepTrainer(self.deep_cfg).fit(
            self._net, train_ds, n_classes=self.n_classes_, val_ds=val_ds,
            forward_fn=lambda m, b: m(b["src"], b["seg"]),
        )
        r = self.fit_result_
        log.info("[byte_net] fit %.1fs, best epoch %d, val macro-F1 %.4f",
                 r.train_seconds, r.best_epoch, r.best_val_macro_f1)
