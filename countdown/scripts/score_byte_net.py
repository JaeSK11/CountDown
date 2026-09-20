#!/usr/bin/env python
"""Baseline-vs-recommended scorecard for the bytes member on the clean packet corpus.

    python scripts/score_byte_net.py                 # train byte_net, score both members
    python scripts/score_byte_net.py --skip-train    # re-score from the saved checkpoints

Both members are fine-tuned from the same public ET-BERT checkpoint on ET-BERT's own
``packet_5000`` split, so the only difference is ``byte_net``'s field-randomising
augmentation (decision D1b).  Each is scored twice:

``released``   the test split as shipped;
``randomised`` the same packets with TCP seq/ack (datagram bytes 2..9) overwritten by random
               bytes -- an input on which a model cannot use those fields as a capture
               fingerprint.  The drop from the first column to the second is how much of a
               member's score was that shortcut.

The baseline is not retrained: its weights come from ``runs/etbert-repro/packet.ckpt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repro_etbert import tokenize_cached  # noqa: E402

from countdown.config import REPO_ROOT, get_logger  # noqa: E402
from countdown.data.etbert_corpus import load_corpus  # noqa: E402
from countdown.eval.metrics import compute_metrics  # noqa: E402
from countdown.features.payload_bytes import ETBertTokenizer  # noqa: E402
from countdown.models import ModelRegistry  # noqa: E402
from countdown.schema import LabelSpace  # noqa: E402

log = get_logger("score-byte-net")


def load_weights(model, ckpt: Path, n_classes: int) -> None:
    import torch

    model.pretrained_path = None                       # the checkpoint already holds everything
    model._net = model._build(n_classes)
    model._net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"])
    model.trained_classes_ = np.arange(n_classes)
    model.is_fitted = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seq-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="truncate every split (smoke tests)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--baseline-ckpt", type=Path, default=REPO_ROOT / "runs/etbert-repro/packet.ckpt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs/byte-net")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus("packet")
    tok = ETBertTokenizer()
    cache = REPO_ROOT / "cache" / "etbert_tokens"
    splits = {}
    for name in ("train", "valid", "test"):
        sp = corpus.split(name)
        texts, y = (sp.texts[: args.limit], sp.y[: args.limit]) if args.limit else (sp.texts, sp.y)
        src, seg = tokenize_cached(texts, tok, args.seq_length,
                                   cache / f"packet_{name}_L{args.seq_length}_{len(texts)}.npz")
        splits[name] = (np.stack([src, seg], axis=1), y)

    space = LabelSpace(target="app", names=tuple(str(i) for i in range(corpus.n_classes)))
    deep = {"epochs": args.epochs, "batch_size": 32, "learning_rate": 2e-5, "warmup_ratio": 0.1,
            "weight_decay": 0.01, "seed": args.seed, "amp": True, "correct_bias": False,
            "eval_batch_size": 512, "checkpoint_path": str(args.out / "packet.ckpt"), "resume": True}

    rec = ModelRegistry.create("byte_net", label_space=space, seq_length=args.seq_length, deep=deep)
    train_seconds = None
    if args.skip_train:
        load_weights(rec, args.out / "packet.ckpt", corpus.n_classes)
    else:
        t0 = time.time()
        rec.fit(*splits["train"], val=splits["valid"])
        train_seconds = time.time() - t0

    base = ModelRegistry.create("byte_net_baseline", label_space=space, seq_length=args.seq_length,
                                deep={"eval_batch_size": 512, "amp": True})
    load_weights(base, args.baseline_ckpt, corpus.n_classes)

    Xte, yte = splits["test"]
    Xrand = rec.randomise_fields(Xte, seed=0)
    changed = float((Xrand[:, 0] != Xte[:, 0]).any(axis=1).mean())
    rows = {}
    for name, model in (("byte_net_baseline", base), ("byte_net", rec)):
        rows[name] = {}
        for col, X in (("released", Xte), ("randomised", Xrand)):
            m = compute_metrics(yte, model.predict_proba(X).argmax(1), corpus.n_classes)
            rows[name][col] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}
    record = {"config": {**{k: str(v) for k, v in vars(args).items()}, "deep": deep},
              "n_test": int(len(yte)), "rows_randomised_frac": changed, "scorecard": rows,
              "train_seconds": train_seconds,
              "fit": None if rec.fit_result_ is None else {"best_epoch": rec.fit_result_.best_epoch,
                                                           "history": rec.fit_result_.history}}
    (args.out / "scorecard.json").write_text(json.dumps(record, indent=2, default=str))

    print(f"\n{'member':<20}{'released F1':>13}{'randomised F1':>15}{'drop':>8}   ({changed:.1%} of test rows randomised)")
    for name, r in rows.items():
        a, b = r["released"]["macro_f1"], r["randomised"]["macro_f1"]
        print(f"{name:<20}{a:>13.4f}{b:>15.4f}{a - b:>+8.4f}")
    log.info("wrote %s", args.out / "scorecard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
