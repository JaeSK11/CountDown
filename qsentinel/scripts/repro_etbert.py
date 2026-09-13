#!/usr/bin/env python
"""Replicate ET-BERT's CSTNET-TLS 1.3 result and record how close we land.

Runs the authors' fine-tuning configuration against their own released corpus and split,
so the only thing that varies is our re-implementation.  Reported targets (Lin et al.,
WWW 2022, Table 3):

    ET-BERT(packet)   AC 0.9737   PR 0.9742   RC 0.9742   F1 0.9741
    ET-BERT(flow)     AC 0.9510   PR 0.9460   RC 0.9419   F1 0.9426

Read the flow-level row against ``audit_endpoint_leakage``: that corpus still carries
Ethernet/IP/TCP headers, and a server-IP lookup table alone reaches 0.80 accuracy on it.

    python scripts/repro_etbert.py --level packet
    python scripts/repro_etbert.py --level packet --no-pretrained   # ablation row
    python scripts/repro_etbert.py --level packet --limit 20000 --epochs 1   # smoke test
    python scripts/repro_etbert.py --level packet --resume   # continue after an interruption
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qsentinel.config import REPO_ROOT, get_logger
from qsentinel.data.etbert_corpus import audit_endpoint_leakage, load_corpus
from qsentinel.eval.metrics import compute_metrics
from qsentinel.features.payload_bytes import ETBertTokenizer
from qsentinel.models.byte_net import DEFAULT_PRETRAINED, ETBertBaseline
from qsentinel.schema import LabelSpace

log = get_logger("repro_etbert")

PAPER = {
    "packet": {"accuracy": 0.9737, "precision": 0.9742, "recall": 0.9742, "macro_f1": 0.9741},
    "flow": {"accuracy": 0.9510, "precision": 0.9460, "recall": 0.9419, "macro_f1": 0.9426},
}


def tokenize_cached(texts, tokenizer, seq_length, cache: Path):
    """Tokenising 465k samples takes minutes; the result is deterministic, so cache it."""
    if cache.exists():
        z = np.load(cache)
        if len(z["src"]) == len(texts) and z["src"].shape[1] == seq_length:
            log.info("tokens from cache %s", cache.name)
            return z["src"], z["seg"]
    t0 = time.time()
    src, seg = tokenizer.encode_batch(list(texts), seq_length)
    log.info("tokenised %d samples in %.1fs -> %s", len(texts), time.time() - t0, cache.name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, src=src, seg=seg)
    return src, seg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=["packet", "flow"], default="packet")
    ap.add_argument("--seq-length", type=int, default=128, help="ET-BERT's fine-tuning default")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: 2e-5 packet, 6e-5 flow")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="truncate every split (smoke tests)")
    ap.add_argument("--no-pretrained", action="store_true", help="the w/o-pre-training ablation")
    ap.add_argument("--no-amp", action="store_true", help="fp32, as the authors ran")
    ap.add_argument("--correct-bias", action="store_true", help="torch AdamW instead of UER's")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last completed epoch; safe "
                         "to pass on the first launch too (there is simply nothing to load)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "etbert-repro")
    args = ap.parse_args()

    lr = args.lr if args.lr is not None else (2e-5 if args.level == "packet" else 6e-5)
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(args.level)

    log.info("endpoint-leakage audit on this corpus:")
    audit = audit_endpoint_leakage(corpus.train, corpus.test)
    for k, v in audit.items():
        log.info("  %-22s %.4f", k, v)

    tok = ETBertTokenizer()
    cache_dir = REPO_ROOT / "cache" / "etbert_tokens"
    splits = {}
    for name in ("train", "valid", "test"):
        sp = corpus.split(name)
        texts, y = sp.texts, sp.y
        if args.limit:
            texts, y = texts[: args.limit], y[: args.limit]
        src, seg = tokenize_cached(
            texts, tok, args.seq_length,
            cache_dir / f"{args.level}_{name}_L{args.seq_length}_{len(texts)}.npz",
        )
        splits[name] = (np.stack([src, seg], axis=1), y)

    space = LabelSpace(target="app", names=tuple(str(i) for i in range(corpus.n_classes)))
    model = ETBertBaseline(
        label_space=space,
        seq_length=args.seq_length,
        dropout=args.dropout,
        pretrained_path=None if args.no_pretrained else DEFAULT_PRETRAINED,
        deep={
            "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": lr,
            "warmup_ratio": 0.1, "weight_decay": 0.01, "seed": args.seed,
            "amp": not args.no_amp, "correct_bias": args.correct_bias, "eval_batch_size": 512,
            "checkpoint_path": str(args.out / f"{args.level}{'_scratch' if args.no_pretrained else ''}.ckpt"),
            # A full fine-tune here runs for hours, so keep enough state to survive a
            # SIGTERM from the host power guard and continue at the next epoch.
            "resume": args.resume,
        },
    )

    Xtr, ytr = splits["train"]
    model.fit(Xtr, ytr, val=splits["valid"])

    Xte, yte = splits["test"]
    t0 = time.time()
    proba = model.predict_proba(Xte)
    infer_ms = (time.time() - t0) * 1000 / len(yte)
    m = compute_metrics(yte, proba.argmax(1), corpus.n_classes)

    paper = PAPER[args.level]
    print(f"\n{'':<16}{'ours':>10}{'paper':>10}{'delta':>10}")
    for key in ("accuracy", "macro_f1"):
        print(f"{key:<16}{m[key]:>10.4f}{paper[key]:>10.4f}{m[key]-paper[key]:>+10.4f}")
    print(f"{'infer ms/sample':<16}{infer_ms:>10.3f}")
    print(f"{'params (M)':<16}{model.fit_result_.n_params/1e6:>10.1f}")
    print(f"{'train minutes':<16}{model.fit_result_.train_seconds/60:>10.1f}")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.level}{'_scratch' if args.no_pretrained else ''}"
    record = {
        "config": {**vars(args), "out": str(args.out), "learning_rate": lr},
        "paper": paper,
        "ours": {k: m[k] for k in ("n", "accuracy", "macro_f1", "macro_f1_all_classes", "weighted_f1")},
        "delta_macro_f1": m["macro_f1"] - paper["macro_f1"],
        "endpoint_leakage_audit": audit,
        "infer_ms_per_sample": infer_ms,
        "n_params": model.fit_result_.n_params,
        "train_seconds": model.fit_result_.train_seconds,
        "best_epoch": model.fit_result_.best_epoch,
        "history": model.fit_result_.history,
    }
    (args.out / f"{tag}.json").write_text(json.dumps(record, indent=2, default=str))
    (args.out / f"{tag}_metrics.json").write_text(json.dumps(m, indent=2))
    log.info("wrote %s", args.out / f"{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
