"""Reader for ET-BERT's *released* CSTNET-TLS 1.3 fine-tuning corpus.

Reproducing a paper's number means holding everything but the model constant, so the byte
baseline is verified against the authors' own tokenised inputs and their own 8:1:1 split
rather than against a re-parse of the pcaps.  Any gap that remains is then attributable to
the model or the training loop, not to a tokenisation or splitting difference.

Two corpora ship in ``data/CSTNET-TLS1.3/extracted/``:

``packet_5000``
    581,709 single packets (465,367 / 58,171 / 58,171), ~5,000 per class, 64 bi-gram
    tokens each.  Matches the paper's ``#Packet`` for CSTNET exactly.  Headers are
    stripped: the TCP header begins at byte 2 of each sample and the first TLS record at
    byte 18, so there are no MAC or IP bytes to key on.  This is ET-BERT(packet).

``flow_500``
    46,372 flows (37,097 / 4,638 / 4,637), ~500 per class, 320 bi-gram tokens each.
    Matches the paper's ``#Flow``.  This is ET-BERT(flow), and it is also available as the
    ``cstnet-tls1.3/*.tsv`` files that ``fine-tuning/run_classifier.py`` consumes directly.

.. warning::
   **The flow-level corpus contradicts the paper's stated preprocessing.**  Section 4.1.2
   says the Ethernet header, IP header and TCP ports were removed.  In ``flow_500`` and
   the TSVs they are present in 100% of samples -- both MACs, both IP addresses and both
   ports decode cleanly from the first 38 bytes.  A lookup table keyed on nothing but the
   server IP scores **0.80 accuracy** on the 120-class test split, against ET-BERT(flow)'s
   reported 0.9510.  Numbers trained on this corpus measure endpoint memorisation more
   than byte modelling; :func:`audit_endpoint_leakage` is here so that stays measured
   rather than assumed.  ``packet_5000`` is clean and is the corpus to prefer.

Labels are ET-BERT's integer ids.  The id -> domain mapping is *not* recoverable: their
``dataset_generation.py`` assigns ids in ``os.walk`` order over the 120 capture
directories, which is their filesystem's ordering, not ours.  Macro-F1 is invariant to the
permutation, so the ids are used as class names and the ambiguity is confined to here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qsentinel.config import REPO_ROOT, get_logger

log = get_logger(__name__)

DEFAULT_ROOT = REPO_ROOT / "data/CSTNET-TLS1.3/extracted"
LEVELS = {"packet": "packet_5000", "flow": "flow_500"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Split:
    """One split's tokenised text and integer labels."""

    texts: np.ndarray  # (N,) unicode, space-separated 4-hex-char bi-grams
    y: np.ndarray  # (N,) int64, ET-BERT label ids

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class Corpus:
    level: str
    n_classes: int
    train: Split
    valid: Split
    test: Split

    def split(self, name: str) -> Split:
        return getattr(self, name)


def load_corpus(level: str = "packet", root: str | Path = DEFAULT_ROOT) -> Corpus:
    """Load ``packet_5000`` or ``flow_500`` from the released ``.npy`` arrays."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
    d = Path(root) / LEVELS[level]
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found. The CSTNET-TLS 1.3 release unpacks to "
            f"data/CSTNET-TLS1.3/extracted/ -- see that folder's README.md."
        )
    parts = {}
    for s in SPLITS:
        x = np.load(d / f"x_datagram_{s}.npy", allow_pickle=True)
        y = np.load(d / f"y_{s}.npy", allow_pickle=True).astype(np.int64)
        if len(x) != len(y):
            raise ValueError(f"{d.name}/{s}: {len(x)} texts but {len(y)} labels")
        parts[s] = Split(texts=x, y=y)
    n_classes = int(max(p.y.max() for p in parts.values())) + 1
    log.info(
        "loaded %s: train=%d valid=%d test=%d, %d classes",
        LEVELS[level], len(parts["train"]), len(parts["valid"]), len(parts["test"]), n_classes,
    )
    return Corpus(level=level, n_classes=n_classes, **parts)


def load_tsv(path: str | Path) -> Split:
    """Read one of ET-BERT's ``label\\ttext_a`` fine-tuning files."""
    labels, texts = [], []
    with Path(path).open(encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        cols = {name: i for i, name in enumerate(header)}
        if "label" not in cols or "text_a" not in cols:
            raise ValueError(f"{path}: expected 'label' and 'text_a' columns, got {header}")
        for row in r:
            labels.append(int(row[cols["label"]]))
            texts.append(row[cols["text_a"]])
    return Split(texts=np.asarray(texts, dtype=object), y=np.asarray(labels, dtype=np.int64))


# ----------------------------------------------------------------------------------------
# Leakage audit
# ----------------------------------------------------------------------------------------


def _tokens_to_bytes(text: str) -> bytes:
    """Invert the overlapping bi-gram encoding: each token after the first adds one byte."""
    t = text.split()
    if not t:
        return b""
    out = bytearray((int(t[0][0:2], 16), int(t[0][2:4], 16)))
    out.extend(int(x[2:4], 16) for x in t[1:])
    return bytes(out)


def _endpoints(raw: bytes) -> dict | None:
    """Ethernet II + IPv4 + TCP/UDP endpoint fields, or ``None`` if it does not parse."""
    if len(raw) < 38 or raw[12:14] != b"\x08\x00" or (raw[14] >> 4) != 4:
        return None
    ihl = (raw[14] & 0x0F) * 4
    o = 14 + ihl
    if len(raw) < o + 4:
        return None
    ip = lambda i: ".".join(str(b) for b in raw[i : i + 4])  # noqa: E731
    mac = lambda i: ":".join(f"{b:02x}" for b in raw[i : i + 6])  # noqa: E731
    return {
        "dst_mac": mac(0), "src_mac": mac(6),
        "src_ip": ip(26), "dst_ip": ip(30), "proto": raw[23],
        "sport": int.from_bytes(raw[o : o + 2], "big"),
        "dport": int.from_bytes(raw[o + 2 : o + 4], "big"),
    }


def audit_endpoint_leakage(train: Split, test: Split, server_port: int = 443) -> dict:
    """How far does a lookup table on the server IP alone get on this corpus?

    Returns the fraction of samples that decode as Ethernet+IPv4 at all, plus the accuracy
    of a majority-vote server-IP table fitted on ``train`` and scored over all of ``test``.
    A high number means the corpus leaks endpoint identity and any model trained on it is
    partly a memorised address book.
    """
    import collections

    def key(text: str) -> str | None:
        p = _endpoints(_tokens_to_bytes(text))
        if p is None:
            return None
        if p["sport"] == server_port:
            return p["src_ip"]
        return p["dst_ip"] if p["dport"] == server_port else None

    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    parsed_train = 0
    for text, y in zip(train.texts, train.y):
        k = key(text)
        if k is not None:
            parsed_train += 1
            table[k][int(y)] += 1
    lut = {k: c.most_common(1)[0][0] for k, c in table.items()}

    hit = covered = parsed_test = 0
    for text, y in zip(test.texts, test.y):
        k = key(text)
        if k is None:
            continue
        parsed_test += 1
        if k in lut:
            covered += 1
            hit += int(lut[k] == int(y))
    n = len(test)
    return {
        "train_parsed_frac": parsed_train / max(1, len(train)),
        "test_parsed_frac": parsed_test / max(1, n),
        "distinct_server_ips": len(lut),
        "test_coverage": covered / max(1, n),
        "accuracy_all_test": hit / max(1, n),
        "accuracy_covered": hit / max(1, covered),
    }
