"""Raw-byte representation: packet bytes -> ET-BERT bi-gram tokens.

Two things live here, because they are the same transformation at different entry points:

* :class:`ETBertTokenizer` -- the *exact* text pipeline ET-BERT fine-tunes on.  Bytes are
  rendered as overlapping bi-grams (``d5 d6 88`` -> ``"d5d6 d688"``), then run through
  BERT WordPiece against the released 60,005-entry vocabulary.  Reproducing the paper
  requires this to be byte-identical to theirs, so it is transcribed from
  ``reference/ET-BERT/uer/utils/tokenizers.py`` rather than reinvented.
* :class:`PayloadBytes` -- the extractor that turns *our* :class:`~qsentinel.schema.Flow`
  objects into the same token matrix, so the byte member can eventually train on
  qsentinel-parsed pcaps rather than only on ET-BERT's shipped arrays.

**Scope limit, stated plainly.**  ``Flow`` retains only the reassembled *handshake window*
of each direction (``handshake_client_bytes`` / ``handshake_server_bytes``), not every
packet's payload.  So :class:`PayloadBytes` yields the opening bytes of the conversation,
which is a legitimate byte representation but is **not** ET-BERT's BURST construction --
that needs per-packet payload retention in ``flows/extract.py``.  Until that lands, the
paper reproduction reads ET-BERT's released corpus directly (see
``qsentinel.data.etbert_corpus``) so that no tokenisation difference can be confused with
a modelling difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from qsentinel.config import REPO_ROOT, get_logger
from qsentinel.features.base import FeatureExtractor, register_extractor
from qsentinel.schema import Flow

log = get_logger(__name__)

#: Where ``scripts/fetch_etbert_assets.sh`` puts the released vocabulary.
DEFAULT_VOCAB = REPO_ROOT / "reference/pretrained/et-bert/encryptd_vocab.txt"

PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, UNK_TOKEN, MASK_TOKEN = "[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"


def bigram_hex(raw: bytes) -> str:
    """``b"\\xd5\\xd6\\x88"`` -> ``"d5d6 d688"``: overlapping 2-byte windows, stride 1.

    This is ET-BERT's ``BURST2Token`` step.  The overlap is what makes the token stream
    carry byte-adjacency information that a non-overlapping split would throw away, and it
    is why one packet of ``n`` bytes becomes ``n - 1`` tokens rather than ``n / 2``.
    """
    if len(raw) < 2:
        return ""
    h = raw.hex()
    return " ".join(h[i : i + 4] for i in range(0, (len(raw) - 1) * 2, 2))


class ETBertTokenizer:
    """BERT WordPiece over the released encrypted-traffic vocabulary.

    Deviations from ``uer.utils.tokenizers.BertTokenizer`` are none that can fire on this
    input: the ``BasicTokenizer`` lower-casing and punctuation splitting it runs first are
    no-ops on lowercase hex separated by single spaces, so whitespace splitting followed by
    greedy longest-match WordPiece is exactly equivalent here.
    """

    def __init__(self, vocab_path: str | Path = DEFAULT_VOCAB) -> None:
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(
                f"ET-BERT vocabulary not found at {path}. Fetch it with "
                f"scripts/fetch_etbert_assets.sh, or pass vocab_path=..."
            )
        self.vocab: dict[str, int] = {}
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        for special in (PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, UNK_TOKEN, MASK_TOKEN):
            if special not in self.vocab:
                raise ValueError(f"vocabulary at {path} is missing {special}")
        self.pad_id = self.vocab[PAD_TOKEN]
        self.cls_id = self.vocab[CLS_TOKEN]
        self.unk_id = self.vocab[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.vocab)

    def _wordpiece(self, word: str) -> list[str]:
        """Greedy longest-match-first, ``##`` on every piece after the first."""
        if word in self.vocab:  # the 95.8% fast path on this corpus
            return [word]
        pieces, start = [], 0
        while start < len(word):
            end = len(word)
            cur = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return [UNK_TOKEN]
            pieces.append(cur)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for word in text.split():
            out.extend(self._wordpiece(word))
        return out

    def encode(self, text: str, seq_length: int) -> tuple[list[int], list[int]]:
        """``(src, seg)`` padded to ``seq_length``, matching UER's ``read_dataset``.

        Note the missing ``[SEP]``: UER's single-sentence path prepends ``[CLS]`` and
        appends nothing.  Adding the trailing ``[SEP]`` that most BERT code would use here
        shifts every position embedding relative to pre-training, so it is left out on
        purpose.  ``seg`` is 1 on real tokens and 0 on padding, because the encoder derives
        its attention mask from ``seg > 0`` rather than from a separate mask tensor.
        """
        ids = [self.cls_id] + [self.vocab.get(t, self.unk_id) for t in self.tokenize(text)]
        ids = ids[:seq_length]
        seg = [1] * len(ids)
        pad = seq_length - len(ids)
        if pad > 0:
            ids.extend([self.pad_id] * pad)
            seg.extend([0] * pad)
        return ids, seg

    def encode_batch(
        self, texts: Sequence[str], seq_length: int, dtype: type = np.int32
    ) -> tuple[np.ndarray, np.ndarray]:
        src = np.empty((len(texts), seq_length), dtype=dtype)
        seg = np.empty((len(texts), seq_length), dtype=dtype)
        for i, t in enumerate(texts):
            s, g = self.encode(t, seq_length)
            src[i] = s
            seg[i] = g
        return src, seg


@register_extractor("payload_bytes")
class PayloadBytes(FeatureExtractor):
    """``(N, seq_length)`` ET-BERT token ids from each flow's handshake window.

    ``direction="both"`` concatenates client-then-server bytes, which keeps the request
    before the response the way a BURST pair would.  See the module docstring for why this
    is the handshake window rather than per-packet payloads.
    """

    version = 1
    input_type = "bytes"

    def __init__(
        self,
        seq_length: int = 128,
        max_bytes: int = 256,
        direction: str = "both",
        vocab_path: str | Path = DEFAULT_VOCAB,
        tokenizer: ETBertTokenizer | None = None,
        **_ignored,
    ) -> None:
        if direction not in ("client", "server", "both"):
            raise ValueError(f"direction must be client/server/both, got {direction!r}")
        self.seq_length = int(seq_length)
        self.max_bytes = int(max_bytes)
        self.direction = direction
        self.tokenizer = tokenizer or ETBertTokenizer(vocab_path)

    @property
    def feature_names(self) -> Sequence[str]:
        return [f"tok{i}" for i in range(self.seq_length)]

    def _raw(self, flow: Flow) -> bytes:
        c = flow.handshake_client_bytes or b""
        s = flow.handshake_server_bytes or b""
        raw = {"client": c, "server": s, "both": c + s}[self.direction]
        return raw[: self.max_bytes]

    def transform_one(self, flow: Flow) -> np.ndarray:
        ids, _ = self.tokenizer.encode(bigram_hex(self._raw(flow)), self.seq_length)
        return np.asarray(ids, dtype=np.int32)

    def transform(self, flows: Sequence[Flow]) -> np.ndarray:
        """Token ids only.  Use :meth:`transform_with_seg` when feeding the encoder."""
        if len(flows) == 0:
            return np.zeros((0, self.seq_length), dtype=np.int32)
        return np.stack([self.transform_one(f) for f in flows])

    def transform_with_seg(self, flows: Sequence[Flow]) -> tuple[np.ndarray, np.ndarray]:
        texts = [bigram_hex(self._raw(f)) for f in flows]
        return self.tokenizer.encode_batch(texts, self.seq_length)
