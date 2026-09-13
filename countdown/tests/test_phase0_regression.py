"""Phase 1 task 1.8: prove the reassembler refactor did not move a single byte of (X, y).

``tests/golden/phase0_flows.json`` was produced by running Phase 0's ``FlowExtractor``
**before** the Phase 1 refactor, over 29 captures: nine synthetic ones covering the corner
cases (idle split, server-first, retransmits, FIN teardown, IPv6, QUIC, garbage frames) and
twenty sampled from all four real datasets, together ~7,000 flows.

For each flow it pins the five-tuple, the packet count, a SHA-1 over the concatenated
``timestamps``/``sizes``/``directions`` arrays, and the ``ParseStats`` for the capture.
Those arrays are exactly the input to every Phase 0 feature extractor, so an unchanged
digest is an unchanged ``(X, y)``.

The synthetic captures ship with the golden and always run.  The real-dataset entries are
skipped when the corpus is not present, so this suite still runs on a bare checkout.

The published ``phase0_flows.json`` carries the nine synthetic captures only.  The
real-dataset entries pin five-tuples, byte counts and timestamps taken from corpora we do
not redistribute, so they stay in ``phase0_flows.full.json``, which is git-ignored and sits
beside it on a machine that has the datasets.  That fuller snapshot wins when present, so
nothing is lost locally; a clone simply covers the synthetic corner cases.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from countdown.config import default_config
from countdown.flows.extract import FlowExtractor, ParseStats

GOLDEN_DIR = Path(__file__).parent / "golden"
#: The local, un-redistributable snapshot wins when it is there; the published
#: synthetic-only one is the fallback.
GOLDEN_FILE = next(
    (GOLDEN_DIR / n for n in ("phase0_flows.full.json", "phase0_flows.json")
     if (GOLDEN_DIR / n).exists()),
    GOLDEN_DIR / "phase0_flows.json",
)
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

pytestmark = pytest.mark.skipif(not GOLDEN_FILE.exists(), reason="no golden snapshot")

_GOLDEN: dict = json.loads(GOLDEN_FILE.read_text()) if GOLDEN_FILE.exists() else {}
SYNTHETIC = sorted(k for k in _GOLDEN if k.startswith("synthetic/"))
REAL = sorted(k for k in _GOLDEN if not k.startswith("synthetic/"))


def _resolve(key: str) -> Path:
    return GOLDEN_DIR / key if key.startswith("synthetic/") else DATA_ROOT / key


def _digest(flows, stats) -> dict:
    out = []
    for f in flows:
        h = hashlib.sha1()
        h.update(f.timestamps.tobytes())
        h.update(f.sizes.tobytes())
        h.update(f.directions.tobytes())
        out.append({
            "flow_id": f.flow_id,
            "five_tuple": list(f.five_tuple),
            "n": int(f.n_packets),
            "arrays_sha1": h.hexdigest(),
            "start": round(f.start_ts, 9),
            "end": round(f.end_ts, 9),
            "meta_keys": sorted(f.meta),
        })
    return {"flows": out, "stats": stats.as_dict()}


def _check(key: str) -> None:
    path = _resolve(key)
    if not path.exists():
        pytest.skip(f"{key} not present")
    stats = ParseStats()
    flows = FlowExtractor(**default_config().flow).extract(
        path, dataset="golden", label="x", label_fields={"t": "x"}, stats=stats
    )
    got = _digest(flows, stats)
    want = _GOLDEN[key]

    assert len(got["flows"]) == len(want["flows"]), f"{key}: flow count moved"
    for i, (g, w) in enumerate(zip(got["flows"], want["flows"])):
        assert g == w, f"{key}: flow {i} differs from the pre-refactor Phase 0 output"
    assert got["stats"] == want["stats"], f"{key}: ParseStats differ"


@pytest.mark.parametrize("key", SYNTHETIC)
def test_synthetic_captures_unchanged(key):
    _check(key)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="dataset corpus not available")
@pytest.mark.parametrize("key", REAL)
def test_real_dataset_captures_unchanged(key):
    _check(key)


@pytest.mark.skipif(not REAL, reason="synthetic-only snapshot carries no dataset entries")
def test_golden_covers_every_dataset():
    """A regression suite that quietly stopped covering a dataset is worse than none.

    Only meaningful against the full snapshot: the published one is synthetic-only by
    design, so there is no dataset coverage to lose.
    """
    covered = {k.split("/")[0] for k in REAL}
    assert covered == {"CSTNET-TLS1.3", "ISCXTor2016", "ISCXVPN2016", "PostQuantumTLS"}


def test_handshake_capture_does_not_alter_the_feature_arrays(tls_handshake_pcap):
    """The Phase 1 additions must be observationally inert for Phase 0 features."""
    with_hs = FlowExtractor(capture_handshake=True).extract(tls_handshake_pcap)
    without = FlowExtractor(capture_handshake=False).extract(tls_handshake_pcap)
    assert len(with_hs) == len(without) == 1
    a, b = with_hs[0], without[0]
    assert (a.timestamps.tobytes(), a.sizes.tobytes(), a.directions.tobytes()) == (
        b.timestamps.tobytes(), b.sizes.tobytes(), b.directions.tobytes())
    assert a.five_tuple == b.five_tuple and a.meta == b.meta


def test_compat_mode_emits_no_extra_meta_keys(tls_handshake_pcap):
    """data/base.py sorts cached flows by json.dumps(meta); a new key reorders datasets."""
    f = FlowExtractor().extract(tls_handshake_pcap)[0]
    assert sorted(f.meta) == ["source_file"]
