"""The Phase 1 acceptance test, verbatim from ``phases/PHASE-1.md`` section 8.

Runs against the real corpus and is skipped when it is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qsentinel.flows import Reassembler
from qsentinel.sources import PcapSource

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
CSTNET_DIR = DATA_ROOT / "CSTNET-TLS1.3" / "extracted"
ISCXTOR_DIR = DATA_ROOT / "ISCXTor2016" / "Tor"

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="dataset corpus not available")


def _first(directory: Path, pattern: str = "**/*.pcap") -> Path:
    if not directory.exists():
        pytest.skip(f"{directory} not present")
    for p in sorted(directory.glob(pattern)):
        if p.is_file():
            return p
    pytest.skip(f"no captures under {directory}")


def test_direct_tls_pcap_recovers_the_handshake():
    """A direct-TLS capture yields a TLS flow whose ServerHello window is intact."""
    flows = list(Reassembler().run(PcapSource(_first(CSTNET_DIR))))
    f = next(x for x in flows if x.l7_hint == "tls")
    assert f.handshake_server_bytes[:1] == b"\x16"      # TLS handshake record
    assert f.client_ip and len(f.packets) >= 4


def test_tunneled_pcap_still_parses_without_crashing():
    """Tor traffic produces flows; the outer layer is all that is observable, by design."""
    tor = list(Reassembler().run(PcapSource(_first(ISCXTOR_DIR))))
    assert len(tor) > 0
    assert all(f.pqc_verdict is None for f in tor)       # Phase 2 fills this in


def test_tunneled_flows_carry_no_pqc_claim():
    """The documented invariant: a tunnel gets flows and bytes, never a PQC verdict."""
    tor = list(Reassembler(context_hint="tor").run(PcapSource(_first(ISCXTOR_DIR))))
    tunneled = [f for f in tor if (f.l7_hint or "").startswith(("tor", "tunnel_"))]
    assert tunneled, "expected the Tor capture to hint tunnelled traffic"
    assert all(f.kem_group is None for f in tunneled)


@pytest.mark.slow
def test_phase0_features_are_unchanged():
    """Regression: the Phase 0 loader path still produces its feature matrix."""
    from qsentinel.data import load

    ds = load("iscxvpn", target="traffic_type")
    assert ds.features("flow_stats").shape[1] > 20
    assert len(ds.labels()) == len(ds)
