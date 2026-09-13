"""Stateful bidirectional flow reassembly -- Pipeline Stage 1 (Phase 1, tasks 1.5-1.6).

``Reassembler`` consumes a ``Source`` and emits ``Flow`` objects carrying everything
Phase 2 needs: the packet-size/timing series Phase 0 already produced, plus the
**reassembled opening bytes of each direction** and a **coarse L7 transport hint**.

Two operating modes
-------------------
``Reassembler()`` (streaming, the default)
    Flows are emitted *as they expire* -- FIN/RST teardown, idle timeout, or max
    duration -- so memory stays bounded on a live NIC or a multi-gigabyte capture.  This
    is the mode Phase 6's live path uses.

``Reassembler(phase0_compat=True)``
    Reproduces Phase 0's batch splitter **exactly**: no FIN/RST teardown, no periodic
    sweep, flows collected and emitted sorted by start time with Phase 0's ``flow_id``
    numbering.  ``FlowExtractor`` uses this so the Phase 0 ``(X, y)`` matrices are
    byte-identical after the refactor (pinned by ``tests/test_phase0_regression.py``).

    The split is not cosmetic and cannot be avoided: closing a flow on FIN changes where
    flow boundaries fall, so a post-FIN straggler starts a new flow instead of joining
    the old one.  That is the *better* behaviour for a live sensor and the *wrong*
    behaviour for reproducing a cached feature matrix, so both are kept and the choice is
    explicit at the call site.

Expiry is driven by the **packet clock**, never ``time.time()``, so a pcap run is
reproducible byte-for-byte regardless of how fast the machine reads it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from qsentinel.config import get_logger
from qsentinel.flows.decode import DecodedPacket, ParseStats, decode
from qsentinel.flows.handshake import (
    DEFAULT_WINDOW,
    HeadBuffer,
    l7_hint,
    opens_handshake,
    records_complete,
    quic_long_header,
)
from qsentinel.schema import BACKWARD, FORWARD, FiveTuple, Flow
from qsentinel.sources.base import RawPacket, Source

log = get_logger(__name__)

#: Ports that mark their endpoint as the server even above the well-known range.
SERVICE_PORTS = frozenset({
    443, 465, 587, 993, 995, 1080, 1194, 3128, 4443, 5222, 5228, 5223, 8000, 8008,
    8080, 8081, 8443, 8888, 9001, 9030, 9050, 9051,
})

#: ``expiry_reason`` values.
EXPIRY_FIN = "fin"
EXPIRY_RST = "rst"
EXPIRY_IDLE = "idle_timeout"
EXPIRY_MAX_DURATION = "max_duration"
EXPIRY_END_OF_CAPTURE = "end_of_capture"
EXPIRY_TABLE_FULL = "table_full"


def is_service_port(port: int) -> bool:
    return port < 1024 or port in SERVICE_PORTS


_is_service_port = is_service_port  # Phase 0 private name


def canonical_key(sip: str, sport: int, dip: str, dport: int, proto: str) -> tuple:
    """Order-independent key so both directions land in the same flow."""
    a, b = (sip, sport), (dip, dport)
    if a <= b:
        return (a[0], a[1], b[0], b[1], proto)
    return (b[0], b[1], a[0], a[1], proto)


@dataclass
class _FlowState:
    """Mutable per-flow buffer.

    Packets are recorded relative to endpoint ``a`` (the canonically-first endpoint of the
    key); which endpoint is the *client* is decided once at emission, so a SYN seen
    mid-capture can still correct the orientation of everything before it.
    """

    key: tuple
    first_sender: tuple[str, int]
    index: int
    source: str = ""
    ts: list[float] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    from_a: list[bool] = field(default_factory=list)
    syn_client: tuple[str, int] | None = None
    head_a: HeadBuffer | None = None
    head_b: HeadBuffer | None = None
    fin_a: bool = False
    fin_b: bool = False
    saw_rst: bool = False
    closing_at: float | None = None
    expiry_reason: str | None = None

    @property
    def last_ts(self) -> float:
        return self.ts[-1]

    @property
    def start_ts(self) -> float:
        return self.ts[0]

    def endpoints(self) -> tuple[tuple[str, int], tuple[str, int]]:
        return (self.key[0], self.key[1]), (self.key[2], self.key[3])

    def client(self) -> tuple[str, int]:
        """Decide which endpoint initiated the conversation.

        1. The sender of a SYN-without-ACK is the client (definitive for TCP).
        2. Otherwise, if exactly one endpoint uses a service port, the other is the client.
        3. Otherwise fall back to the first-seen sender.
        """
        if self.syn_client is not None:
            return self.syn_client
        a, b = self.endpoints()
        a_srv, b_srv = is_service_port(a[1]), is_service_port(b[1])
        if a_srv != b_srv:
            return b if a_srv else a
        return self.first_sender


class Reassembler:
    """Flow table + TCP head reassembly + handshake capture.

    Parameters
    ----------
    flow_timeout / idle_timeout:
        Seconds of idleness that closes a flow (``idle_timeout`` wins if both are given;
        ``flow_timeout`` is Phase 0's name for the same knob).
    min_packets:
        Flows shorter than this are dropped and counted in ``ParseStats``.
    max_flows_per_pcap:
        Guard against pathological captures.
    handshake_window:
        Bytes of the opening of each direction to reassemble.  8 KB comfortably holds a
        ClientHello plus a ServerHello and the start of a certificate chain.
    max_duration:
        Hard cap on flow lifetime, so a long-lived connection still gets emitted.
    close_on_fin:
        Emit on RST, or once both directions have sent FIN and ``fin_grace`` seconds of
        packet clock have passed (letting the final ACK land in the same flow).
    capture_handshake:
        Set False to skip head reassembly entirely when only sizes/timings are wanted.
    context_hint:
        Dataset-level knowledge (``"tor"`` / ``"vpn"``) applied to ``l7_hint`` only when
        the wire evidence is unspecific.
    phase0_compat:
        Reproduce Phase 0's batch splitter exactly.  See the module docstring.
    """

    def __init__(
        self,
        flow_timeout: float = 64.0,
        idle_timeout: float | None = None,
        min_packets: int = 4,
        max_flows_per_pcap: int = 200_000,
        handshake_window: int = DEFAULT_WINDOW,
        max_duration: float = 3600.0,
        close_on_fin: bool = True,
        fin_grace: float = 2.0,
        capture_handshake: bool = True,
        emit_handshake_meta: bool | None = None,
        sweep_interval: float = 8.0,
        context_hint: str | None = None,
        phase0_compat: bool = False,
        **_ignored: Any,
    ) -> None:
        self.idle_timeout = float(idle_timeout if idle_timeout is not None else flow_timeout)
        self.min_packets = int(min_packets)
        self.max_flows_per_pcap = int(max_flows_per_pcap)
        self.handshake_window = int(handshake_window)
        self.max_duration = float(max_duration)
        self.capture_handshake = bool(capture_handshake)
        # Phase 0 cached flows are ordered by json.dumps(meta) in data/base.py, so an
        # extra meta key would reorder every dataset's rows.  Off in compat mode.
        self.emit_handshake_meta = (
            (not phase0_compat) if emit_handshake_meta is None else bool(emit_handshake_meta)
        )
        self.context_hint = context_hint
        self.phase0_compat = bool(phase0_compat)
        # Phase 0 had no teardown handling and no periodic sweep; honouring either would
        # move flow boundaries and break the (X, y) regression.
        self.close_on_fin = bool(close_on_fin) and not self.phase0_compat
        self.fin_grace = float(fin_grace)
        self.sweep_interval = math.inf if self.phase0_compat else float(sweep_interval)
        if self.phase0_compat:
            self.max_duration = math.inf

    @classmethod
    def from_config(cls, config, **overrides: Any) -> "Reassembler":
        """Build from ``config.flow`` + ``config.reassembly`` (see configs/features.yaml)."""
        params: dict[str, Any] = {}
        if config is not None:
            params.update(dict(getattr(config, "flow", {}) or {}))
            params.update(dict(getattr(config, "reassembly", {}) or {}))
        params.update(overrides)
        return cls(**params)

    # -- main entry point --------------------------------------------------------------
    def run(
        self,
        source: Source | Iterable[RawPacket],
        dataset: str = "",
        label: str = "",
        label_fields: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        stats: ParseStats | None = None,
    ) -> Iterator[Flow]:
        """Consume a source, yield ``Flow`` objects.

        Streaming mode yields flows at expiry (bounded memory).  ``phase0_compat`` mode
        collects everything, then yields sorted by start time with Phase 0's ids.
        """
        stats = stats if stats is not None else ParseStats()
        label_fields = dict(label_fields or {})
        base_meta = dict(meta or {})
        multi = getattr(source, "multi_capture", False)

        open_flows: dict[tuple, _FlowState] = {}
        finished: list[_FlowState] = []
        seq = 0
        current_source = ""
        next_sweep = math.inf
        truncated = False

        def _new_state(key, sender, src) -> _FlowState:
            nonlocal seq
            st = _FlowState(key=key, first_sender=sender, index=seq, source=src)
            seq += 1
            return st

        # A file source counts packets_read as it reads.  Only count here when it is
        # writing into a *different* ParseStats (or is a bare iterable of RawPacket),
        # otherwise every frame would be counted twice.
        source_counts_reads = getattr(source, "stats", None) is stats

        for raw in source:
            if not source_counts_reads:
                stats.packets_read += 1

            if multi and raw.source != current_source:
                # New capture: never merge flows across files on a colliding 5-tuple.
                for st_open in open_flows.values():
                    st_open.expiry_reason = EXPIRY_END_OF_CAPTURE
                    finished.append(st_open)
                open_flows.clear()
                current_source = raw.source
                next_sweep = math.inf

            pkt = decode(raw.ts, raw.data, raw.linktype, stats)
            if pkt is None:
                continue
            ts = pkt.ts

            # ---- periodic expiry sweep (packet clock, never wall clock) --------------
            if ts >= next_sweep:
                for st in self._sweep(open_flows, ts):
                    finished.append(st)
                next_sweep = ts + self.sweep_interval
            elif math.isinf(next_sweep) and not self.phase0_compat:
                next_sweep = ts + self.sweep_interval

            key = canonical_key(pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port, pkt.proto)
            st = open_flows.get(key)

            if st is not None and ts - st.last_ts > self.idle_timeout:
                st.expiry_reason = EXPIRY_IDLE
                finished.append(st)  # idle split
                st = None

            if st is None:
                if len(finished) + len(open_flows) >= self.max_flows_per_pcap:
                    stats.truncated_at_max_flows = True
                    if self.phase0_compat:
                        truncated = True
                        break
                    # Streaming (and live) must not stop reading: evict the stalest flow
                    # so the table stays bounded and capture continues.
                    stalest = min(open_flows, key=lambda k: open_flows[k].last_ts)
                    victim = open_flows.pop(stalest)
                    victim.expiry_reason = EXPIRY_TABLE_FULL
                    finished.append(victim)
                st = _new_state(key, pkt.src, raw.source or current_source)
                open_flows[key] = st

            self._absorb(st, pkt, key)
            stats.packets_used += 1

            # ---- teardown ------------------------------------------------------------
            if self.close_on_fin and pkt.proto == "tcp":
                if pkt.is_rst:
                    st.saw_rst = True
                    st.expiry_reason = EXPIRY_RST
                    finished.append(st)
                    del open_flows[key]
                elif st.fin_a and st.fin_b and st.closing_at is None:
                    st.closing_at = ts + self.fin_grace

            if not self.phase0_compat and finished:
                for st_done in finished:
                    flow = self._emit(st_done, dataset, label, label_fields, base_meta, stats)
                    if flow is not None:
                        yield flow
                finished.clear()

        # ---- drain ------------------------------------------------------------------
        remaining = list(open_flows.values())
        if self.phase0_compat:
            finished.extend(remaining)
            finished.sort(key=lambda s: (s.ts[0], s.index))
            out_index = 0
            for st in finished:
                flow = self._emit(st, dataset, label, label_fields, base_meta, stats,
                                  flow_index=out_index)
                if flow is not None:
                    out_index += 1
                    yield flow
        else:
            for st in finished:
                flow = self._emit(st, dataset, label, label_fields, base_meta, stats)
                if flow is not None:
                    yield flow
            for st in sorted(remaining, key=lambda s: (s.ts[0], s.index)):
                st.expiry_reason = (
                    EXPIRY_TABLE_FULL if truncated else EXPIRY_END_OF_CAPTURE
                )
                flow = self._emit(st, dataset, label, label_fields, base_meta, stats)
                if flow is not None:
                    yield flow

    # -- per-packet ---------------------------------------------------------------------
    def _absorb(self, st: _FlowState, pkt: DecodedPacket, key: tuple) -> None:
        from_a = pkt.src == (key[0], key[1])

        if st.syn_client is None and pkt.is_pure_syn:
            st.syn_client = pkt.src

        st.ts.append(pkt.ts)
        st.sizes.append(pkt.size)
        st.from_a.append(from_a)

        if pkt.proto == "tcp" and pkt.is_fin:
            if from_a:
                st.fin_a = True
            else:
                st.fin_b = True

        if not self.capture_handshake:
            return

        is_tcp = pkt.proto == "tcp"
        if from_a:
            if st.head_a is None:
                st.head_a = HeadBuffer(self.handshake_window, is_tcp)
            buf = st.head_a
        else:
            if st.head_b is None:
                st.head_b = HeadBuffer(self.handshake_window, is_tcp)
            buf = st.head_b

        if is_tcp and pkt.is_syn:
            buf.note_syn(pkt.seq)
        if pkt.payload:
            buf.add(pkt.payload, pkt.seq)

    # -- expiry -------------------------------------------------------------------------
    def _sweep(self, open_flows: dict[tuple, _FlowState], now: float) -> list[_FlowState]:
        """Evict every flow whose expiry condition has fired.  Packet-clock driven."""
        expired: list[_FlowState] = []
        for key, st in list(open_flows.items()):
            reason = None
            if st.closing_at is not None and now >= st.closing_at:
                reason = EXPIRY_FIN
            elif now - st.last_ts > self.idle_timeout:
                reason = EXPIRY_IDLE
            elif now - st.start_ts > self.max_duration:
                reason = EXPIRY_MAX_DURATION
            if reason is not None:
                st.expiry_reason = reason
                expired.append(st)
                del open_flows[key]
        return expired

    # -- emission -----------------------------------------------------------------------
    def _emit(
        self,
        st: _FlowState,
        dataset: str,
        label: str,
        label_fields: dict[str, Any],
        base_meta: dict[str, Any],
        stats: ParseStats,
        flow_index: int | None = None,
    ) -> Flow | None:
        if len(st.ts) < self.min_packets:
            stats.flows_dropped_short += 1
            return None

        endpoint_a, endpoint_b = st.endpoints()
        client = st.client()
        server = endpoint_b if client == endpoint_a else endpoint_a
        client_is_a = client == endpoint_a

        # +1 means client->server; from_a is True when the sender was endpoint_a
        from_a = np.asarray(st.from_a, dtype=bool)
        directions = np.where(from_a == client_is_a, FORWARD, BACKWARD).astype(np.int8)
        ft = FiveTuple(client[0], client[1], server[0], server[1], st.key[4])

        head_c = st.head_a if client_is_a else st.head_b
        head_s = st.head_b if client_is_a else st.head_a
        cb = head_c.data() if head_c is not None else b""
        sb = head_s.data() if head_s is not None else b""

        hint = l7_hint(st.key[4], cb, sb, client[1], server[1], self.context_hint)
        incomplete = self._handshake_incomplete(head_c, head_s, cb, sb, hint)

        meta = dict(base_meta)
        stem = Path(st.source).stem if st.source else ""
        if st.source:
            meta.setdefault("source_file", st.source)
        if self.emit_handshake_meta:
            meta["handshake_client_complete"] = bool(head_c is not None and head_c.complete)
            meta["handshake_server_complete"] = bool(head_s is not None and head_s.complete)
            meta["handshake_bytes_seen"] = [
                int(head_c.bytes_seen) if head_c else 0,
                int(head_s.bytes_seen) if head_s else 0,
            ]

        idx = flow_index if flow_index is not None else st.index
        prefix = stem or dataset or "flow"
        flow = Flow(
            flow_id=f"{prefix}#{idx}",
            five_tuple=ft,
            timestamps=np.asarray(st.ts, dtype=np.float64),
            sizes=np.asarray(st.sizes, dtype=np.int32),
            directions=directions,
            dataset=dataset,
            label=label,
            label_fields=dict(label_fields),
            meta=meta,
            l7_hint=hint,
            handshake_client_bytes=cb or None,
            handshake_server_bytes=sb or None,
            handshake_incomplete=incomplete,
            expiry_reason=st.expiry_reason or EXPIRY_END_OF_CAPTURE,
        )
        stats.flows_built += 1
        # Head buffers are dead once their bytes are on the Flow; drop them so a long
        # capture does not accumulate 2 x handshake_window per emitted flow.
        st.head_a = st.head_b = None
        return flow

    @staticmethod
    def _handshake_incomplete(head_c: HeadBuffer | None, head_s: HeadBuffer | None,
                              cb: bytes, sb: bytes, hint: str) -> bool:
        """Whether the captured window can be trusted to hold the opening handshake.

        Two distinct failures, both of which must set this flag, because Phase 2 has to
        tell "no PQC" apart from "could not see":

        * **A gap.** Head reassembly stopped at an unfilled hole, so the window may be
          silently truncated mid-record.
        * **No handshake present.** The flow was captured mid-session -- the TCP
          connection predates the capture -- so the window is record-aligned but opens on
          application data or an alert.  Measured on PostQuantumTLS, this is ~9% of :443
          flows and is a property of the dataset, not something reassembly can fix.

        The test is applied to the **server** window specifically.  Phase 2 reads the
        PQC verdict from the ServerHello's ``key_share`` -- the group actually negotiated
        -- so a flow carrying only a ClientHello can say what was *offered* but can never
        yield a verdict, and must not be presented as observable.  Requiring the server
        side also matches how these captures truncate: every CSTNET capture begins after
        the ClientHello, with the client window opening on a ChangeCipherSpec while the
        server window still carries the ServerHello.

        ``handshake_client_bytes`` is still populated in that case; the flag only says a
        verdict is not derivable, not that the bytes are worthless.
        """
        if hint not in ("tls", "quic", "tor"):
            return False  # nothing to be incomplete about
        if hint == "quic":
            # The Initial that carries the ClientHello comes from the client.
            return bool(head_c is not None and head_c.has_gap) or not (
                quic_long_header(cb)[0] or quic_long_header(sb)[0]
            )
        # Only the server window's integrity matters: a hole in the client's stream has
        # no bearing on whether the ServerHello can be read.  Client-side integrity is
        # still reported separately in meta["handshake_client_complete"].
        if not opens_handshake(sb):
            return True
        # A gap matters only if it truncates the handshake itself.  Holes further into
        # the 8 KB window -- common on CSTNET, where the certificate chain that follows
        # the ServerHello is partly missing -- leave a whole, parseable ServerHello
        # record behind, and writing those flows off would discard usable handshakes.
        if head_s is not None and head_s.has_gap and not records_complete(sb):
            return True
        return False
