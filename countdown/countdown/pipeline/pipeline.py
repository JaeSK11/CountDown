"""The end-to-end pipeline: pcap -> per-flow verdict records (Phase 6, task 6.1).

Per flow, in order -- the thesis of the project:

1. **Stage 1** reassemble: flow + handshake bytes + ``l7_hint`` (``flows/reassembly.py``).
2. **Stages 2-3** parse the handshake and give the PQC verdict (``crypto.enrich_flow``).
3. **Branch.** A negotiated ``hybrid`` / ``pqc`` exchange is quantum-safe: classification is
   skipped (``classify_quantum_safe=True`` tags it anyway).  Everything else -- classical,
   not observable (tunnels), unknown -- goes on to Stage 4.
4. **Stage 4** route to one expert, then classify:

   * ``context`` given (the operator says what the capture point sees, e.g. ``mobile``) ->
     the context's expert, for every flow (``route_reason = "context"``);
   * otherwise the Stage-1 hint (``route_reason = "hint"``): ``tls``/``quic`` -> E2, tunnels
     and the unspecific rest -> E1;
   * **confidence fallback** (``PHASE-5.md`` section 10.5): a flow the hint sent to E2 whose
     calibrated E2 confidence is below E2's exported threshold is re-classified by E1
     (``route_reason = "fallback"``); E2's guess is kept in ``fallback_from`` so the
     decision stays auditable.

``mode="fast"`` keeps only each expert's GBDT (no GPU, no byte graphs) and uses the
threshold exported for that member set.

5. **Phase 7** -- an expert with a per-class decision bias (``configs/decision_bias.yaml``,
   E1 only: ``ensemble/decision.py``) has it applied to the combined probabilities; every record gets the certificate verdict (``auth_verdict``, ``cert_key``)
   and a harvest-now-decrypt-later score with mitigation codes (``countdown/risk.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from countdown.crypto.handshake_only import is_handshake_only
from countdown.flows.short_flows import ShortFlowTally
from countdown.config import default_config, get_logger
from countdown.ensemble.decision import apply_bias, bias_for
from countdown.ensemble.router import Router
from countdown.pipeline.artifacts import DeployedExpert, load_store

log = get_logger(__name__)

def _pq_names() -> set[str]:
    from countdown.crypto import kem_registry

    return {g.name for c, g in kem_registry.all_groups().items()
            if kem_registry.is_quantum_resistant(c)}


_PQ_NAMES = _pq_names()
FALLBACK_FROM = "E2"
FALLBACK_TO = "E1"
QUANTUM_SAFE = ("hybrid", "pqc")


@dataclass
class PipelineConfig:
    mode: str = "full"                       # full | fast
    context: str | None = None               # capture-level override, e.g. "mobile"
    fallback: bool = True
    fallback_threshold: float | None = None  # None -> the threshold exported with E2
    classify_quantum_safe: bool = False
    include_members: bool = False            # per-member breakdown in each record
    redact: bool = False                     # drop IPs and SNI from records
    store: str | None = None                 # models_store/ directory
    allow_unverified: bool = False           # load artifacts not in models_store.sha256
    decision_bias: bool = True               # per-class biases from configs/decision_bias.yaml


@dataclass
class Pipeline:
    config: PipelineConfig = field(default_factory=PipelineConfig)
    experts: dict[str, DeployedExpert] | None = None
    router: Router = field(default_factory=Router)
    _tier_of: dict[str, str] | None = field(default=None, init=False, repr=False)
    _biases: dict[str, np.ndarray | None] = field(default_factory=dict, init=False, repr=False)
    #: Flows dropped before classification (shorter than ``min_packets``), by shape, over
    #: everything this pipeline has read -- scans live here (``flows/short_flows.py``).
    short_flows: ShortFlowTally = field(default_factory=ShortFlowTally, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.mode not in ("full", "fast"):
            raise ValueError(f"mode must be full or fast, got {self.config.mode!r}")
        if self.experts is None:
            self.experts = load_store(self.config.store, self.config.allow_unverified)

    # -- entry points ------------------------------------------------------------------
    def analyze_pcap(self, path: str | Path) -> list[dict[str, Any]]:
        from countdown.flows.reassembly import Reassembler
        from countdown.sources.pcap import PcapSource

        path = Path(path)
        reasm = Reassembler.from_config(default_config())
        with PcapSource(path) as src:
            flows = list(reasm.run(src, meta={"source_file": str(path)}))
        self.short_flows.merge(reasm.short_flows)
        log.info("[pipeline] %s: %d flows", path.name, len(flows))
        byte_rows = None
        if self._needs_bytes():
            byte_rows = byte_matrix_for_flows(path, flows)
        return self.analyze_flows(flows, byte_rows=byte_rows)

    def analyze_stream(self, source: Any, on_records: Any, batch_seconds: float = 2.0,
                       on_status: Any = None, status_seconds: float = 60.0) -> int:
        """Live mode (task 6.4): classify flows as the reassembler closes them.

        Closed flows are batched and handed to ``on_records`` at most every ``batch_seconds``;
        the flush is checked on every frame *and heartbeat* the source yields, so a quiet
        wire still delivers its last flows.  In ``full`` mode a :class:`ByteCollector` keeps
        each 5-tuple's first payload packets for the graph member.  Returns flows classified.
        ``on_status()`` is called at most every ``status_seconds`` (frames and heartbeats
        both count), so the dropped-short-flow tally shows up even when a scan is all there
        is on the wire and nothing gets classified.
        """
        import time

        from countdown.features.byte_prep import ByteCollector
        from countdown.flows.reassembly import Reassembler

        collector = ByteCollector() if self._needs_bytes() else None
        batch: list = []
        rows: list[np.ndarray] = []
        state = {"last": time.monotonic(), "n": 0, "status": time.monotonic()}

        def flush() -> None:
            if batch:
                byte_rows = np.stack(rows) if collector is not None else None
                on_records(self.analyze_flows(list(batch), byte_rows=byte_rows))
                state["n"] += len(batch)
                batch.clear()
                rows.clear()
            state["last"] = time.monotonic()

        def tee():
            for raw in source:
                if collector is not None and raw.data:
                    collector.add(raw.linktype, raw.data)
                if batch and time.monotonic() - state["last"] >= batch_seconds:
                    flush()
                if on_status is not None and time.monotonic() - state["status"] >= status_seconds:
                    on_status()
                    state["status"] = time.monotonic()
                yield raw

        reasm = Reassembler.from_config(default_config())
        reasm.short_flows = self.short_flows          # live: tally in place, readable mid-run
        name = getattr(source, "iface", None) or "live"
        for flow in reasm.run(tee(), meta={"source_file": f"live:{name}"}):
            batch.append(flow)
            if collector is not None:
                m = collector.take(flow_byte_key(flow))
                rows.append(m if m is not None else np.full((50, 190), -1, dtype=np.int16))
        flush()
        return state["n"]

    def analyze_flows(self, flows: list, byte_rows: np.ndarray | None = None) -> list[dict[str, Any]]:
        """``byte_rows``: ``(len(flows), 50, 190)`` byte matrices, NaN-free int rows, with
        ``-1`` in the first cell of a flow that has none (the graph member then abstains)."""
        from countdown.crypto import enrich_flow

        verdicts = [enrich_flow(f) for f in flows]
        records: list[dict[str, Any]] = [self._base_record(f, v) for f, v in zip(flows, verdicts)]

        # Phase 10 (2b): a handshake and nothing after it has a PQC verdict but no traffic
        # to classify -- scanners, our own census included (crypto/handshake_only.py).
        for f, rec in zip(flows, records):
            if is_handshake_only(f):
                rec.update(handshake_only=True, traffic_class="handshake_only",
                           route_reason="handshake_only")
        todo = [i for i, v in enumerate(verdicts)
                if not records[i]["handshake_only"]
                and (self.config.classify_quantum_safe or not v.is_quantum_resistant)]
        routed: dict[str, list[int]] = {}
        for i in todo:
            exp = self.router.route(flows[i].l7_hint, self.config.context)
            reason = "context" if self.config.context in self.router.context_rules else "hint"
            if exp not in self.experts:
                exp, reason = FALLBACK_TO, f"{reason}->missing:{exp}"
            records[i].update(expert=exp, route_reason=reason)
            routed.setdefault(exp, []).append(i)

        # E2 first: its low-confidence flows join E1's batch.
        fallback: list[int] = []
        if FALLBACK_FROM in routed:
            thr = self._threshold(FALLBACK_FROM)
            for i, label, conf, members in self._run(FALLBACK_FROM, flows, routed[FALLBACK_FROM], byte_rows):
                if (thr is not None and conf < thr and records[i]["route_reason"] == "hint"
                        and FALLBACK_TO in self.experts):
                    records[i]["fallback_from"] = {"expert": FALLBACK_FROM, "label": label,
                                                   "confidence": round(conf, 4),
                                                   "threshold": round(thr, 4)}
                    records[i].update(expert=FALLBACK_TO, route_reason="fallback")
                    fallback.append(i)
                else:
                    self._fill(records[i], FALLBACK_FROM, label, conf, members)
        rest = {n: list(ix) for n, ix in routed.items() if n != FALLBACK_FROM}
        if fallback:
            rest.setdefault(FALLBACK_TO, []).extend(fallback)
        for name, idx in rest.items():
            for i, label, conf, members in self._run(name, flows, idx, byte_rows):
                self._fill(records[i], name, label, conf, members)
        self._assess_risk(flows, records)
        return records

    # -- internals ---------------------------------------------------------------------
    def _assess_risk(self, flows: list, records: list[dict[str, Any]]) -> None:
        """Phase 7: HNDL score + mitigation codes (``countdown/risk.py``), after Stage 4 so the
        predicted traffic type can set the sensitivity through its readiness tier."""
        from countdown.risk import assess

        if self._tier_of is None:
            from countdown.eval.tiers import load_tiers

            self._tier_of = {c: t for t, cs in load_tiers().items() for c in cs}
        for f, rec in zip(flows, records):
            tier = (self._tier_of.get(rec["traffic_class"])
                    if rec.get("target") == "traffic_type" else None)
            rec.update(assess(rec, f.meta.get("crypto"), tier))

    def _bias(self, exp: DeployedExpert) -> np.ndarray | None:
        if not self.config.decision_bias:
            return None
        if exp.name not in self._biases:
            self._biases[exp.name] = bias_for(exp.name, exp.label_space.names)
        return self._biases[exp.name]

    def _needs_bytes(self) -> bool:
        return self.config.mode == "full" and any(
            r.features == "byte_matrix"
            for e in self.experts.values() for m, r in e.recipes.items()
            if m in e.member_names("full"))

    def _threshold(self, name: str) -> float | None:
        if not self.config.fallback or name != FALLBACK_FROM:
            return None
        if self.config.fallback_threshold is not None:
            return float(self.config.fallback_threshold)
        return self.experts[name].fallback_threshold.get(self.config.mode)

    def _run(self, name: str, flows: list, idx: list[int], byte_rows: np.ndarray | None):
        """``(row, label, confidence, per-member)`` for each flow index routed to ``name``."""
        if not idx:
            return []
        exp = self.experts[name]
        proba, per = self._classify(exp, [flows[i] for i in idx],
                                    None if byte_rows is None else byte_rows[idx])
        names = exp.label_space.names
        return [(i, names[int(proba[k].argmax())], float(proba[k].max()),
                 {m: per[m][k] for m in per}) for k, i in enumerate(idx)]

    def _fill(self, rec: dict[str, Any], name: str, label: str, conf: float,
              members: dict[str, Any]) -> None:
        rec.update(target=self.experts[name].target, traffic_class=label,
                   confidence=round(conf, 4))
        if self.config.include_members:
            rec["members"] = members

    def _classify(self, exp: DeployedExpert, flows: list, byte_rows: np.ndarray | None
                  ) -> tuple[np.ndarray, dict[str, list[dict[str, Any]]]]:
        from countdown.ensemble.views import member_input
        from countdown.schema import FlowDataset

        fds = FlowDataset(list(flows), name="pipeline", default_target=exp.target,
                          config=default_config())
        n = len(flows)
        member_proba: dict[str, np.ndarray] = {}
        for m in exp.member_names(self.config.mode):
            r, model = exp.recipes[m], exp.members[m]
            if r.features == "byte_matrix":
                p = np.full((n, len(exp.label_space)), np.nan)
                if byte_rows is not None:
                    ok = byte_rows[:, 0, 0] >= 0
                    if ok.any():
                        p[ok] = model.predict_proba(byte_rows[ok])
                member_proba[m] = p
                continue
            X, parent = member_input(fds, r.features, r.feature_params, r.windows)
            if parent is None:
                member_proba[m] = model.predict_proba(X)
            else:
                pw = model.predict_proba(X)
                out = np.zeros((n, pw.shape[1]))
                np.add.at(out, parent, pw)
                cnt = np.bincount(parent, minlength=n).astype(np.float64)
                out[cnt == 0] = np.nan
                cnt[cnt == 0] = 1.0
                member_proba[m] = out / cnt[:, None]
        final = exp.combine(member_proba)
        bias = self._bias(exp)
        if bias is not None:
            final = apply_bias(final, bias)
        names = exp.label_space.names
        per = {}
        for m, p in member_proba.items():
            per[m] = [None if np.isnan(row).any() else
                      {"label": names[int(row.argmax())], "confidence": round(float(row.max()), 4)}
                      for row in p]
        return final, per

    def _base_record(self, f: Any, v: Any) -> dict[str, Any]:
        crypto = f.meta.get("crypto", {})
        rec = {
            "flow_id": f.flow_id,
            "five_tuple": None if self.config.redact else str(f.five_tuple),
            "proto": f.five_tuple.l4_proto,
            "start": float(f.timestamps[0]) if len(f.timestamps) else None,
            "packets": int(len(f.timestamps)),
            "bytes": int(np.asarray(f.sizes).sum()),
            "l7": f.l7_hint,
            "tls_version": f.tls_version,
            "cipher": f.cipher_suite,
            "kem_group": f.kem_group,
            "sni": None if self.config.redact else crypto.get("sni"),
            "pqc_verdict": v.label,
            "pqc_basis": v.basis,
            "pqc_confidence": v.confidence,
            "quantum_safe": bool(v.is_quantum_resistant),
            # Does the ClientHello list any post-quantum group at all (HRR could select it)?
            "pqc_listed": any(g in _PQ_NAMES for g in crypto.get("offered_groups") or []),
            "at_risk": bool(v.is_at_risk),
            # Phase 7: the certificate chain, readable before TLS 1.3 only (crypto/x509.py).
            "auth_verdict": crypto.get("auth", {}).get("label"),
            "cert_key": crypto.get("auth", {}).get("leaf_key"),
            "expert": None, "route_reason": "skipped:quantum_safe" if v.is_quantum_resistant else None,
            "target": None, "traffic_class": None, "confidence": None,
            "handshake_only": False,
        }
        return rec


def byte_matrix_for_flows(path: str | Path, flows: list) -> np.ndarray:
    """TFE-GNN byte matrices for reassembled flows, joined on the canonical 5-tuple.

    ``byte_prep`` keeps one sample per 5-tuple (its first 50 payload packets); when the
    reassembler split that 5-tuple into several flows (idle timeout), the sample belongs to
    the earliest of them and the rest get ``-1`` rows -- the graph member abstains on them
    rather than scoring bytes from another flow.
    """
    from countdown.features.byte_prep import extract_byte_samples

    samples = extract_byte_samples(path)
    by_key = {s.meta["five_tuple"]: s for s in samples}
    out = np.full((len(flows), 50, 190), -1, dtype=np.int16)
    order = sorted(range(len(flows)),
                   key=lambda i: float(flows[i].timestamps[0]) if len(flows[i].timestamps) else 0.0)
    used: set = set()
    for i in order:
        key = flow_byte_key(flows[i])
        s = by_key.get(key)
        if s is None or key in used:
            continue
        used.add(key)
        out[i] = np.concatenate([s.header, s.payload], axis=1)
    return out


def flow_byte_key(flow: Any) -> tuple:
    """A flow's canonical 5-tuple in ``byte_prep``'s form (order-independent endpoints)."""
    t = flow.five_tuple
    a, b = (t.src_ip, int(t.src_port)), (t.dst_ip, int(t.dst_port))
    lo, hi = (a, b) if a <= b else (b, a)
    return (lo[0], lo[1], hi[0], hi[1], t.l4_proto)
