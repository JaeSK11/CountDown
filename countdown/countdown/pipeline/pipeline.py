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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from countdown.config import default_config, get_logger
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


@dataclass
class Pipeline:
    config: PipelineConfig = field(default_factory=PipelineConfig)
    experts: dict[str, DeployedExpert] | None = None
    router: Router = field(default_factory=Router)

    def __post_init__(self) -> None:
        if self.config.mode not in ("full", "fast"):
            raise ValueError(f"mode must be full or fast, got {self.config.mode!r}")
        if self.experts is None:
            self.experts = load_store(self.config.store)

    # -- entry points ------------------------------------------------------------------
    def analyze_pcap(self, path: str | Path) -> list[dict[str, Any]]:
        from countdown.flows.reassembly import Reassembler
        from countdown.sources.pcap import PcapSource

        path = Path(path)
        reasm = Reassembler.from_config(default_config())
        with PcapSource(path) as src:
            flows = list(reasm.run(src, meta={"source_file": str(path)}))
        log.info("[pipeline] %s: %d flows", path.name, len(flows))
        byte_rows = None
        if self._needs_bytes():
            byte_rows = byte_matrix_for_flows(path, flows)
        return self.analyze_flows(flows, byte_rows=byte_rows)

    def analyze_flows(self, flows: list, byte_rows: np.ndarray | None = None) -> list[dict[str, Any]]:
        """``byte_rows``: ``(len(flows), 50, 190)`` byte matrices, NaN-free int rows, with
        ``-1`` in the first cell of a flow that has none (the graph member then abstains)."""
        from countdown.crypto import enrich_flow

        verdicts = [enrich_flow(f) for f in flows]
        records: list[dict[str, Any]] = [self._base_record(f, v) for f, v in zip(flows, verdicts)]

        todo = [i for i, v in enumerate(verdicts)
                if self.config.classify_quantum_safe or not v.is_quantum_resistant]
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
        return records

    # -- internals ---------------------------------------------------------------------
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
            "expert": None, "route_reason": "skipped:quantum_safe" if v.is_quantum_resistant else None,
            "target": None, "traffic_class": None, "confidence": None,
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
        t = flows[i].five_tuple
        a, b = (t.src_ip, int(t.src_port)), (t.dst_ip, int(t.dst_port))
        lo, hi = (a, b) if a <= b else (b, a)
        key = (lo[0], lo[1], hi[0], hi[1], t.l4_proto)
        s = by_key.get(key)
        if s is None or key in used:
            continue
        used.add(key)
        out[i] = np.concatenate([s.header, s.payload], axis=1)
    return out
