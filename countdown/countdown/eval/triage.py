"""PQC-triage statistics -- the project's headline research output (Phase 6, task 6.5).

Two questions, over any set of flows:

1. **How much traffic is quantum-safe, exposed, or unobservable?**  Each flow's Phase-2 verdict,
   counted in flows *and* bytes (harvest-now-decrypt-later is about what an adversary can
   store, so volume matters as much as connection count).  ``quantum_safe`` is a negotiated
   hybrid/PQC exchange; ``at_risk`` a negotiated classical one; ``at_risk_ceiling`` a flow whose
   ClientHello offered only classical groups (server reply unseen -- still certainly classical);
   ``pqc_offered_undecided`` the reverse (PQC offered, outcome unseen).  ``not_observable``
   (tunnels, no handshake) is a statement about the capture, never counted as exposed.
2. **What does the exposed traffic look like?**  The class breakdown of the at-risk flows, plus
   readiness tiers (``configs/readiness_tiers.yaml``) when the classes are traffic types.

``triage_flows`` answers both from ground-truth labels on a labelled dataset (the research
table); ``triage_records`` answers them from ``countdown analyze`` records, i.e. from the
pipeline's *predicted* classes (what an operator gets on their own capture).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

AT_RISK = "at_risk"
SAFE = "quantum_safe"


#: Exposed because the client offered nothing post-quantum: only its ClientHello was seen,
#: but a server cannot select a group the client did not offer, so the outcome is classical.
AT_RISK_CEILING = "at_risk_ceiling"
EXPOSED = (AT_RISK, AT_RISK_CEILING)


def _bucket(label: str, basis: str, any_pqc_supported: bool | None = None) -> str:
    """quantum_safe / at_risk / at_risk_ceiling / pqc_offered_undecided / not_observable /
    unknown -- the verdict's label *and* how good its evidence was.

    An ``offered`` classical verdict is read from the client's key shares; a client can still
    *list* a hybrid group in ``supported_groups`` and be asked for it by a HelloRetryRequest.
    So it is a certain classical ceiling only when no listed group is post-quantum
    (``any_pqc_supported`` False); when that is unknown (records), it stays undecided.
    """
    if label in ("hybrid", "pqc"):
        return SAFE if basis == "negotiated" else "pqc_offered_undecided"
    if label == "classical":
        if basis == "negotiated":
            return AT_RISK
        return AT_RISK_CEILING if any_pqc_supported is False else "pqc_offered_undecided"
    if label == "not_observable":
        return "not_observable"
    return "unknown"


def _shares(counter: Counter, total: float) -> dict[str, dict[str, float]]:
    return {k: {"n": round(float(v), 1) if isinstance(v, float) else int(v),
                "share": round(float(v) / total, 4) if total else 0.0}
            for k, v in counter.most_common()}


def _summarise(rows: Sequence[dict[str, Any]], tiers: dict[str, list[str]] | None) -> dict[str, Any]:
    """``rows``: ``{bucket, label, basis, kem_group, tls_version, cls, bytes}`` per flow."""
    n = len(rows)
    nbytes = float(sum(r["bytes"] for r in rows))
    by_flow, by_bytes = Counter(), Counter()
    for r in rows:
        by_flow[r["bucket"]] += 1
        by_bytes[r["bucket"]] += r["bytes"]
    risk = [r for r in rows if r["bucket"] in EXPOSED]
    cls_flow, cls_bytes = Counter(), Counter()
    for r in risk:
        cls_flow[str(r["cls"])] += 1
        cls_bytes[str(r["cls"])] += r["bytes"]
    out: dict[str, Any] = {
        "flows": n, "bytes": int(nbytes),
        "verdict_by_flows": _shares(by_flow, n),
        "verdict_by_bytes": _shares(by_bytes, nbytes),
        "verdict_detail": dict(Counter(f"{r['label']}/{r['basis']}" for r in rows).most_common()),
        "observable_share": round(1 - by_flow["not_observable"] / n, 4) if n else 0.0,
        "exposed_share_flows": round(len(risk) / n, 4) if n else 0.0,
        "exposed_share_bytes": round(sum(r["bytes"] for r in risk) / nbytes, 4) if nbytes else 0.0,
        "kem_groups": dict(Counter(str(r["kem_group"]) for r in rows
                                   if r["bucket"] in (SAFE, AT_RISK)).most_common(10)),
        "tls_versions": dict(Counter(str(r["tls_version"]) for r in rows
                                     if r["bucket"] != "not_observable").most_common()),
        "at_risk_classes_by_flows": _shares(cls_flow, len(risk)),
        "at_risk_classes_by_bytes": _shares(cls_bytes, float(sum(cls_bytes.values()))),
    }
    if tiers:
        lookup = {c: t for t, cs in tiers.items() for c in cs}
        if any(str(r["cls"]) in lookup for r in risk):
            tf, tb = Counter(), Counter()
            for r in risk:
                t = lookup.get(str(r["cls"]), "unmapped")
                tf[t] += 1
                tb[t] += r["bytes"]
            out["at_risk_tiers_by_flows"] = _shares(tf, len(risk))
            out["at_risk_tiers_by_bytes"] = _shares(tb, float(sum(tb.values())))
    return out


def triage_flows(flows: Iterable[Any], label_field: str | None = None,
                 tiers: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Ground-truth triage of dataset flows (runs stages 2-3 on each)."""
    from countdown.crypto import enrich_flow, kem_registry, parse_handshake

    rows = []
    for f in flows:
        v = enrich_flow(f)
        pqc_listed = None
        if v.basis == "offered":
            pqc_listed = any(kem_registry.is_quantum_resistant(g)
                             for g in parse_handshake(f).offered_groups)
        cls = f.label_fields.get(label_field) if label_field else None
        rows.append({"bucket": _bucket(v.label, v.basis, pqc_listed), "label": v.label, "basis": v.basis,
                     "kem_group": f.kem_group, "tls_version": f.tls_version, "cls": cls,
                     "bytes": int(np.asarray(f.sizes).sum())})
    out = _summarise(rows, tiers)
    out["label_field"] = label_field
    return out


def triage_records(records: Sequence[dict[str, Any]], flows_bytes: Sequence[int] | None = None,
                   tiers: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Triage from ``countdown analyze`` records, using the **predicted** class.

    Records from several experts carry different targets; the class breakdown is reported per
    target (``traffic_type`` / ``app`` / ``activity``), tiers only for ``traffic_type``.
    """
    rows = []
    for i, r in enumerate(records):
        rows.append({"bucket": _bucket(r["pqc_verdict"], r["pqc_basis"], r.get("pqc_listed")),
                     "label": r["pqc_verdict"],
                     "basis": r["pqc_basis"], "kem_group": r.get("kem_group"),
                     "tls_version": r.get("tls_version"),
                     "cls": r.get("traffic_class"), "target": r.get("target"),
                     "bytes": int(flows_bytes[i]) if flows_bytes is not None else int(r.get("bytes") or 0)})
    out = _summarise(rows, None)
    # Classes from different experts are different vocabularies; mix them only per target.
    out["at_risk_classes_by_flows"], out["at_risk_classes_by_bytes"] = {}, {}
    per_target: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["bucket"] in EXPOSED and r["target"]:
            per_target[r["target"]].append(r)
    out["at_risk_classes_by_target"] = {
        t: _summarise(rs, tiers if t == "traffic_type" else None) for t, rs in per_target.items()}
    for t in out["at_risk_classes_by_target"].values():
        for k in ("verdict_by_flows", "verdict_by_bytes", "verdict_detail", "observable_share",
                  "kem_groups", "tls_versions"):
            t.pop(k, None)
    return out


def breakdown_error(true_cls: Sequence[Any], pred_cls: Sequence[Any]) -> dict[str, Any]:
    """How far a predicted class *mix* is from the true one: total-variation distance between
    the two distributions (0 = identical mix, 1 = disjoint), plus the largest per-class gap."""
    t, p = Counter(map(str, true_cls)), Counter(map(str, pred_cls))
    n = max(len(true_cls), 1)
    keys = set(t) | set(p)
    gaps = {k: (p[k] - t[k]) / n for k in keys}
    worst = max(gaps, key=lambda k: abs(gaps[k])) if gaps else None
    return {"n": len(true_cls), "total_variation": round(0.5 * sum(abs(g) for g in gaps.values()), 4),
            "worst_class": worst, "worst_gap": round(gaps[worst], 4) if worst else 0.0}


def classes_markdown(title: str, t: dict[str, Any], top: int = 8) -> str:
    """Just the class (and tier) tables of a triage summary."""
    lines = [f"#### {title}", ""]
    for key, label in (("at_risk_classes", "class"), ("at_risk_tiers", "readiness tier")):
        flows, nbytes = t.get(f"{key}_by_flows"), t.get(f"{key}_by_bytes", {})
        if not flows:
            continue
        lines += [f"| {label} | flows | share | bytes share |", "| --- | --- | --- | --- |"]
        for k, v in list(flows.items())[:top]:
            lines.append(f"| {k} | {v['n']:,} | {v['share']:.1%} | {nbytes.get(k, {}).get('share', 0):.1%} |")
        lines.append("")
    return "\n".join(lines)


def to_markdown(name: str, t: dict[str, Any]) -> str:
    lines = [f"### {name}", "",
             f"{t['flows']:,} flows, {t['bytes'] / 1e6:,.1f} MB; handshake observable on "
             f"{t['observable_share']:.1%} of flows; **exposed (at_risk + at_risk_ceiling): "
             f"{t['exposed_share_flows']:.1%} of flows, {t['exposed_share_bytes']:.1%} of bytes**.", "",
             "| verdict | flows | share | bytes share |", "| --- | --- | --- | --- |"]
    vb = t["verdict_by_bytes"]
    for k, v in t["verdict_by_flows"].items():
        lines.append(f"| {k} | {v['n']:,} | {v['share']:.1%} | {vb.get(k, {}).get('share', 0):.1%} |")
    if t.get("at_risk_classes_by_flows"):
        lines += ["", f"Exposed flows by {t.get('label_field') or 'class'} (top 8):", "",
                  "| class | flows | share | bytes share |", "| --- | --- | --- | --- |"]
        cb = t["at_risk_classes_by_bytes"]
        for k, v in list(t["at_risk_classes_by_flows"].items())[:8]:
            lines.append(f"| {k} | {v['n']:,} | {v['share']:.1%} | {cb.get(k, {}).get('share', 0):.1%} |")
    if t.get("at_risk_tiers_by_flows"):
        lines += ["", "Exposed flows by readiness tier:", "",
                  "| tier | flows | share | bytes share |", "| --- | --- | --- | --- |"]
        tb = t["at_risk_tiers_by_bytes"]
        for k, v in t["at_risk_tiers_by_flows"].items():
            lines.append(f"| {k} | {v['n']:,} | {v['share']:.1%} | {tb.get(k, {}).get('share', 0):.1%} |")
    return "\n".join(lines) + "\n"
