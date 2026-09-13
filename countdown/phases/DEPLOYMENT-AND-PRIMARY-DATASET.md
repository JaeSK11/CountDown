# Deployment & Primary Dataset

**Decision:** the tool's job is to **passively collect pcap on a real network and run it through the
ensemble.** That deployment distribution — not the training convenience — decides the primary
dataset.

## Primary dataset: CSTNET-TLS 1.3 (not ISCX)

Judge each dataset by closeness to passively-captured real traffic:

| Dataset | What it actually is | Match to passive capture |
| --- | --- | --- |
| **CSTNET-TLS1.3** | **real traffic on a live network** (CSTNET, 2021), 120 real services, **TLS 1.3**, direct | **Best** — real network, current protocol, **KEM observable** |
| ISCXVPN2016 | testbed, one activity/capture, mostly **VPN-tunnelled**, 2016 | Poor — synthetic, tunneled, dated |
| ISCXTor2016 | testbed, **Tor-tunnelled**, 2016 | Poor — tunneled, dated |
| PostQuantumTLS | 90 real Android captures (1/app) | Too small — PQC ground-truth, not a training set |
| MobileApp | Wi-Fi testbed, scripted actions | Poor — synthetic, mobile-only |

Decisive point for **this** project: ISCX VPN/Tor traffic is **tunneled → crypto verdict
`not_observable`**, so the novel PQC axis can't even fire there. CSTNET is direct TLS → the KEM is
visible and the triage works.

## Generalizes vs closed-world (what to trust on arbitrary passive traffic)

- **Generalizes to any passive flow** — the real product:
  - **Crypto/PQC verdict** (deterministic; dataset-independent; works on any TLS/QUIC).
  - **Coarse `traffic_type`** (browsing/streaming/file-transfer/voip…) — behavioural, transfers.
- **Closed-world — only recognises training classes** (bonus, not backbone):
  - **Website/App-ID** (CSTNET's 120) and **in-app activity** — cannot name classes outside training.

## Dataset roles in the passive-capture build

- **CSTNET-TLS1.3 → primary.** Deployment-distribution match; validates Stage-1→2→3 end-to-end on
  realistic flows; trains the website-ID head (E2).
- **ISCXVPN + ISCXTor (pooled) → auxiliary only**, to train the coarse `traffic_type` head (E1) —
  the only source of behavioural labels — accepting 2016-testbed → real-traffic domain shift.
- **PostQuantumTLS + self-generated handshakes → Phase-2 PQC ground truth.**
- **MobileApp → only if mobile in-app activity is an actual deployment goal.**

## Gaps this exposes (act on these)

1. **QUIC/HTTP-3** is a large share of 2026 passive traffic; CSTNET (2021) is mostly TLS-1.3-over-TCP
   → raises priority of the **Phase-2 QUIC Initial parser** (Phase-1 already captures the Initial).
2. **Your services ≠ CSTNET's 120** → website-ID won't transfer to your network's sites.
3. **No public set is *your* network** → **capture a small labelled validation set from the actual
   deployment environment** and measure transfer there. Train on CSTNET, *validate on your capture*.

## One-line
Primary = **CSTNET-TLS 1.3**; ISCX = auxiliary (behavioural head only); the parts trustworthy on
arbitrary passive traffic are the **deterministic crypto verdict + coarse traffic-type**, with
website-ID a closed-world extra.
