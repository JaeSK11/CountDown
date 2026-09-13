# Cascade & Harvest-Now-Decrypt-Later (HNDL) Ordering

Stage 4 is **not** a flat ensemble applied uniformly. It's a **cost-and-value-ordered cascade with
confidence gating**: cheap/deterministic discriminators first, expensive fine-grained models last
and only on the flows that warrant them. This reduces compute, lifts macro-F1 on the tail, and makes
the pipeline double as an **HNDL harvest-priority scorer**.

## The one hard structural limit
Targets live on **different datasets** (ISCX=traffic_type, CSTNET=app), so you cannot chain
`traffic_type → app` on a single flow — except **MobileApp (App→Activity)**, the one within-flow
hierarchy. So it's **staged triage + local hierarchies**, not a monolithic conditional tree.

## The order (cheapest / most-certain / highest-value first)

| Stage | Runs | Cost | Buys |
| --- | --- | --- | --- |
| **S0 Crypto verdict** | rule-based KEM lookup (Phase 2) | free | drop `pqc`/`hybrid`; keep `classical` + `not_observable` |
| **S1 Context** | tunnel/protocol from Stage-1 `l7_hint` | ~free | **you already *know* VPN/Tor/direct — don't spend a model on it.** Biggest reduction |
| **S2 Coarse behavioural** | `traffic_type` GBDT (pooled ISCX) | cheap | 8 broad classes; **attach HNDL value weights here** |
| **S3 Fine-grained** | app/activity deep models | expensive | run **only** on the right distribution + where value/ambiguity justify it |

## Why ordering *raises* accuracy (not just saves compute)
1. **Confidence-gated escalation** — S2 resolves the easy majority; only low-confidence flows
   escalate to the deep model → less overfitting on easy cases, capacity spent on the hard residual.
2. **Label-space reduction where a hierarchy exists** — MobileApp App(8)→Activity(92): disambiguate
   ~4–22 activities within an app, not 92 globally.
3. **Tail protection for macro-F1** — resolve the broad mass cheaply so tail classes get attention.
4. **Keep routing *soft*** — pass probabilities, not hard labels, so a wrong coarse call doesn't doom
   the flow. The Phase-5 combiner already speaks `predict_proba`, so this is an extension, not a
   rewrite.

## HNDL value ordering (the payoff)
The pipeline is a **harvest-priority queue**, not just a classifier:
- **Value-weight the coarse classes** by residual intelligence value once decrypted. **Working
  default (adjustable):** `file_transfer ≈ email > chat ≈ app-data > voip > audio/video_streaming`.
- **`not_observable` (VPN/Tor) classical traffic ranks *high*** — deliberately tunnelled
  (intent-to-hide) and un-verdictable → prime behavioural + harvest target, not "ignore".
- **Spend expensive stages + storage top-of-queue** — run the 120-class transformer and reserve
  capture storage for high-value, classical, ambiguous flows; skip it on confidently-cheap streaming.

So the order = **free crypto filter → free context → cheap value-scoring coarse pass → expensive
fine-grained ID only on the high-value residual.**

## Constraints from the verified data that bound this
- **Pool ISCXVPN+ISCXTor for `traffic_type`** — mandatory: VPN has no `browsing`, `p2p` is one file;
  Tor supplies both.
- **`is_tor` is 77:1** → Tor is a flag/anomaly, never a balanced cascade branch.
- **`app` on ISCX is invalid** (capture stems, guarded in code) → not a cascade target; only CSTNET
  app / MobileApp activity are real fine targets.
- **CSTNET is 1 pcap = 1 flow, no traffic_type** → can't chain traffic_type→app there; per-flow
  graphs only.
- **activity = 4 samples/class** → few-shot; the App→Activity cascade is the one clean hierarchy.

## Open knob
The HNDL value ranking is a **threat-model judgement**, not data-derived. The default above is a
placeholder — set your own priority for what's worth harvesting and it flows straight into S2's
scoring.
