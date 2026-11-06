# Ensemble Map — the tailored ensembles, their members, and their outputs

Not one linear flow. A deterministic crypto axis runs in parallel, and a **router dispatches each
flow to one of a few target-specific ensembles** (the Phase-4 "domain experts"), each with its own
member composition **and its own output class set**.

## The ensembles

> **As built (Phase 5–7).** The table below is the design. What is deployed
> (`countdown/experts/__init__.py`, `deploy=`; `phases/PHASE-5.md` §10):
>
> | expert | deployed members | combiner | why |
> | --- | --- | --- | --- |
> | E1 | `flow_gbdt` only (+ per-class decision bias, Phase 7.3) | — | no subset beat the GBDT (0.665) |
> | E2 | `flow_gbdt` + `graph_gnn` (GIN) | calibrated soft vote | 0.890 vs 0.868; `byte_net` has no flow-aligned input yet |
> | E3 | `flow_gbdt` + `seq_cnn` + `flow_image_cnn` | calibrated soft vote | 0.513 vs 0.441 (92 test flows) |
>
> Routing: capture context (e.g. `mobile` → E3) wins; otherwise the Stage-1 hint (`tls`/`quic` → E2,
> everything else → E1); a flow the hint sent to E2 whose calibrated confidence is below E2's
> threshold is **re-classified by E1** (`route_reason = fallback`), because E2 confidently labels
> non-CSTNET TLS as one of its 120 sites (`PHASE-5.md` §10.5).

| # | Ensemble (= a Phase-4 expert) | Routes on | Member combo that fits | Output classes |
| --- | --- | --- | --- | --- |
| **—** | **Crypto verdict** *(not ML — Phase 2)* | any flow w/ visible handshake | rule-based KEM lookup | **5:** classical · hybrid · pqc · not_observable · unknown |
| **E1** | **Traffic-type** (behavioural) | tunneled (VPN/Tor) or direct classical | **flow-stats GBDT + sequence CNN** (+ FlowPic) | **8:** browsing · email · chat · audio_streaming · video_streaming · file_transfer · voip · p2p |
| **E2** | **Website/App-ID** | direct TLS, CSTNET-like, data-rich | **byte transformer + graph GNN + GBDT floor** | **120** website/SNI classes |
| **E3** | **In-app activity** | Wi-Fi / mobile-app | **App→Activity cascade: light seq/FlowPic + few-shot** | **8 apps → 92 activities** |
| **—** | tunnel_type / is_vpn / is_tor | — | *not an ensemble* — given by Stage-1 `l7_hint` | none · vpn · tor (free) |

Each flow's final record joins two axes: `{crypto verdict}` **×** `{the one applicable ML ensemble}`
— e.g. `pqc_verdict=classical, traffic_class=file_transfer, conf=0.86`.

## Why the member combos differ — the capacity-matching rule

Match **model capacity to (class count × samples-per-class)**:

| Regime | Example | Good combo | Bad combo (why) |
| --- | --- | --- | --- |
| Coarse + any data | E1: 8 classes | GBDT + light CNN | byte transformer / GNN → **no gain, wasted cost** |
| Fine + data-rich | E2: 120 × ~475 | byte transformer + GNN (GBDT floor) | GBDT alone → underfits |
| Fine + data-poor | E3: 92 × **4** | few-shot + augmented light CNN/FlowPic | deep transformer / GNN → **overfits catastrophically** |

The same "byte transformer + GNN" that *wins* E2 is *wrong* for E1 (overkill) and E3 (overfit).
That's why it's several tailored ensembles, not one stack applied everywhere.

## Runtime composition (branched, not linear)
```
flow ─▶ S0 crypto verdict ─┬─ pqc/hybrid ─▶ mark quantum-safe, stop
                           └─ classical / not_observable
                                   │  S1 context (l7_hint, deterministic)
                        ┌──────────┼───────────────┬───────────────┐
                     tunneled    direct-TLS     mobile/wifi      opaque
                        │        (CSTNET-like)      │               │
                       E1           E2             E3            E1 (fallback)
                        └──────────┴──────────────┴────────► per-flow verdict record + HNDL score
```
The router (Phase 5) is the dispatcher; a flow gets **one** applicable ensemble, not all three.
As built, an E2 flow below E2's confidence threshold falls back to E1, keeping E2's guess in the
record. The crypto branch stops classification for negotiated hybrid/PQC flows.

## Not an ensemble (deliberately)
- **PostQuantumTLS `app`** (1/class) → Phase-2 KEM ground truth.
- **`app` on ISCX** → capture-filename stems (guarded in code via `invalid_targets`).
- **`is_vpn`/`is_tor`, tunnel_type** → redundant with deterministic Stage-1 context.

## Caveat
E1/E2/E3 live on **different data distributions**. The router applies the *applicable* one by
context; cross-distribution use (E2 on a random tunneled flow) is weak and out of scope.
