"""The three tailored ensembles (decision D2): E1 traffic type, E2 app-ID, E3 activity.

Member choice follows ``ENSEMBLE-MAP.md``'s capacity-matching rule -- match model capacity to
(class count x samples per class) -- and the Phase-4 scorecards in ``phases/phase4-models/``.
"""

from countdown.experts.base import Expert, MemberSpec

_GBDT = MemberSpec("flow_gbdt", "flow_stats", role="universal floor; fast, interpretable")

E1 = Expert(
    name="E1", title="traffic type (behavioural)", dataset="iscx_pooled", target="traffic_type",
    routes_on="tunnelled (VPN/Tor) or direct-classical flows; also the opaque-flow fallback",
    members=(
        _GBDT,
        MemberSpec("seq_cnn", "packet_seq", {"n": 256, "channels": 2},
                   role="universal deep voter on sizes + timing"),
        MemberSpec("flow_image_cnn", "flow_image", {"construction": "flowpic", "channels": 3},
                   windows=(15.0, 30.0, 60.0),
                   role="FlowPic view; +0.05-0.07 macro-F1 over scatter on VPN/Tor traffic type"),
    ),
    notes="Coarse, 8 classes: byte transformer / GNN would be wasted cost here.",
)

E2 = Expert(
    name="E2", title="website / app-ID", dataset="cstnet", target="app",
    routes_on="direct TLS with an observable handshake (CSTNET-like, data-rich)",
    members=(
        _GBDT,
        MemberSpec("byte_net", "bytes", role="high-capacity member for 120 fine-grained classes",
                   driver="script", script="python scripts/score_byte_net.py"),
        MemberSpec("graph_gnn", "byte_matrix", role="relational view of the same bytes",
                   driver="script", script="python scripts/train_graph_gnn.py --model graph_gnn --split <cache>"),
    ),
    notes="Fine + data-rich (120 x ~475). The GBDT floor is 0.81 macro-F1 and must be beaten.",
)

E3 = Expert(
    name="E3", title="in-app activity", dataset="mobileapp", target="activity",
    routes_on="Wi-Fi / mobile-app captures",
    members=(
        _GBDT,
        MemberSpec("seq_cnn", "packet_seq", {"n": 256, "channels": 2}, role="light sequence model"),
        MemberSpec("flow_image_cnn", "flow_image", {"construction": "flowpic", "channels": 3},
                   windows=(5.0, 10.0), role="FlowPic on 5/10 s windows (median capture is 11 s)"),
    ),
    test_size=0.25,
    notes="Fine + data-poor (92 x 4 captures): few-shot; deep transformers / GNNs overfit here.",
)

EXPERTS: dict[str, Expert] = {e.name: e for e in (E1, E2, E3)}


def get_expert(name: str) -> Expert:
    if name not in EXPERTS:
        raise KeyError(f"unknown expert {name!r}; available: {sorted(EXPERTS)}")
    return EXPERTS[name]


__all__ = ["E1", "E2", "E3", "EXPERTS", "Expert", "MemberSpec", "get_expert"]
