"""TFE-GNN (Zhang et al., WWW'23) -- the Phase-4 graph member's paper baseline.

Reference: "TFE-GNN: A Temporal Fusion Encoder Using Graph Neural Networks for
Fine-grained Encrypted Traffic Classification", arXiv:2307.16713, and the authors'
implementation at github.com/ViktorAxelsen/TFE-GNN.

Shape of the thing, per packet in a flow:

    header bytes -> byte graph -\
                                 |-- dual embed -> 4x GraphSAGE -> JKN -> mean pool
    payload bytes -> byte graph -/                        (two towers, no shared weights)
                                          |
                                          v
                            cross-gated fusion -> z  (one vector per packet)
                                          |
                    z_1 .. z_n  ->  BiLSTM(2) -> 2-layer PReLU classifier -> logits

Graph construction lives in :mod:`countdown.features.traffic_graph`; this module is only
the network.

**Where the paper and the authors' code disagree**, we follow the code, because the code
is what produced the published numbers:

===========================  ==============  ================
Hyperparameter               Paper           Repo (used here)
===========================  ==============  ================
Dual embedding dimension     50 (Sec. 4.6)   64
Max training epochs          120             20
Batch size                   512             102 x 5 accum
===========================  ==============  ================

Both agree on: PMI window 5, <=50 packets/sample, 40 header bytes, 150 payload bytes,
lr 1e-2 -> 1e-4 with 0.1 warmup, dropout 0.2, overlong-flow threshold 10000 packets.
Downstream dropout is 0.0 in the authors' config, so the classifier has none.

Eq. 10 reads ``CE(Classifier(LSTM(z_1..z_n)), y)`` and the text only says "its output
vectors are fed into a two-layer linear classifier", which does not disambiguate final
state from pooling.  The authors' code settles it: ``cat(h_n[-1], h_n[-2])`` -- the last
layer's two directions.  Since samples are padded to exactly 50 packets, every sequence
is full length and that final state is well defined without any masking.  A masked-mean
alternative is available via ``readout="mean"`` so the choice can be scored.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from countdown.features.traffic_graph import N_BYTE_VALUES


class TrafficGraphEncoder(nn.Module):
    """One tower: byte values -> graph vector.

    Stacked GraphSAGE with PReLU + BatchNorm, then a Jumping-Knowledge-style
    concatenation of every layer's output before mean pooling (Eqs. 4-6).

    The JKN concat is not decoration.  The paper caps the stack at 4 layers because of
    over-smoothing, and concatenating the per-layer outputs is what preserves the shallow
    representations that a 4-deep mean would otherwise blur -- their ablation puts it at
    roughly 1.6 F1 points on ISCX-VPN.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(N_BYTE_VALUES, embedding_dim)
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = embedding_dim
        for _ in range(n_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            # PReLU with per-channel slopes: the paper leans on this deliberately, calling
            # the per-channel negative scale "similar to an attention mechanism".  A plain
            # ReLU here is one of the ablations that collapses the model (F1 0.53).
            self.acts.append(nn.PReLU(num_parameters=hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * n_layers

    def forward(
        self,
        node_values: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        n_graphs: int | None = None,
    ) -> torch.Tensor:
        """``(n_graphs, out_dim)`` graph vectors.

        ``node_values`` is ``(n_nodes,)`` int64 byte values, ``batch`` maps each node to
        its graph in the disjoint-union minibatch.

        ``n_graphs`` must be passed whenever the batch can contain **empty** graphs -- a
        packet with no payload bytes yields one.  Without it, pooling sizes the output
        from ``batch.max() + 1`` and silently returns *fewer* rows than there are packets,
        which then pairs header graph ``i`` with payload graph ``j != i`` in the fusion
        step.  That misalignment does not raise unless the two towers happen to differ in
        count, so it can survive as a plausible-but-wrong accuracy number.  Empty graphs
        pool to a zero vector instead, which is the honest representation of "no bytes".
        """
        h = self.embedding(node_values)
        per_layer = []
        for conv, act, norm in zip(self.convs, self.acts, self.norms):
            h = norm(act(conv(h, edge_index)))
            per_layer.append(h)
        h = torch.cat(per_layer, dim=-1)  # JKN-like concat (Eq. 5)
        h = self.dropout(h)
        return global_mean_pool(h, batch, size=n_graphs)  # Eq. 6


class CrossGatedFusion(nn.Module):
    """Fuse header and payload graph vectors (Eqs. 7-9).

    Each side computes a gate from *itself* and applies it to the *other* side, then the
    two filtered vectors are concatenated.  The crossing is the point: the header
    describes the packet, so it is the header's gate that says which payload channels
    matter, and vice versa.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_h = nn.Sequential(nn.Linear(dim, dim), nn.PReLU(), nn.Linear(dim, dim))
        self.gate_p = nn.Sequential(nn.Linear(dim, dim), nn.PReLU(), nn.Linear(dim, dim))
        self.out_dim = dim * 2

    def forward(self, g_h: torch.Tensor, g_p: torch.Tensor) -> torch.Tensor:
        s_h = torch.sigmoid(self.gate_h(g_h))
        s_p = torch.sigmoid(self.gate_p(g_p))
        return torch.cat([s_h * g_p, s_p * g_h], dim=-1)  # Eq. 9


class TFEGNNNet(nn.Module):
    """The full network: per-packet byte graphs -> per-flow class logits.

    Parameters
    ----------
    readout
        ``"last_state"`` (default) reproduces the authors' ``cat(h_n[-1], h_n[-2])``.
        ``"mean"`` masked-averages real packet positions instead and requires
        ``lengths``; it is an alternative to score, not the paper's design.
    """

    def __init__(
        self,
        n_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_gnn_layers: int = 4,
        lstm_hidden: int | None = None,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        readout: str = "last_state",
    ) -> None:
        super().__init__()
        if readout not in ("last_state", "mean"):
            raise ValueError(f"readout must be 'last_state' or 'mean', got {readout!r}")
        self.readout = readout

        # Two towers, same architecture, deliberately *not* shared -- that separation is
        # the "dual" in dual embedding, worth 3.63 F1 on ISCX-VPN in their ablation.
        self.header_encoder = TrafficGraphEncoder(embedding_dim, hidden_dim, n_gnn_layers, dropout)
        self.payload_encoder = TrafficGraphEncoder(embedding_dim, hidden_dim, n_gnn_layers, dropout)

        self.fusion = CrossGatedFusion(self.header_encoder.out_dim)

        # The authors size the LSTM state to the fused packet vector itself
        # (``hidden_size = gcn_out_dim * 2``), not to some smaller bottleneck.  That one
        # line is most of the model: it puts TFE-GNN at ~44M parameters, matching the
        # figure reported for it in the literature, where a 128-wide state would give
        # ~2.9M.  A 15x-undersized network is the kind of "faithful" reimplementation
        # that quietly fails to reproduce and gets blamed on hyperparameters.
        lstm_hidden = self.fusion.out_dim if lstm_hidden is None else int(lstm_hidden)
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=self.fusion.out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Reference `fc` + `cls`: Linear(4*gcn_out -> gcn_out) + PReLU, then the class
        # head.  No dropout here -- the authors ship DOWNSTREAM_DROPOUT = 0.0, and adding
        # one is a silent deviation that shows up as a reproduction gap.
        clf_hidden = self.header_encoder.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, clf_hidden),
            nn.PReLU(clf_hidden),
            nn.Linear(clf_hidden, n_classes),
        )

    def encode_packets(
        self,
        h_values: torch.Tensor,
        h_edge_index: torch.Tensor,
        h_batch: torch.Tensor,
        p_values: torch.Tensor,
        p_edge_index: torch.Tensor,
        p_batch: torch.Tensor,
        n_packets: int | None = None,
    ) -> torch.Tensor:
        """``(n_packets, fusion_dim)`` -- one representation vector ``z`` per packet.

        Pass ``n_packets`` (header and payload graph counts are equal by construction, one
        of each per packet) so empty graphs cannot desynchronise the two towers.
        """
        g_h = self.header_encoder(h_values, h_edge_index, h_batch, n_packets)
        g_p = self.payload_encoder(p_values, p_edge_index, p_batch, n_packets)
        if g_h.size(0) != g_p.size(0):
            raise RuntimeError(
                f"header/payload tower desync: {g_h.size(0)} vs {g_p.size(0)} graphs. "
                f"Pass n_packets to encode_packets()."
            )
        return self.fusion(g_h, g_p)

    def forward(self, z: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, n_classes)`` logits from ``(B, T, fusion_dim)`` packet vectors.

        ``readout="last_state"`` (the default, and what the authors do) needs no
        ``lengths``: samples are padded to exactly ``FLOW_PAD_TRUNC_LENGTH`` packets with
        all-PAD packets, so every sequence is genuinely full length and the final hidden
        state is well defined.  The PAD packets are not noise to be masked away -- they
        encode "this flow was short", which is information the model is entitled to.

        ``readout="mean"`` is the masked alternative and *does* require ``lengths``.  It
        is not the paper's design; it is here so the choice can be scored rather than
        argued about.
        """
        if self.readout == "mean":
            if lengths is None:
                raise ValueError("readout='mean' requires lengths")
            packed = nn.utils.rnn.pack_padded_sequence(
                z, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=z.size(1)
            )
            mask = (
                torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
            ).unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1)
        else:
            _, (h_n, _) = self.lstm(z)
            # Last layer's two directions, in the reference's order.
            pooled = torch.cat([h_n[-1], h_n[-2]], dim=-1)

        return self.classifier(pooled)
