"""TFE-GNN baseline: graph construction fidelity and network invariants.

The centrepiece is :func:`_reference_construct_graph`, a verbatim port of
``construct_graph`` from the authors' repository (github.com/ViktorAxelsen/TFE-GNN,
``utils.py``) with the DGL calls unrolled.  Our vectorised implementation is checked
against it directly, because graph construction is the part of TFE-GNN that the ablations
show is not recoverable downstream -- a wrong edge rule yields a model that trains
happily and reports the wrong number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qsentinel.features.traffic_graph import (
    PAD_TRUNC_DIGIT,
    build_byte_graph,
    compute_pmi_edges,
    pad_truncate_part,
    strip_addressing,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def _reference_construct_graph(byte_values: list[int], w_size: int, k: int = 1):
    """Verbatim port of the authors' ``construct_graph``; ``dgl`` calls unrolled.

    Returns ``(node_values_in_reference_order, {(local_src, local_dst)})``.
    """
    words = list(byte_values)
    length = len(words)
    windows: list[list[int]] = []
    if length <= w_size:
        windows.append(words)
    else:
        for j in range(length - w_size + 1):
            windows.append(words[j : j + w_size])

    word_window_freq: dict[int, int] = {}
    for window in windows:
        appeared: set[int] = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            word_window_freq[window[i]] = word_window_freq.get(window[i], 0) + 1
            appeared.add(window[i])

    word_pair_count: dict[str, int] = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                w_i, w_j = window[i], window[j]
                if w_i == w_j:
                    continue
                for key in (f"{w_i},{w_j}", f"{w_j},{w_i}"):
                    word_pair_count[key] = word_pair_count.get(key, 0) + 1

    src: list[int] = []
    dst: list[int] = []
    num_window = len(windows)
    for key, count in word_pair_count.items():
        a, b = key.split(",")
        i, j = int(a), int(b)
        pmi = math.log(
            (1.0 * count / num_window) ** k
            / (1.0 * word_window_freq[i] * word_window_freq[j] / (num_window * num_window))
        )
        if pmi <= 0:
            continue
        src.append(i)
        dst.append(j)

    bytes2id: dict[int, int] = {}
    feat: list[int] = []
    for byte in src:  # node ids follow first appearance in src
        if byte not in bytes2id:
            bytes2id[byte] = len(feat)
            feat.append(byte)

    edges = {(bytes2id[a], bytes2id[b]) for a, b in zip(src, dst)}
    edges |= {(n, n) for n in range(len(feat))}  # dgl.add_self_loop
    return feat, edges


def _as_value_edges(node_values: np.ndarray, edge_index: np.ndarray) -> set[tuple[int, int]]:
    """Edges as (byte value, byte value), so local node numbering cannot matter."""
    return {(int(node_values[a]), int(node_values[b])) for a, b in edge_index.T.tolist()}


@pytest.mark.parametrize("window", [2, 5, 9])
def test_matches_reference_construct_graph(window: int) -> None:
    rng = np.random.default_rng(1234)
    for _ in range(150):
        n = int(rng.integers(1, 160))
        alphabet = int(rng.integers(2, 60))
        values = rng.integers(0, alphabet, size=n).astype(np.int16)
        if rng.random() < 0.4:  # exercise PAD taking part in the graph
            values = pad_truncate_part(values, 150)

        node_values, edge_index = compute_pmi_edges(values, window=window)
        ref_feat, ref_edges = _reference_construct_graph(values.tolist(), window)

        assert set(node_values.tolist()) == set(ref_feat)
        ref_map = dict(enumerate(ref_feat))
        assert _as_value_edges(node_values, edge_index) == {
            (ref_map[a], ref_map[b]) for a, b in ref_edges
        }


def test_every_node_has_a_self_loop() -> None:
    """``dgl.add_self_loop`` is applied after filtering; ours must match."""
    rng = np.random.default_rng(7)
    g = build_byte_graph(rng.integers(0, 20, size=80).astype(np.int16))
    pairs = set(map(tuple, g.edge_index.T.tolist()))
    assert all((i, i) in pairs for i in range(g.n_nodes))


def test_nodes_are_only_edge_incident_values() -> None:
    """A byte that never clears PMI is dropped -- node count != distinct values."""
    rng = np.random.default_rng(3)
    values = rng.integers(0, 50, size=100).astype(np.int16)
    g = build_byte_graph(values)
    assert set(g.node_values.tolist()) <= set(values.tolist())
    non_loop = [(a, b) for a, b in g.edge_index.T.tolist() if a != b]
    incident = {a for a, _ in non_loop} | {b for _, b in non_loop}
    assert incident == set(range(g.n_nodes)), "every node must carry a real edge"


def test_pad_participates_in_the_graph() -> None:
    """PAD is a first-class byte value here, not something masked out."""
    padded = pad_truncate_part(b"\x45\x00\x02\x4f\x0a\x0d", 40)
    assert padded.tolist().count(PAD_TRUNC_DIGIT) == 34
    g = build_byte_graph(padded)
    assert PAD_TRUNC_DIGIT in g.node_values.tolist()


def test_pad_truncate_part() -> None:
    assert pad_truncate_part(bytes(range(250)), 150).tolist() == list(range(150))
    short = pad_truncate_part(b"\x01\x02", 5)
    assert short.tolist() == [1, 2, PAD_TRUNC_DIGIT, PAD_TRUNC_DIGIT, PAD_TRUNC_DIGIT]


def test_graph_degenerate_inputs() -> None:
    assert build_byte_graph(b"").is_empty()
    # A single value, or all-identical values, yields no positive-PMI pair at all, so the
    # reference produces an empty graph rather than an isolated node.
    assert build_byte_graph(np.array([7], dtype=np.int16)).is_empty()
    assert build_byte_graph(np.full(200, 0x41, dtype=np.int16)).is_empty()


def test_strip_addressing_removes_ips_and_ports() -> None:
    ip_header = bytes(range(20))  # bytes 12..19 are src/dst IPv4
    transport = bytes(range(100, 120))  # bytes 0..3 are src/dst port
    out = strip_addressing(ip_header, transport)
    assert out == bytes(range(12)) + bytes(range(100 + 4, 120))
    for addr_byte in range(12, 20):
        assert addr_byte not in out[:12]
    assert 100 not in out and 103 not in out  # port bytes gone


# -- network -------------------------------------------------------------------------


def _tiny_batch(n_flows: int, lengths: list[int], t: int, seed: int = 0):
    from torch_geometric.data import Batch, Data

    from qsentinel.features.traffic_graph import (
        BYTE_PAD_TRUNC_LENGTH,
        HEADER_BYTE_PAD_TRUNC_LENGTH,
    )

    rng = np.random.default_rng(seed)
    hg, pg = [], []
    for b in range(n_flows):
        for i in range(t):
            real = i < lengths[b]
            head = rng.integers(0, 256, size=30).astype(np.int16) if real else np.zeros(0, np.int16)
            pay = rng.integers(0, 256, size=90).astype(np.int16) if real else np.zeros(0, np.int16)
            for part, length, acc in (
                (head, HEADER_BYTE_PAD_TRUNC_LENGTH, hg),
                (pay, BYTE_PAD_TRUNC_LENGTH, pg),
            ):
                g = build_byte_graph(pad_truncate_part(part, length))
                acc.append(
                    Data(
                        x=torch.tensor(g.node_values, dtype=torch.long),
                        edge_index=torch.tensor(g.edge_index, dtype=torch.long),
                        num_nodes=g.n_nodes,
                    )
                )
    return Batch.from_data_list(hg), Batch.from_data_list(pg)


def test_param_count_matches_published_model_size() -> None:
    from qsentinel.models.tfe_gnn import TFEGNNNet

    n_params = sum(p.numel() for p in TFEGNNNet(n_classes=6).parameters())
    # ~44M is the figure reported for TFE-GNN in the literature (LGR-CE Table 6).  This
    # guards the LSTM sizing: hidden_size is the fused packet dim (1024).  A 128-wide
    # state gives a 2.9M-parameter model that will not reproduce.
    assert 40e6 < n_params < 50e6, f"expected ~44M parameters, got {n_params / 1e6:.1f}M"


def test_forward_shapes_and_gradients() -> None:
    from qsentinel.models.tfe_gnn import TFEGNNNet

    n_flows, t, n_classes = 3, 8, 6
    hb, pb = _tiny_batch(n_flows, [8, 3, 1], t)
    net = TFEGNNNet(n_classes=n_classes)

    z = net.encode_packets(
        hb.x, hb.edge_index, hb.batch, pb.x, pb.edge_index, pb.batch, n_packets=n_flows * t
    )
    assert z.shape == (n_flows * t, net.fusion.out_dim)

    logits = net(z.view(n_flows, t, -1))
    assert logits.shape == (n_flows, n_classes)

    torch.nn.functional.cross_entropy(logits, torch.zeros(n_flows, dtype=torch.long)).backward()
    assert all(p.grad is not None for p in net.parameters())


def test_empty_graphs_do_not_desync_towers() -> None:
    """Payload-less packets must not shift the header<->payload pairing."""
    from qsentinel.models.tfe_gnn import TFEGNNNet

    n_flows, t = 3, 8
    hb, pb = _tiny_batch(n_flows, [8, 1, 4], t)
    net = TFEGNNNet(n_classes=6).eval()
    z = net.encode_packets(
        hb.x, hb.edge_index, hb.batch, pb.x, pb.edge_index, pb.batch, n_packets=n_flows * t
    )
    assert z.shape[0] == n_flows * t, "one z per packet slot, empty graphs included"
    assert torch.isfinite(net(z.view(n_flows, t, -1))).all()


def test_mean_readout_requires_lengths() -> None:
    from qsentinel.models.tfe_gnn import TFEGNNNet

    net = TFEGNNNet(n_classes=6, readout="mean").eval()
    z = torch.randn(2, 5, net.fusion.out_dim)
    with pytest.raises(ValueError, match="requires lengths"):
        net(z)
    assert net(z, torch.tensor([5, 2])).shape == (2, 6)


def test_rejects_unknown_readout() -> None:
    from qsentinel.models.tfe_gnn import TFEGNNNet

    with pytest.raises(ValueError, match="readout must be"):
        TFEGNNNet(n_classes=6, readout="last")
