"""Byte-level traffic graphs (TFE-GNN, WWW'23) -> the Phase-4 graph member.

This is the paper's actual contribution, and the part worth getting exactly right: the
backbone is interchangeable (Zhang et al. Fig. 2 puts GAT/GIN/GCN/SGC within a few points
of GraphSAGE), but a wrong edge rule produces a model that trains fine and reports the
wrong number.

Verified line-by-line against the authors' ``construct_graph`` in
github.com/ViktorAxelsen/TFE-GNN ``utils.py``, because the paper's prose under-specifies
it in ways that matter.  Five details are load-bearing and none are stated in the paper:

1. **Pair counts use multiplicity, not window occupancy.**  Within each window the
   reference enumerates every ordered *position* pair, so a window ``[A, A, B]``
   contributes **2** to ``count(A,B)``, not 1.  Marginals ``freq(a)`` meanwhile *are*
   window occupancy -- counted once per window.  The asymmetry is easy to miss and
   changes which edges clear the threshold.
2. **Only bytes with at least one surviving edge become nodes.**  A byte value that
   appears but never clears PMI is dropped from the graph entirely, so node count is
   not the number of distinct values.
3. **Self-loops are added to every node** (``dgl.add_self_loop``) after edge filtering.
4. **Padding is part of the graph.**  Sequences are padded to a fixed length with
   ``PAD_TRUNC_DIGIT = 256`` -- deliberately outside the byte range -- and those PAD
   tokens participate in PMI like any other value.  The embedding therefore spans 257
   values, not 256.
5. **The header/payload split is protocol-aware**, not a fixed byte offset: payload is
   whatever is in the transport layer's data, header is everything before it.

PMI is estimated *per packet part*, over that part's own windows -- not from a global
corpus table.  Contrast Okonkwo et al. (Inf. Sci. 2026), whose mutual-information edges
come from a dataset-wide lookup; their Table 10 shows that design losing 8.2 points under
WTF-PAD when the table is built from partial data.  TFE-GNN has no such coupling, so
graphs can be built streaming, one packet at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Padding token.  Deliberately outside ``0..255`` so a padded byte is never confusable
#: with a real one -- the authors' ``config.PAD_TRUNC_DIGIT``.
PAD_TRUNC_DIGIT = 256

#: Embedding cardinality: the 256 byte values plus :data:`PAD_TRUNC_DIGIT`.
N_BYTE_VALUES = 257

#: Fixed sample geometry (paper Sec. 4.1.3 and the authors' config, which agree).
FLOW_PAD_TRUNC_LENGTH = 50
BYTE_PAD_TRUNC_LENGTH = 150
HEADER_BYTE_PAD_TRUNC_LENGTH = 40
ANOMALOUS_FLOW_THRESHOLD = 10_000
PMI_WINDOW_SIZE = 5


@dataclass(slots=True)
class ByteGraph:
    """One packet part as a graph.

    Attributes
    ----------
    node_values
        ``(V,)`` int16 -- the byte values that survived edge filtering, ascending.  This
        doubles as the node feature: node ``v`` is initialised with its byte value, which
        the embedding layer looks up.  int16 rather than uint8 because
        :data:`PAD_TRUNC_DIGIT` does not fit in a byte.
    edge_index
        ``(2, E)`` int64 in **local** node indices, both directions present, self-loops
        included.
    """

    node_values: np.ndarray
    edge_index: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.node_values.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def is_empty(self) -> bool:
        return self.n_nodes == 0


def _windows_matrix(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """``(present, counts)`` where ``counts`` is ``(n_windows, V)`` int32.

    ``counts[w, i]`` is how many *positions* in window ``w`` hold ``present[i]`` -- the
    multiplicity the reference's nested position loop implies.  Occupancy (``counts > 0``)
    gives the marginals.
    """
    present, local = np.unique(values, return_inverse=True)
    n, v = values.shape[0], present.shape[0]

    # The reference branches on `length <= window_size` -> a single window covering
    # everything.  Otherwise a full stride-1 sweep.
    if n <= window:
        counts = np.zeros((1, v), dtype=np.int32)
        np.add.at(counts, (0, local), 1)
        return present, counts

    n_win = n - window + 1
    counts = np.zeros((n_win, v), dtype=np.int32)
    rows = np.arange(n_win)
    for offset in range(window):
        np.add.at(counts, (rows, local[offset : offset + n_win]), 1)
    return present, counts


def compute_pmi_edges(
    values: np.ndarray, window: int = PMI_WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """``(present_values, edge_index)`` for byte values joined by positive PMI.

    Follows the reference exactly::

        count(a,b) = sum over windows of n_w[a] * n_w[b]        (a != b, multiplicity)
        freq(a)    = number of windows containing a at least once
        PMI        = log( (count/W)^k / (freq_a * freq_b / W^2) )      with k = 1

    An edge survives iff ``PMI > 0``.  With ``k = 1`` that is exactly
    ``count * W > freq_a * freq_b``, which we test in integer arithmetic -- the float
    ``log`` form rounds exact ties to a hair above zero and would admit edges the
    threshold excludes.

    Nodes are then the values that appear in at least one surviving edge, and every node
    receives a self-loop.
    """
    if values.size == 0:
        return np.zeros(0, dtype=np.int16), np.zeros((2, 0), dtype=np.int64)

    present, counts = _windows_matrix(values, window)
    v, n_win = present.shape[0], counts.shape[0]

    occupancy = (counts > 0).sum(axis=0).astype(np.int64)  # freq(a)
    pair = counts.astype(np.int64).T @ counts.astype(np.int64)  # count(a,b), multiplicity
    np.fill_diagonal(pair, 0)  # the reference skips word_i == word_j

    keep = pair * n_win > np.outer(occupancy, occupancy)
    src, dst = np.nonzero(keep)
    if src.size == 0:
        return np.zeros(0, dtype=np.int16), np.zeros((2, 0), dtype=np.int64)

    # Only values incident to a surviving edge become nodes; relabel to a dense range.
    node_local = np.unique(src)
    remap = np.full(v, -1, dtype=np.int64)
    remap[node_local] = np.arange(node_local.shape[0])

    edges = np.stack([remap[src], remap[dst]])
    loops = np.arange(node_local.shape[0], dtype=np.int64)
    edge_index = np.concatenate([edges, np.stack([loops, loops])], axis=1)

    return present[node_local].astype(np.int16), edge_index


def build_byte_graph(part, window: int = PMI_WINDOW_SIZE) -> ByteGraph:
    """Build the byte-level traffic graph for one (already padded) packet part."""
    values = (
        np.frombuffer(part, dtype=np.uint8).astype(np.int16)
        if isinstance(part, (bytes, bytearray))
        else np.asarray(part, dtype=np.int16)
    )
    node_values, edge_index = compute_pmi_edges(values, window=window)
    return ByteGraph(node_values, edge_index)


def pad_truncate_part(part, length: int) -> np.ndarray:
    """Truncate to ``length``, or pad to it with :data:`PAD_TRUNC_DIGIT`.

    Padding is not cosmetic here.  Because PAD participates in graph construction, a
    short packet's graph genuinely contains a PAD node wired to whatever it co-occurs
    with -- that is the reference's behaviour, and dropping the padding instead would
    change every short packet's topology.
    """
    values = (
        np.frombuffer(part, dtype=np.uint8).astype(np.int16)
        if isinstance(part, (bytes, bytearray))
        else np.asarray(part, dtype=np.int16)
    )
    if values.shape[0] >= length:
        return values[:length]
    out = np.full(length, PAD_TRUNC_DIGIT, dtype=np.int16)
    out[: values.shape[0]] = values
    return out


def strip_addressing(ip_header: bytes, transport_header: bytes) -> bytes:
    """Drop the source/destination IP addresses and ports from a packet header.

    The paper says the Ethernet header, both IP addresses and both port numbers are
    removed (Sec. 4.1.2), which is what this does: IPv4 bytes 12..19 are the address
    pair, and transport bytes 0..3 are the port pair.

    .. warning::
       The authors' ``remove()`` slices ``p[:12]`` and ``p[20:][4:]`` from an array that
       **still carries the 14-byte Ethernet header** (``pcap2npy.py`` hexdumps the whole
       frame).  Applied at that offset the slice removes the MAC trailer and parts of the
       IP header instead, and leaves the addresses and ports in place.  We implement the
       paper's stated intent rather than replicate that offset, because a model that can
       see server IPs is not doing traffic classification.  If a strict byte-for-byte
       replication is ever needed to chase a delta, it belongs behind an explicit flag,
       not as the default.
    """
    return bytes(ip_header[:12]) + bytes(ip_header[20:]) + bytes(transport_header[4:])
