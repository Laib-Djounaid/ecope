"""Graph construction and GNN architectures (PyTorch Geometric)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import coalesce, to_undirected


# --------------------------------------------------------------------------- graphs
def knn_graph(Z: np.ndarray, k: int, setting: str, train_idx: np.ndarray
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """kNN graph in (standardised) trait space. Returns edge_index [2, E] (src -> dst) and
    edge weights w = 1 / (1 + d).

    transductive: kNN over all nodes, symmetrised.
    inductive:    kNN among training nodes (symmetrised); every non-training node receives
                  directed edges from its k nearest TRAINING nodes and sends none, so training
                  node representations never depend on held-out nodes.
    """
    n = len(Z)
    if setting == "transductive":
        nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        d, j = nn_.kneighbors(Z)
        src = j[:, 1:].ravel()
        dst = np.repeat(np.arange(n), k)
        dist = d[:, 1:].ravel()
        ei = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
        ew = torch.as_tensor(dist, dtype=torch.float32)
        ei, ew = to_undirected(ei, ew, num_nodes=n, reduce="min")
    elif setting == "inductive":
        tr = np.asarray(train_idx)
        nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Z[tr])
        d, j = nn_.kneighbors(Z[tr])
        e1 = torch.as_tensor(np.vstack([tr[j[:, 1:].ravel()], np.repeat(tr, k)]), dtype=torch.long)
        w1 = torch.as_tensor(d[:, 1:].ravel(), dtype=torch.float32)
        e1, w1 = to_undirected(e1, w1, num_nodes=n, reduce="min")
        other = np.setdiff1d(np.arange(n), tr)
        d2, j2 = nn_.kneighbors(Z[other], n_neighbors=k)
        e2 = torch.as_tensor(np.vstack([tr[j2.ravel()], np.repeat(other, k)]), dtype=torch.long)
        w2 = torch.as_tensor(d2.ravel(), dtype=torch.float32)
        ei, ew = coalesce(torch.cat([e1, e2], 1), torch.cat([w1, w2]), num_nodes=n, reduce="min")
        trset = torch.zeros(n, dtype=torch.bool)
        trset[torch.as_tensor(tr)] = True
        assert trset[ei[0]].all(), "inductive graph: a non-training node sends a message"
    else:
        raise ValueError(setting)
    return ei, (1.0 / (1.0 + ew)).float()


# --------------------------------------------------------------------------- models
class EcoPE(nn.Module):
    """Ecological positional encoder: 2-layer MLP (LayerNorm, GELU) of the context vector."""

    def __init__(self, d_ctx: int, d_hidden: int = 64, d_out: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_ctx, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_out), nn.LayerNorm(d_out), nn.GELU())

    def forward(self, c):
        return self.net(c)


class GNN(nn.Module):
    """input projection -> 2 graph layers -> MLP readout (one hidden layer of 64, ReLU)."""

    def __init__(self, conv: str, d_in: int, hidden: int, heads: int, dropout: float,
                 d_ctx: int = 0, ecope_dim: int = 32):
        super().__init__()
        self.conv_type = conv
        self.ecope = EcoPE(d_ctx, 64, ecope_dim) if d_ctx else None
        d0 = d_in + (ecope_dim if d_ctx else 0)
        self.inp = nn.Sequential(nn.Linear(d0, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.drop = nn.Dropout(dropout)
        if conv == "gcn":
            self.c1, self.c2 = GCNConv(hidden, hidden), GCNConv(hidden, hidden)
            d_out = hidden
        elif conv == "sage":
            self.c1, self.c2 = SAGEConv(hidden, hidden), SAGEConv(hidden, hidden)
            d_out = hidden
        elif conv == "gat":
            # heads split a fixed total width: each head has hidden // heads channels
            assert hidden % heads == 0
            self.c1 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, edge_dim=1)
            self.c2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, edge_dim=1)
            d_out = hidden
        else:
            raise ValueError(conv)
        self.act = nn.ELU()
        self.readout = nn.Sequential(nn.Linear(d_out, 64), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(64, 1))

    def _conv(self, layer, h, ei, ew, return_attention=False):
        if self.conv_type == "gcn":
            return layer(h, ei, edge_weight=ew)
        if self.conv_type == "sage":
            return layer(h, ei)
        return layer(h, ei, edge_attr=ew.view(-1, 1), return_attention_weights=return_attention)

    def embed_input(self, data):
        x = data["x"]
        if self.ecope is not None:
            x = torch.cat([x, self.ecope(data["ctx"])], dim=1)
        return self.inp(x)

    def forward(self, data, return_attention: bool = False):
        h = self.drop(self.embed_input(data))
        ei, ew = data["edge_index"], data["edge_weight"]
        h = self.act(self._conv(self.c1, h, ei, ew))
        h = self.drop(h)
        if return_attention and self.conv_type == "gat":
            h, att = self._conv(self.c2, h, ei, ew, return_attention=True)
            return self.readout(self.act(h)).squeeze(-1), att
        h = self.act(self._conv(self.c2, h, ei, ew))
        return self.readout(h).squeeze(-1)
