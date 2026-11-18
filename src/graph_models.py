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
                 d_ctx: int = 0, ecope_dim: int = 32, edge_dim: int = 1, ecope_form: str = "mlp"):
        super().__init__()
        self.conv_type = conv
        # ecological position encoder applied BEFORE message passing (D13: form is tuned)
        #   mlp = EcoPE (v1 form, 64 -> 32); linear = Linear(d_ctx, 32); identity = raw concatenation
        if not d_ctx:
            self.ecope, d_enc = None, 0
        elif ecope_form == "mlp":
            self.ecope, d_enc = EcoPE(d_ctx, 64, ecope_dim), ecope_dim
        elif ecope_form == "linear":
            self.ecope, d_enc = nn.Linear(d_ctx, ecope_dim), ecope_dim
        elif ecope_form == "identity":
            self.ecope, d_enc = nn.Identity(), d_ctx
        else:
            raise ValueError(ecope_form)
        d0 = d_in + d_enc
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
            self.c1 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, edge_dim=edge_dim)
            self.c2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout, edge_dim=edge_dim)
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
        ea = ew.view(-1, 1) if ew.dim() == 1 else ew          # 1 or several edge features
        return layer(h, ei, edge_attr=ea, return_attention_weights=return_attention)

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


# --------------------------------------------------------------------------- v2 graphs
def geo_knn_graph(latlon_deg: np.ndarray, k: int, scale_km: float, R: float = 6371.0088
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Transductive kNN graph by great-circle distance (BallTree, haversine), symmetrised.
    Edge weight w = 1 / (1 + d_km / scale_km)."""
    from sklearn.neighbors import BallTree
    n = len(latlon_deg)
    tree = BallTree(np.radians(latlon_deg), metric="haversine")
    d, j = tree.query(np.radians(latlon_deg), k=k + 1)
    src, dst = j[:, 1:].ravel(), np.repeat(np.arange(n), k)
    ei = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
    ew = torch.as_tensor(d[:, 1:].ravel() * R, dtype=torch.float32)
    ei, ew = to_undirected(ei, ew, num_nodes=n, reduce="min")
    return ei, (1.0 / (1.0 + ew / scale_km)).float()


def two_channel(ei_t, w_t, ei_g, w_g, n: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge attributes [w_trait, w_geo] (0 where the edge is absent from that graph).
    which = 'trait' | 'geo' | 'combined' (union of both edge sets)."""
    parts_e, parts_a = [], []
    if which in ("trait", "combined"):
        parts_e.append(ei_t); parts_a.append(torch.stack([w_t, torch.zeros_like(w_t)], 1))
    if which in ("geo", "combined"):
        parts_e.append(ei_g); parts_a.append(torch.stack([torch.zeros_like(w_g), w_g], 1))
    ei, ea = coalesce(torch.cat(parts_e, 1), torch.cat(parts_a, 0), num_nodes=n, reduce="add")
    return ei, ea


# --------------------------------------------------------------------------- occupancy-detection
class ODGNN(nn.Module):
    """EcoPE-OD-GNN: P(reported) = P(present) x P(detected | present).

    presence  : GNN (GAT + optional EcoPE) on the node features and ecological context;
                it never receives effort.
    detection : depends ONLY on the standardised log effort z (data['det'], shape [n, 1]):
      'cloglog' p_det = 1 - exp(-exp(a + softplus(b) * z))   monotone increasing, -> 1 at high
                effort (Poisson-thinning / effort-as-exposure form; anchors the scale)
      'mlp'     p_det = sigmoid(MLP(z))                       free small MLP (no constraint)
      'none'    p_det = 1                                     presence branch alone
    forward() returns logit(P(reported)) so the standard BCE-with-logits trainer applies
    (use pos_weight = 1: the product is a likelihood only without class weighting).
    """

    def __init__(self, presence: GNN, detection: str = "cloglog", hidden: int = 8):
        super().__init__()
        self.presence, self.detection = presence, detection
        if detection == "cloglog":
            self.a = nn.Parameter(torch.zeros(1))
            self.b_raw = nn.Parameter(torch.zeros(1))
        elif detection == "mlp":
            self.det_mlp = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        elif detection != "none":
            raise ValueError(detection)

    def log_p_det(self, data):
        z = data["det"].view(-1)
        if self.detection == "cloglog":
            rate = torch.exp(self.a + nn.functional.softplus(self.b_raw) * z)
            return torch.log((-torch.expm1(-rate)).clamp(min=1e-12))       # log(1 - exp(-rate))
        if self.detection == "mlp":
            return nn.functional.logsigmoid(self.det_mlp(z.view(-1, 1)).view(-1))
        return torch.zeros_like(z)

    def presence_logit(self, data):
        return self.presence(data)

    def forward(self, data):
        log_p = nn.functional.logsigmoid(self.presence(data)) + self.log_p_det(data)
        log_p = log_p.clamp(max=-1e-7)
        return log_p - torch.log(-torch.expm1(log_p))                       # logit(P(reported))


def time_causal(ei: torch.Tensor, ew: torch.Tensor, years: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only messages j -> i with survey_year[j] <= survey_year[i] (decision D12).

    A node's prior presence counts target records before ITS OWN survey year. A neighbour surveyed
    later can therefore carry records from the receiving plot's label window (temporal leakage,
    tools/neighbour_time_leak.py). With this filter every quantity reaching plot i, also over
    several hops, predates year_i."""
    yr = torch.as_tensor(np.asarray(years), dtype=torch.long)
    keep = yr[ei[0]] <= yr[ei[1]]
    return ei[:, keep], ew[keep]


# --------------------------------------------------------------------------- D14: as-of model
class AsOfGAT(nn.Module):
    """EcoPE-GAT_asof (decision D14, option C). One attention layer over edge-level messages whose
    time-stamped part is computed AS OF THE RECEIVER's date (src/asof_features.py), so every
    time-stamped quantity reaching node i uses records with year < t_i, with no edge removed.

    h_i     = inp([x_i, enc(ctx_i)])
    m_{j->i} = msg([xe_{j->i}, enc(ctx_j), ea_{j->i}])
    a_{ji}  = softmax_j( w^T LeakyReLU(W [h_i, m_{j->i}]) )     (over the in-edges of i)
    logit_i = readout([h_i, sum_j a_{ji} m_{j->i}])
    data: x [n, d_in], ctx [n, d_ctx], edge_index [2, E], xe [E, d_in], ea [E, d_ea].
    """

    def __init__(self, d_in: int, d_ctx: int, d_ea: int, hidden: int, dropout: float,
                 ecope_form: str = "mlp", ecope_dim: int = 32):
        super().__init__()
        if ecope_form == "mlp":
            self.ecope, d_enc = EcoPE(d_ctx, 64, ecope_dim), ecope_dim
        elif ecope_form == "linear":
            self.ecope, d_enc = nn.Linear(d_ctx, ecope_dim), ecope_dim
        elif ecope_form == "identity":
            self.ecope, d_enc = nn.Identity(), d_ctx
        else:
            raise ValueError(ecope_form)
        self.inp = nn.Sequential(nn.Linear(d_in + d_enc, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.msg = nn.Sequential(nn.Linear(d_in + d_enc + d_ea, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.att = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(64, 1))

    def forward(self, data, return_attention: bool = False):
        from torch_geometric.utils import scatter, softmax
        src, dst = data["edge_index"]
        c = self.ecope(data["ctx"])
        h = self.drop(self.inp(torch.cat([data["x"], c], 1)))
        m = self.drop(self.msg(torch.cat([data["xe"], c[src], data["ea"]], 1)))
        a = softmax(self.att(torch.cat([h[dst], m], 1)).squeeze(-1), dst, num_nodes=h.shape[0])
        agg = scatter(a.unsqueeze(-1) * m, dst, dim=0, dim_size=h.shape[0], reduce="sum")
        out = self.readout(torch.cat([h, agg], 1)).squeeze(-1)
        return (out, a) if return_attention else out
