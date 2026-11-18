"""H4 (pre-registration D13, section 3): dose-response and sensitivity, development data only.

H4a  remove a nested random fraction q of the non-causal edges (t_j > t_i) of the geo kNN graph,
     q in {0, .25, .5, .75, 1}; EcoPE-GAT with the FROZEN per-fold hyperparameters of
     EcoPE-GAT_leaky (results/d13/params), 5 seeds, N = 5; statistic = spatial-CV fold-mean PR-AUC
     of the seed-mean score; least-squares slope on q, 95% block-bootstrap CI (2,000, seed 0).
     q = 0 is the D13 EcoPE-GAT_leaky refit (identical graph, hyperparameters and seeds; reused).
H4b  development positives: Spearman(share of the plot's in-neighbours surveyed later,
     out-of-fold percentile-rank gain leaky - causal) from the D13 cv refits; block-bootstrap CI.
H4c  descriptive: D1, D2 (N = 5/7/10) and the leaky - causal fold-mean gap at N = 5 for geo kNN
     k in {5, 20} and radius graphs 50 / 100 km (capped at 50 nearest); each side uses its own D13
     per-fold hyperparameters (leaky / causal). k = 10 and N = 7/10 gaps come from the D13 refits.
Outputs: results/h4/refits (checkpoints), reports/dev_results/h4*.{csv,json}.

Usage: python src/13_h4.py [--parts a b c] [--threads 8]
"""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score as AP
from sklearn.neighbors import BallTree
from torch_geometric.utils import to_undirected

from checkpoint import cached_refit
from common import ROOT, load_config, p, variant_name
from features import FoldPreprocessor, assemble, labels, load_modelling_table, raw_blocks
from graph_models import GNN, geo_knn_graph, time_causal
from leakage_diagnostic import d1, d2
from nn_train import predict, train_full_batch

d13 = importlib.import_module("09_d13")
QS = [0.0, 0.25, 0.5, 0.75, 1.0]


def radius_graph(ll, r_km, cap, scale, R):
    """Neighbours within r_km (at most `cap` nearest), symmetrised; w = 1/(1 + d/scale)."""
    tree = BallTree(np.radians(ll), metric="haversine")
    d, j = tree.query(np.radians(ll), k=cap + 1)
    d, j = d[:, 1:] * R, j[:, 1:]
    keep = (d <= r_km).ravel()
    dst = np.repeat(np.arange(len(ll)), cap)[keep]
    ei = torch.as_tensor(np.vstack([j.ravel()[keep], dst]), dtype=torch.long)
    ew = torch.as_tensor(d.ravel()[keep], dtype=torch.float32)
    ei, ew = to_undirected(ei, ew, num_nodes=len(ll), reduce="min")
    return ei, (1.0 / (1.0 + ew / scale)).float()


def boot_indices(e, ctxs, folds, n_boot, seed=0):
    """Block-bootstrap index sets (within each test fold), shared by all compared models."""
    blocks = e["block_id"].to_numpy()
    members = []
    for c in folds:
        te = ctxs[c][2]
        ub, inv = np.unique(blocks[te], return_inverse=True)
        members.append([np.flatnonzero(inv == u) for u in range(len(ub))])
    rng = np.random.default_rng(seed)
    return [[np.concatenate([m[u] for u in rng.integers(0, len(m), len(m))]) for m in members]
            for _ in range(n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--parts", nargs="+", default=["a", "b", "c"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config()
    dc = cfg["d13"]
    res, rep = p(cfg, "results_dir"), ROOT / "reports" / "dev_results"
    rep.mkdir(parents=True, exist_ok=True)
    e = load_modelling_table(cfg)
    assert not e["holdout"].any()
    t = e["survey_year"].to_numpy().astype(int)
    Y = {N: labels(e, cfg, N) for N in dc["horizons"]}
    blk = raw_blocks(e, cfg)
    ctxs = d13.contexts(e, cfg)
    folds = [f"cv{k}" for k in range(cfg["evaluation"]["n_folds"])]
    ll = e[["Latitude", "Longitude"]].to_numpy()
    R, B, K = cfg["spatial"]["earth_radius_km"], cfg["spatial"]["buffer_km"], cfg["graph"]["k"]
    ei0, ew0 = geo_knn_graph(ll, K, B, R)
    prm = {(m, c): json.load(open(res / "d13" / "params" / f"{m}_{c}.json"))["best_params"]
           for m in ["EcoPE-GAT_leaky", "EcoPE-GAT_causal"] for c in folds}

    def d13_score(m, c, N):
        return np.mean([np.load(res / "d13" / "refits" / f"{m}_{c}_N{N}_s{s}.npz")["test"]
                        for s in dc["seeds"]], 0)

    def train(tag, c, ei, ew, hp, N=5):
        tr, iv, te = ctxs[c]
        x = FoldPreprocessor().fit(assemble(blk, d13.NODE), tr).transform(assemble(blk, d13.NODE))
        C = FoldPreprocessor().fit(assemble(blk, d13.CTX), tr).transform(assemble(blk, d13.CTX))
        data = {"x": torch.as_tensor(x), "ctx": torch.as_tensor(C), "edge_index": ei, "edge_weight": ew}
        y = Y[N]
        scores = []
        for sd in dc["seeds"]:
            def fit(sd=sd):
                torch.manual_seed(sd)
                net = GNN("gat", x.shape[1], hp["hidden"], hp["heads"], hp["dropout"], d_ctx=C.shape[1],
                          ecope_form=hp["encoder"])
                net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=hp["lr"],
                                             weight_decay=hp["weight_decay"],
                                             pos_weight=float((y[tr] == 0).sum() / (y[tr] == 1).sum()),
                                             max_epochs=dc["max_epochs"], patience=dc["patience"], seed=sd)
                s = predict(net, data)
                return ({"test": s[te].astype(np.float32)},
                        {"tag": tag, "context": c, "N": N, "seed": sd, "test_pr_auc": float(AP(y[te], s[te])),
                         **{k: v for k, v in info.items() if np.isscalar(v)}})
            arrs, _ = cached_refit(res / "h4" / "refits" / f"{tag}_{c}_N{N}_s{sd}", fit)
            scores.append(arrs["test"])
        return np.mean(scores, 0)

    BOOT = boot_indices(e, ctxs, folds, dc["n_boot"]) if {"a", "c"} & set(a.parts) else None

    def fold_mean(S, N, ix=None):
        v = []
        for f, c in enumerate(folds):
            y, s = Y[N][ctxs[c][2]], S[c]
            if ix is not None:
                y, s = y[ix[f]], s[ix[f]]
            if y.sum():
                v.append(AP(y, s))
        return float(np.mean(v))

    # ------------------------------------------------------------------ H4a
    if "a" in a.parts:
        src, dst = ei0.numpy()
        nc = np.flatnonzero(t[src] > t[dst])
        perm = np.random.default_rng(1000).permutation(nc)          # one nested removal order
        S = {}
        for q in QS:
            keep = np.ones(ei0.shape[1], dtype=bool)
            keep[perm[: int(round(q * len(nc)))]] = False
            S[q] = {c: (d13_score("EcoPE-GAT_leaky", c, 5) if q == 0 else
                        train(f"h4a_q{q:.2f}", c, ei0[:, keep], ew0[keep], prm[("EcoPE-GAT_leaky", c)]))
                    for c in folds}
            print(f"H4a q={q}: fold-mean {fold_mean(S[q], 5):.4f}", flush=True)
        obs = [fold_mean(S[q], 5) for q in QS]
        slope = lambda yv: float(np.polyfit(QS, yv, 1)[0])
        bs = [slope([fold_mean(S[q], 5, ix) for q in QS]) for ix in BOOT]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out = {"q": QS, "fold_mean_pr_auc": obs, "slope": slope(obs), "ci": [float(lo), float(hi)],
               "supported": bool(hi < 0), "n_noncausal_edges": int(len(nc)), "removal_seed": 1000}
        json.dump(out, open(rep / "h4a.json", "w"), indent=1)
        print("H4a", out, flush=True)

    # ------------------------------------------------------------------ H4b
    if "b" in a.parts:
        _, share = d1(ei0.numpy(), t)
        rows = []
        for c in folds:
            te = ctxs[c][2]
            rl = rankdata(d13_score("EcoPE-GAT_leaky", c, 5)) / len(te)
            rc = rankdata(d13_score("EcoPE-GAT_causal", c, 5)) / len(te)
            pos = Y[5][te] == 1
            rows.append(pd.DataFrame({"idx": te[pos], "share_later": share[te[pos]], "gain": (rl - rc)[pos]}))
        df = pd.concat(rows, ignore_index=True)
        df["block"] = e["block_id"].to_numpy()[df["idx"]]
        rho = spearmanr(df.share_later, df.gain).statistic
        grp = {b: g for b, g in df.groupby("block")}
        ub = np.array(list(grp))
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(dc["n_boot"]):
            g = pd.concat([grp[b] for b in rng.choice(ub, len(ub))])
            bs.append(spearmanr(g.share_later, g.gain).statistic)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out = {"n_positives": int(len(df)), "n_blocks": int(len(ub)), "spearman": float(rho),
               "ci": [float(lo), float(hi)], "supported": bool(lo > 0)}
        json.dump(out, open(rep / "h4b.json", "w"), indent=1)
        print("H4b", out, flush=True)

    # ------------------------------------------------------------------ H4c
    if "c" in a.parts:
        V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
        rec = pd.read_parquet(p(cfg, "processed_dir") / f"gbif_filtered_{V}.parquet",
                              columns=["decimalLatitude", "decimalLongitude", "year"])
        rec = rec[rec["year"].notna()]
        ry = rec["year"].to_numpy().astype(int)
        near = BallTree(np.radians(rec[["decimalLatitude", "decimalLongitude"]].to_numpy()),
                        metric="haversine").query_radius(np.radians(ll), r=B / R)
        near_years = [ry[ix] for ix in near]
        graphs = {"knn5": geo_knn_graph(ll, 5, B, R), "knn10": (ei0, ew0), "knn20": geo_knn_graph(ll, 20, B, R),
                  "radius50": radius_graph(ll, 50, 50, B, R), "radius100": radius_graph(ll, 100, 50, B, R)}
        rows = []
        for g, (ei, ew) in graphs.items():
            r1, _ = d1(ei.numpy(), t)
            for N in dc["horizons"]:
                r2, _ = d2(ei.numpy(), t, Y[N], near_years, N)
                row = {"graph": g, **r1, **r2}
                L = None
                if g == "knn10":
                    L = {c: d13_score("EcoPE-GAT_leaky", c, N) for c in folds}
                    Cs = {c: d13_score("EcoPE-GAT_causal", c, N) for c in folds}
                elif N == 5:
                    eic, ewc = time_causal(ei, ew, t)
                    L = {c: train(f"h4c_{g}_leaky", c, ei, ew, prm[("EcoPE-GAT_leaky", c)]) for c in folds}
                    Cs = {c: train(f"h4c_{g}_causal", c, eic, ewc, prm[("EcoPE-GAT_causal", c)]) for c in folds}
                if L is not None:
                    row.update({"leaky": fold_mean(L, N), "causal": fold_mean(Cs, N)})
                    row["gap"] = row["leaky"] - row["causal"]
                rows.append(row)
                print(row, flush=True)
                pd.DataFrame(rows).to_csv(rep / "h4c.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
