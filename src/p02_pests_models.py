"""D15 step 2: tuning and development refits of the pest replication (reports/preregistration_D15*.md).

Development data only (data/processed/pests/<case>_dev.parquet; the hold-out file is never read here).
Features (from src/p01_build_pests.py):
  NODE  prior_log_n = log1p(n_prior), prior_any, prior_years, effort_log = log1p(effort_prior)
  CTX   bio1, bio5, bio6, bio12, lat, lon
  G_lgbm = NODE + CTX; G_lgbm_nbr(_leaky) adds the kNN neighbour block (causal / all neighbours)
  and the radius block (target records within 100 km and in the 50-100 km ring, year < t_i).
Graph: geographic kNN (k = graph.k, symmetrised, w = 1/(1 + d/50 km)); causal rule t_j <= t_i;
as-of edge features (D14) recomputed here for pests with a self-check (j = i reproduces NODE).
Protocol (pre-registration D15, section 2): per (model, case) one Optuna TPE study (seed 0, 20 trials)
on the blocked all-development inner validation at N = 3; these hyperparameters are reused for the 5
spatial folds (train = other folds minus inner validation f_k, early stopping on it) and, later, for
the hold-out refit. Refits N = 3 and 5, seeds 0-2. N = 5 labels are undefined for t > 2020: such nodes
stay in the graph but are excluded from training, validation and scoring.
Outputs: results/pests/<case>/{optuna,params,refits}, reports/pests/<case>_diagnostics.json.

Usage: python src/p02_pests_models.py P1_halys_europe [--models ...] [--threads 8] [--smoke]
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import average_precision_score as AP
from sklearn.neighbors import BallTree

from checkpoint import cached_refit, run_study
from common import ROOT, load_config, p
from features import FoldPreprocessor
from graph_models import GNN, AsOfGAT, geo_knn_graph, time_causal
from leakage_diagnostic import d1, d2
from nn_train import predict, train_full_batch

tab = importlib.import_module("03_tabular")
pb = importlib.import_module("p01_build_pests")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS = ["G_lgbm", "G_lgbm_nbr", "G_lgbm_nbr_leaky", "EcoPE-GAT_leaky", "EcoPE-GAT_causal", "EcoPE-GAT_asof"]
# D16 candidate (exploratory, development only; not part of the frozen D15 model set, never run by default):
# EcoPE-GAT_asof_plus = EcoPE-GAT_asof whose node and as-of edge features also carry the radius block of
# G_lgbm_nbr (target records within 100 km and in the 50-100 km ring, year < t_i), so the graph model sees
# the same information as the tabular reference.
EXTRA = ["EcoPE-GAT_asof_plus"]
PLUS = ["prior100_log_n", "prior_ring50_100_log_n"]
NODE = ["prior_log_n", "prior_any", "prior_years", "effort_log"]
CTX = ["bio1", "bio5", "bio6", "bio12", "lat", "lon"]
N_TRIALS, SEEDS, HORIZONS, TUNE_N = 20, [0, 1, 2], [3, 5], 3
EFF_Y0 = 2005


def target_years(d: pd.DataFrame, tgt: str, r_km: float, R: float) -> list[np.ndarray]:
    T = pd.read_parquet(pb.RAW / f"{tgt}.parquet").dropna(subset=["decimalLatitude", "decimalLongitude", "year"])
    ty = T["year"].to_numpy().astype(int)
    tree = BallTree(np.radians(T[["decimalLatitude", "decimalLongitude"]].to_numpy()), metric="haversine")
    return [np.sort(ty[ix]) for ix in tree.query_radius(np.radians(d[["lat", "lon"]].to_numpy()), r=r_km / R)]


def effort_table(bg: str) -> pd.DataFrame:
    eff = pd.read_parquet(pb.RAW / f"{bg}_effort.parquet")
    piv = eff.pivot_table(index="gid2", columns="year", values="n", aggfunc="sum", fill_value=0)
    return piv.reindex(columns=range(EFF_Y0, 2026), fill_value=0).cumsum(axis=1)


def feats_asof(j: np.ndarray, ti: np.ndarray, yrs50, uyrs50, cum, gid2) -> pd.DataFrame:
    """Time-stamped features of node j as of year ti (records with year < ti)."""
    n = np.array([np.searchsorted(yrs50[a], b, side="left") for a, b in zip(j, ti)])
    ny = np.array([np.searchsorted(uyrs50[a], b, side="left") for a, b in zip(j, ti)])
    g = gid2[j]
    known = pd.Series(g).isin(cum.index).to_numpy()
    e = np.full(len(j), np.nan)
    kk = np.flatnonzero(known & (ti > EFF_Y0))
    e[kk] = cum.loc[g[kk]].to_numpy()[np.arange(len(kk)), np.minimum(ti[kk], 2026) - 1 - EFF_Y0]
    e[known & (ti <= EFF_Y0)] = 0.0
    return pd.DataFrame({"prior_log_n": np.log1p(n), "prior_any": (n > 0).astype(float),
                         "prior_years": ny.astype(float), "effort_log": np.log1p(e)})


def plus_asof(j, ti, yrs50, yrs100) -> pd.DataFrame:
    n100 = np.array([np.searchsorted(yrs100[a], b, side="left") for a, b in zip(j, ti)])
    n50 = np.array([np.searchsorted(yrs50[a], b, side="left") for a, b in zip(j, ti)])
    return pd.DataFrame({"prior100_log_n": np.log1p(n100), "prior_ring50_100_log_n": np.log1p(n100 - n50)})


def neighbour_block(d: pd.DataFrame, X: pd.DataFrame, ei, ew, causal: bool, yrs100, yrs50) -> pd.DataFrame:
    t = d["t"].to_numpy()
    if causal:
        ei, _ = time_causal(ei, ew, t)
    src, dst = ei.numpy()
    n = len(d)
    deg = np.bincount(dst, minlength=n).astype(float)

    def mean_(v):                                       # mean over neighbours with a finite value
        f = np.isfinite(v[src])
        c = np.bincount(dst, weights=f, minlength=n)
        s = np.bincount(dst, weights=np.where(f, v[src], 0.0), minlength=n)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    amax = np.zeros(n)
    np.maximum.at(amax, dst, X["prior_any"].to_numpy()[src])
    n100 = np.array([np.searchsorted(a, b, side="left") for a, b in zip(yrs100, t)])
    n50 = np.array([np.searchsorted(a, b, side="left") for a, b in zip(yrs50, t)])
    return pd.DataFrame({"nbr_prior_log_n_mean": mean_(X["prior_log_n"].to_numpy()),
                         "nbr_prior_any_max": np.where(deg > 0, amax, np.nan),
                         "nbr_effort_log_mean": mean_(X["effort_log"].to_numpy()), "nbr_n": deg,
                         "prior100_log_n": np.log1p(n100), "prior_ring50_100_log_n": np.log1p(n100 - n50)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(pb.CASES))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS + EXTRA)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="code check only (2 trials, 2 epochs, 1 seed), "
                    "writes to results/pests_smoke, never a result")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config()
    dc = cfg["d13"]
    n_trials, seeds, max_ep, fixed = N_TRIALS, SEEDS, dc["max_epochs"], dict(cfg["tabular"]["fixed"])
    out = p(cfg, "results_dir") / "pests" / a.case
    if a.smoke:
        n_trials, seeds, max_ep = 2, [0], 2
        fixed["lgbm_max_rounds"] = 5
        out = p(cfg, "results_dir") / "pests_smoke" / a.case
    out.mkdir(parents=True, exist_ok=True)
    tgt, bg = pb.CASES[a.case]
    d = pd.read_parquet(ROOT / "data" / "processed" / "pests" / f"{a.case}_dev.parquet")
    assert not d["holdout"].any()
    R, B, K = cfg["spatial"]["earth_radius_km"], cfg["spatial"]["buffer_km"], cfg["graph"]["k"]
    assert B == pb.B
    t = d["t"].to_numpy().astype(int)
    Y = {N: d[f"y{N}"].to_numpy() for N in HORIZONS}
    ok = {N: np.isfinite(Y[N]) for N in HORIZONS}

    # ---- features, with self-checks against the stored table
    yrs50 = target_years(d, tgt, B, R)
    yrs100 = target_years(d, tgt, pb.R100, R)
    uyrs50 = [np.unique(v) for v in yrs50]
    cum = effort_table(bg)
    gid2 = d["gid2"].to_numpy().astype(object)
    me = np.arange(len(d))
    X = feats_asof(me, t, yrs50, uyrs50, cum, gid2)
    assert (np.expm1(X["prior_log_n"]).round().astype(int).to_numpy() == d["n_prior"].to_numpy()).all()
    assert (X["prior_years"].to_numpy() == d["prior_years"].to_numpy()).all()
    assert np.allclose(np.expm1(X["effort_log"]), d["effort_prior"], equal_nan=True)
    assert np.allclose(np.log1p([np.searchsorted(a_, b) for a_, b in zip(yrs100, t)]), d["persist100"])
    for N in HORIZONS:                                   # labels reproduced from the record index
        yy = np.array([float(((v > ti) & (v <= ti + N)).any()) for v, ti in zip(yrs50, t)])
        assert (yy[ok[N]] == Y[N][ok[N]]).all()
    Cdf = d[CTX].reset_index(drop=True)
    ei, ew = geo_knn_graph(d[["lat", "lon"]].to_numpy(), K, B, R)
    graphs = {"leaky": (ei, ew), "causal": time_causal(ei, ew, t)}
    nbr = {"G_lgbm_nbr": neighbour_block(d, X, ei, ew, True, yrs100, yrs50),
           "G_lgbm_nbr_leaky": neighbour_block(d, X, ei, ew, False, yrs100, yrs50)}
    src, dst = ei.numpy()
    xe_raw = feats_asof(src, t[dst], yrs50, uyrs50, cum, gid2)
    Xp = pd.concat([X, plus_asof(me, t, yrs50, yrs100)], axis=1)
    assert np.allclose(Xp["prior100_log_n"], d["persist100"])
    xe_plus = pd.concat([xe_raw, plus_asof(src, t[dst], yrs50, yrs100)], axis=1)
    ea = torch.as_tensor(np.stack([B * (1 / ew.numpy() - 1) / B, (t[dst] - t[src]) / 10.0], 1), dtype=torch.float32)
    # as-of leak check (all edges): counted records of j must all be < t_i
    n_used = np.round(np.expm1(xe_raw["prior_log_n"].to_numpy())).astype(int)
    bad = sum(int((yrs50[j][:k] >= t[i]).any()) for j, i, k in zip(src, dst, n_used))
    assert bad == 0
    diag = {"case": a.case, "n_dev": int(len(d)), **d1(ei.numpy(), t)[0],
            "D2_N3": d2(ei.numpy(), t, Y[3].astype(int), yrs50, 3)[0],
            "asof_edges_with_records_at_or_after_t_i": bad, "asof_edges_removed": 0,
            "min_in_degree": int(np.bincount(dst, minlength=len(d)).min())}
    rep = ROOT / "reports" / ("pests_smoke" if a.smoke else "pests")
    rep.mkdir(parents=True, exist_ok=True)
    json.dump(diag, open(rep / f"{a.case}_diagnostics.json", "w"), indent=1)
    print(json.dumps(diag), flush=True)

    # ---- contexts: all_dev (tuning) and cv0..cv4 (refits)
    fold = d["spatial_fold"].to_numpy()
    ctxs = {"all_dev": (np.flatnonzero(~d["alldev_inner_val"].to_numpy()),
                        np.flatnonzero(d["alldev_inner_val"].to_numpy()), np.array([], dtype=int))}
    for k in range(5):
        iv = d[f"spatial_inner_val_f{k}"].to_numpy()
        ctxs[f"cv{k}"] = (np.flatnonzero((fold != k) & ~iv), np.flatnonzero(iv), np.flatnonzero(fold == k))

    def build(m, tr):
        if m.startswith("G_lgbm"):
            Z = pd.concat([X, Cdf] + ([nbr[m]] if m in nbr else []), axis=1)
            return {"X": FoldPreprocessor().fit(Z, tr).transform(Z)}
        C = FoldPreprocessor().fit(Cdf, tr).transform(Cdf)
        if m == "EcoPE-GAT_asof_plus":
            pre = FoldPreprocessor().fit(Xp, tr)
            return {"x": torch.as_tensor(pre.transform(Xp)), "xe": torch.as_tensor(pre.transform(xe_plus[NODE + PLUS])),
                    "ctx": torch.as_tensor(C), "edge_index": ei, "ea": ea}
        pre = FoldPreprocessor().fit(X, tr)
        if m == "EcoPE-GAT_asof":
            return {"x": torch.as_tensor(pre.transform(X)), "xe": torch.as_tensor(pre.transform(xe_raw[NODE])),
                    "ctx": torch.as_tensor(C), "edge_index": ei, "ea": ea}
        e_, w_ = graphs["leaky" if m == "EcoPE-GAT_leaky" else "causal"]
        return {"x": torch.as_tensor(pre.transform(X)), "ctx": torch.as_tensor(C), "edge_index": e_,
                "edge_weight": w_}

    def run(m, data, prm, N, seed, tr, iv):
        tr, iv = tr[ok[N][tr]], iv[ok[N][iv]]
        y = np.nan_to_num(Y[N]).astype(np.int64)
        if m.startswith("G_lgbm"):
            return tab.fit_predict("lgbm", prm, data["X"], y, tr, iv, seed, a.threads, fixed)
        torch.manual_seed(seed)
        if m.startswith("EcoPE-GAT_asof"):
            net = AsOfGAT(data["x"].shape[1], data["ctx"].shape[1], ea.shape[1], prm["hidden"], prm["dropout"],
                          ecope_form=prm["encoder"])
        else:
            net = GNN("gat", data["x"].shape[1], prm["hidden"], prm["heads"], prm["dropout"],
                      d_ctx=data["ctx"].shape[1], ecope_form=prm["encoder"])
        net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=prm["lr"],
                                     weight_decay=prm["weight_decay"],
                                     pos_weight=float((y[tr] == 0).sum() / (y[tr] == 1).sum()),
                                     max_epochs=max_ep, patience=dc["patience"], seed=seed)
        return predict(net, data), info

    for m in a.models:
        t0 = time.time()
        tr, iv, _ = ctxs["all_dev"]
        data = build(m, tr)
        ivN = iv[ok[TUNE_N][iv]]

        def objective(trial):
            if m.startswith("G_lgbm"):
                prm = tab.suggest("lgbm", trial, cfg["tabular"]["search_space"])
            elif m.startswith("EcoPE-GAT_asof"):
                prm = tab.suggest("gnn", trial, {"gnn": cfg["d14"]["asof_space"]})
            else:
                prm = tab.suggest("gnn", trial, {"gnn": dc["gnn_space"]})
            s, info = run(m, data, prm, TUNE_N, 0, tr, iv)
            return float(AP(Y[TUNE_N][ivN].astype(int), s[ivN]))

        study = run_study(f"{m}_all_dev", out / "optuna", objective, n_trials, seed=0)
        best = study.best_params
        (out / "params").mkdir(parents=True, exist_ok=True)
        json.dump({"case": a.case, "model": m, "context": "all_dev", "best_params": best,
                   "best_value": study.best_value, "n_trials": len(study.trials)},
                  open(out / "params" / f"{m}_all_dev.json", "w"), indent=1)
        print(f"[{a.case} {m}] tuned: inner PR-AUC {study.best_value:.3f} {best} ({time.time() - t0:.0f}s)",
              flush=True)
        for k in range(5):
            c = f"cv{k}"
            tr, iv, te = ctxs[c]
            data = build(m, tr)
            for N in HORIZONS:
                teN = te[ok[N][te]]
                for sd in seeds:
                    def refit(N=N, sd=sd, data=data, tr=tr, iv=iv, teN=teN):
                        s, info = run(m, data, best, N, sd, tr, iv)
                        info = {k_: v for k_, v in info.items() if np.isscalar(v)}
                        return ({"test": s[teN].astype(np.float32), "test_idx": teN},
                                {"case": a.case, "model": m, "context": c, "N": N, "seed": sd,
                                 "test_pr_auc": float(AP(Y[N][teN].astype(int), s[teN])), **info})
                    cached_refit(out / "refits" / f"{m}_{c}_N{N}_s{sd}", refit)
        print(f"[{a.case} {m}] refits done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
