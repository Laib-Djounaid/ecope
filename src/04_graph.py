"""Step 3b: graph models (GCN, GraphSAGE, GAT, GAT+context concat, EcoPE-GAT) and the hybrid
LightGBM + graph-neighbourhood-label model, in transductive and inductive settings.

Protocol per (scheme, setting, model, outer fold) - identical to 03_tabular.py:
  preprocessing + kNN-space scaler fitted on training rows; Optuna (n_trials) maximising
  inner-validation PR-AUC on the tuning horizon; refit the best parameters for every
  horizon x seed; evaluate on the outer test fold.

Usage: python src/04_graph.py --scheme spatial [--settings transductive inductive]
       [--models ...] [--k 10] [--trials 30] [--n-jobs 3 --threads 3]
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors

from common import load_config, p
from features import (CONTEXT_BLOCKS, FEATURE_SETS, FoldPreprocessor, assemble, folds_of,
                      labels, load_modelling_table, raw_blocks, split_indices)
from graph_models import GNN, knn_graph
from checkpoint import cached_refit, run_study
from metrics import evaluate, safe_ap
from nn_train import predict, set_seed, train_full_batch

tab = importlib.import_module("03_tabular")
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

GNN_MODELS = {             # name -> (conv, node feature set, use EcoPE)
    "gcn": ("gcn", "A", False),
    "sage": ("sage", "A", False),
    "gat": ("gat", "A", False),
    "gat_concat": ("gat", "B", False),
    "ecope_gat": ("gat", "A", True),
}


# --------------------------------------------------------------------------- hybrid features
def neighbour_label_features(Z: np.ndarray, y: np.ndarray, tr: np.ndarray, k: int,
                             groups: np.ndarray | None, seed: int) -> np.ndarray:
    """[weighted fraction of positive neighbours, # positive neighbours, mean neighbour distance]
    using TRAINING labels only. Training rows are cross-fitted: a training row's neighbours are
    drawn from training rows in other groups (spatial blocks, or random folds for the random
    scheme), so neither its own label nor its block's labels enter its feature. Non-training
    rows (inner-val, test) use all training rows."""
    n = len(Z)
    F = np.zeros((n, 3), dtype=np.float32)

    def feats(query, ref):
        nn_ = NearestNeighbors(n_neighbors=k).fit(Z[ref])
        d, j = nn_.kneighbors(Z[query])
        yl = y[ref][j]
        w = 1.0 / (1.0 + d)
        return np.c_[(w * yl).sum(1) / w.sum(1), yl.sum(1), d.mean(1)]

    cv = GroupKFold(5) if groups is not None else KFold(5, shuffle=True, random_state=seed)
    for a, b in cv.split(tr, groups=None if groups is None else groups[tr]):
        F[tr[b]] = feats(tr[b], tr[a])
    other = np.setdiff1d(np.arange(n), tr)
    F[other] = feats(other, tr)
    return F


# --------------------------------------------------------------------------- one job
def prepare(e, cfg, scheme, fold, N, k, setting):
    """Fold-wise features (train-fitted preprocessing) and the kNN trait graph."""
    y = labels(e, cfg, N)
    tr, iv, te = split_indices(e, scheme, fold)
    blocks = raw_blocks(e, cfg)
    pa = FoldPreprocessor().fit(assemble(blocks, FEATURE_SETS["A"]), tr)
    pc = FoldPreprocessor().fit(assemble(blocks, CONTEXT_BLOCKS), tr)
    XA = pa.transform(assemble(blocks, FEATURE_SETS["A"]))
    CTX = pc.transform(assemble(blocks, CONTEXT_BLOCKS))
    n_tr_cols = len(blocks["traits"].columns)
    Ztrait = XA[:, :n_tr_cols]                     # standardised trait CWM/CWV (train stats)
    ei, ew = knn_graph(Ztrait, k, setting, tr)
    return y, tr, iv, te, XA, CTX, Ztrait, ei, ew


def run_job(scheme, setting, model, fold, cfg, n_trials, threads, k, horizons, ckpt_dir):
    """Tune once (inner-val PR-AUC, tuning horizon), then refit every horizon x seed."""
    torch.set_num_threads(threads)
    t0 = time.time()
    gc = cfg["graph"]
    e = load_modelling_table(cfg)
    Nt = cfg["evaluation"]["tuning_horizon"]
    yt, tr, iv, te, XA, CTX, Ztrait, ei, ew = prepare(e, cfg, scheme, fold, Nt, k, setting)
    assert not set(tr) & set(te) and not set(iv) & set(te) and not set(tr) & set(iv)
    Y = {N: labels(e, cfg, N) for N in horizons}
    rows, preds = [], {"test_idx": te}

    if model == "hybrid_lgbm":
        blk = raw_blocks(e, cfg)
        Xb = FoldPreprocessor().fit(assemble(blk, FEATURE_SETS["B"]), tr) \
            .transform(assemble(blk, FEATURE_SETS["B"]))
        groups = e["block_id"].to_numpy() if scheme != "random" else None
        Xcache: dict[int, np.ndarray] = {}

        def fp(prm, seed, y):
            key = hash(y.tobytes())
            if key not in Xcache:   # neighbour features: TRAINING labels only
                F = neighbour_label_features(Ztrait, y, tr, k, groups, seed=0)
                Xcache[key] = np.c_[Xb, F].astype(np.float32)
            return tab.fit_predict("lgbm", prm, Xcache[key], y, tr, iv, seed, threads,
                                   cfg["tabular"]["fixed"])

        def suggest_fn(trial):
            return tab.suggest("lgbm", trial, cfg["tabular"]["search_space"])
    else:
        conv, fs, use_ecope = GNN_MODELS[model]
        x = XA if fs == "A" else np.c_[XA, CTX].astype(np.float32)
        data = {"x": torch.as_tensor(x), "ctx": torch.as_tensor(CTX),
                "edge_index": ei, "edge_weight": ew}

        def fp(prm, seed, y):
            set_seed(seed)
            pos_w = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
            heads = prm.get("heads", 1) if conv == "gat" else 1
            net = GNN(conv, x.shape[1], prm["hidden"], heads, prm["dropout"],
                      d_ctx=CTX.shape[1] if use_ecope else 0)
            net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=prm["lr"],
                                         weight_decay=prm["weight_decay"],
                                         pos_weight=pos_w if prm["balanced"] else 1.0,
                                         max_epochs=gc["max_epochs"], patience=gc["patience"],
                                         seed=seed)
            return predict(net, data), info

        def suggest_fn(trial):
            space = dict(gc["search_space"]["gnn"])
            if conv != "gat":
                space.pop("heads")
            return tab.suggest("gnn", trial, {"gnn": space})

    # checkpoints: Optuna study saved after every trial, each refit saved when it finishes
    ck = ckpt_dir / f"{setting}_{model}_f{str(fold).replace(' ', '_')}"
    study = run_study("optuna", ck, lambda tl: safe_ap(yt[iv], fp(suggest_fn(tl), 0, yt)[0][iv]),
                      n_trials, cfg["evaluation"]["split_seed"])
    best = study.best_params
    t_tune = time.time() - t0
    for N in horizons:
        y = Y[N]
        for seed in cfg["seeds"]:
            def refit(seed=seed, y=y):
                s, info = fp(best, seed, y)
                return {"te": s[te], "iv": s[iv]}, info
            arrs, info = cached_refit(ck / f"N{N}_seed{seed}", refit)
            rows.append({**evaluate(y[te], arrs["te"]), "seed": seed, "horizon": N,
                         "inner_val_pr_auc": safe_ap(y[iv], arrs["iv"]), **info})
            preds[f"N{N}_seed{seed}"] = arrs["te"]
        preds[f"y_test_N{N}"] = y[te]
    for r in rows:
        r.update({"scheme": scheme, "setting": setting, "model": model, "k": k, "fold": str(fold),
                  "n_edges": int(ei.shape[1]), "tune_seconds": t_tune, "seconds": time.time() - t0})
    return rows, {"scheme": scheme, "setting": setting, "model": model, "k": k, "fold": str(fold),
                  "tuning_horizon": Nt, "n_trials": n_trials, "best_params": best}, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="spatial", choices=["spatial", "random", "loco"])
    ap.add_argument("--settings", nargs="+", default=["transductive", "inductive"])
    ap.add_argument("--models", nargs="+", default=["hybrid_lgbm"] + list(GNN_MODELS))
    ap.add_argument("--k", type=int)
    ap.add_argument("--trials", type=int)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--folds", nargs="+")
    ap.add_argument("--tag", default="", help="suffix for the output directory (ablations)")
    ap.add_argument("--out", help="output directory override (debugging only)")
    a = ap.parse_args()
    cfg = load_config()
    H = cfg["labels"]["horizons"]
    k = a.k or cfg["graph"]["k"]
    n_trials = a.trials or cfg["graph"]["n_trials"]
    out = Path(a.out) if a.out else p(cfg, "results_dir") / "graph" / f"{a.scheme}_k{k}{a.tag}"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "metrics_parts").mkdir(exist_ok=True)
    e = load_modelling_table(cfg)
    folds = [f for f in folds_of(e, a.scheme, cfg) if not a.folds or str(f) in a.folds]
    jobs = []
    for m in a.models:
        for st in a.settings:
            if m == "hybrid_lgbm" and st != "transductive":
                continue      # the hybrid uses training labels only; no message passing
            for f in folds:
                tag = f"{st}_{m}_f{str(f).replace(' ', '_')}"
                if not (out / "preds" / f"{tag}.npz").exists():
                    jobs.append((st, m, f, tag))
    jobs.sort(key=lambda j: 0 if j[1] in ("gat", "gat_concat", "ecope_gat") else 1)
    print(f"{len(jobs)} jobs -> {out}, {n_trials} trials", flush=True)
    T0 = time.time()

    def wrapped(st, m, f, tag):
        rows, bp, preds = run_job(a.scheme, st, m, f, cfg, n_trials, a.threads, k, H, out / "checkpoints")
        np.savez_compressed(out / "preds" / f"{tag}.npz", **preds)
        pd.DataFrame(rows).to_csv(out / "metrics_parts" / f"{tag}.csv", index=False)
        json.dump(bp, open(out / "metrics_parts" / f"{tag}.json", "w"))
        r = pd.DataFrame(rows).groupby("horizon").pr_auc.mean().round(3).to_dict()
        print(f"  {tag}: PR-AUC by horizon {r} ({time.time() - T0:.0f}s since start)", flush=True)

    Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(wrapped)(*j) for j in jobs)
    tab.collect(out)


if __name__ == "__main__":
    main()
