"""Pre-registered pilot of EcoPE-GAT v2 (decision D9, reports/decision_log.md).

EcoPE-GAT v2: EcoPE inputs = SoilClim PC1/PC2, lat/lon, biome one-hot, independent effort
(non-target Fungi before the survey, log1p); node features = prior presence (P) or prior presence
+ traits (PT); graphs = trait kNN (v1 reference), geo kNN (haversine), combined (union; edge
attributes [w_trait, w_geo]). Controls on the same folds: GAT with the same variables
concatenated (no encoder) and LightGBM on feature set G.

N = 5, spatial 5-fold CV (same splits as pilot v2), seed 0, default hyperparameters (those of the
original paper, as in pilot v1). Transductive graphs. Every (fold, configuration) result is saved
as soon as it finishes (resumable). Output: reports/pilot_runs/<tag>/.

Usage: python src/pilot_ecope_v2.py --tag v3_ecope_v2 [--configs ...]
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

from common import ROOT, load_config
from features import (FEATURE_SETS, FoldPreprocessor, assemble, labels, load_modelling_table,
                      raw_blocks, split_indices)
from graph_models import GNN, geo_knn_graph, knn_graph, two_channel
from nn_train import predict, train_full_batch

pilot = importlib.import_module("pilot_n5")
N, SEED = 5, 0
CTX_V2 = ["env", "geo", "biome", "effort"]                       # EcoPE v2 inputs
NODES = {"P": ["prior"], "PT": ["prior", "traits", "attrs"]}
GRAPHS = ["trait", "geo", "combined"]
DEFAULTS = dict(hidden=64, heads=4, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                max_epochs=200, patience=20)                    # original-paper defaults


def config_names() -> list[str]:
    names = ["G_lgbm"]
    for nd in NODES:
        for gr in GRAPHS:
            names += [f"ecope_v2_{nd}_{gr}", f"gat_concat_{nd}_{gr}"]
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--configs", nargs="+", help="subset of configurations (default: all)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"], help="smoke tests only")
    ap.add_argument("--folds", type=int, nargs="+", help="smoke tests only")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config()
    out = ROOT / "reports" / "pilot_runs" / a.tag
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    e = load_modelling_table(cfg)
    assert "effort_fungi_prior" in e.columns, "effort layer missing"
    y = labels(e, cfg, N)
    blk = raw_blocks(e, cfg)
    R, k = cfg["spatial"]["earth_radius_km"], cfg["graph"]["k"]
    configs = a.configs or config_names()

    for fold in (a.folds if a.folds is not None else range(cfg["evaluation"]["n_folds"])):
        tr, iv, te = split_indices(e, "spatial", fold)
        assert not set(tr) & set(te) and not set(iv) & set(te)
        todo = [c for c in configs if not (parts / f"{c}_f{fold}.json").exists()]
        if not todo:
            continue
        t0 = time.time()
        # fold-wise preprocessing, fitted on training rows only
        Xn = {nd: FoldPreprocessor().fit(assemble(blk, b), tr).transform(assemble(blk, b))
              for nd, b in NODES.items()}
        C = FoldPreprocessor().fit(assemble(blk, CTX_V2), tr).transform(assemble(blk, CTX_V2))
        XA = FoldPreprocessor().fit(assemble(blk, FEATURE_SETS["A"]), tr).transform(assemble(blk, FEATURE_SETS["A"]))
        Ztrait = XA[:, :len(blk["traits"].columns)]
        ei_t, w_t = knn_graph(Ztrait, k, "transductive", tr)
        ei_g, w_g = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), k, cfg["spatial"]["buffer_km"], R)
        graphs = {g: two_channel(ei_t, w_t, ei_g, w_g, len(e), g) for g in GRAPHS}
        print(f"fold {fold}: edges trait {graphs['trait'][0].shape[1]:,}, geo {graphs['geo'][0].shape[1]:,}, "
              f"combined {graphs['combined'][0].shape[1]:,} ({time.time() - t0:.0f}s)", flush=True)
        pos_w = float((y[tr] == 0).sum() / (y[tr] == 1).sum())

        for c in todo:
            t1 = time.time()
            info = {}
            if c == "G_lgbm":       # same code path as pilot v2 (default LightGBM, train rows)
                X = FoldPreprocessor().fit(assemble(blk, FEATURE_SETS["G"]), tr).transform(assemble(blk, FEATURE_SETS["G"]))
                s = lgb.LGBMClassifier(random_state=SEED, n_jobs=a.threads, verbose=-1).fit(X[tr], y[tr]).predict_proba(X)[:, 1]
            else:
                kind, nd, gr = (("ecope", *c.split("_")[2:]) if c.startswith("ecope_v2")
                                else ("concat", *c.split("_")[2:]))
                ei, ea = graphs[gr]
                x = Xn[nd] if kind == "ecope" else np.c_[Xn[nd], C].astype(np.float32)
                data = {"x": torch.as_tensor(x), "ctx": torch.as_tensor(C), "edge_index": ei, "edge_weight": ea}
                torch.manual_seed(SEED)
                net = GNN("gat", x.shape[1], DEFAULTS["hidden"], DEFAULTS["heads"], DEFAULTS["dropout"],
                          d_ctx=C.shape[1] if kind == "ecope" else 0, edge_dim=2)
                net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=DEFAULTS["lr"],
                                             weight_decay=DEFAULTS["weight_decay"], pos_weight=pos_w,
                                             max_epochs=a.max_epochs, patience=DEFAULTS["patience"], seed=SEED)
                s = predict(net, data)
            np.save(parts / f"{c}_f{fold}.npy", s[te].astype(np.float32))
            rec = {"config": c, "fold": fold, "n_test": int(len(te)), "n_pos": int(y[te].sum()),
                   "chance": float(y[te].mean()), "pr_auc": AP(y[te], s[te]), "roc_auc": AUC(y[te], s[te]),
                   "seconds": time.time() - t1, **{k_: v for k_, v in info.items()}}
            json.dump(rec, open(parts / f"{c}_f{fold}.json", "w"))          # written last = done
            print(f"  fold {fold} {c}: PR-AUC {rec['pr_auc']:.3f} ({rec['seconds']:.0f}s)", flush=True)

    # ---------------- collect + pre-registered tests (only once every part exists)
    recs = [json.load(open(f)) for f in sorted(parts.glob("*.json"))]
    df = pd.DataFrame(recs)
    df.to_csv(out / "pilot_folds.csv", index=False)
    complete = [c for c in config_names() if (df.config == c).sum() == cfg["evaluation"]["n_folds"]]
    oof = {}
    for c in complete:
        o = np.full(len(y), np.nan)
        for fold in range(cfg["evaluation"]["n_folds"]):
            _, _, te = split_indices(e, "spatial", fold)
            o[te] = np.load(parts / f"{c}_f{fold}.npy")
        oof[c] = o
    summ = df.groupby("config").agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                                    roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                                    epochs=("epochs", "mean") if "epochs" in df else ("pr_auc", "size")).reset_index()
    summ["pooled_pr_auc"] = summ.config.map(lambda c: AP(y, oof[c]) if c in oof else np.nan)
    summ.to_csv(out / "pilot_summary.csv", index=False)

    blocks = e["block_id"].to_numpy()
    tests = []
    for c in complete:
        if c == "G_lgbm":
            continue
        refs = [("G_lgbm", "competitive_vs_G")]
        if c.startswith("ecope_v2"):
            refs.append((c.replace("ecope_v2", "gat_concat"), "ecope_adds_vs_concat"))
        for ref, test in refs:
            if ref not in oof:
                continue
            lo, hi = pilot.block_bootstrap_diff(y, oof[ref], oof[c], blocks, n_boot=2000, seed=0)
            d = AP(y, oof[c]) - AP(y, oof[ref])
            verdict = ("superior" if lo > 0 else "non-inferior (competitive)" if lo > -0.02 else "inferior")
            if test == "ecope_adds_vs_concat":
                verdict = "EcoPE adds (lower bound > 0)" if lo > 0 else (
                          "non-inferior to concat" if lo > -0.02 else "concat better or equal")
            tests.append({"config": c, "reference": ref, "test": test, "pooled_diff": d,
                          "ci95_low": lo, "ci95_high": hi, "verdict": verdict})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "preregistered_tests.csv", index=False)
    pd.DataFrame(oof).assign(y=y, block=blocks).to_parquet(out / "pilot_oof.parquet")
    pd.set_option("display.width", 230)
    print(summ.round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
