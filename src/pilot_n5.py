"""Pilot (decision point before tuning): default-hyperparameter models, N = 5, spatial 5-fold CV,
one seed. Answers: do traits add signal beyond prior presence + sampling effort (D vs E)?

All models are trained on the training part of each outer fold (inner-validation blocks excluded,
used only for GNN early stopping) and evaluated on the outer test fold.
Outputs: results/pilot/pilot_folds.csv, pilot_summary.csv, pilot_paired.csv, pilot_oof.parquet

Usage: python src/pilot_n5.py [--gnn]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

from common import ROOT, load_config, p, variant_name
from features import (CONTEXT_BLOCKS, FEATURE_SETS, FoldPreprocessor, assemble, labels,
                      load_modelling_table, raw_blocks, split_indices)

N, SEED, THREADS = 5, 0, 8


def block_bootstrap_diff(y, s1, s2, blocks, n_boot=2000, seed=0):
    """95% CI of AP(s2) - AP(s1) on pooled out-of-fold predictions, resampling spatial blocks."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx_by_b = {b: np.flatnonzero(blocks == b) for b in ub}
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([idx_by_b[b] for b in rng.choice(ub, len(ub))])
        if 0 < y[ix].sum() < len(ix):
            d.append(AP(y[ix], s2[ix]) - AP(y[ix], s1[ix]))
    return np.percentile(d, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", action="store_true", help="also run plain GAT and EcoPE-GAT")
    ap.add_argument("--tag", required=True, help="version name; results go to reports/pilot_runs/<tag>/")
    a = ap.parse_args()
    cfg = load_config()
    out = ROOT / "reports" / "pilot_runs" / a.tag     # versioned and tracked in git; never overwritten
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists: choose a new --tag (results are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    e = load_modelling_table(cfg)
    has_eff = "effort_fungi_prior" in e.columns
    y = labels(e, cfg, N)
    blk = raw_blocks(e, cfg)
    # Without the independent effort layer, C/D/E are run without it and labelled C0/D0/E0.
    fsets = {"A": FEATURE_SETS["A"]}
    for fs in ["C", "D", "E", "G"]:
        fsets[fs if has_eff else fs + "0"] = [b for b in FEATURE_SETS[fs] if has_eff or b != "effort"]
    sfx = "" if has_eff else "0"
    print("independent effort layer:", "yes" if has_eff else "NO (sets C0/D0/E0)", flush=True)
    folds = range(cfg["evaluation"]["n_folds"])
    rows, oof = [], {}

    def record(model, fs, k, te, s, t0):
        rows.append({"model": model, "fset": fs, "fold": k, "n_test": len(te), "n_pos": int(y[te].sum()),
                     "chance": float(y[te].mean()), "pr_auc": AP(y[te], s), "roc_auc": AUC(y[te], s),
                     "seconds": time.time() - t0})
        oof.setdefault(f"{model}_{fs}", np.full(len(y), np.nan))[te] = s

    for k in folds:
        tr, iv, te = split_indices(e, "spatial", k)
        # non-learned scores
        t0 = time.time()
        V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
        record("score", "prior_count", k, te, e[f"n_prior_b{cfg['spatial']['buffer_km']}_{V}"].to_numpy()[te], t0)
        if has_eff:
            record("score", "fungi_effort_prior", k, te, e["effort_fungi_prior"].to_numpy()[te], t0)
            record("score", "fungi_effort_all(diag)", k, te, e["effort_fungi_all"].to_numpy()[te], t0)
        for fs, blocks_ in fsets.items():
            Xdf = assemble(blk, blocks_)
            X = FoldPreprocessor().fit(Xdf, tr).transform(Xdf)
            assert not set(tr) & set(te)
            import lightgbm as lgb
            t0 = time.time()
            m = lgb.LGBMClassifier(random_state=SEED, n_jobs=THREADS, verbose=-1).fit(X[tr], y[tr])
            record("lgbm", fs, k, te, m.predict_proba(X[te])[:, 1], t0)
            t0 = time.time()
            m = RandomForestClassifier(random_state=SEED, n_jobs=THREADS).fit(X[tr], y[tr])
            record("rf", fs, k, te, m.predict_proba(X[te])[:, 1], t0)
        print(f"fold {k} done", flush=True)

    if a.gnn:
        import importlib
        from graph_models import GNN
        from nn_train import predict, train_full_batch
        g = importlib.import_module("04_graph")
        torch.set_num_threads(THREADS)
        for k in folds:
            yk, tr, iv, te, XA, CTX, Z, ei, ew = g.prepare(e, cfg, "spatial", k, N, cfg["graph"]["k"], "transductive")
            data = {"x": torch.as_tensor(XA), "ctx": torch.as_tensor(CTX), "edge_index": ei, "edge_weight": ew}
            pos_w = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
            for name, ctx in [("gat", 0), ("ecope_gat", CTX.shape[1])]:
                t0 = time.time()
                torch.manual_seed(SEED)
                net = GNN("gat", XA.shape[1], 64, 4, 0.3, d_ctx=ctx)     # defaults of the original paper
                net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=1e-3, weight_decay=1e-4,
                                             pos_weight=pos_w, max_epochs=200, patience=20, seed=SEED)
                record(name, "A", k, te, predict(net, data)[te], t0)
                print(f"  fold {k} {name}: PR-AUC {rows[-1]['pr_auc']:.3f} epochs {info['epochs']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "pilot_folds.csv", index=False)
    summ = df.groupby(["model", "fset"]).agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                                             roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
                                             chance_mean=("chance", "mean")).reset_index()
    summ.to_csv(out / "pilot_summary.csv", index=False)

    # paired comparisons
    pair_rows = []
    blocks = e["block_id"].to_numpy()
    for model in ["lgbm", "rf"]:
        for f1, f2 in [("D" + sfx, "E" + sfx), ("A", "E" + sfx), ("D" + sfx, "C" + sfx),
                       ("G" + sfx, "C" + sfx)]:
            a1 = df[(df.model == model) & (df.fset == f1)].sort_values("fold")
            a2 = df[(df.model == model) & (df.fset == f2)].sort_values("fold")
            for met in ["pr_auc", "roc_auc"]:
                d = a2[met].to_numpy() - a1[met].to_numpy()
                lo, hi = (block_bootstrap_diff(y, oof[f"{model}_{f1}"], oof[f"{model}_{f2}"], blocks)
                          if met == "pr_auc" else (np.nan, np.nan))
                pair_rows.append({"model": model, "comparison": f"{f2} - {f1}", "metric": met,
                                  "mean_diff": d.mean(), "folds_improved": int((d > 0).sum()),
                                  "per_fold_diff": " ".join(f"{x:+.3f}" for x in d),
                                  "wilcoxon_p": wilcoxon(d).pvalue,
                                  "pooled_diff_block_boot_CI95": f"[{lo:+.3f}, {hi:+.3f}]" if met == "pr_auc" else ""})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "pilot_paired.csv", index=False)
    pd.DataFrame(oof).assign(y=y, block=blocks).to_parquet(out / "pilot_oof.parquet")
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string())
    print(pairs.round(4).to_string())


if __name__ == "__main__":
    main()
