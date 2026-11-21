"""D14 tuning and development refits of EcoPE-GAT_asof (reports/preregistration_D14.md, tag prereg-D14).

Same protocol as src/09_d13.py (contexts cv0..cv4, dev_transfer, all_dev; Optuna TPE seed 0,
d13.n_trials trials on inner-validation PR-AUC at N = 5; refits N = 5/7/10 x seeds 0-4, all_dev
tuning only), so EcoPE-GAT_asof is comparable with the D13 models. Development data only.
Outputs: results/d14/{optuna,params,refits}, results/d14/asof_leak_check.json.

Usage: python src/11_d14.py [--contexts cv0 ...] [--threads 8] [--smoke]
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np
import pandas as pd
import optuna
import torch
from sklearn.metrics import average_precision_score as AP

from asof_features import COLS, PLUS, asof_edges, asof_radius, leak_check
from neighbour_features import neighbour_block
from checkpoint import cached_refit, run_study
from common import load_config, p
from features import FoldPreprocessor, assemble, labels, load_modelling_table, raw_blocks
from graph_models import AsOfGAT, geo_knn_graph
from nn_train import predict, train_full_batch

d13 = importlib.import_module("09_d13")
tab = importlib.import_module("03_tabular")
optuna.logging.set_verbosity(optuna.logging.WARNING)
MODEL = "EcoPE-GAT_asof"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", nargs="+", default=d13.CONTEXTS, choices=d13.CONTEXTS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--plus", action="store_true", help="D16 candidate EcoPE-GAT_asof_plus (exploratory, "
                    "development only): adds the radius block of G_lgbm_nbr, as of t_i, to node and edge "
                    "features; outputs to <results_dir>/d16")
    ap.add_argument("--holdout", action="store_true", help="single confirmatory run (src/holdout.py): no tuning, "
                    "dev hyperparameters, settings T and S, outputs to <results_dir>/holdout/pathogens")
    ap.add_argument("--smoke", action="store_true", help="code check only (2 trials, 2 epochs, 1 seed, N=5), "
                    "writes to <results_dir>/d14_smoke, deleted afterwards, never a result")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config()
    dc = cfg["d13"]
    global MODEL
    out = p(cfg, "results_dir") / ("d16" if a.plus else "d14")
    if a.plus:
        MODEL = "EcoPE-GAT_asof_plus"
    if a.smoke:
        dc = {**dc, "n_trials": 2, "max_epochs": 2, "seeds": [0], "horizons": [5]}
        out = p(cfg, "results_dir") / "d14_smoke"
    out.mkdir(parents=True, exist_ok=True)
    if a.holdout:
        import holdout as H
        dev_out, out = out, H.out_dir(cfg, "pathogens")
        e, ctxs = H.pathogen_setup(cfg)
        globals()["cached_refit"] = H.guard(cached_refit)
        a.contexts = list(ctxs)
    else:
        e = load_modelling_table(cfg)                   # development only
        assert "holdout" in e.columns and not e["holdout"].any()
    Y = {N: labels(e, cfg, N) for N in dc["horizons"]}
    blk = raw_blocks(e, cfg)
    assert list(assemble(blk, d13.NODE).columns) == COLS
    ei, ew = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), cfg["graph"]["k"],
                           cfg["spatial"]["buffer_km"], cfg["spatial"]["earth_radius_km"])
    t0 = time.time()
    xe_raw, ea = asof_edges(e, cfg, ei, ew)
    if a.plus:
        rn, re_ = asof_radius(e, cfg, ei)
        ref = neighbour_block(e, blk, cfg, causal=True)
        assert np.allclose(rn[PLUS].to_numpy(), ref[PLUS].to_numpy()), "radius block not reproduced"
        blk = {**blk, "plus": rn}
        xe_raw = pd.concat([xe_raw.reset_index(drop=True), re_], axis=1)
    chk = {f"N{N}": leak_check(e, cfg, ei, xe_raw, N) for N in dc["horizons"]}
    chk.update({"n_edges": int(ei.shape[1]), "edges_removed": 0,
                "min_in_degree": int(np.bincount(ei[1].numpy(), minlength=len(e)).min())})
    json.dump(chk, open(out / (f"{MODEL}_asof_leak_check.json" if a.holdout else "asof_leak_check.json"), "w"), indent=1)
    assert all(v["edges_with_window_information"] == 0 for k, v in chk.items() if k.startswith("N"))
    print(f"as-of edges {ei.shape[1]:,} ({time.time() - t0:.0f}s); leak check {chk}", flush=True)
    if not a.holdout:
        ctxs = d13.contexts(e, cfg)

    for c in a.contexts:
        tr, iv, te = ctxs[c]
        name = f"{MODEL}_{c}"
        t0 = time.time()
        Xn = assemble(blk, d13.NODE + (["plus"] if a.plus else []))
        pre = FoldPreprocessor().fit(Xn, tr)            # one preprocessor for node and edge features
        x, xe = pre.transform(Xn), pre.transform(xe_raw[list(Xn.columns)])
        C = FoldPreprocessor().fit(assemble(blk, d13.CTX), tr).transform(assemble(blk, d13.CTX))
        data = {"x": torch.as_tensor(x), "xe": torch.as_tensor(xe), "ctx": torch.as_tensor(C),
                "edge_index": ei, "ea": ea}

        def run(prm, N, seed):
            y = Y[N]
            torch.manual_seed(seed)
            net = AsOfGAT(x.shape[1], C.shape[1], ea.shape[1], prm["hidden"], prm["dropout"],
                          ecope_form=prm["encoder"])
            pos_w = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
            net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=prm["lr"],
                                         weight_decay=prm["weight_decay"], pos_weight=pos_w,
                                         max_epochs=dc["max_epochs"], patience=dc["patience"], seed=seed)
            return predict(net, data), info

        def objective(trial):
            prm = tab.suggest("gnn", trial, {"gnn": cfg["d14"]["asof_space"]})
            _, info = run(prm, dc["tuning_horizon"], 0)
            trial.set_user_attr("epochs", info.get("epochs"))
            return info["best_val_pr_auc"]

        if a.holdout:
            best = H.best_params(dev_out, MODEL, c)
        else:
            study = run_study(name, out / "optuna", objective, dc["n_trials"], seed=0)
            best = study.best_params
            (out / "params").mkdir(parents=True, exist_ok=True)
            json.dump({"model": MODEL, "context": c, "best_params": best, "best_value": study.best_value,
                       "n_trials": len(study.trials)}, open(out / "params" / f"{name}.json", "w"), indent=1)
            print(f"[{name}] tuned: inner PR-AUC {study.best_value:.3f} {best} ({time.time() - t0:.0f}s)", flush=True)
        if c == "all_dev":
            continue
        for N in dc["horizons"]:
            for sd in dc["seeds"]:
                def refit(N=N, sd=sd):
                    s, info = run(best, N, sd)
                    info = {k_: v for k_, v in info.items() if np.isscalar(v)}
                    return ({"test": s[te].astype(np.float32), "val": s[iv].astype(np.float32), "test_idx": te},
                            {"model": MODEL, "context": c, "N": N, "seed": sd,
                             "test_pr_auc": float(AP(Y[N][te], s[te])), **info})
                cached_refit(out / "refits" / f"{name}_N{N}_s{sd}", refit)
        print(f"[{name}] refits done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
