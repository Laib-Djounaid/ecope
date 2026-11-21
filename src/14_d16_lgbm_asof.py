"""D16: G_lgbm_nbr_asof on the pathogen data (development only; hold-out never loaded).

G_lgbm_nbr with its kNN neighbour block replaced by the AS-OF encoding (D14): for each plot i, the
neighbours' time-stamped features recomputed as of t_i (src/asof_features.asof_edges), averaged over ALL
geographic neighbours (no edge removed, no window information), plus the max of prior_any, the number of
neighbours, and the radius block of G_lgbm_nbr (records within 100 km and in the 50-100 km ring, year < t_i).
Protocol identical to G_lgbm_nbr in src/09_d13.py (contexts cv0..cv4, dev_transfer, all_dev; Optuna TPE
seed 0, d13.n_trials trials at N = 5; refits N = 5/7/10 x seeds 0-4). Outputs: results/d16/{optuna,params,refits}.

Usage: python src/14_d16_lgbm_asof.py [--contexts ...] [--threads 8]
"""
from __future__ import annotations

import argparse
import importlib
import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score as AP

from asof_features import COLS, asof_edges
from checkpoint import cached_refit, run_study
from common import load_config, p
from features import FoldPreprocessor, assemble, labels, load_modelling_table, raw_blocks
from graph_models import geo_knn_graph
from neighbour_features import neighbour_block

d13 = importlib.import_module("09_d13")
tab = importlib.import_module("03_tabular")
optuna.logging.set_verbosity(optuna.logging.WARNING)
MODEL = "G_lgbm_nbr_asof"


def asof_block(e, cfg, blk) -> pd.DataFrame:
    ei, ew = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), cfg["graph"]["k"],
                           cfg["spatial"]["buffer_km"], cfg["spatial"]["earth_radius_km"])
    xe, _ = asof_edges(e, cfg, ei, ew)
    dst = ei[1].numpy()
    n = len(e)
    deg = np.bincount(dst, minlength=n).astype(float)
    out = {f"nbr_asof_{c}_mean": np.bincount(dst, weights=xe[c].to_numpy(), minlength=n) / np.maximum(deg, 1)
           for c in COLS}
    mx = np.zeros(n)
    np.maximum.at(mx, dst, xe["prior_any"].to_numpy())
    out["nbr_asof_prior_any_max"] = mx
    out["nbr_n"] = deg
    rad = neighbour_block(e, blk, cfg, causal=True)[["prior100_log_n", "prior_ring50_100_log_n"]]
    return pd.concat([pd.DataFrame(out), rad.reset_index(drop=True)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", nargs="+", default=d13.CONTEXTS, choices=d13.CONTEXTS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--holdout", action="store_true", help="single confirmatory run (src/holdout.py)")
    a = ap.parse_args()
    cfg = load_config()
    dc = cfg["d13"]
    out = p(cfg, "results_dir") / "d16"
    if a.holdout:
        import holdout as H
        dev_out, out = out, H.out_dir(cfg, "pathogens")
        e, ctxs = H.pathogen_setup(cfg)
        globals()["cached_refit"] = H.guard(cached_refit)
        a.contexts = list(ctxs)
    else:
        e = load_modelling_table(cfg)
        assert "holdout" in e.columns and not e["holdout"].any()
    Y = {N: labels(e, cfg, N) for N in dc["horizons"]}
    blk = raw_blocks(e, cfg)
    blk["nbr_asof"] = asof_block(e, cfg, blk)
    if not a.holdout:
        ctxs = d13.contexts(e, cfg)
    fixed = cfg["tabular"]["fixed"]
    for c in a.contexts:
        tr, iv, te = ctxs[c]
        t0 = time.time()
        Z = assemble(blk, d13.G + ["nbr_asof"])
        X = FoldPreprocessor().fit(Z, tr).transform(Z)

        def run(prm, N, seed):
            return tab.fit_predict("lgbm", prm, X, Y[N], tr, iv, seed, a.threads, fixed)

        def objective(trial):
            prm = tab.suggest("lgbm", trial, cfg["tabular"]["search_space"])
            s, _ = run(prm, dc["tuning_horizon"], 0)
            return AP(Y[dc["tuning_horizon"]][iv], s[iv])

        name = f"{MODEL}_{c}"
        if a.holdout:
            best = H.best_params(dev_out, MODEL, c)
        else:
            study = run_study(name, out / "optuna", objective, dc["n_trials"], seed=0)
            best = study.best_params
            (out / "params").mkdir(parents=True, exist_ok=True)
            json.dump({"model": MODEL, "context": c, "best_params": best, "best_value": study.best_value,
                       "n_trials": len(study.trials)}, open(out / "params" / f"{name}.json", "w"), indent=1)
            print(f"[{name}] tuned: inner PR-AUC {study.best_value:.3f} ({time.time() - t0:.0f}s)", flush=True)
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
