"""D13 tuning and development refits (pre-registration `reports/preregistration_D13.md`, tag prereg-D13).

Models (H1-H5): G_lgbm, G_lgbm_nbr (causal neighbourhood), G_lgbm_nbr_leaky, EcoPE-GAT_leaky,
EcoPE-GAT_causal, GAT_noctx_causal.
Training sets ("contexts"), development data only (the sealed hold-out is never loaded):
  cv0..cv4      spatial folds (train = other folds minus the blocked inner validation)
  dev_transfer  train_high blocks (inner validation = transfer_inner_val), test = development test_low
  all_dev       whole development set, blocked inner validation (seed split_seed + 500); tuning only,
                its hyperparameters are used for the hold-out (S) setting.
Per (model, context): Optuna TPE (30 trials, seed 0) on inner-validation PR-AUC at N = 5; then, except
for all_dev, refits with the best hyperparameters for N = 5, 7, 10 x seeds 0-4 (hyperparameters tuned
at N = 5 are reused for N = 7 and 10). Checkpoints: Optuna SQLite after each trial, one file per
refit (resumable; ECOPE_RESULTS_DIR / ECOPE_LOCAL_TMP overrides for Colab + Drive).

Usage: python src/09_d13.py --models G_lgbm G_lgbm_nbr ... [--contexts cv0 ... all_dev] [--threads 8]
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

from checkpoint import cached_refit, run_study
from common import load_config, p
from features import FoldPreprocessor, assemble, labels, load_modelling_table, raw_blocks
from graph_models import GNN, geo_knn_graph, time_causal
from neighbour_features import neighbour_block
from nn_train import predict, train_full_batch

tab = importlib.import_module("03_tabular")
s1 = importlib.import_module("02_splits")
optuna.logging.set_verbosity(optuna.logging.WARNING)

LGBM_MODELS = {"G_lgbm": [], "G_lgbm_nbr": ["nbr_causal"], "G_lgbm_nbr_leaky": ["nbr_leaky"]}
GNN_MODELS = {"EcoPE-GAT_leaky": ("leaky", True), "EcoPE-GAT_causal": ("causal", True),
              "GAT_noctx_causal": ("causal", False)}
ALL_MODELS = list(LGBM_MODELS) + list(GNN_MODELS)
CONTEXTS = [f"cv{k}" for k in range(5)] + ["dev_transfer", "all_dev"]
G = ["prior", "effort", "env", "geo", "biome"]
NODE = ["prior", "effort"]                 # GAT node features: prior presence + dated effort
CTX = ["env", "geo", "biome"]              # ecological position (EcoPE input)


def contexts(e: pd.DataFrame, cfg: dict) -> dict:
    ev = cfg["evaluation"]
    f = e["spatial_fold"].to_numpy()
    out = {}
    for k in range(ev["n_folds"]):
        iv = e[f"spatial_inner_val_f{k}"].to_numpy()
        out[f"cv{k}"] = (np.flatnonzero((f != k) & ~iv), np.flatnonzero(iv), np.flatnonzero(f == k))
    role, iv = e["transfer_role"].to_numpy(), e["transfer_inner_val"].to_numpy()
    out["dev_transfer"] = (np.flatnonzero((role == "train_high") & ~iv), np.flatnonzero(iv),
                           np.flatnonzero(role == "test_low"))
    strat = labels(e, cfg, ev["stratify_horizon"])
    blocks = e["block_id"].to_numpy()
    m = np.zeros(len(e), dtype=bool)
    m[s1.inner_split(strat, blocks, ev["split_seed"] + cfg["d13"]["alldev_inner_seed_offset"], ev["n_inner"])] = True
    out["all_dev"] = (np.flatnonzero(~m), np.flatnonzero(m), np.array([], dtype=int))
    for tr, iv, te in out.values():
        assert not set(tr) & set(iv) and not set(tr) & set(te) and not set(iv) & set(te)
    return out


def suggest_gnn(trial: optuna.Trial, space: dict, with_encoder: bool) -> dict:
    sp = {k: v for k, v in space.items() if with_encoder or k != "encoder"}
    return tab.suggest("gnn", trial, {"gnn": sp})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--contexts", nargs="+", default=CONTEXTS, choices=CONTEXTS)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--holdout", action="store_true", help="single confirmatory run (src/holdout.py): no tuning, "
                    "dev hyperparameters, settings T and S, outputs to <results_dir>/holdout/pathogens")
    ap.add_argument("--smoke", action="store_true", help="code check only: 2 trials, 2 epochs, 1 seed, N=5, "
                    "writes to <results_dir>/d13_smoke (deleted afterwards, never a result)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config()
    dc = cfg["d13"]
    out = p(cfg, "results_dir") / "d13"
    if a.smoke:
        dc = {**dc, "n_trials": 2, "max_epochs": 2, "seeds": [0], "horizons": [5]}
        cfg["tabular"]["fixed"] = {**cfg["tabular"]["fixed"], "lgbm_max_rounds": 5}
        out = p(cfg, "results_dir") / "d13_smoke"
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
    if any(m in LGBM_MODELS for m in a.models):
        blk["nbr_causal"] = neighbour_block(e, blk, cfg, causal=True)
        blk["nbr_leaky"] = neighbour_block(e, blk, cfg, causal=False)
    ei0, ew0 = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), cfg["graph"]["k"],
                             cfg["spatial"]["buffer_km"], cfg["spatial"]["earth_radius_km"])
    graphs = {"leaky": (ei0, ew0), "causal": time_causal(ei0, ew0, e["survey_year"].to_numpy())}
    if not a.holdout:
        ctxs = contexts(e, cfg)
    fixed = cfg["tabular"]["fixed"]
    print(f"development plots {len(e):,}; edges leaky {ei0.shape[1]:,} causal {graphs['causal'][0].shape[1]:,}", flush=True)

    for c in a.contexts:
        tr, iv, te = ctxs[c]

        def fit(names):
            X = assemble(blk, names)
            return FoldPreprocessor().fit(X, tr).transform(X)

        for m in a.models:
            name = f"{m}_{c}"
            t0 = time.time()
            if m in LGBM_MODELS:
                X = fit(G + LGBM_MODELS[m])

                def run(prm, N, seed):
                    s, info = tab.fit_predict("lgbm", prm, X, Y[N], tr, iv, seed, a.threads, fixed)
                    return s, info

                def objective(trial):
                    prm = tab.suggest("lgbm", trial, cfg["tabular"]["search_space"])
                    s, _ = run(prm, dc["tuning_horizon"], 0)
                    return AP(Y[dc["tuning_horizon"]][iv], s[iv])
            else:
                gname, with_ctx = GNN_MODELS[m]
                x = fit(NODE)
                C = fit(CTX) if with_ctx else None
                ei, ew = graphs[gname]
                data = {"x": torch.as_tensor(x), "edge_index": ei, "edge_weight": ew}
                if with_ctx:
                    data["ctx"] = torch.as_tensor(C)

                def run(prm, N, seed):
                    y = Y[N]
                    torch.manual_seed(seed)
                    net = GNN("gat", x.shape[1], prm["hidden"], prm["heads"], prm["dropout"],
                              d_ctx=C.shape[1] if with_ctx else 0, ecope_form=prm.get("encoder", "mlp"))
                    pos_w = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
                    net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=prm["lr"],
                                                 weight_decay=prm["weight_decay"], pos_weight=pos_w,
                                                 max_epochs=dc["max_epochs"], patience=dc["patience"], seed=seed)
                    return predict(net, data), info

                def objective(trial):
                    prm = suggest_gnn(trial, dc["gnn_space"], with_ctx)
                    _, info = run(prm, dc["tuning_horizon"], 0)
                    trial.set_user_attr("epochs", info.get("epochs"))
                    return info["best_val_pr_auc"]

            if a.holdout:
                best = H.best_params(dev_out, m, c)
            else:
                study = run_study(name, out / "optuna", objective, dc["n_trials"], seed=0)
                best = study.best_params
                (out / "params").mkdir(parents=True, exist_ok=True)
                json.dump({"model": m, "context": c, "best_params": best, "best_value": study.best_value,
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
                                {"model": m, "context": c, "N": N, "seed": sd,
                                 "test_pr_auc": float(AP(Y[N][te], s[te])), **info})
                    _, info = cached_refit(out / "refits" / f"{name}_N{N}_s{sd}", refit)
            print(f"[{name}] refits done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
