"""Step 3a: tabular models (LR, RF, LightGBM, MLP) x feature sets (A, B, C) + score baselines.

Protocol per (scheme, horizon, feature set, model, outer fold):
  1. preprocessing fitted on the training rows of the outer fold (inner-val + test excluded)
  2. Optuna TPE search (n_trials from config) maximising PR-AUC on the inner validation set
  3. refit with the best parameters for each seed in config.seeds; evaluate on the outer test fold
Outputs: results/tabular/<scheme>/metrics.csv (one row per fold x seed),
         results/tabular/<scheme>/best_params.jsonl, results/tabular/<scheme>/preds/*.npz

Usage: python src/03_tabular.py --scheme spatial [--horizons 5 7 10] [--models lr rf lgbm mlp]
       [--fsets A B C] [--n-jobs 5] [--trials 30]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from common import load_config, p
from features import (FEATURE_SETS, FoldPreprocessor, assemble, folds_of, labels,
                      load_modelling_table, raw_blocks, split_indices)
from metrics import evaluate, safe_ap

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- models
def suggest(model: str, trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for k, spec in space[model].items():
        kind = spec[0]
        if kind == "float":
            out[k] = trial.suggest_float(k, spec[1], spec[2], log=bool(spec[3]) if len(spec) > 3 else False)
        elif kind == "int":
            out[k] = trial.suggest_int(k, spec[1], spec[2], log=bool(spec[3]) if len(spec) > 3 else False)
        elif kind == "cat":
            out[k] = trial.suggest_categorical(k, spec[1])
        else:
            raise ValueError(kind)
    return out


def fit_predict(model: str, prm: dict, X: np.ndarray, y: np.ndarray, tr: np.ndarray,
                iv: np.ndarray, seed: int, threads: int, fixed: dict) -> tuple[np.ndarray, dict]:
    """Fit on tr (iv used only for early stopping where applicable); return probs for all rows."""
    pos_w = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    info = {}
    if model == "lr":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=prm["C"], class_weight="balanced" if prm["balanced"] else None,
                               max_iter=3000, random_state=seed)
        m.fit(X[tr], y[tr])
        return m.predict_proba(X)[:, 1], info
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        cw = {"none": None}.get(prm["class_weight"], prm["class_weight"])
        m = RandomForestClassifier(n_estimators=fixed["rf_n_estimators"], max_depth=prm["max_depth"] or None,
                                   min_samples_leaf=prm["min_samples_leaf"],
                                   max_features=prm["max_features"], class_weight=cw,
                                   n_jobs=threads, random_state=seed)
        m.fit(X[tr], y[tr])
        return m.predict_proba(X)[:, 1], info
    if model == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=fixed["lgbm_max_rounds"], learning_rate=prm["learning_rate"],
                               num_leaves=prm["num_leaves"], min_child_samples=prm["min_child_samples"],
                               subsample=prm["subsample"], subsample_freq=1,
                               colsample_bytree=prm["colsample_bytree"], reg_lambda=prm["reg_lambda"],
                               scale_pos_weight=pos_w if prm["balanced"] else 1.0,
                               random_state=seed, n_jobs=threads, verbose=-1)
        m.fit(X[tr], y[tr], eval_X=(X[iv],), eval_y=(y[iv],), eval_metric="average_precision",
              callbacks=[lgb.early_stopping(fixed["lgbm_early_stopping"], verbose=False)])
        info["best_iteration"] = int(m.best_iteration_ or fixed["lgbm_max_rounds"])
        return m.predict_proba(X)[:, 1], info
    if model == "mlp":
        import torch
        from nn_train import MLP, predict, set_seed, train_full_batch
        torch.set_num_threads(threads)
        set_seed(seed)
        net = MLP(X.shape[1], prm["hidden"], prm["n_layers"], prm["dropout"])
        data = {"x": torch.as_tensor(X)}
        net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=prm["lr"],
                                     weight_decay=prm["weight_decay"],
                                     pos_weight=pos_w if prm["balanced"] else 1.0,
                                     max_epochs=fixed["mlp_max_epochs"], patience=fixed["patience"],
                                     minibatch=fixed["mlp_batch"], seed=seed)
        return predict(net, data), info
    raise ValueError(model)


# --------------------------------------------------------------------------- one job
def run_job(scheme: str, fs: str, model: str, fold, cfg: dict, n_trials: int, threads: int,
            horizons: list[int]) -> tuple[list[dict], dict, dict]:
    """Tune once on the inner-validation PR-AUC of the tuning horizon, then refit the best
    parameters with every seed for every horizon and evaluate on the outer test fold."""
    t0 = time.time()
    tc = cfg["tabular"]
    e = load_modelling_table(cfg)
    tr, iv, te = split_indices(e, scheme, fold)
    assert not set(tr) & set(te) and not set(iv) & set(te) and not set(tr) & set(iv)
    Xdf = assemble(raw_blocks(e, cfg), FEATURE_SETS[fs])
    pre = FoldPreprocessor(scale=True).fit(Xdf, tr)        # fitted on training rows only
    X = pre.transform(Xdf)
    Y = {N: labels(e, cfg, N) for N in horizons}
    preds, rows, best = {"test_idx": te}, [], {}

    if model.startswith("score_"):                            # non-learned ranking baselines
        s = raw_blocks(e, cfg)["prior"]["prior_log_n"].to_numpy()
        for N in horizons:
            for seed in cfg["seeds"]:
                rows.append({**evaluate(Y[N][te], s[te], prob=False), "seed": seed, "horizon": N})
        preds["scores"] = s[te]
    else:
        yt = labels(e, cfg, cfg["evaluation"]["tuning_horizon"])
        if yt[tr].sum() == 0 or yt[iv].sum() == 0:
            raise RuntimeError(f"no positives in train/inner-val for {scheme} fold {fold}")
        sampler = optuna.samplers.TPESampler(seed=cfg["evaluation"]["split_seed"])
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            prm = suggest(model, trial, tc["search_space"])
            s, _ = fit_predict(model, prm, X, yt, tr, iv, seed=0, threads=threads, fixed=tc["fixed"])
            return safe_ap(yt[iv], s[iv])

        study.optimize(objective, n_trials=n_trials)
        best = study.best_params
        for N in horizons:
            y = Y[N]
            for seed in cfg["seeds"]:
                s, info = fit_predict(model, best, X, y, tr, iv, seed=seed, threads=threads,
                                      fixed=tc["fixed"])
                rows.append({**evaluate(y[te], s[te]), "seed": seed, "horizon": N,
                             "inner_val_pr_auc": safe_ap(y[iv], s[iv]), **info})
                preds[f"N{N}_seed{seed}"] = s[te]
    for N in horizons:
        preds[f"y_test_N{N}"] = Y[N][te]
    for r in rows:
        r.update({"scheme": scheme, "fset": fs, "model": model, "fold": str(fold),
                  "n_train": len(tr), "n_inner_val": len(iv), "seconds": time.time() - t0})
    return rows, {"scheme": scheme, "fset": fs, "model": model, "fold": str(fold),
                  "tuning_horizon": cfg["evaluation"]["tuning_horizon"], "n_trials": n_trials,
                  "best_params": best, "n_features": int(X.shape[1])}, preds


def collect(out) -> None:
    parts = sorted((out / "metrics_parts").glob("*.csv"))
    if parts:
        pd.concat([pd.read_csv(x) for x in parts]).to_csv(out / "metrics.csv", index=False)
        with open(out / "best_params.jsonl", "w") as fh:
            for x in sorted((out / "metrics_parts").glob("*.json")):
                fh.write(open(x).read().strip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="spatial", choices=["spatial", "random", "loco"])
    ap.add_argument("--models", nargs="+", default=["score_prior", "lr", "lgbm", "rf", "mlp"])
    ap.add_argument("--fsets", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--trials", type=int)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--folds", nargs="+", help="subset of outer folds (debugging only)")
    ap.add_argument("--out", help="output directory override (debugging only)")
    a = ap.parse_args()
    cfg = load_config()
    H = cfg["labels"]["horizons"]
    n_trials = a.trials or cfg["tabular"]["n_trials"]
    out = Path(a.out) if a.out else p(cfg, "results_dir") / "tabular" / a.scheme
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "metrics_parts").mkdir(exist_ok=True)
    e = load_modelling_table(cfg)
    folds = [f for f in folds_of(e, a.scheme, cfg) if not a.folds or str(f) in a.folds]

    jobs = []
    for model in a.models:
        for fs in (["C"] if model.startswith("score_") else a.fsets):
            for f in folds:
                tag = f"{fs if not model.startswith('score_') else '-'}_{model}_f{str(f).replace(' ', '_')}"
                if not (out / "preds" / f"{tag}.npz").exists():          # resume support
                    jobs.append((fs, model, f, tag))
    # slowest first for better load balancing
    jobs.sort(key=lambda j: {"mlp": 0, "rf": 1, "lgbm": 2}.get(j[1], 3))
    print(f"{len(jobs)} jobs ({a.scheme}), {n_trials} trials each, n_jobs={a.n_jobs}", flush=True)

    def wrapped(fs, model, f, tag):
        rows, bp, preds = run_job(a.scheme, fs, model, f, cfg, n_trials, a.threads, H)
        if model.startswith("score_"):
            for r in rows:
                r["fset"] = "-"
            bp["fset"] = "-"
        np.savez_compressed(out / "preds" / f"{tag}.npz", **preds)
        pd.DataFrame(rows).to_csv(out / "metrics_parts" / f"{tag}.csv", index=False)
        with open(out / "metrics_parts" / f"{tag}.json", "w") as fh:
            json.dump(bp, fh)
        r = pd.DataFrame(rows).groupby("horizon").pr_auc.mean().round(3).to_dict()
        print(f"  {tag}: PR-AUC by horizon {r} ({time.time() - T0:.0f}s since start)", flush=True)

    T0 = time.time()
    Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(wrapped)(*j) for j in jobs)
    collect(out)


if __name__ == "__main__":
    main()
