"""Pre-registered pilot of EcoPE-OD-GNN (decision D10; criteria in reports/decision_log.md).

Development data only (the sealed hold-out is never loaded). N = 5, seed 0, default
hyperparameters. Two settings:
  cv       : spatial 5-fold CV on the development set (splits_v2)
  transfer : train on high-effort development blocks, test on low-effort blocks
Configurations:
  G_lgbm            LightGBM on feature set G (prior + effort + context)
  gat_concat_P_geo  GAT, nodes = [prior, context incl. effort], geo graph (best pilot-v3 model)
  od_cloglog        EcoPE-OD-GNN (PRIMARY): presence = EcoPE-GAT on geo graph, nodes = prior +
                    traits, context = climate PCs, lat/lon, biome (NO effort); detection =
                    monotone cloglog in log effort only
  od_mlp            same, detection = free small MLP of log effort
  od_concat         same as primary, context concatenated instead of EcoPE encoder
  od_nodet          presence branch alone (no detection branch, no effort anywhere)
Outputs (reports/pilot_runs/<tag>/): per-part predictions (resumable), metrics, the three
pre-registered tests, out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

from common import ROOT, load_config
from features import FEATURE_SETS, FoldPreprocessor, assemble, labels, load_modelling_table, raw_blocks
from graph_models import GNN, ODGNN, geo_knn_graph
from nn_train import predict, train_full_batch

N, SEED = 5, 0
DEF = dict(hidden=64, heads=4, dropout=0.3, lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=20)
CTX_PRES = ["env", "geo", "biome"]                 # presence context: NO effort
CTX_FULL = ["env", "geo", "biome", "effort"]
CONFIGS = ["G_lgbm", "gat_concat_P_geo", "od_cloglog", "od_mlp", "od_concat", "od_nodet"]


def settings(e, cfg):
    """(setting, fold, train_idx, inner_val_idx, test_idx)"""
    out = []
    f = e["spatial_fold"].to_numpy()
    for k in range(cfg["evaluation"]["n_folds"]):
        iv = e[f"spatial_inner_val_f{k}"].to_numpy()
        out.append(("cv", k, np.flatnonzero((f != k) & ~iv), np.flatnonzero(iv), np.flatnonzero(f == k)))
    role, iv = e["transfer_role"].to_numpy(), e["transfer_inner_val"].to_numpy()
    out.append(("transfer", 0, np.flatnonzero((role == "train_high") & ~iv), np.flatnonzero(iv),
                np.flatnonzero(role == "test_low")))
    for _, _, tr, va, te in out:
        assert not set(tr) & set(te) and not set(va) & set(te) and not set(tr) & set(va)
    return out


def prior_block(blk, e, mode):
    if mode == "raw":
        return blk["prior"]
    # effort-normalised prior presence: share of target-pathogen records among all prior
    # fungal + target records near the plot (target-group ratio)
    n_t = np.expm1(blk["prior"]["prior_log_n"])
    return pd.DataFrame({"prior_ratio": n_t / (n_t + e["effort_fungi_prior"] + 1.0),
                         "prior_any": blk["prior"]["prior_any"]})


def fold_mean_boot(y, s1, s2, folds, blocks, n_boot=2000, seed=0):
    """CI of mean-over-folds AP(s2) - AP(s1); blocks resampled within each test fold."""
    rng = np.random.default_rng(seed)
    idx = {k: {b: np.flatnonzero((folds == k) & (blocks == b)) for b in np.unique(blocks[folds == k])}
           for k in np.unique(folds)}
    d = []
    for _ in range(n_boot):
        diffs = []
        for k, bb in idx.items():
            keys = list(bb)
            ix = np.concatenate([bb[b] for b in rng.choice(keys, len(keys))])
            if 0 < y[ix].sum() < len(ix):
                diffs.append(AP(y[ix], s2[ix]) - AP(y[ix], s1[ix]))
        d.append(np.mean(diffs))
    return np.percentile(d, [2.5, 97.5])


def abs_rho_boot(eff, s_ref, s_new, blocks, n_boot=2000, seed=0):
    """CI of |rho(s_new, effort)| - |rho(s_ref, effort)|, spatial blocks resampled."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    by = {b: np.flatnonzero(blocks == b) for b in ub}
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([by[b] for b in rng.choice(ub, len(ub))])
        d.append(abs(spearmanr(s_new[ix], eff[ix]).statistic) - abs(spearmanr(s_ref[ix], eff[ix]).statistic))
    return np.percentile(d, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prior-mode", choices=["raw", "ratio"], required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=DEF["max_epochs"], help="smoke tests only")
    ap.add_argument("--only", nargs="+", help="smoke tests only: subset of configs")
    ap.add_argument("--first-n-settings", type=int, help="smoke tests only")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config()
    out = ROOT / "reports" / "pilot_runs" / a.tag
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    e = load_modelling_table(cfg)                      # development set only (hold-out dropped)
    assert "holdout" in e.columns and not e["holdout"].any()
    y = labels(e, cfg, N)
    blk = raw_blocks(e, cfg)
    pri = prior_block(blk, e, a.prior_mode)
    log_eff = np.log1p(e["effort_fungi_prior"].to_numpy())
    ei, ew = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), cfg["graph"]["k"],
                           cfg["spatial"]["buffer_km"], cfg["spatial"]["earth_radius_km"])
    print(f"development plots {len(e):,}; geo graph edges {ei.shape[1]:,}; prior mode {a.prior_mode}", flush=True)
    configs = a.only or CONFIGS
    sets = settings(e, cfg)[: a.first_n_settings] if a.first_n_settings else settings(e, cfg)

    for setting, k, tr, iv, te in sets:
        def fit(x_df):
            return FoldPreprocessor().fit(x_df, tr).transform(x_df)
        PT = fit(pd.concat([pri, blk["traits"], blk["attrs"]], axis=1))     # OD presence: --prior-mode
        P_only = fit(blk["prior"])                 # references always use RAW prior presence
        Cp = fit(assemble(blk, CTX_PRES))
        Cf = fit(assemble(blk, CTX_FULL))
        z = (log_eff - log_eff[tr].mean()) / log_eff[tr].std()
        for c in configs:
            tag = f"{setting}_{c}_f{k}"
            if (parts / f"{tag}.json").exists():
                continue
            t0 = time.time()
            info, pres = {}, None
            if c == "G_lgbm":
                Xdf = assemble(blk, ["prior", "effort", "env", "geo", "biome"])    # = feature set G, raw prior
                X = fit(Xdf)
                m = lgb.LGBMClassifier(random_state=SEED, n_jobs=a.threads, verbose=-1).fit(X[tr], y[tr])
                s = m.predict_proba(X)[:, 1]
                # effort-standardised prediction (Warton et al. 2013 style): effort set to the
                # training median for every plot (used by test c only)
                Xs = Xdf.copy()
                Xs["effort_fungi_prior_log"] = float(np.median(Xdf["effort_fungi_prior_log"].to_numpy()[tr]))
                pres = m.predict_proba(FoldPreprocessor().fit(Xdf, tr).transform(Xs))[:, 1]
            else:
                if c == "gat_concat_P_geo":
                    x, ctx, det, model_kind, pos_w = np.c_[P_only, Cf], Cp, "none", "plain", None
                else:
                    x = PT if c != "od_concat" else np.c_[PT, Cp]
                    ctx, model_kind, pos_w = Cp, "od", 1.0
                    det = {"od_cloglog": "cloglog", "od_mlp": "mlp", "od_concat": "cloglog", "od_nodet": "none"}[c]
                data = {"x": torch.as_tensor(x.astype(np.float32)), "ctx": torch.as_tensor(ctx),
                        "edge_index": ei, "edge_weight": ew,
                        "det": torch.as_tensor(z.astype(np.float32)).view(-1, 1)}
                torch.manual_seed(SEED)
                use_enc = c not in ("gat_concat_P_geo", "od_concat")
                pres_net = GNN("gat", x.shape[1], DEF["hidden"], DEF["heads"], DEF["dropout"],
                               d_ctx=ctx.shape[1] if use_enc else 0)
                if model_kind == "plain":
                    net = pres_net
                    pos_w = float((y[tr] == 0).sum() / (y[tr] == 1).sum())       # as in pilot v3
                else:
                    net = ODGNN(pres_net, det)
                net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=DEF["lr"],
                                             weight_decay=DEF["weight_decay"], pos_weight=pos_w,
                                             max_epochs=a.max_epochs, patience=DEF["patience"], seed=SEED)
                s = predict(net, data)
                if model_kind == "od":
                    with torch.no_grad():
                        net.eval()
                        pres = torch.sigmoid(net.presence_logit(data)).numpy()
                        if det == "cloglog":
                            info["det_a"] = float(net.a)
                            info["det_b"] = float(torch.nn.functional.softplus(net.b_raw))
            np.save(parts / f"{tag}.npy", s[te].astype(np.float32))
            if pres is not None:
                np.save(parts / f"{tag}_presence.npy", pres[te].astype(np.float32))
            rec = {"setting": setting, "config": c, "fold": k, "n_test": int(len(te)),
                   "n_pos": int(y[te].sum()), "chance": float(y[te].mean()),
                   "pr_auc": AP(y[te], s[te]), "roc_auc": AUC(y[te], s[te]),
                   "seconds": time.time() - t0, **info}
            json.dump(rec, open(parts / f"{tag}.json", "w"))
            print(f"  {tag}: PR-AUC {rec['pr_auc']:.3f} ROC {rec['roc_auc']:.3f} ({rec['seconds']:.0f}s)", flush=True)

    # ---------------- collect + pre-registered tests (once all parts exist)
    df = pd.DataFrame([json.load(open(f)) for f in sorted(parts.glob("*.json"))])
    df.to_csv(out / "pilot_parts.csv", index=False)
    need = {(s_, c) for s_, *_ in settings(e, cfg) for c in CONFIGS}
    have = {(r.setting, r.config) for r in df.itertuples()}
    if not need <= have or len(df) < len(CONFIGS) * (cfg["evaluation"]["n_folds"] + 1):
        print("incomplete: tests not computed"); return
    sets = settings(e, cfg)
    folds = e["spatial_fold"].to_numpy()
    blocks = e["block_id"].to_numpy()
    oof, oof_pres = {}, {}
    for c in CONFIGS:
        o, op = np.full(len(e), np.nan), np.full(len(e), np.nan)
        for setting, k, tr, iv, te in sets:
            if setting != "cv":
                continue
            o[te] = np.load(parts / f"cv_{c}_f{k}.npy")
            f_ = parts / f"cv_{c}_f{k}_presence.npy"
            if f_.exists():
                op[te] = np.load(f_)
        oof[c], oof_pres[c] = o, op
    summ = df.groupby(["setting", "config"]).agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                                                 roc_auc_mean=("roc_auc", "mean")).reset_index()
    summ.to_csv(out / "pilot_summary.csv", index=False)

    rows = []
    P = "od_cloglog"
    # (a) spatial CV, fold-mean PR-AUC
    for ref in ["G_lgbm", "gat_concat_P_geo", "od_concat", "od_nodet", "od_mlp"]:
        lo, hi = fold_mean_boot(y, oof[ref], oof[P], folds, blocks)
        d = np.mean([AP(y[folds == k], oof[P][folds == k]) - AP(y[folds == k], oof[ref][folds == k])
                     for k in np.unique(folds)])
        rows.append({"test": "a_cv_foldmean", "config": P, "reference": ref, "diff": d, "ci_low": lo, "ci_high": hi})
    # (b) transfer high -> low effort
    _, _, trT, ivT, teT = sets[-1]
    yt, bt = y[teT], blocks[teT]
    sT = {c: np.load(parts / f"transfer_{c}_f0.npy") for c in CONFIGS}
    for ref in ["G_lgbm", "gat_concat_P_geo", "od_concat", "od_nodet", "od_mlp"]:
        lo, hi = importlib_boot(yt, sT[ref], sT[P], bt)
        rows.append({"test": "b_transfer", "config": P, "reference": ref,
                     "diff": AP(yt, sT[P]) - AP(yt, sT[ref]), "ci_low": lo, "ci_high": hi})
    # (c) effort dependence of the presence output (development out-of-fold predictions)
    rho = {"od_presence": spearmanr(oof_pres[P], log_eff).statistic,
           **{c: spearmanr(oof[c], log_eff).statistic for c in ["G_lgbm", "gat_concat_P_geo"]},
           "G_lgbm_effort_standardised": spearmanr(oof_pres["G_lgbm"], log_eff).statistic}
    for ref, s_ref in [("G_lgbm", oof["G_lgbm"]), ("gat_concat_P_geo", oof["gat_concat_P_geo"]),
                       ("G_lgbm_effort_standardised", oof_pres["G_lgbm"])]:
        lo, hi = abs_rho_boot(log_eff, s_ref, oof_pres[P], blocks)
        rows.append({"test": "c_abs_rho_presence_minus_ref", "config": "od_cloglog presence", "reference": ref,
                     "diff": abs(rho["od_presence"]) - abs(rho[ref]), "ci_low": lo, "ci_high": hi})
    tests = pd.DataFrame(rows)
    tests.to_csv(out / "preregistered_tests.csv", index=False)
    json.dump({k_: float(v) for k_, v in rho.items()}, open(out / "rho_effort.json", "w"), indent=1)
    pd.DataFrame({**{f"oof_{c}": oof[c] for c in CONFIGS},
                  **{f"oof_presence_{c}": oof_pres[c] for c in CONFIGS},
                  "y": y, "block": blocks, "fold": folds, "log_effort": log_eff,
                  "lat": e["Latitude"], "lon": e["Longitude"]}).to_parquet(out / "pilot_oof.parquet")
    pd.set_option("display.width", 230)
    print(summ.round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))
    print(json.dumps({k_: round(float(v), 3) for k_, v in rho.items()}))


def importlib_boot(y, s_ref, s_new, blocks):
    import importlib
    return importlib.import_module("pilot_n5").block_bootstrap_diff(y, s_ref, s_new, blocks, n_boot=2000, seed=0)


if __name__ == "__main__":
    main()
