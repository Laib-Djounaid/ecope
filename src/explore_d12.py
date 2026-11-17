"""Decision D12, B2 (EXPLORATORY, development data only, already-used splits): effort transfer
(train on high-effort blocks, test on low-effort blocks), N = 5, seed 0, default hyperparameters.

Compares, with block-bootstrap 95% CIs of PR-AUC differences on the transfer test set:
  EcoPE-GAT (current best configuration = gat_concat_P_geo) vs G_lgbm, vs G_lgbm + neighbourhood,
  and vs the same geo GAT without ecological context (prior presence only),
each with the ORIGINAL geo graph (as in pilots v3/v4; temporal leakage through neighbours, see
tools/neighbour_time_leak.py) and with the TIME-CAUSAL geo graph (graph_models.time_causal).
Output: reports/pilot_runs/<tag>/.   Usage: python src/explore_d12.py --tag x1_d12_transfer
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
from features import FoldPreprocessor, assemble, labels, load_modelling_table, raw_blocks
from graph_models import GNN, geo_knn_graph, time_causal
from neighbour_features import neighbour_block
from nn_train import predict, train_full_batch

pilot_od = importlib.import_module("pilot_od")
boot = importlib.import_module("pilot_n5").block_bootstrap_diff
N, SEED = 5, 0
GAT_SEEDS = [0, 1, 2, 3, 4]
DEF = pilot_od.DEF
G = ["prior", "effort", "env", "geo", "biome"]
CONFIGS = ["G_lgbm", "G_lgbm_nbr", "gat_concat_P_geo_leaky", "gat_P_geo_noctx_leaky",
           "gat_concat_P_geo_causal", "gat_P_geo_noctx_causal"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = load_config()
    out = ROOT / "reports" / "pilot_runs" / a.tag
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    e = load_modelling_table(cfg)
    assert not e["holdout"].any()
    y = labels(e, cfg, N)
    blk = raw_blocks(e, cfg)
    blk["nbr"] = neighbour_block(e, blk, cfg)
    ei0, ew0 = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), cfg["graph"]["k"],
                             cfg["spatial"]["buffer_km"], cfg["spatial"]["earth_radius_km"])
    graphs = {"leaky": (ei0, ew0), "causal": time_causal(ei0, ew0, e["survey_year"].to_numpy())}
    print({g: int(v[0].shape[1]) for g, v in graphs.items()}, flush=True)
    _, _, tr, iv, te = [s for s in pilot_od.settings(e, cfg) if s[0] == "transfer"][0]

    def fit(x_df: pd.DataFrame) -> np.ndarray:
        return FoldPreprocessor().fit(x_df, tr).transform(x_df)

    # LightGBM is deterministic here: one fit. GATs: 5 seeds (a single seed proved unstable:
    # seed-0 run stopped at epoch 4 with near-constant scores), tests use the seed-mean score.
    def seeds_of(c):
        return [SEED] if c.startswith("G_lgbm") else GAT_SEEDS

    for c in CONFIGS:
        for sd in seeds_of(c):
            name = f"{c}_s{sd}"
            if (parts / f"{name}.json").exists():
                continue
            t0, info = time.time(), {}
            if c.startswith("G_lgbm"):
                X = fit(assemble(blk, G + (["nbr"] if c == "G_lgbm_nbr" else [])))
                s = lgb.LGBMClassifier(random_state=sd, n_jobs=a.threads, verbose=-1).fit(X[tr], y[tr]).predict_proba(X)[:, 1]
            else:
                # node features: prior presence only (noctx) | prior + effort + context (= set G)
                x = fit(blk["prior"]) if "noctx" in c else fit(assemble(blk, G))
                ei, ew = graphs[c.rsplit("_", 1)[1]]
                data = {"x": torch.as_tensor(x), "edge_index": ei, "edge_weight": ew}
                torch.manual_seed(sd)
                net = GNN("gat", x.shape[1], DEF["hidden"], DEF["heads"], DEF["dropout"])
                pos_w = float((y[tr] == 0).sum() / (y[tr] == 1).sum())
                net, info = train_full_batch(net, data, torch.as_tensor(y), tr, iv, lr=DEF["lr"],
                                             weight_decay=DEF["weight_decay"], pos_weight=pos_w,
                                             max_epochs=DEF["max_epochs"], patience=DEF["patience"], seed=sd)
                s = predict(net, data)
            np.save(parts / f"{name}.npy", s[te].astype(np.float32))
            rec = {"config": c, "seed": sd, "pr_auc": AP(y[te], s[te]), "roc_auc": AUC(y[te], s[te]),
                   "seconds": time.time() - t0, **info}
            json.dump(rec, open(parts / f"{name}.json", "w"))
            print(f"  {name}: PR-AUC {rec['pr_auc']:.3f} ROC {rec['roc_auc']:.3f} "
                  f"(best epoch {info.get('best_epoch', '-')}, {rec['seconds']:.0f}s)", flush=True)

    yt, bt = y[te], e["block_id"].to_numpy()[te]
    S = {c: np.mean([np.load(parts / f"{c}_s{sd}.npy") for sd in seeds_of(c)], axis=0) for c in CONFIGS}
    per_seed = pd.DataFrame([json.load(open(f)) for f in sorted(parts.glob("*.json"))])
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summ = per_seed.groupby("config").agg(pr_auc_seed_mean=("pr_auc", "mean"), pr_auc_seed_sd=("pr_auc", "std"),
                                          pr_auc_seed_min=("pr_auc", "min"), pr_auc_seed_max=("pr_auc", "max"),
                                          roc_auc_seed_mean=("roc_auc", "mean")).reindex(CONFIGS).reset_index()
    summ["pr_auc_of_seed_mean_score"] = [AP(yt, S[c]) for c in CONFIGS]
    summ["chance"] = yt.mean()
    rows = []
    for g in ["leaky", "causal"]:
        P = f"gat_concat_P_geo_{g}"
        for ref in ["G_lgbm", "G_lgbm_nbr", f"gat_P_geo_noctx_{g}"]:
            lo, hi = boot(yt, S[ref], S[P], bt, n_boot=2000, seed=0)
            rows.append({"graph": g, "model": P, "reference": ref,
                         "diff": AP(yt, S[P]) - AP(yt, S[ref]), "ci_low": lo, "ci_high": hi})
    tests = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "exploratory_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
