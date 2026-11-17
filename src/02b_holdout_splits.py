"""Decision D10, step 0: sealed confirmatory hold-out + development splits (splits_v2).

1. Confirmatory hold-out: random spatial blocks (~15% of plots), stratified by continent x
   effort tercile (block-median log1p of independent prior effort), seed fixed in config. It is
   excluded from every pilot and tuning run and evaluated ONCE at the very end
   (features.load_modelling_table drops it unless include_holdout=True is passed explicitly).
2. Development spatial CV: StratifiedGroupKFold(5) over the remaining blocks, stratified on the
   N = 10 label, with spatially blocked inner validation (as in 02_splits.py).
3. Effort-transfer split (pre-registered test b): development blocks whose block-median
   log1p(prior effort) is above the development median train the model (with a blocked inner
   validation); blocks at or below the median are the transfer test set.

Output: data/processed/splits_v2.parquet, splits_v2_summary.json, holdout_manifest.csv
(block ids + a SHA-256 of the sorted hold-out plot ids, to show it never changed).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import label_col, load_config, p, variant_name

import importlib
s1 = importlib.import_module("02_splits")


def main() -> None:
    cfg = load_config()
    ev, hc = cfg["evaluation"], cfg["holdout"]
    proc = p(cfg, "processed_dir")
    d = pd.read_parquet(proc / "plots.parquet")
    eff = pd.read_parquet(proc / "effort_layer.parquet")
    V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
    B = cfg["spatial"]["buffer_km"]
    e = d[d["eligible_N5"]].merge(eff, on="PlotObservationID", how="left", validate="one_to_one")
    e = e.reset_index(drop=True)
    blocks = s1.spatial_blocks(e["Latitude"].to_numpy(), e["Longitude"].to_numpy(),
                               ev["block_km"], cfg["spatial"]["earth_radius_km"])
    e["block_id"] = blocks
    e["log_eff"] = np.log1p(e["effort_fungi_prior"])

    # ---------------- 1. hold-out: stratified random blocks
    bt = e.groupby("block_id").agg(n=("PlotObservationID", "size"), eff=("log_eff", "median"),
                                   cont=("Continent", lambda s: s.mode()[0]))
    bt["eff_t"] = pd.qcut(bt["eff"].rank(method="first"), 3, labels=["low", "mid", "high"])
    rng = np.random.default_rng(hc["seed"])
    hold = []
    for _, g in bt.groupby(["cont", "eff_t"], observed=True):
        target = hc["fraction"] * g["n"].sum()
        cum = 0
        for b in rng.permutation(g.index.to_numpy()):
            if cum >= target:
                break
            if cum + g.loc[b, "n"] > 1.6 * target and cum > 0:
                continue                       # skip a block that would overshoot badly
            hold.append(b)
            cum += g.loc[b, "n"]
    is_hold = e["block_id"].isin(hold).to_numpy()

    # ---------------- 2. development spatial CV
    dev = np.flatnonzero(~is_hold)
    strat = e[label_col(ev["stratify_horizon"], B, V)].to_numpy()
    fold = np.full(len(e), -1)
    sg = StratifiedGroupKFold(ev["n_folds"], shuffle=True, random_state=ev["split_seed"])
    for k, (_, te) in enumerate(sg.split(np.zeros(len(dev)), strat[dev], blocks[dev])):
        fold[dev[te]] = k
    out = pd.DataFrame({"PlotObservationID": e["PlotObservationID"], "block_id": blocks,
                        "holdout": is_hold, "spatial_fold": fold, "continent": e["Continent"]})
    for k in range(ev["n_folds"]):
        tr = np.flatnonzero((fold != k) & ~is_hold)
        m = np.zeros(len(e), dtype=bool)
        m[tr[s1.inner_split(strat[tr], blocks[tr], ev["split_seed"] + 100 + k, ev["n_inner"])]] = True
        out[f"spatial_inner_val_f{k}"] = m

    # ---------------- 3. effort-transfer split (development set only)
    bdev = e.loc[dev].groupby("block_id")["log_eff"].median()
    thr = float(bdev.median())
    high_blocks = set(bdev[bdev > thr].index)
    tr_role = np.where(is_hold, "holdout",
                       np.where(e["block_id"].isin(high_blocks), "train_high", "test_low"))
    out["transfer_role"] = tr_role
    trh = np.flatnonzero(tr_role == "train_high")
    m = np.zeros(len(e), dtype=bool)
    m[trh[s1.inner_split(strat[trh], blocks[trh], ev["split_seed"] + 400, ev["n_inner"])]] = True
    out["transfer_inner_val"] = m

    # ---------------- sanity checks
    assert (fold[~is_hold] >= 0).all() and (fold[is_hold] == -1).all()
    assert not set(blocks[is_hold]) & set(blocks[~is_hold]), "hold-out block leaks into dev"
    for k in range(ev["n_folds"]):
        assert not set(blocks[fold == k]) & set(blocks[(fold != k) & (fold >= 0)])
    assert not set(blocks[tr_role == "train_high"]) & set(blocks[tr_role == "test_low"])

    ids = np.sort(e.loc[is_hold, "PlotObservationID"].to_numpy())
    digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    pd.DataFrame({"block_id": sorted(hold)}).to_csv(proc / "holdout_manifest.csv", index=False)
    out.to_parquet(proc / "splits_v2.parquet", index=False)

    def stats(mask):
        return {"plots": int(mask.sum()), "blocks": int(len(np.unique(blocks[mask]))),
                **{f"pos_N{N}": int(e.loc[mask, label_col(N, B, V)].sum()) for N in cfg["labels"]["horizons"]},
                **{f"prev_N{N}": round(float(e.loc[mask, label_col(N, B, V)].mean()), 4) for N in cfg["labels"]["horizons"]},
                "median_log_effort": round(float(e.loc[mask, "log_eff"].median()), 3)}
    summ = {"holdout": {**stats(is_hold), "fraction_plots": round(float(is_hold.mean()), 4),
                        "sha256_sorted_plot_ids": digest,
                        "by_continent": e.loc[is_hold, "Continent"].value_counts().to_dict()},
            "development": stats(~is_hold),
            "dev_folds": {k: stats(fold == k) for k in range(ev["n_folds"])},
            "transfer": {"threshold_block_median_log_effort": thr,
                         "train_high": stats(tr_role == "train_high"),
                         "test_low": stats(tr_role == "test_low")}}
    json.dump(summ, open(proc / "splits_v2_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
