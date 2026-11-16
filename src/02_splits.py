"""Step 2: evaluation splits.

Writes data/processed/splits.parquet with, for every eligible plot:
  block_id                     ~500 km equal-area-ish spatial block
  spatial_fold                 primary: spatially blocked 5-fold CV (StratifiedGroupKFold)
  spatial_inner_val            True if the plot is in the inner validation set of the
                               training part when its own fold is NOT the test fold -> stored
                               as one column per outer fold: inner_val_f{k}
  random_fold                  secondary: stratified random 5-fold CV
  random_inner_val_f{k}        inner validation for the random CV (random 20% of training)
  continent                    for leave-one-continent-out (folds = continents)
  loco_inner_val_{continent}   inner validation (spatial blocks) for LOCO
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import label_col, load_config, p, variant_name


def spatial_blocks(lat: np.ndarray, lon: np.ndarray, block_km: float, R: float) -> np.ndarray:
    """Grid of ~block_km x block_km cells: fixed-height latitude bands, longitude width
    scaled by 1/cos(latitude) so that cells have roughly constant area."""
    dlat = np.degrees(block_km / R)
    band = np.floor((lat + 90.0) / dlat).astype(int)
    centre = -90.0 + (band + 0.5) * dlat
    dlon = np.minimum(360.0, dlat / np.maximum(np.cos(np.radians(centre)), 1e-3))
    ncell = np.maximum(1, np.floor(360.0 / dlon)).astype(int)
    cell = np.floor((lon + 180.0) / 360.0 * ncell).astype(int) % ncell
    return band * 10000 + cell


def inner_split(y: np.ndarray, groups: np.ndarray | None, seed: int, n_inner: int) -> np.ndarray:
    """Boolean mask: one of n_inner folds of the given rows (grouped if groups given)."""
    if groups is None:
        cv = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
        _, va = next(cv.split(np.zeros(len(y)), y))
    else:
        cv = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
        _, va = next(cv.split(np.zeros(len(y)), y, groups))
    m = np.zeros(len(y), dtype=bool)
    m[va] = True
    return m


def main() -> None:
    cfg = load_config()
    ev = cfg["evaluation"]
    proc = p(cfg, "processed_dir")
    d = pd.read_parquet(proc / "plots.parquet")
    V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
    e = d[d["eligible_N5"]].reset_index(drop=True)
    strat = e[label_col(ev["stratify_horizon"], cfg["spatial"]["buffer_km"], V)].to_numpy()

    out = pd.DataFrame({"PlotObservationID": e["PlotObservationID"]})
    blocks = spatial_blocks(e["Latitude"].to_numpy(), e["Longitude"].to_numpy(),
                            ev["block_km"], cfg["spatial"]["earth_radius_km"])
    out["block_id"] = blocks
    K, seed = ev["n_folds"], ev["split_seed"]

    # primary: spatially blocked CV
    sg = StratifiedGroupKFold(K, shuffle=True, random_state=seed)
    fold = np.full(len(e), -1)
    for k, (_, te) in enumerate(sg.split(np.zeros(len(e)), strat, blocks)):
        fold[te] = k
    out["spatial_fold"] = fold
    for k in range(K):
        tr = np.flatnonzero(fold != k)
        m = np.zeros(len(e), dtype=bool)
        m[tr[inner_split(strat[tr], blocks[tr], seed + 100 + k, ev["n_inner"])]] = True
        out[f"spatial_inner_val_f{k}"] = m

    # secondary: stratified random CV
    rf = np.full(len(e), -1)
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=seed)
                                .split(np.zeros(len(e)), strat)):
        rf[te] = k
    out["random_fold"] = rf
    for k in range(K):
        tr = np.flatnonzero(rf != k)
        m = np.zeros(len(e), dtype=bool)
        m[tr[inner_split(strat[tr], None, seed + 200 + k, ev["n_inner"])]] = True
        out[f"random_inner_val_f{k}"] = m

    # leave-one-continent-out
    cont = e["Continent"].to_numpy()
    out["continent"] = cont
    for c in sorted(pd.unique(cont)):
        tr = np.flatnonzero(cont != c)
        m = np.zeros(len(e), dtype=bool)
        m[tr[inner_split(strat[tr], blocks[tr], seed + 300, ev["n_inner"])]] = True
        out[f"loco_inner_val_{c.replace(' ', '_')}"] = m

    # ---------------- sanity checks
    assert (fold >= 0).all() and (rf >= 0).all()
    for k in range(K):
        te_blocks = set(blocks[fold == k])
        tr_blocks = set(blocks[fold != k])
        assert not te_blocks & tr_blocks, "a spatial block is in both train and test"
        iv = out[f"spatial_inner_val_f{k}"].to_numpy()
        assert not (iv & (fold == k)).any(), "inner validation overlaps test fold"
        assert not set(blocks[iv]) & set(blocks[(fold != k) & ~iv]), "inner val block leak"
        assert not (out[f"random_inner_val_f{k}"].to_numpy() & (rf == k)).any()

    summ = {"n_plots": len(e), "n_blocks": int(len(np.unique(blocks))), "block_km": ev["block_km"],
            "spatial_folds": {}, "random_folds": {}, "loco": {}}
    for name, f in [("spatial_folds", fold), ("random_folds", rf)]:
        for k in range(K):
            mk = f == k
            summ[name][k] = {"n": int(mk.sum()), "n_blocks": int(len(np.unique(blocks[mk]))),
                             **{f"prev_N{N}": round(float(e.loc[mk, label_col(N, cfg['spatial']['buffer_km'], V)].mean()), 4)
                                for N in cfg["labels"]["horizons"]}}
    for c in sorted(pd.unique(cont)):
        mk = cont == c
        summ["loco"][c] = {"n": int(mk.sum()), **{
            f"prev_N{N}": round(float(e.loc[mk, label_col(N, cfg['spatial']['buffer_km'], V)].mean()), 4)
            for N in cfg["labels"]["horizons"]}}
    out.to_parquet(proc / "splits.parquet", index=False)
    with open(proc / "splits_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
