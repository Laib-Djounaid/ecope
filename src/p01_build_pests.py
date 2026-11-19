"""D15 step 1: build the node tables, labels, features and splits of the pest replication
(reports/preregistration_D15.md). No model is fitted here.

Per case: nodes = background sampling events (tools/gbif_api_background.py), years t in [2008, 2022],
random subsample of at most 30,000 (seed 0). Target records = tools/gbif_api_target.py.
  labels   y3 / y5: target record within 50 km in (t, t + N]   (y5 only defined for t <= 2020)
  prior    n_prior (target within 50 km, year < t), prior_any, prior_years (distinct years)
  effort   background records of the node's GADM-2 district, years 2005 .. t - 1 (API facet counts;
           2005 = first year downloaded), log1p; NaN when the district is unknown (imputed later)
  context  WorldClim 2.1 10' bio1, bio5, bio6, bio12; latitude, longitude
  persist  log1p(target within 100 km, year < t)  (persistence heuristic, H7/R7)
Splits: ~500 km blocks; sealed hold-out = one of 5 StratifiedGroupKFold folds over blocks (stratified
on y3, seed 0); development: 5 spatial folds (seed 42) + inner validations (seed 42 + 100 + k) and
the all-development inner validation (seed 542), as in D13.
Outputs: data/processed/pests/<case>_dev.parquet, <case>_holdout.parquet (hold-out written once, never
read by development code), reports/pests/<case>_summary.json with the hold-out SHA-256.
"""
from __future__ import annotations

import hashlib
import importlib
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree

from common import ROOT, load_config

s2 = importlib.import_module("02_splits")
RAW = ROOT / "data" / "raw_pests"
CASES = {
    "P1_halys_europe": ("Halyomorpha_halys", "bg_Hemiptera_europe"),
    "P2_halys_namerica": ("Halyomorpha_halys", "bg_Hemiptera_north_america"),
    "P3_popillia_namerica": ("Popillia_japonica", "bg_Coleoptera_north_america"),
}
T0, T1, N_MAX_NODES, B, R100 = 2008, 2022, 30_000, 50.0, 100.0
BIO = [1, 5, 6, 12]


def worldclim(lat, lon):
    out = {}
    r = np.clip(np.floor((90 - lat) * 6).astype(int), 0, 1079)
    c = np.clip(np.floor((lon + 180) * 6).astype(int), 0, 2159)
    for b in BIO:
        a = np.array(Image.open(ROOT / "data" / "external" / "worldclim" / f"wc2.1_10m_bio_{b}.tif"))
        v = a[r, c].astype(float)
        v[v < -1e30] = np.nan
        out[f"bio{b}"] = v
    return pd.DataFrame(out)


def build(case: str, cfg: dict) -> None:
    tgt, bg = CASES[case]
    Rk = cfg["spatial"]["earth_radius_km"]
    T = pd.read_parquet(RAW / f"{tgt}.parquet").dropna(subset=["decimalLatitude", "decimalLongitude", "year"])
    E = pd.read_parquet(RAW / f"{bg}_events.parquet").dropna(subset=["lat", "lon", "year"])
    E = E[E["year"].between(T0, T1)].drop_duplicates("key")
    E = E.sample(n=min(N_MAX_NODES, len(E)), random_state=0).sort_values("key").reset_index(drop=True)
    t = E["year"].to_numpy().astype(int)
    ty = T["year"].to_numpy().astype(int)
    tree = BallTree(np.radians(T[["decimalLatitude", "decimalLongitude"]].to_numpy()), metric="haversine")
    X = np.radians(E[["lat", "lon"]].to_numpy())
    near50 = tree.query_radius(X, r=B / Rk)
    near100 = tree.query_radius(X, r=R100 / Rk)
    d = pd.DataFrame({"key": E["key"], "lat": E["lat"], "lon": E["lon"], "t": t, "countryCode": E["countryCode"],
                      "gid2": E["gid2"]})
    d["n_prior"] = [int((ty[ix] < ti).sum()) for ix, ti in zip(near50, t)]
    d["prior_years"] = [len(np.unique(ty[ix][ty[ix] < ti])) for ix, ti in zip(near50, t)]
    d["prior_any"] = (d["n_prior"] > 0).astype(int)
    d["persist100"] = np.log1p([int((ty[ix] < ti).sum()) for ix, ti in zip(near100, t)])
    for N in (3, 5):
        y = np.array([int(((ty[ix] > ti) & (ty[ix] <= ti + N)).any()) for ix, ti in zip(near50, t)], dtype=float)
        if 2025 - N < T1:
            y[t > 2025 - N] = np.nan
        d[f"y{N}"] = y
    # dated effort from district-year counts
    eff = pd.read_parquet(RAW / f"{bg}_effort.parquet")
    piv = eff.pivot_table(index="gid2", columns="year", values="n", aggfunc="sum", fill_value=0)
    cum = piv.reindex(columns=range(2005, 2026), fill_value=0).cumsum(axis=1)
    known = d["gid2"].isin(cum.index).to_numpy()
    e_prior = np.full(len(d), np.nan)
    kk = np.flatnonzero(known & (t > 2005))
    e_prior[kk] = cum.loc[d["gid2"].to_numpy()[kk]].to_numpy()[np.arange(len(kk)), t[kk] - 1 - 2005]
    e_prior[known & (t <= 2005)] = 0.0
    d["effort_prior"] = e_prior
    d = pd.concat([d, worldclim(d["lat"].to_numpy(), d["lon"].to_numpy())], axis=1)
    # splits
    ev = cfg["evaluation"]
    d["block_id"] = s2.spatial_blocks(d["lat"].to_numpy(), d["lon"].to_numpy(), ev["block_km"], Rk)
    y3 = d["y3"].to_numpy().astype(int)
    blocks = d["block_id"].to_numpy()
    _, ho = next(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(d)), y3, blocks))
    d["holdout"] = False
    d.loc[ho, "holdout"] = True
    dev = d[~d["holdout"]].reset_index(drop=True)
    hold = d[d["holdout"]].reset_index(drop=True)
    assert not set(dev["block_id"]) & set(hold["block_id"])
    yd, bd = dev["y3"].to_numpy().astype(int), dev["block_id"].to_numpy()
    fold = np.full(len(dev), -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=ev["split_seed"]).split(
            np.zeros(len(dev)), yd, bd)):
        fold[te] = k
    dev["spatial_fold"] = fold
    for k in range(5):
        tr = np.flatnonzero(fold != k)
        m = np.zeros(len(dev), dtype=bool)
        m[tr[s2.inner_split(yd[tr], bd[tr], ev["split_seed"] + 100 + k, ev["n_inner"])]] = True
        dev[f"spatial_inner_val_f{k}"] = m
        assert not set(bd[m]) & set(bd[(fold != k) & ~m]) and not (m & (fold == k)).any()
    m = np.zeros(len(dev), dtype=bool)
    m[s2.inner_split(yd, bd, ev["split_seed"] + 500, ev["n_inner"])] = True
    dev["alldev_inner_val"] = m
    out = ROOT / "data" / "processed" / "pests"
    out.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(out / f"{case}_dev.parquet", index=False)
    hpath = out / f"{case}_holdout.parquet"
    hold.to_parquet(hpath, index=False)
    sha = hashlib.sha256(",".join(map(str, sorted(hold["key"]))).encode()).hexdigest()
    summ = {"case": case, "target": tgt, "background": bg, "n_nodes": int(len(d)), "n_dev": int(len(dev)),
            "n_holdout": int(len(hold)), "blocks_dev": int(dev["block_id"].nunique()),
            "blocks_holdout": int(hold["block_id"].nunique()), "holdout_keys_sha256": sha,
            "prev_y3_dev": float(dev["y3"].mean()), "prev_y5_dev": float(dev["y5"].mean()),
            "share_prior_any_dev": float(dev["prior_any"].mean()),
            "emergence_nodes_dev": int((dev["prior_any"] == 0).sum()),
            "emergence_pos_y3_dev": int(((dev["prior_any"] == 0) & (dev["y3"] == 1)).sum()),
            "share_effort_missing": float(d["effort_prior"].isna().mean()),
            "share_climate_missing": float(d["bio1"].isna().mean()),
            "years": dev["t"].value_counts().sort_index().to_dict(),
            "fold_prev_y3": dev.groupby("spatial_fold")["y3"].mean().round(4).to_dict()}
    rep = ROOT / "reports" / "pests"
    rep.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(rep / f"{case}_summary.json", "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in summ.items() if k != "years"}, indent=1, default=int), flush=True)


if __name__ == "__main__":
    import sys
    cfg = load_config()
    for c in (sys.argv[1:] or CASES):
        build(c, cfg)
