"""Step 1: build the plot-level dataset (features, labels, prior presence, effort).

Inputs (paths in config.yaml):
  - sPlotOpen header + CWM/CWV tables (Sabatini et al., 2021)
  - GBIF occurrence download 0046318-260806074905277 (DOI 10.15468/dl.ps65sp)

Outputs (data/processed/):
  - gbif_filter_counts.csv          per-species record counts after each filter, per variant
  - gbif_filtered_<variant>.parquet filtered GBIF records per variant
  - plots.parquet                   one row per sPlotOpen plot with features, metadata,
                                    eligibility flags, labels, prior presence and effort,
                                    for every (variant, buffer, horizon)
"""
from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from common import ROOT, label_col, load_config, p, variant_name


# --------------------------------------------------------------------------- sPlotOpen
def load_splot(cfg: dict) -> tuple[pd.DataFrame, dict]:
    hdr = pd.read_csv(p(cfg, "splot_header"), sep="\t", low_memory=False)
    cwm = pd.read_csv(p(cfg, "splot_cwm"), sep="\t", low_memory=False)
    info = {"n_header": len(hdr), "n_cwm": len(cwm)}
    assert hdr["PlotObservationID"].is_unique and cwm["PlotObservationID"].is_unique

    traits = cfg["features"]["traits"]
    trait_cols = [f"{t}_CWM" for t in traits] + [f"{t}_CWV" for t in traits]
    keep_cwm = ["PlotObservationID"] + cfg["features"]["plot_attrs"] + trait_cols
    missing = [c for c in keep_cwm if c not in cwm.columns]
    assert not missing, f"missing CWM columns: {missing}"
    keep_hdr = cfg["features"]["meta"] + cfg["features"]["env"]
    missing = [c for c in keep_hdr if c not in hdr.columns]
    assert not missing, f"missing header columns: {missing}"

    df = hdr[keep_hdr].merge(cwm[keep_cwm], on="PlotObservationID", how="left",
                             validate="one_to_one", indicator=True)
    info["n_header_without_cwm"] = int((df["_merge"] == "left_only").sum())
    df = df.drop(columns="_merge")

    d = pd.to_datetime(df["Date_of_recording"], errors="coerce")
    df["survey_year"] = d.dt.year.astype("Int64")
    info["n_no_survey_year"] = int(df["survey_year"].isna().sum())
    info["n_no_coords"] = int(df[["Latitude", "Longitude"]].isna().any(axis=1).sum())
    info["survey_year_min"] = int(df["survey_year"].min())
    info["survey_year_max"] = int(df["survey_year"].max())
    return df, info


# --------------------------------------------------------------------------- GBIF
def load_gbif(cfg: dict) -> pd.DataFrame:
    # GBIF "simple CSV" downloads are tab-separated with no quoting.
    g = pd.read_csv(p(cfg, "gbif_download"), sep="\t", quoting=csv.QUOTE_NONE,
                    dtype=str, keep_default_na=False, na_values=[""])
    for c in ["decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters", "year"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g


def filter_gbif(g: pd.DataFrame, cfg: dict, unc_mode: str, bor_mode: str
                ) -> tuple[pd.DataFrame, list[tuple[str, pd.Series]]]:
    """Apply the documented filters in order; return filtered records + per-step counts."""
    gc = cfg["gbif"]
    steps: list[tuple[str, pd.Series]] = []

    def log(name: str, frame: pd.DataFrame) -> None:
        steps.append((name, frame.groupby("species").size()))

    x = g.copy()
    log("0_raw", x)
    x = x[x["species"].notna()]
    log("1_species_resolved", x)
    x = x[x["occurrenceStatus"] == "PRESENT"]
    log("2_status_present", x)
    x = x[x["year"].between(gc["year_min"], gc["year_max"])]
    log("3_valid_year", x)
    ok = (x["decimalLatitude"].between(-90, 90) & x["decimalLongitude"].between(-180, 180)
          & ~((x["decimalLatitude"] == 0) & (x["decimalLongitude"] == 0)))
    x = x[ok]
    log("4_valid_coords", x)
    unc = x["coordinateUncertaintyInMeters"]
    if unc_mode == "keep":
        x = x[unc.isna() | (unc <= gc["max_uncertainty_m"])]
    elif unc_mode == "drop":
        x = x[unc.notna() & (unc <= gc["max_uncertainty_m"])]
    else:
        raise ValueError(unc_mode)
    log(f"5_uncertainty_le_{gc['max_uncertainty_m']}m_missing-{unc_mode}", x)
    excl = gc["basis_of_record_excluded"].get(bor_mode, [])
    x = x[~x["basisOfRecord"].isin(excl)]
    log(f"6_basis_of_record_{bor_mode}", x)
    r = gc["dedup_round_deg"]
    key = pd.DataFrame({
        "s": x["species"],
        "la": np.round(x["decimalLatitude"] / r).astype(int),
        "lo": np.round(x["decimalLongitude"] / r).astype(int),
        "y": x["year"].astype(int),
    })
    x = x[~key.duplicated(keep="first")]
    log("7_dedup_species_coord0.01_year", x)
    return x.reset_index(drop=True), steps


# --------------------------------------------------------------------------- spatial join
def join_plots_records(plots: pd.DataFrame, recs: pd.DataFrame, variants: dict[str, np.ndarray],
                       cfg: dict, species_codes: np.ndarray) -> pd.DataFrame:
    """For every plot, count records in each buffer for each variant.

    `recs` is the union of all variants' records; `variants[v]` is a boolean mask over recs.
    The BallTree is queried once at the widest buffer; narrower buffers are distance masks.
    """
    R = cfg["spatial"]["earth_radius_km"]
    buffers = sorted(cfg["spatial"]["buffer_km_sensitivity"])
    horizons = cfg["labels"]["horizons"]
    rmax = max(buffers) / R

    rec_xy = np.radians(recs[["decimalLatitude", "decimalLongitude"]].to_numpy())
    rec_year = recs["year"].to_numpy().astype(int)
    tree = BallTree(rec_xy, metric="haversine")

    ok = plots[["Latitude", "Longitude"]].notna().all(axis=1).to_numpy()
    idx_ok = np.flatnonzero(ok)
    plot_xy = np.radians(plots.loc[ok, ["Latitude", "Longitude"]].to_numpy())
    sy_all = plots["survey_year"].to_numpy(dtype=float, na_value=np.nan)

    n = len(plots)
    out: dict[str, np.ndarray] = {}

    def acc(name: str, vals: np.ndarray, rows: np.ndarray) -> None:
        if name not in out:
            out[name] = np.zeros(n, dtype=np.int64)
        out[name][rows] += vals

    chunk = 4000
    t0 = time.time()
    for s in range(0, len(idx_ok), chunk):
        rows = idx_ok[s:s + chunk]
        ind, dist = tree.query_radius(plot_xy[s:s + chunk], r=rmax, return_distance=True)
        lens = np.fromiter((len(a) for a in ind), dtype=np.int64, count=len(ind))
        if lens.sum() == 0:
            continue
        local = np.repeat(np.arange(len(rows)), lens)
        ri = np.concatenate(ind)
        dk = np.concatenate(dist) * R
        sy = sy_all[rows][local]
        delta = rec_year[ri] - sy            # NaN if survey year unknown
        has_sy = ~np.isnan(delta)
        m = len(rows)

        for v, vmask in variants.items():
            mv = vmask[ri]
            for b in buffers:
                mb = mv & (dk <= b)
                bc = lambda w: np.bincount(local, weights=w, minlength=m).astype(np.int64)
                acc(f"n_all_b{b}_{v}", bc(mb), rows)                       # effort, any year
                acc(f"n_prior_b{b}_{v}", bc(mb & has_sy & (delta < 0)), rows)
                acc(f"n_same_b{b}_{v}", bc(mb & has_sy & (delta == 0)), rows)
                # distinct species recorded before the survey
                sel = mb & has_sy & (delta < 0)
                if sel.any():
                    u = np.unique(local[sel] * 64 + species_codes[ri[sel]])
                    acc(f"nsp_prior_b{b}_{v}", np.bincount(u // 64, minlength=m), rows)
                else:
                    acc(f"nsp_prior_b{b}_{v}", np.zeros(m, dtype=np.int64), rows)
                for N in horizons:
                    win = mb & has_sy & (delta > 0) & (delta <= N)
                    acc(f"n_win_N{N}_b{b}_{v}", bc(win), rows)
        print(f"  joined {min(s + chunk, len(idx_ok))}/{len(idx_ok)} plots "
              f"({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(out, index=plots.index)
    res.loc[~ok, :] = -1   # no coordinates -> sentinel, excluded downstream
    return res


# --------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    out_dir = p(cfg, "processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    gc, lc = cfg["gbif"], cfg["labels"]

    print("Loading sPlotOpen ...")
    plots, sinfo = load_splot(cfg)
    print(json.dumps(sinfo, indent=1))

    print("Loading GBIF download ...")
    g = load_gbif(cfg)
    print(f"  raw records: {len(g)}  species: {g['species'].nunique()}")

    # Filter variants
    variants_recs: dict[str, pd.DataFrame] = {}
    count_tables = []
    for unc in gc["missing_uncertainty_sensitivity"]:
        for bor in gc["basis_of_record_sensitivity"]:
            v = variant_name(unc, bor)
            x, steps = filter_gbif(g, cfg, unc, bor)
            variants_recs[v] = x
            t = pd.concat({k: s for k, s in steps}, axis=1).fillna(0).astype(int)
            t.loc["TOTAL"] = t.sum()
            t.insert(0, "variant", v)
            count_tables.append(t.reset_index().rename(columns={"index": "species"}))
            x.to_parquet(out_dir / f"gbif_filtered_{v}.parquet", index=False)
            print(f"  variant {v}: {len(x)} records, {x['species'].nunique()} species")
    counts = pd.concat(count_tables, ignore_index=True)
    counts.to_csv(out_dir / "gbif_filter_counts.csv", index=False)

    # Union of all variants (by gbifID) so we query the tree once
    union = pd.concat(variants_recs.values()).drop_duplicates("gbifID").reset_index(drop=True)
    vmasks = {v: union["gbifID"].isin(x["gbifID"]).to_numpy() for v, x in variants_recs.items()}
    for v, x in variants_recs.items():
        assert vmasks[v].sum() == len(x), v
    sp_codes = pd.Categorical(union["species"]).codes.astype(np.int64)
    assert sp_codes.max() < 64

    print("Spatial join (haversine BallTree) ...")
    j = join_plots_records(plots, union, vmasks, cfg, sp_codes)

    # Eligibility and labels
    sy = plots["survey_year"]
    for N in lc["horizons"]:
        plots[f"eligible_N{N}"] = ((sy >= gc["year_min"] - 1) & (sy + N <= gc["year_max"])
                                   & plots[["Latitude", "Longitude"]].notna().all(axis=1)
                                   ).fillna(False).astype(bool)
    new: dict[str, pd.Series] = {}
    for v in variants_recs:
        for b in cfg["spatial"]["buffer_km_sensitivity"]:
            for N in lc["horizons"]:
                new[label_col(N, b, v)] = (j[f"n_win_N{N}_b{b}_{v}"] > 0).astype(np.int8)
            new[f"prior_any_b{b}_{v}"] = (j[f"n_prior_b{b}_{v}"] > 0).astype(np.int8)
            # Effort excluding the records that can define the label (diagnostic only)
            for N in lc["horizons"]:
                new[f"n_outwin_N{N}_b{b}_{v}"] = j[f"n_all_b{b}_{v}"] - j[f"n_win_N{N}_b{b}_{v}"]
    df = pd.concat([plots, j, pd.DataFrame(new, index=plots.index)], axis=1)

    # Sanity checks
    assert df["PlotObservationID"].is_unique
    assert len(df) == sinfo["n_header"]
    for v in variants_recs:
        for N in lc["horizons"]:
            a, b_ = (df[label_col(N, 25, v)], df[label_col(N, 50, v)])
            c = df[label_col(N, 100, v)]
            assert (a <= b_).all() and (b_ <= c).all(), "labels must be monotone in buffer"
        for b in cfg["spatial"]["buffer_km_sensitivity"]:
            y5, y7, y10 = (df[label_col(N, b, v)] for N in (5, 7, 10))
            assert (y5 <= y7).all() and (y7 <= y10).all(), "labels must be monotone in N"

    df.to_parquet(out_dir / "plots.parquet", index=False)
    meta = {"splot": sinfo, "gbif_raw_records": int(len(g)),
            "gbif_raw_species": int(g["species"].nunique()),
            "variants": {v: int(len(x)) for v, x in variants_recs.items()},
            "n_plots": int(len(df)),
            "n_eligible": {N: int(df[f"eligible_N{N}"].sum()) for N in lc["horizons"]},
            "python": sys.version}
    with open(out_dir / "build_info.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
