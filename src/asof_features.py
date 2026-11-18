"""As-of neighbour features for EcoPE-GAT_asof (decision D14, option C; reports/preregistration_D14*.md).

For every directed geographic edge j -> i, the neighbour's time-stamped features are recomputed
AS OF THE RECEIVER's survey year t_i (records with year < t_i), instead of j's own year t_j:
  prior_log_n      log1p(# target records within buffer_km of j, year < t_i)
  prior_n_species  # distinct target species among them
  prior_any        1 if any
  effort_fungi_prior_log  log1p(non-target fungal records in the 0.5-degree cells within
                   buffer_km of j, years [year_min, t_i - 1])   (same rule as 01d_effort_from_sql)
Self-check: for j = i the as-of features reproduce the stored node features exactly (asserted), so
edge and node features share one definition and one preprocessing.
Edge attributes: distance (km, from the stored weight) and signed gap (t_i - t_j) / 10.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree

from common import ROOT, p, variant_name

COLS = ["prior_log_n", "prior_n_species", "prior_any", "effort_fungi_prior_log"]


def _record_index(e: pd.DataFrame, cfg: dict):
    """Per plot: sorted years of target records within the buffer, and sorted first-years per species."""
    R, B = cfg["spatial"]["earth_radius_km"], cfg["spatial"]["buffer_km"]
    V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
    rec = pd.read_parquet(p(cfg, "processed_dir") / f"gbif_filtered_{V}.parquet",
                          columns=["decimalLatitude", "decimalLongitude", "year", "species"])
    rec = rec[rec["year"].notna()]
    ry = rec["year"].to_numpy().astype(int)
    sp = pd.factorize(rec["species"])[0]
    tree = BallTree(np.radians(rec[["decimalLatitude", "decimalLongitude"]].to_numpy()), metric="haversine")
    ind = tree.query_radius(np.radians(e[["Latitude", "Longitude"]].to_numpy()), r=B / R)
    yrs, first = [], []
    for ix in ind:
        yrs.append(np.sort(ry[ix]))
        if len(ix):
            f = pd.Series(ry[ix]).groupby(sp[ix]).min().to_numpy()
            first.append(np.sort(f))
        else:
            first.append(np.array([], dtype=int))
    return yrs, first, V, B


def _effort_cum(e: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, int]:
    """Per plot: cumulative non-target fungal records over years (column k = years < year_min + k)."""
    m01d = importlib.import_module("01d_effort_from_sql")
    res, R, B = cfg["effort"]["cell_deg"], cfg["spatial"]["earth_radius_km"], cfg["spatial"]["buffer_km"]
    y0, y1 = cfg["gbif"]["year_min"], cfg["gbif"]["year_max"]
    sql_dir = ROOT / "data" / "gbif_effort_sql"
    key = (sql_dir / "download_key_v2.txt").read_text().strip()
    raw = pd.read_csv(sql_dir / f"{key}.csv", sep="\t", dtype={"cell05deg": str})
    raw = raw[raw["cell05deg"].notna() & raw["year"].between(y0, y1)]
    cells = pd.DataFrame({"cell05deg": raw["cell05deg"].unique()})
    ll = np.array([m01d.parse_cell(c) for c in cells["cell05deg"]])
    cells["lat"], cells["lon"] = ll[:, 0] + res / 2, ll[:, 1] + res / 2
    nt = raw[raw["is_target"] == 0].pivot_table(index="cell05deg", columns="year", values="n_occurrences",
                                                 aggfunc="sum", fill_value=0)
    years = np.arange(y0, y1 + 1)
    F = nt.reindex(index=cells["cell05deg"], columns=years, fill_value=0).to_numpy(float)
    cum = np.concatenate([np.zeros((len(F), 1)), np.cumsum(F, axis=1)], axis=1)
    tree = BallTree(np.radians(cells[["lat", "lon"]].to_numpy()), metric="haversine")
    ind = tree.query_radius(np.radians(e[["Latitude", "Longitude"]].to_numpy()), r=B / R)
    P = np.stack([cum[c].sum(0) if len(c) else np.zeros(cum.shape[1]) for c in ind])
    return P, y0


def asof_edges(e: pd.DataFrame, cfg: dict, ei: torch.Tensor, ew: torch.Tensor
               ) -> tuple[pd.DataFrame, torch.Tensor]:
    """Edge-level as-of features (DataFrame, columns COLS, one row per edge) and edge attributes."""
    src, dst = ei.numpy()
    t = e["survey_year"].to_numpy().astype(int)
    yrs, first, V, B = _record_index(e, cfg)
    P, y0 = _effort_cum(e, cfg)

    def feats(j: np.ndarray, ti: np.ndarray) -> pd.DataFrame:
        n = np.array([np.searchsorted(yrs[a], b, side="left") for a, b in zip(j, ti)])
        s = np.array([np.searchsorted(first[a], b, side="left") for a, b in zip(j, ti)])
        eff = P[j, np.clip(ti - y0, 0, P.shape[1] - 1)]
        return pd.DataFrame({"prior_log_n": np.log1p(n), "prior_n_species": s.astype(float),
                             "prior_any": (n > 0).astype(float), "effort_fungi_prior_log": np.log1p(eff)})

    # self-check: j = i reproduces the stored node features
    me = np.arange(len(e))
    s0 = feats(me, t)
    assert (np.expm1(s0["prior_log_n"]).round().astype(int).to_numpy() == e[f"n_prior_b{B}_{V}"].to_numpy()).all(), \
        "as-of prior count does not reproduce the node feature"
    assert (s0["prior_n_species"].to_numpy() == e[f"nsp_prior_b{B}_{V}"].to_numpy()).all(), \
        "as-of species count does not reproduce the node feature"
    assert np.allclose(np.expm1(s0["effort_fungi_prior_log"]), e["effort_fungi_prior"].to_numpy()), \
        "as-of effort does not reproduce the node feature"

    xe = feats(src, t[dst])
    # causal property: no record counted for edge j -> i has year >= t_i (searchsorted side='left')
    scale = cfg["spatial"]["buffer_km"]
    dist_km = scale * (1.0 / ew.numpy() - 1.0)
    ea = torch.as_tensor(np.stack([dist_km / scale, (t[dst] - t[src]) / 10.0], 1), dtype=torch.float32)
    return xe, ea


def leak_check(e: pd.DataFrame, cfg: dict, ei: torch.Tensor, xe: pd.DataFrame, N: int, n_sample: int = 20000,
               seed: int = 0) -> dict:
    """D2 for the as-of graph: share of edges whose message counts a record in the receiver's label
    window (t_i, t_i + N]. Must be 0 by construction; brute-force on a random sample of edges."""
    yrs, _, _, _ = _record_index(e, cfg)
    src, dst = ei.numpy()
    t = e["survey_year"].to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(src), size=min(n_sample, len(src)), replace=False)
    n_used = np.round(np.expm1(xe["prior_log_n"].to_numpy()[idx])).astype(int)
    bad = 0
    for k, n in zip(idx, n_used):
        used = yrs[src[k]][:n]                    # the n earliest records = those with year < t_i
        bad += int(((used > t[dst[k]]) & (used <= t[dst[k]] + N)).any() or (used >= t[dst[k]]).any())
    return {"edges_checked": int(len(idx)), "edges_with_window_information": bad}
