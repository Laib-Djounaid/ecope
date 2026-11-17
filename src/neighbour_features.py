"""Neighbourhood features for the tabular reference "G_lgbm + voisinage" (decision D12, B1).

All quantities are prior presence / effort (never labels) and are time-safe: a plot only sees
neighbours surveyed in the same year or earlier (same rule as graph_models.time_causal), and
radius counts use target records strictly before the plot's own survey year.
- kNN block: over the k geographic neighbours (k = graph.k, same symmetrised geo graph as the GNNs,
  causal edges only): mean log prior count, max prior_any, mean log effort, number of neighbours kept.
- radius block: target records within 100 km before the survey year (log1p), and in the ring
  50-100 km (log1p), from the filtered GBIF records of the main variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from common import p, variant_name
from graph_models import geo_knn_graph, time_causal


def neighbour_block(e: pd.DataFrame, blk: dict, cfg: dict) -> pd.DataFrame:
    R, k, B = cfg["spatial"]["earth_radius_km"], cfg["graph"]["k"], cfg["spatial"]["buffer_km"]
    years = e["survey_year"].to_numpy().astype(int)
    ei, ew = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), k, B, R)
    ei, _ = time_causal(ei, ew, years)
    src, dst = ei.numpy()
    n = len(e)
    deg = np.bincount(dst, minlength=n).astype(float)

    def mean_(v: np.ndarray) -> np.ndarray:
        s = np.bincount(dst, weights=v[src], minlength=n)
        return np.where(deg > 0, s / np.maximum(deg, 1), np.nan)   # NaN -> imputed + indicator

    pr = blk["prior"]
    any_max = np.zeros(n)
    np.maximum.at(any_max, dst, pr["prior_any"].to_numpy()[src])
    out = pd.DataFrame({"nbr_prior_log_n_mean": mean_(pr["prior_log_n"].to_numpy()),
                        "nbr_prior_any_max": np.where(deg > 0, any_max, np.nan),
                        "nbr_effort_log_mean": mean_(blk["effort"]["effort_fungi_prior_log"].to_numpy()),
                        "nbr_n_causal": deg})
    # radius counts of target records before the plot's own survey year
    V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
    rec = pd.read_parquet(p(cfg, "processed_dir") / f"gbif_filtered_{V}.parquet")
    tree = BallTree(np.radians(rec[["decimalLatitude", "decimalLongitude"]].to_numpy()), metric="haversine")
    ry = rec["year"].to_numpy().astype(int)
    ind, dist = tree.query_radius(np.radians(e[["Latitude", "Longitude"]].to_numpy()), r=100 / R,
                                  return_distance=True)
    n100 = np.array([(ry[ix] < years[i]).sum() for i, ix in enumerate(ind)])
    n50 = np.array([((ry[ix] < years[i]) & (d * R <= B)).sum() for i, (ix, d) in enumerate(zip(ind, dist))])
    assert (n50 == e[f"n_prior_b{B}_{V}"].to_numpy()).all(), "50 km prior count not reproduced"
    out["prior100_log_n"] = np.log1p(n100)
    out["prior_ring50_100_log_n"] = np.log1p(n100 - n50)
    return out
