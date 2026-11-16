"""Independent sampling-effort layer from the GBIF SQL download (citable; see
data/gbif_effort_sql/README.md and reports/decision_log.md D8).

Input: <key>.csv with columns cell05deg (GBIF_DMSGCode 1800 = 0.5-degree cell, lower-left corner),
year, is_target (1 = one of the 20 fungal target species, matched by name), n_occurrences.
Effort uses NON-target Fungi only (is_target = 0).

Plot-level effort = sum over 0.5-degree cells whose centre lies within buffer_km of the plot:
  effort_fungi_prior   years [2000, survey_year - 1]   (legitimate feature: past only)
  effort_fungi_all     years [2000, 2025]              (diagnostic only: uses future years)
Output: data/processed/effort_layer.parquet (same columns as the API version, which it replaces).
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from common import ROOT, load_config, p
from features import load_modelling_table

DMS = re.compile(r"LON(-?)(\d+)-(\d+)-(\d+)LAT(-?)(\d+)-(\d+)-(\d+)")


def parse_cell(code: str) -> tuple[float, float]:
    """'CRS4326RES0-30-0LON10-30-0LAT52-0-0' -> lower-left (lat, lon) in decimal degrees."""
    m = DMS.search(code)
    if not m:
        raise ValueError(f"unexpected cell code: {code}")
    s1, d1, m1, x1, s2, d2, m2, x2 = m.groups()
    lon = (int(d1) + int(m1) / 60 + int(x1) / 3600) * (-1 if s1 else 1)
    lat = (int(d2) + int(m2) / 60 + int(x2) / 3600) * (-1 if s2 else 1)
    return lat, lon


def main() -> None:
    cfg = load_config()
    res, R, b = cfg["effort"]["cell_deg"], cfg["spatial"]["earth_radius_km"], cfg["spatial"]["buffer_km"]
    y0, y1 = cfg["gbif"]["year_min"], cfg["gbif"]["year_max"]
    sql_dir = ROOT / "data" / "gbif_effort_sql"
    key = (sql_dir / "download_key_v2.txt").read_text().strip()
    raw = pd.read_csv(sql_dir / f"{key}.csv", sep="\t", dtype={"cell05deg": str})
    print(f"{len(raw):,} rows; target rows {int((raw.is_target == 1).sum()):,}; "
          f"non-target occurrences {int(raw.loc[raw.is_target == 0, 'n_occurrences'].sum()):,}")
    raw = raw[raw["cell05deg"].notna() & raw["year"].between(y0, y1)]

    cells = pd.DataFrame({"cell05deg": raw["cell05deg"].unique()})
    ll = np.array([parse_cell(c) for c in cells["cell05deg"]])
    cells["lat"], cells["lon"] = ll[:, 0] + res / 2, ll[:, 1] + res / 2      # cell centres
    assert cells["lat"].between(-90, 90).all() and cells["lon"].between(-180, 180).all()
    # sanity: cell corners must sit on the 0.5-degree lattice
    assert np.allclose((ll / res) % 1, 0) or np.allclose(((ll / res) % 1) % 1, 0), "grid mismatch"

    nt = raw[raw["is_target"] == 0].pivot_table(index="cell05deg", columns="year",
                                                 values="n_occurrences", aggfunc="sum", fill_value=0)
    years = np.arange(y0, y1 + 1)
    F = nt.reindex(index=cells["cell05deg"], columns=years, fill_value=0).to_numpy(float)
    cum = np.concatenate([np.zeros((len(F), 1)), np.cumsum(F, axis=1)], axis=1)

    e = load_modelling_table(cfg)
    tree = BallTree(np.radians(cells[["lat", "lon"]].to_numpy()), metric="haversine")
    ind = tree.query_radius(np.radians(e[["Latitude", "Longitude"]].to_numpy()), r=b / R)
    sy = e["survey_year"].to_numpy().astype(int)
    prior = np.array([cum[c, np.clip(s - y0, 0, len(years))].sum() for c, s in zip(ind, sy)])
    allyrs = np.array([cum[c, -1].sum() for c in ind])
    out = pd.DataFrame({"PlotObservationID": e["PlotObservationID"], "effort_fungi_prior": prior,
                        "effort_fungi_all": allyrs, "effort_n_cells": [len(c) for c in ind]})
    out.to_parquet(p(cfg, "processed_dir") / "effort_layer.parquet", index=False)

    # which target names were matched (synonyms under the new taxonomy may be missed)
    meta = {"source": "GBIF SQL download", "download_key": key, "cell_deg": res, "buffer_km": b,
            "n_rows": int(len(raw)), "n_cells_with_fungi": int(len(cells)),
            "nontarget_occurrences": int(raw.loc[raw.is_target == 0, "n_occurrences"].sum()),
            "target_occurrences": int(raw.loc[raw.is_target == 1, "n_occurrences"].sum()),
            "plots_with_zero_cells": int((out.effort_n_cells == 0).sum())}
    json.dump(meta, open(p(cfg, "processed_dir") / "effort_layer_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print(out.describe().T)


if __name__ == "__main__":
    main()
