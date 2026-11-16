"""Cross-check the GBIF SQL effort layer against the GBIF API counts (independent route) on the
cells queried through both. Validates cell-code decoding (corner convention) and filters.
Usage: python tools/check_effort_cells.py"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
m = importlib.import_module("01d_effort_from_sql")
key = (ROOT / "data/gbif_effort_sql/download_key_v2.txt").read_text().strip()
sql = pd.read_csv(ROOT / f"data/gbif_effort_sql/{key}.csv", sep="\t")
ll = np.array([m.parse_cell(c) for c in sql.cell05deg])
sql["lat0"], sql["lon0"] = np.round(ll[:, 0], 3), np.round(ll[:, 1], 3)

res = 0.5
lat_c, lon_c = np.arange(-90 + res / 2, 90, res), np.arange(-180 + res / 2, 180, res)
G = np.array(np.meshgrid(lat_c, lon_c, indexing="ij")).reshape(2, -1).T
api = pd.read_csv(ROOT / "data/external/gbif_fungi_effort_cells_0.5deg.csv")
both = api.groupby("cell").group.nunique()
api = api[api.cell.isin(both[both == 2].index) & (api.year >= 0)]
a = api.pivot_table(index="cell", columns="group", values="n", aggfunc="sum", fill_value=0)
a = a.reindex(both[both == 2].index, fill_value=0)
a["api_nontarget"] = a.get("fungi", 0) - a.get("target", 0)
a["api_target"] = a.get("target", 0)
a["lat0"] = np.round(G[a.index, 0] - res / 2, 3)
a["lon0"] = np.round(G[a.index, 1] - res / 2, 3)

for dl, dn in [(0, 0), (0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5)]:
    s = sql.assign(lat0=np.round(sql.lat0 + dl, 3), lon0=np.round(sql.lon0 + dn, 3))
    nt = s[s.is_target == 0].groupby(["lat0", "lon0"]).n_occurrences.sum().rename("sql_nontarget")
    tg = s[s.is_target == 1].groupby(["lat0", "lon0"]).n_occurrences.sum().rename("sql_target")
    j = a.reset_index().merge(nt.reset_index(), on=["lat0", "lon0"], how="left") \
         .merge(tg.reset_index(), on=["lat0", "lon0"], how="left").fillna(0)
    rho = j[["api_nontarget", "sql_nontarget"]].corr("spearman").iloc[0, 1]
    big = j.api_nontarget >= 50
    ratio = (j.sql_nontarget / j.api_nontarget)[big].median()
    label = "as decoded" if (dl, dn) == (0, 0) else f"shifted ({dl:+},{dn:+})"
    print(f"{label:>18}: cells {len(j)}, Spearman non-target {rho:.3f}, median SQL/API {ratio:.3f}, "
          f"target totals SQL {int(j.sql_target.sum())} vs API {int(j.api_target.sum())}")
