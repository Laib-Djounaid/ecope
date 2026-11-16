"""Independent sampling-effort layer from the GBIF occurrence search API.

For every 0.5-degree grid cell whose centre lies within the buffer of at least one eligible plot,
we count, per year 2000-2025, GBIF records of kingdom Fungi (taxonKey 5) with coordinates, no
geospatial issue, occurrenceStatus PRESENT, and basisOfRecord other than MATERIAL_SAMPLE /
FOSSIL_SPECIMEN (same exclusion as the main label variant). A second query counts the records of
the target pathogen species that are Fungi; it is subtracted so that the effort group is
independent of the label species.

Plot-level effort = sum over cells whose centre lies within buffer_km of the plot:
  effort_fungi_prior   records in years [2000, survey_year - 1]   (legitimate feature)
  effort_fungi_all     records in years [2000, 2025]              (diagnostic only: future years)

Counts are cached in data/external/ (resumable). The query template and access date are written
to data/external/effort_layer_meta.json for citation.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from common import ROOT, load_config, p
from features import load_modelling_table

API = "https://api.gbif.org/v1/occurrence/search"
BASIS = ["HUMAN_OBSERVATION", "PRESERVED_SPECIMEN", "OBSERVATION", "MATERIAL_CITATION",
         "LIVING_SPECIMEN", "MACHINE_OBSERVATION", "OCCURRENCE"]


def query(lat0, lon0, res, taxon_keys, y0, y1) -> dict[int, int]:
    params = [("decimalLatitude", f"{lat0},{lat0 + res}"), ("decimalLongitude", f"{lon0},{lon0 + res}"),
              ("year", f"{y0},{y1}"), ("hasCoordinate", "true"), ("hasGeospatialIssue", "false"),
              ("occurrenceStatus", "PRESENT"), ("limit", "0"), ("facet", "year"), ("facetLimit", "50")]
    params += [("taxonKey", str(k)) for k in taxon_keys] + [("basisOfRecord", b) for b in BASIS]
    url = API + "?" + urllib.parse.urlencode(params)
    attempt, waits = 0, 0
    while attempt < 6:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                js = json.load(r)
            facets = js.get("facets") or [{"counts": []}]
            return {int(c["name"]): int(c["count"]) for c in facets[0]["counts"]}
        except urllib.error.HTTPError as err:
            LAST_ERR[0] = repr(err)[:200]
            if err.code == 429 and waits < 20:          # rate limited: wait as asked, do not
                ra = err.headers.get("Retry-After")     # count it as a failed attempt
                time.sleep(float(ra) if ra and ra.isdigit() else 30)
                waits += 1
                continue
            attempt += 1
            time.sleep(2 ** attempt)
        except Exception as err:
            LAST_ERR[0] = repr(err)[:200]
            attempt += 1
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GBIF query failed: {url}")


LAST_ERR = [""]


def main() -> None:
    cfg = load_config()
    ec = cfg["effort"]
    res, R, b = ec["cell_deg"], cfg["spatial"]["earth_radius_km"], cfg["spatial"]["buffer_km"]
    y0, y1 = cfg["gbif"]["year_min"], cfg["gbif"]["year_max"]
    out_dir = ROOT / "data" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"gbif_fungi_effort_cells_{res}deg.csv"

    e = load_modelling_table(cfg)
    lat_c = np.arange(-90 + res / 2, 90, res)
    lon_c = np.arange(-180 + res / 2, 180, res)
    G = np.array(np.meshgrid(lat_c, lon_c, indexing="ij")).reshape(2, -1).T
    tree = BallTree(np.radians(G), metric="haversine")
    ind = tree.query_radius(np.radians(e[["Latitude", "Longitude"]].to_numpy()), r=b / R)
    cells = np.unique(np.concatenate(ind))

    # target species that are Fungi (their records are subtracted)
    raw = pd.read_csv(p(cfg, "gbif_download"), sep="\t", quoting=3, dtype=str,
                      keep_default_na=False, na_values=[""])
    target_keys = sorted(raw.loc[raw["kingdom"] == "Fungi", "speciesKey"].dropna().unique().astype(int))

    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["cell", "group", "year", "n"])
    have = set(zip(done["cell"], done["group"]))
    todo = [(c, g) for c in cells for g in ("fungi", "target") if (c, g) not in have]
    print(f"{len(cells)} cells, {len(todo)} queries to run", flush=True)
    rows, t0 = [], time.time()

    def work(c, g):
        lat0, lon0 = G[c, 0] - res / 2, G[c, 1] - res / 2
        cnt = query(lat0, lon0, res, [5] if g == "fungi" else target_keys, y0, y1)
        return [(c, g, yr, n) for yr, n in cnt.items()] or [(c, g, -1, 0)]   # -1: no records

    # Batches of 200 queries, cached after each batch. A watchdog abandons a batch that makes no
    # progress for `stall_s` seconds (a hung connection) and retries its unfinished queries with a
    # fresh thread pool, so one stuck request can no longer block the whole download.
    stall_s, batch, n_done = 600, 200, 0
    queue = list(todo)
    while queue:
        cur, queue = queue[:batch], queue[batch:]
        ex = ThreadPoolExecutor(max_workers=ec["api_workers"])
        futs = {ex.submit(work, c, g): (c, g) for c, g in cur}
        finished = set()
        n_err = 0
        try:
            for f in as_completed(futs, timeout=stall_s):
                try:
                    rows += f.result()
                    finished.add(futs[f])
                except Exception:                      # this query failed after all retries
                    n_err += 1
        except TimeoutError:
            pass
        if n_err:
            print(f"  {n_err} queries failed after retries (requeued); last error: {LAST_ERR[0]}", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        retry = [cg for cg in cur if cg not in finished]
        queue = retry + queue
        n_done += len(finished)
        pd.concat([done, pd.DataFrame(rows, columns=done.columns)]).to_csv(cache, index=False)
        print(f"  {n_done}/{len(todo)} queries ({time.time() - t0:.0f}s)"
              + (f", {len(retry)} retried after stall" if retry else ""), flush=True)
        if retry and len(retry) == len(cur):
            time.sleep(60)                             # nothing worked: back off before retrying
    allc = pd.concat([done, pd.DataFrame(rows, columns=done.columns)]) if rows else done
    allc.to_csv(cache, index=False)

    # cell x year matrix of non-target fungal records
    allc = allc[allc["year"] >= 0]
    tab = allc.pivot_table(index="cell", columns=["group", "year"], values="n", aggfunc="sum", fill_value=0)
    years = np.arange(y0, y1 + 1)
    F = np.zeros((len(G), len(years)))
    for g, sign in (("fungi", 1), ("target", -1)):
        if g in tab.columns.get_level_values(0):
            sub = tab[g].reindex(columns=years, fill_value=0)
            F[sub.index.to_numpy(dtype=int)] += sign * sub.to_numpy()
    F = np.clip(F, 0, None)
    cum = np.concatenate([np.zeros((len(G), 1)), np.cumsum(F, axis=1)], axis=1)  # cum[:, j] = years < y0+j

    sy = e["survey_year"].to_numpy().astype(int)
    prior, allyrs = np.zeros(len(e)), np.zeros(len(e))
    for i, cs in enumerate(ind):
        j = np.clip(sy[i] - y0, 0, len(years))            # number of years in [y0, sy-1]
        prior[i] = cum[cs, j].sum()
        allyrs[i] = cum[cs, -1].sum()
    out = pd.DataFrame({"PlotObservationID": e["PlotObservationID"], "effort_fungi_prior": prior,
                        "effort_fungi_all": allyrs, "effort_n_cells": [len(c) for c in ind]})
    out.to_parquet(p(cfg, "processed_dir") / "effort_layer.parquet", index=False)
    meta = {"api": API, "basisOfRecord": BASIS, "fungi_taxonKey": 5,
            "subtracted_target_speciesKeys": [int(k) for k in target_keys], "cell_deg": res,
            "buffer_km": b, "years": [y0, y1], "n_cells": int(len(cells)),
            "accessed": dt.date.today().isoformat()}
    json.dump(meta, open(out_dir / "effort_layer_meta.json", "w"), indent=1)
    print(out.describe().T)


if __name__ == "__main__":
    main()
    os._exit(0)   # do not wait for abandoned (hung) request threads
