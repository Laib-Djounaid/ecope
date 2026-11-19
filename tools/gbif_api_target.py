"""D15 (agricultural pest replication): download the occurrence records of one target pest from the
public GBIF occurrence-search API (no account needed), year by year (and month by month when a
year exceeds the API paging limit). Filters: coordinates present, no geospatial issue, status
PRESENT, years 2000-2025. Raw API pages are not kept; one parquet per species plus a JSON manifest
(query parameters, counts per year, retrieval date) for the citation of a GBIF derived dataset.

Usage: python tools/gbif_api_target.py "Halyomorpha halys" [--out data/raw_pests]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.gbif.org/v1"
FIELDS = ["key", "decimalLatitude", "decimalLongitude", "year", "month", "day", "eventDate", "basisOfRecord",
          "coordinateUncertaintyInMeters", "countryCode", "datasetKey", "publishingOrgKey", "species",
          "taxonKey", "establishmentMeans", "issues"]
LIMIT, MAX_OFFSET = 300, 100_000


def get(url, params, tries=6):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60, headers={"User-Agent": "research-client/1.0"})
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(min(2 ** k, 120))
    raise RuntimeError(f"GBIF API failed: {url} {params}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--out", default="data/raw_pests")
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2025)
    a = ap.parse_args()
    m = get(f"{API}/species/match", {"name": a.name, "strict": "true"})
    assert m.get("matchType") == "EXACT" and m.get("rank") == "SPECIES", m
    base = {"taxonKey": m["usageKey"], "hasCoordinate": "true", "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT"}
    rows, per_year = [], {}

    def pull(params):
        n = get(f"{API}/occurrence/search", {**base, **params, "limit": 0})["count"]
        if n > MAX_OFFSET:
            return None, n
        got = []
        for off in range(0, n, LIMIT):
            res = get(f"{API}/occurrence/search", {**base, **params, "limit": LIMIT, "offset": off})["results"]
            got += [{f: (r.get(f) if f != "issues" else ";".join(r.get("issues", []))) for f in FIELDS} for r in res]
        return got, n

    for y in range(a.y0, a.y1 + 1):
        got, n = pull({"year": y})
        if got is None:                                   # too many for one query: split by month
            got = []
            for mo in range(1, 13):
                g, nm = pull({"year": y, "month": mo})
                assert g is not None, f"{y}-{mo}: {nm} records exceed the API limit"
                got += g
        per_year[y] = {"api_count": n, "retrieved": len(got)}
        rows += got
        print(f"{a.name} {y}: {n} -> {len(got)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("key")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = a.name.replace(" ", "_")
    df.to_parquet(out / f"{stem}.parquet", index=False)
    json.dump({"name": a.name, "taxonKey": m["usageKey"], "filters": base, "years": [a.y0, a.y1],
               "retrieved_utc": dt.datetime.utcnow().isoformat(timespec="seconds"), "n_records": int(len(df)),
               "per_year": per_year, "datasetKeys": df["datasetKey"].value_counts().to_dict()},
              open(out / f"{stem}_manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
