"""D15: sampling events (graph nodes) and dated effort for the pest replication, from the public GBIF
occurrence-search API (no account).

For one insect order (target group) and a list of countries, per country x year:
  1. effort: exact record counts per GADM level-2 district (API facet GADM_LEVEL_2_GID);
  2. events: a random sample of background records (up to `--per` records per country-year),
     drawn as small pages at uniformly random offsets (seeded), with coordinates, date and district.
     When a country-year exceeds the API paging limit the sample is drawn month by month, in
     proportion to the monthly counts.
The target species itself is excluded from the background (taxonKey filter applied afterwards).
Outputs: data/raw_pests/bg_<order>_<region>_{events,effort}.parquet + manifest JSON.

Usage: python tools/gbif_api_background.py Hemiptera europe --exclude-key 4485843
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gbif_api_target import API, get

REGIONS = {
    "europe": ["AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR",
               "HU", "IE", "IT", "LT", "LU", "LV", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE",
               "SI", "SK", "UA", "MD", "BY", "XK", "LI", "AD", "MC", "SM"],
    "north_america": ["US", "CA"],
}
MAX_OFFSET, PAGE = 100_000, 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("order")
    ap.add_argument("region", choices=list(REGIONS))
    ap.add_argument("--exclude-key", type=int, action="append", default=[])
    ap.add_argument("--per", type=int, default=400, help="sampled events per country-year")
    ap.add_argument("--y0", type=int, default=2005)
    ap.add_argument("--y1", type=int, default=2025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/raw_pests")
    a = ap.parse_args()
    m = get(f"{API}/species/match", {"name": a.order, "rank": "ORDER", "strict": "true"})
    assert m.get("rank") == "ORDER" and m.get("canonicalName") == a.order, m
    base = {"taxonKey": m["usageKey"], "hasCoordinate": "true", "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT"}
    rng = np.random.default_rng(a.seed)
    ev, eff, counts = [], [], {}

    def sample(params, n, k):
        offs = np.unique(rng.integers(0, max(n - PAGE, 0) + 1, size=max(1, int(np.ceil(k / PAGE)))))
        out = []
        for o in offs:
            for r in get(f"{API}/occurrence/search", {**base, **params, "limit": PAGE, "offset": int(o)})["results"]:
                g = r.get("gadm", {})
                out.append({"key": r["key"], "lat": r.get("decimalLatitude"), "lon": r.get("decimalLongitude"),
                            "year": r.get("year"), "month": r.get("month"), "countryCode": r.get("countryCode"),
                            "gid1": g.get("level1", {}).get("gid"), "gid2": g.get("level2", {}).get("gid"),
                            "speciesKey": r.get("speciesKey"), "datasetKey": r.get("datasetKey"),
                            "basisOfRecord": r.get("basisOfRecord")})
        return out

    for cc in REGIONS[a.region]:
        for y in range(a.y0, a.y1 + 1):
            q = {"country": cc, "year": y}
            r0 = get(f"{API}/occurrence/search", {**base, **q, "limit": 0, "facet": "gadmLevel2Gid",
                                                  "facetLimit": 5000})
            n = r0["count"]
            counts[f"{cc}-{y}"] = n
            if n == 0:
                continue
            for f in r0.get("facets", []):
                eff += [{"countryCode": cc, "year": y, "gid2": c["name"], "n": c["count"]} for c in f["counts"]]
            if n <= MAX_OFFSET:
                ev += sample(q, n, min(a.per, n))
            else:
                for mo in range(1, 13):
                    nm = get(f"{API}/occurrence/search", {**base, **q, "month": mo, "limit": 0})["count"]
                    assert nm <= MAX_OFFSET, f"{cc}-{y}-{mo}: {nm}"
                    if nm:
                        ev += sample({**q, "month": mo}, nm, max(1, round(a.per * nm / n)))
            print(f"{a.order} {cc} {y}: {n:,} records; events so far {len(ev):,}", flush=True)
    E = pd.DataFrame(ev).drop_duplicates("key")
    E = E[~E["speciesKey"].isin(a.exclude_key)]
    out = Path(a.out)
    stem = f"bg_{a.order}_{a.region}"
    E.to_parquet(out / f"{stem}_events.parquet", index=False)
    pd.DataFrame(eff).to_parquet(out / f"{stem}_effort.parquet", index=False)
    json.dump({"order": a.order, "taxonKey": m["usageKey"], "region": a.region, "filters": base,
               "exclude_speciesKeys": a.exclude_key, "per_country_year": a.per, "page": PAGE, "seed": a.seed,
               "years": [a.y0, a.y1], "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "n_events": int(len(E)), "counts": counts}, open(out / f"{stem}_manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
