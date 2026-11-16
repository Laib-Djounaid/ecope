# Independent sampling-effort layer: GBIF SQL download

Citable replacement for the API counts of `src/01c_effort_layer.py`.

**What it returns:** number of GBIF occurrences of kingdom Fungi (taxonKey 5) per 0.5-degree
cell (`GBIF_DMSGCode(1800, ...)`, lower-left corner, no randomisation) and year, 2000-2025,
with occurrenceStatus PRESENT, coordinates, no geospatial issues, basisOfRecord not
MATERIAL_SAMPLE / FOSSIL_SPECIMEN, and **excluding the 20 fungal target species** of the label
download (10.15468/dl.ps65sp), so the effort group is independent of the label species.

- `query.sql`: the query
- `query.json`: the request body (validated by GBIF on 2026-11-16: HTTP 201)

## Submit (needs your GBIF.org username and password; never commit them)

```bash
curl --include --user YOUR_GBIF_USERNAME:YOUR_PASSWORD \
     --header "Content-Type: application/json" \
     --data @data/gbif_effort_sql/query.json \
     https://api.gbif.org/v1/occurrence/download/request
```

The last line of the response is the download key. Check status (until `SUCCEEDED`; the DOI
appears there):

```bash
curl -Ss https://api.gbif.org/v1/occurrence/download/<KEY>
```

Download: `https://api.gbif.org/v1/occurrence/download/request/<KEY>.zip`, then unzip it into
`data/gbif_effort_sql/` and tell me the key and DOI. GBIF SQL downloads are an experimental
service; processing time is not documented (typically hours).

Cite as: GBIF.org (date) GBIF Occurrence Download https://doi.org/<DOI>
