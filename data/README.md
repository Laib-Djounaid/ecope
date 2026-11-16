# Data

Raw data are not versioned. Place them at the paths given in `config.yaml` (`paths:`), or edit those paths.

| Source | File(s) | Where to get it |
|---|---|---|
| sPlotOpen v2.0 (Sabatini et al., 2021) | `sPlotOpen_header.txt`, `sPlotOpen_CWM_CWV.txt` | https://doi.org/10.25829/idiv.3474-bb7k72 |
| GBIF occurrence download 0046318-260806074905277 | `0046318-260806074905277.csv` (simple CSV, 22,129 records) | https://doi.org/10.15468/dl.ps65sp |

Then run `python src/01_build_dataset.py` and `python src/01b_data_report.py` (~30 s on CPU).
Outputs: `data/processed/plots.parquet` (one row per plot; see the column naming below), filtered GBIF records per variant, and `gbif_filter_counts.csv`.

Column naming in `plots.parquet` (variant = `unc-{keep|drop}_bor-{all|no-ms}`, b = buffer km):
- `eligible_N{N}`: survey window (t, t+N] lies inside 2000-2025
- `y_N{N}_b{b}_{variant}`: label (>= 1 record with 0 < year - t <= N within b km)
- `n_prior_b{b}_{variant}` / `prior_any_...` / `nsp_prior_...`: records / any record / distinct species strictly before t
- `n_same_b{b}_{variant}`: records in the survey year itself (used neither as label nor prior)
- `n_all_b{b}_{variant}`: all records, any year (LEAKY: contains the label records; diagnostic only)
- `n_outwin_N{N}_b{b}_{variant}`: all records outside the label window (uses future years; diagnostic only)
