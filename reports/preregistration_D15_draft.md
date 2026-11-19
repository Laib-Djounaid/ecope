# Pre-registration D15 (DRAFT): replication on agricultural pests

Purpose: test whether the D13/D14 findings (neighbour temporal leakage, causal rule, as-of model)
replicate on crop pests with independent data, regions and taxa. Written before any model is fitted
on these data; only the raw downloads and their counts have been inspected (decision log, D15 data).

## 0. Cases and data

| Case | Target (crop system) | Region | Background group (nodes, effort) |
|---|---|---|---|
| P1 | *Halyomorpha halys* (orchards, vegetables) | Europe (42 countries) | Hemiptera without the target |
| P2 | *Halyomorpha halys* | USA + Canada | Hemiptera without the target |
| P3 | *Popillia japonica* (turf, fruit, vines, soy/maize) | USA + Canada | Coleoptera without the target |

*Spodoptera frugiperda* was dropped before any analysis (too few records in Africa, see log).
Source: public GBIF occurrence API (filters: coordinates, no geospatial issue, status PRESENT);
manifests in `data/raw_pests/*_manifest.json`; cited as GBIF derived datasets.

- **Nodes** (sampling events, the analogue of sPlotOpen plots): background records sampled at random
  per country-year (seed 0), years t_i in [2008, 2025 - N]. To keep the compute budget, a fixed random
  subsample of at most **30,000 nodes per case** (seed 0) is used.
- **Label** y_i^N = 1 if a target record lies within 50 km of node i in years (t_i, t_i + N].
  **Primary horizon N = 3** (invasions are fast, and it allows t_i up to 2022); N = 5 secondary.
- **Node features (dated, year < t_i):** prior target count within 50 km (log1p), prior_any, number
  of years with a prior target record; dated effort = background records of the node's GADM level-2
  district, years 2000 .. t_i - 1 (log1p, exact API counts).
- **Context (static):** WorldClim 2.1 bio1, bio5, bio6, bio12 (10 arc-min) and latitude/longitude.
- **Splits:** ~500 km spatial blocks; sealed hold-out = 20% of blocks (stratified on N = 3 block
  prevalence, seed 0), SHA-256 written to a manifest before any modelling and never loaded during
  development; 5 spatial folds on the development blocks; code of `02_splits` / `02b_holdout_splits`.

## 1. Graph and causal rule

Geographic kNN graph (haversine, k = 10, symmetrised, w = 1/(1 + d/50 km)), as in D13. Causal rule
t_j <= t_i (D13 section 1) and as-of features (D14) unchanged. D1 and D2 are reported for each case
on development data and, before prediction, on the hold-out.

## 2. Models (reduced set and budget)

G_lgbm, G_lgbm_nbr (causal), G_lgbm_nbr_leaky, EcoPE-GAT_leaky, EcoPE-GAT_causal, EcoPE-GAT_asof,
identical code to D13/D14 (GAT_noctx_causal omitted for budget).
**Tuning (reduced, fixed now):** one Optuna study per (model, case), 20 trials, on the blocked inner
validation of the whole development set at N = 3; the hyperparameters are reused for the spatial
folds and the hold-out refit. 3 seeds (0-2); seed-mean score. Same search spaces as D13/D14.

## 3. Hypotheses (hold-out, setting S: train on all development, test on the hold-out)

Statistics as in D13: PR-AUC of the seed-mean score, 2,000 block-bootstrap replicates (seed 0),
replicates without positives dropped and counted; supported = CI lower bound > 0. Each hypothesis is
tested per case; **replication** = supported in at least 2 of the 3 cases (stated in advance).

- **R1 (leakage):** EcoPE-GAT_leaky - EcoPE-GAT_causal > 0.
- **R1b (rank reversal):** EcoPE-GAT_leaky - G_lgbm_nbr CI lower bound > 0 and EcoPE-GAT_causal -
  G_lgbm_nbr CI lower bound <= 0.
- **R5 (scope):** G_lgbm_nbr_leaky - G_lgbm_nbr > 0.
- **R8 (as-of model):** EcoPE-GAT_asof - G_lgbm_nbr, superiority, reported as is.
- **R9:** EcoPE-GAT_asof - EcoPE-GAT_causal > 0.
- **R7 (emergence):** on hold-out nodes with no prior target record within 50 km, the final model
  (rule of D14 section 3, applied per case on development CV) minus the persistence heuristic
  (log1p target records within 100 km before t_i) > 0.
Development CV results and N = 5 are descriptive. The transfer setting (T) is not used here.

## 4. Order of work

Build data and hold-out manifest -> tag `prereg-D15` (this file frozen) -> tuning and development
CV -> tag `frozen-D15` -> one hold-out run for the three cases, together with D13/D14.
