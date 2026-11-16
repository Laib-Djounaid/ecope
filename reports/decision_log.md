# Decision log

One entry per decision: what was decided, why (with the numbers that motivated it), what it
changes, and how to go back. Newest entries at the bottom. Git tags mark decision points:
`git checkout <tag>` restores the code and config of that moment.

---

## D1 - Main GBIF filtering variant (2026-11-16, user)
- **Decision:** missing coordinate uncertainty KEPT; basisOfRecord MATERIAL_SAMPLE EXCLUDED
  (variant `unc-keep_bor-no-ms`).
- **Why:** 71% of raw records lack an uncertainty value (dropping them would remove most data).
  54% of raw records are MATERIAL_SAMPLE, mainly eDNA metabarcoding (Australian Microbiome 18S,
  INSDC sequences, Global soil organisms, airborne spores, marine eDNA), i.e. DNA detections,
  not reports on plants.
- **Effect:** 8,276 filtered records; prevalence 5.0 / 6.7 / 8.5 % at N = 5 / 7 / 10 (vs
  6.7 / 9.7 / 12.9 % with eDNA kept).
- **Reversible:** yes, all four variants are computed for every plot (`plots.parquet`,
  columns `y_N*_b*_<variant>`); switch with `gbif.basis_of_record_main` /
  `gbif.missing_uncertainty` in `config.yaml`.

## D2 - Horizons and eligibility (2026-11-16, user)
- **Decision:** N = 5, 7, 10; plots eligible if survey_year >= 1999 and survey_year + N <= 2025.
- **Effect:** 44,772 eligible plots, identical for all N (latest sPlotOpen survey is 2015).

## D3 - Tuning protocol under CPU budget (2026-11-16, Claude; user: "do what you think is ok")
- **Decision:** Optuna tuned once per outer fold on the N = 10 inner-validation PR-AUC, best
  parameters reused for N = 5 and 7; 30 trials for every model; GNN settings tuned separately.
  GAT heads split a fixed width (hidden // heads) instead of 4 x 64.
- **Why:** full GAT epoch 5.5 s on the laptop CPU; per-horizon tuning would take months.
- **Reversible:** `evaluation.tuning_horizon`, `graph.search_space` in `config.yaml`; if runs
  move to a GPU (Colab), per-horizon tuning becomes affordable.

## D4 - Trait signal is largely geographic (2026-11-16, finding)
- Fixed RF on traits (set A), spatial CV: pooled ROC 0.79, ROC of the *dataset-mean* score 0.85,
  *within-dataset* ROC 0.62; traits predict continent with 72% balanced accuracy (chance 17%).
- **Consequence:** location/dataset confounding must be controlled before any claim that traits
  carry ecological signal. Script: to be moved into `src/05_analyses.py`.

## D5 - Full spatial run paused (2026-11-16, user)
- **Decision:** pause the 30 h Optuna run; first check the effort variable for leakage, add
  feature sets D and E, run a default-hyperparameter pilot, then decide the paper's framing.
- **State at pause:** 5 / 65 tabular jobs done (MLP, old feature set C without effort) in
  `results/tabular/spatial/`. **Superseded** (feature set C changed); not to be reused.

## D6 - Effort variable: leakage audit (2026-11-16)
- Effort was never a model feature. Three diagnostics existed:
  `n_all` (27 species, any year: leaks the label), `n_outwin` (27 species, all years except the
  label window, uses the future: the "ROC 0.88" quoted in the data report and draft manuscript
  is this diagnostic and must not be presented as a baseline), `n_prior` (27 species before the
  survey = identical to prior presence).
- **Decision:** build an independent target-group effort layer: GBIF Fungi records excluding
  the 20 fungal target species, 0.5-degree cells, years before the survey only for features.
- **Route 1 (abandoned):** GBIF occurrence-search API (`src/01c_effort_layer.py`), stopped after
  rate limiting (HTTP 429; 171 queries in 31 min). Partial cache kept in `data/external/`.
- **Route 2 (current):** GBIF SQL download (`data/gbif_effort_sql/`), citable with a DOI.
  First submission 0010697-261110092616502 (DOI 10.15468/dl.98g8ed) returned **0 rows**
  (a filter column behaves differently in the SQL table); diagnostic download
  0010758-261110092616502 submitted to identify the column. The empty download is not used
  and must not be cited.

## D7 - New feature sets (2026-11-16, user + Claude)
- D = prior presence + effort (no-ecology baseline); E = D + traits; C = B + prior + effort.
- G = D + context (climate PCs, lat/lon, biome), no traits (proposed by Claude, accepted):
  C vs G tests whether traits add anything beyond effort, prior presence AND location.

## Pilot v1 - without the independent effort layer (2026-11-16)
- Files: `reports/pilot_runs/v1_no_indep_effort/` (per-fold metrics, paired tests, out-of-fold
  predictions, log). N = 5, spatial 5-fold CV, 1 seed, default hyperparameters.
  Sets C0/D0/E0 = C/D/E without the independent effort layer.
- Positives per test fold: 429, 576, 413, 387, 426 (prevalence 4.3-6.4 %).
- PR-AUC (mean over folds): prior count score 0.301; D0 0.300 (RF) / 0.302 (LGBM);
  E0 0.350 / 0.371; C0 0.417 / 0.421; A 0.154 / 0.168; plain GAT 0.144; EcoPE-GAT 0.191.
- E0 - D0: LGBM +0.069 PR-AUC (5/5 folds, block-bootstrap 95% CI [+0.001, +0.117]);
  RF +0.051 (4/5, CI [-0.006, +0.100]); ROC-AUC +0.11 for both (5/5).
- EcoPE-GAT > plain GAT in 5/5 folds, but EcoPE receives location that the plain GAT lacks;
  the proper control (GAT + context concatenated) was not in this pilot.
- **Provisional reading:** traits add a modest, consistent gain over prior presence; context
  adds more; part of the trait gain is probably location in disguise. Decision deferred to
  pilot v2 (with independent effort and set G).

## D8 - GBIF SQL effort query corrected (2026-11-16)
- **Diagnosis** (download 0010758-261110092616502, Luxembourg 2020, values of the filter columns):
  in the GBIF SQL table `kingdomKey` uses the new letter keys (Fungi = `F`, not 5) and
  `occurrenceStatus` uses `DETECTED` / `NOT_DETECTED` (not `PRESENT`). Each of these two filters
  alone removed every row of download v1 (0010697-261110092616502, empty, DOI not to be cited).
  The old integer `speciesKey` values of the label download do not match the new taxonomy either.
- **Fix (query v2, `data/gbif_effort_sql/query_v2.sql`):** filter on `kingdom = 'Fungi'` and
  `occurrenceStatus IN ('DETECTED', 'PRESENT')`; instead of excluding target species by key,
  output an `is_target` flag from the 20 fungal target species NAMES, and subtract target counts
  ourselves (also shows which names matched; synonyms under the new taxonomy may not match and
  will be reported).
- Download v2: 0010811-261110092616502.

## Independent effort layer built (2026-11-16)
- GBIF SQL download v2 0010811-261110092616502, **DOI 10.15468/dl.4739rd** (cite this one):
  193,814 rows (0.5-degree cell x year x is_target); 24.5 M non-target Fungi occurrences in
  26,369 cells; 8,159 target-species occurrences matched by name. Script `src/01d_effort_from_sql.py`.
- Validation (`tools/check_effort_cells.py`) against the independent API counts on 1,972 cells:
  Spearman 0.994, median SQL/API ratio 0.999 (half-cell shifts drop to ~0.68: decoding correct);
  name-based target matching recovers ~95% of target records (218 vs 229).

## Pilot v2 - with independent effort and set G (2026-11-16)
- Files: `reports/pilot_runs/v2_indep_effort/`. N = 5, spatial 5-fold CV, 1 seed, defaults.
- Effort-only score (non-target Fungi before survey, no model): PR-AUC 0.342, ROC 0.775
  (prior count 0.301 / 0.722; all-years diagnostic 0.415 / 0.861).
- PR-AUC LGBM / RF: A 0.168 / 0.154; D 0.361 / 0.241 (RF default weak on 4 features);
  E 0.388 / 0.373; C 0.407 / 0.429; **G 0.421 / 0.450** (best). GATs (v1): 0.144 / 0.191.
- **C - G (traits beyond effort + prior + location): LGBM -0.013 (1/5 folds, CI [-0.088, +0.018]);
  RF -0.021 (2/5, CI [-0.028, +0.030]); ROC change ~0.** Traits add nothing once effort, prior
  presence and location are known. E - D (LGBM) +0.027, CI [-0.022, +0.080].
- **Reading:** the trait gain in pilot v1 was location in disguise. Labels are explained by
  surveillance effort, prior presence and geography. Recommendation (Claude): evaluation-paper
  framing with EcoPE-GAT as a case study. Awaiting user decision (D9).
- Caveats: single horizon, single seed, default hyperparameters, high fold variance.
