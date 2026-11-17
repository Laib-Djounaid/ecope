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

## D9 - EcoPE-GAT stays central; pre-registered test of EcoPE-GAT v2 (2026-11-16, user)
**Pre-registration. Written and committed (tag `prereg-ecope-v2`) BEFORE the pilot was run.**

- **Model (EcoPE-GAT v2):** EcoPE inputs = SoilClim PC1/PC2, lat/lon, biome one-hot, independent
  effort (non-target Fungi records within 50 km BEFORE the survey year, log1p; DOI
  10.15468/dl.4739rd). Node features = prior presence (P) or prior presence + traits (PT).
  Graphs (transductive, k = 10): trait kNN (v1 reference), geo kNN (haversine), combined (union),
  edge attributes [w_trait, w_geo].
- **Primary variant (user choice): PT + combined graph** (`ecope_v2_PT_combined`). The verdicts
  below apply to it alone. The other 5 variants (P/PT x trait/geo/combined, minus primary) are
  secondary ablations, reported in full but without a verdict.
- **Controls on the same folds:** GAT with the same variables concatenated, no encoder
  (`gat_concat_<nodes>_<graph>`); LightGBM on feature set G (`G_lgbm`, same code path as pilot v2).
- **Protocol:** N = 5; the same spatial 5-fold splits as pilot v2; seed 0; default
  hyperparameters of the original paper (hidden 64, 4 heads, dropout 0.3, lr 1e-3, wd 1e-4,
  balanced positive weight, max 200 epochs, early stopping patience 20 on inner-val PR-AUC);
  training on the training part of each fold. Script `src/pilot_ecope_v2.py`.
- **Statistic:** difference in pooled out-of-fold PR-AUC (all 5 test folds), 95% percentile CI
  from 2,000 bootstrap resamples of the spatial blocks (seed 0).
- **Criteria (fixed in advance):**
  - EcoPE-GAT v2 is **competitive** if the CI lower bound of (v2 - G_lgbm) > -0.02
    (non-inferiority), **superior** if the lower bound > 0.
  - EcoPE **adds** if (v2 - GAT+concat, same nodes and graph) has a CI lower bound > 0 (same rule).
- **Consequence (fixed in advance):** competitive -> method paper centred on EcoPE-GAT v2 (v1 as
  ablation); otherwise -> evaluation paper with EcoPE-GAT v1/v2 as case studies. In both cases:
  full tuning, 3 horizons, 5 seeds, all variants reported.
- **Transparency note:** a smoke test (fold 0 only, 4 epochs, 4 configurations) was run to check the
  code before this entry; its outputs were deleted and are not results. It confirmed that G_lgbm
  reproduces pilot v2 exactly on fold 0 (PR-AUC 0.287).

## Pilot v3 - pre-registered EcoPE-GAT v2 test: RESULT (2026-11-17)
- Files: `reports/pilot_runs/v3_ecope_v2/` (per-fold metrics, summary, `preregistered_tests.csv`,
  out-of-fold predictions, per-part checkpoints, log). Run 22:32 -> 13:20 (laptop slept 01:27-11:50;
  resumed from checkpoints, no part recomputed).
- Positives per test fold: 429, 576, 413, 387, 426.
- **Primary (`ecope_v2_PT_combined`) vs G_lgbm: pooled PR-AUC 0.279 vs 0.440, diff -0.162,
  95% CI [-0.304, +0.017]. Lower bound < -0.02 -> NOT competitive (non-inferiority not shown).**
- **EcoPE adds? primary vs GAT+concat (same nodes, graph): diff -0.078, CI [-0.223, +0.017] -> no.**
- **Pre-registered consequence: evaluation paper, with EcoPE-GAT v1/v2 as case studies.**
- Secondary variants (no verdict, as pre-registered): trait graphs are worst (pooled 0.23-0.24 for
  all four); geo graphs best; `gat_concat_P_geo` (no EcoPE, prior-only nodes, geo graph) has the
  highest pooled PR-AUC 0.465 (vs G 0.440; diff +0.024, CI [-0.065, +0.122]). EcoPE never beats
  concatenation; for P + geo concatenation is better with CI excluding 0 (diff -0.144,
  CI [-0.253, -0.018]).
- **Observation for the full run (not a change of this verdict):** fold-mean PR-AUC differs
  markedly from pooled PR-AUC for the GNNs (e.g. `ecope_v2_P_geo` fold-mean 0.456 vs pooled 0.320;
  G_lgbm 0.421 vs 0.440). Pooling out-of-fold scores penalises models whose score scale varies
  between folds (GNNs trained with class weights and variable early stopping). Fold-mean of the
  primary: 0.380 vs G 0.421 (per-fold diffs +0.078, -0.146, -0.005, -0.222, +0.095).
  Any alternative statistic for the full run must be specified BEFORE that run.

## D10 - Method contribution: occupancy-detection GNN; sealed confirmatory hold-out (2026-11-17, user)
- **User decision:** no evaluation paper; develop a methodological contribution, a GNN that
  separates presence from detection (EcoPE-OD-GNN). **This overrides the consequence that D9 fixed
  in advance** (pilot v3: EcoPE-GAT v2 not competitive). Disclosure required in the paper: EcoPE-GAT
  v1/v2 failed their pre-registered tests, and the OD model was designed after seeing pilots v1-v3.
- **Step 0, sealed confirmatory hold-out** (`src/02b_holdout_splits.py`, created BEFORE any OD
  model code was run): random spatial blocks, stratified by continent x tercile of block-median
  log1p(independent prior effort), seed 20261117, target 15% of plots per stratum.
  Result: 7,920 plots (17.7%; large blocks cannot be split), 92 blocks, all 6 continents;
  positives N5/N7/N10 = 355/483/562. SHA-256 of sorted plot ids:
  31e674c06b13b0ac4b5afb0fc3f0c66af44c9e7e3ab05d464cbe4c4ff3cd3a7f (`holdout_manifest.csv`).
  Chosen over a recent-period hold-out because sPlotOpen ends in 2015 and a 2011-2015 test would
  be confounded with GBIF growth after 2012 and with the prior-presence look-back length.
  **Caveat:** pilots v1-v3 already used these plots (exploratory); the hold-out is untouched only
  with respect to the OD model, its pilot and its tuning.
  `features.load_modelling_table` drops the hold-out unless `include_holdout=True` (to be passed
  ONCE, by the final confirmatory evaluation).
- **Development splits (splits_v2):** 36,852 plots, 178 blocks, spatial 5-fold CV with blocked
  inner validation. **Effort-transfer split:** development blocks with block-median log effort
  above the development median (= 0) train (87 blocks, 1,758 positives at N5); the others are the
  transfer test (91 blocks, median effort 0, 118 positives at N5, 281 at N10).

## D11 - Pre-registration of the EcoPE-OD-GNN pilot (2026-11-17)
**Written and committed (tag `prereg-od`) BEFORE the pilot was run.** Literature and
identifiability rationale: `reports/literature_od.md`.

- **Data:** development set only (splits_v2; the sealed hold-out is never loaded). N = 5, seed 0,
  default hyperparameters (hidden 64, 4 heads, dropout 0.3, lr 1e-3, wd 1e-4, max 200 epochs,
  early stopping patience 20 on inner-validation PR-AUC). Geo kNN graph (haversine, k = 10),
  transductive. Script `src/pilot_od.py`.
- **Primary model `od_cloglog` (user choices, AskUserQuestion 2026-11-17):**
  presence = EcoPE-GAT on the geo graph; node features = **effort-normalised prior presence**
  (target-group ratio: prior target records / (prior target + prior non-target fungal + 1), plus
  an any-prior flag) + traits + plot attributes; EcoPE context = SoilClim PC1/PC2, lat/lon, biome
  one-hot (**no effort**). Detection = **monotone cloglog** in standardised log1p(independent prior
  effort) only: p_det = 1 - exp(-exp(a + softplus(b) z)). P(reported) = P(present) x p_det;
  unweighted BCE (likelihood). No country/dataset in detection.
- **Identifiability (stated limits):** single label per plot -> separation relies on covariate
  separation (effort only in detection; no effort, and effort-normalised prior, in presence) and
  on the monotone, saturating detection form (Lele et al. 2012; Knape & Korner-Nievergelt 2015).
  P(present) is a model-based quantity, not observed occurrence.
- **References / variants on the same splits:** `G_lgbm` (feature set G, raw prior);
  `gat_concat_P_geo` (best pilot-v3 model; raw prior + context incl. effort, concatenated,
  class-weighted as in v3); `od_concat` (primary without the EcoPE encoder); `od_nodet` (presence
  branch only, no effort anywhere); `od_mlp` (detection = free MLP). Secondary run with **raw**
  prior presence in the OD presence branch (`--prior-mode raw`): reported, no verdict.
- **Statistics and criteria (fixed in advance; CI = 95% percentile, 2,000 block-bootstrap reps,
  seed 0):**
  - **(a) spatial CV**, statistic = **mean over the 5 folds of PR-AUC** (blocks resampled within
    each test fold). `od_cloglog` vs G_lgbm and vs gat_concat_P_geo: competitive if the CI lower
    bound > -0.02, superior if > 0. Component tests (same rule, lower bound > 0 = "adds"):
    vs od_concat (EcoPE adds), vs od_nodet (detection branch adds), vs od_mlp (constraint adds).
  - **(b) effort transfer (MAIN test of the contribution)**: train on high-effort blocks, test on
    low-effort blocks (91 blocks, 118 positives at N5 -> low power, stated in advance); statistic =
    PR-AUC on the transfer test set (blocks resampled). **Contribution supported if `od_cloglog` -
    G_lgbm AND `od_cloglog` - gat_concat_P_geo both have CI lower bound > 0.** Component
    comparisons as in (a), reported without verdict.
  - **(c) effort dependence**: Spearman rho with log effort on development out-of-fold predictions.
    Supported if |rho(P(present) of od_cloglog)| - |rho(ref)| has CI upper bound < 0 for ref =
    G_lgbm score and ref = gat_concat_P_geo score. Also reported vs G_lgbm with effort set to the
    training median (effort-standardised, Warton et al. 2013), no verdict. Risk maps of P(present)
    and of naive scores are produced from the same out-of-fold predictions.
- **Decision rule for the full run:** whatever the outcome, the full tuning (3 horizons,
  5 seeds) reports all variants; the claim "EcoPE-OD-GNN improves transfer to low-effort regions"
  requires (b), and "its presence output is less effort-dependent" requires (c). A final single
  evaluation on the sealed hold-out decides the confirmatory claims.
- **Transparency note:** two smoke tests were run before this entry to check the code (fold 0 with
  3 epochs; all settings with 1 epoch, incl. the test computations). Their outputs were deleted and
  are not results.
