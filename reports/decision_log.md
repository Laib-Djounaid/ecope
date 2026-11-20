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

## Pilot v4 (EcoPE-OD-GNN, pre-registered in D11) - primary run `v4_od_primary` (2026-11-17)

Run as pre-registered (N = 5, seed 0, defaults, `--prior-mode ratio`), finished about 16:30 after one
PC sleep (resumed from per-part files). Outputs in `reports/pilot_runs/v4_od_primary/`.

| setting | G_lgbm | gat_concat_P_geo | od_cloglog (primary) | od_concat | od_mlp | od_nodet |
|---|---|---|---|---|---|---|
| CV fold-mean PR-AUC | **0.467** | 0.323 | 0.351 | 0.358 | 0.367 | 0.341 |
| Transfer PR-AUC (chance 0.006) | 0.040 | **0.189** | 0.014 | 0.019 | 0.006 | 0.007 |
| Transfer ROC-AUC | 0.788 | 0.781 | 0.558 | 0.699 | 0.389 | 0.508 |

Pre-registered verdicts:
- **(a) spatial CV: FAILED.** od_cloglog - G_lgbm = -0.116 [-0.134, +0.047] (not competitive);
  vs gat_concat_P_geo +0.028 [-0.058, +0.078] (not shown competitive). EcoPE vs concat -0.008
  [-0.058, +0.056], detection branch vs none +0.010 [-0.048, +0.104], cloglog vs MLP -0.016
  [-0.110, +0.057]: no component shown to add.
- **(b) effort transfer (main test): FAILED.** od_cloglog - G_lgbm = -0.027 [-0.145, +0.018];
  vs gat_concat_P_geo -0.176 [-0.410, +0.017]. The OD models are near chance on low-effort blocks
  (median log effort there = 0, so the detection branch is almost constant and ranking comes from
  the presence branch alone). gat_concat_P_geo transfers best.
- **(c) effort dependence: SUPPORTED as pre-registered.** |rho| of P(present) with log effort 0.094
  vs 0.429 (G_lgbm), 0.344 (gat_concat), 0.369 (G_lgbm effort-standardised); differences -0.335
  [-0.472, -0.122], -0.250 [-0.405, -0.063], -0.274 [-0.415, -0.052]. Caveat: the presence branch
  receives no effort input by construction and an effort-normalised prior, so low |rho| is partly
  by design and does not show the presence output is a better estimate of occurrence; given (b),
  the presence output does not rank low-effort plots well.

Consequence under D11: the claim "improves transfer to low-effort regions" is not supported; the
claim "presence output less effort-dependent" passes its pilot criterion with the caveat above.
No tuning started. Secondary run (`--prior-mode raw`, no verdict) in progress.

### Pilot v4 - secondary run `v4_od_rawprior` (raw prior in the OD presence branch; no verdict)

| setting | G_lgbm | gat_concat_P_geo | od_cloglog | od_concat | od_mlp | od_nodet |
|---|---|---|---|---|---|---|
| CV fold-mean PR-AUC | **0.467** | 0.323 | 0.357 | 0.369 | 0.378 | 0.386 |
| Transfer PR-AUC | 0.040 | **0.189** | 0.010 | 0.028 | 0.006 | 0.006 |
| Transfer ROC-AUC | 0.788 | 0.781 | 0.557 | 0.772 | 0.447 | 0.504 |

Same pattern as the primary run: od_cloglog - G_lgbm in CV -0.110 [-0.146, +0.067], in transfer
-0.031 [-0.154, +0.008]; vs gat_concat in transfer -0.180 [-0.413, +0.009]; |rho| of P(present)
0.040 (differences vs refs all with CI < 0). The prior encoding does not change the conclusions.

Risk maps (`risk_maps.png` in each run folder, script `src/pilot_od_maps.py`): percentile ranks of
out-of-fold predictions. Visually, all models concentrate high ranks where effort and observed
positives are (Europe, eastern Australia, western USA); the od_cloglog presence map spreads
high ranks to some low-effort areas (S. America, S. Africa, Siberia), which is what lower effort
dependence looks like, but (b) shows those ranks are not validated by the low-effort labels.

Status: STOP before any tuning; awaiting the user's decision.

## D12 - EcoPE-GAT reframed; confirmatory target = transfer to low-effort regions (user, 2026-11-17)

(The user's text calls it "D11"; D11 is already the OD pre-registration, so it is logged as D12.)
EcoPE-GAT = GAT on a geographic graph + ecological position encoding (SoilClim PC1/PC2, coordinates,
biome) before message passing, prior presence as node feature; encoder form (identity/concat,
linear, MLP) and graph (geo, trait, combined) become tested choices; every variant and every
pre-registered failure (v1, v2, OD) is reported. Before pre-registration: B1 add the reference
"G_lgbm + neighbourhood"; B2 exploratory transfer CIs; B3 hold-out counts; E check the repository
for the GBIF password. Pre-registration (H1-H3) to be shown to the user before any tuning.

### D12 checks (2026-11-17, before any pre-registration)

- **E. GBIF password:** the credential string and the GBIF user name occur 0 times in the working
  tree (tracked and untracked files, logs) and 0 times in `git log --all -p` (all branches, tags).
  No purge needed. (Search done by script; the string is never printed or stored.)
- **B3. Hold-out low-effort part** (same rule as the development transfer split: block-median
  log1p prior effort <= development threshold = 0, i.e. blocks with no prior non-target fungal
  record), counts only, no prediction: 3,629 plots, 27 blocks, positives 97 / 134 / 139 at
  N = 5 / 7 / 10 (`reports/holdout_transfer_counts.json`; `tools/holdout_transfer_counts.py`).
  High-effort part: 4,291 plots, 65 blocks. 27 blocks limits the block bootstrap.
- **NEW: temporal leakage through graph neighbours** (`tools/neighbour_time_leak.py`,
  `reports/neighbour_time_leak.json`). A node's prior presence counts target records before ITS
  OWN survey year; a neighbour surveyed later can carry records from the receiving plot's label
  window. On the geo kNN graph (k = 10, symmetrised) used in pilots v3 and v4: 15.2% of edges
  come from a later-surveyed neighbour, 0.5% carry window records; but 30.2% of positive plots vs
  0.5% of negative plots have such a neighbour (P(y=1 | leaky neighbour) = 0.77; the indicator
  alone gives PR-AUC 0.27 at prevalence 0.05, and 0.11 on the transfer test set at prevalence
  0.006). **All GNN results of pilots v3 and v4 that use the geo graph are affected** (the trait
  graph of v1/v3 likewise passes neighbours' priors, not audited yet). Fix: time-causal graph,
  messages j -> i only if survey_year[j] <= survey_year[i] (`graph_models.time_causal`; keeps
  389,780 of 459,584 directed edges). The 04_graph hybrid (cross-fitted neighbour labels) has the
  same problem and must be made time-causal or dropped.
- **B1. G_lgbm + neighbourhood** (`src/neighbour_features.py`): time-safe aggregates over the
  causal geo kNN neighbours (mean log prior count, max prior_any, mean log effort, number of
  causal neighbours) + target records within 100 km and in the 50-100 km ring before the survey
  year. No labels used.
- **B2. Exploratory transfer** (`src/explore_d12.py`, `reports/pilot_runs/x1_d12_transfer/`;
  N = 5, defaults; LightGBM 1 fit, GATs 5 seeds, tests on the seed-mean score; transfer test
  chance 0.006). A single seed proved unstable (seed 0 of the leaky concat GAT stopped at epoch 4,
  PR-AUC 0.006, while v4 seed 0 had given 0.189).

  | model | PR-AUC (seed-mean score) | per-seed range |
  |---|---|---|
  | G_lgbm | 0.040 | - |
  | G_lgbm + neighbourhood | 0.052 | - |
  | gat_concat_P_geo, original (leaky) graph | 0.243 | 0.006 - 0.285 |
  | GAT prior only, leaky graph | 0.189 | 0.168 - 0.190 |
  | gat_concat_P_geo, **causal** graph | **0.011** | 0.006 - 0.032 |
  | GAT prior only, **causal** graph | 0.077 | 0.071 - 0.080 |

  CIs (2,000 block bootstraps): leaky concat - G_lgbm +0.203 [+0.001, +0.382]; - G_lgbm_nbr
  +0.191 [-0.000, +0.351]. **Causal** concat - G_lgbm -0.029 [-0.150, +0.009]; - G_lgbm_nbr
  -0.041 [-0.152, +0.010]; - causal prior-only GAT -0.065 [-0.222, +0.028]. Extra (not asked,
  exploratory): causal prior-only GAT - G_lgbm +0.036 [-0.053, +0.131]; - G_lgbm_nbr +0.024
  [-0.042, +0.099].
  **Reading:** the transfer advantage of the geo GAT seen in pilot v4 was produced by the temporal
  leakage; with a causal graph EcoPE-GAT (best current configuration) is not better than G_lgbm or
  G_lgbm + neighbourhood in transfer, and adding the ecological context makes it worse than the
  prior-only GAT. The premise of H1/H2 is not supported by the exploratory data.

Status: pre-registration NOT written; no tuning. Awaiting the user's decision.

## D13 - option 2: paper centred on temporal leakage through neighbours (user, 2026-11-17)

The paper is centred on the methodological finding (temporal leakage through graph neighbours in
graph SDMs), with EcoPE-GAT (v1, v2, OD, geo GAT) as the central case study; every variant and
pre-registered failure is reported. Hypotheses requested: H1 leaky > causal on the sealed
hold-out in transfer and in spatial evaluation; H2 EcoPE-GAT_causal vs G_lgbm_nbr (reported as
is); H3 EcoPE-GAT_causal vs no-context GAT (as is); H4 sensitivity / dose-response. Formal
causal rule + diagnostic. Full tuning only for the H1-H3 models.
Prepared (not frozen): `reports/preregistration_D13_DRAFT.md`, `reports/literature_leakage.md`
(references verified via Crossref / arXiv API), `reports/paper_plan_D13.md`. Awaiting approval
of the pre-registration before any tuning.

### D13 pre-registration approved and frozen (2026-11-17)

The user approved the draft with: (1) effort as a dated node feature; (2) tuning on N = 5,
hyperparameters reused for N = 7 and 10; (3) neighbour-label hybrid removed from the confirmatory
analysis (illustration only, leaky vs strict t_j + N <= t_i). Added: H1b (rank reversal), H5
(scope: leaky vs causal neighbourhood features in LightGBM; framing widened if supported), and the
cost of the causal rule (15.2% edges removed, 232 isolated nodes = 0.6%, sensitivity excluding
isolated nodes). Frozen text: `reports/preregistration_D13.md`, tag `prereg-D13`.

## D14 (2026-11-18, draft): applied framing (B) + new causal model (C)
User asked for both options: applied paper on plant-pathogen emergence forecasting, plus a new
causal model. Draft addendum `reports/preregistration_D14_draft.md` (H6-H9, model EcoPE-GAT_asof,
final-model selection rule). D13 H1-H5 unchanged. Not frozen; awaiting user approval. Hold-out
never loaded. Development D13 numbers seen so far (cv0-cv3, N = 5) motivated C, which is why C
is a new pre-registered hypothesis tested only on the sealed hold-out.

## D14 frozen (2026-11-18)
User approved the D14 draft ("vas-y, garde trace de tout"). Implemented `src/asof_features.py`,
`graph_models.AsOfGAT`, `src/11_d14.py`, config `d14`. Smoke test passed (self-checks, 0 leaking
edges in 20,000 sampled; smoke outputs deleted). Draft renamed to `reports/preregistration_D14.md`,
frozen at tag `prereg-D14`. D14 tuning queued after the D13 GNN run (log `logs/d14_asof.log`).

## Autonomy while the user is away (2026-11-18)
User: if results are not publishable after D13/D14, try new variants without waiting. Rule adopted
by Claude to keep the hold-out valid: publishability is judged on DEVELOPMENT results; any new
variant is explored on development data only, pre-registered (D15, ...) and tagged before the single
hold-out run; every autonomous decision is logged here as "made by Claude, user absent".

## D13 tuning done; H4b; dev summary scripts (2026-11-18/19, Claude, user absent)
D13 GNN tuning finished 2026-11-18 23:44 (42 parameter files). D14 tuning started right after
(as-of leak check N=5/7/10: 0 of 20,000 sampled edges). Added `src/leakage_diagnostic.py` (D1, D2),
`src/12_dev_summary.py` (development seed-mean PR-AUC + block-bootstrap differences, outputs in
`reports/dev_results/`), `src/13_h4.py`. H4b (development, pre-registered): Spearman 0.374,
95% CI [0.204, 0.504], 1,876 positives, 37 blocks -> supported. H4a and H4c queued after D14
(log `logs/h4.log`); H4a q=0 reuses the D13 EcoPE-GAT_leaky refits (same graph, params, seeds).

## Dev summary, first run (2026-11-19 00:15, Claude, user absent)
`reports/dev_results/` (seed-mean score). N = 5 fold-mean: G_lgbm_nbr_leaky 0.503, EcoPE-GAT_leaky 0.478,
G_lgbm_nbr 0.456, G_lgbm 0.410, EcoPE-GAT_causal 0.364, GAT_noctx_causal 0.360. Dev transfer:
EcoPE-GAT_leaky 0.257, G_lgbm_nbr_leaky 0.129, G_lgbm 0.075, G_lgbm_nbr 0.057, GAT_noctx 0.050,
EcoPE-GAT_causal 0.050 (prevalence 0.006). Block-bootstrap CIs are wide and skewed (positives in
only 37 of 178 blocks). Implementation detail fixed now, BEFORE the hold-out: a bootstrap replicate
whose resampled test set has no positive is dropped and the number of valid replicates is reported
(the pre-registration did not specify this case; it produced NaN in one development CI). The same
rule will be used in `src/10_holdout_d13.py`.

## D15 (2026-11-19, draft direction): agricultural pest replication
User: crop pests are acceptable for the thesis; left the choice of pests to Claude ("real contribution").
Decision (Claude): replicate on three pests, two continents, three crop systems: Halyomorpha halys
(Europe, orchards/vegetables), Spodoptera frugiperda (Africa, maize), Popillia japonica (Europe).
Same questions as D13/D14 (neighbour leakage, causal rule, EcoPE-GAT_asof vs tabular). Target records
from the public GBIF occurrence API (`tools/gbif_api_target.py`, manifest per species, to be cited as a
GBIF derived dataset). Nodes (sampling events) = target-group background records (same insect order)
aggregated to cell-year events; this needs a GBIF SQL download with the user account (credentials
read from a local file outside the repository, never printed or committed). Full D15 pre-registration
(regions, nodes, horizons, splits, sealed hold-out) to be written and shown BEFORE any modelling.

## D15 data (2026-11-19, Claude, user absent for this choice)
Target downloads (GBIF API, 2000-2025, no geospatial issue): Halyomorpha halys 150,578 records,
Popillia japonica 80,302, Spodoptera frugiperda 9,943. **S. frugiperda dropped**: only ~260 records in
Africa (KE 158, CM 64, ZA 41), its region of emergence; elsewhere native (Americas). Cases kept:
H. halys Europe, H. halys North America, P. japonica North America (European mainland too sparse:
IT 1,468, CH 291). Node design: random sample of target-group background records (Hemiptera /
Coleoptera) as sampling events (Europe 150, North America 1,500 per country-year, seed 0); dated
effort = exact GADM level-2 district counts per year (API facet). No GBIF account needed.
`tools/gbif_api_background.py`.

## H4 bug fixed (2026-11-19, Claude)
First H4a run failed: tag `h4a_q0.25` made `Path.with_suffix` collapse every refit onto one file
(`h4a_q0`), so refits were overwritten/reused (crash on length mismatch; no result produced or used).
Fixed: tag `h4a_q025` etc.; `checkpoint.cached_refit` now asserts that names contain no dot. All D13/D14
refit names checked: none contains a dot (1,260 files match the expected pattern). H4 a/c restarted.

## Dev summary with EcoPE-GAT_asof (2026-11-19 ~14:30)
N = 5 fold-mean: asof 0.419 (G_lgbm_nbr 0.456, EcoPE-GAT_causal 0.364); dev transfer: asof 0.091, best of
the causal models (G_lgbm 0.075, G_lgbm_nbr 0.058, causal GAT 0.050). Diff asof - G_lgbm_nbr: CV -0.037
[-0.074, +0.074], transfer +0.034 [-0.037, +0.165]. Per the D14 selection rule (development CV) the final
model would be G_lgbm_nbr. Note: the block-bootstrap percentile CIs are skewed (point estimate close
to the upper bound in several comparisons); kept as pre-registered, to be discussed in the paper.

## 2026-11-19 — H4a result; D15 downloader made resumable; P1 tables

- H4a (dev, frozen D13 protocol): fold-mean PR-AUC by q = 0/.25/.5/.75/1: 0.478, 0.451, 0.470, 0.451, 0.373;
  slope -0.084, 95% block-bootstrap CI [-0.107, 0.069]; **not supported** (reported as is). The drop is
  concentrated at q = 1 (removing all non-causal edges), consistent with H4b but not a monotone dose-response.
- User said "arrette" mid-turn, then "reprends tout". The North-American background download had failed
  meanwhile (GBIF API: 6 failed retries at Hemiptera CA 2020; nothing saved). Fix: `tools/gbif_api_background.py`
  now checkpoints after every country-year (partial results + RNG state, so the sample equals an
  uninterrupted run), 12 retries with waits capped at 120 s; relaunched with identical arguments
  (--per 1500). Europe (P1) download unaffected.
- `src/p01_build_pests.py`: P1 built (30,000 nodes; dev 24,270 / hold-out 5,730; hold-out SHA-256
  591038eb…62975f3). Dev prevalence y3 = 0.376 (high: H. halys spread across W. Europe 2016-2025; PR-AUC
  baselines will be high, so differences and the emergence subset R7 carry the information).
  Note vs draft: effort counts start in 2005 (first downloaded year), not 2000; draft corrected before freeze.

## 2026-11-19 — D15 frozen (tag prereg-D15)

- `src/p02_pests_models.py` written, smoke-tested (outputs deleted). Pre-registration renamed from draft
  and frozen with an implementation section. P2/P3 hold-out SHAs not yet available (download running);
  the frozen build code defines them; they will be logged before any fit on P2/P3. Decision by Claude
  (user away after "reprends tout"), within the autonomy rule.
- Tagging slip: tag `prereg-D15` (commit d292207) captured only the file rename, because a failed
  `git add` left the edited content unstaged. Tags are not rewritten; the frozen D15 content is the commit
  tagged **`prereg-D15-content`** (next commit). Both precede any non-smoke fit on pest data.

## 2026-11-19 — P1 development run done; P2 built

- P1 tuning + refits finished (`logs/p02_P1.log`). Inner-validation PR-AUC (all_dev, N = 3, tuning):
  G_lgbm 0.851, G_lgbm_nbr 0.892, G_lgbm_nbr_leaky 0.901, EcoPE-GAT_leaky 0.801, EcoPE-GAT_causal 0.845,
  EcoPE-GAT_asof 0.882 (tuning values, optimistic; not results).
- P2 built with the frozen `p01_build_pests.py` before any fit on it: 30,000 nodes, dev 22,607 / hold-out
  7,393 (62 / 33 blocks), hold-out SHA-256 `3b99294f4987bb1844e87817d560edb5183495860cfe14a8b58317ef4da296a4`,
  dev prevalence y3 0.422, prior_any 0.265, emergence nodes 16,614 (3,679 y3-positive).

## 2026-11-19 — P1 development summary (`src/p03_pests_dev_summary.py`, descriptive)

CV fold-mean PR-AUC (seed-mean, 3 seeds), N = 3 / N = 5: EcoPE-GAT_asof 0.868 / 0.842, G_lgbm_nbr_leaky
0.867 / 0.848, G_lgbm_nbr 0.855 / 0.817, EcoPE-GAT_causal 0.811 / 0.771, G_lgbm 0.801 / 0.771,
EcoPE-GAT_leaky 0.761 / 0.763. Paired differences (95% block bootstrap within folds), N = 3:
asof - causal +0.058 [0.020, 0.119]; asof - G_lgbm_nbr +0.014 [-0.025, 0.064] (N = 5: +0.024 [0.004, 0.108]);
G_lgbm_nbr_leaky - G_lgbm_nbr +0.012 [-0.005, 0.053] (N = 5: +0.030 [0.006, 0.095]);
**EcoPE-GAT_leaky - causal -0.050 [-0.066, -0.003]** (opposite of the D13 direction; reported as is).
Final model (rule D14 s3 at N = 3): EcoPE-GAT_asof. R7 analogue, emergence nodes (19,689; 4,807 pos):
asof - persistence +0.162 [0.087, 0.209]; PR-AUC - prevalence +0.385.
Reading: leakage exists in the data (D2) and the tabular model exploits it, but the leaky GAT does not
gain from it here; the as-of model is the best causal model. No change to any frozen plan.

## 2026-11-20 — D16 candidate EcoPE-GAT_asof_plus (exploratory, development only)

- User (2026-11-20): the contribution must be a model ("une réelle contrib comme EcoPE-GAT ou variant").
  Contribution framed as the model **EcoPE-GAT_asof** (temporally consistent message passing), with
  leaky/causal/tabular models as references.
- Weak point: on pathogens asof < G_lgbm_nbr (-0.037). Likely cause: G_lgbm_nbr sees the 100 km / 50-100 km
  ring record counts, asof does not. Variant **EcoPE-GAT_asof_plus**: same architecture and search space;
  node features + as-of edge features also include those two counts (computed as of t_i, so no window
  information; self-check j = i reproduces persist100). Added to `src/p02_pests_models.py` as an extra
  model never run by default (the frozen D15 model set and outputs are unchanged); summary pairs added.
- Plan: P1, P2, P3 dev runs, then the pathogen data. If kept, pre-registered as D16 (incl. this having been
  chosen after seeing development results) and tagged before the single hold-out run. Decision by Claude.

## 2026-11-20 — P3 built

- Coleoptera North America background download complete (no retry needed after the fix).
- P3 built with the frozen `p01_build_pests.py` before any fit on it: 30,000 nodes, dev 24,401 / hold-out
  5,599 (80 / 20 blocks), hold-out SHA-256 `64023ddb219c74141334fd2c5f05ccb10f42c67d1745814fdac392506782ca67`,
  dev prevalence y3 0.359, prior_any 0.329, emergence nodes 16,380 (1,464 y3-positive).

## 2026-11-20 — P2 development summary (descriptive)

CV fold-mean PR-AUC N = 3 / N = 5: G_lgbm_nbr_leaky 0.920 / 0.895, G_lgbm_nbr 0.905 / 0.848,
EcoPE-GAT_asof 0.900 / 0.842, EcoPE-GAT_leaky 0.897 / 0.873, G_lgbm 0.895 / 0.869, EcoPE-GAT_causal
0.889 / 0.847 (near ceiling). Leakage gains: GAT leaky - causal N = 5 +0.026 [0.001, 0.082]; LGBM leaky -
causal N = 5 +0.047 [0.012, 0.080] (N = 3 CIs include 0). asof - G_lgbm_nbr N = 3 -0.006 [-0.019, 0.014];
asof - causal N = 3 +0.010 [-0.008, 0.045]. Final model (rule, N = 3): G_lgbm_nbr (asof 0.0059 below,
just outside the 0.005 tie margin). Emergence: G_lgbm_nbr - persistence +0.125 [0.080, 0.192].

## 2026-11-20 — H4c done (descriptive)

Leaky - causal EcoPE-GAT CV fold-mean gap at N = 5, by graph (D2 P(leaky | pos) / P(leaky | neg)):
knn5 +0.053 (0.215 / 0.002), knn10 +0.114 (0.302 / 0.005), knn20 +0.097 (0.413 / 0.012), radius50 +0.110
(0.481 / 0.016), radius100 +0.146 (0.533 / 0.024). The gap is positive for every graph and largest for the
graph with most leakage; the k = 5 graph leaks least. Output: reports/dev_results/h4c.csv.

## 2026-11-20 — asof_plus on P1 (dev, exploratory)

EcoPE-GAT_asof_plus CV fold-mean 0.872 (N = 3) / 0.846 (N = 5), best causal model on P1 (asof 0.868 / 0.842).
asof_plus - G_lgbm_nbr N = 3 +0.018 [-0.010, 0.071], N = 5 +0.029 [0.010, 0.102]; asof_plus - asof +0.004
[-0.010, 0.022]; asof_plus - G_lgbm_nbr_leaky +0.006 [-0.038, 0.037] (N = 3): matches the leaky tabular
model without leakage. Note: adding pairs shifts the shared bootstrap RNG stream, so CIs of the earlier
pairs moved in the third decimal versus the previous P1 entry (point estimates identical).

## 2026-11-20 — asof_plus on P2 (dev, exploratory)

EcoPE-GAT_asof_plus 0.903 (N = 3) / 0.838 (N = 5); G_lgbm_nbr 0.905 / 0.848. asof_plus - G_lgbm_nbr
N = 3 -0.002 [-0.012, 0.018] (tie), N = 5 -0.011 [-0.055, 0.035]; asof_plus - asof +0.004 [-0.006, 0.016].
P2 is near ceiling; no model separates from G_lgbm_nbr.

## 2026-11-20 — P3 development summary (descriptive)

CV fold-mean N = 3 / N = 5: G_lgbm_nbr_leaky 0.950 / 0.947, G_lgbm_nbr 0.948 / 0.939, G_lgbm 0.945 / 0.931,
EcoPE-GAT_leaky 0.943 / 0.949, EcoPE-GAT_asof 0.942 / 0.936, EcoPE-GAT_causal 0.929 / 0.920 (ceiling).
GAT leaky - causal N = 5 +0.030 [0.002, 0.111]; asof - G_lgbm_nbr N = 3 -0.006 [-0.036, 0.008];
asof - causal +0.013 [-0.014, 0.047]. Final model (rule): G_lgbm_nbr. Emergence (16,380 nodes, 1,464 pos):
G_lgbm_nbr - persistence +0.244 [0.077, 0.341]; PR-AUC - prevalence +0.612.
