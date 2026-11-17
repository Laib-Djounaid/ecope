# Pre-registration D13 (FROZEN at git tag `prereg-D13`, before any tuning)

Paper focus (decision D13): temporal leakage through neighbours in graph-based species
distribution models, with EcoPE-GAT (v1, v2, OD, geo GAT) as the central case study. All variants
and all pre-registered failures are reported. The draft (commit e671d42) was approved by the user
on 2026-11-17 with the corrections and additions integrated below (marked [user]).

## 0. Data, splits, labels (unchanged, frozen)

- Development set: 36,852 plots, 178 blocks (~500 km), 5 spatial folds (`splits_v2.parquet`).
  Development transfer split: train_high (87 blocks) / test_low (91 blocks).
- Sealed hold-out: 7,920 plots, 92 blocks, SHA-256 `31e674c06b13b0ac4b5afb0fc3f0c66af44c9e7e3ab05d464cbe4c4ff3cd3a7f`;
  its low-effort part (same rule as the development split): 3,629 plots, 27 blocks, 97 / 134 / 139
  positives at N = 5 / 7 / 10. High-effort part: 4,291 plots, 65 blocks.
- Label y_i^N = 1 if at least one target record lies within 50 km of plot i in the years
  (t_i, t_i + N], t_i = survey year. **Primary horizon N = 5**; N = 7 and 10 secondary (reported,
  no verdict).

## 1. Formal causal-graph rule

Every time-stamped node feature of plot j uses records with year < t_j: prior target records,
number of prior target species, and the independent effort (non-target fungal GBIF records in the
plot's 0.5-degree cells, years 2000 .. t_j - 1; `src/01d_effort_from_sql.py`) [user: effort is a
node feature, dated]. Static covariates (SoilClim PCs, coordinates, biome, plot traits/attributes)
are treated as carrying no target information (assumption stated in the paper).

**Rule.** A message j -> i is allowed, in every layer, only if t_j <= t_i. Self-loops allowed.

**Property (proof by induction over layers).** After L layers, h_i depends on the node features of
the nodes j_0 on directed paths j_0 -> ... -> j_L = i. Along any allowed path t_{j0} <= ... <= t_i,
so every time-stamped quantity reaching i uses records with year < t_{j0} <= t_i: nothing from i's
label window (t_i, t_i + N]. The weaker rule "j's information predates t_i" (t_j <= t_i + 1) is
not closed under composition over two layers (a one-year leak remains), hence the strict order.

The rule applies to every graph (geo, trait, combined), transductive or inductive, and to
neighbourhood aggregates of tabular models. Implementation: `graph_models.time_causal`,
`neighbour_features.neighbour_block`.

**Cost of the rule [user]** (development geo graph, k = 10): 15.2% of directed edges removed;
median in-degree 12 -> 10; **232 nodes (0.6%) receive no causal message** ("isolated"; 12 positives,
prevalence 0.052 vs 0.051 elsewhere; 77 in development test_low, 0 positives). Treatment: GATConv
self-loops, so an isolated node is predicted from its own features only; no imputation of
neighbours. Reported for the hold-out before prediction. **Sensitivity:** H1, H1b and H2 are also
computed on test nodes that are not isolated in the causal graph (same subset for both models), to
check that a loss of the causal model is not due to isolated nodes only.

**Diagnostic for other authors** (`src/leakage_diagnostic.py`):
- D1 (needs only dates): share of directed edges with t_j > t_i, overall and per node.
- D2 (needs the raw time-stamped source data): share of positive vs negative nodes receiving a
  message from a neighbour whose features contain information from the node's label window, and
  the PR-AUC of that indicator alone.
- D3 (needs a model): performance of the same model with the original vs the causal graph.
Recommendation: always report D1; report D2 when possible; use the causal graph by default.

## 2. Models

- **EcoPE-GAT**: GAT (PyTorch Geometric GATConv, 2 layers, self-loops) on the geographic kNN graph
  (haversine, k = 10, symmetrised, w = 1/(1 + d/50 km)). Node features = prior presence block
  (log prior count, prior species count, prior_any) + dated independent effort (log1p).
  Ecological position (SoilClim PC1/PC2, latitude/longitude, biome one-hot) enters through an
  encoder **before** message passing; the encoder form is a tuned categorical hyperparameter:
  identity (concatenation), linear (to 32), MLP (v1 form, 64 -> 32). Class-balanced BCE.
- `EcoPE-GAT_leaky`: original graph. `EcoPE-GAT_causal`: causal graph.
- `GAT_noctx_causal`: EcoPE-GAT_causal without the ecological position (node features = prior
  presence + dated effort only).
- `G_lgbm`: LightGBM on feature set G (prior + effort + context).
- `G_lgbm_nbr` (= G_lgbm + neighbourhood, causal): G + time-safe neighbourhood block (aggregates
  over causal kNN neighbours: mean log prior count, max prior_any, mean log effort, number of
  causal neighbours; target records within 100 km and in the 50-100 km ring before t_i).
- `G_lgbm_nbr_leaky` [user, for H5]: same, but kNN aggregates over ALL kNN neighbours (original
  graph). The radius counts are defined relative to t_i and are identical in both versions.

## 3. Hypotheses, statistics, criteria

Common rules: statistic = PR-AUC of the **seed-mean score** (5 seeds; LightGBM: 5 seeds of
`random_state`) on the test set; 95% CI = percentile of 2,000 bootstrap replicates resampling
spatial blocks of the test set (seed 0); "supported" = CI lower bound > 0. Hold-out settings:
- **(T) transfer**: train on development train_high blocks (inner validation =
  `transfer_inner_val`), test on the hold-out low-effort part (3,629 plots, 27 blocks);
- **(S) spatial**: train on the whole development set (blocked inner validation built with
  `02_splits.inner_split`, stratified on N = 10, seed = evaluation.split_seed + 500 = 542), test on the whole hold-out (7,920 plots, 92 blocks).
The graph is built transductively over development + hold-out nodes; hold-out labels are never
used. Development results (fold-mean spatial CV, development transfer) are reported alongside.

- **H1 (main):** PR-AUC(EcoPE-GAT_leaky) - PR-AUC(EcoPE-GAT_causal) > 0 in (T) **and** in (S).
  Both must hold. Each model has its own tuning with the same budget.
- **H1b (rank reversal) [user]:** in (T) and in (S): EcoPE-GAT_leaky - G_lgbm_nbr has CI lower
  bound > 0 **and** EcoPE-GAT_causal - G_lgbm_nbr has CI lower bound <= 0 (causal not superior).
  Supported if all four conditions hold.
- **H2:** EcoPE-GAT_causal vs G_lgbm_nbr, superiority test in (T) and (S). Reported as is.
- **H3:** EcoPE-GAT_causal vs GAT_noctx_causal, superiority test in (T) and (S). Reported as is.
- **H4 (dose-response; development data only, frozen hyperparameters of EcoPE-GAT_leaky, 5 seeds,
  spatial CV fold-mean, N = 5):**
  - H4a (experimental): for q in {0, 0.25, 0.5, 0.75, 1}, remove a random fraction q of the
    non-causal edges (t_j > t_i); q = 0 = original graph, q = 1 = causal. Supported if the
    least-squares slope of fold-mean PR-AUC on q has 95% block-bootstrap CI upper bound < 0.
  - H4b (observational): among development positives, Spearman correlation between the share of a
    plot's neighbours surveyed after it and the gain in its out-of-fold percentile rank (leaky minus
    causal). Supported if the CI lower bound > 0.
  - H4c (sensitivity, descriptive): D1, D2 and the leaky-causal gap for k in {5, 10, 20}, radius
    graphs of 50 and 100 km (capped at 50 nearest neighbours), and N in {5, 7, 10}.
- **H5 (scope) [user]:** PR-AUC(G_lgbm_nbr_leaky) - PR-AUC(G_lgbm_nbr) in (T) and (S).
  Supported if the CI lower bound > 0 in both. Interpretation fixed in advance: if H5 is supported,
  the leak is a property of dated neighbourhood variables in general, not of GNNs only, and the
  paper's title and framing are widened accordingly ("dated neighbourhood features in spatial
  models"); if not, the framing stays on graph message passing. The ratio of the H5 gap to the H1
  gap is reported (descriptive).
- Multiplicity: H1, H1b and H5 are conjunctions over settings; no other correction. Secondary
  horizons and development results are descriptive.

## 4. Tuning (development data only), reduced budget

- Optuna TPE (seed 0), **30 trials** per (model, training set), objective = inner-validation PR-AUC
  at **N = 5** [user: the primary horizon]; the best hyperparameters are **reused without retuning
  for N = 7 and N = 10** (documented in the paper). Refits: N = 5, 7, 10 x seeds 0-4.
  SQLite checkpoints after each trial; GNNs can run on Colab GPU with Drive checkpoints.
- **Full tuning only for the models of H1-H5**: EcoPE-GAT_leaky, EcoPE-GAT_causal,
  GAT_noctx_causal, G_lgbm, G_lgbm_nbr, G_lgbm_nbr_leaky. Training sets: 5 development folds,
  development train_high (= hold-out (T) training set), whole development set (= hold-out (S)).
  GNN search space: encoder {identity, linear, MLP} (EcoPE models only), hidden {64, 128},
  heads {2, 4}, dropout [0.1, 0.5], lr [1e-4, 1e-2] log, weight decay [1e-6, 1e-3] log;
  max 200 epochs, patience 20. LightGBM search space: config `tabular.search_space.lgbm`.
- **Case-study variants, defaults only, development data only, original and causal graph,
  5 seeds**: EcoPE v1 (trait graph, traits + MLP encoder), v2 (P/PT x trait/geo/combined), OD
  models of pilot v4, GCN/SAGE.
- **Hybrid with neighbour labels [user]: removed from the confirmatory analysis.** Shown only as an
  illustration in the case study: leaky version vs strictly causal version, where neighbour j's
  label may be used for plot i only if j's label window has ended: t_j + N <= t_i.
- Earlier pilot numbers (v1-v4) are reported as obtained, flagged as using the original graph.

## 5. Final evaluation

After all configurations are frozen (git tag `frozen-D13`), `src/10_holdout_d13.py` loads the
hold-out (`include_holdout=True`), checks the SHA-256, builds both graphs over development +
hold-out nodes, reports D1 and the isolated-node count for hold-out nodes before any prediction,
refits the frozen models on the (T) and (S) training sets with 5 seeds, and computes H1, H1b,
H2, H3, H5 (and the isolated-node sensitivity) once. No change after this run.
