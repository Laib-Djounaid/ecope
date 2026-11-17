# Pre-registration D13 (DRAFT, not yet frozen; no tuning before the user approves it)

Paper focus (decision D13): temporal leakage through neighbours in graph-based species
distribution models, with EcoPE-GAT (v1, v2, OD, geo GAT) as the central case study. All variants
and all pre-registered failures are reported.

## 0. Data, splits, labels (unchanged, frozen)

- Development set: 36,852 plots, 178 blocks (~500 km), 5 spatial folds (`splits_v2.parquet`).
  Development transfer split: train_high (87 blocks) / test_low (91 blocks).
- Sealed hold-out: 7,920 plots, 92 blocks, SHA-256 `31e674c0...3a7f`; its low-effort part (same
  rule as the development split): 3,629 plots, 27 blocks, 97 / 134 / 139 positives at
  N = 5 / 7 / 10. High-effort part: 4,291 plots, 65 blocks.
- Label y_i^N = 1 if at least one target record lies within 50 km of plot i in the years
  (t_i, t_i + N], where t_i is the survey year. **Primary horizon N = 5**; N = 7 and 10 secondary
  (reported, no verdict).

## 1. Formal causal-graph rule

Every time-stamped node feature of plot j (prior target records, number of prior species, prior
non-target fungal effort) uses records with year < t_j. Static covariates (SoilClim PCs,
coordinates, biome, plot traits/attributes) are treated as carrying no target information
(assumption stated in the paper).

**Rule.** A message j -> i is allowed, in every layer, only if t_j <= t_i. Self-loops allowed.

**Property (proof by induction over layers).** After L layers, h_i depends on node features of the
nodes j_0 on directed paths j_0 -> j_1 -> ... -> j_L = i. Along any allowed path t_{j0} <= ... <= t_i,
so every time-stamped quantity reaching i uses records with year < t_{j0} <= t_i: nothing from
i's label window (t_i, t_i + N]. The weaker rule "information of j predates t_i" (t_j <= t_i + 1)
is NOT closed under composition over two layers, which is why the strict order is used.

The same rule applies to every graph (geo, trait, combined), transductive or inductive, and to
neighbourhood aggregates used by tabular models (G_lgbm + neighbourhood). Implementation:
`graph_models.time_causal`, `neighbour_features.neighbour_block`.

**Diagnostic for other authors** (`src/leakage_diagnostic.py`, to be packaged):
- D1 (needs only dates): share of directed edges with t_j > t_i, overall and per node.
- D2 (needs the raw time-stamped source data): share of positive vs negative nodes receiving a
  message from a neighbour whose features contain information from the node's label window, and
  the PR-AUC of that indicator alone.
- D3 (needs a model): performance of the same model with the original vs the causal graph.
Recommendation: always report D1; report D2 when possible; use the causal graph by default.

## 2. Models

- **EcoPE-GAT**: GAT (PyTorch Geometric GATConv, 2 layers) on the geographic kNN graph (haversine,
  k = 10, symmetrised, w = 1/(1 + d/50 km)). Node features = prior presence block (log prior count,
  prior species count, prior_any) + independent prior effort. Ecological position (SoilClim PC1/PC2,
  latitude/longitude, biome one-hot) enters through an encoder **before** message passing; encoder
  form is a tuned categorical hyperparameter: identity (concatenation), linear, MLP (v1 form).
  Class-balanced BCE.
- `EcoPE-GAT_leaky`: same, original graph. `EcoPE-GAT_causal`: same, causal graph.
- `GAT_noctx_causal`: same as EcoPE-GAT_causal without the ecological position (node features =
  prior presence + effort only).
- `G_lgbm`: LightGBM on feature set G (prior + effort + context).
- `G_lgbm_nbr`: G + time-safe neighbourhood block (causal kNN aggregates + target records within
  100 km and in the 50-100 km ring before t_i).

## 3. Hypotheses, statistics, criteria

Common rules: statistic = PR-AUC of the **seed-mean score** (5 seeds) on the test set; 95% CI =
percentile of 2,000 bootstrap replicates resampling spatial blocks of the test set (seed 0);
"supported" = CI lower bound > 0. Hold-out settings:
- **(T) transfer**: train on development train_high blocks (inner validation as in
  `transfer_inner_val`), test on the hold-out low-effort part (3,629 plots, 27 blocks);
- **(S) spatial**: train on the whole development set (blocked inner validation), test on the whole
  hold-out (7,920 plots, 92 blocks).
The graph is built transductively over development + hold-out nodes; hold-out labels are never
used. Development results (fold-mean spatial CV, development transfer) are reported alongside.

- **H1 (main):** PR-AUC(EcoPE-GAT_leaky) - PR-AUC(EcoPE-GAT_causal) > 0 in (T) **and** in (S).
  Both must hold for H1 to be supported. Each model has its own tuning with the same budget.
- **H2:** EcoPE-GAT_causal vs G_lgbm_nbr, superiority test in (T) and (S) (CI lower bound > 0 =
  superior). Reported as is, whatever the result; our exploratory expectation is "not superior".
- **H3:** EcoPE-GAT_causal vs GAT_noctx_causal, superiority test in (T) and (S). Reported as is.
- **H4 (dose-response; development data only, frozen hyperparameters of EcoPE-GAT from the tuning,
  5 seeds, spatial CV fold-mean, N = 5):**
  - H4a (experimental): for q in {0, 0.25, 0.5, 0.75, 1}, remove a random fraction q of the
    non-causal edges (t_j > t_i), q = 0 = original graph, q = 1 = causal. Supported if the slope of
    fold-mean PR-AUC on q is negative with 95% block-bootstrap CI upper bound < 0.
  - H4b (observational): among development positives, Spearman correlation between the share of
    a plot's neighbours surveyed after it and the gain in its percentile rank (leaky minus causal,
    out-of-fold). Supported if the CI lower bound > 0.
  - H4c (sensitivity, descriptive): D1, D2 and the leaky-causal gap for k in {5, 10, 20}, radius
    graphs of 50 and 100 km (capped at 50 nearest neighbours), and N in {5, 7, 10}.
- Multiplicity: H1 requires both settings (conjunction); no other correction. Secondary horizons
  and the development results are descriptive.

## 4. Tuning (development data only) and reduced budget

- Optuna TPE, 30 trials per (model, training set), inner validation spatially blocked, tuning
  horizon N = 10 (config `evaluation.tuning_horizon`, unchanged), refits for N = 5, 7, 10 x 5 seeds.
  SQLite checkpoints on Google Drive; GNNs on Colab GPU.
- **Full tuning only for the models of H1-H3**: EcoPE-GAT_leaky, EcoPE-GAT_causal,
  GAT_noctx_causal, G_lgbm_nbr, G_lgbm. Training sets: 5 development folds, development
  train_high (also the hold-out (T) training set), whole development set (hold-out (S)).
  GNN search space: encoder {identity, linear, MLP}, hidden {64, 128}, heads {2, 4}, dropout
  [0.1, 0.5], lr [1e-4, 1e-2] log, weight decay [1e-6, 1e-3] log (config `graph`).
- **Case-study variants, defaults only, development data only, original and causal graph,
  5 seeds**: EcoPE v1 (trait graph, traits + MLP encoder), v2 (P/PT x trait/geo/combined), OD
  models of pilot v4, GCN/SAGE, tabular sets A-F with LR/RF/LGBM/MLP. The hybrid of `04_graph.py`
  (cross-fitted neighbour labels) is made time-causal (neighbour labels only from neighbours whose
  label window ends before t_i) or dropped; decision recorded before running.
- Earlier pilot numbers (v1-v4) are reported as obtained, flagged as using the original graph.

## 5. Final evaluation

After all configurations are frozen (git tag `frozen-D13`), `src/08_holdout.py` loads the
hold-out (`include_holdout=True`), checks the SHA-256, builds both graphs, refits the frozen H1-H3
models on the (T) and (S) training sets with 5 seeds, and computes H1-H3 once. Before any
prediction, D1 (share of later-surveyed neighbours) is reported for hold-out nodes. No change
after this run.
