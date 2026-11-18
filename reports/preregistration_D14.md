# Pre-registration addendum D14 (FROZEN at git tag `prereg-D14`, before any D14 tuning and any hold-out access)

Draft commit 8fbdc51, approved by the user on 2026-11-18 ("vas-y").

Decision D14 (user, 2026-11-18): the paper becomes an **applied** paper on predicting
plant-pathogen emergence (option B) **and** proposes a new causal graph model (option C).
D13 (tag `prereg-D13`) stays frozen and unchanged: H1-H5 are tested exactly as registered and
become the "methodological rigour" section of the paper. D14 only **adds** hypotheses H6-H9 and
one model. The hold-out has not been loaded at any point (`assert not e["holdout"].any()`).

Working title: "Forecasting plant-pathogen emergence from vegetation plots and GBIF records with
time-causal graph models" (EMS).

## 1. New model (option C): EcoPE-GAT_asof ("as-of" neighbour features)

Motivation from D13 development results: the causal rule removes later-surveyed neighbours (15% of
edges, 232 isolated nodes), and EcoPE-GAT_causal loses to G_lgbm_nbr. The leak comes from the
*date at which a neighbour's features are computed*, not from the neighbour itself.

**Idea.** Keep every geographic edge j -> i, but compute the neighbour's time-stamped features
**as of the receiver's date**: x_{j->i} = (log1p prior target count within 50 km of j with year < t_i,
prior_any, dated effort of j's cells for years 2000 .. t_i - 1). Static context of j (SoilClim PCs,
coordinates, biome) through the EcoPE encoder; edge attributes: distance (km), t_i - t_j (signed).

**Architecture.** One attention layer over edge-level messages:
m_{j->i} = MLP([x_{j->i}, enc(ctx_j), edge_attr]); a_{ji} = softmax_j(w^T LeakyReLU([h_i, m_{j->i}]));
h_i' = [h_i, sum_j a_{ji} m_{j->i}] -> MLP -> logit, with h_i = [x_i, enc(ctx_i)].
One layer only, so the property below is exact (multi-layer as-of would need features evaluated at
the root's date along whole paths; not done).

**Property.** Every time-stamped quantity reaching i is computed from records with year < t_i;
no edge is removed, no isolated node. D2 = 0 by construction (verified numerically).

**Tuning.** Same protocol and budget as D13 (Optuna TPE seed 0, 30 trials, N = 5, contexts
cv0-cv4, dev_transfer, all_dev; refits N = 5/7/10 x seeds 0-4). Space: encoder {identity, linear,
mlp}, hidden {64, 128}, dropout [0.1, 0.5], lr [1e-4, 1e-2] log, weight decay [1e-6, 1e-3] log.

## 2. Applied hypotheses (option B) and model hypotheses (option C)

Same statistics as D13: PR-AUC of the seed-mean score, 2,000 block-bootstrap replicates (seed 0),
supported = CI lower bound > 0, settings (T) transfer to the hold-out low-effort part and (S)
whole hold-out. Primary horizon N = 5.

- **H6 (forecasting skill beyond persistence).** The final model (selection rule in 3) minus a
  **persistence heuristic** that needs no fitting: score = log1p(target records within 100 km
  before t_i) (`prior100_log_n`, ties broken by prior count within 50 km). Supported if CI lower
  bound > 0 in (T) and (S).
- **H7 (true emergence).** Restricted to hold-out plots with **no target record within 50 km
  before t_i** (prior_any = 0), i.e. the pathogen is not yet known locally: final model minus
  persistence heuristic, and final model PR-AUC minus subset prevalence. Supported if both CI lower
  bounds > 0 in (S); (T) reported (expected low power). The number of positives in the subset is
  reported before prediction.
- **H8 (new model vs best tabular).** EcoPE-GAT_asof minus G_lgbm_nbr, in (T) and (S). Superiority
  test, reported as is.
- **H9 (as-of vs removal).** EcoPE-GAT_asof minus EcoPE-GAT_causal, in (T) and (S). Supported if
  CI lower bound > 0 in both.
- Descriptive: calibration (Brier score, reliability curve) of the final model; risk maps from the
  (S) model; N = 7 and 10.

## 3. Final-model selection rule (fixed now, applied on development data only)

Among the causal models {G_lgbm_nbr, EcoPE-GAT_causal, GAT_noctx_causal, EcoPE-GAT_asof}, the final
model is the one with the highest development spatial-CV fold-mean PR-AUC (seed-mean score, N = 5);
ties (difference < 0.005) go to the simpler model in the order listed. Leaky models are never
eligible. The choice is written to `results/d14/final_model.json` and committed before `frozen-D14`.

## 4. Evaluation

`src/10_holdout_d13.py` is extended (same single run, after tag `frozen-D14`, which replaces
`frozen-D13`): H1-H5 as frozen in D13, plus H6-H9, plus the H7 subset counts reported before
prediction. No change after this run.

## 5. Implementation (frozen with this addendum)

- Edge features: `src/asof_features.py` (`asof_edges`). Self-check asserted: for j = i the as-of
  features reproduce the stored node features exactly (prior count, species count, effort).
- Model: `graph_models.AsOfGAT`. Tuning/refits: `src/11_d14.py` (same contexts, inner validations,
  seeds and budget as `src/09_d13.py`; space in `config.yaml` section `d14.asof_space`).
- Smoke run on development data (2 trials, 2 epochs; outputs deleted, no result used): 459,584
  directed edges, 0 removed, minimum in-degree 10; brute-force check on 20,000 random edges:
  0 edges carry a record from the receiver's label window (N = 5). The full check for N = 5/7/10
  is written to `results/d14/asof_leak_check.json` by the real run.
- D14 tuning starts automatically after the D13 GNN run finishes (same machine, CPU).
