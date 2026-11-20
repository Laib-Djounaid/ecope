# Pre-registration D16 (DRAFT): the EcoPE as-of neighbour encoding as the proposed method

Status: draft, not frozen. Decided with the user on 2026-11-20 ("On va pour 2 et 1 en même temps"):
the contribution is the **as-of neighbour encoding** (EcoPE family), instantiated in a graph model and
a tabular model, with **EcoPE-GAT_asof_plus** as the flagship model.

**Disclosure.** Both proposed models were designed AFTER seeing development results of D13-D15
(decision log, 2026-11-20). Their development results are therefore exploratory. Only the single
hold-out run, after this file is frozen, is confirmatory.

## 1. Proposed models (code fixed at freeze)

- **EcoPE-GAT_asof_plus**: EcoPE-GAT_asof (D14: messages from every geographic neighbour j, with j's
  time-stamped features recomputed as of the receiver's year t_i; context encoder; edge attributes
  distance and time gap), whose node and as-of edge features also include the target-record counts
  within 100 km and in the 50-100 km ring before t_i. Pathogens: `src/11_d14.py --plus`; pests:
  `src/p02_pests_models.py --models EcoPE-GAT_asof_plus`. Search space = D14 asof_space.
- **G_lgbm_nbr_asof**: G_lgbm_nbr in which the kNN neighbour block is computed with the as-of encoding over
  ALL neighbours (means of the as-of features, max prior_any, number of neighbours) plus the same
  radius block. Pathogens: `src/14_d16_lgbm_asof.py`; pests: `src/p02_pests_models.py --models
  G_lgbm_nbr_asof`. Search space = D13 LightGBM space.
- Same tuning, seeds, horizons and contexts as the models they are compared with in each data set
  (pathogens: D13 protocol; pests: D15 protocol).

## 2. Hypotheses (hold-out; statistics as in D13/D15)

Data sets: pathogens (setting S, and T reported) and the pest cases P1, P2, P3 (setting S).
"Supported" = 95% block-bootstrap CI lower bound > 0 (2,000 replicates, seed 0, replicates without
positives dropped and counted). Primary horizon: N = 5 for pathogens, N = 3 for pests.

- **E1 (as-of beats removal, graph model):** EcoPE-GAT_asof_plus - EcoPE-GAT_causal > 0.
- **E2 (as-of beats removal, tabular model):** G_lgbm_nbr_asof - G_lgbm_nbr > 0.
- **E3 (general method claim):** E1 and E2 supported in at least 3 of the 4 data sets each.
- **E4 (flagship vs best tabular reference):** EcoPE-GAT_asof_plus - G_lgbm_nbr; superiority
  tested, result reported per data set whatever its sign.
- **E5 (emergence):** on hold-out nodes without a prior target record within 50 km, the best of the two
  proposed models on development CV (chosen per data set before the hold-out run, written to
  `results/d16/final_model_<dataset>.json`) minus the persistence heuristic > 0.
- **E6 (leaky ceiling, descriptive):** the proposed models minus their leaky counterparts
  (EcoPE-GAT_leaky, G_lgbm_nbr_leaky), reported as the share of the leakage gain recovered without leakage.

## 3. Order of work

Development runs of both models on all data sets -> fill section 4 (implementation, smoke tests,
development summary) -> tag `prereg-D16` -> extend the single hold-out script (D13-D16 hypotheses,
one run) -> tag `frozen-D16` -> hold-out run.

## 4. Implementation (to be completed at freeze)
