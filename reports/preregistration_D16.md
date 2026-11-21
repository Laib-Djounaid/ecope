# Pre-registration D16: the EcoPE as-of neighbour encoding (primary and secondary proposed models)

Status: FROZEN at tag `prereg-D16` (2026-11-21). Decided with the user on 2026-11-20 ("On va pour 2 et 1
en même temps") and 2026-11-21 ("Tu peux pas faire les deux ?"): the contribution is the **as-of
neighbour encoding** (as-of message passing). Both graph variants are carried to the hold-out:

- **Primary proposed model: EcoPE-GAT_asof.** Specified in D14 (pathogens, tag `prereg-D14`) and D15
  (pests, tag `prereg-D15-content`) BEFORE any of its development results were seen. Its hold-out tests
  (D14 H8/H9, D15 R8/R9) are confirmatory and are unchanged by this file.
- **Secondary proposed model: EcoPE-GAT_asof_plus**, and the tabular instance **G_lgbm_nbr_asof**.
  Designed AFTER seeing development results of D13-D15 (decision log, 2026-11-20). Their hold-out
  tests below are pre-registered here but labelled secondary in the paper.

## 1. Models (code fixed at this tag)

- **EcoPE-GAT_asof** (D14 section 1): messages from every geographic neighbour j, with j's time-stamped
  features recomputed as of the receiver's year t_i; context encoder; edge attributes (distance, time gap).
- **EcoPE-GAT_asof_plus**: EcoPE-GAT_asof whose node and as-of edge features also include the target-record
  counts within 100 km and in the 50-100 km ring before t_i. Pathogens: `src/11_d14.py --plus`; pests:
  `src/p02_pests_models.py --models EcoPE-GAT_asof_plus`. Search space = D14 asof_space.
- **G_lgbm_nbr_asof**: G_lgbm_nbr with the kNN neighbour block computed with the as-of encoding over ALL
  neighbours (means of the as-of features, max prior_any, number of neighbours) plus the same radius
  block. Pathogens: `src/14_d16_lgbm_asof.py`; pests: `src/p02_pests_models.py --models G_lgbm_nbr_asof`.
  Search space = D13 LightGBM space.
- Same tuning, seeds, horizons and contexts as the models they are compared with (pathogens: D13
  protocol; pests: D15 protocol).

## 2. Hypotheses (hold-out; statistics as in D13/D15)

Data sets: pathogens (setting S; T reported) and pest cases P1, P2, P3 (setting S). "Supported" = 95%
block-bootstrap CI lower bound > 0 (2,000 replicates, seed 0, replicates without positives dropped and
counted). Primary horizon: N = 5 (pathogens), N = 3 (pests); other horizons reported.

Primary (already pre-registered, restated for the paper): D14 H8, H9; D15 R8, R9 (EcoPE-GAT_asof vs
G_lgbm_nbr and vs EcoPE-GAT_causal).

Secondary (new here):
- **E1:** EcoPE-GAT_asof_plus - EcoPE-GAT_causal > 0.
- **E1b (ablation):** EcoPE-GAT_asof_plus - EcoPE-GAT_asof, two-sided, reported as is.
- **E2:** G_lgbm_nbr_asof - G_lgbm_nbr > 0.
- **E3 (general method claim):** H9/R9 and E2 each supported in at least 3 of the 4 data sets.
- **E4:** EcoPE-GAT_asof_plus - G_lgbm_nbr, superiority, reported per data set whatever its sign.
- **E5 (emergence):** on hold-out nodes without a prior target record within 50 km, EcoPE-GAT_asof minus
  the persistence heuristic > 0 (the D14/D15 final-model emergence tests H7/R7 are unchanged).
- **E6 (leaky ceiling, descriptive):** EcoPE-GAT_asof, EcoPE-GAT_asof_plus and G_lgbm_nbr_asof minus their
  leaky counterparts, reported as the share of the leakage gain recovered.

The final-model selection rules of D14 section 3 and D15 section 3 are NOT changed: the D16 models do
not enter the selection set.

## 3. Order of work

Development runs (done) -> this file frozen, tag `prereg-D16` -> single hold-out script for D13-D16 ->
tag `frozen-D16` -> one hold-out run.

## 4. Implementation and development summary (exploratory)

- Code at freeze: commit tagged `prereg-D16` on branch `rebuild-step1-3`. Pathogen outputs `results/d16/`;
  pest outputs `results/pests/<case>/`; summaries `reports/dev_results/` and `reports/pests/`.
- Self-checks (all passed in the real runs): as-of leak check 0 edges with window information over
  20,000 sampled edges per horizon, 0 edges removed (`results/d16/asof_leak_check.json`); the radius
  block of asof_plus equals `neighbour_block(..., causal=True)` columns; pests: stored table and labels
  reproduced, leak check on all edges = 0.
- Development results (CV fold-mean PR-AUC, primary horizon):

| Data set | EcoPE-GAT_causal | EcoPE-GAT_asof | EcoPE-GAT_asof_plus | G_lgbm_nbr | G_lgbm_nbr_asof |
|---|---|---|---|---|---|
| Pathogens (N=5) | 0.364 | 0.419 | 0.416 | 0.456 | 0.448 |
| P1 H. halys Europe (N=3) | 0.811 | 0.868 | 0.872 | 0.855 | 0.854 |
| P2 H. halys N. America (N=3) | 0.889 | 0.900 | 0.903 | 0.905 | 0.910 |
| P3 P. japonica N. America (N=3) | 0.929 | 0.942 | 0.944 | 0.948 | 0.949 |

- Development differences: asof - causal pathogens N5 +0.055 [-0.018, 0.079], P1 N3 +0.058
  [0.020, 0.119]; asof_plus - asof pathogens -0.003, P1 +0.004 [-0.010, 0.022]; G_lgbm_nbr_asof -
  G_lgbm_nbr pathogens N5 -0.008 [-0.046, 0.070]. On development data E2 and E3 look unlikely to be
  supported; they are kept as pre-registered tests and will be reported whatever the outcome.
