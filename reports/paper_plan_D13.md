# Paper plan (D13), in the original section structure of main.tex

Working title: "Temporal leakage through neighbours in graph-based species distribution models:
lessons from EcoPE-GAT for plant-pathogen emergence" (Environmental Modelling & Software).

**Abstract.** Graph models pass neighbours' features at prediction time; when node features are
time-stamped (prior occurrences), neighbours observed later leak future records even under
spatial-block validation. Case study, size of the inflation on a sealed hold-out, causal-graph rule,
diagnostic; EcoPE-GAT results after correction, reported honestly.

1. **Introduction.** Graph SDMs and the appeal of message passing; plant-pathogen emergence from
   GBIF + sPlotOpen; validation literature focuses on the split, not on model inputs; our finding
   that a spatially blocked evaluation still leaked; contributions (rule + proof, diagnostic,
   quantification on a sealed hold-out, re-evaluation of EcoPE-GAT); pre-registration statement.
2. **Related work**
   - 2.1 Species distribution models and transferability (Elith & Leathwick 2009; Yates et al. 2018).
   - 2.2 Sampling bias and observation processes in occurrence data (Phillips 2009; Beck 2014;
     Warton 2013; Dorazio 2014; Fithian 2015; occupancy-detection refs).
   - 2.3 Validation and leakage (Roberts 2017; Hijmans 2012; Meyer 2018; Ploton 2020; Wadoux 2021;
     Kaufman 2012; Kapoor & Narayanan 2023).
   - 2.4 Graph neural networks for spatial and spatio-temporal data (DCRNN, STGCN, Graph WaveNet,
     IGNNK, PE-GNN, GNN-SDM, heterogeneous GNN-SDM).
   - 2.5 Temporal leakage in graph learning (JODIE, TGN, Poursafaei 2022, TGB) and the gap.
3. **Proposed approach**
   - 3.1 Motivation: how the leak arises (figure: timeline of plot i and later neighbour j, with
     i's label window inside j's prior).
   - 3.2 Methodology
     - 3.2.1 Dataset: sPlotOpen, GBIF download and filters, 50 km join, labels, eligibility,
       independent effort layer (GBIF SQL), splits incl. sealed hold-out and transfer split.
     - 3.2.2 Architecture: EcoPE-GAT (encoder forms, graphs), references (G_lgbm, G_lgbm_nbr,
       GAT_noctx), variants v1/v2/OD; the causal-graph rule, proof sketch, neighbourhood
       features; diagnostic D1-D3.
     - 3.2.3 Predictions: training, tuning (budget), seeds, statistics, pre-registered H1-H4.
4. **Experiments**
   - 4.1 Leakage audit (D1, D2 on development and, before prediction, hold-out).
   - 4.2 H1: leaky vs causal on the hold-out (T) and (S), plus development results.
   - 4.3 H2/H3: EcoPE-GAT after correction vs G_lgbm_nbr and vs no-context GAT.
   - 4.4 H4: dose-response (edge removal q, plot-level), sensitivity to k, radius, N.
   - 4.5 Case-study history: all pilots (v1-v4) and variants, original vs causal graph; tabular
     feature sets; within-dataset trait diagnostic; effort dependence (OD test c).
5. **Discussion.** What the leak means for published graph SDMs; when it can occur (time-stamped
   node features, mixed survey dates, transductive graphs); limits (one taxon group, single
   label per plot, low power in low-effort hold-out with 27 blocks); recommendations checklist.
6. **Conclusion.**
Appendices / supplementary: consistency table paper <-> code, decision log summary, software and
data availability (DOIs 10.15468/dl.ps65sp and 10.15468/dl.4739rd, MIT code).
