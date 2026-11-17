# Literature: information leakage, validation, and graph models for spatial / spatio-temporal data (D13)

Verified on 2026-11-17: DOIs against Crossref metadata (authors, year, title, venue), arXiv ids against
the arXiv API. References already verified for `literature_od.md` are marked (prev.).

## 1. Leakage in machine learning (general)

- **Kaufman, Rosset, Perlich & Stitelman (2012)**, "Leakage in data mining: formulation, detection,
  and avoidance". ACM TKDD 6, 1-21. https://doi.org/10.1145/2382577.2382579
  Formal definition: a feature leaks when it carries information about the target that would not
  be available at prediction time; proposes the "legitimacy" principle by time-stamping
  ("no-time-machine" rule). **Our causal-graph rule is this principle applied to message passing.**
- **Kapoor & Narayanan (2023)**, "Leakage and the reproducibility crisis in machine-learning-based
  science". Patterns 4, 100804. https://doi.org/10.1016/j.patter.2023.100804
  Taxonomy of 8 leakage types found in 294 papers across 17 fields, including "temporal leakage"
  (test information from the future); proposes "model info sheets".
- **Whalen, Schreiber, Noble & Pollard (2022)**, "Navigating the pitfalls of applying machine
  learning in genomics". Nat. Rev. Genet. 23, 169-181. https://doi.org/10.1038/s41576-021-00434-9
  Leakage through dependence between examples (distributional, dependency, confounding); the
  closest analogue in another field of "information shared between related samples".

## 2. Validation of spatial / spatio-temporal ecological models

- **Roberts et al. (2017)**, cross-validation for data with temporal, spatial, hierarchical or
  phylogenetic structure. Ecography 40, 913-929. https://doi.org/10.1111/ecog.02881 (prev.)
- **Hijmans (2012)**, spatial sorting bias in cross-validation. Ecology 93, 679-688.
  https://doi.org/10.1890/11-0826.1 (prev.)
- **Valavi et al. (2019)**, blockCV. Methods Ecol. Evol. 10, 225-232.
  https://doi.org/10.1111/2041-210X.13107 (prev.)
- **Meyer, Reudenbach, Hengl, Katurji & Nauss (2018)**, "Improving performance of spatio-temporal
  machine learning models using forward feature selection and target-oriented validation".
  Environ. Model. Softw. 101, 1-9. https://doi.org/10.1016/j.envsoft.2017.12.001
  Shows random CV overestimates spatio-temporal ML performance; "leave-location-and-time-out"
  validation. Same journal as our target venue.
- **Ploton et al. (2020)**, spatial validation reveals poor performance of large-scale ecological
  mapping models. Nat. Commun. 11, 4540. https://doi.org/10.1038/s41467-020-18321-y (prev.)
- **Meyer & Pebesma (2021)**, area of applicability. Methods Ecol. Evol. 12, 1620-1633.
  https://doi.org/10.1111/2041-210X.13650
- **Wadoux, Heuvelink, de Bruin & Brus (2021)**, "Spatial cross-validation is not the right way to
  evaluate map accuracy". Ecol. Model. 457, 109692. https://doi.org/10.1016/j.ecolmodel.2021.109692
  Counterpoint: spatial CV can be pessimistic; probability sampling for map accuracy.
- **Bergmeir & Benitez (2012)**, cross-validation for time-series predictor evaluation.
  Inf. Sci. 191, 192-213. https://doi.org/10.1016/j.ins.2011.12.028
- **Yates et al. (2018)**, outstanding challenges in the transferability of ecological models.
  Trends Ecol. Evol. 33, 790-802. https://doi.org/10.1016/j.tree.2018.08.001
- **Elith & Leathwick (2009)**, SDMs across space and time. Annu. Rev. Ecol. Evol. Syst. 40,
  677-697. https://doi.org/10.1146/annurev.ecolsys.110308.120159
- **Pichler & Hartig (2023)**, machine learning and deep learning, a review for ecologists.
  Methods Ecol. Evol. 14, 994-1016. https://doi.org/10.1111/2041-210X.14061

**Gap:** this literature deals with leakage through the *split* (train and test samples that are
close in space or time). It does not address leakage through the *model input*: in a graph model,
a test node receives its neighbours' features at prediction time, so even a perfect spatial/temporal
split leaks if a neighbour's features were computed from data posterior to the test node's date.
Our split was spatially blocked and the leakage still occurred.

## 3. Graph neural networks for spatio-temporal data and for SDMs

- **Li, Yu, Shahabi & Liu (2018)**, DCRNN, ICLR. arXiv:1707.01926; **Yu, Yin & Zhu (2018)**, STGCN,
  IJCAI, https://doi.org/10.24963/ijcai.2018/505; **Wu, Pan, Long, Jiang & Zhang (2019)**,
  Graph WaveNet, IJCAI, https://doi.org/10.24963/ijcai.2019/264. Spatio-temporal GNNs on sensor
  graphs: all nodes share the same time axis, inputs are windows strictly before the forecast time,
  so the cross-node timing problem does not arise.
- **Wu, Zhuang, Labbe & Sun (2021)**, inductive GNNs for spatio-temporal kriging (IGNNK).
  AAAI 35, 4478-4485. https://doi.org/10.1609/aaai.v35i5.16575
- **Klemmer et al. (2023)** PE-GNN; **Wu et al. (2025)** GNN-SDM; **Harrell et al. (2025)**
  heterogeneous GNN for SDM, arXiv:2503.11900 (prev.). None of these describes a time-ordering
  constraint on edges between observations made at different dates.

## 4. Temporal leakage in graph learning (outside ecology)

- **Kumar, Zhang & Leskovec (2019)**, JODIE, KDD, https://doi.org/10.1145/3292500.3330895;
  **Rossi et al. (2020)**, Temporal Graph Networks, arXiv:2006.10637. Continuous-time dynamic graph
  models that only aggregate past events by construction.
- **Poursafaei, Huang, Pelrine & Rabbany (2022)**, "Towards better evaluation for dynamic link
  prediction", NeurIPS Datasets & Benchmarks, arXiv:2207.10128; **Huang et al. (2023)**, Temporal
  Graph Benchmark, arXiv:2307.01026. Show that evaluation choices inflate temporal-graph results.
- **Shchur, Mumme, Bojchevski & Gunnemann (2018)**, "Pitfalls of graph neural network
  evaluation", arXiv:1811.05868.

**Gap:** in temporal-graph learning the "no information from the future" principle is standard,
but it is applied to *edges/events with timestamps* in dynamic graphs. Static spatial kNN graphs
over observations made at different dates (the usual construction of GNN-SDMs, built from
coordinates only) do not carry timestamps, and we found no ecological or SDM paper that tests for,
or constrains, the timing of neighbours' information. A web search (2026-11-17) found practitioner
guides on graph temporal leakage in enterprise data but no formal study for spatial node features.
**Positioning (stated carefully):** the principle is known (Kaufman et al. 2012); the contribution
is to show that it is violated by the standard construction of graph SDMs from time-stamped
occurrence data, to quantify how much it inflates evaluation (including under spatial-block CV),
to give a formal sufficient rule (time-ordered edges) with a proof that it holds over several
message-passing layers, and a simple diagnostic.
