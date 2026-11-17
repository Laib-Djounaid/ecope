# Literature: occupancy-detection, sampling bias, GNNs for SDM (positioning of EcoPE-OD-GNN)

All references verified on 2026-11-17 (Crossref metadata for DOIs, arxiv.org for arXiv ids,
proceedings / journal pages otherwise). One candidate DOI was rejected (it resolved to an
unrelated paper) and one corrected (Knape & Korner-Nievergelt is in MEE, not Ecology).

## 1. Occupancy-detection models

- **MacKenzie et al. (2002)** introduced site-occupancy models that separate the probability that a
  site is occupied (psi) from the probability that the species is detected given occupancy (p),
  using **repeated surveys** of the same sites. Ecology 83, 2248-2255.
  https://doi.org/10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2
- **Kellner & Swihart (2014)** reviewed 537 studies: imperfect detection is common and ignoring it
  biases occupancy estimates. PLoS ONE 9, e111436. https://doi.org/10.1371/journal.pone.0111436
- **Guillera-Arroita (2017)** reviews SDMs under imperfect detection, including the
  identifiability conditions and the data designs that make psi and p separable.
  Ecography 40, 281-295. https://doi.org/10.1111/ecog.02445
- **Single-visit identifiability.** Lele, Moreno & Bayne (2012) show that psi and p can be
  identified from a **single survey** only if they depend on **different covariates**, at least
  one continuous, with a correctly specified (e.g. logistic) link. J. Plant Ecol. 5, 22-31.
  https://doi.org/10.1093/jpe/rtr042
  **Knape & Korner-Nievergelt (2015)** show that such non-replicated estimates **rely on strong,
  untestable assumptions** (functional form, covariate separation) and can be badly biased when
  they fail. Methods Ecol. Evol. 6, 298-306. https://doi.org/10.1111/2041-210X.12329
  -> **Directly relevant**: our data have one label per plot (no repeated visits), so the
  presence/detection split is identified only through covariate separation and functional-form
  constraints. Any "presence" output must be presented as a model-based quantity, not an
  estimate of true occurrence.

## 2. Sampling bias in presence-only data and SDMs

- **Phillips et al. (2009)**: target-group background (records of related taxa collected with
  the same methods) as a proxy for sampling bias. Ecol. Appl. 19, 181-197.
  https://doi.org/10.1890/07-2153.1  (already cited)
- **Warton, Renner & Ramp (2013)**: model observer bias explicitly as covariates in a point-process
  model, then predict with the bias covariates set to a common value ("bias-corrected"
  predictions). PLoS ONE 8, e79168. https://doi.org/10.1371/journal.pone.0079168
  -> Closest analogue of our "risk maps corrected for effort".
- **Dorazio (2014)**: a model combining a point process for occurrence with a thinning
  (detection) process driven by survey covariates, for presence-only data.
  Global Ecol. Biogeogr. 23, 1472-1484. https://doi.org/10.1111/geb.12216
  -> The statistical template for presence x detection with effort covariates.
- **Fithian et al. (2015)**: share the bias (effort) component across many species to separate it
  from species' environmental responses. Methods Ecol. Evol. 6, 424-438.
  https://doi.org/10.1111/2041-210X.12242
- **Koshkina et al. (2017)**; **Isaac et al. (2020)**: integrated SDMs combining presence-only data
  (with a detection/bias sub-model) and structured survey data.
  Methods Ecol. Evol. 8, 420-430, https://doi.org/10.1111/2041-210X.12738 ;
  Trends Ecol. Evol. 35, 56-67, https://doi.org/10.1016/j.tree.2019.08.006
- **Johnston et al. (2021)**: effort covariates in community-science (eBird) models; predict at
  standardised effort. Divers. Distrib. 27, 1265-1277. https://doi.org/10.1111/ddi.13271
- **Beck et al. (2014)**: spatial bias in GBIF. (already cited)
- **Spatial evaluation**: Roberts et al. (2017), Valavi et al. (2019), Ploton et al. (2020)
  (already cited); Hijmans (2012) on spatial sorting bias in cross-validation, Ecology 93,
  679-688, https://doi.org/10.1890/11-0826.1

## 3. Neural and graph models for SDM / hierarchical ecology

- **Joseph (2020)**: "neural hierarchical models", i.e. neural networks inside the process and
  observation components of hierarchical ecological models, including occupancy. Ecol. Lett.
  23, 734-747. https://doi.org/10.1111/ele.13462
- **Seo et al. (2021)**: StatEcoNet, neural occupancy-detection models for species
  distributions. arXiv:2102.08534. https://arxiv.org/abs/2102.08534
- **Wu et al. (2025)** GNN-SDM; **Harrell et al. (2025)** heterogeneous GNN for SDM; **Klemmer
  et al. (2023)** PE-GNN (already cited).

## 4. Positioning of EcoPE-OD-GNN

Neural occupancy-detection models exist (Joseph 2020; Seo et al. 2021), and presence x detection
(thinning) models with effort covariates are established for presence-only data (Dorazio 2014;
Warton et al. 2013; Fithian et al. 2015). GNNs for SDM exist, but without an observation model.
What is new here, to our knowledge:
1. a **graph** model whose **presence** branch uses message passing on a geographic graph with an
   ecological positional encoding, combined with an explicit **detection** branch driven by an
   **independent target-group effort** layer;
2. applied to **plant-pathogen records** derived from GBIF, with a **pre-registered** evaluation of
   transfer from high- to low-effort regions and of the effort-dependence of the presence output.

What must be stated plainly (Lele et al. 2012; Knape & Korner-Nievergelt 2015): with a single
label per plot, the presence/detection split is identified only through covariate separation and
the functional form of the detection branch; the presence output is a model-based, not an
observed, quantity.
