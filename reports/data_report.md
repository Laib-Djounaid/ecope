# Step 1 - Data report

Main setting: buffer **50 km**, GBIF variant **`unc-keep_bor-no-ms`** (missing uncertainty kept; all basisOfRecord). All numbers below are produced by `src/01_build_dataset.py` + `src/01b_data_report.py` from `config.yaml`.

## 1. sPlotOpen

- Plots in header / CWM-CWV tables: **95,104** / 95,104 (all matched one-to-one).
- Plots without a parseable survey date: **15,019** (excluded from labelled sets).
- Survey years span 1888-2015; plots surveyed before 1999: 35,313 (44.1% of dated plots).
- Eligible plots per horizon (window (t, t+N] inside GBIF 2000-2025, i.e. t >= 1999 and t + N <= 2025): N=5: **44,772**, N=7: **44,772**, N=10: **44,772**.
  Because the latest sPlotOpen survey is 2015, the upper constraint never binds and the eligible set is identical for all three horizons.
- Eligible plots: 41,132 unique coordinates, 93 source datasets, max location uncertainty 2,500 m.

Missing values among eligible plots (columns with any missing):

|                        |   fraction_missing |
|:-----------------------|-------------------:|
| Leaffreshmass_CWV      |              0.042 |
| Stem_cond_dens_CWV     |              0.042 |
| LeafArea_CWV           |              0.042 |
| StemDens_CWV           |              0.042 |
| Disp_unit_leng_CWV     |              0.042 |
| Wood_vessel_length_CWV |              0.042 |
| SeedMass_CWV           |              0.042 |
| PlantHeight_CWV        |              0.042 |
| Seed_num_rep_unit_CWV  |              0.042 |
| Leaf_delta_15N_CWV     |              0.042 |
| LeafNPratio_CWV        |              0.042 |
| LeafNperArea_CWV       |              0.042 |
| LDMC_CWV               |              0.042 |
| Seed_length_CWV        |              0.042 |
| LeafC_perdrymass_CWV   |              0.042 |
| SLA_CWV                |              0.042 |
| LeafP_CWV              |              0.042 |
| LeafN_CWV              |              0.042 |
| StemDens_CWM           |              0.006 |
| LeafArea_CWM           |              0.006 |
| LeafC_perdrymass_CWM   |              0.006 |
| SLA_CWM                |              0.006 |
| Disp_unit_leng_CWM     |              0.006 |
| Wood_vessel_length_CWM |              0.006 |
| Seed_num_rep_unit_CWM  |              0.006 |
| Leaf_delta_15N_CWM     |              0.006 |
| Leaffreshmass_CWM      |              0.006 |
| Stem_cond_dens_CWM     |              0.006 |
| SeedMass_CWM           |              0.006 |
| PlantHeight_CWM        |              0.006 |
| LDMC_CWM               |              0.006 |
| Seed_length_CWM        |              0.006 |
| LeafP_CWM              |              0.006 |
| LeafN_CWM              |              0.006 |
| LeafNperArea_CWM       |              0.006 |
| LeafNPratio_CWM        |              0.006 |

## 2. GBIF download (DOI 10.15468/dl.ps65sp)

- Raw records: **22,129**, species: **27**, years 2000-2025.
- coordinateUncertaintyInMeters missing for 70.7% of raw records.
- Filters, in order: species resolved -> occurrenceStatus = PRESENT -> year in [2000, 2025] -> valid coordinates (not 0,0) -> uncertainty <= 10 km (missing kept / dropped) -> basisOfRecord variant -> de-duplication (species, lat/lon rounded to 0.01 deg, year).

Total records after each filter, per variant:

|                                      |   unc-keep_bor-all |   unc-keep_bor-no-ms |   unc-drop_bor-all |   unc-drop_bor-no-ms |
|:-------------------------------------|-------------------:|---------------------:|-------------------:|---------------------:|
| 0_raw                                |              22129 |                22129 |              22129 |                22129 |
| 1_species_resolved                   |              22129 |                22129 |              22129 |                22129 |
| 2_status_present                     |              21674 |                21674 |              21674 |                21674 |
| 3_valid_year                         |              21674 |                21674 |              21674 |                21674 |
| 4_valid_coords                       |              21674 |                21674 |              21674 |                21674 |
| 5_uncertainty_le_10000m_missing-keep |              21468 |                21468 |                    |                      |
| 6_basis_of_record_all                |              21468 |                      |               6274 |                      |
| 7_dedup_species_coord0.01_year       |              11475 |                 8276 |               5553 |                 5538 |
| 6_basis_of_record_no-ms              |                    |                 9926 |                    |                 6256 |
| 5_uncertainty_le_10000m_missing-drop |                    |                      |               6274 |                 6274 |

Per-species counts, main variant (full table: `reports/tables_step1/gbif_filter_counts_main.csv`):

| species                       |     raw |   present |   valid year+coords |   unc<=10km |   dedup |   kept_% |
|:------------------------------|--------:|----------:|--------------------:|------------:|--------:|---------:|
| Alternaria solani             |    77.0 |      77.0 |                77.0 |        76.0 |    49.0 |     63.6 |
| Blumeria graminis             |  1730.0 |    1730.0 |              1730.0 |      1725.0 |   614.0 |     35.5 |
| Botrytis cinerea              |  1280.0 |    1050.0 |              1050.0 |      1027.0 |   627.0 |     49.0 |
| Cercospora beticola           |    88.0 |      88.0 |                88.0 |        88.0 |    59.0 |     67.0 |
| Cochliobolus heterostrophus   |    67.0 |      67.0 |                67.0 |        67.0 |     4.0 |      6.0 |
| Colletotrichum lindemuthianum |    78.0 |      78.0 |                78.0 |        78.0 |     3.0 |      3.8 |
| Erwinia amylovora             |    69.0 |      69.0 |                69.0 |        64.0 |    43.0 |     62.3 |
| Erysiphe necator              |   519.0 |     519.0 |               519.0 |       494.0 |   431.0 |     83.0 |
| Fusarium graminearum          |  1583.0 |    1583.0 |              1583.0 |      1582.0 |   209.0 |     13.2 |
| Fusarium oxysporum            |  1625.0 |    1622.0 |              1622.0 |      1618.0 |   189.0 |     11.6 |
| Hemileia vastatrix            |    83.0 |      83.0 |                83.0 |        80.0 |    67.0 |     80.7 |
| Peronospora farinosa          |   471.0 |     471.0 |               471.0 |       466.0 |   337.0 |     71.5 |
| Phakopsora pachyrhizi         |    99.0 |      99.0 |                99.0 |        97.0 |    65.0 |     65.7 |
| Phytophthora infestans        |   682.0 |     682.0 |               682.0 |       677.0 |   190.0 |     27.9 |
| Plasmodiophora brassicae      |   127.0 |     127.0 |               127.0 |       126.0 |    24.0 |     18.9 |
| Plasmopara viticola           |   423.0 |     423.0 |               423.0 |       409.0 |   271.0 |     64.1 |
| Pseudocercospora fijiensis    |    14.0 |      14.0 |                14.0 |        12.0 |     2.0 |     14.3 |
| Pseudomonas syringae          |  2041.0 |    2041.0 |              2041.0 |      2004.0 |  1376.0 |     67.4 |
| Puccinia graminis             |  1982.0 |    1760.0 |              1760.0 |      1717.0 |  1369.0 |     69.1 |
| Puccinia recondita            |    74.0 |      74.0 |                74.0 |        72.0 |    25.0 |     33.8 |
| Puccinia striiformis          |   453.0 |     453.0 |               453.0 |       446.0 |   220.0 |     48.6 |
| Pyricularia oryzae            |  1088.0 |    1088.0 |              1088.0 |      1088.0 |    34.0 |      3.1 |
| Rhizoctonia solani            |  4782.0 |    4782.0 |              4782.0 |      4782.0 |    93.0 |      1.9 |
| Sclerotinia sclerotiorum      |   452.0 |     452.0 |               452.0 |       449.0 |   246.0 |     54.4 |
| Ustilago maydis               |  1383.0 |    1383.0 |              1383.0 |      1383.0 |  1108.0 |     80.1 |
| Venturia inaequalis           |   797.0 |     797.0 |               797.0 |       779.0 |   620.0 |     77.8 |
| Xanthomonas oryzae            |    62.0 |      62.0 |                62.0 |        62.0 |     1.0 |      1.6 |
| TOTAL                         | 22129.0 |   21674.0 |             21674.0 |     21468.0 |  8276.0 |     37.4 |

basisOfRecord (raw vs. filtered main):

| basisOfRecord       |   raw |   filtered_main |
|:--------------------|------:|----------------:|
| HUMAN_OBSERVATION   |  7148 |            6143 |
| LIVING_SPECIMEN     |   382 |             196 |
| MACHINE_OBSERVATION |    42 |              37 |
| MATERIAL_CITATION   |    74 |              67 |
| MATERIAL_SAMPLE     | 12001 |               0 |
| OBSERVATION         |    14 |               9 |
| OCCURRENCE          |     4 |               4 |
| PRESERVED_SPECIMEN  |  2464 |            1820 |

Most MATERIAL_SAMPLE records come from environmental-DNA datasets (Australian Microbiome 18S soil samples, INSDC (host-organism) sequences, Global soil organisms, Global Spore Sampling Project, a marine fungal eDNA time series). These are detections of the organism, not reports of disease.

## 3. Label prevalence

Prevalence among the 44,772 eligible plots, all variants and buffers:

|                             |   N=5 %pos |   N=5 n_pos |   N=7 %pos |   N=7 n_pos |   N=10 %pos |   N=10 n_pos |   prior>0 % |
|:----------------------------|-----------:|------------:|-----------:|------------:|------------:|-------------:|------------:|
| ('unc-keep_bor-all', 25)    |       2.87 |     1283.00 |       3.88 |     1739.00 |        5.58 |      2499.00 |        2.54 |
| ('unc-keep_bor-all', 50)    |       6.67 |     2986.00 |       9.74 |     4363.00 |       12.89 |      5771.00 |        6.08 |
| ('unc-keep_bor-all', 100)   |      17.16 |     7685.00 |      22.82 |    10215.00 |       28.26 |     12654.00 |       12.60 |
| ('unc-keep_bor-no-ms', 25)  |       2.35 |     1052.00 |       3.06 |     1368.00 |        4.08 |      1825.00 |        2.43 |
| ('unc-keep_bor-no-ms', 50)  |       4.98 |     2231.00 |       6.72 |     3007.00 |        8.50 |      3804.00 |        5.72 |
| ('unc-keep_bor-no-ms', 100) |      11.83 |     5298.00 |      14.96 |     6697.00 |       18.10 |      8102.00 |       11.66 |
| ('unc-drop_bor-all', 25)    |       1.16 |      520.00 |       1.57 |      703.00 |        1.94 |       868.00 |        0.72 |
| ('unc-drop_bor-all', 50)    |       2.34 |     1048.00 |       3.21 |     1438.00 |        3.85 |      1725.00 |        2.38 |
| ('unc-drop_bor-all', 100)   |       4.96 |     2219.00 |       6.14 |     2751.00 |        7.19 |      3217.00 |        4.90 |
| ('unc-drop_bor-no-ms', 25)  |       1.16 |      520.00 |       1.57 |      703.00 |        1.94 |       868.00 |        0.72 |
| ('unc-drop_bor-no-ms', 50)  |       2.34 |     1048.00 |       3.21 |     1438.00 |        3.85 |      1725.00 |        2.38 |
| ('unc-drop_bor-no-ms', 100) |       4.96 |     2219.00 |       6.14 |     2751.00 |        7.19 |      3217.00 |        4.90 |

By continent (main setting):

| Continent     |   n_plots |   N=5 %pos |   N=7 %pos |   N=10 %pos |
|:--------------|----------:|-----------:|-----------:|------------:|
| North America |  14890.00 |       0.52 |       1.27 |        1.38 |
| Europe        |   8001.00 |      16.27 |      21.91 |       26.45 |
| Oceania       |   7373.00 |       8.68 |      10.13 |       13.89 |
| Asia          |   7259.00 |       2.91 |       3.77 |        5.48 |
| Africa        |   4366.00 |       0.00 |       0.87 |        0.92 |
| South America |   2883.00 |       0.00 |       0.21 |        0.73 |

By biome (main setting):

| Biome                           |   n_plots |   N=5 %pos |   N=7 %pos |   N=10 %pos |
|:--------------------------------|----------:|-----------:|-----------:|------------:|
| Temperate midlatitudes          |  10115.00 |      11.38 |      15.32 |       18.01 |
| Dry tropics and subtropics      |   8742.00 |       0.46 |       0.54 |        0.58 |
| Subtropics with year-round rain |   7031.00 |       7.65 |      10.04 |       14.11 |
| Dry midlatitudes                |   5633.00 |       0.80 |       1.24 |        1.24 |
| Tropics with summer rain        |   3831.00 |       4.70 |       5.12 |        6.32 |
| Boreal zone                     |   3168.00 |       0.16 |       0.16 |        0.92 |
| Subtropics with winter rain     |   2545.00 |       7.03 |      11.83 |       15.09 |
| Tropics with year-round rain    |   1701.00 |       5.00 |       6.00 |        7.76 |
| Alpine                          |   1152.00 |       0.69 |       2.60 |        6.60 |
| Polar and subpolar zone         |    854.00 |       0.00 |       0.00 |        0.70 |

By survey-year bin (main setting):

| year_bin   |   n_plots |   N=5 %pos |   N=7 %pos |   N=10 %pos |
|:-----------|----------:|-----------:|-----------:|------------:|
| 1999-2002  |  15730.00 |       3.96 |       5.96 |        7.64 |
| 2003-2006  |  11412.00 |       6.27 |       7.83 |        9.43 |
| 2007-2010  |  10999.00 |       5.83 |       7.59 |        9.40 |
| 2011-2015  |   6631.00 |       3.79 |       5.13 |        7.43 |

By source dataset: of 56 datasets with >= 100 eligible plots, 43% have **zero** positives at N=10 and 5% have >= 50% positives. Labels are therefore strongly clustered by dataset (and hence by region/survey campaign); random splits will leak this structure. Top 15 datasets:

| Dataset                      |   n_plots |   N=5 %pos |   N=10 %pos |
|:-----------------------------|----------:|-----------:|------------:|
| USA_VegBank                  |   12226.0 |        0.4 |         0.9 |
| Australia                    |    7216.0 |        8.9 |        14.2 |
| Siberia                      |    3627.0 |        1.0 |         1.0 |
| USA_CVS                      |    1659.0 |        1.7 |         5.8 |
| Brazil_forest                |    1541.0 |        0.0 |         0.0 |
| France_sophy                 |    1117.0 |       35.5 |        45.4 |
| Taiwan                       |     912.0 |        9.8 |        15.4 |
| BIOTA_South_Africa           |     902.0 |        0.0 |         4.2 |
| TAVA Apes                    |     884.0 |        0.0 |         0.0 |
| Nordic Vegetation Database A |     678.0 |        1.5 |         7.1 |
| BIOTA_Morocco                |     621.0 |        0.0 |         0.0 |
| Greece_nat2000               |     580.0 |        6.7 |         6.7 |
| Saudi Arabia                 |     577.0 |        0.0 |         0.0 |
| TAVA Sahel                   |     555.0 |        0.0 |         0.0 |
| Korea                        |     461.0 |       18.4 |        35.4 |

Note: with missing uncertainty dropped, the two basisOfRecord variants are almost identical, because nearly all MATERIAL_SAMPLE records lack an uncertainty value.

## 4. Prior presence and sampling effort

Prior presence (>= 1 record in 50 km strictly before the survey year): 5.7% of eligible plots.

|                            |   n_plots |   N=5 %pos |   N=7 %pos |   N=10 %pos |
|:---------------------------|----------:|-----------:|-----------:|------------:|
| no prior record in buffer  |  42210.00 |       2.67 |       4.02 |        5.54 |
| >=1 prior record in buffer |   2562.00 |      43.09 |      51.13 |       57.22 |

Note: the look-back window available for prior presence grows with survey year (0 years for t = 1999, 15 years for t = 2015), because the download starts in 2000. Prior presence is therefore itself partly a function of survey year.

How much of the label is explained by effort alone? Each count is used directly as a ranking score (no model fitting, no tuning) on all eligible plots:

|                                                                        |   spearman_rho |   ROC-AUC |   PR-AUC |   chance PR-AUC |
|:-----------------------------------------------------------------------|---------------:|----------:|---------:|----------------:|
| (5, 'prior records (before t) [legitimate]')                           |          0.435 |     0.732 |    0.307 |           0.050 |
| (5, 'distinct species before t [legitimate]')                          |          0.432 |     0.731 |    0.271 |           0.050 |
| (5, 'records outside label window (any other year) [diagnostic]')      |          0.430 |     0.876 |    0.504 |           0.050 |
| (5, 'all records, any year (incl. label window) [leaky, diagnostic]')  |          0.529 |     0.972 |    0.618 |           0.050 |
| (7, 'prior records (before t) [legitimate]')                           |          0.440 |     0.704 |    0.316 |           0.067 |
| (7, 'distinct species before t [legitimate]')                          |          0.438 |     0.703 |    0.294 |           0.067 |
| (7, 'records outside label window (any other year) [diagnostic]')      |          0.486 |     0.865 |    0.562 |           0.067 |
| (7, 'all records, any year (incl. label window) [leaky, diagnostic]')  |          0.613 |     0.975 |    0.718 |           0.067 |
| (10, 'prior records (before t) [legitimate]')                          |          0.433 |     0.680 |    0.315 |           0.085 |
| (10, 'distinct species before t [legitimate]')                         |          0.431 |     0.680 |    0.299 |           0.085 |
| (10, 'records outside label window (any other year) [diagnostic]')     |          0.472 |     0.808 |    0.513 |           0.085 |
| (10, 'all records, any year (incl. label window) [leaky, diagnostic]') |          0.680 |     0.974 |    0.743 |           0.085 |

The target-group effort proxy is built from the same 27 species that define the label, so 'all records, any year' contains the label records themselves and is reported only as an upper bound. 'Records outside the label window' excludes them but uses future years, so it is a diagnostic of surveillance intensity, not a usable predictor. Only the two 'before t' scores are legitimate features.
