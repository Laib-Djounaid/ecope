"""Step 1 data report: tables (CSV + markdown) and figures from data/processed/.

Run after 01_build_dataset.py. Writes reports/data_report.md and reports/figures_step1/.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from common import label_col, load_config, p, variant_name

# Reference categorical palette (slots 1-3 validate all-pairs); neutrals for context.
C = {5: "#2a78d6", 7: "#eb6834", 10: "#1baf7a"}
POS, NEG, INK, MUTED, GRID = "#eb6834", "#b9b8b3", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 100,
})


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def main() -> None:
    cfg = load_config()
    proc = p(cfg, "processed_dir")
    rep = p(cfg, "reports_dir")
    fig_dir = rep / "figures_step1"
    tab_dir = rep / "tables_step1"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    H = cfg["labels"]["horizons"]
    B = cfg["spatial"]["buffer_km"]
    V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
    d = pd.read_parquet(proc / "plots.parquet")
    info = json.load(open(proc / "build_info.json"))
    counts = pd.read_csv(proc / "gbif_filter_counts.csv")
    raw = pd.read_csv(p(cfg, "gbif_download"), sep="\t", quoting=3, dtype=str,
                      keep_default_na=False, na_values=[""])
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    g = pd.read_parquet(proc / f"gbif_filtered_{V}.parquet")

    e = d[d["eligible_N5"]].copy()
    # With sPlotOpen ending in 2015 the eligible set is the same for every horizon.
    for N in H:
        assert (d[f"eligible_N{N}"] == d["eligible_N5"]).all(), "eligible sets differ by N"
    elig_n = {N: int(d[f"eligible_N{N}"].sum()) for N in H}
    y = {N: e[label_col(N, B, V)] for N in H}
    L = []  # markdown lines

    # ------------------------------------------------------------------ 1 sPlotOpen
    sy = d["survey_year"]
    L += ["# Step 1 - Data report", "",
          f"Main setting: buffer **{B} km**, GBIF variant **`{V}`** "
          "(missing uncertainty kept; all basisOfRecord). All numbers below are produced by "
          "`src/01_build_dataset.py` + `src/01b_data_report.py` from `config.yaml`.", "",
          "## 1. sPlotOpen", "",
          f"- Plots in header / CWM-CWV tables: **{info['splot']['n_header']:,}** / "
          f"{info['splot']['n_cwm']:,} (all matched one-to-one).",
          f"- Plots without a parseable survey date: **{info['splot']['n_no_survey_year']:,}** "
          "(excluded from labelled sets).",
          f"- Survey years span {info['splot']['survey_year_min']}-{info['splot']['survey_year_max']}; "
          f"plots surveyed before 1999: {int((sy < 1999).sum()):,} "
          f"({(sy < 1999).sum() / sy.notna().sum():.1%} of dated plots).",
          "- Eligible plots per horizon (window (t, t+N] inside GBIF 2000-2025, i.e. "
          "t >= 1999 and t + N <= 2025): " + ", ".join(f"N={N}: **{elig_n[N]:,}**" for N in H) + ".",
          "  Because the latest sPlotOpen survey is 2015, the upper constraint never binds and "
          "the eligible set is identical for all three horizons.",
          f"- Eligible plots: {e[['Latitude', 'Longitude']].drop_duplicates().shape[0]:,} unique "
          f"coordinates, {e['Dataset'].nunique()} source datasets, "
          f"max location uncertainty {e['Location_uncertainty'].max():,.0f} m.", ""]

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    yrs = sy.dropna().astype(int)
    ax.hist(yrs, bins=np.arange(yrs.min(), yrs.max() + 2) - 0.5, color=NEG, edgecolor="white",
            linewidth=0.3)
    ax.hist(yrs[yrs >= 1999], bins=np.arange(1999, yrs.max() + 2) - 0.5, color=C[5],
            edgecolor="white", linewidth=0.3, label="eligible (1999-2015)")
    ax.set_xlabel("Survey year"); ax.set_ylabel("Plots"); ax.legend(frameon=False)
    ax.set_title("sPlotOpen plots by survey year", loc="left", fontsize=10)
    save(fig, fig_dir, "fig_s1_survey_years")

    # trait missingness (eligible)
    tcols = [f"{t}_{s}" for s in ("CWM", "CWV") for t in cfg["features"]["traits"]]
    miss = e[tcols + cfg["features"]["plot_attrs"] + cfg["features"]["env"]].isna().mean()
    miss = miss[miss > 0].sort_values(ascending=False)
    L += ["Missing values among eligible plots (columns with any missing):", "",
          md(miss.to_frame("fraction_missing"), ".3f"), ""]

    # ------------------------------------------------------------------ 2 GBIF
    L += ["## 2. GBIF download (DOI 10.15468/dl.ps65sp)", "",
          f"- Raw records: **{info['gbif_raw_records']:,}**, species: **{info['gbif_raw_species']}**, "
          f"years {int(raw['year'].min())}-{int(raw['year'].max())}.",
          f"- coordinateUncertaintyInMeters missing for "
          f"{raw['coordinateUncertaintyInMeters'].isna().mean():.1%} of raw records.",
          "- Filters, in order: species resolved -> occurrenceStatus = PRESENT -> year in "
          "[2000, 2025] -> valid coordinates (not 0,0) -> uncertainty <= 10 km "
          "(missing kept / dropped) -> basisOfRecord variant -> de-duplication "
          "(species, lat/lon rounded to 0.01 deg, year).", ""]
    tot = counts[counts["species"] == "TOTAL"].set_index("variant").drop(columns="species")
    tot = tot.T.dropna(how="all")
    L += ["Total records after each filter, per variant:", "", md(tot.fillna(-1).astype(int)
          .replace(-1, ""), ".0f"), ""]
    main_c = counts[counts["variant"] == V].set_index("species").drop(columns="variant")
    main_c = main_c.dropna(axis=1, how="all").astype(int)
    main_c.to_csv(tab_dir / "gbif_filter_counts_main.csv")
    pick = {"raw": "0_raw", "present": "2_status_present", "valid year+coords": "4_valid_coords",
            "unc<=10km": next(c for c in main_c.columns if c.startswith("5_")),
            "dedup": "7_dedup_species_coord0.01_year"}
    short = pd.DataFrame({k: main_c[c] for k, c in pick.items()})
    short["kept_%"] = 100 * short["dedup"] / short["raw"]
    L += [f"Per-species counts, main variant (full table: `reports/tables_step1/"
          "gbif_filter_counts_main.csv`):", "", md(short, ".1f"), ""]

    bor = raw.groupby("basisOfRecord").size().rename("raw").to_frame()
    bor["filtered_main"] = g.groupby("basisOfRecord").size()
    L += ["basisOfRecord (raw vs. filtered main):", "", md(bor.fillna(0).astype(int), ".0f"), "",
          "Most MATERIAL_SAMPLE records come from environmental-DNA datasets (Australian "
          "Microbiome 18S soil samples, INSDC (host-organism) sequences, Global soil organisms, "
          "Global Spore Sampling Project, a marine fungal eDNA time series). These are "
          "detections of the organism, not reports of disease.", ""]

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    gy = g.assign(ms=g["basisOfRecord"].eq("MATERIAL_SAMPLE")).groupby(["year", "ms"]).size()\
        .unstack(fill_value=0)
    ax.bar(gy.index, gy.get(False, 0), color=C[5], label="other basisOfRecord", width=0.8)
    ax.bar(gy.index, gy.get(True, 0), bottom=gy.get(False, 0), color=C[7],
           label="MATERIAL_SAMPLE (mostly eDNA)", width=0.8)
    ax.set_xlabel("Record year"); ax.set_ylabel("Filtered records")
    ax.set_title("Filtered GBIF records per year (main variant)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    save(fig, fig_dir, "fig_s2_gbif_records_per_year")

    # maps
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(e["Longitude"], e["Latitude"], s=0.3, c=NEG, lw=0, label="eligible plots")
    ax.scatter(g["decimalLongitude"], g["decimalLatitude"], s=1.2, c=C[5], lw=0,
               label="GBIF records (filtered)")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); ax.set_aspect("equal")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(frameon=False, markerscale=8, loc="lower left")
    ax.set_title("Eligible plots and filtered pathogen records", loc="left", fontsize=10)
    save(fig, fig_dir, "fig_s3_map_plots_records")

    fig, axes = plt.subplots(3, 1, figsize=(7, 10))
    for ax, N in zip(axes, H):
        yy = y[N].to_numpy() == 1
        ax.scatter(e["Longitude"][~yy], e["Latitude"][~yy], s=0.3, c=NEG, lw=0, label="y = 0")
        ax.scatter(e["Longitude"][yy], e["Latitude"][yy], s=1.5, c=POS, lw=0, label="y = 1")
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); ax.set_aspect("equal")
        ax.set_title(f"Labels, N = {N} years, {B} km ({yy.mean():.1%} positive)", loc="left",
                     fontsize=10)
        ax.legend(frameon=False, markerscale=8, loc="lower left")
    save(fig, fig_dir, "fig_s4_map_labels")

    # ------------------------------------------------------------------ 3 labels
    L += ["## 3. Label prevalence", ""]
    rows = []
    for unc in cfg["gbif"]["missing_uncertainty_sensitivity"]:
        for bo in cfg["gbif"]["basis_of_record_sensitivity"]:
            v = variant_name(unc, bo)
            for b in cfg["spatial"]["buffer_km_sensitivity"]:
                r = {"variant": v, "buffer_km": b}
                for N in H:
                    r[f"N={N} %pos"] = 100 * e[label_col(N, b, v)].mean()
                    r[f"N={N} n_pos"] = int(e[label_col(N, b, v)].sum())
                r["prior>0 %"] = 100 * e[f"prior_any_b{b}_{v}"].mean()
                rows.append(r)
    prev = pd.DataFrame(rows)
    prev.to_csv(tab_dir / "prevalence_variants.csv", index=False)
    L += [f"Prevalence among the {len(e):,} eligible plots, all variants and buffers:", "",
          md(prev.set_index(["variant", "buffer_km"]), ".2f"), ""]

    def by(col, order=None):
        t = pd.DataFrame({f"N={N} %pos": 100 * y[N].groupby(e[col]).mean() for N in H})
        t.insert(0, "n_plots", e.groupby(col).size())
        return t.loc[order] if order is not None else t.sort_values("n_plots", ascending=False)

    bins = cfg["report"]["survey_year_bins"]
    e["year_bin"] = pd.cut(e["survey_year"].astype(float), bins=bins, right=False,
                           labels=[f"{a}-{b - 1}" for a, b in zip(bins[:-1], bins[1:])])
    t_cont, t_biome, t_year = by("Continent"), by("Biome"), by("year_bin")
    t_year = t_year.sort_index()
    for name, t in [("continent", t_cont), ("biome", t_biome), ("survey-year bin", t_year)]:
        t.to_csv(tab_dir / f"prevalence_by_{name.split()[0].replace('-', '_')}.csv")
        L += [f"By {name} (main setting):", "", md(t, ".2f"), ""]

    # per-year line chart
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for N in H:
        grp = y[N].groupby(e["survey_year"])
        s = 100 * grp.mean()[grp.size() >= 100]     # years with < 100 plots omitted
        ax.plot(s.index.astype(int), s.values, color=C[N], lw=2, marker="o", ms=4, label=f"N = {N}")
    ax.set_xlabel("Survey year (years with < 100 eligible plots omitted)"); ax.set_ylabel("% positive")
    ax.set_title(f"Label prevalence by survey year ({B} km)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    save(fig, fig_dir, "fig_s5_prevalence_by_year")

    for name, t, fn in [("continent", t_cont, "fig_s6_prevalence_by_continent"),
                        ("biome", t_biome, "fig_s7_prevalence_by_biome")]:
        fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(t) + 1))
        idx = np.arange(len(t))
        hgt = 0.26
        for i, N in enumerate(H):
            ax.barh(idx + (i - 1) * hgt, t[f"N={N} %pos"], height=hgt - 0.03, color=C[N],
                    label=f"N = {N}")
        ax.set_yticks(idx, [f"{k} (n={n:,})" for k, n in zip(t.index, t["n_plots"])])
        ax.invert_yaxis(); ax.set_xlabel("% positive")
        ax.set_title(f"Label prevalence by {name} ({B} km)", loc="left", fontsize=10)
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="y", visible=False)
        save(fig, fig_dir, fn)

    # per dataset (fingerprint confound)
    t_ds = pd.DataFrame({"n_plots": e.groupby("Dataset").size(),
                         "N=5 %pos": 100 * y[5].groupby(e["Dataset"]).mean(),
                         "N=10 %pos": 100 * y[10].groupby(e["Dataset"]).mean()})
    t_ds = t_ds.sort_values("n_plots", ascending=False)
    t_ds.to_csv(tab_dir / "prevalence_by_dataset.csv")
    big = t_ds[t_ds["n_plots"] >= 100]
    share0 = (big["N=10 %pos"] == 0).mean()
    share_hi = (big["N=10 %pos"] >= 50).mean()
    L += [f"By source dataset: of {len(big)} datasets with >= 100 eligible plots, "
          f"{share0:.0%} have **zero** positives at N=10 and {share_hi:.0%} have >= 50% positives. "
          "Labels are therefore strongly clustered by dataset (and hence by region/survey "
          "campaign); random splits will leak this structure. Top 15 datasets:", "",
          md(t_ds.head(15), ".1f"), ""]
    L += ["Note: with missing uncertainty dropped, the two basisOfRecord variants are almost "
          "identical, because nearly all MATERIAL_SAMPLE records lack an uncertainty value.", ""]

    # ------------------------------------------------------------------ 4 prior + effort
    L += ["## 4. Prior presence and sampling effort", ""]
    pr = e[f"prior_any_b{B}_{V}"]
    ct = pd.DataFrame({f"N={N} %pos": 100 * y[N].groupby(pr).mean() for N in H})
    ct.insert(0, "n_plots", pr.value_counts())
    ct.index = ["no prior record in buffer", ">=1 prior record in buffer"]
    L += [f"Prior presence (>= 1 record in {B} km strictly before the survey year): "
          f"{pr.mean():.1%} of eligible plots.", "", md(ct, ".2f"), "",
          "Note: the look-back window available for prior presence grows with survey year "
          "(0 years for t = 1999, 15 years for t = 2015), because the download starts in 2000. "
          "Prior presence is therefore itself partly a function of survey year.", ""]

    eff = {
        "prior records (before t) [legitimate]": f"n_prior_b{B}_{V}",
        "distinct species before t [legitimate]": f"nsp_prior_b{B}_{V}",
        "records outside label window (any other year) [diagnostic]": None,
        "all records, any year (incl. label window) [leaky, diagnostic]": f"n_all_b{B}_{V}",
    }
    rows = []
    for N in H:
        for nm, col in eff.items():
            s = e[col] if col else e[f"n_outwin_N{N}_b{B}_{V}"]
            rho = spearmanr(s, y[N]).statistic
            rows.append({"horizon": N, "score": nm, "spearman_rho": rho,
                         "ROC-AUC": roc_auc_score(y[N], s), "PR-AUC": average_precision_score(y[N], s),
                         "chance PR-AUC": y[N].mean()})
    te = pd.DataFrame(rows)
    te.to_csv(tab_dir / "effort_only_scores.csv", index=False)
    L += ["How much of the label is explained by effort alone? Each count is used directly "
          "as a ranking score (no model fitting, no tuning) on all eligible plots:", "",
          md(te.set_index(["horizon", "score"]), ".3f"), "",
          "The target-group effort proxy is built from the same 27 species that define the "
          "label, so 'all records, any year' contains the label records themselves and is "
          "reported only as an upper bound. 'Records outside the label window' excludes them "
          "but uses future years, so it is a diagnostic of surveillance intensity, not a "
          "usable predictor. Only the two 'before t' scores are legitimate features.", ""]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    s = e[f"n_outwin_N5_b{B}_{V}"]
    qb = pd.cut(s, bins=[-1, 0, 2, 5, 10, 25, 50, 100, 1e9],
                labels=["0", "1-2", "3-5", "6-10", "11-25", "26-50", "51-100", ">100"])
    for N in H:
        t = 100 * y[N].groupby(qb, observed=False).mean()
        ax.plot(range(len(t)), t.values, color=C[N], lw=2, marker="o", ms=4, label=f"N = {N}")
    ax.set_xticks(range(len(t)), t.index)
    ax.set_xlabel(f"Target-group records within {B} km outside the N = 5 label window")
    ax.set_ylabel("% positive")
    ax.set_title("Label prevalence rises with surveillance intensity", loc="left", fontsize=10)
    ax.legend(frameon=False)
    save(fig, fig_dir, "fig_s8_prevalence_vs_effort")

    (rep / "data_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
