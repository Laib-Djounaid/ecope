"""Figures of the manuscript main_asof.tex, drawn from the FROZEN outputs of the single hold-out run.

Reads only: reports/holdout/tests.csv, reports/holdout/pre_prediction.json and the stored hold-out
scores (results/holdout/**/refits/*.npz). No model is fitted and no test is recomputed.

Outputs (figures/):
  fig_holdout_tests.pdf  pre-registered differences with 95% block-bootstrap intervals
  fig_leak_audit.pdf     leakage diagnostics D1 / D2 computed before any hold-out prediction
  fig_risk_maps.pdf      hold-out risk ranks of EcoPE-GAT_as-of (pathogens S, N = 5; P1, N = 3)
"""
from __future__ import annotations

import importlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

from common import ROOT, load_config
from features import labels

H = importlib.import_module("holdout")
EV = importlib.import_module("15_holdout_eval")

OUT = ROOT / "figures"
REP = ROOT / "reports" / "holdout"
# reference palette (dataviz skill, light mode): categorical slots 1-2, text and grid tokens
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

DS = [("pathogens", "S", 5, "Pathogens (S)"), ("pathogens", "T", 5, "Pathogens (T)"),
      ("P1_halys_europe", "S", 3, "P1 H. halys EU"), ("P2_halys_namerica", "S", 3, "P2 H. halys NA"),
      ("P3_popillia_namerica", "S", 3, "P3 P. japonica NA")]


def _save(fig, name, **kw):
    fig.savefig(OUT / f"{name}.pdf", dpi=300, **kw)
    (OUT / "preview").mkdir(exist_ok=True)
    fig.savefig(OUT / "preview" / f"{name}.png", dpi=160, **kw)


def fig_tests():
    t = pd.read_csv(REP / "tests.csv")
    panels = [("H9", "R9", "all", "As-of minus\nedge deletion\n(H9 / R9)"),
              ("H8", "R8", "all", "As-of minus\nLGBM$_{nbr}$\n(H8 / R8)"),
              ("E5", "E5", "emergence", "As-of minus persistence,\nemergence sites\n(E5)"),
              ("H1", "R1", "all", "Leaky minus\nedge deletion\n(H1 / R1)")]
    fig, axs = plt.subplots(1, 4, figsize=(6.7, 2.7), sharey=True)
    ypos = np.arange(len(DS))[::-1]
    for ax, (hp, hr, sub, title) in zip(axs, panels):
        for y0, (ds, st, N, _) in zip(ypos, DS):
            h = hp if ds == "pathogens" else hr
            r = t[(t.dataset == ds) & (t.setting == st) & (t.N == N) & (t.subset == sub) & (t.hypothesis == h)]
            if h == "E5":
                r = r[r.a == "EcoPE-GAT_asof"]
            r = r.iloc[0]
            ax.plot([r.ci_lo, r.ci_hi], [y0, y0], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(r["diff"], y0, "o", ms=6, mec=BLUE, mew=1.5, mfc=BLUE if r.supported else "white", zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=8, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("PR-AUC difference")
    axs[0].set_yticks(ypos, [d[3] for d in DS])
    axs[0].tick_params(axis="y", length=0)
    fig.text(0.99, 0.01, "filled = supported (lower bound > 0); open = not supported",
             ha="right", va="bottom", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig_holdout_tests")
    plt.close(fig)


def fig_leak():
    pre = json.load(open(REP / "pre_prediction.json"))
    names = [d[3] for d in DS]
    d1 = [pre["pathogens"]["S"]["D1_receivers_in_test"]["D1_share_edges_later"],
          pre["pathogens"]["T"]["D1_receivers_in_test"]["D1_share_edges_later"]] + \
         [pre["pests"][c]["D1_receivers_in_test"]["D1_share_edges_later"] for c, *_ in DS[2:]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.7, 2.5), gridspec_kw={"width_ratios": [1, 1.1]})
    x = np.arange(len(names))
    a.bar(x, np.array(d1) * 100, width=0.6, color=BLUE)
    for xi, v in zip(x, d1):
        a.text(xi, v * 100 + 0.8, f"{v*100:.1f}%", ha="center", fontsize=7, color=INK)
    a.set_xticks(x, names, rotation=30, ha="right")
    a.set_ylabel("% of received edges")
    a.set_title("(a) D1: edges from a later-surveyed neighbour", fontsize=8, color=INK, loc="left")
    a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    # D2, N = 3, test nodes of the pest hold-outs (computed before prediction); pathogens: development graph
    cases = [c for c, *_ in DS[2:]]
    pos = [pre["pests"][c]["D2_N3_test"]["P_leaky_given_pos"] for c in cases] + [0.302]
    neg = [pre["pests"][c]["D2_N3_test"]["P_leaky_given_neg"] for c in cases] + [0.005]
    lab = [d[3] for d in DS[2:]] + ["Pathogens (dev, N=5)"]
    x = np.arange(len(lab)); w = 0.36
    b.bar(x - w / 2 - 0.01, np.array(pos) * 100, width=w, color=BLUE, label="positive sites")
    b.bar(x + w / 2 + 0.01, np.array(neg) * 100, width=w, color=ORANGE, label="negative sites")
    for xi, v in zip(x, pos):
        b.text(xi - w / 2, v * 100 + 0.8, f"{v*100:.0f}", ha="center", fontsize=7, color=INK)
    for xi, v in zip(x, neg):
        b.text(xi + w / 2, v * 100 + 0.8, f"{v*100:.1f}", ha="center", fontsize=7, color=INK)
    b.set_xticks(x, lab, rotation=30, ha="right")
    b.set_ylabel("% of sites")
    b.set_title("(b) D2: site receives a record from its window", fontsize=8, color=INK, loc="left")
    b.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    b.set_ylim(0, 72)
    b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig_leak_audit")
    plt.close(fig)


def _frame(ax, lon_bg, lat_bg, title, extent):
    ax.scatter(lon_bg, lat_bg, s=0.6, c=GRID, lw=0, rasterized=True)          # development sites
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(extent[2:]))))
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.tick_params(labelsize=7)


def _map(axp, axo, lon_bg, lat_bg, lon, lat, score, y, name, extent):
    """Left: hold-out risk rank of the model; right: observed outcome of the same sites."""
    _frame(axp, lon_bg, lat_bg, f"{name}: predicted risk rank", extent)
    r = rankdata(score) / len(score)
    o = np.argsort(r)
    sc = axp.scatter(lon[o], lat[o], s=2.2, c=r[o], cmap=RAMP, vmin=0, vmax=1, lw=0, rasterized=True)
    _frame(axo, lon_bg, lat_bg, f"{name}: observed new records", extent)
    axo.scatter(lon[y == 0], lat[y == 0], s=2.2, c=RAMP(0.15), lw=0, rasterized=True, label="none")
    axo.scatter(lon[y == 1], lat[y == 1], s=2.2, c=INK, lw=0, rasterized=True, label="new record")
    return sc


def fig_maps():
    cfg = load_config()
    e, ctxs = H.pathogen_setup(cfg)
    s, idx = EV.seed_mean(H.out_dir(cfg, "pathogens") / "refits", "EcoPE-GAT_asof", "S", 5, EV.PATH_SEEDS)
    assert (idx == ctxs["S"][2]).all()
    y = labels(e, cfg, 5)[idx]
    lat, lon = e["Latitude"].to_numpy(), e["Longitude"].to_numpy()
    dev = np.setdiff1d(np.arange(len(e)), idx)
    pe, pctx = H.pest_setup("P1_halys_europe")
    ps, pidx = EV.seed_mean(H.out_dir(cfg, "pests/P1_halys_europe") / "refits", "EcoPE-GAT_asof", "S", 3, EV.PEST_SEEDS)
    assert (pidx == pctx["S"][2]).all()
    py = pe["y3"].to_numpy()[pidx]
    plat, plon = pe["lat"].to_numpy(), pe["lon"].to_numpy()
    pdev = np.setdiff1d(np.arange(len(pe)), pidx)
    fig = plt.figure(figsize=(6.7, 5.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.25, 0.06], hspace=0.45, wspace=0.12)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    b1, b2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    sc = _map(a1, a2, lon[dev], lat[dev], lon[idx], lat[idx], s, y, "(a) Pathogens, N = 5", (-180, 180, -60, 80))
    _map(b1, b2, plon[pdev], plat[pdev], plon[pidx], plat[pidx], ps, py, "(b) P1 H. halys, N = 3", (-12, 35, 34, 62))
    for ax in (a2, b2):
        ax.set_yticklabels([])
    b2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, markerscale=3)
    cax = fig.add_subplot(gs[2, 0])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("risk rank of EcoPE-GAT as-of among hold-out sites (0 = lowest, 1 = highest)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    _save(fig, "fig_risk_maps", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["tests", "leak", "maps"]
    if "tests" in which:
        fig_tests()
    if "leak" in which:
        fig_leak()
    if "maps" in which:
        fig_maps()
    print("figures written to", OUT)
