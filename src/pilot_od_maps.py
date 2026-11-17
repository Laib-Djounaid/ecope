"""Risk maps for pilot v4 (D11): out-of-fold development predictions, shown as within-map
percentile ranks so that models with different score scales are comparable.
Usage: python src/pilot_od_maps.py --tag v4_od_primary"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

PANELS = [("log_effort", "Independent effort (log1p non-target Fungi, prior)"),
          ("oof_G_lgbm", "G_lgbm: P(reported)"),
          ("oof_gat_concat_P_geo", "gat_concat_P_geo: P(reported)"),
          ("oof_presence_od_cloglog", "od_cloglog: P(present) (presence branch)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = ROOT / "reports" / "pilot_runs" / a.tag
    o = pd.read_parquet(out / "pilot_oof.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), PANELS):
        r = o[col].rank(pct=True)
        idx = r.sort_values().index                       # high ranks drawn on top
        sc = ax.scatter(o.loc[idx, "lon"], o.loc[idx, "lat"], c=r[idx], s=2, cmap="Blues",
                        vmin=0, vmax=1, linewidths=0)
        pos = o["y"] == 1
        ax.scatter(o.loc[pos, "lon"], o.loc[pos, "lat"], s=4, facecolors="none",
                   edgecolors="#c0392b", linewidths=0.4, label="observed positive (N = 5)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbbbbb")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.colorbar(sc, ax=axes, shrink=0.6, label="percentile rank within map")
    fig.suptitle(f"Pilot {a.tag}: development out-of-fold predictions (spatial CV, N = 5, seed 0)",
                 fontsize=11)
    fig.savefig(out / "risk_maps.png", dpi=150)
    print(out / "risk_maps.png")


if __name__ == "__main__":
    main()
