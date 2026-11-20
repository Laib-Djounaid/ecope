"""Development results summary for D13/D14 (descriptive; the sealed hold-out is never loaded).

Statistic (as pre-registered): PR-AUC of the seed-mean score. Spatial CV: fold-mean over cv0..cv4;
development transfer: train_high -> test_low. Paired differences with 95% CIs from 2,000 block
bootstrap replicates (seed 0), resampling spatial blocks within each test fold.
Outputs (tracked): reports/dev_results/summary.csv, differences.csv, params/*.json.

Usage: python src/12_dev_summary.py
"""
from __future__ import annotations

import importlib
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

from common import ROOT, load_config, p
from features import labels, load_modelling_table

d13 = importlib.import_module("09_d13")
PAIRS = [("EcoPE-GAT_leaky", "EcoPE-GAT_causal"), ("EcoPE-GAT_leaky", "G_lgbm_nbr"),
         ("EcoPE-GAT_causal", "G_lgbm_nbr"), ("EcoPE-GAT_causal", "GAT_noctx_causal"),
         ("G_lgbm_nbr_leaky", "G_lgbm_nbr"), ("G_lgbm_nbr", "G_lgbm"),
         ("EcoPE-GAT_asof", "G_lgbm_nbr"), ("EcoPE-GAT_asof", "EcoPE-GAT_causal"),
         # D16 (exploratory until prereg-D16)
         ("G_lgbm_nbr_asof", "G_lgbm_nbr"), ("G_lgbm_nbr_asof", "G_lgbm_nbr_leaky"),
         ("EcoPE-GAT_asof_plus", "EcoPE-GAT_causal"), ("EcoPE-GAT_asof_plus", "G_lgbm_nbr"),
         ("EcoPE-GAT_asof_plus", "G_lgbm_nbr_asof")]
EXTRA = ["EcoPE-GAT_asof", "EcoPE-GAT_asof_plus", "G_lgbm_nbr_asof"]


def main() -> None:
    cfg = load_config()
    dc = cfg["d13"]
    e = load_modelling_table(cfg)
    assert not e["holdout"].any()
    ctxs = d13.contexts(e, cfg)
    blocks = e["block_id"].to_numpy()
    res = p(cfg, "results_dir")
    S = {}                                               # (model, context, N) -> seed-mean test score
    for d in ["d13", "d14", "d16"]:
        for f in sorted((res / d / "refits").glob("*.npz")):
            m, rest = f.stem.rsplit("_N", 1)
            N, sd = rest.split("_s")
            model, c = next((mm, m[len(mm) + 1:]) for mm in sorted(d13.ALL_MODELS + EXTRA, key=len, reverse=True)
                            if m.startswith(mm + "_") and m[len(mm) + 1:] in d13.CONTEXTS)
            S.setdefault((model, c, int(N)), []).append(np.load(f)["test"])
    S = {k: np.mean(v, 0) for k, v in S.items() if len(v) == len(dc["seeds"])}
    Y = {N: labels(e, cfg, N) for N in dc["horizons"]}
    rows = []
    for (model, c, N), s in S.items():
        te = ctxs[c][2]
        rows.append({"model": model, "context": c, "N": N, "pr_auc": AP(Y[N][te], s), "prevalence": Y[N][te].mean()})
    tab = pd.DataFrame(rows)
    out = ROOT / "reports" / "dev_results"
    (out / "params").mkdir(parents=True, exist_ok=True)
    cv = tab[tab.context.str.startswith("cv")].groupby(["model", "N"]).agg(pr_auc=("pr_auc", "mean"), n_folds=("pr_auc", "size"))
    cv = cv.reset_index().assign(context="cv_fold_mean")
    summ = pd.concat([tab, cv], ignore_index=True).sort_values(["N", "context", "pr_auc"], ascending=[True, True, False])
    summ.to_csv(out / "summary.csv", index=False, float_format="%.4f")

    rng = np.random.default_rng(0)
    diffs = []
    for N in dc["horizons"]:
        for setting, cl in [("cv_fold_mean", [f"cv{k}" for k in range(5)]), ("dev_transfer", ["dev_transfer"])]:
            for a_, b_ in PAIRS:
                if not all((m, c, N) in S for m in (a_, b_) for c in cl):
                    continue
                parts = []
                for c in cl:
                    te = ctxs[c][2]
                    ub, inv = np.unique(blocks[te], return_inverse=True)
                    parts.append((te, ub, [np.flatnonzero(inv == u) for u in range(len(ub))]))
                obs = np.mean([AP(Y[N][te], S[(a_, c, N)]) - AP(Y[N][te], S[(b_, c, N)]) for c, (te, _, _) in zip(cl, parts)])
                boot = []
                for _ in range(dc["n_boot"]):
                    v = []
                    for c, (te, ub, members) in zip(cl, parts):
                        ix = np.concatenate([members[u] for u in rng.integers(0, len(ub), len(ub))])
                        y = Y[N][te][ix]
                        if y.sum() == 0:
                            continue
                        v.append(AP(y, S[(a_, c, N)][ix]) - AP(y, S[(b_, c, N)][ix]))
                    boot.append(np.mean(v) if v else np.nan)   # replicate without positives: dropped
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                diffs.append({"N": N, "setting": setting, "a": a_, "b": b_, "diff": obs, "ci_lo": lo, "ci_hi": hi,
                              "n_valid_replicates": int(np.isfinite(boot).sum())})
    pd.DataFrame(diffs).to_csv(out / "differences.csv", index=False, float_format="%.4f")
    for d in ["d13", "d14", "d16"]:
        for f in (res / d / "params").glob("*.json"):
            shutil.copy(f, out / "params" / f.name)
    print(summ[summ.N == 5].to_string(index=False))
    print(pd.DataFrame(diffs).query("N == 5").to_string(index=False))


if __name__ == "__main__":
    main()
