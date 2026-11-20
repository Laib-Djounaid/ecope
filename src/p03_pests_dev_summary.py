"""D15 development summary per pest case (descriptive; the hold-out file is never read).

As in src/12_dev_summary.py: PR-AUC of the seed-mean score per spatial fold, fold mean; paired
differences (R1, R1b, R5, R8, R9 analogues) with 95% CIs from 2,000 block-bootstrap replicates (seed 0)
within the test folds, replicates without positives dropped and counted. Also, on development data:
- final model per case (D14 section 3 rule applied as D15 section 3 states: highest CV fold-mean among
  the causal models {G_lgbm_nbr, EcoPE-GAT_causal, EcoPE-GAT_asof} at the primary horizon N = 3,
  ties < 0.005 to the earlier one in that order); written to results/pests/<case>/final_model.json;
- R7 analogue: on emergence nodes (prior_any = 0) the final model minus the persistence heuristic
  (persist100, ties by n_prior), and minus the subset prevalence.
Outputs: reports/pests/<case>_dev_{summary,differences}.csv, reports/pests/<case>_final_model.json.

Usage: python src/p03_pests_dev_summary.py P1_halys_europe [...]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

from common import ROOT, load_config, p

MODELS = ["G_lgbm", "G_lgbm_nbr", "G_lgbm_nbr_leaky", "EcoPE-GAT_leaky", "EcoPE-GAT_causal", "EcoPE-GAT_asof",
          "EcoPE-GAT_asof_plus"]          # last one: D16 candidate, summarised only if its refits exist
CAUSAL = ["G_lgbm_nbr", "EcoPE-GAT_causal", "EcoPE-GAT_asof"]
PAIRS = [("EcoPE-GAT_leaky", "EcoPE-GAT_causal"), ("EcoPE-GAT_leaky", "G_lgbm_nbr"),
         ("EcoPE-GAT_causal", "G_lgbm_nbr"), ("G_lgbm_nbr_leaky", "G_lgbm_nbr"), ("G_lgbm_nbr", "G_lgbm"),
         ("EcoPE-GAT_asof", "G_lgbm_nbr"), ("EcoPE-GAT_asof", "EcoPE-GAT_causal"),
         ("EcoPE-GAT_asof_plus", "G_lgbm_nbr"), ("EcoPE-GAT_asof_plus", "G_lgbm_nbr_leaky"),
         ("EcoPE-GAT_asof_plus", "EcoPE-GAT_asof")]
SEEDS, HORIZONS, PRIMARY, N_BOOT = [0, 1, 2], [3, 5], 3, 2000


def boot_diff(fold_data, n_boot, rng):
    """fold_data: list of (y, s_a, s_b, block) per fold -> observed fold-mean diff, replicates, n valid."""
    obs = np.mean([AP(y, a) - AP(y, b) for y, a, b, _ in fold_data])
    mem = []
    for y, _, _, bl in fold_data:
        ub, inv = np.unique(bl, return_inverse=True)
        mem.append([np.flatnonzero(inv == u) for u in range(len(ub))])
    bs = []
    for _ in range(n_boot):
        v = []
        for (y, a, b, _), m in zip(fold_data, mem):
            ix = np.concatenate([m[u] for u in rng.integers(0, len(m), len(m))])
            if y[ix].sum() == 0:
                continue
            v.append(AP(y[ix], a[ix]) - AP(y[ix], b[ix]))
        bs.append(np.mean(v) if v else np.nan)
    bs = np.array(bs)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return obs, lo, hi, int(np.isfinite(bs).sum())


def summarise(case: str, cfg: dict) -> None:
    d = pd.read_parquet(ROOT / "data" / "processed" / "pests" / f"{case}_dev.parquet")
    assert not d["holdout"].any()
    ref = p(cfg, "results_dir") / "pests" / case / "refits"
    blocks = d["block_id"].to_numpy()
    S, IDX = {}, {}
    for m in MODELS:
        for k in range(5):
            for N in HORIZONS:
                fs = [ref / f"{m}_cv{k}_N{N}_s{s}.npz" for s in SEEDS]
                if all(f.exists() for f in fs):
                    z = [np.load(f) for f in fs]
                    S[(m, k, N)] = np.mean([q["test"] for q in z], 0)
                    IDX[(k, N)] = z[0]["test_idx"]
    rows = []
    for (m, k, N), s in S.items():
        ix = IDX[(k, N)]
        y = d[f"y{N}"].to_numpy()[ix].astype(int)
        rows.append({"model": m, "fold": k, "N": N, "pr_auc": AP(y, s), "prevalence": y.mean(), "n": len(ix)})
    tab = pd.DataFrame(rows)
    fm = tab.groupby(["model", "N"]).agg(pr_auc=("pr_auc", "mean"), n_folds=("pr_auc", "size")).reset_index()
    fm["fold"] = "mean"
    rep = ROOT / "reports" / "pests"
    pd.concat([tab, fm]).sort_values(["N", "model", "fold"], key=lambda c: c.astype(str)).to_csv(
        rep / f"{case}_dev_summary.csv", index=False, float_format="%.4f")

    rng = np.random.default_rng(0)
    diffs = []

    def fold_data(a_, b_, N, sub=None):
        out = []
        for k in range(5):
            ix = IDX[(k, N)]
            keep = np.ones(len(ix), bool) if sub is None else sub[ix]
            y = d[f"y{N}"].to_numpy()[ix].astype(int)[keep]
            sa = S[(a_, k, N)][keep] if isinstance(a_, str) else a_[ix][keep]
            sb = S[(b_, k, N)][keep] if isinstance(b_, str) else b_[ix][keep]
            out.append((y, sa, sb, blocks[ix][keep]))
        return out

    for N in HORIZONS:
        for a_, b_ in PAIRS:
            if all((m, k, N) in S for m in (a_, b_) for k in range(5)):
                o, lo, hi, nv = boot_diff(fold_data(a_, b_, N), N_BOOT, rng)
                diffs.append({"N": N, "subset": "all", "a": a_, "b": b_, "diff": o, "ci_lo": lo, "ci_hi": hi,
                              "n_valid_replicates": nv})
    # final model and the R7 analogue (development, descriptive)
    fmP = fm[fm.N == PRIMARY].set_index("model")["pr_auc"]
    final, info = None, {}
    if all(m in fmP for m in CAUSAL):
        best = max(fmP[m] for m in CAUSAL)
        final = next(m for m in CAUSAL if fmP[m] >= best - 0.005)
        info = {"case": case, "final_model": final, "rule": "D14 s3 / D15 s3: highest dev CV fold-mean among "
                f"{CAUSAL} at N={PRIMARY}, ties < 0.005 to the earlier", "fold_means": fmP[CAUSAL].to_dict()}
        json.dump(info, open(rep / f"{case}_final_model.json", "w"), indent=1)
        persist = d["persist100"].to_numpy() + 1e-6 * np.log1p(d["n_prior"].to_numpy())
        emer = d["prior_any"].to_numpy() == 0
        for N in HORIZONS:
            if all((final, k, N) in S for k in range(5)):
                o, lo, hi, nv = boot_diff(fold_data(final, persist, N, emer), N_BOOT, rng)
                diffs.append({"N": N, "subset": "emergence", "a": final, "b": "persistence", "diff": o,
                              "ci_lo": lo, "ci_hi": hi, "n_valid_replicates": nv})
                o, lo, hi, nv = boot_diff(fold_data(final, persist, N), N_BOOT, rng)
                diffs.append({"N": N, "subset": "all", "a": final, "b": "persistence", "diff": o,
                              "ci_lo": lo, "ci_hi": hi, "n_valid_replicates": nv})
                fdx = fold_data(final, persist, N, emer)
                diffs.append({"N": N, "subset": "emergence", "a": final, "b": "prevalence",
                              "diff": float(np.mean([AP(y, s) - y.mean() for y, s, _, _ in fdx])),
                              "n_pos": int(sum(y.sum() for y, *_ in fdx)), "n": int(sum(len(y) for y, *_ in fdx))})
    D = pd.DataFrame(diffs)
    D.to_csv(rep / f"{case}_dev_differences.csv", index=False, float_format="%.4f")
    print(f"=== {case}\n", fm.pivot(index="model", columns="N", values="pr_auc").round(3).to_string())
    print(D.round(4).to_string(index=False))
    print(info)


if __name__ == "__main__":
    cfg = load_config()
    for c in sys.argv[1:]:
        summarise(c, cfg)
