"""Single confirmatory hold-out evaluation (pre-registrations D13, D14, D15, D16; tag `frozen-D16`).

Two steps, run once each, in this order:
  --pre    BEFORE any hold-out model is fitted: hold-out diagnostics that need no prediction (D1, isolated
           nodes in the causal graph, D2 for pests), positives in the emergence subsets (H7 / R7 / E5), and
           the pathogen final model (D14 section 3 rule on development results). -> reports/holdout/pre_prediction.json
  --final  AFTER the hold-out refits (`--holdout` mode of 09_d13, 11_d14 [--plus], 14_d16_lgbm_asof,
           p02_pests_models): every pre-registered test, once. -> reports/holdout/{scores,tests}.csv, verdicts.json

Statistics (D13 section 3): PR-AUC of the seed-mean score; 95% CI = percentile interval of 2,000 bootstrap
replicates resampling spatial blocks of the test set; replicates without positives dropped and counted;
"supported" = CI lower bound > 0. Each comparison uses its own generator seeded 0, so every comparison on a
given test set resamples the same blocks (paired) and the result does not depend on the order of tests.
Persistence heuristic (D14 H6): log1p(target records within 100 km before t_i), ties broken by the prior
count within 50 km (+1e-6 * log1p count), as on development data.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from sklearn.metrics import brier_score_loss

import holdout as H
from common import ROOT, load_config, p
from features import labels, raw_blocks
from graph_models import geo_knn_graph, time_causal
from leakage_diagnostic import d2
from neighbour_features import neighbour_block

import os
REP = Path(os.environ.get("ECOPE_HOLDOUT_REP", ROOT / "reports" / "holdout"))   # override only for the synthetic code test
PEST_CASES = ["P1_halys_europe", "P2_halys_namerica", "P3_popillia_namerica"]
PATH_N, PEST_N = [5, 7, 10], [3, 5]
PATH_SEEDS, PEST_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
PATH_MODELS = ["G_lgbm", "G_lgbm_nbr", "G_lgbm_nbr_leaky", "EcoPE-GAT_leaky", "EcoPE-GAT_causal",
               "GAT_noctx_causal", "EcoPE-GAT_asof", "EcoPE-GAT_asof_plus", "G_lgbm_nbr_asof"]
PEST_MODELS = ["G_lgbm", "G_lgbm_nbr", "G_lgbm_nbr_leaky", "EcoPE-GAT_leaky", "EcoPE-GAT_causal",
               "EcoPE-GAT_asof", "EcoPE-GAT_asof_plus", "G_lgbm_nbr_asof"]
PATH_FINAL_CANDIDATES = ["G_lgbm_nbr", "EcoPE-GAT_causal", "GAT_noctx_causal", "EcoPE-GAT_asof"]
N_BOOT = 2000


# ----------------------------------------------------------------------------- statistics
def boot(y, blocks, stat, n_boot=N_BOOT):
    """stat(ix) -> float on resampled rows ix. Returns observed value, CI, number of valid replicates."""
    obs = float(stat(np.arange(len(y))))
    ub, inv = np.unique(blocks, return_inverse=True)
    mem = [np.flatnonzero(inv == u) for u in range(len(ub))]
    rng = np.random.default_rng(0)
    bs = np.full(n_boot, np.nan)
    for r in range(n_boot):
        ix = np.concatenate([mem[u] for u in rng.integers(0, len(ub), len(ub))])
        if y[ix].sum() > 0:
            bs[r] = stat(ix)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return obs, float(lo), float(hi), int(np.isfinite(bs).sum())


def diff_test(y, blocks, a, b):
    return boot(y, blocks, lambda ix: AP(y[ix], a[ix]) - AP(y[ix], b[ix]))


def minus_prev_test(y, blocks, a):
    return boot(y, blocks, lambda ix: AP(y[ix], a[ix]) - y[ix].mean())


def persistence(prior100_log_n, prior50_log_n):
    return np.asarray(prior100_log_n, float) + 1e-6 * np.asarray(prior50_log_n, float)


def seed_mean(ref, model, setting, N, seeds):
    z = [np.load(ref / f"{model}_{setting}_N{N}_s{s}.npz") for s in seeds]
    idx = z[0]["test_idx"]
    assert all((q["test_idx"] == idx).all() for q in z)
    return np.mean([q["test"] for q in z], 0), idx


# ----------------------------------------------------------------------------- data
def pathogen_data(cfg):
    e, ctxs = H.pathogen_setup(cfg)
    blk = raw_blocks(e, cfg)
    nb = neighbour_block(e, blk, cfg, causal=True)
    pers = persistence(nb["prior100_log_n"], blk["prior"]["prior_log_n"])
    emer = blk["prior"]["prior_any"].to_numpy() == 0
    ei, ew = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), cfg["graph"]["k"],
                           cfg["spatial"]["buffer_km"], cfg["spatial"]["earth_radius_km"])
    t = e["survey_year"].to_numpy().astype(int)
    cei, _ = time_causal(ei, ew, t)
    isolated = np.bincount(cei[1].numpy(), minlength=len(e)) == 0
    Y = {N: labels(e, cfg, N) for N in PATH_N}
    return e, ctxs, Y, pers, emer, isolated, ei, t


def pest_data(case, cfg):
    p02 = importlib.import_module("p02_pests_models")
    pb = importlib.import_module("p01_build_pests")
    e, ctxs = H.pest_setup(case)
    pers = persistence(e["persist100"], np.log1p(e["n_prior"]))
    emer = e["prior_any"].to_numpy() == 0
    Y = {N: e[f"y{N}"].to_numpy() for N in PEST_N}
    R, B, K = cfg["spatial"]["earth_radius_km"], cfg["spatial"]["buffer_km"], cfg["graph"]["k"]
    ei, ew = geo_knn_graph(e[["lat", "lon"]].to_numpy(), K, B, R)
    t = e["t"].to_numpy().astype(int)
    tgt, _ = pb.CASES[case]
    return e, ctxs, Y, pers, emer, ei, ew, t, (p02, tgt, B, R)


def d1_test(ei, t, te):
    """D1 restricted to messages received by test nodes (the graph itself spans development + hold-out)."""
    src, dst = ei.numpy()
    k = np.isin(dst, te)
    later = t[src[k]] > t[dst[k]]
    deg = np.bincount(dst[k], minlength=len(t))[te]
    cdeg = deg - np.bincount(dst[k], weights=later, minlength=len(t))[te]
    share = np.where(deg > 0, np.bincount(dst[k], weights=later, minlength=len(t))[te] / np.maximum(deg, 1), np.nan)
    return {"edges_received": int(k.sum()), "D1_share_edges_later": float(later.mean()),
            "D1_median_node_share_later": float(np.nanmedian(share)),
            "test_nodes_isolated_in_causal_graph": int((cdeg == 0).sum()),
            "median_in_degree": float(np.median(deg)), "median_causal_in_degree": float(np.median(cdeg))}


# ----------------------------------------------------------------------------- step 1: before prediction
def pre(cfg):
    REP.mkdir(parents=True, exist_ok=True)
    out = {"note": "computed before any hold-out model was fitted; no prediction involved"}
    # pathogen final model (D14 section 3), from the development summary
    sm = pd.read_csv(ROOT / "reports" / "dev_results" / "summary.csv")
    fm = sm[(sm.context == "cv_fold_mean") & (sm.N == 5)].set_index("model")["pr_auc"]
    best = max(fm[m] for m in PATH_FINAL_CANDIDATES)
    final = next(m for m in PATH_FINAL_CANDIDATES if fm[m] >= best - 0.005)
    fmj = {"final_model": final, "rule": "D14 s3: highest dev spatial-CV fold-mean PR-AUC (N = 5) among "
           f"{PATH_FINAL_CANDIDATES}, ties < 0.005 to the earlier", "fold_means": fm[PATH_FINAL_CANDIDATES].to_dict()}
    json.dump(fmj, open(p(cfg, "results_dir") / "d14" / "final_model.json", "w"), indent=1)
    out["pathogens_final_model"] = fmj

    e, ctxs, Y, pers, emer, isolated, ei, t = pathogen_data(cfg)
    po = {}
    for s, (_, _, te) in ctxs.items():
        po[s] = {"test_plots": int(len(te)), "test_blocks": int(e["block_id"].iloc[te].nunique()),
                 "D1_receivers_in_test": d1_test(ei, t, te), "isolated_test_nodes_causal_graph": int(isolated[te].sum()),
                 **{f"pos_N{N}": int(Y[N][te].sum()) for N in PATH_N},
                 "emergence_subset": {"plots": int(emer[te].sum()),
                                      **{f"pos_N{N}": int(Y[N][te][emer[te]].sum()) for N in PATH_N}},
                 "non_isolated_subset": {"plots": int((~isolated[te]).sum()),
                                         **{f"pos_N{N}": int(Y[N][te][~isolated[te]].sum()) for N in PATH_N}}}
    out["pathogens"] = po
    out["pests"] = {}
    for case in PEST_CASES:
        e, ctxs, Y, pers, emer, ei, ew, t, (p02, tgt, B, R) = pest_data(case, cfg)
        te = ctxs["S"][2]
        yrs50 = p02.target_years(e, tgt, B, R)
        r = {"test_nodes": int(len(te)), "test_blocks": int(e["block_id"].iloc[te].nunique()),
             "D1_receivers_in_test": d1_test(ei, t, te)}
        for N in PEST_N:
            ok = np.isfinite(Y[N])
            teN = te[ok[te]]
            _, has = d2(ei.numpy(), t, np.nan_to_num(Y[N]).astype(int), yrs50, N)
            y_ = Y[N][teN].astype(int)
            r[f"D2_N{N}_test"] = {"P_leaky_given_pos": float(has[teN][y_ == 1].mean()),
                                  "P_leaky_given_neg": float(has[teN][y_ == 0].mean()),
                                  "PR_AUC_indicator_alone": float(AP(y_, has[teN])), "prevalence": float(y_.mean())}
            r[f"N{N}"] = {"scored_nodes": int(len(teN)), "pos": int(Y[N][teN].sum()),
                          "emergence_nodes": int(emer[teN].sum()), "emergence_pos": int(Y[N][teN][emer[teN]].sum())}
        r["final_model"] = json.load(open(ROOT / "reports" / "pests" / f"{case}_final_model.json"))["final_model"]
        out["pests"][case] = r
    json.dump(out, open(REP / "pre_prediction.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


# ----------------------------------------------------------------------------- step 2: the tests
def final(cfg):
    pre_ = json.load(open(REP / "pre_prediction.json"))
    rows_s, rows_t = [], []

    def add(ds, setting, N, hyp, a, b, res, subset="all"):
        o, lo, hi, nv = res
        rows_t.append({"dataset": ds, "setting": setting, "N": N, "subset": subset, "hypothesis": hyp, "a": a, "b": b,
                       "diff": o, "ci_lo": lo, "ci_hi": hi, "n_valid_replicates": nv, "supported": lo > 0})

    # ---------------- pathogens
    e, ctxs, Y, pers, emer, isolated, _, _ = pathogen_data(cfg)
    ref = H.out_dir(cfg, "pathogens") / "refits"
    fin = pre_["pathogens_final_model"]["final_model"]
    blocks = e["block_id"].to_numpy()
    for st, (_, _, te) in ctxs.items():
        for N in PATH_N:
            S = {}
            for m in PATH_MODELS:
                s, idx = seed_mean(ref, m, st, N, PATH_SEEDS)
                assert (idx == te).all()
                S[m] = s
            y, bl = Y[N][te], blocks[te]
            for m, s in S.items():
                rows_s.append({"dataset": "pathogens", "setting": st, "N": N, "model": m, "pr_auc": AP(y, s),
                               "prevalence": y.mean(), "brier": brier_score_loss(y, s), "n": len(y), "n_pos": int(y.sum())})
            rows_s.append({"dataset": "pathogens", "setting": st, "N": N, "model": "persistence",
                           "pr_auc": AP(y, pers[te]), "prevalence": y.mean(), "n": len(y), "n_pos": int(y.sum())})
            T = lambda a, b: diff_test(y, bl, S[a] if isinstance(a, str) else a, S[b] if isinstance(b, str) else b)
            for hyp, a, b in [("H1", "EcoPE-GAT_leaky", "EcoPE-GAT_causal"), ("H1b_i", "EcoPE-GAT_leaky", "G_lgbm_nbr"),
                              ("H1b_ii", "EcoPE-GAT_causal", "G_lgbm_nbr"), ("H2", "EcoPE-GAT_causal", "G_lgbm_nbr"),
                              ("H3", "EcoPE-GAT_causal", "GAT_noctx_causal"), ("H5", "G_lgbm_nbr_leaky", "G_lgbm_nbr"),
                              ("H8", "EcoPE-GAT_asof", "G_lgbm_nbr"), ("H9", "EcoPE-GAT_asof", "EcoPE-GAT_causal"),
                              ("E1", "EcoPE-GAT_asof_plus", "EcoPE-GAT_causal"), ("E1b", "EcoPE-GAT_asof_plus", "EcoPE-GAT_asof"),
                              ("E2", "G_lgbm_nbr_asof", "G_lgbm_nbr"), ("E4", "EcoPE-GAT_asof_plus", "G_lgbm_nbr"),
                              ("E6", "EcoPE-GAT_asof", "EcoPE-GAT_leaky"), ("E6", "EcoPE-GAT_asof_plus", "EcoPE-GAT_leaky"),
                              ("E6", "G_lgbm_nbr_asof", "G_lgbm_nbr_leaky")]:
                add("pathogens", st, N, hyp, a, b, T(a, b))
            add("pathogens", st, N, "H6", fin, "persistence", T(fin, pers[te]))
            # isolated-node sensitivity (H1, H1b, H2) on test nodes not isolated in the causal graph
            k = ~isolated[te]
            for hyp, a, b in [("H1", "EcoPE-GAT_leaky", "EcoPE-GAT_causal"), ("H1b_i", "EcoPE-GAT_leaky", "G_lgbm_nbr"),
                              ("H1b_ii", "EcoPE-GAT_causal", "G_lgbm_nbr"), ("H2", "EcoPE-GAT_causal", "G_lgbm_nbr")]:
                add("pathogens", st, N, hyp, a, b, diff_test(y[k], bl[k], S[a][k], S[b][k]), "non_isolated")
            # emergence (H7, E5)
            k = emer[te]
            add("pathogens", st, N, "H7_i", fin, "persistence", diff_test(y[k], bl[k], S[fin][k], pers[te][k]), "emergence")
            add("pathogens", st, N, "H7_ii", fin, "prevalence", minus_prev_test(y[k], bl[k], S[fin][k]), "emergence")
            add("pathogens", st, N, "E5", "EcoPE-GAT_asof", "persistence",
                diff_test(y[k], bl[k], S["EcoPE-GAT_asof"][k], pers[te][k]), "emergence")

    # ---------------- pests (setting S)
    for case in PEST_CASES:
        e, ctxs, Y, pers, emer, *_ = pest_data(case, cfg)
        ref = H.out_dir(cfg, f"pests/{case}") / "refits"
        fin = pre_["pests"][case]["final_model"]
        blocks = e["block_id"].to_numpy()
        for N in PEST_N:
            S, te = {}, None
            for m in PEST_MODELS:
                S[m], idx = seed_mean(ref, m, "S", N, PEST_SEEDS)
                assert te is None or (idx == te).all()
                te = idx
            y, bl = Y[N][te].astype(int), blocks[te]
            for m, s in S.items():
                rows_s.append({"dataset": case, "setting": "S", "N": N, "model": m, "pr_auc": AP(y, s),
                               "prevalence": y.mean(), "brier": brier_score_loss(y, s), "n": len(y), "n_pos": int(y.sum())})
            rows_s.append({"dataset": case, "setting": "S", "N": N, "model": "persistence", "pr_auc": AP(y, pers[te]),
                           "prevalence": y.mean(), "n": len(y), "n_pos": int(y.sum())})
            T = lambda a, b: diff_test(y, bl, S[a] if isinstance(a, str) else a, S[b] if isinstance(b, str) else b)
            for hyp, a, b in [("R1", "EcoPE-GAT_leaky", "EcoPE-GAT_causal"), ("R1b_i", "EcoPE-GAT_leaky", "G_lgbm_nbr"),
                              ("R1b_ii", "EcoPE-GAT_causal", "G_lgbm_nbr"), ("R5", "G_lgbm_nbr_leaky", "G_lgbm_nbr"),
                              ("R8", "EcoPE-GAT_asof", "G_lgbm_nbr"), ("R9", "EcoPE-GAT_asof", "EcoPE-GAT_causal"),
                              ("E1", "EcoPE-GAT_asof_plus", "EcoPE-GAT_causal"), ("E1b", "EcoPE-GAT_asof_plus", "EcoPE-GAT_asof"),
                              ("E2", "G_lgbm_nbr_asof", "G_lgbm_nbr"), ("E4", "EcoPE-GAT_asof_plus", "G_lgbm_nbr"),
                              ("E6", "EcoPE-GAT_asof", "EcoPE-GAT_leaky"), ("E6", "EcoPE-GAT_asof_plus", "EcoPE-GAT_leaky"),
                              ("E6", "G_lgbm_nbr_asof", "G_lgbm_nbr_leaky")]:
                add(case, "S", N, hyp, a, b, T(a, b))
            k = emer[te]
            add(case, "S", N, "R7", fin, "persistence", diff_test(y[k], bl[k], S[fin][k], pers[te][k]), "emergence")
            add(case, "S", N, "E5", "EcoPE-GAT_asof", "persistence",
                diff_test(y[k], bl[k], S["EcoPE-GAT_asof"][k], pers[te][k]), "emergence")

    sc, ts = pd.DataFrame(rows_s), pd.DataFrame(rows_t)
    sc.to_csv(REP / "scores.csv", index=False, float_format="%.4f")
    ts.to_csv(REP / "tests.csv", index=False, float_format="%.4f")
    verdicts(sc, ts)


def verdicts(sc, ts):
    """Pre-registered verdicts at the primary horizons (pathogens N = 5, pests N = 3)."""
    P = ts[((ts.dataset == "pathogens") & (ts.N == 5)) | ((ts.dataset != "pathogens") & (ts.N == 3))]
    sup = lambda ds, st, hyp, sub="all", a=None: bool(
        P[(P.dataset == ds) & (P.setting == st) & (P.hypothesis == hyp) & (P.subset == sub)
          & ((P.a == a) if a else True)]["supported"].all())
    lo = lambda ds, st, hyp, sub="all": float(
        P[(P.dataset == ds) & (P.setting == st) & (P.hypothesis == hyp) & (P.subset == sub)]["ci_lo"].iloc[0])
    v = {"pathogens": {
        "H1": sup("pathogens", "T", "H1") and sup("pathogens", "S", "H1"),
        "H1b": all(lo("pathogens", s, "H1b_i") > 0 and lo("pathogens", s, "H1b_ii") <= 0 for s in "TS"),
        "H5": sup("pathogens", "T", "H5") and sup("pathogens", "S", "H5"),
        "H6": sup("pathogens", "T", "H6") and sup("pathogens", "S", "H6"),
        "H7": sup("pathogens", "S", "H7_i", "emergence") and sup("pathogens", "S", "H7_ii", "emergence"),
        "H9": sup("pathogens", "T", "H9") and sup("pathogens", "S", "H9"),
        "H2_H3_H8": "reported as is (superiority tests, no verdict)",
        "E1": sup("pathogens", "S", "E1"), "E2": sup("pathogens", "S", "E2"),
        "E5": sup("pathogens", "S", "E5", "emergence")}}
    for c in PEST_CASES:
        v[c] = {"R1": sup(c, "S", "R1"), "R1b": lo(c, "S", "R1b_i") > 0 and lo(c, "S", "R1b_ii") <= 0,
                "R5": sup(c, "S", "R5"), "R7": sup(c, "S", "R7", "emergence"), "R9": sup(c, "S", "R9"),
                "E1": sup(c, "S", "E1"), "E2": sup(c, "S", "E2"), "E5": sup(c, "S", "E5", "emergence")}
    v["replication_D15 (>= 2 of 3 cases)"] = {r: sum(v[c][r] for c in PEST_CASES) >= 2 for r in ["R1", "R1b", "R5", "R7", "R9"]}
    ds4 = ["pathogens"] + PEST_CASES
    h9 = {"pathogens": sup("pathogens", "S", "H9"), **{c: v[c]["R9"] for c in PEST_CASES}}
    e2 = {d: v[d]["E2"] for d in ds4}
    v["E3 (H9/R9 and E2 each in >= 3 of 4)"] = {"as-of vs removal, graph": h9, "as-of vs removal, tabular": e2,
                                                "supported": sum(h9.values()) >= 3 and sum(e2.values()) >= 3}
    # E6: share of the leakage gain recovered (descriptive, point estimates)
    pr = sc.set_index(["dataset", "setting", "N", "model"])["pr_auc"]
    e6 = {}
    for d in ds4:
        st, N = ("S", 5) if d == "pathogens" else ("S", 3)
        g = lambda m: pr[(d, st, N, m)]
        gl, gc = g("EcoPE-GAT_leaky") - g("EcoPE-GAT_causal"), g("G_lgbm_nbr_leaky") - g("G_lgbm_nbr")
        e6[d] = {"EcoPE-GAT_asof": (g("EcoPE-GAT_asof") - g("EcoPE-GAT_causal")) / gl if gl else None,
                 "EcoPE-GAT_asof_plus": (g("EcoPE-GAT_asof_plus") - g("EcoPE-GAT_causal")) / gl if gl else None,
                 "G_lgbm_nbr_asof": (g("G_lgbm_nbr_asof") - g("G_lgbm_nbr")) / gc if gc else None,
                 "leaky_gain_gnn": gl, "leaky_gain_lgbm": gc}
    v["E6_share_of_leakage_gain_recovered"] = e6
    json.dump(v, open(REP / "verdicts.json", "w"), indent=1, default=float)
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pre", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    pre(cfg) if a.pre else final(cfg)
