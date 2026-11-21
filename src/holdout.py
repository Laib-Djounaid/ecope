"""Single confirmatory hold-out run (pre-registrations D13-D16; frozen at tag `frozen-D16`).

Shared helpers for the `--holdout` mode of the model scripts (09_d13, 11_d14, 14_d16_lgbm_asof,
p02_pests_models). In this mode the scripts do NOT tune: they reload the hyperparameters tuned on
development data (setting T <- context dev_transfer, setting S <- context all_dev), refit with the
pre-registered seeds and horizons, and save the hold-out scores. The graph is transductive over
development + hold-out nodes; hold-out labels are never used for training or early stopping.

Pathogens (D13 section 3):
  T  train = development train_high minus transfer_inner_val; inner val = transfer_inner_val;
     test = hold-out low-effort part (block-median log1p effort <= development threshold).
  S  train = whole development set minus its blocked inner validation (seed split_seed + 500,
     identical to the all_dev tuning context); test = whole hold-out.
Pests (D15 section 3): S only (train = development minus alldev_inner_val, test = hold-out).
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, p
from features import labels, load_modelling_table

s1 = importlib.import_module("02_splits")

PATHOGEN_SHA = "31e674c06b13b0ac4b5afb0fc3f0c66af44c9e7e3ab05d464cbe4c4ff3cd3a7f"
PATHOGEN_COUNTS = {"holdout": (7920, 92), "holdout_low": (3629, 27)}
DEV_CONTEXT = {"T": "dev_transfer", "S": "all_dev"}


def out_dir(cfg: dict, data: str) -> Path:
    d = p(cfg, "results_dir") / "holdout" / data
    d.mkdir(parents=True, exist_ok=True)
    return d


def best_params(dev_out: Path, model: str, setting: str) -> dict:
    f = Path(dev_out) / "params" / f"{model}_{DEV_CONTEXT[setting]}.json"
    return json.load(open(f))["best_params"]


def pathogen_setup(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Development + hold-out table (dev rows in development order) and the T / S contexts."""
    e = load_modelling_table(cfg, include_holdout=True)
    h = e["holdout"].to_numpy()
    ids = np.sort(e.loc[h, "PlotObservationID"].to_numpy())
    assert hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() == PATHOGEN_SHA, "hold-out changed"
    # development rows keep their development order (needed to reproduce the inner validation)
    e_dev = load_modelling_table(cfg)
    assert (e.loc[~h, "PlotObservationID"].to_numpy() == e_dev["PlotObservationID"].to_numpy()).all()
    ev = cfg["evaluation"]
    dev = np.flatnonzero(~h)
    blocks = e["block_id"].to_numpy()
    strat = labels(e, cfg, ev["stratify_horizon"])
    m = s1.inner_split(strat[dev], blocks[dev], ev["split_seed"] + cfg["d13"]["alldev_inner_seed_offset"], ev["n_inner"])
    S = (dev[~m], dev[m], np.flatnonzero(h))
    # same inner validation as the all_dev tuning context (asserted against 09_d13.contexts)
    d13 = importlib.import_module("09_d13")
    ref = d13.contexts(e_dev, cfg)["all_dev"]
    assert (e["PlotObservationID"].to_numpy()[S[1]] == e_dev["PlotObservationID"].to_numpy()[ref[1]]).all()
    thr = json.load(open(p(cfg, "processed_dir") / "splits_v2_summary.json"))["transfer"]["threshold_block_median_log_effort"]
    le = pd.Series(np.log1p(e["effort_fungi_prior"].to_numpy()))
    bmed = le[h].groupby(blocks[h]).median()
    low = np.isin(blocks, bmed[bmed <= thr].index.to_numpy()) & h
    role, tiv = e["transfer_role"].to_numpy(), e["transfer_inner_val"].to_numpy()
    T = (np.flatnonzero((role == "train_high") & ~tiv), np.flatnonzero(tiv), np.flatnonzero(low))
    assert (len(S[2]), len(np.unique(blocks[S[2]]))) == PATHOGEN_COUNTS["holdout"]
    assert (len(T[2]), len(np.unique(blocks[T[2]]))) == PATHOGEN_COUNTS["holdout_low"]
    for tr, iv, te in (T, S):
        assert not h[tr].any() and not h[iv].any() and h[te].all()
        assert not set(blocks[tr]) & set(blocks[te]) and not set(blocks[iv]) & set(blocks[te])
    return e, {"T": T, "S": S}


def pest_setup(case: str) -> tuple[pd.DataFrame, dict]:
    """Development + hold-out pest table (development rows first, in development order) and context S."""
    pr = ROOT / "data" / "processed" / "pests"
    d = pd.read_parquet(pr / f"{case}_dev.parquet")
    h = pd.read_parquet(pr / f"{case}_holdout.parquet")
    want = json.load(open(ROOT / "reports" / "pests" / f"{case}_summary.json"))["holdout_keys_sha256"]
    got = hashlib.sha256(",".join(map(str, sorted(h["key"]))).encode()).hexdigest()   # as in p01
    assert got == want, f"{case}: hold-out changed"
    assert not d["holdout"].any() and h["holdout"].all()
    for c in d.columns:
        if c not in h.columns:
            h[c] = False if d[c].dtype == bool else (-1 if c == "spatial_fold" else np.nan)
    e = pd.concat([d, h[d.columns]], ignore_index=True)
    n = len(d)
    iv = d["alldev_inner_val"].to_numpy().astype(bool)
    dev = np.arange(n)
    S = (dev[~iv], dev[iv], np.arange(n, len(e)))
    blocks = e["block_id"].to_numpy()
    assert not set(blocks[S[0]]) & set(blocks[S[2]]) and not set(blocks[S[1]]) & set(blocks[S[2]])
    return e, {"S": S}



def guard(cached_refit):
    """In a dry run (env ECOPE_HOLDOUT_DRYRUN=1) list the refits instead of running them: the set-up,
    feature self-checks, leak checks and hyperparameter loading are exercised, no model is fitted and
    no hold-out score is produced."""
    import os
    if os.environ.get("ECOPE_HOLDOUT_DRYRUN") != "1":
        return cached_refit

    def dry(path, fn):
        print(f"[dry run] would fit {Path(path).name}", flush=True)
        return None, {}
    return dry
