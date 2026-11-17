"""D12 B3: counts (no predictions) of the sealed hold-out's low-effort part, using the SAME rule
as the development transfer split (block-median log1p prior effort <= development threshold)."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from common import label_col, load_config, variant_name  # noqa: E402
from features import load_modelling_table  # noqa: E402

cfg = load_config()
B, V = cfg["spatial"]["buffer_km"], variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
thr = json.load(open("data/processed/splits_v2_summary.json"))["transfer"]["threshold_block_median_log_effort"]
e = load_modelling_table(cfg, include_holdout=True)   # counts only: no model sees these rows
h = e[e["holdout"]].copy()
h["log_eff"] = np.log1p(h["effort_fungi_prior"])
bmed = h.groupby("block_id")["log_eff"].median()
low_blocks = set(bmed[bmed <= thr].index)
res = {"threshold": thr}
for name, m in [("holdout_low", h["block_id"].isin(low_blocks)), ("holdout_high", ~h["block_id"].isin(low_blocks))]:
    res[name] = {"plots": int(m.sum()), "blocks": int(h.loc[m, "block_id"].nunique()),
                 **{f"pos_N{N}": int(h.loc[m, label_col(N, B, V)].sum()) for N in cfg["labels"]["horizons"]}}
res["dev_test_low_for_comparison"] = json.load(open("data/processed/splits_v2_summary.json"))["transfer"]["test_low"]
json.dump(res, open("reports/holdout_transfer_counts.json", "w"), indent=1)
print(json.dumps(res, indent=1))
