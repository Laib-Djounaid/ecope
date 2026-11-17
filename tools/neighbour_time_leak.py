"""D12 audit: does a neighbour's prior presence (records before the NEIGHBOUR's survey year) contain
target records from the FOCAL plot's label window (year_i, year_i + N]? Uses the same geo kNN graph
as the GAT pilots (k from config, symmetrised). Development set only; no model, no hold-out."""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.neighbors import BallTree

sys.path.insert(0, "src")
from common import label_col, load_config, p, variant_name  # noqa: E402
from features import labels, load_modelling_table  # noqa: E402
from graph_models import geo_knn_graph  # noqa: E402

N = 5
cfg = load_config()
R, k, B = cfg["spatial"]["earth_radius_km"], cfg["graph"]["k"], cfg["spatial"]["buffer_km"]
V = variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
e = load_modelling_table(cfg)
y = labels(e, cfg, N)
yr = e["survey_year"].to_numpy().astype(int)
rec = pd.read_parquet(p(cfg, "processed_dir") / f"gbif_filtered_{V}.parquet")
tree = BallTree(np.radians(rec[["decimalLatitude", "decimalLongitude"]].to_numpy()), metric="haversine")
ryear = rec["year"].to_numpy().astype(int)
near = tree.query_radius(np.radians(e[["Latitude", "Longitude"]].to_numpy()), r=B / R)
# check: reproduces the stored prior count
own = np.array([(ryear[ix] < yr[i]).sum() for i, ix in enumerate(near)])
assert (own == e[f"n_prior_b{B}_{V}"].to_numpy()).all(), "prior count not reproduced"

ei, _ = geo_knn_graph(e[["Latitude", "Longitude"]].to_numpy(), k, B, R)
src, dst = ei.numpy()                       # message j=src -> i=dst
later = yr[src] > yr[dst]
leak = np.zeros(len(src), dtype=int)
for t, (j, i) in enumerate(zip(src, dst)):
    ry = ryear[near[j]]
    leak[t] = ((ry > yr[i]) & (ry <= yr[i] + N) & (ry < yr[j])).sum()
has_leak = np.zeros(len(e), dtype=bool)
np.logical_or.at(has_leak, dst, leak > 0)
res = {"edges": int(len(src)), "frac_edges_neighbour_surveyed_later": float(later.mean()),
       "frac_edges_with_window_records_in_neighbour_prior": float((leak > 0).mean()),
       "frac_plots_with_any_leaky_neighbour": float(has_leak.mean()),
       "P(leaky neighbour | y=1)": float(has_leak[y == 1].mean()),
       "P(leaky neighbour | y=0)": float(has_leak[y == 0].mean()),
       "P(y=1 | leaky neighbour)": float(y[has_leak].mean()),
       "P(y=1 | no leaky neighbour)": float(y[~has_leak].mean()),
       "ROC_of_leak_indicator_alone": float(AUC(y, has_leak)),
       "PR_of_leak_indicator_alone": float(AP(y, has_leak)), "prevalence": float(y.mean())}
tl = e["transfer_role"].to_numpy() == "test_low"
res["test_low: P(leaky | y=1)"] = float(has_leak[tl & (y == 1)].mean())
res["test_low: P(leaky | y=0)"] = float(has_leak[tl & (y == 0)].mean())
res["test_low: PR_of_leak_indicator_alone"] = float(AP(y[tl], has_leak[tl]))
json.dump(res, open("reports/neighbour_time_leak.json", "w"), indent=1)
print(json.dumps(res, indent=1))
