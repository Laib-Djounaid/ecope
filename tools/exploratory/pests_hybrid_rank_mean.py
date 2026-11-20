import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as AP
from common import ROOT, load_config, p
cfg = load_config()
def S(ref, m, k, N):
    z = [np.load(ref / f"{m}_cv{k}_N{N}_s{s}.npz") for s in (0, 1, 2)]
    return z[0]["test_idx"], np.mean([q["test"] for q in z], 0)
for case in sys.argv[1:]:
    d = pd.read_parquet(ROOT / "data/processed/pests" / f"{case}_dev.parquet")
    ref = p(cfg, "results_dir") / "pests" / case / "refits"
    em = d["prior_any"].to_numpy() == 0
    for N in (3, 5):
        r = {"lgbm_nbr": [], "asof_plus": [], "hybrid": [], "lgbm_nbr_em": [], "asof_plus_em": [], "hybrid_em": []}
        for k in range(5):
            ix, a = S(ref, "G_lgbm_nbr", k, N); _, b = S(ref, "EcoPE-GAT_asof_plus", k, N)
            h = (rankdata(a) + rankdata(b)) / (2 * len(a))
            y = d[f"y{N}"].to_numpy()[ix].astype(int); e = em[ix]
            for nm, s in [("lgbm_nbr", a), ("asof_plus", b), ("hybrid", h)]:
                r[nm].append(AP(y, s)); r[nm + "_em"].append(AP(y[e], s[e]))
        print(case, N, {k: round(float(np.mean(v)), 3) for k, v in r.items()})
