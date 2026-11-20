import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from sklearn.metrics import average_precision_score as AP
from common import ROOT, load_config, p
cfg = load_config()
for case in ["P1_halys_europe", "P2_halys_namerica", "P3_popillia_namerica"]:
    d = pd.read_parquet(ROOT / "data/processed/pests" / f"{case}_dev.parquet")
    ref = p(cfg, "results_dir") / "pests" / case / "refits"
    em = d["prior_any"].to_numpy() == 0
    for N in (3, 5):
        row = {}
        for m in ["G_lgbm", "G_lgbm_nbr", "G_lgbm_nbr_leaky", "EcoPE-GAT_causal", "EcoPE-GAT_leaky", "EcoPE-GAT_asof", "EcoPE-GAT_asof_plus", "G_lgbm_nbr_asof"]:
            v = []
            for k in range(5):
                fs = [ref / f"{m}_cv{k}_N{N}_s{s}.npz" for s in (0, 1, 2)]
                if not all(f.exists() for f in fs): break
                z = [np.load(f) for f in fs]; ix = z[0]["test_idx"]; s = np.mean([q["test"] for q in z], 0)
                keep = em[ix]; y = d[f"y{N}"].to_numpy()[ix][keep].astype(int)
                v.append(AP(y, s[keep]))
            if len(v) == 5: row[m] = round(float(np.mean(v)), 3)
        print(case, N, row)
