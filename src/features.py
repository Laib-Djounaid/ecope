"""Feature sets and fold-wise preprocessing (fitted on training rows only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from common import label_col, load_config, p, variant_name

BIOMES = ["Alpine", "Boreal zone", "Dry midlatitudes", "Dry tropics and subtropics",
          "Polar and subpolar zone", "Subtropics with winter rain",
          "Subtropics with year-round rain", "Temperate midlatitudes",
          "Tropics with summer rain", "Tropics with year-round rain"]


def load_modelling_table(cfg: dict) -> pd.DataFrame:
    """Eligible plots merged with their split assignments (row order = splits.parquet)."""
    proc = p(cfg, "processed_dir")
    d = pd.read_parquet(proc / "plots.parquet")
    s = pd.read_parquet(proc / "splits.parquet")
    e = s.merge(d, on="PlotObservationID", how="left", validate="one_to_one")
    assert e["eligible_N5"].all()
    return e


def trait_cols(cfg: dict) -> list[str]:
    t = cfg["features"]["traits"]
    return [f"{x}_CWM" for x in t] + [f"{x}_CWV" for x in t]


def raw_blocks(e: pd.DataFrame, cfg: dict, buffer_km: int | None = None,
               variant: str | None = None) -> dict[str, pd.DataFrame]:
    """Deterministic (non-fitted) transforms, grouped into named blocks."""
    b = buffer_km or cfg["spatial"]["buffer_km"]
    v = variant or variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
    tr = e[trait_cols(cfg)].copy()
    cwv = [c for c in tr.columns if c.endswith("_CWV")]
    tr[cwv] = np.log1p(tr[cwv].clip(lower=0))           # CWV are right-skewed variances
    attrs = e[cfg["features"]["plot_attrs"]].copy()
    attrs["Species_richness"] = np.log1p(attrs["Species_richness"])
    env = e[cfg["features"]["env"]].copy()
    geo = e[["Latitude", "Longitude"]].copy()
    biome = pd.DataFrame({f"biome_{x}": (e["Biome"] == x).astype(float) for x in BIOMES})
    assert set(e["Biome"].dropna().unique()) <= set(BIOMES), "unexpected biome category"
    prior = pd.DataFrame({
        "prior_log_n": np.log1p(e[f"n_prior_b{b}_{v}"]),
        "prior_n_species": e[f"nsp_prior_b{b}_{v}"].astype(float),
        "prior_any": e[f"prior_any_b{b}_{v}"].astype(float),
    })
    return {"traits": tr, "attrs": attrs, "env": env, "geo": geo, "biome": biome, "prior": prior}


FEATURE_SETS = {
    "A": ["traits", "attrs"],                                   # traits + plot attributes
    "B": ["traits", "attrs", "env", "geo", "biome"],            # A + ecological context
    "C": ["traits", "attrs", "env", "geo", "biome", "prior"],   # B + prior presence
}
CONTEXT_BLOCKS = ["env", "geo", "biome"]                         # EcoPE inputs


def assemble(blocks: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    return pd.concat([blocks[n] for n in names], axis=1)


class FoldPreprocessor:
    """Median imputation (+ missing indicators) and standardisation, fitted on train rows."""

    def __init__(self, scale: bool = True):
        self.scale = scale

    def fit(self, X: pd.DataFrame, train_idx: np.ndarray) -> "FoldPreprocessor":
        self.cols = list(X.columns)
        self.imp = SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True).fit(X.iloc[train_idx])
        Z = self.imp.transform(X.iloc[train_idx])
        self.scaler = StandardScaler().fit(Z) if self.scale else None
        n_ind = Z.shape[1] - len(self.cols)
        self.out_cols = self.cols + [f"missing_{self.cols[i]}" for i in
                                     self.imp.indicator_.features_][:n_ind]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.imp.transform(X)
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        Z = np.asarray(Z, dtype=np.float32)
        assert np.isfinite(Z).all()
        return Z


def labels(e: pd.DataFrame, cfg: dict, N: int, buffer_km: int | None = None,
           variant: str | None = None) -> np.ndarray:
    b = buffer_km or cfg["spatial"]["buffer_km"]
    v = variant or variant_name(cfg["gbif"]["missing_uncertainty"], cfg["gbif"]["basis_of_record_main"])
    return e[label_col(N, b, v)].to_numpy().astype(np.int64)


def split_indices(e: pd.DataFrame, scheme: str, fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train, inner_val, test) row indices for one outer fold of a scheme."""
    if scheme == "spatial":
        te = e["spatial_fold"].to_numpy() == fold
        iv = e[f"spatial_inner_val_f{fold}"].to_numpy()
    elif scheme == "random":
        te = e["random_fold"].to_numpy() == fold
        iv = e[f"random_inner_val_f{fold}"].to_numpy()
    elif scheme == "loco":
        te = e["continent"].to_numpy() == fold
        iv = e[f"loco_inner_val_{fold.replace(' ', '_')}"].to_numpy()
    else:
        raise ValueError(scheme)
    tr = ~te & ~iv
    assert not (iv & te).any()
    return np.flatnonzero(tr), np.flatnonzero(iv), np.flatnonzero(te)


def folds_of(e: pd.DataFrame, scheme: str, cfg: dict) -> list:
    if scheme in ("spatial", "random"):
        return list(range(cfg["evaluation"]["n_folds"]))
    return sorted(e["continent"].unique())
