"""Shared helpers: config loading and path resolution."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (or the file named by $ECOPE_CONFIG, used only for smoke tests)."""
    path = path or os.environ.get("ECOPE_CONFIG") or ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repository root."""
    q = Path(cfg["paths"][key])
    return q if q.is_absolute() else (ROOT / q).resolve()


def variant_name(unc: str, bor: str) -> str:
    """Name of a GBIF filtering variant, e.g. 'unc-keep_bor-all'."""
    return f"unc-{unc}_bor-{bor}"


def label_col(n: int, buffer_km: int, variant: str) -> str:
    return f"y_N{n}_b{buffer_km}_{variant}"
