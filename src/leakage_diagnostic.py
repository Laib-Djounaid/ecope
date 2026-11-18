"""Leakage diagnostic for graph / neighbourhood models with time-stamped node features
(pre-registration D13, section 1; reusable by other authors).

D1 (dates only): share of directed edges j -> i with t_j > t_i, overall and per receiving node.
D2 (needs the raw time-stamped records): share of positive vs negative nodes receiving a message
   from a neighbour whose features count a record from the node's label window (t_i, t_i + N],
   and the PR-AUC of that indicator alone.
D3 (needs a model) is computed by the model scripts (leaky vs causal graph).

Functions take arrays only, so they apply to any data set:
  d1(edge_index, t) ; d2(edge_index, t, y, near_years, N)
where near_years[j] = years of the records counted in node j's features (window: year < t_j).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score as AP


def d1(ei: np.ndarray, t: np.ndarray) -> dict:
    src, dst = np.asarray(ei)
    later = t[src] > t[dst]
    n = len(t)
    deg = np.bincount(dst, minlength=n)
    share = np.where(deg > 0, np.bincount(dst, weights=later, minlength=n) / np.maximum(deg, 1), np.nan)
    causal_deg = deg - np.bincount(dst, weights=later, minlength=n)
    return {"edges": int(len(src)), "D1_share_edges_later": float(later.mean()),
            "D1_median_node_share_later": float(np.nanmedian(share)),
            "nodes_isolated_in_causal_graph": int((causal_deg == 0).sum()),
            "median_in_degree": float(np.median(deg)), "median_causal_in_degree": float(np.median(causal_deg))}, share


def d2(ei: np.ndarray, t: np.ndarray, y: np.ndarray, near_years: list, N: int) -> dict:
    src, dst = np.asarray(ei)
    def window(j: int, i: int) -> bool:
        ry = near_years[j]
        return bool(((ry > t[i]) & (ry <= t[i] + N) & (ry < t[j])).any())

    leak = np.fromiter((window(j, i) for j, i in zip(src, dst)), dtype=bool, count=len(src))
    has = np.zeros(len(t), dtype=bool)
    np.logical_or.at(has, dst, leak)
    return {"N": N, "D2_edges_with_window_records": float(leak.mean()),
            "D2_P_leaky_given_pos": float(has[y == 1].mean()), "D2_P_leaky_given_neg": float(has[y == 0].mean()),
            "D2_PR_AUC_indicator_alone": float(AP(y, has)), "prevalence": float(y.mean())}, has
