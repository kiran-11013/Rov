"""Cluster bootstrap confidence intervals, an exact McNemar test, Holm correction, and the Brier score."""
from math import comb

import numpy as np


def cluster_bootstrap(values, clusters, n_boot, rng, stat=np.mean):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    groups = [values[clusters == u] for u in uniq]
    point = float(stat(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(len(groups), size=len(groups))
        boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired binary outcomes. Returns (p, n_a_only, n_b_only)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    k = min(n10, n01)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return float(min(1.0, p)), n10, n01


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def brier(pred, label):
    return (np.asarray(pred, float) - np.asarray(label, float)) ** 2
