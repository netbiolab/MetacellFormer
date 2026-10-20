"""Canonical ranking metrics used by the revalidated Exact and Functional runs."""
from __future__ import annotations

from typing import Any

import numpy as np


METRIC_NAMES = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "map",
    "purity_at_10",
)


def stable_descending_order(scores: np.ndarray) -> np.ndarray:
    """Return a deterministic descending order; input columns are canonical item order."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("ranking scores must be a finite two-dimensional array")
    return np.argsort(-scores, axis=1, kind="stable")


def metrics_from_relevance(relevance: np.ndarray, topk: int = 10) -> dict[str, np.ndarray]:
    """Compute bounded IR metrics from a query-by-reference binary relevance matrix."""
    rel = np.asarray(relevance, dtype=bool)
    if rel.ndim != 2:
        raise ValueError(f"relevance must be 2-D, got {rel.shape}")
    n_query, n_ref = rel.shape
    if n_ref <= 0:
        raise ValueError("reference candidate set is empty")
    n_pos = rel.sum(axis=1).astype(np.int64)
    if np.any(n_pos <= 0):
        bad = np.flatnonzero(n_pos <= 0)[:10].tolist()
        raise ValueError(f"queries without a positive reference: {bad}")
    k = min(int(topk), n_ref)
    ranks = np.arange(1, n_ref + 1, dtype=np.float64)[None, :]
    cumulative = np.cumsum(rel, axis=1, dtype=np.int64)
    ap = ((cumulative / ranks) * rel).sum(axis=1) / n_pos
    first = np.argmax(rel, axis=1).astype(np.float64) + 1.0
    out = {
        "recall_at_1": rel[:, :1].any(axis=1).astype(np.float64),
        "recall_at_5": rel[:, : min(5, n_ref)].any(axis=1).astype(np.float64),
        "recall_at_10": rel[:, : min(10, n_ref)].any(axis=1).astype(np.float64),
        "mrr": 1.0 / first,
        "map": ap,
        "purity_at_10": rel[:, :k].sum(axis=1).astype(np.float64) / float(k),
    }
    validate_metric_arrays(out)
    if np.any(out["recall_at_1"] > out["recall_at_5"] + 1e-12) or np.any(out["recall_at_5"] > out["recall_at_10"] + 1e-12):
        raise ValueError("Recall monotonicity invariant failed")
    return out


def validate_metric_arrays(metrics: dict[str, Any], tolerance: float = 1e-9) -> None:
    for name in METRIC_NAMES:
        if name not in metrics:
            raise ValueError(f"missing metric: {name}")
        values = np.asarray(metrics[name], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite metric values: {name}")
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            lo = float(values.min())
            hi = float(values.max())
            raise ValueError(f"metric outside [0,1]: {name} min={lo} max={hi}")


def summarize_metric_arrays(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    validate_metric_arrays(metrics)
    return {name: float(np.mean(np.asarray(metrics[name], dtype=np.float64))) for name in METRIC_NAMES}
