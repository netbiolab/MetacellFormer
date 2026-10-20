from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ValueBinner:
    edges: np.ndarray
    n_bins: int

    @property
    def value_pad_id(self) -> int:
        return 0

    @property
    def value_mask_id(self) -> int:
        return self.n_bins + 1

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0,), dtype=np.uint8)
        # 1..n_bins (0 is reserved for PAD)
        bins = np.digitize(arr, self.edges, right=False).astype(np.int64) + 1
        bins = np.clip(bins, 1, self.n_bins)
        return bins.astype(np.uint8)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.edges.astype(np.float32))

    @staticmethod
    def load(path: str | Path, n_bins: int) -> "ValueBinner":
        edges = np.load(str(path), mmap_mode=None).astype(np.float32)
        return ValueBinner(edges=edges, n_bins=int(n_bins))


def fit_global_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.linspace(0.0, 1.0, max(1, n_bins - 1), dtype=np.float32)

    q = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)[1:-1]
    edges = np.quantile(arr, q, method="linear").astype(np.float32)

    # keep monotonic non-decreasing; repeated edges are acceptable
    edges = np.maximum.accumulate(edges)
    return edges
