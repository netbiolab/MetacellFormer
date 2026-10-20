"""Sum raw cell counts by a supplied cell-to-metacell assignment.

The assignment is an input produced by MetaQ; this script does not rerun MetaQ.
All measured features are retained so CP10K uses the original denominator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
from anndata.experimental import read_elem, sparse_dataset
import numpy as np
import pandas as pd
from scipy import sparse


def aggregate(source: Path, membership: Path, output: Path, source_id: str,
              counts_layer: str | None = None, chunk_size: int = 2048) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with h5py.File(source, "r") as handle:
        cell_names = read_elem(handle["obs"]).index
        var = read_elem(handle["var"])
    if not cell_names.is_unique:
        raise ValueError("Duplicate source cell IDs")
    assignment = pd.read_csv(membership, sep="\t", dtype=str, keep_default_na=False)
    required = {"source_cell_id", "metacell_id"}
    if not required.issubset(assignment.columns):
        raise ValueError("Membership needs source_cell_id and metacell_id columns")
    if assignment.empty or (assignment[list(required)] == "").any().any():
        raise ValueError("Membership must contain nonempty cell and metacell IDs")
    if assignment.source_cell_id.duplicated().any():
        raise ValueError("Each cell can have only one metacell assignment")
    indices = cell_names.get_indexer(assignment.source_cell_id)
    if (indices < 0).any():
        raise ValueError("Membership contains cells absent from the count matrix")
    ids, groups = np.unique(assignment.metacell_id.to_numpy(), return_inverse=True)
    group_for_cell = np.full(len(cell_names), -1, dtype=np.int64)
    group_for_cell[indices] = groups
    summed = None
    with h5py.File(source, "r") as handle:
        node = handle["layers"][counts_layer] if counts_layer else handle["X"]
        counts = sparse_dataset(node) if isinstance(node, h5py.Group) else node
        if counts.shape != (len(cell_names), len(var)):
            raise ValueError("Count matrix shape does not match metadata")
        for start in range(0, len(cell_names), chunk_size):
            stop = min(start + chunk_size, len(cell_names))
            chunk = sparse.csr_matrix(counts[start:stop])
            if chunk.nnz and (not np.isfinite(chunk.data).all() or (chunk.data < 0).any()):
                raise ValueError("Input must be a finite, nonnegative raw-count matrix")
            selected = group_for_cell[start:stop] >= 0
            rows = np.flatnonzero(selected)
            incidence = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.int64),
                 (group_for_cell[start:stop][selected], rows)),
                shape=(len(ids), stop-start))
            partial = incidence @ chunk
            summed = partial if summed is None else summed + partial
    summed = summed.tocsr()
    summed.sum_duplicates()
    summed.eliminate_zeros()
    obs = pd.DataFrame(index=pd.Index(ids, name="metacell_id"))
    obs["source_id"] = source_id
    obs["n_cells"] = np.bincount(groups, minlength=len(ids))
    obs["total_counts_all_features"] = np.asarray(summed.sum(axis=1)).ravel()
    result = ad.AnnData(X=summed, obs=obs, var=var)
    result.uns["aggregation"] = "sum of member-cell raw counts across all measured features"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("membership", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--counts-layer")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    aggregate(args.source, args.membership, args.output, args.source_id, args.counts_layer, args.chunk_size)


if __name__ == "__main__":
    main()
