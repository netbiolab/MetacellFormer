"""Select released source-cell memberships from the matching upstream count H5AD.

CELLxGENE inputs must be the filtered Census view described in the corpus README,
not a Discover H5AD with a different barcode namespace. This command does not
redraw cells, rerun MetaQ, normalize expression, or change the gene universe.
"""
from pathlib import Path
import argparse

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def prepare(source, corpus, dataset_id, output, counts_layer=None, chunk_size=2048):
    manifest = pd.read_csv(corpus / "source_manifest.tsv", sep="\t", dtype={"dataset_id": str})
    match = manifest[manifest.dataset_id.eq(dataset_id)]
    if len(match) != 1:
        raise ValueError("Expected exactly one dataset in the corpus manifest")
    record = match.iloc[0]
    membership = pd.read_csv(corpus / record.membership_file, sep="\t", dtype=str)
    a = ad.read_h5ad(source, backed="r")
    reference = ad.read_h5ad(corpus / record.metacell_file, backed="r")
    try:
        if not a.obs_names.is_unique or not membership.source_cell_id.is_unique:
            raise ValueError("Source cell IDs must be unique")
        rows = a.obs_names.get_indexer(membership.source_cell_id)
        if (rows < 0).any():
            raise ValueError("Missing source cells; check the source release and ID namespace")
        genes = (a.var.feature_id.astype(str) if "feature_id" in a.var
                 else pd.Series(a.var_names.astype(str)))
        expected = reference.var.feature_id.astype(str)
        if not genes.is_unique or set(genes) != set(expected):
            raise ValueError("Source gene universe differs from the frozen release")
        columns = pd.Index(genes).get_indexer(expected)
        matrix = a.layers[counts_layer] if counts_layer else a.X
        blocks = []
        for start in range(0, len(rows), chunk_size):
            block = sparse.csr_matrix(matrix[rows[start:start + chunk_size]])[:, columns]
            if not np.isfinite(block.data).all() or (block.data < 0).any():
                raise ValueError("Counts must be finite and nonnegative")
            blocks.append(block)
        counts = sparse.vstack(blocks, format="csr")
        obs = pd.DataFrame(index=pd.Index(membership.source_cell_id, name="cell_id"))
        out = ad.AnnData(counts, obs=obs, var=reference.var.copy())
        output.parent.mkdir(parents=True, exist_ok=True)
        out.write_h5ad(output, compression="gzip")
    finally:
        a.file.close()
        reference.file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "corpus-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--counts-layer")
    p.add_argument("--chunk-size", type=int, default=2048)
    a = p.parse_args()
    if a.chunk_size <= 0:
        p.error("--chunk-size must be positive")
    if a.output.exists():
        p.error("--output already exists; choose a new file")
    prepare(a.input, a.corpus_root, a.dataset_id, a.output, a.counts_layer, a.chunk_size)


if __name__ == "__main__":
    main()
