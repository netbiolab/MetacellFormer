"""Chunking must preserve direct raw sums, IDs and all measured genes."""
import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

spec = importlib.util.spec_from_file_location(
    "aggregate_metacells", Path(__file__).parents[1] / "scripts/aggregate_metacells.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("storage", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
@pytest.mark.parametrize("layer", [None, "counts"])
def test_chunked_raw_sums(tmp_path, storage, layer):
    x = np.arange(35, dtype=np.float32).reshape(7, 5)
    cells = ad.AnnData(storage(x), obs=pd.DataFrame(index=list("abcdefg")),
                       var=pd.DataFrame(index=["MT-A", "RPS1", "g2", "g3", "g4"]))
    if layer:
        cells.layers[layer] = storage(x)
        cells.X = sparse.csr_matrix(x.shape)
    source = tmp_path / "cells.h5ad"
    cells.write_h5ad(source)
    membership = tmp_path / "membership.tsv"
    pd.DataFrame({"source_cell_id": ["g", "b", "d"],
                  "metacell_id": ["NA", "10", "NA"]}).to_csv(membership, sep="\t", index=False)
    for chunk in [1, 3, 2048]:
        output = tmp_path / "metacells.h5ad"
        module.aggregate(source, membership, output, "test", layer, chunk)
        result = ad.read_h5ad(output)
        np.testing.assert_array_equal(result.X.toarray(), np.stack([x[1], x[6] + x[3]]))
        assert result.obs_names.tolist() == ["10", "NA"]
        assert result.obs.n_cells.tolist() == [1, 2]
        assert result.var_names.equals(cells.var_names)
