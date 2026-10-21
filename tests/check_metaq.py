"""Run the public MetaQ example in its separate environment (not a pytest test).

Usage: python tests/check_metaq.py
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def main():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(METAQ_NUM_WORKERS="0", METAQ_PIN_MEMORY="0",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with tempfile.TemporaryDirectory(prefix="metaq_example_") as temporary:
        work = Path(temporary)
        rng = np.random.default_rng(1)
        counts = sparse.csr_matrix(rng.poisson(1, size=(100, 64)).astype(np.float32))
        gene_names = ["MT-CO1", "IGHV1", "RPL1", "TRAV1"] + [f"gene_{i}" for i in range(60)]
        cells = ad.AnnData(counts,
                          obs=pd.DataFrame(index=[f"cell_{i}" for i in range(100)]),
                          var=pd.DataFrame(index=gene_names))
        from MetaQ_sc.data_utils import preprocess
        selected, _, raw_selected, _ = preprocess(cells.copy(), "RNA")
        assert selected.shape == raw_selected.shape == counts.shape
        cells.write_h5ad(work / "small_counts.h5ad")
        def run(*args):
            subprocess.run([sys.executable, *map(str, args)], cwd=work, env=env, check=True)
        run(root / "scripts/prepare_metaq.py", "--input", "small_counts.h5ad",
            "--output-dir", "small_metaq", "--device", "cpu", "--batch-size", "16")
        run(root / "scripts/aggregate_metacells.py", "small_counts.h5ad",
            "small_metaq/memberships.tsv", "small_metacells.h5ad",
            "--source-id", "synthetic", "--chunk-size", "17")
        cells = ad.read_h5ad(work / "small_counts.h5ad")
        result = ad.read_h5ad(work / "small_metacells.h5ad")
        membership = pd.read_csv(work / "small_metaq/memberships.tsv", sep="\t", dtype=str)
        assert membership.source_cell_id.tolist() == cells.obs_names.tolist()
        assert result.var_names.equals(cells.var_names)
        assert int(result.obs.n_cells.sum()) == cells.n_obs
        for group in result.obs_names:
            selected = membership.metacell_id.to_numpy() == group
            expected = np.asarray(cells.X[selected].sum(axis=0)).ravel()
            np.testing.assert_array_equal(result[group].X.toarray().ravel(), expected)
        print("MetaQ check passed: 100 cells, 300 epochs, IDs and raw sums verified")


if __name__ == "__main__":
    main()
