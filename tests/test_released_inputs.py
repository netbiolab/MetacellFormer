"""Mapping invariants for the archived evaluation feature axes."""
from pathlib import Path
from types import SimpleNamespace
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from benchmarks.embeddings.extract_embeddings import feature_tokenizer, mapped_geneformer


def test_historical_alias_features_are_not_collapsed_before_ranking(tmp_path):
    # Collapsing A's two aliases would incorrectly move it ahead of B.
    var = pd.DataFrame({
        "metacellformer_gene_id": ["ENSG_A", "ENSG_A", "ENSG_B", ""],
        "geneformer_ensembl_id": ["ENSG_A", "ENSG_A", "ENSG_B", ""],
    }, index=["A_old", "A_new", "B", "unmapped"])
    a = ad.AnnData(sparse.csr_matrix([[6., 6., 10., 200.]]), var=var)
    tokenizer = SimpleNamespace(gene_to_token={"ENSG_A": 4, "ENSG_B": 5},
                                medians=np.ones(2, np.float32),
                                special={"<pad>": 0, "<cls>": 1})
    directory = tmp_path / "training_resources"
    directory.mkdir()
    np.save(directory / "value_bin_edges.npy", np.linspace(0, 10, 63))
    encode, _ = feature_tokenizer(a, tokenizer, tmp_path)
    result = encode(a.X[0])
    np.testing.assert_array_equal(result["input_ids"][:4], [1, 5, 4, 4])
    assert result["attention_mask"].sum() == 4
    mapped = mapped_geneformer(a, None)
    assert mapped.var.ensembl_id.tolist() == ["ENSG_A", "ENSG_A", "ENSG_B"]
    assert mapped.obs.n_counts.iloc[0] == 22
    assert a.X.sum() == 222
