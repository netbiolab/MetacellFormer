"""Verify that token caches retain every profile in the supplied fixed split."""

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def test_tokenization_keeps_profiles_without_vocabulary_genes(tmp_path):
    model = tmp_path / "model"
    resources = model / "training_resources"
    resources.mkdir(parents=True)
    (model / "gene_vocabulary.json").write_text(json.dumps({
        "genes": ["g0"],
        "special_tokens": {"<pad>": 0, "<cls>": 1, "<mask>": 2, "<unk>": 3},
    }))
    np.save(model / "gene_reference_medians.npy", np.ones(1, dtype=np.float32))
    np.save(resources / "value_bin_edges.npy", np.linspace(0, 10, 63))

    profiles = ad.AnnData(
        sparse.csr_matrix([[3, 0], [0, 5], [4, 2]], dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame({"feature_id": ["g0", "unmapped"]}, index=["g0", "unmapped"]),
    )
    input_path = tmp_path / "metacells.h5ad"
    profiles.write_h5ad(input_path)
    split_path = tmp_path / "split.tsv"
    pd.DataFrame({
        "profile_id": ["a", "b", "c"],
        "split": ["train", "eval", "train"],
    }).to_csv(split_path, sep="\t", index=False)
    output = tmp_path / "tokens"
    script = Path(__file__).resolve().parents[1] / "scripts" / "tokenize_training.py"
    subprocess.run([
        sys.executable, str(script),
        "--input", str(input_path), "--model-dir", str(model),
        "--split", str(split_path), "--dataset-id", "example",
        "--dataset-idx", "7", "--output-dir", str(output),
    ], check=True)

    ids = np.load(output / "input_ids.npy")
    attention = np.load(output / "attention_mask.npy")
    assert ids.shape == attention.shape == (3, 1024)
    np.testing.assert_array_equal(attention.sum(axis=1), [2, 1, 2])
    np.testing.assert_array_equal(ids[1, :2], [1, 0])
    np.testing.assert_array_equal(np.load(output / "split.npy"), [0, 1, 0])
    np.testing.assert_array_equal(np.load(output / "metacell.npy"), [0, 1, 2])
    np.testing.assert_array_equal(np.load(output / "train_rows_by_mc.npy"), [0, 2])
    np.testing.assert_array_equal(np.load(output / "val_rows_by_mc.npy"), [1])
    index = pd.read_csv(output / "token_index.csv")
    assert index.dataset_id.tolist() == ["example"]
    assert index.dataset_idx.tolist() == [7]
