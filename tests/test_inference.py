"""Exercise the installed inference loader without downloading paper weights."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import BertConfig, BertModel

from metacellformer import GeneRankTokenizer, cell_embeddings, load_encoder


def test_small_checkpoint_loading_and_pooling(tmp_path):
    torch.manual_seed(1)
    config = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=32)
    config.position_embedding_alpha = 0.25
    config.to_json_file(tmp_path / "config.json")
    reference = BertModel(config, add_pooling_layer=False).eval()
    save_file({"bert." + key: value.contiguous() for key, value in reference.state_dict().items()},
              tmp_path / "model.safetensors")
    (tmp_path / "gene_vocabulary.json").write_text(json.dumps({
        "genes": ["g0", "g1", "g2", "g3"],
        "special_tokens": {"<pad>": 0, "<cls>": 1, "<mask>": 2, "<unk>": 3}}))
    np.save(tmp_path / "gene_reference_medians.npy", np.ones(4, dtype=np.float32))
    tokenizer = GeneRankTokenizer(tmp_path, max_genes=4)
    profile = tokenizer.encode(["g0", "g1", "g2", "g3"], [5, 1, 0, 3],
                               total_counts_all_features=12)
    assert profile["selected_gene_ids"].tolist() == ["g0", "g3", "g1"]
    model = load_encoder(tmp_path)
    # Scaling the original positional weights is numerically equivalent to alpha scaling.
    with torch.no_grad():
        reference.embeddings.position_embeddings.weight.mul_(0.25)
    ids = torch.tensor(profile["input_ids"][None, :])
    mask = torch.tensor(profile["attention_mask"][None, :])
    with torch.no_grad():
        hidden = reference(input_ids=ids, attention_mask=mask,
                           token_type_ids=torch.zeros_like(ids)).last_hidden_state
    expected = hidden[:, :int(mask.sum())].mean(dim=1)
    actual = cell_embeddings(model, ids, mask)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 16) and torch.isfinite(actual).all()
    with pytest.raises(ValueError, match="at least one gene"):
        cell_embeddings(model, ids, torch.tensor([[1, 0, 0, 0, 0]]))
