"""Load the paper checkpoint encoder and extract representations.

This inference loader loads encoder parameters from the full pretraining
checkpoint. The training architecture and losses are in training_runtime.py.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings


class AlphaScaledBertEmbeddings(BertEmbeddings):
    """Apply the checkpoint's scale to absolute position embeddings."""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.position_embedding_alpha = float(getattr(config, "position_embedding_alpha", 1.0))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None,
                inputs_embeds=None, past_key_values_length=0):
        shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        length = shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                token_type_ids = self.token_type_ids[:, :length].expand(shape[0], length)
            else:
                token_type_ids = torch.zeros(shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden = inputs_embeds + self.token_type_embeddings(token_type_ids)
        if self.position_embedding_type == "absolute":
            hidden = hidden + self.position_embedding_alpha * self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(hidden))


def load_encoder(model_dir: str | Path, device: str = "cpu") -> BertModel:
    """Load every encoder tensor from a MetacellFormer model directory.

    The full checkpoint also contains pretraining heads. This loader selects
    only encoder tensors and requires an exact encoder state-dict match.
    """
    directory = Path(model_dir)
    config = BertConfig.from_json_file(str(directory / "config.json"))
    if str(getattr(config, "encoder_layout", "unique")) != "unique":
        raise ValueError("This loader supports the paper's unique-layer encoder only")
    if str(getattr(config, "norm_position", "post")) != "post":
        raise ValueError("This loader supports the paper's post-normalization encoder only")
    model = BertModel(config, add_pooling_layer=False)
    model.embeddings = AlphaScaledBertEmbeddings(config)
    weights = directory / "model.safetensors"
    with safe_open(str(weights), framework="pt", device="cpu") as archive:
        names = [name for name in archive.keys() if name.startswith("bert.")]
        state = {name.removeprefix("bert."): archive.get_tensor(name) for name in names}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Encoder tensors do not match config: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


@torch.inference_mode()
def cell_embeddings(model: BertModel, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, token_type_id: int = 0) -> torch.Tensor:
    """Mean-pool non-padding final-layer states, including CLS as in the paper evaluation."""
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have matching [cells, tokens] shapes")
    types = torch.full_like(input_ids, int(token_type_id))
    hidden = model(input_ids=input_ids, attention_mask=attention_mask,
                   token_type_ids=types).last_hidden_state
    keep = attention_mask.bool()
    counts = keep.sum(dim=1)
    if bool((counts <= 1).any()):
        raise ValueError("A cell must contain CLS and at least one gene token")
    return (hidden * keep.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


@torch.inference_mode()
def gene_embeddings(model: BertModel, token_ids: torch.Tensor) -> torch.Tensor:
    """Return input gene embeddings for vocabulary token IDs."""
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=model.device)
    if bool(((ids < 4) | (ids >= model.config.vocab_size)).any()):
        raise ValueError("Gene token IDs must be in the vocabulary range")
    return model.embeddings.word_embeddings(ids)


def perturbation_response(perturbed: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    """Difference between mean perturbed and control cell representations."""
    if perturbed.ndim != 2 or controls.ndim != 2 or perturbed.shape[1] != controls.shape[1]:
        raise ValueError("Both inputs must have matching embedding dimensions")
    if not len(perturbed) or not len(controls):
        raise ValueError("Both groups must contain cells")
    return perturbed.mean(dim=0) - controls.mean(dim=0)
