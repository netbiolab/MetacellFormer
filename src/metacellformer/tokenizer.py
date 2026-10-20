"""Rank observed genes using whole-profile CP10K and reference medians."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class GeneRankTokenizer:
    def __init__(self, model_dir: str | Path, max_genes: int = 1023):
        directory = Path(model_dir)
        vocab = json.loads((directory / "gene_vocabulary.json").read_text())
        self.genes = list(vocab["genes"])
        self.special = dict(vocab["special_tokens"])
        self.medians = np.load(directory / "gene_reference_medians.npy").astype(np.float32)
        if len(self.genes) != len(self.medians):
            raise ValueError("Vocabulary and reference median lengths differ")
        self.gene_to_token = {gene: index + len(self.special) for index, gene in enumerate(self.genes)}
        self.max_genes = int(max_genes)
        if self.max_genes < 1 or self.max_genes + 1 > 1024:
            raise ValueError("max_genes must be between 1 and 1023")

    def encode(self, gene_ids: Sequence[str], raw_counts: Sequence[float],
               total_counts_all_features: float | None = None) -> dict[str, np.ndarray]:
        """Tokenize one raw-count profile with deterministic token-ID tie breaking.

        If the input omits measured features outside the vocabulary, provide
        their complete total through total_counts_all_features.
        """
        counts = np.asarray(raw_counts, dtype=np.float32)
        if counts.ndim != 1 or len(counts) != len(gene_ids):
            raise ValueError("gene_ids and raw_counts must be equal-length vectors")
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Counts must be finite and nonnegative")
        if len(set(gene_ids)) != len(gene_ids):
            raise ValueError("Duplicate gene IDs must be resolved before tokenization")
        observed_total = float(counts.sum())
        total = observed_total if total_counts_all_features is None else float(total_counts_all_features)
        if not np.isfinite(total) or total < observed_total - 1e-5 or total <= 0:
            raise ValueError("The whole-profile count total must be positive and include all supplied counts")
        pairs = [(self.gene_to_token[gene], count) for gene, count in zip(gene_ids, counts)
                 if gene in self.gene_to_token and count > 0]
        if not pairs:
            raise ValueError("No positive-count vocabulary genes in profile")
        tokens = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        cp10k = np.asarray([pair[1] for pair in pairs], dtype=np.float32) / total * 10_000.0
        medians = self.medians[tokens - len(self.special)]
        medians = np.where(np.isfinite(medians) & (medians > 0), medians, np.float32(1.0))
        scores = cp10k / medians
        order = np.lexsort((tokens, -scores))[:self.max_genes]
        selected = tokens[order]
        input_ids = np.full(self.max_genes + 1, self.special["<pad>"], dtype=np.int64)
        attention = np.zeros(self.max_genes + 1, dtype=np.int64)
        input_ids[0] = self.special["<cls>"]
        input_ids[1:len(selected) + 1] = selected
        attention[:len(selected) + 1] = 1
        return {"input_ids": input_ids, "attention_mask": attention,
                "selected_gene_ids": np.asarray([self.genes[token - len(self.special)] for token in selected])}
