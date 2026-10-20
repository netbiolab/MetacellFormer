from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .binning import ValueBinner


@dataclass
class TokenizedCell:
    input_ids: np.ndarray
    gene_ids: np.ndarray
    value_bins: np.ndarray
    attention_mask: np.ndarray


def _extract_sparse_row(row: object) -> Tuple[np.ndarray, np.ndarray]:
    # scipy sparse row path
    if hasattr(row, "indices") and hasattr(row, "data"):
        idx = np.asarray(row.indices, dtype=np.int64)
        val = np.asarray(row.data, dtype=np.float32)
        return idx, val

    dense = np.asarray(row).reshape(-1).astype(np.float32)
    idx = np.nonzero(dense > 0)[0].astype(np.int64)
    val = dense[idx].astype(np.float32)
    return idx, val


def _ordering_mode_uses_rank_medians(ordering_mode: str) -> bool:
    mode = str(ordering_mode).lower()
    return mode in {
        "geneformer_median_scaled_cp10k",
        "geneformer_score_only",
        "geneformer_median_scaled_cp10k_score_only",
    }


def _empty_tokenized_cell(
    max_genes: int,
    cls_token_id: int,
    pad_token_id: int,
    value_pad_id: int,
) -> TokenizedCell:
    seq_len = max_genes + 1
    input_ids = np.full((seq_len,), pad_token_id, dtype=np.int64)
    gene_ids = np.full((seq_len,), -1, dtype=np.int64)
    value_bins = np.full((seq_len,), value_pad_id, dtype=np.uint8)
    attention = np.zeros((seq_len,), dtype=np.uint8)
    input_ids[0] = int(cls_token_id)
    attention[0] = 1
    return TokenizedCell(input_ids=input_ids, gene_ids=gene_ids, value_bins=value_bins, attention_mask=attention)


def compute_pflogpf_shifted_clr_values(
    raw_counts: np.ndarray,
    total_count: float,
    center_gene_count: int,
    pseudocount: float = 1.0,
) -> np.ndarray:
    vals = np.asarray(raw_counts, dtype=np.float32)
    d = int(center_gene_count)
    c = float(pseudocount)
    if vals.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if total_count <= 0 or d <= 0:
        return np.zeros_like(vals, dtype=np.float32)
    if c <= 0:
        raise ValueError(f"PFlogPF pseudocount must be positive, got {pseudocount}")

    u = vals.astype(np.float64, copy=False) / float(total_count)
    log_vals = np.log(u + c)
    zero_contrib = max(0, d - int(vals.size)) * np.log(c)
    center = (float(np.sum(log_vals)) + float(zero_contrib)) / float(d)
    return (log_vals - center).astype(np.float32)


def tokenize_single_cell_from_indices(
    gene_idx: np.ndarray,
    raw_counts: np.ndarray,
    gene_to_token_by_index: np.ndarray,
    max_genes: int,
    binner: ValueBinner,
    cls_token_id: int,
    pad_token_id: int,
    value_pad_id: int = 0,
    target_sum: float = 10_000.0,
    gene_subset_mask: Optional[np.ndarray] = None,
    ordering_mode: str = "cp10k_desc",
    rank_normalization_factors_by_index: Optional[np.ndarray] = None,
    rank_median_exponent: float = 1.0,
    value_transform: str = "log1p_cp10k",
    pflogpf_center_gene_count: Optional[int] = None,
    pflogpf_pseudocount: float = 1.0,
) -> TokenizedCell:
    gene_idx = np.asarray(gene_idx, dtype=np.int64)
    raw_counts = np.asarray(raw_counts, dtype=np.float32)
    total = float(np.sum(raw_counts))
    if total <= 0:
        return _empty_tokenized_cell(max_genes, cls_token_id, pad_token_id, value_pad_id)

    cp10k = (raw_counts / total) * float(target_sum)
    transform = str(value_transform).lower()
    if transform == "log1p_cp10k":
        value_expr = np.log1p(cp10k)
        subset_mask_for_tokenize = gene_subset_mask
    elif transform == "pflogpf_shifted_clr":
        if gene_subset_mask is not None:
            keep = gene_subset_mask[gene_idx]
            gene_idx = gene_idx[keep]
            raw_counts = raw_counts[keep]
            cp10k = cp10k[keep]
            subset_mask_for_tokenize = None
            center_gene_count = int(
                pflogpf_center_gene_count
                if pflogpf_center_gene_count is not None
                else np.count_nonzero(gene_subset_mask)
            )
        else:
            subset_mask_for_tokenize = None
            center_gene_count = int(
                pflogpf_center_gene_count
                if pflogpf_center_gene_count is not None
                else gene_to_token_by_index.shape[0]
            )
        if gene_idx.size == 0:
            return _empty_tokenized_cell(max_genes, cls_token_id, pad_token_id, value_pad_id)
        value_expr = compute_pflogpf_shifted_clr_values(
            raw_counts=raw_counts,
            total_count=total,
            center_gene_count=center_gene_count,
            pseudocount=pflogpf_pseudocount,
        )
    else:
        raise ValueError(f"Unsupported tokenizer.value_transform={value_transform}")

    return tokenize_expression_from_indices(
        gene_idx=gene_idx,
        log1p_cp10k=value_expr,
        cp10k_for_order=cp10k,
        gene_to_token_by_index=gene_to_token_by_index,
        max_genes=max_genes,
        binner=binner,
        cls_token_id=cls_token_id,
        pad_token_id=pad_token_id,
        value_pad_id=value_pad_id,
        gene_subset_mask=subset_mask_for_tokenize,
        ordering_mode=ordering_mode,
        rank_normalization_factors_by_index=rank_normalization_factors_by_index,
        rank_median_exponent=rank_median_exponent,
    )


def tokenize_expression_from_indices(
    gene_idx: np.ndarray,
    log1p_cp10k: np.ndarray,
    cp10k_for_order: np.ndarray,
    gene_to_token_by_index: np.ndarray,
    max_genes: int,
    binner: ValueBinner,
    cls_token_id: int,
    pad_token_id: int,
    value_pad_id: int = 0,
    gene_subset_mask: Optional[np.ndarray] = None,
    ordering_mode: str = "cp10k_desc",
    rank_normalization_factors_by_index: Optional[np.ndarray] = None,
    rank_median_exponent: float = 1.0,
) -> TokenizedCell:
    seq_len = max_genes + 1  # +1 for [CLS]

    input_ids = np.full((seq_len,), pad_token_id, dtype=np.int64)
    gene_ids = np.full((seq_len,), -1, dtype=np.int64)
    value_bins = np.full((seq_len,), value_pad_id, dtype=np.uint8)
    attention = np.zeros((seq_len,), dtype=np.uint8)

    input_ids[0] = int(cls_token_id)
    attention[0] = 1

    if gene_idx.size == 0:
        return TokenizedCell(input_ids=input_ids, gene_ids=gene_ids, value_bins=value_bins, attention_mask=attention)

    if gene_subset_mask is not None:
        keep = gene_subset_mask[gene_idx]
        gene_idx = gene_idx[keep]
        cp10k_for_order = np.asarray(cp10k_for_order, dtype=np.float32)[keep]
        log1p_cp10k = np.asarray(log1p_cp10k, dtype=np.float32)[keep]
        if gene_idx.size == 0:
            return TokenizedCell(input_ids=input_ids, gene_ids=gene_ids, value_bins=value_bins, attention_mask=attention)
    cp10k = np.asarray(cp10k_for_order, dtype=np.float32)
    logv = np.asarray(log1p_cp10k, dtype=np.float32)

    token_ids = gene_to_token_by_index[gene_idx]
    mode = str(ordering_mode).lower()
    if mode == "cp10k_desc":
        order_score = cp10k
    elif _ordering_mode_uses_rank_medians(mode):
        if rank_normalization_factors_by_index is None:
            raise ValueError("rank_normalization_factors_by_index is required for Geneformer-style ordering")
        exponent = float(rank_median_exponent)
        if not np.isfinite(exponent) or exponent < 0:
            raise ValueError(f"rank_median_exponent must be finite and nonnegative, got {rank_median_exponent}")
        norm = np.asarray(rank_normalization_factors_by_index[gene_idx], dtype=np.float32)
        norm = np.where(np.isfinite(norm) & (norm > 0), norm, 1.0).astype(np.float32, copy=False)
        order_score = cp10k / np.power(norm, exponent)
    else:
        raise ValueError(f"Unsupported tokenizer.ordering_mode={ordering_mode}")

    if mode in {"geneformer_score_only", "geneformer_median_scaled_cp10k_score_only"}:
        # Diagnostic mode for direct comparison to Geneformer-style score-only ranking.
        order = np.argsort(-order_score, kind="stable")
    else:
        # Default standalone path: descending score with deterministic token-id tie-break.
        order = np.lexsort((token_ids, -order_score))

    if order.size > max_genes:
        order = order[:max_genes]

    sel_gene = gene_idx[order]
    sel_tok = token_ids[order]
    sel_logv = logv[order]

    n = int(sel_tok.size)
    if n > 0:
        input_ids[1 : 1 + n] = sel_tok
        gene_ids[1 : 1 + n] = sel_gene
        value_bins[1 : 1 + n] = binner.transform(sel_logv)
        attention[1 : 1 + n] = 1

    return TokenizedCell(input_ids=input_ids, gene_ids=gene_ids, value_bins=value_bins, attention_mask=attention)


def tokenize_log1p_cp10k_from_indices(
    gene_idx: np.ndarray,
    log1p_cp10k: np.ndarray,
    gene_to_token_by_index: np.ndarray,
    max_genes: int,
    binner: ValueBinner,
    cls_token_id: int,
    pad_token_id: int,
    value_pad_id: int = 0,
    gene_subset_mask: Optional[np.ndarray] = None,
    ordering_mode: str = "cp10k_desc",
    rank_normalization_factors_by_index: Optional[np.ndarray] = None,
    rank_median_exponent: float = 1.0,
) -> TokenizedCell:
    logv = np.asarray(log1p_cp10k, dtype=np.float32)
    cp10k = np.expm1(logv).astype(np.float32, copy=False)
    return tokenize_expression_from_indices(
        gene_idx=gene_idx,
        log1p_cp10k=logv,
        cp10k_for_order=cp10k,
        gene_to_token_by_index=gene_to_token_by_index,
        max_genes=max_genes,
        binner=binner,
        cls_token_id=cls_token_id,
        pad_token_id=pad_token_id,
        value_pad_id=value_pad_id,
        gene_subset_mask=gene_subset_mask,
        ordering_mode=ordering_mode,
        rank_normalization_factors_by_index=rank_normalization_factors_by_index,
        rank_median_exponent=rank_median_exponent,
    )


def tokenize_single_cell_row(
    row: object,
    gene_to_token_by_index: np.ndarray,
    max_genes: int,
    binner: ValueBinner,
    cls_token_id: int,
    pad_token_id: int,
    value_pad_id: int = 0,
    target_sum: float = 10_000.0,
    gene_subset_mask: Optional[np.ndarray] = None,
    ordering_mode: str = "cp10k_desc",
    rank_normalization_factors_by_index: Optional[np.ndarray] = None,
    rank_median_exponent: float = 1.0,
    value_transform: str = "log1p_cp10k",
    pflogpf_center_gene_count: Optional[int] = None,
    pflogpf_pseudocount: float = 1.0,
) -> TokenizedCell:
    idx, vals = _extract_sparse_row(row)
    return tokenize_single_cell_from_indices(
        gene_idx=idx,
        raw_counts=vals,
        gene_to_token_by_index=gene_to_token_by_index,
        max_genes=max_genes,
        binner=binner,
        cls_token_id=cls_token_id,
        pad_token_id=pad_token_id,
        value_pad_id=value_pad_id,
        target_sum=target_sum,
        gene_subset_mask=gene_subset_mask,
        ordering_mode=ordering_mode,
        rank_normalization_factors_by_index=rank_normalization_factors_by_index,
        rank_median_exponent=rank_median_exponent,
        value_transform=value_transform,
        pflogpf_center_gene_count=pflogpf_center_gene_count,
        pflogpf_pseudocount=pflogpf_pseudocount,
    )
