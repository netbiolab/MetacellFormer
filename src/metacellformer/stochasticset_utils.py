from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


TOP_NONZERO_SOURCE_ID = 0
RANDOM_NONZERO_SOURCE_ID = 1
FILL_ZERO_SOURCE_ID = 2

RANK_BUCKET_EDGES = (32, 128, 512, 2048)


def gene_token_universe_from_token_dictionary(token_dictionary: dict[str, int]) -> np.ndarray:
    special_ids = {
        int(token_id)
        for token, token_id in token_dictionary.items()
        if str(token).startswith("<")
    }
    gene_ids = sorted(int(token_id) for token_id in token_dictionary.values() if int(token_id) not in special_ids)
    return np.asarray(gene_ids, dtype=np.int32)


def feature_to_int_array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _rank_bucket(raw_rank_indices: np.ndarray) -> np.ndarray:
    if raw_rank_indices.size == 0:
        return np.zeros((0,), dtype=np.int8)
    buckets = np.zeros((raw_rank_indices.size,), dtype=np.int8)
    for bucket_id, edge in enumerate(RANK_BUCKET_EDGES, start=1):
        buckets[raw_rank_indices >= int(edge)] = int(bucket_id)
    return buckets


def _sample_fill_from_complement(
    *,
    gene_universe: np.ndarray,
    observed_gene_token_ids: np.ndarray,
    n_fill: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_fill <= 0:
        return np.zeros((0,), dtype=np.int32)
    max_token_id = int(gene_universe.max()) + 1
    excluded = np.zeros((max_token_id,), dtype=bool)
    excluded[observed_gene_token_ids.astype(np.int64, copy=False)] = True
    fill_candidates = gene_universe[~excluded[gene_universe.astype(np.int64, copy=False)]]
    if fill_candidates.size < n_fill:
        raise ValueError(
            f"not enough complement genes to fill row: need={n_fill} candidates={int(fill_candidates.size)}"
        )
    if fill_candidates.size == n_fill:
        return np.asarray(fill_candidates, dtype=np.int32)
    picked = rng.choice(fill_candidates.size, size=n_fill, replace=False)
    return np.asarray(fill_candidates[np.sort(picked)], dtype=np.int32)


def sample_candidate_subset(
    feature: dict[str, Any],
    *,
    cls_token_id: int,
    max_input_size: int,
    value_pad_id: int,
    gene_universe: np.ndarray,
    input_subset_mode: str,
    subset_top_count: int,
    subset_random_count: int,
    subset_fill_count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    k = int(max_input_size) - 1
    observed_gene_token_ids = feature_to_int_array(feature["observed_gene_token_ids"], dtype=np.int32)
    observed_value_bins = feature_to_int_array(feature["observed_value_bins"], dtype=np.int16)
    if observed_gene_token_ids.ndim != 1 or observed_value_bins.ndim != 1:
        raise ValueError("observed candidate arrays must be rank-1")
    if observed_gene_token_ids.size != observed_value_bins.size:
        raise ValueError(
            "observed_gene_token_ids / observed_value_bins length mismatch: "
            f"{observed_gene_token_ids.size} vs {observed_value_bins.size}"
        )
    if observed_gene_token_ids.size and np.unique(observed_gene_token_ids).size != observed_gene_token_ids.size:
        raise ValueError("duplicate observed genes inside candidate cache row")
    if np.any(observed_value_bins <= 0):
        raise ValueError("candidate cache should contain nonzero observed bins only")

    observed_count = int(observed_gene_token_ids.size)
    observed_rank = np.arange(observed_count, dtype=np.int32)

    if input_subset_mode == "static_top_nonzero_4096":
        take_top = min(k, observed_count)
        top_idx = np.arange(take_top, dtype=np.int32)
        random_idx = np.zeros((0,), dtype=np.int32)
    elif input_subset_mode == "stochastic_mixed_4096":
        take_top = min(int(subset_top_count), k, observed_count)
        top_idx = np.arange(take_top, dtype=np.int32)
        remaining_slots = k - take_top
        remaining_obs = max(0, observed_count - take_top)
        take_random = min(int(subset_random_count), remaining_slots, remaining_obs)
        if take_random > 0:
            pool = np.arange(take_top, observed_count, dtype=np.int32)
            chosen = rng.choice(pool.size, size=take_random, replace=False)
            random_idx = np.sort(pool[chosen])
        else:
            random_idx = np.zeros((0,), dtype=np.int32)
    else:
        raise ValueError(f"unsupported input_subset_mode={input_subset_mode}")

    selected_obs_idx = np.concatenate([top_idx, random_idx], axis=0)
    selected_obs_gene_ids = observed_gene_token_ids[selected_obs_idx]
    selected_obs_values = observed_value_bins[selected_obs_idx]
    selected_obs_rank = observed_rank[selected_obs_idx]
    selected_obs_sources = np.concatenate(
        [
            np.full((top_idx.size,), TOP_NONZERO_SOURCE_ID, dtype=np.int8),
            np.full((random_idx.size,), RANDOM_NONZERO_SOURCE_ID, dtype=np.int8),
        ],
        axis=0,
    )

    n_fill = max(0, k - int(selected_obs_idx.size))
    fill_gene_ids = _sample_fill_from_complement(
        gene_universe=gene_universe,
        observed_gene_token_ids=observed_gene_token_ids,
        n_fill=n_fill,
        rng=rng,
    )
    fill_values = np.zeros((n_fill,), dtype=np.int16)
    fill_rank = np.full((n_fill,), max(observed_count, 1_000_000), dtype=np.int32)
    fill_sources = np.full((n_fill,), FILL_ZERO_SOURCE_ID, dtype=np.int8)

    combined_gene_ids = np.concatenate([selected_obs_gene_ids, fill_gene_ids], axis=0).astype(np.int32, copy=False)
    combined_values = np.concatenate([selected_obs_values, fill_values], axis=0).astype(np.int16, copy=False)
    combined_rank = np.concatenate([selected_obs_rank, fill_rank], axis=0).astype(np.int32, copy=False)
    combined_sources = np.concatenate([selected_obs_sources, fill_sources], axis=0).astype(np.int8, copy=False)

    if combined_gene_ids.size != k:
        raise ValueError(f"expected exactly {k} selected genes, got {combined_gene_ids.size}")
    if np.unique(combined_gene_ids).size != combined_gene_ids.size:
        raise ValueError("duplicate gene ids after subset sampling")

    perm = rng.permutation(k)
    shuffled_gene_ids = combined_gene_ids[perm]
    shuffled_values = combined_values[perm]
    shuffled_sources = combined_sources[perm]
    shuffled_rank_bucket = _rank_bucket(combined_rank)[perm]

    input_ids = np.empty((max_input_size,), dtype=np.int32)
    input_ids[0] = int(cls_token_id)
    input_ids[1:] = shuffled_gene_ids

    value_bins = np.empty((max_input_size,), dtype=np.int16)
    value_bins[0] = int(value_pad_id)
    value_bins[1:] = shuffled_values

    attention_mask = np.ones((max_input_size,), dtype=np.int8)
    subset_source_ids = np.full((max_input_size,), -1, dtype=np.int8)
    subset_source_ids[1:] = shuffled_sources
    pre_shuffle_rank_bucket = np.full((max_input_size,), -1, dtype=np.int8)
    pre_shuffle_rank_bucket[1:] = shuffled_rank_bucket

    return {
        "input_ids": input_ids,
        "value_bins": value_bins,
        "attention_mask": attention_mask,
        "subset_source_ids": subset_source_ids,
        "pre_shuffle_rank_bucket": pre_shuffle_rank_bucket,
        "observed_count": np.asarray(observed_count, dtype=np.int32),
    }


def apply_value_masking(
    materialized_row: dict[str, np.ndarray],
    *,
    value_mask_ratio: float,
    value_mask_id: int,
    value_target_mode: str,
    target_nonzero_fraction: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int | float]:
    value_bins = np.asarray(materialized_row["value_bins"], dtype=np.int16)
    subset_source_ids = np.asarray(materialized_row["subset_source_ids"], dtype=np.int8)
    seq_len = int(value_bins.shape[0])
    k = seq_len - 1
    target_n = int(round(k * float(value_mask_ratio)))
    gene_value_bins = value_bins[1:]
    gene_subset_source_ids = subset_source_ids[1:]

    nonzero_idx = np.flatnonzero(gene_value_bins > 0).astype(np.int32)
    lowzero_idx = np.flatnonzero(gene_value_bins == 0).astype(np.int32)

    masked_gene_positions: np.ndarray
    if value_target_mode == "observed_nonzero_only":
        take_n = min(target_n, int(nonzero_idx.size))
        if take_n > 0:
            chosen = rng.choice(nonzero_idx.size, size=take_n, replace=False)
            masked_gene_positions = np.sort(nonzero_idx[chosen])
        else:
            masked_gene_positions = np.zeros((0,), dtype=np.int32)
    elif value_target_mode == "mixed_nonzero_lowzero":
        want_nonzero = int(round(target_n * float(target_nonzero_fraction)))
        want_nonzero = min(want_nonzero, target_n)
        want_lowzero = max(0, target_n - want_nonzero)

        take_nonzero = min(want_nonzero, int(nonzero_idx.size))
        take_lowzero = min(want_lowzero, int(lowzero_idx.size))
        if take_nonzero + take_lowzero < target_n:
            remaining = target_n - (take_nonzero + take_lowzero)
            extra_nonzero = min(remaining, int(nonzero_idx.size) - take_nonzero)
            take_nonzero += max(0, extra_nonzero)
            remaining = target_n - (take_nonzero + take_lowzero)
            extra_lowzero = min(remaining, int(lowzero_idx.size) - take_lowzero)
            take_lowzero += max(0, extra_lowzero)

        chosen_parts: list[np.ndarray] = []
        if take_nonzero > 0:
            chosen = rng.choice(nonzero_idx.size, size=take_nonzero, replace=False)
            chosen_parts.append(np.sort(nonzero_idx[chosen]))
        if take_lowzero > 0:
            chosen = rng.choice(lowzero_idx.size, size=take_lowzero, replace=False)
            chosen_parts.append(np.sort(lowzero_idx[chosen]))
        masked_gene_positions = (
            np.sort(np.concatenate(chosen_parts, axis=0)).astype(np.int32, copy=False)
            if chosen_parts
            else np.zeros((0,), dtype=np.int32)
        )
    else:
        raise ValueError(f"unsupported value_target_mode={value_target_mode}")

    value_labels = np.full((seq_len,), -100, dtype=np.int16)
    masked_positions = masked_gene_positions + 1
    if masked_positions.size > 0:
        value_labels[masked_positions] = value_bins[masked_positions]
    corrupted_value_bins = np.array(value_bins, copy=True)
    corrupted_value_bins[masked_positions] = int(value_mask_id)

    masked_subset_sources = gene_subset_source_ids[masked_gene_positions] if masked_gene_positions.size else np.zeros((0,), dtype=np.int8)
    nonzero_target_n = int((value_labels[masked_positions] > 0).sum()) if masked_positions.size else 0
    lowzero_target_n = int((value_labels[masked_positions] == 0).sum()) if masked_positions.size else 0

    return {
        "value_bins": corrupted_value_bins,
        "value_labels": value_labels,
        "masked_positions": masked_positions.astype(np.int32, copy=False),
        "masked_target_n": int(masked_positions.size),
        "observed_nonzero_target_n": int(nonzero_target_n),
        "lowzero_target_n": int(lowzero_target_n),
        "realized_value_mask_ratio": float(masked_positions.size / max(1, k)),
        "realized_nonzero_target_fraction": float(nonzero_target_n / max(1, masked_positions.size)),
        "masked_subset_source_ids": masked_subset_sources.astype(np.int8, copy=False),
    }


@dataclass
class CandidateShard:
    dataset_idx: int
    dataset_id: str
    n_rows: int
    gene_token_ids: np.memmap
    value_bins: np.memmap
    row_offsets: np.memmap
    metacell_local_idx: np.memmap


class FullObservedCandidateDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest_path = self.root / "candidate_cache_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing candidate cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("format_version")) != "full_observed_candidate_cache_v1":
            raise ValueError(f"unsupported candidate cache format: {manifest.get('format_version')}")

        self.manifest = manifest
        self.max_input_size = int(manifest["max_input_size"])
        self.total_rows = int(manifest["rows"])
        self.gene_universe_size = int(manifest["gene_universe_size"])
        self.value_pad_id = int(manifest["value_pad_id"])
        self.value_mask_id = int(manifest["value_mask_id"])
        self.n_value_bins = int(manifest["n_value_bins"])

        self.shards: list[CandidateShard] = []
        self.cumulative_rows: list[int] = []
        running = 0
        for shard_meta in manifest["shards"]:
            shard_dir = self.root / str(shard_meta["shard_dir"])
            n_rows = int(shard_meta["n_rows"])
            shard = CandidateShard(
                dataset_idx=int(shard_meta["dataset_idx"]),
                dataset_id=str(shard_meta["dataset_id"]),
                n_rows=n_rows,
                gene_token_ids=np.load(str(shard_dir / "observed_gene_token_ids.npy"), mmap_mode="r"),
                value_bins=np.load(str(shard_dir / "observed_value_bins.npy"), mmap_mode="r"),
                row_offsets=np.load(str(shard_dir / "row_offsets.npy"), mmap_mode="r"),
                metacell_local_idx=np.load(str(shard_dir / "metacell_local_idx.npy"), mmap_mode="r"),
            )
            if int(shard.row_offsets.shape[0]) != n_rows + 1:
                raise ValueError(f"row_offsets shape mismatch in shard {shard_dir}")
            self.shards.append(shard)
            running += n_rows
            self.cumulative_rows.append(running)
        if running != self.total_rows:
            raise ValueError(f"manifest rows mismatch: manifest={self.total_rows} discovered={running}")

    def __len__(self) -> int:
        return self.total_rows

    def _locate(self, index: int) -> tuple[CandidateShard, int]:
        if index < 0:
            index += self.total_rows
        if index < 0 or index >= self.total_rows:
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cumulative_rows, index)
        shard = self.shards[shard_idx]
        shard_start = 0 if shard_idx == 0 else self.cumulative_rows[shard_idx - 1]
        local_idx = index - shard_start
        return shard, local_idx

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard, local_idx = self._locate(index)
        start = int(shard.row_offsets[local_idx])
        end = int(shard.row_offsets[local_idx + 1])
        observed_gene_token_ids = np.asarray(shard.gene_token_ids[start:end], dtype=np.int32)
        observed_value_bins = np.asarray(shard.value_bins[start:end], dtype=np.int16)
        return {
            "observed_gene_token_ids": observed_gene_token_ids,
            "observed_value_bins": observed_value_bins,
            "observed_count": int(end - start),
            "dataset_idx": int(shard.dataset_idx),
            "dataset_id": str(shard.dataset_id),
            "metacell_local_idx": int(shard.metacell_local_idx[local_idx]),
            "length": int(self.max_input_size),
        }

