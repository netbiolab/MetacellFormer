#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Convert standalone token-cache rows into Hugging Face dataset shards")
    ap.add_argument("--token_index_csv", required=True)
    ap.add_argument("--out_train_dir", required=True)
    ap.add_argument("--out_eval_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--summary_out", default=None)
    ap.add_argument("--max_input_size", type=int, default=1024)
    ap.add_argument("--shard_size", type=int, default=4096)
    ap.add_argument("--dataset_limit", type=int, default=None)
    ap.add_argument("--clean", action="store_true")
    return ap.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_lengths(path: Path, lengths: list[int]) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(list(map(int, lengths)), handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_chunk_dataset(
    *,
    input_ids: np.ndarray,
    value_bins: np.ndarray,
    attention_mask: np.ndarray,
    metacell_local_idx: np.ndarray,
    raw_cell_count: np.ndarray | None,
    dataset_idx: int,
    dataset_id: str,
    row_indices: np.ndarray,
    max_input_size: int,
) -> tuple[Dataset, list[int]]:
    if row_indices.size == 0:
        raise ValueError("cannot build a dataset chunk from zero rows")

    input_chunk = np.asarray(input_ids[row_indices, :max_input_size], dtype=np.int32)
    value_chunk = np.asarray(value_bins[row_indices, :max_input_size], dtype=np.int16)
    attention_chunk = np.asarray(attention_mask[row_indices, :max_input_size], dtype=np.int8)
    lengths = np.asarray(attention_chunk.sum(axis=1), dtype=np.int64)
    lengths = np.clip(lengths, 0, max_input_size)

    payload = {
            "input_ids": input_chunk,
            "value_bins": value_chunk,
            "attention_mask": attention_chunk,
            "length": lengths,
            "dataset_idx": np.full((row_indices.size,), int(dataset_idx), dtype=np.int32),
            "metacell_local_idx": np.asarray(metacell_local_idx[row_indices], dtype=np.int32),
            "dataset_id": np.asarray([str(dataset_id)] * int(row_indices.size), dtype=object),
            "source_row_idx": np.asarray(row_indices, dtype=np.int64),
    }
    if raw_cell_count is not None:
        payload["raw_cell_count"] = np.asarray(raw_cell_count[row_indices], dtype=np.int32)
    ds = Dataset.from_dict(payload)
    return ds, lengths.astype(np.int64).tolist()


def infer_split_indices(
    *,
    split_name: str,
    split_array: np.ndarray,
    indexed_rows: np.ndarray,
) -> np.ndarray:
    if indexed_rows.size > 0:
        return np.asarray(indexed_rows, dtype=np.int64)
    target = 0 if split_name == "train" else 1
    return np.flatnonzero(np.asarray(split_array, dtype=np.int64) == target).astype(np.int64)


def write_split_shards(
    *,
    split_name: str,
    out_dir: Path,
    out_prefix: str,
    shard_counter: int,
    dataset_idx: int,
    dataset_id: str,
    input_ids: np.ndarray,
    value_bins: np.ndarray,
    attention_mask: np.ndarray,
    metacell_local_idx: np.ndarray,
    raw_cell_count: np.ndarray | None,
    row_indices: np.ndarray,
    max_input_size: int,
    shard_size: int,
) -> tuple[int, list[int], int]:
    lengths_accum: list[int] = []
    rows_written = 0
    for start in range(0, int(row_indices.size), int(shard_size)):
        chunk = np.asarray(row_indices[start : start + int(shard_size)], dtype=np.int64)
        ds, lengths = build_chunk_dataset(
            input_ids=input_ids,
            value_bins=value_bins,
            attention_mask=attention_mask,
            metacell_local_idx=metacell_local_idx,
            raw_cell_count=raw_cell_count,
            dataset_idx=dataset_idx,
            dataset_id=dataset_id,
            row_indices=chunk,
            max_input_size=max_input_size,
        )
        shard_path = out_dir / f"{out_prefix}.{split_name}.shard{shard_counter:05d}.dataset"
        ds.save_to_disk(str(shard_path))
        shard_counter += 1
        rows_written += int(chunk.size)
        lengths_accum.extend(lengths)
    return shard_counter, lengths_accum, rows_written


def main() -> None:
    args = parse_args()
    token_index_csv = Path(args.token_index_csv)
    out_train_dir = Path(args.out_train_dir)
    out_eval_dir = Path(args.out_eval_dir)
    summary_out = Path(args.summary_out) if args.summary_out else (out_train_dir.parent / f"{args.out_prefix}_summary.json")

    df = pd.read_csv(token_index_csv).sort_values("dataset_idx").reset_index(drop=True)
    if args.dataset_limit is not None:
        df = df.head(int(args.dataset_limit)).copy()
    if df.empty:
        raise ValueError(f"no rows found in token index: {token_index_csv}")

    if args.clean:
        clean_output_dir(out_train_dir)
        clean_output_dir(out_eval_dir)

    ensure_dir(out_train_dir)
    ensure_dir(out_eval_dir)

    train_lengths: list[int] = []
    eval_lengths: list[int] = []
    train_dataset_counts: Counter[str] = Counter()
    eval_dataset_counts: Counter[str] = Counter()
    train_shard_counter = 0
    eval_shard_counter = 0
    source_seq_lens: list[int] = []
    dataset_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        dataset_idx = int(row["dataset_idx"])
        dataset_id = str(row["dataset_id"])

        input_ids = np.load(str(row["input_ids_path"]), mmap_mode="r")
        value_bins = np.load(str(row["value_bins_path"]), mmap_mode="r")
        attention_mask = np.load(str(row["attention_mask_path"]), mmap_mode="r")
        metacell_local_idx = np.load(str(row["metacell_path"]), mmap_mode="r")
        raw_cell_count_path = row.get("raw_cell_count_path")
        raw_cell_count = (
            np.load(str(raw_cell_count_path), mmap_mode="r")
            if raw_cell_count_path and Path(str(raw_cell_count_path)).is_file()
            else None
        )
        split_array = np.load(str(row["split_path"]), mmap_mode="r")
        train_rows = np.load(str(row["train_rows_by_mc_path"]), mmap_mode="r")
        val_rows = np.load(str(row["val_rows_by_mc_path"]), mmap_mode="r")

        if input_ids.shape != attention_mask.shape or input_ids.shape != value_bins.shape:
            raise ValueError(f"input/value/attention shape mismatch for dataset_id={dataset_id}")
        if input_ids.shape[0] != int(np.asarray(metacell_local_idx).shape[0]):
            raise ValueError(f"input/metacell shape mismatch for dataset_id={dataset_id}")

        seq_len = int(input_ids.shape[1])
        source_seq_lens.append(seq_len)
        train_indices = infer_split_indices(
            split_name="train",
            split_array=split_array,
            indexed_rows=train_rows,
        )
        eval_indices = infer_split_indices(
            split_name="eval",
            split_array=split_array,
            indexed_rows=val_rows,
        )

        train_shard_counter, new_train_lengths, train_rows_written = write_split_shards(
            split_name="train",
            out_dir=out_train_dir,
            out_prefix=args.out_prefix,
            shard_counter=train_shard_counter,
            dataset_idx=dataset_idx,
            dataset_id=dataset_id,
            input_ids=input_ids,
            value_bins=value_bins,
            attention_mask=attention_mask,
            metacell_local_idx=metacell_local_idx,
            raw_cell_count=raw_cell_count,
            row_indices=train_indices,
            max_input_size=int(args.max_input_size),
            shard_size=int(args.shard_size),
        )
        eval_shard_counter, new_eval_lengths, eval_rows_written = write_split_shards(
            split_name="eval",
            out_dir=out_eval_dir,
            out_prefix=args.out_prefix,
            shard_counter=eval_shard_counter,
            dataset_idx=dataset_idx,
            dataset_id=dataset_id,
            input_ids=input_ids,
            value_bins=value_bins,
            attention_mask=attention_mask,
            metacell_local_idx=metacell_local_idx,
            raw_cell_count=raw_cell_count,
            row_indices=eval_indices,
            max_input_size=int(args.max_input_size),
            shard_size=int(args.shard_size),
        )

        train_lengths.extend(new_train_lengths)
        eval_lengths.extend(new_eval_lengths)
        if train_rows_written:
            train_dataset_counts[dataset_id] += int(train_rows_written)
        if eval_rows_written:
            eval_dataset_counts[dataset_id] += int(eval_rows_written)

        dataset_rows.append(
            {
                "dataset_idx": dataset_idx,
                "dataset_id": dataset_id,
                "source_seq_len": seq_len,
                "train_rows": int(train_rows_written),
                "eval_rows": int(eval_rows_written),
                "train_rows_indexed": int(train_rows.size),
                "eval_rows_indexed": int(val_rows.size),
            }
        )

    if not train_lengths:
        raise ValueError("converted train split is empty")
    if not eval_lengths:
        raise ValueError("converted eval split is empty")

    train_lengths_path = out_train_dir / f"{args.out_prefix}_train_lengths.pkl"
    eval_lengths_path = out_eval_dir / f"{args.out_prefix}_eval_lengths.pkl"
    save_lengths(train_lengths_path, train_lengths)
    save_lengths(eval_lengths_path, eval_lengths)

    summary = {
        "token_index_csv": str(token_index_csv),
        "out_train_dir": str(out_train_dir),
        "out_eval_dir": str(out_eval_dir),
        "out_prefix": args.out_prefix,
        "dataset_limit": args.dataset_limit,
        "datasets_processed": int(len(dataset_rows)),
        "source_seq_lens": sorted(set(source_seq_lens)),
        "source_seq_len": int(source_seq_lens[0]),
        "output_seq_len": int(args.max_input_size),
        "truncation_policy": f"keep the first {int(args.max_input_size)} tokens from each cached row",
        "includes_value_bins": True,
        "includes_metacell_local_idx": True,
        "train_examples": int(len(train_lengths)),
        "eval_examples": int(len(eval_lengths)),
        "train_shards": int(train_shard_counter),
        "eval_shards": int(eval_shard_counter),
        "train_lengths_pkl": str(train_lengths_path),
        "eval_lengths_pkl": str(eval_lengths_path),
        "train_dataset_counts": dict(sorted(train_dataset_counts.items())),
        "eval_dataset_counts": dict(sorted(eval_dataset_counts.items())),
        "datasets": dataset_rows,
    }
    write_json(summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
