"""Portable HuRI cosine scorer on the frozen 20-fold U_bio contract."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


from benchmarks.shared._common import MODELS



def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Match the tie-aware AP used for the Fig. 3 cosine scores."""
    order = np.argsort(scores)
    total_positive = int(np.count_nonzero(y))
    if not total_positive:
        return float("nan")
    tp = 0.0
    total = 0.0
    previous_tp = 0.0
    ap = 0.0
    for j in range(len(y) - 1, -1, -1):
        i = order[j]
        total += 1
        if y[i] == 1:
            tp += 1
        if j == 0 or scores[order[j - 1]] != scores[i]:
            ap += (tp - previous_tp) / total_positive * (tp / total)
            previous_tp = tp
    return float(ap)


def symbol_to_entrez(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: header.index(name) for name in ("GeneID", "Symbol", "Synonyms")}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            gene_id = fields[index["GeneID"]]
            if gene_id == "-":
                continue
            for symbol in [fields[index["Symbol"]], *fields[index["Synonyms"]].split("|")]:
                if symbol not in {"-", "NEWENTRY", ""}:
                    mapping.setdefault(symbol.upper(), gene_id)
    return mapping


def cosine_for_pairs(path: Path, nodes: np.ndarray, endpoints: np.ndarray,
                     symbol_map: dict[str, str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        genes = data["genes"].astype(str)
        embedding = data["embedding"].astype(np.float32)
    if embedding.ndim != 2 or len(genes) != len(embedding) or len(set(genes)) != len(genes):
        raise ValueError(f"Invalid static embedding: {path}")
    norms = np.linalg.norm(embedding, axis=1)
    index: dict[str, int] = {}
    for i, gene in enumerate(genes):
        if np.isfinite(norms[i]) and norms[i] > 0:
            entrez = symbol_map.get(gene.upper())
            if entrez is not None:
                index.setdefault(entrez, i)
    missing = set(nodes) - index.keys()
    if missing:
        raise ValueError(f"{path.name}: missing {len(missing)} frozen HuRI nodes")
    rows = np.asarray([index[node] for node in nodes], dtype=np.int32)
    scores = np.empty(len(endpoints), dtype=np.float32)
    for start in range(0, len(endpoints), 4096):
        stop = min(start + 4096, len(endpoints))
        batch = endpoints[start:stop]
        left = embedding[rows[batch[:, 0]]]
        right = embedding[rows[batch[:, 1]]]
        left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
        right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
        scores[start:stop] = (left * right).sum(axis=1)
    return scores


def run(root: Path, model: str, embedding_path: Path, output_root: Path) -> None:
    contract = root / "contracts" / "protein_interaction_prediction"
    with np.load(contract / "shared_pairs_folds.npz", allow_pickle=False) as z:
        nodes = z["nodes"].astype(str)
        pairs = z["pairs"].astype(str)
        labels = z["labels"].astype(np.int8)
        outer = z["outer_fold"].astype(np.int8)
    if len(nodes) != 7826 or len(pairs) != 88581 or set(np.unique(outer)) != set(range(20)):
        raise ValueError("Unexpected frozen HuRI 20-fold contract")
    node_index = {node: i for i, node in enumerate(nodes)}
    endpoints = np.asarray([[node_index[a], node_index[b]] for a, b in pairs], dtype=np.int32)
    eligible = outer[endpoints[:, 0]] == outer[endpoints[:, 1]]
    pair_index = np.flatnonzero(eligible)
    fold_index = outer[endpoints[pair_index, 0]]
    if len(pair_index) != 4538:
        raise ValueError(f"Unexpected 20-fold held-out pair count: {len(pair_index)}")
    mapping = symbol_to_entrez(contract / "Homo_sapiens.gene_info.gz")
    output = output_root
    output.mkdir(parents=True, exist_ok=True)
    for model in (model,):
        scores = cosine_for_pairs(embedding_path, nodes, endpoints, mapping)
        selected = scores[pair_index]
        rows = []
        for fold in range(20):
            mask = fold_index == fold
            y = labels[pair_index[mask]]
            if set(np.unique(y)) != {0, 1}:
                raise ValueError(f"Empty or one-class HuRI fold: {fold}")
            rows.append({"model_id": model, "fold": fold,
                         "n_nodes": int(np.count_nonzero(outer == fold)),
                         "n_test_pairs": int(mask.sum()), "n_positive": int(y.sum()),
                         "n_unknown": int(len(y) - y.sum()),
                         "metric_primary": average_precision(y, selected[mask])})
        frame = pd.DataFrame(rows)
        model_out = output / model
        model_out.mkdir(exist_ok=True)
        frame.to_csv(model_out / "fold_metrics.tsv", sep="\t", index=False)
        np.savez_compressed(model_out / "predictions.npz", fold=fold_index, pair_index=pair_index,
                            label=labels[pair_index], cosine=selected)
        print(f"HURI {model} {frame.metric_primary.mean():.15f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root, args.model, args.embedding, args.output_dir)

if __name__ == "__main__": main()
