"""Paired statistical inference from saved evaluation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from metacellformer.names import read_table, model_directory


def exact_signflip(delta: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Two-sided exact sign-flip P value over independent paired blocks."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or not 1 <= len(delta) <= 20 or not np.isfinite(delta).all():
        raise ValueError("Exact sign-flip requires 1–20 finite paired blocks")
    if weights is None:
        weights = np.ones(len(delta), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != delta.shape or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be positive and match the paired blocks")
    weighted = delta * (weights / weights.sum())
    observed = abs(float(weighted.sum()))
    extreme = 0
    n_states = 1 << len(delta)
    bits = np.arange(len(delta), dtype=np.uint32)
    for start in range(0, n_states, 16384):
        states = np.arange(start, min(start + 16384, n_states), dtype=np.uint32)
        signs = np.where(((states[:, None] >> bits) & 1) == 1, 1.0, -1.0)
        extreme += int(np.count_nonzero(np.abs(signs @ weighted) >= observed - 1e-12))
    return extreme / n_states


def paired_inference(first: np.ndarray, second: np.ndarray, seed: int = 42,
                     bootstrap_reps: int = 2000, signflip_reps: int = 50000) -> dict:
    """Paired percentile CI and Monte Carlo sign-flip P value."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("Paired vectors must have equal, nonzero length")
    delta = first - second
    if not np.isfinite(delta).all() or bootstrap_reps < 1 or signflip_reps < 1:
        raise ValueError("Values and replicate counts must be valid")
    rng = np.random.default_rng(seed)
    n = len(delta)
    bootstrap = np.empty(bootstrap_reps)
    for start in range(0, bootstrap_reps, 1000):
        size = min(1000, bootstrap_reps - start)
        sample = rng.integers(0, n, size=(size, n))
        bootstrap[start:start + size] = delta[sample].mean(axis=1)
    observed = float(delta.mean())
    extreme = 0
    for start in range(0, signflip_reps, 1000):
        size = min(1000, signflip_reps - start)
        signs = rng.choice((-1, 1), size=(size, n))
        extreme += int((np.abs((signs * delta).mean(axis=1)) >= abs(observed)).sum())
    return {"n_pairs": n, "mean_difference": observed,
            "ci_95_low": float(np.quantile(bootstrap, 0.025)),
            "ci_95_high": float(np.quantile(bootstrap, 0.975)),
            "p_two_sided": (extreme + 1) / (signflip_reps + 1), "seed": seed,
            "bootstrap_reps": bootstrap_reps, "signflip_reps": signflip_reps}


def cell_type_annotation(root: Path) -> dict:
    table = read_table(root / "tasks/cell_type_annotation/per_dataset_scores.tsv", sep="\t")
    wide = table[table.model_id.isin(("MetacellFormer", "MetacellFormer-SC"))].pivot(
        index="dataset", columns="model_id", values="metric_primary")
    if wide.shape != (9, 2) or wide.isna().any().any():
        raise ValueError("Expected nine paired cell-type annotation datasets")
    delta = (wide["MetacellFormer"] - wide["MetacellFormer-SC"]).to_numpy()
    return {"analysis": "cell_type_annotation", "n_datasets": len(delta),
            "mean_difference": float(delta.mean()), "p_exact_signflip": exact_signflip(delta)}


def protein_interaction_prediction(root: Path) -> dict:
    folder = root / "tasks/protein_interaction_prediction"
    mc = read_table(folder / "MetacellFormer/fold_metrics.tsv", sep="\t").set_index("fold")
    sc = read_table(folder / "MetacellFormer-SC/fold_metrics.tsv", sep="\t").set_index("fold")
    if not mc.index.equals(sc.index) or len(mc) != 20 or mc.index.has_duplicates:
        raise ValueError("Expected 20 matched HuRI folds")
    delta = (mc.metric_primary - sc.metric_primary).to_numpy()
    return {"analysis": "protein_interaction_prediction", "n_folds": len(delta),
            "mean_difference": float(delta.mean()), "p_exact_signflip": exact_signflip(delta)}


def same_group_neighbors(root: Path) -> dict:
    """Fig. 6a: average sampling repeats, then flip seven functional groups."""
    table = read_table(root / "tasks/perturbation_response_structure/group_seed.tsv", sep="\t")
    table["model"] = table.model.replace({"MetacellFormer-SC": "MetacellFormer-SC"})
    wide = table.pivot(index=["seed", "author38_group"], columns="model",
                       values="mean_same_group_fraction")
    if wide.shape != (70, 2) or wide.isna().any().any():
        raise ValueError("Expected 10 seeds across seven paired groups")
    counts = table.groupby("author38_group").n_targets.agg(["first", "nunique"])
    if not counts["nunique"].eq(1).all() or int(counts["first"].sum()) != 152:
        raise ValueError("Expected fixed group sizes for 152 targets")
    delta = (wide["MetacellFormer"] - wide["MetacellFormer-SC"]).groupby(
        "author38_group").mean().sort_index()
    weights = counts.loc[delta.index, "first"].to_numpy(dtype=float)
    return {"analysis": "fig6a_same_group_neighbors", "n_groups": len(delta),
            "n_targets": int(weights.sum()),
            "mean_difference": float(np.average(delta, weights=weights)),
            "p_exact_signflip": exact_signflip(delta.to_numpy(), weights)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="analysis", required=True)
    for name in ("cell_type_annotation", "protein_interaction_prediction"):
        sub.add_parser(name).add_argument("--root", type=Path, required=True)
    fig6a = sub.add_parser("fig6a", help="Fig. 6a paired test from submitted Source Data")
    fig6a.add_argument("--root", type=Path, required=True,
                       help="result directory containing tasks/")
    generic = sub.add_parser("paired", help="Two paired columns from a TSV table")
    generic.add_argument("table", type=Path)
    generic.add_argument("--first", required=True)
    generic.add_argument("--second", required=True)
    generic.add_argument("--seed", type=int, default=42)
    generic.add_argument("--bootstrap-reps", type=int, default=2000)
    generic.add_argument("--signflip-reps", type=int, default=50000)
    args = parser.parse_args()
    if args.analysis == "cell_type_annotation":
        result = cell_type_annotation(args.root)
    elif args.analysis == "protein_interaction_prediction":
        result = protein_interaction_prediction(args.root)
    elif args.analysis == "fig6a":
        result = same_group_neighbors(args.root)
    else:
        table = read_table(args.table, sep="\t")
        result = paired_inference(table[args.first], table[args.second], args.seed,
                                  args.bootstrap_reps, args.signflip_reps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
