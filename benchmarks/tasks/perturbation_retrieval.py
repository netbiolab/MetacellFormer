"""Original cross-batch exact and functional perturbation retrieval (Figs. 3, 4).

Cell L2 -> target/batch mean minus batch-control mean -> delta L2. Stable
rank AP is used here; it differs from threshold-based classification AP.
Functional retrieval excludes both same-batch and same-target candidates.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import argparse
import numpy as np
import pandas as pd
import torch
from benchmarks.shared._common import MODELS, embeddings, write_table
from benchmarks.shared._retrieval_metrics import METRIC_NAMES,metrics_from_relevance,validate_metric_arrays
NORMALIZATIONS={'l2cell':'L2 cell -> mean -> subtract batch control -> L2 delta'}
TOPK=10

def make_delta_items(emb: np.ndarray, index: pd.DataFrame, normalization: str, min_cells: int) -> tuple[np.ndarray, pd.DataFrame]:
    if normalization not in NORMALIZATIONS:
        raise ValueError(normalization)
    if normalization == "l2cell":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("zero-norm cell embedding in l2cell variant")
        cells = emb / norms
    else:
        cells = emb
    index = index.copy().reset_index(drop=True)
    control = index[index["cell_role"] == "control"]
    control_batches, control_codes = np.unique(control["batch_group"].astype(str).to_numpy(), return_inverse=True)
    control_order = np.argsort(control_codes, kind="stable")
    control_starts = np.r_[0, np.flatnonzero(np.diff(control_codes[control_order])) + 1]
    control_sums = np.add.reduceat(cells[control.index.to_numpy()[control_order]], control_starts, axis=0)
    control_counts = np.bincount(control_codes)
    controls = control_sums / control_counts[:, None]
    control_lookup = {str(batch): pos for pos, batch in enumerate(control_batches)}

    pert = index[index["cell_role"] == "perturbed"]
    item_names = (pert["target_gene"].astype(str) + "|batch" + pert["batch_group"].astype(str)).to_numpy()
    unique_items, item_codes = np.unique(item_names, return_inverse=True)
    item_order = np.argsort(item_codes, kind="stable")
    item_starts = np.r_[0, np.flatnonzero(np.diff(item_codes[item_order])) + 1]
    item_sums = np.add.reduceat(cells[pert.index.to_numpy()[item_order]], item_starts, axis=0)
    item_counts = np.bincount(item_codes)
    item_targets = np.asarray([name.rsplit("|batch", 1)[0] for name in unique_items], dtype=str)
    item_batches = np.asarray([name.rsplit("|batch", 1)[1] for name in unique_items], dtype=str)
    eligible = np.asarray([batch in control_lookup for batch in item_batches], dtype=bool)
    eligible &= item_counts >= int(min_cells)
    if not eligible.any():
        raise RuntimeError(f"no delta items for {normalization}, min_cells={min_cells}")
    eligible_positions = np.flatnonzero(eligible)
    vectors = item_sums[eligible_positions] / item_counts[eligible_positions, None]
    vectors -= controls[[control_lookup[batch] for batch in item_batches[eligible_positions]]]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        bad = np.flatnonzero((~np.isfinite(norms[:, 0])) | (norms[:, 0] <= 1e-12))[:10].tolist()
        raise RuntimeError(f"zero/non-finite delta vectors at positions {bad} ({normalization}, min={min_cells})")
    vectors = vectors / norms
    meta = pd.DataFrame(
        {
            "item_id": unique_items[eligible_positions],
            "target_gene": item_targets[eligible_positions],
            "label": item_targets[eligible_positions],
            "batch_group": item_batches[eligible_positions],
            "group": item_batches[eligible_positions],
            "split_id": [f"replogle_k562_essential_batchheldout_batch{batch}" for batch in item_batches[eligible_positions]],
            "n_cells": item_counts[eligible_positions].astype(int),
        }
    ).sort_values("item_id").reset_index(drop=True)
    if meta.empty:
        raise RuntimeError(f"no delta items for {normalization}, min_cells={min_cells}")
    vector_positions = {item: pos for item, pos in zip(unique_items[eligible_positions], range(len(eligible_positions)))}
    arr = np.vstack([vectors[vector_positions[item]] for item in meta["item_id"]]).astype(np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError("non-finite delta vectors")
    return arr, meta

def torch_metrics(scores: "torch.Tensor", positive: "torch.Tensor", n_candidate: "torch.Tensor | None" = None) -> dict[str, np.ndarray]:
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise RuntimeError("NaN or positive-infinite similarity matrix")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ordered = positive.gather(1, order)
    if n_candidate is not None:
        positions = torch.arange(scores.shape[1], device=scores.device)[None, :]
        ordered = ordered & (positions < n_candidate[:, None])
    n_pos = ordered.sum(dim=1)
    if torch.any(n_pos <= 0):
        raise RuntimeError("query without positive reference in torch evaluation")
    ranks = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=torch.float32)[None, :]
    cumulative = ordered.cumsum(dim=1).to(torch.float32)
    ap = ((cumulative / ranks) * ordered.to(torch.float32)).sum(dim=1) / n_pos.to(torch.float32)
    first = torch.argmax(ordered.to(torch.int64), dim=1).to(torch.float32) + 1.0
    k = min(TOPK, scores.shape[1])
    out = {
        "recall_at_1": ordered[:, :1].any(dim=1).float(),
        "recall_at_5": ordered[:, : min(5, scores.shape[1])].any(dim=1).float(),
        "recall_at_10": ordered[:, : min(10, scores.shape[1])].any(dim=1).float(),
        "mrr": 1.0 / first,
        "map": ap,
        "purity_at_10": ordered[:, :k].sum(dim=1).float() / float(k),
    }
    return {key: value.detach().cpu().numpy().astype(np.float64) for key, value in out.items()}

def evaluate_retrieval(X: np.ndarray, meta: pd.DataFrame, task: str, device: str, query_mask: np.ndarray | None = None) -> pd.DataFrame:
    labels = meta["label"].astype(str).to_numpy()
    genes = meta["target_gene"].astype(str).to_numpy()
    label_codes, _ = pd.factorize(labels, sort=True)
    gene_codes, _ = pd.factorize(genes, sort=True)
    groups = meta["group"].astype(str).to_numpy()
    item_ids = meta["item_id"].astype(str).to_numpy()
    if query_mask is None:
        query_mask = np.ones(len(meta), dtype=bool)
    query_mask = np.asarray(query_mask, dtype=bool)
    if query_mask.shape != (len(meta),):
        raise ValueError("query_mask shape mismatch")
    rows: list[dict[str, Any]] = []
    use_torch = device == "cuda"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device) if use_torch else None
    for batch in sorted(set(groups), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
        q_idx = np.flatnonzero((groups == batch) & query_mask)
        r_idx = np.flatnonzero(groups != batch)
        if not q_idx.size:
            continue
        if not r_idx.size:
            raise RuntimeError(f"empty reference batch partition: {batch}")
        if set(item_ids[q_idx]).intersection(set(item_ids[r_idx])):
            raise RuntimeError(f"query/reference item overlap in batch {batch}")
        if use_torch:
            with torch.no_grad():
                scores = Xt[torch.as_tensor(q_idx, device=device)] @ Xt[torch.as_tensor(r_idx, device=device)].T
                q_gene = torch.as_tensor(gene_codes[q_idx], device=device)
                r_gene = torch.as_tensor(gene_codes[r_idx], device=device)
                q_label = torch.as_tensor(label_codes[q_idx], device=device)
                r_label = torch.as_tensor(label_codes[r_idx], device=device)
                if task == "exact":
                    positive = r_label[None, :] == q_label[:, None]
                    n_candidate = None
                else:
                    candidate = r_gene[None, :] != q_gene[:, None]
                    scores = scores.masked_fill(~candidate, -torch.inf)
                    positive = (r_label[None, :] == q_label[:, None]) & candidate
                    n_candidate = candidate.sum(dim=1)
                metrics = torch_metrics(scores, positive, n_candidate)
                positive_counts = positive.sum(dim=1).detach().cpu().numpy().astype(int)
                candidate_counts = (n_candidate.detach().cpu().numpy().astype(int) if n_candidate is not None else np.full(q_idx.size, r_idx.size, dtype=int))
        else:
            scores = X[q_idx] @ X[r_idx].T
            if task == "exact":
                positive = labels[r_idx][None, :] == labels[q_idx][:, None]
                candidate_counts = np.full(q_idx.size, r_idx.size, dtype=int)
            else:
                candidate = genes[r_idx][None, :] != genes[q_idx][:, None]
                scores = np.where(candidate, scores, -np.inf)
                positive = (labels[r_idx][None, :] == labels[q_idx][:, None]) & candidate
                candidate_counts = candidate.sum(axis=1).astype(int)
            order = np.argsort(-scores, axis=1, kind="stable")
            ordered = np.take_along_axis(positive, order, axis=1)
            if task == "exact":
                metrics = metrics_from_relevance(ordered, TOPK)
            else:
                valid = np.arange(r_idx.size)[None, :] < candidate_counts[:, None]
                metrics = metrics_from_relevance(ordered & valid, TOPK)
            positive_counts = positive.sum(axis=1).astype(int)
        validate_metric_arrays(metrics)
        if np.any(metrics["recall_at_1"] > metrics["recall_at_5"] + 1e-12) or np.any(metrics["recall_at_5"] > metrics["recall_at_10"] + 1e-12):
            raise RuntimeError(f"Recall monotonicity failed for {task}, batch={batch}")
        for pos, qi in enumerate(q_idx):
            if positive_counts[pos] <= 0:
                raise RuntimeError(f"no positive reference for {item_ids[qi]}")
            row = {
                "task": task,
                "query_item_id": item_ids[qi],
                "query_target_gene": genes[qi],
                "query_batch_group": groups[qi],
                "n_cells": int(meta.iloc[qi]["n_cells"]),
                "n_reference_items": int(candidate_counts[pos]),
                "n_positive_reference_items": int(positive_counts[pos]),
            }
            row.update({metric: float(metrics[metric][pos]) for metric in METRIC_NAMES})
            rows.append(row)
        if use_torch:
            del scores, positive
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)

def attach_functional_labels(meta: pd.DataFrame, manifest: pd.DataFrame, assignment_column: str) -> pd.DataFrame:
    required = ["target_gene", "primary_functional_group", "primary_clean_eligible", "primary_group_size_genes", assignment_column]
    out = meta.merge(manifest[required], on="target_gene", how="left", validate="many_to_one")
    if out["primary_functional_group"].isna().any():
        raise RuntimeError("functional annotation missing for delta item")
    out["label"] = out["primary_functional_group"].astype(str)
    return out

def functional_query_mask(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = meta["label"].astype(str).to_numpy()
    genes = meta["target_gene"].astype(str).to_numpy()
    batches = meta["group"].astype(str).to_numpy()
    primary = meta["primary_clean_eligible"].astype(int).to_numpy() == 1
    by_label_batch: dict[tuple[str, str], set[str]] = {}
    for label, batch, gene in zip(labels, batches, genes):
        by_label_batch.setdefault((label, batch), set()).add(gene)
    labels_to_batches: dict[str, list[str]] = {}
    for label, batch in by_label_batch:
        labels_to_batches.setdefault(label, []).append(batch)
    other_genes: dict[tuple[str, str], set[str]] = {}
    for label, batch_list in labels_to_batches.items():
        for batch in batch_list:
            merged: set[str] = set()
            for other_batch in batch_list:
                if other_batch != batch:
                    merged.update(by_label_batch[(label, other_batch)])
            other_genes[(label, batch)] = merged
    positive = np.fromiter(
        (bool(other_genes.get((label, batch), set()) - {gene})
         for label, batch, gene in zip(labels, batches, genes)),
        dtype=bool,
        count=len(meta),
    )
    return primary & positive, primary

def run_functional_variant(
    X: np.ndarray,
    meta: pd.DataFrame,
    manifest: pd.DataFrame,
    assignment_column: str,
    annotation_source: str,
    model_id: str,
    normalization: str,
    min_cells: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    functional_meta = attach_functional_labels(meta, manifest, assignment_column)
    query_mask, primary = functional_query_mask(functional_meta)
    functional = evaluate_retrieval(X, functional_meta, "functional", device, query_mask=query_mask)
    group_lookup = manifest.set_index("target_gene")["primary_functional_group"]
    functional["functional_group"] = functional["query_target_gene"].map(group_lookup)
    functional["annotation_source"] = annotation_source
    functional["model_id"] = model_id
    functional["normalization"] = normalization
    functional["min_cells"] = min_cells
    functional["variant"] = "delta"
    functional["setting"] = "primary_clean"
    excluded = functional_meta[primary & ~query_mask].copy()
    excluded["annotation_source"] = annotation_source
    excluded["task"] = "functional"
    excluded["exclusion_reason"] = "primary-clean item has no same-group different-target item in another retained batch"
    return functional, excluded[["item_id", "target_gene", "group", "n_cells", "annotation_source", "task", "exclusion_reason"]]

def run(embedding_path,metadata_path,annotation_path,model,output,device):
    meta=pd.read_csv(metadata_path,sep='	',dtype=str)
    x,ids,valid=embeddings(embedding_path,meta.cell_id)
    if not valid.all():raise ValueError('The paper retrieval roster requires all cell embeddings')
    if not set(meta.cell_role).issubset({'control','perturbed'}):raise ValueError('Unknown cell roles')
    vectors,items=make_delta_items(np.asarray(x,dtype=np.float32),meta,'l2cell',0)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.save(output/'delta_vectors.npy',vectors);write_table(items,output/'delta_metadata.tsv')
    counts=items.groupby('target_gene').group.nunique();mask=items.target_gene.map(counts).ge(2).to_numpy()
    exact=evaluate_retrieval(vectors,items,'exact',device,query_mask=mask);exact['model_id']=model
    annotations=pd.read_csv(annotation_path,sep='	').fillna('')
    functional,_=run_functional_variant(vectors,items,annotations,'expert_assignment_status',
                                        'Replogle_2022_Table_S3_manual_annotation',model,'l2cell',0,device)
    for name,frame in [('exact_perturbation_state_retrieval',exact),('functional_perturbation_retrieval',functional)]:
        write_table(frame,output/name/'query_results.tsv.gz')
        summary=frame.groupby(['model_id','query_target_gene'],as_index=False)['map'].mean()
        write_table(summary,output/name/'target_scores.tsv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('embedding','metadata','annotations','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();run(a.embedding,a.metadata,a.annotations,a.model,a.output_dir,a.device)
if __name__=='__main__':main()
