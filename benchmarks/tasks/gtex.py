"""Reproduce the frozen U_bio GTEx GPU probe from packaged static embeddings.

This implements the 2026-09-27 float32/PCA-lowrank/L-BFGS protocol. Its output
contains all 150 tissue/fold scores and held-out predictions; no source tree is
needed at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score


def stable_seed(model_id: str, tissue_index: int, fold: int) -> int:
    return int.from_bytes(hashlib.sha256(f"42|{model_id}|{tissue_index}|{fold}".encode()).digest()[:4], "little")


def run(root: Path, model_id: str, embedding_path: Path, output_root: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen U_bio GTEx GPU protocol")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    contract = root / "contracts" / "tissue_specificity_prediction"
    with np.load(contract / "labels_folds.npz", allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    support = json.loads((contract / "shared_genes.json").read_text())
    shared = np.asarray(support["shared_genes"], dtype=str)
    with np.load(embedding_path, allow_pickle=False) as z:
        genes = z["genes"].astype(str)
        embedding = z["embedding"].astype(np.float32)
    if len(set(genes)) != len(genes) or embedding.shape[0] != len(genes):
        raise ValueError("Invalid static gene embedding")
    index = {gene: i for i, gene in enumerate(genes)}
    if not set(shared).issubset(index):
        raise ValueError("Model is missing genes in the frozen U_bio shared support")
    x_np = embedding[[index[gene] for gene in shared]]
    if not np.isfinite(x_np).all() or not np.all(np.linalg.norm(x_np, axis=1) > 0):
        raise ValueError("Nonfinite or zero static embedding in shared support")
    ubio_index = {gene: i for i, gene in enumerate(arrays["genes"].astype(str))}
    selected = np.asarray([ubio_index[gene] for gene in shared], dtype=np.int64)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    rows: list[dict] = []
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("tissue", "gene_index", "fold", "label", "score")}
    for tissue_index, tissue in enumerate(arrays["tissues"].astype(str)):
        labels = arrays[f"y_{tissue_index}"][selected].astype(np.int8)
        folds = arrays[f"fold_{tissue_index}"][selected].astype(np.int8)
        if set(np.unique(folds)) != set(range(5)):
            raise ValueError(f"Incomplete folds: {tissue}")
        for fold in range(5):
            test = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            y_train, y_test = labels[train], labels[test]
            if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
                raise ValueError(f"One-class fold: {tissue}/{fold}")
            seed_ids = json.loads((Path(__file__).resolve().parents[2] / "configs/gtex_seed_identifiers.json").read_text())
            torch.manual_seed(stable_seed(seed_ids[model_id], tissue_index, fold))
            train_raw = x.index_select(0, torch.as_tensor(train, dtype=torch.long, device=device))
            test_raw = x.index_select(0, torch.as_tensor(test, dtype=torch.long, device=device))
            mean = train_raw.mean(dim=0)
            scale = torch.sqrt(((train_raw - mean) ** 2).mean(dim=0))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            train_x, test_x = (train_raw - mean) / scale, (test_raw - mean) / scale
            components = min(256, int(train_x.shape[0]) - 1, int(train_x.shape[1]))
            q = min(components + 16, int(train_x.shape[0]), int(train_x.shape[1]))
            _, _, vectors = torch.pca_lowrank(train_x, q=q, center=False, niter=4)
            projection = vectors[:, :components]
            train_z, test_z = train_x @ projection, test_x @ projection
            y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
            positives = float(y.sum().item())
            negatives = float(len(y_train) - positives)
            sample_weight = torch.where(y > 0, torch.full_like(y, len(y_train) / (2 * positives)),
                                        torch.full_like(y, len(y_train) / (2 * negatives)))
            weight = torch.zeros(components, dtype=torch.float32, device=device, requires_grad=True)
            bias = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
            optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=300, max_eval=375,
                                          tolerance_grad=1e-8, tolerance_change=1e-12,
                                          history_size=20, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = (sample_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                    train_z @ weight + bias, y, reduction="none")).sum()
                loss = loss + 0.5 * weight.square().sum()
                loss.backward()
                return loss

            final_loss = optimizer.step(closure)
            n_iter = int(optimizer.state[weight].get("n_iter", 0))
            with torch.no_grad():
                score = (test_z @ weight + bias).detach().cpu().numpy()
            if not np.isfinite(score).all() or not np.isfinite(float(final_loss.detach().cpu())) or n_iter >= 300:
                raise RuntimeError(f"L-BFGS did not converge: {tissue}/{fold}: {n_iter}")
            order = np.argsort(score)[::-1][:min(50, len(test))]
            rows.append({"model_id": model_id, "unit": str(tissue), "fold": fold,
                         "metric_primary": float(average_precision_score(y_test, score)),
                         "auroc": float(roc_auc_score(y_test, score)),
                         "recall_at_50": float(y_test[order].sum() / max(y_test.sum(), 1)),
                         "ndcg_at_50": float(ndcg_score(y_test[None, :], score[None, :], k=min(50, len(test)))),
                         "n_candidates": len(labels), "n_positive": int(labels.sum()),
                         "protocol_id": "gtex_ubio_fixed_contract_cuda_torch_pcalowrank_lbfgs_20260927",
                         "status": "PASS", "optimizer_iterations": n_iter})
            for key, value in (("tissue", np.full(len(test), tissue_index, np.int16)),
                               ("gene_index", selected[test].astype(np.int32)),
                               ("fold", np.full(len(test), fold, np.int8)),
                               ("label", y_test.astype(np.int8)), ("score", score.astype(np.float32))):
                parts[key].append(value)
            print(f"GTEX {model_id} {tissue_index + 1}/30 fold {fold + 1}/5", flush=True)
            del train_raw, test_raw, train_x, test_x, projection, train_z, test_z, weight, bias, optimizer
            torch.cuda.empty_cache()
    out = output_root / model_id
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.tsv", sep="\t", index=False)
    np.savez_compressed(out / "predictions.npz", **{key: np.concatenate(value) for key, value in parts.items()})


def main() -> None:
    from benchmarks.shared._common import MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root, args.model, args.embedding, args.output_dir)

if __name__ == "__main__": main()
