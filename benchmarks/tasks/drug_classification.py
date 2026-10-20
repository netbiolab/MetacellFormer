"""Paper category-wise 10-fold scDrugMap MLP, with the original RNG sequence."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from benchmarks.shared._drug_head import OfficialEBDMLP, set_seed
from benchmarks.shared._common import MODELS, embeddings, table, write_table
AXES={'Tissue':'tissue','Drug_type':'drug','Regimen':'regimen','Cancer_type_level1':'cancer'}

def metric_dict(y: np.ndarray, logits: np.ndarray, prefix: str) -> dict[str, object]:
    pred = np.argmax(logits, axis=1).astype(np.int64)
    score = logits[:, 1]
    cm = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    tn, fp, fn, tp = [int(x) for x in cm]
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_f1": float(f1_score(y, pred, zero_division=1)),
        f"{prefix}_auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else 0.0,
        f"{prefix}_tn": tn,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_tp": tp,
        f"{prefix}_n": int(len(y)),
        f"{prefix}_negative": int(np.sum(y == 0)),
        f"{prefix}_positive": int(np.sum(y == 1)),
        f"{prefix}_prevalence": float(np.mean(y == 1)),
        f"{prefix}_predicted_positive_rate": float(np.mean(pred == 1)),
    }

def train_fold(
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    set_seed(args.training_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = OfficialEBDMLP(int(x_all.shape[1]), args.hidden_size).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, object]] = []
    train_y = y_all[train_rows]
    test_y = y_all[test_rows]
    x_train_device = None
    x_test_device = None
    if args.stage_on_gpu:
        x_train_device = torch.from_numpy(x_all[train_rows]).to(device)
        x_test_device = torch.from_numpy(x_all[test_rows]).to(device)
        train_labels_device = torch.from_numpy(train_y).to(device)
        test_labels_device = torch.from_numpy(test_y).to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows)).numpy()
        train_loss_total = 0.0
        train_batches = 0
        for start in range(0, len(order), args.train_batch_size):
            local = order[start : start + args.train_batch_size]
            if x_train_device is None:
                xb = torch.from_numpy(x_all[train_rows[local]]).to(device)
                yb = torch.nn.functional.one_hot(torch.from_numpy(train_y[local]), num_classes=2).float().to(device)
            else:
                idx = torch.from_numpy(local).to(device)
                xb = x_train_device.index_select(0, idx)
                yb = torch.nn.functional.one_hot(train_labels_device.index_select(0, idx), num_classes=2).float()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach().cpu())
            train_batches += 1

        model.eval()
        test_order = torch.randperm(len(test_rows)).numpy()
        train_logits_parts: list[np.ndarray] = []
        test_logits_parts: list[np.ndarray] = []
        test_loss_total = 0.0
        test_batches = 0
        with torch.no_grad():
            # Training metrics are computed in deterministic row order after
            # optimization, while test metrics follow the source DataLoader's
            # shuffled evaluation semantics.
            for start in range(0, len(train_rows), args.test_batch_size):
                stop = start + args.test_batch_size
                if x_train_device is None:
                    xb = torch.from_numpy(x_all[train_rows[start:stop]]).to(device)
                    labels = torch.from_numpy(train_y[start:stop]).to(device)
                else:
                    xb = x_train_device[start:stop]
                    labels = train_labels_device[start:stop]
                train_logits_parts.append(model(xb).cpu().numpy())
            for start in range(0, len(test_order), args.test_batch_size):
                local = test_order[start : start + args.test_batch_size]
                if x_test_device is None:
                    xb = torch.from_numpy(x_all[test_rows[local]]).to(device)
                    labels = torch.from_numpy(test_y[local]).to(device)
                else:
                    idx = torch.from_numpy(local).to(device)
                    xb = x_test_device.index_select(0, idx)
                    labels = test_labels_device.index_select(0, idx)
                logits = model(xb)
                one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
                test_loss_total += float(loss_fn(logits, one_hot).cpu())
                test_batches += 1
                test_logits_parts.append(logits.cpu().numpy())
        train_logits = np.concatenate(train_logits_parts, axis=0)
        test_logits_shuffled = np.concatenate(test_logits_parts, axis=0)
        test_y_shuffled = test_y[test_order]
        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(train_batches, 1),
            "test_loss": test_loss_total / max(test_batches, 1),
            "device": str(device),
        }
        row.update(metric_dict(train_y, train_logits, "train"))
        row.update(metric_dict(test_y_shuffled, test_logits_shuffled, "test"))
        history.append(row)
        # Optional audit hook; does not alter optimization, RNG, or metrics.
        callback = getattr(args, "prediction_callback", None)
        if callback is not None and epoch == args.epochs:
            callback(test_rows[test_order], test_y_shuffled, test_logits_shuffled)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history

def run(embedding,metadata,folds_path,model,output,device='cpu',axis=None,category=None,fold=None):
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    meta=table(metadata)
    x,ids,valid=embeddings(embedding,meta.cell_id)
    if not valid.all():raise ValueError('Incomplete native embeddings')
    y=meta.label.to_numpy(dtype=np.int64)
    with np.load(folds_path,allow_pickle=False) as z:folds={k:z[k] for k in z.files}
    if any(len(v)!=len(meta) for v in folds.values()):raise ValueError('Fold roster length differs')
    torch.set_num_threads(2)
    args=SimpleNamespace(training_seed=24,device=device,hidden_size=512,lr=1e-4,epochs=10,
                         train_batch_size=128,test_batch_size=256,stage_on_gpu=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);records=[]
    for name,field in AXES.items():
        if axis is not None and axis!=name:continue
        assignment=folds['fold_'+field]
        values=sorted(set(meta.loc[assignment>=0,field].astype(str)))
        if category is not None:values=[v for v in values if v==category]
        for value in values:
            rows=np.flatnonzero(meta[field].astype(str).eq(value)&(assignment>=0))
            for f in (range(10) if fold is None else [fold]):
                tr=rows[assignment[rows]!=f];te=rows[assignment[rows]==f]
                if set(y[tr])!={0,1} or set(y[te])!={0,1}:raise ValueError('One-class fold')
                dest=output/name/value/f'fold{f}';dest.mkdir(parents=True,exist_ok=True)
                def capture(rows,truth,logits):
                    np.savez_compressed(dest/'predictions.npz',cell_ids=ids[rows],label=truth,logits=logits)
                args.prediction_callback=capture
                history=train_fold(np.asarray(x,dtype=np.float32),y,tr,te,args)
                write_table(pd.DataFrame(history),dest/'epochs.tsv')
                records.append(dict(model_id=model,axis=name,category_value=value,fold=f,**history[-1]))
    if not records:raise ValueError('No eligible folds selected')
    write_table(pd.DataFrame(records),output/'pooled_fold_scores.tsv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('embedding','metadata','folds','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--axis',choices=AXES);p.add_argument('--category');p.add_argument('--fold',type=int,choices=range(10))
    a=p.parse_args();run(a.embedding,a.metadata,a.folds,a.model,a.output_dir,a.device,a.axis,a.category,a.fold)
if __name__=='__main__':main()
