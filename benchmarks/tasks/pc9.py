"""Train the paper PC9 MLP on source-study cells and score external comparisons.

Metadata TSVs have cell_id, study, label (0=control, 1=surviving/DTEP).
External comparison TSV has comparison_id, cell_id, label; shared controls may
occur in both gefitinib and osimertinib comparisons. Each embedding is encoded
once. No external labels are used by the training loop.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from benchmarks.shared._common import MODELS, embeddings, table, write_table

COMPARISONS=('GSE303411_PC9_persister','GSE196018','GSE282074_PC9_gefitinib','GSE282074_PC9_osimertinib')

def evaluate_comparisons(logits,ids,comparisons):
    index=pd.Index(np.asarray(ids).astype(str))
    if not index.is_unique:raise ValueError('External cell IDs must be unique')
    rows=[]
    for name,part in comparisons.groupby('comparison_id',sort=False):
        if part.cell_id.duplicated().any():raise ValueError('Duplicate cell within a comparison')
        pos=index.get_indexer(part.cell_id.astype(str))
        if (pos<0).any():raise ValueError('A comparison cell has no embedding')
        y=part.label.to_numpy(dtype=np.int64)
        if set(y)!={0,1}:raise ValueError('Each comparison needs two classes')
        rows.append(dict(comparison_id=name,AP=float(average_precision_score(y,logits[pos,1])),
                         n_test=len(y),n_positive=int(y.sum())))
    return pd.DataFrame(rows)

def run(train_embedding,test_embedding,train_metadata,test_metadata,comparison_path,model,output,device='cpu'):
    import torch
    from benchmarks.shared._drug_head import train_mlp
    tr=table(train_metadata);te=table(test_metadata);comparisons=table(comparison_path)
    for frame in (tr,te):
        if not {'cell_id','study','label'}.issubset(frame):raise ValueError('Metadata requires cell_id, study, label')
        if frame.cell_id.duplicated().any():raise ValueError('Duplicate metadata cell IDs')
    if set(tr.study)&set(te.study) or set(tr.cell_id)&set(te.cell_id):raise ValueError('Training/external overlap')
    if set(tr.study)!={'GSE149214'}:raise ValueError('PC9 source training must be GSE149214 only')
    if set(comparisons.comparison_id)!=set(COMPARISONS):raise ValueError('Expected the four paper comparisons')
    if len(tr)!=2393:raise ValueError('Expected 2,393 source PC9 cells')
    if set(tr.label)!={0,1}:raise ValueError('Training must contain parental and DTEP cells')
    expected=te.set_index('cell_id').label
    if not np.array_equal(comparisons.cell_id.map(expected),comparisons.label):raise ValueError('External labels differ')
    xt,_,vt=embeddings(train_embedding,tr.cell_id);xe,ids,ve=embeddings(test_embedding,te.cell_id)
    if not vt.all() or not ve.all() or xt.shape[1]!=xe.shape[1]:raise ValueError('Incomplete/mismatched embeddings')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args=SimpleNamespace(hidden_size=512,epochs=10,lr=1e-4,batch_size_train=128,batch_size_test=256,device=device)
    rows=[]
    for seed in (1,2,3):
        # Dummy external labels: inference is independent of external outcomes.
        result=train_mlp(np.asarray(xt,dtype=np.float32),tr.label.to_numpy(dtype=np.int64),
                         np.asarray(xe,dtype=np.float32),np.zeros(len(xe),np.int64),args,seed)
        logits=result['logits']
        np.savez_compressed(output/f'predictions_s{seed}.npz',cell_ids=ids,logits=logits)
        part=evaluate_comparisons(logits,ids,comparisons);part['model']=model;part['seed']=seed;rows.append(part)
    scores=pd.concat(rows,ignore_index=True);write_table(scores,output/'seed_metrics.tsv')
    write_table(scores.groupby(['comparison_id','model'],as_index=False).AP.mean(),output/'summary.tsv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train-embedding','test-embedding','train-metadata','test-metadata','comparisons','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=MODELS[:-1],required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();run(a.train_embedding,a.test_embedding,a.train_metadata,a.test_metadata,a.comparisons,a.model,a.output_dir,a.device)
if __name__=='__main__':main()
