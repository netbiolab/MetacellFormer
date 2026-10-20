"""Balanced logistic probes on native frozen embeddings for the nine datasets."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from benchmarks.shared._common import MODELS, DATASETS, embeddings, roster, write_table

def run(train_path,test_path,roster_path,model,dataset,output):
    frame=roster(roster_path);tr=frame[frame.split.eq('0')];te=frame[frame.split.eq('2')]
    xt,_,vt=embeddings(train_path,tr.cell_id);xe,ids,ve=embeddings(test_path,te.cell_id)
    if not vt.all() or not ve.all():raise ValueError('Main benchmark requires complete embeddings')
    encoder=LabelEncoder();y=encoder.fit_transform(tr.label.astype(str))
    keep=np.isin(te.label,encoder.classes_)
    if not keep.any():raise ValueError('No test labels occur in training')
    truth=te.label.to_numpy(dtype=str)[keep]
    output=Path(output);output.mkdir(parents=True,exist_ok=True);scores=[]
    with threadpool_limits(limits=2):
        scaler=StandardScaler().fit(xt)
        xtrain=scaler.transform(xt).astype(np.float32);xtest=scaler.transform(xe[keep]).astype(np.float32)
        for seed in (1,2,3):
            probe=LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=seed,n_jobs=1)
            probe.fit(xtrain,y);prediction=encoder.inverse_transform(probe.predict(xtest))
            scores.append(dict(model_id=model,dataset=dataset,seed=seed,
                               macro_f1=float(f1_score(truth,prediction,average='macro',zero_division=0)),
                               n_test=int(keep.sum()),n_test_total=len(te)))
            np.savez_compressed(output/f'predictions_s{seed}.npz',cell_ids=ids[keep],truth=truth,prediction=prediction)
    scores=pd.DataFrame(scores);write_table(scores,output/'seed_scores.tsv')
    mean=scores.groupby(['model_id','dataset'],as_index=False).macro_f1.mean().rename(columns={'macro_f1':'metric_primary'})
    write_table(mean,output/'per_dataset_scores.tsv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('train-embedding','test-embedding','roster','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--dataset',choices=DATASETS,required=True)
    a=p.parse_args();run(a.train_embedding,a.test_embedding,a.roster,a.model,a.dataset,a.output_dir)
if __name__=='__main__':main()
