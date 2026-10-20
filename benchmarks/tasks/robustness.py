"""Gene-specific Gaussian inputs and frozen clean-probe evaluation (Figs. 3, 5).

prepare uses the published roster and unmodified archived benchmark matrix. score consumes
native embeddings, produced with extract_embeddings.py in each model environment.
Integer-valued entries do not establish raw-count provenance; see the dataset
input_manifest.tsv, particularly its DC expression-type annotation.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from benchmarks.shared._common import DATASETS, MODELS, embeddings, roster, write_table, write_json

RATIOS=(.1,.5,.8,1.)
DRAW_SEEDS=(42,43,44,45,46)
PROBE_SEEDS=(1,2,3)

def gaussian(raw,variance,ratio,rng):
    """Normal variance = ratio * population variance; SD is its square root.

    Preserve 128-row generation, float64 noise and final float32 conversion used
    in the original experiment. Zeros receive noise; no rounding/rescaling.
    """
    if ratio<0:raise ValueError('Noise ratio must be nonnegative')
    raw=sparse.csr_matrix(raw)
    sd=np.sqrt(ratio*np.maximum(variance,0))
    chunks=[]
    for start in range(0,raw.shape[0],128):
        block=raw[start:start+128].toarray().astype(np.float64)
        block+=rng.standard_normal(block.shape)*sd[None,:]
        np.maximum(block,0,out=block)
        chunks.append(sparse.csr_matrix(block.astype(np.float32)))
    out=sparse.vstack(chunks,format='csr');out.eliminate_zeros()
    return out

def prepare(input_path,roster_path,dataset,output,layer=None,ratios=RATIOS,seeds=DRAW_SEEDS):
    import anndata as ad
    frame=roster(roster_path)
    if frame.cell_id.duplicated().any() or not set(frame.split).issubset({'0','2'}):
        raise ValueError('Expected unique cells and fixed train=0/test=2 splits')
    a=ad.read_h5ad(input_path)
    positions=a.obs_names.get_indexer(frame.cell_id.astype(str))
    if (positions<0).any():raise ValueError('Roster cells missing from the raw input')
    a=a[positions].copy()
    raw=sparse.csr_matrix(a.layers[layer] if layer else a.X)
    if not np.isfinite(raw.data).all() or (raw.data<0).any() or not np.allclose(raw.data,np.round(raw.data),rtol=0,atol=1e-6):
        raise ValueError('Gaussian preparation requires the archived nonnegative integer-valued input')
    a.X=raw;a.layers.clear()
    a.obs=pd.DataFrame({'label':frame.label.to_numpy(),'split':frame.split.to_numpy()},index=frame.cell_id.astype(str))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    train=a[frame.split.eq('0').to_numpy()].copy();test=a[frame.split.eq('2').to_numpy()].copy()
    train.write_h5ad(output/'train_clean.h5ad');test.write_h5ad(output/'clean.h5ad')
    x=sparse.csr_matrix(test.X,dtype=np.float64)
    variance=np.asarray(x.power(2).mean(axis=0)).ravel()-np.asarray(x.mean(axis=0)).ravel()**2
    variance=np.maximum(variance,0)
    np.save(output/'gene_variance.npy',variance)
    for seed in seeds:
        for ratio in ratios:
            rng=np.random.default_rng(np.random.SeedSequence([seed,DATASETS.index(dataset),11,int(round(ratio*10))]))
            noisy=test.copy();noisy.X=gaussian(test.X,variance,ratio,rng)
            noisy.write_h5ad(output/f'gaussian_{ratio:g}_s{seed}.h5ad')
    write_json({'dataset':dataset,'ratios':list(ratios),'draw_seeds':list(seeds),
                'variance_ddof':0,'generation_rows':128,'n_train':len(train),'n_test':len(test)},output/'preparation.json')

def fit_clean(train_x,train_y,clean_x,truth,seeds=PROBE_SEEDS):
    classes=sorted(set(truth))
    with threadpool_limits(limits=2):
        scaler=StandardScaler().fit(train_x)
        ztrain=scaler.transform(train_x);zclean=scaler.transform(clean_x)
        probes={s:LogisticRegression(C=1,class_weight='balanced',max_iter=3000,
                   random_state=s,n_jobs=1).fit(ztrain,train_y) for s in seeds}
        clean={s:float(f1_score(truth,p.predict(zclean),labels=classes,average='macro',zero_division=0)) for s,p in probes.items()}
    if any(v<=0 for v in clean.values()):raise ValueError('Retention is undefined for a zero clean macro-F1')
    return scaler,probes,clean

def evaluate_noisy(x,valid,truth,scaler,probes,clean):
    rows=[];predictions={}
    with threadpool_limits(limits=2):
        z=scaler.transform(x)
        for seed,probe in probes.items():
            pred=probe.predict(z).astype(object);pred[~valid]='__NO_VALID_EMBEDDING__'
            value=float(f1_score(truth,pred,labels=sorted(set(truth)),average='macro',zero_division=0))
            rows.append(dict(probe_seed=seed,macro_f1=value,clean_macro_f1=clean[seed],
                retention=value/clean[seed],f1_loss=clean[seed]-value,n_test=len(truth),
                coverage=float(valid.mean()),n_failed=int((~valid).sum())))
            predictions[f'pred_s{seed}']=pred.astype(str)
    return rows,predictions

def score(roster_path,embedding_dir,model,dataset,output,ratios=RATIOS,seeds=DRAW_SEEDS):
    frame=roster(roster_path);train=frame[frame.split.eq('0')];test=frame[frame.split.eq('2')]
    embedding_dir=Path(embedding_dir);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def read(name,ids):
        paths=[embedding_dir/(name+'.npz'),embedding_dir/(name+'.npy')]
        path=next((p for p in paths if p.is_file()),paths[0])
        return embeddings(path,ids)
    xt,_,vt=read('train_clean',train.cell_id);xc,ids,vc=read('clean',test.cell_id)
    if not vt.all() or not vc.all():raise ValueError('The published clean baseline requires complete native embeddings')
    truth=test.label.to_numpy(dtype=str)
    scaler,probes,clean=fit_clean(xt,train.label.to_numpy(dtype=str),xc,truth)
    write_table(pd.DataFrame([dict(model=model,dataset=dataset,probe_seed=s,macro_f1=v,
                     n_test=len(test),retention=1.,coverage=1.,f1_loss=0.) for s,v in clean.items()]),output/'clean_scores.tsv')
    rows=[]
    for seed in seeds:
        for ratio in ratios:
            condition=f'gaussian_{ratio:g}'
            x,_,valid=read(f'{condition}_s{seed}',test.cell_id)
            part,pred=evaluate_noisy(x,valid,truth,scaler,probes,clean)
            rows.extend(dict(model=model,dataset=dataset,condition=condition,level=ratio,draw_seed=seed,**r) for r in part)
            np.savez_compressed(output/f'{condition}_s{seed}_predictions.npz',cell_ids=ids,truth=truth,valid=valid,**pred)
    write_table(pd.DataFrame(rows),output/'all_scores.tsv')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    for stage in ('prepare','score'):
        q=sub.add_parser(stage);q.add_argument('--roster',type=Path,required=True)
        q.add_argument('--dataset',choices=DATASETS,required=True);q.add_argument('--output-dir',type=Path,required=True)
        q.add_argument('--ratios',type=float,nargs='+',default=RATIOS,choices=RATIOS)
        q.add_argument('--seeds',type=int,nargs='+',default=DRAW_SEEDS,choices=DRAW_SEEDS)
        if stage=='prepare':q.add_argument('--input',type=Path,required=True);q.add_argument('--layer')
        else:q.add_argument('--embedding-dir',type=Path,required=True);q.add_argument('--model',choices=MODELS,required=True)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.input,a.roster,a.dataset,a.output_dir,a.layer,a.ratios,a.seeds)
    else:score(a.roster,a.embedding_dir,a.model,a.dataset,a.output_dir,a.ratios,a.seeds)
if __name__=='__main__':main()
