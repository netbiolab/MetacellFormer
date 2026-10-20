"""Corpus counts, fixed-vocabulary sparsity and full-corpus PCA/UMAP calculations."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from benchmarks.shared._common import embeddings,write_table

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('umap');q.add_argument('--embedding',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('counts');q.add_argument('--manifest',type=Path,required=True);q.add_argument('--output-dir',type=Path,required=True)
    q.add_argument('--split',choices=['train','validation','all'],default='train')
    q=sub.add_parser('cost');q.add_argument('--stages',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('sparsity');q.add_argument('--counts',type=Path,required=True);q.add_argument('--vocabulary',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--gene-column',default='feature_id')
    a=p.parse_args()
    if a.command=='umap':
        from sklearn.decomposition import PCA
        import umap
        x,ids,valid=embeddings(a.embedding)
        if len(x)!=799562 or not valid.all():raise ValueError('Expected all 799,562 train metacells')
        z=PCA(n_components=50,svd_solver='randomized',random_state=42).fit_transform(x).astype(np.float32)
        xy=umap.UMAP(n_neighbors=30,min_dist=.3,metric='cosine',random_state=42,n_jobs=1,low_memory=True).fit_transform(z).astype(np.float32)
        write_table(pd.DataFrame({'cell_id':ids,'UMAP1':xy[:,0],'UMAP2':xy[:,1]}),a.output)
    elif a.command=='counts':
        d=pd.read_csv(a.manifest,sep='\t')
        if d.dataset_id.duplicated().any():raise ValueError('One row per dataset required')
        if 'split' in d:
            if a.split!='all':d=d[d.split.eq(a.split)].copy()
        elif a.split!='train':
            raise ValueError('This legacy manifest has no split column')
        # Legacy training-only manifests remain accepted.
        aliases={'tissue_group':'tissue','n_sc':'single_cells','n_mc':'metacells'}
        d=d.rename(columns={k:v for k,v in aliases.items() if k in d and v not in d})
        if d.empty:raise ValueError('No datasets in the requested split')
        for c in ['single_cells','metacells']:
            if (d[c]<0).any():raise ValueError('Negative counts')
        summary=d.groupby('tissue',dropna=False)[['single_cells','metacells']].sum().reset_index()
        summary['single_cell_fraction']=summary.single_cells/summary.single_cells.sum()
        write_table(summary,a.output_dir/'tissue_counts.tsv')
        write_table(pd.DataFrame([dict(single_cells=int(d.single_cells.sum()),metacells=int(d.metacells.sum()),datasets=len(d),compression=float(d.single_cells.sum()/d.metacells.sum()))]),a.output_dir/'corpus_counts.tsv')
    elif a.command=='cost':
        d=pd.read_csv(a.stages,sep='\t');rows=[]
        for arm,part in d.groupby('Arm',sort=False):
            training=pd.to_numeric(part.Epoch,errors='coerce').between(1,3)
            preprocessing=part.Epoch.isna()
            pre=float(part.loc[preprocessing,'Wall-clock time (h)'].sum());fit=float(part.loc[training,'Wall-clock time (h)'].sum())
            rows.append({'Arm':arm,'Preprocessing time (h)':pre,'Epochs 1–3 active training time (h)':fit,'Accounted total time (h)':pre+fit,'Training GPU count':int(part.loc[training,'GPU count'].iloc[0])})
        write_table(pd.DataFrame(rows),a.output)
    else:
        import anndata as ad,json
        d=ad.read_h5ad(a.counts,backed='r');v=json.loads(a.vocabulary.read_text())['genes']
        names=d.var[a.gene_column].astype(str).to_numpy() if a.gene_column in d.var else d.var_names.astype(str)
        if len(set(names))!=len(names):raise ValueError('Duplicate gene IDs')
        columns=np.flatnonzero(np.isin(names,v));nonzero=0
        for start in range(0,len(d),1024):
            x=sparse.csr_matrix(d.X[start:start+1024])[:,columns];x.eliminate_zeros();nonzero+=x.nnz
        value=1-nonzero/(len(d)*len(v))
        write_table(pd.DataFrame([dict(profiles=len(d),vocabulary_genes=len(v),zero_fraction_mean=value)]),a.output);d.file.close()
if __name__=='__main__':main()
