"""Create original rank/value-bin token caches from raw profiles and a fixed split.
The split TSV has profile_id,split (train/eval); no new split is drawn here.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy import sparse
from metacellformer import GeneRankTokenizer
from metacellformer.binning import ValueBinner
from metacellformer.pretraining_tokenizer import tokenize_single_cell_from_indices

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input','model-dir','split','output-dir']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--dataset-id',required=True);p.add_argument('--dataset-idx',type=int,required=True)
    p.add_argument('--gene-column',default='feature_id');a=p.parse_args()
    import anndata as ad
    data=ad.read_h5ad(a.input,backed='r');split=pd.read_csv(a.split,sep='\t',dtype=str)
    if not split.profile_id.is_unique or not np.array_equal(split.profile_id.to_numpy(),data.obs_names.astype(str)):raise ValueError('Split must match raw profile IDs and order')
    if not set(split.split).issubset({'train','eval'}):raise ValueError('Unknown split')
    tok=GeneRankTokenizer(a.model_dir);genes=data.var[a.gene_column].astype(str).to_numpy() if a.gene_column in data.var else data.var_names.astype(str).to_numpy()
    if len(set(genes))!=len(genes):raise ValueError('Collapse duplicate gene IDs first')
    token=np.array([tok.gene_to_token.get(g,0) for g in genes]);keep=token>=4
    median=np.ones(len(genes),np.float32);median[keep]=tok.medians[token[keep]-4]
    median=np.where(np.isfinite(median)&(median>0),median,1).astype(np.float32)
    binner=ValueBinner.load(a.model_dir/'training_resources/value_bin_edges.npy',64)
    a.output_dir.mkdir(parents=True,exist_ok=True);n=len(data);arrays={}
    for name,dtype in [('input_ids',np.int32),('value_bins',np.uint8),('attention_mask',np.uint8)]:
        arrays[name]=np.lib.format.open_memmap(a.output_dir/f'{name}.npy',mode='w+',dtype=dtype,shape=(n,1024))
    for start in range(0,n,128):
        block=sparse.csr_matrix(data.X[start:start+128]);block.sum_duplicates();block.eliminate_zeros()
        if not np.isfinite(block.data).all() or (block.data<0).any():raise ValueError('Invalid raw counts')
        for j in range(block.shape[0]):
            row=block[j]
            z=tokenize_single_cell_from_indices(row.indices,row.data,token,1023,binner,tok.special['<cls>'],tok.special['<pad>'],gene_subset_mask=keep,ordering_mode='geneformer_median_scaled_cp10k',rank_normalization_factors_by_index=median)
            if z.attention_mask.sum()<=1:raise ValueError(f'No gene tokens at profile {start+j}; fix the frozen roster')
            for name,array in arrays.items():array[start+j]=getattr(z,name)
    for v in arrays.values():v.flush()
    extra={'metacell':np.arange(n,dtype=np.int64),'split':split.split.eq('eval').to_numpy(dtype=np.int8),
           'train_rows_by_mc':np.flatnonzero(split.split.eq('train')),'val_rows_by_mc':np.flatnonzero(split.split.eq('eval'))}
    for name,v in extra.items():np.save(a.output_dir/f'{name}.npy',v)
    record={'dataset_id':a.dataset_id,'dataset_idx':a.dataset_idx}
    for name in [*arrays,*extra]:record[name+'_path']=str((a.output_dir/f'{name}.npy').resolve())
    if 'n_cells' in data.obs:
        np.save(a.output_dir/'raw_cell_count.npy',data.obs.n_cells.to_numpy(dtype=np.int32));record['raw_cell_count_path']=str((a.output_dir/'raw_cell_count.npy').resolve())
    pd.DataFrame([record]).to_csv(a.output_dir/'token_index.csv',index=False)
    data.file.close()
if __name__=='__main__':main()
