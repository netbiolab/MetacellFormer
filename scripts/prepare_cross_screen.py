"""Select the fixed union cells and measured common genes before any normalization."""
from pathlib import Path
import argparse,json
import numpy as np
import anndata as ad
from scipy import sparse

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['input','contracts','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--screen',choices=['K562','RPE1'],required=True);p.add_argument('--seed42',action='store_true');a=p.parse_args()
    data=ad.read_h5ad(a.input);suffix='_seed42' if a.seed42 else '';ids=np.load(a.contracts/f'{a.screen}{suffix}_cell_ids.npy').astype(str)
    rows=data.obs_names.get_indexer(ids)
    if (rows<0).any():raise ValueError('Missing fixed cells')
    allowed=np.isin(data.var_names.astype(str),json.loads((a.contracts/'common_measured_genes.json').read_text()))
    # Subsetting is equivalent to zeroing other measured genes; the total must be recomputed.
    data=data[rows,allowed].copy();data.X=sparse.csr_matrix(data.X);data.layers.clear()
    if 'total_counts_all_features' in data.obs:del data.obs['total_counts_all_features']
    a.output.parent.mkdir(parents=True,exist_ok=True);data.write_h5ad(a.output,compression='lzf')
if __name__=='__main__':main()
