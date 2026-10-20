"""Input checks shared by the paper evaluation commands."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

MODELS = ('MetacellFormer', 'Geneformer-V2', 'Geneformer-V1', 'scPRINT', 'scGPT', 'CellFM', 'MetacellFormer-SC')
DATASETS = ('Cell_Lines', 'DC', 'HumanPBMC', 'Immune', 'Liver_cross', 'MCA', 'MHSP', 'PBMC_368K', 'Pancrm')

def table(path):
    return pd.read_csv(path, sep='\t', float_precision='round_trip')

def write_table(frame, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,sep='\t',index=False,float_format='%.17g')

def write_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n')

def roster(path):
    path=Path(path)
    if path.suffix=='.json':
        value=json.loads(path.read_text())
        return pd.DataFrame({'cell_id':value['obs_name'], 'label':value['cell_type'],
                             'split':np.asarray(value['official_split']).astype(str)})
    frame=table(path)
    for column in ('cell_id','label','split'):
        if column not in frame:raise ValueError(f'Missing roster column: {column}')
    frame['split']=frame['split'].astype(str)
    return frame

def embeddings(path, expected_ids=None):
    """Read a portable NPZ or an original NPY with an explicit ID sidecar."""
    path=Path(path)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:
            x=z['embeddings'];ids=z['cell_ids'].astype(str)
            valid=z['valid'].astype(bool) if 'valid' in z else np.ones(len(x),bool)
    else:
        x=np.load(path,mmap_mode='r',allow_pickle=False)
        candidates=[path.with_suffix('.ids.npy'),path.with_name('cell_ids.npy')]
        sidecar=next((p for p in candidates if p.is_file()),None)
        if sidecar is None:raise FileNotFoundError(f'An ID sidecar is required for {path}')
        ids=np.load(sidecar,allow_pickle=False).astype(str)
        valid_path=path.with_suffix('.valid.npy')
        valid=np.load(valid_path,allow_pickle=False).astype(bool) if valid_path.is_file() else np.ones(len(x),bool)
    if x.ndim!=2 or len(ids)!=len(x) or valid.shape!=(len(x),):raise ValueError('Embedding shapes differ')
    if len(set(ids))!=len(ids) or not np.isfinite(x).all():raise ValueError('Duplicate IDs or nonfinite embeddings')
    if expected_ids is not None and not np.array_equal(ids,np.asarray(expected_ids).astype(str)):
        raise ValueError('Embedding cell order differs from the fixed roster')
    return x,ids,valid

def save_embeddings(path,x,ids,valid=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if valid is None:valid=np.ones(len(ids),bool)
    np.savez_compressed(path,embeddings=np.asarray(x,dtype=np.float32),cell_ids=np.asarray(ids,dtype=str),valid=valid)

def unit_vectors(x):
    x=np.asarray(x)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
