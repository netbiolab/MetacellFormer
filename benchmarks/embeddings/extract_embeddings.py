"""Extract native frozen cell representations with the paper task conventions.

Run each model in its own environment. --config supplies checkpoint/code paths.
--sampling-key is required for scPRINT: robustness uses cell/DATASET for all
conditions; other tasks retain the recorded job key. No labels reach encoders.
"""
from pathlib import Path
import argparse, hashlib, json, os, random, sys, tempfile
import numpy as np
import pandas as pd
from scipy import sparse
from benchmarks.shared._common import MODELS, save_embeddings, write_json

def mapped_geneformer(a,mapping_path):
    a=a.copy()
    if mapping_path:
        mapping=pd.read_csv(mapping_path,sep='\t',dtype=str)
        pos=a.var_names.get_indexer(mapping.source_gene)
        if (pos<0).any():raise ValueError('Geneformer source genes missing')
        a=a[:,pos].copy();a.var['ensembl_id']=mapping.ensembl_id.to_numpy()
    elif 'geneformer_ensembl_id' in a.var:
        # Frozen per-feature mapping; retain the historical alias columns.
        genes=a.var.geneformer_ensembl_id.fillna('').astype(str)
        keep=genes.str.startswith('ENSG').to_numpy()
        a=a[:,keep].copy();a.var['ensembl_id']=genes.to_numpy()[keep]
    elif 'human_ensembl_gene' in a.var:a.var['ensembl_id']=a.var.human_ensembl_gene.astype(str)
    elif 'ensembl_id' not in a.var:
        if not all(str(g).startswith('ENSG') for g in a.var_names):
            raise ValueError('Geneformer requires --gene-map or var[ensembl_id]')
        a.var['ensembl_id']=a.var_names.astype(str)
    a.X=sparse.csr_matrix(a.X);a.X.eliminate_zeros()
    a.obs['n_counts']=np.asarray(a.X.sum(1)).ravel();a.layers.clear()
    return a

def feature_tokenizer(a,tokenizer,model_dir):
    """Use the frozen feature mapping without collapsing historical aliases.

    Multiple source features may map to one gene. The original training
    tokenizer retains those features independently; all features contribute to
    the normalization denominator, including unmapped genes.
    """
    from metacellformer.binning import ValueBinner
    from metacellformer.pretraining_tokenizer import tokenize_single_cell_from_indices
    genes=a.var.metacellformer_gene_id.fillna('').astype(str).to_numpy()
    tokens=np.asarray([tokenizer.gene_to_token.get(g,0) for g in genes])
    keep=tokens>=4
    medians=np.ones(len(genes),np.float32)
    medians[keep]=tokenizer.medians[tokens[keep]-4]
    binner=ValueBinner.load(Path(model_dir)/'training_resources/value_bin_edges.npy',64)
    def encode(row):
        item=tokenize_single_cell_from_indices(
            row.indices,row.data,tokens,1023,binner,
            tokenizer.special['<cls>'],tokenizer.special['<pad>'],
            gene_subset_mask=keep,ordering_mode='geneformer_median_scaled_cp10k',
            rank_normalization_factors_by_index=medians)
        return {'input_ids':item.input_ids,'attention_mask':item.attention_mask}
    return encode,keep

def metacellformer(a,config,device,batch):
    import torch
    from metacellformer import GeneRankTokenizer,load_encoder,cell_embeddings
    tokenizer=GeneRankTokenizer(config['checkpoint']);encoder=load_encoder(config['checkpoint'],device)
    genes=a.var_names.astype(str).to_numpy()
    if 'human_ensembl_gene' in a.var:genes=a.var.human_ensembl_gene.astype(str).to_numpy()
    elif 'ensembl_id' in a.var:genes=a.var.ensembl_id.astype(str).to_numpy()
    feature_encode,feature_keep=(feature_tokenizer(a,tokenizer,config['checkpoint'])
                                 if 'metacellformer_gene_id' in a.var else (None,None))
    if feature_encode is None and len(set(genes))!=len(genes):raise ValueError('Collapse duplicate Ensembl IDs before tokenization')
    raw=sparse.csr_matrix(a.X);values=np.zeros((len(a),encoder.config.hidden_size),np.float32);valid=np.zeros(len(a),bool)
    for start in range(0,len(a),batch):
        encoded=[];rows=[]
        for row in range(start,min(start+batch,len(a))):
            x=raw[row];total=float(a.obs.iloc[row]['total_counts_all_features']) if 'total_counts_all_features' in a.obs else float(x.sum())
            if feature_encode is not None:
                if total<=0 or not np.any(feature_keep[x.indices] & (x.data>0)):continue
                item=feature_encode(x)
            else:
                if total<=0 or not any(genes[i] in tokenizer.gene_to_token and v>0 for i,v in zip(x.indices,x.data)):continue
                item=tokenizer.encode(genes[x.indices],x.data,total_counts_all_features=total)
            encoded.append(item);rows.append(row)
        if encoded:
            ids=torch.as_tensor(np.stack([x['input_ids'] for x in encoded]),device=device)
            mask=torch.as_tensor(np.stack([x['attention_mask'] for x in encoded]),device=device)
            values[rows]=cell_embeddings(encoder,ids,mask).float().cpu().numpy();valid[rows]=True
    return values,valid

def geneformer(a,config,device,batch,task,mapping,work):
    import torch
    from geneformer import TranscriptomeTokenizer,EmbExtractor
    from datasets import load_from_disk
    a=mapped_geneformer(a,mapping);a.obs['cell_id']=a.obs_names.astype(str)
    source=work/'input';source.mkdir(parents=True);a.write_h5ad(source/'input.h5ad')
    version=config['tokenizer_version'];kwargs={}
    if config.get('token_dictionary'):kwargs['token_dictionary_file']=config['token_dictionary']
    tok=TranscriptomeTokenizer(custom_attr_name_dict={'cell_id':'cell_id'},nproc=2,model_version=version,**kwargs)
    tok.tokenize_data(source,work,'input',file_format='h5ad')
    ds=load_from_disk(str(work/'input.dataset'));expected=a.obs_names.astype(str).to_numpy()
    if set(ds['cell_id'])!=set(expected) or len(ds)!=len(expected):raise ValueError('Geneformer changed the cell roster')
    if not device.startswith('cuda'):raise ValueError('The official Geneformer extractor requires CUDA')
    if task=='robustness':
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    ex=EmbExtractor(model_type='Pretrained',emb_mode='cell',max_ncells=None,emb_layer=-1,
                    emb_label=['cell_id'],forward_batch_size=batch,nproc=2,model_version=version,**kwargs)
    result=pd.DataFrame(ex.extract_embs(model_directory=config['checkpoint'],input_data_file=str(work/'input.dataset'),
                      output_directory=str(work),output_prefix='native',output_torch_embs=False))
    values=result.set_index('cell_id').loc[expected].to_numpy(dtype=np.float32)
    return values,np.ones(len(a),bool)


def symbol_input(a):
    """Sum repeated native gene symbols, retaining the original cell order."""
    import anndata as ad
    if 'gene_symbol' not in a.var:return a
    names=a.var.gene_symbol.astype(str).tolist();unique=list(dict.fromkeys(names));index={g:i for i,g in enumerate(unique)}
    collapse=sparse.csr_matrix((np.ones(len(names),np.float32),(np.arange(len(names)),[index[g] for g in names])),shape=(len(names),len(unique)))
    return ad.AnnData(sparse.csr_matrix(a.X)@collapse,obs=a.obs.copy(),var=pd.DataFrame(index=unique))

def scgpt(a,config,device,batch,task):
    a=symbol_input(a)
    import torch
    from scgpt.tasks.cell_emb import embed_data
    random.seed(123);np.random.seed(123);torch.manual_seed(123)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(123)
    if 'gene_symbol' in a.var:a.var['feature_name']=a.var.gene_symbol.astype(str)
    else:a.var['feature_name']=a.var_names.astype(str)
    z=embed_data(a,model_dir=config['checkpoint'],gene_col='feature_name',max_length=1200,
                 batch_size=batch,obs_to_save=['cell_id'],device=device,
                 use_fast_transformer=config.get('fast_transformer',{}).get(task,False),return_new_adata=False)
    if not np.array_equal(z.obs_names.astype(str),a.obs_names.astype(str)):raise ValueError('scGPT reordered cells')
    return np.asarray(z.obsm['X_scGPT'],np.float32),np.ones(len(a),bool)

def scprint(a,config,device,batch,task,sampling_key):
    import torch
    from benchmarks.embeddings import _scprint as w
    if not sampling_key:raise ValueError('scPRINT requires the recorded --sampling-key')
    w.CHECKPOINT=Path(config['checkpoint']);w.GENE_INFO=Path(config['gene_info'])
    w.REPO=Path(config['code']);w.SCDL_REPO=Path(config['scdataloader_code'])
    torch.manual_seed(0);np.random.seed(0);torch.set_float32_matmul_precision('medium')
    dev=torch.device(device);transformer='normal' if task=='robustness' else config.get('transformer','flash')
    model,_=w.load_model(dev,transformer)
    x=sparse.csr_matrix(a.X);depths=np.asarray(x.sum(1)).ravel().astype(np.float32);ids=a.obs_names.to_numpy(dtype=str)
    mapping,human,_=w.source_to_model(a.var,model.genes)
    indices=[model.classes.index(n)+(2 if model.depth_atinput else 1)+(1 if model.use_metacell_token else 0) for n in w.PRED_EMBEDDING]
    values=np.empty((len(a),model.d_model),np.float32)
    with torch.inference_mode():
        for start in range(0,len(a),batch):
            rows=range(start,min(start+batch,len(a)))
            gene,expression,depth=w.collate_rows(x,depths,ids,rows,mapping,human,sampling_key)
            gene,expression,depth=gene.to(dev),expression.to(dev),depth.to(dev)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
                result=model(gene,expression,depth_mult=expression.sum(1),req_depth=depth,do_class=False)
                values[start:start+len(rows)]=result['cell_embs'][:,indices,:].mean(1).float().cpu().numpy()
    return values,np.ones(len(a),bool)

def cellfm(a,config,device,batch,work):
    """Run the official CellFM MindSpore encoder with fixed input sampling."""
    import scanpy as sc
    import mindspore as ms
    import mindspore.dataset as ds
    from scipy.sparse import issparse

    a=symbol_input(a)
    os.environ['METACELLFORMER_CELLFM_CODE']=config['code']
    repo=Path(config['code'])
    sys.path.insert(0,str(repo));sys.path.insert(1,str(repo/'tutorials/CellAnnotation'))
    from data_process import Prepare
    from config import Config
    from model import CellFM

    def digest(values):
        payload=json.dumps(np.asarray(values).astype(str).tolist(),ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    class Data:
        def __init__(self,job,gene_info):
            source=sc.read_h5ad(job['source'],backed='r')
            geneset={gene:i+1 for i,gene in enumerate(gene_info.index.astype(str))}
            genes=np.intersect1d(source.var_names.astype(str),gene_info.index.astype(str))
            if not len(genes):raise RuntimeError(f"{job['name']}: no CellFM genes")
            try:
                selected=source[int(job['start']):int(job['end']),genes].to_memory()
                x=selected.layers[job['layer']] if job['layer']!='X' else selected.X
                self.data=x.toarray().astype(np.float32) if issparse(x) else np.asarray(x,dtype=np.float32)
                self.gene=np.asarray([geneset[gene] for gene in selected.var_names.astype(str)],dtype=np.int32)
                self.ids=np.asarray(selected.obs['cell_barcode'] if job['kind']=='scdrug' and 'cell_barcode' in selected.obs else selected.obs_names).astype(str)
            finally:
                source.file.close()
            if digest(self.ids)!=job['expected_ids_sha256']:raise RuntimeError(f"{job['name']}: source IDs changed")

        def __len__(self):return len(self.data)

        def __getitem__(self,i):
            x=self.data[i].reshape(-1).astype(np.float32,copy=True);total=float(x.sum())
            if total>0:x*=np.float32(1e4/total);np.log1p(x,out=x)
            return x,self.gene,np.int32(0)

    class DeterministicPrepare(Prepare):
        def sample(self,data,nonz,zero):
            # Stable expression-derived sampling independent of worker order.
            h=hashlib.blake2b(digest_size=8)
            h.update(np.asarray(nonz,dtype=np.int32).tobytes())
            h.update(np.asarray(data[nonz],dtype=np.float32).tobytes())
            rng=np.random.default_rng(int.from_bytes(h.digest(),'little'))
            cutted=np.array([],dtype=np.int64)
            if len(nonz)>self.cut:
                weights=np.log1p(data[nonz]);weights=weights/weights.sum()
                order=rng.choice(np.arange(len(nonz)),len(nonz),replace=False,p=weights)
                order=nonz[order];nonz=np.sort(order[:self.cut]);cutted=np.sort(order[self.cut:])
            length=self.zero_len or (self.pad_len-len(nonz))
            sampled_zero=rng.choice(zero,length,replace=False)
            return data,nonz,cutted,sampled_zero,len(nonz)

    def loader(data,prep):
        dataset=ds.GeneratorDataset(data,column_names=['data','gene','label'],shuffle=False)
        dataset=dataset.map(prep.seperate,input_columns=['data'],output_columns=['data','nonz','zero'])
        dataset=dataset.map(prep.sample,input_columns=['data','nonz','zero'],output_columns=['data','nonz','cuted','z_sample','seq_len'])
        dataset=dataset.map(prep.compress,input_columns=['data','nonz'],output_columns=['data','nonz_data','nonz'])
        dataset=dataset.map(prep.compress,input_columns=['gene','nonz'],output_columns=['gene','nonz_gene','nonz'])
        dataset=dataset.map(prep.attn_mask,input_columns=['seq_len'],output_columns=['zero_idx'])
        dataset=dataset.map(prep.pad_zero,input_columns=['nonz_data'])
        dataset=dataset.map(prep.pad_zero,input_columns=['nonz_gene'])
        return dataset.project(columns=['nonz_data','nonz_gene','zero_idx','label']).batch(batch,drop_remainder=False,num_parallel_workers=4)

    ms.set_context(device_target='GPU' if device.startswith('cuda') else 'CPU',device_id=int(device.split(':')[1]) if ':' in device else 0,mode=ms.GRAPH_MODE)
    ms.set_seed(0)
    path=work/'input.h5ad';a.write_h5ad(path)
    job=dict(source=str(path),name='input',kind='cell',layer='X',start=0,end=len(a),expected_ids_sha256=digest(a.obs_names))
    info=pd.read_csv(config['gene_info'],index_col=0);data=Data(job,info)
    cfg=Config();cfg.recompute=False;net=CellFM(len(info),cfg)
    wanted=set(net.parameters_dict())
    tensors=ms.load_checkpoint(str(config['checkpoint']),choice_func=lambda name:name in wanted)
    missing,_=ms.load_param_into_net(net,tensors,strict_load=True)
    if missing:raise ValueError(f'Missing CellFM parameters: {missing}')
    net.set_train(False)
    prep=DeterministicPrepare(cfg.nonz_len,pad=1,mask_ratio=0,random=False)
    parts=[]
    for item in loader(data,prep).create_dict_iterator():
        parts.append(net.encode(item['nonz_data'],item['nonz_gene'],item['zero_idx'])[0][:,0,:].asnumpy().astype(np.float32))
    return np.concatenate(parts),np.ones(len(a),bool)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--task',choices=['cell','drug','robustness','pc9','perturbation'],required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--gene-map',type=Path);p.add_argument('--sampling-key');p.add_argument('--layer')
    args=p.parse_args()
    import anndata as ad
    config=json.loads(args.config.read_text())[args.model]
    a=ad.read_h5ad(args.input)
    if args.layer:a.X=a.layers[args.layer].copy()
    a.layers.clear();ids=a.obs_names.to_numpy(dtype=str)
    if len(set(ids))!=len(ids):raise ValueError('Duplicate cells')
    totals=a.obs['total_counts_all_features'].to_numpy() if 'total_counts_all_features' in a.obs else None
    a.obs=pd.DataFrame({'cell_id':ids},index=ids)
    if totals is not None:a.obs['total_counts_all_features']=totals
    if not np.isfinite(a.X.data if sparse.issparse(a.X) else a.X).all():raise ValueError('Nonfinite expression')
    with tempfile.TemporaryDirectory(prefix='metacellformer_') as tmp:
        work=Path(tmp)
        if args.model.startswith('MetacellFormer'):values,valid=metacellformer(a,config,args.device,args.batch_size)
        elif args.model.startswith('Geneformer'):values,valid=geneformer(a,config,args.device,args.batch_size,args.task,args.gene_map,work)
        elif args.model=='scGPT':values,valid=scgpt(a,config,args.device,args.batch_size,args.task)
        elif args.model=='scPRINT':values,valid=scprint(a,config,args.device,args.batch_size,args.task,args.sampling_key)
        else:values,valid=cellfm(a,config,args.device,args.batch_size,work)
    if not np.isfinite(values).all():raise ValueError('Nonfinite embeddings')
    save_embeddings(args.output,values,ids,valid)
    write_json({'model':args.model,'task':args.task,'rows':len(ids),'dimension':values.shape[1],
                'sampling_key':args.sampling_key},args.output.with_suffix('.json'))
if __name__=='__main__':main()
