"""Export native input gene embeddings as genes/embedding NPZ for GTEx and HuRI.
Supply the exact native vocabulary and NCBI mapping snapshot used in the run.
"""
from pathlib import Path
import argparse,json,gzip,pickle
import numpy as np
import pandas as pd
from benchmarks.shared._common import MODELS

def export(args):
    cfg=json.loads(args.config.read_text())[args.model]
    path=Path(cfg['checkpoint'])
    if args.model.startswith('MetacellFormer'):
        from safetensors import safe_open
        with safe_open(str(path/'model.safetensors'),framework='np') as z:matrix=z.get_tensor('bert.embeddings.word_embeddings.weight')
        genes=json.loads((path/'gene_vocabulary.json').read_text())['genes'];rows=np.arange(len(genes))+4
    elif args.model.startswith('Geneformer'):
        from transformers import BertForMaskedLM
        model=BertForMaskedLM.from_pretrained(path,local_files_only=True)
        matrix=model.bert.embeddings.word_embeddings.weight.detach().float().numpy()
        # Only load a trusted official Geneformer token dictionary.
        with open(cfg['token_dictionary'],'rb') as f:vocab=pickle.load(f)
        genes=[g for g in vocab if str(g).startswith('ENSG')];rows=[vocab[g] for g in genes]
    elif args.model=='scGPT':
        import torch
        checkpoint=torch.load(path/'best_model.pt',map_location='cpu',weights_only=False)
        matrix=checkpoint['encoder.embedding.weight'].float().numpy();vocab=json.loads((path/'vocab.json').read_text())
        genes=[g for g in vocab if not g.startswith('<')];rows=[vocab[g] for g in genes]
    elif args.model=='scPRINT':
        from benchmarks.embeddings import _scprint as w
        w.CHECKPOINT=path;w.GENE_INFO=args.gene_info;w.export_static(args.output);return
    else:
        import mindspore as ms
        matrix=ms.load_checkpoint(str(path),choice_func=lambda n:n=='gene_emb')['gene_emb'].asnumpy()
        genes=pd.read_csv(cfg['gene_info'],index_col=0).index.astype(str).tolist();rows=np.arange(len(genes))+1
    info=pd.read_csv(args.gene_info,sep='\t',dtype=str);mapping={}
    for record in info.itertuples(index=False):
        for ref in record.dbXrefs.split('|'):
            if ref.startswith('Ensembl:ENSG'):mapping.setdefault(ref.split(':')[1].split('.')[0],record.Symbol)
    names=[];selected=[];seen=set()
    for gene,row in zip(genes,rows):
        name=mapping.get(gene.split('.')[0]) if gene.startswith('ENSG') else gene
        if name and name not in seen:
            seen.add(name);names.append(name);selected.append(row)
    values=np.asarray(matrix[selected],dtype=np.float32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite input gene weights')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,genes=np.asarray(names,dtype=str),embedding=values)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','gene-info','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);export(p.parse_args())
if __name__=='__main__':main()
