"""Prepare the four PC9 comparisons from downloaded GEO count files (no label-based QC)."""
from pathlib import Path
import argparse,gzip,io,zipfile
import numpy as np,pandas as pd,anndata as ad
from scipy import sparse
from scipy.io import mmread
def table_matrix(p,sep):
 blocks=[];genes=[];ids=None
 for chunk in pd.read_csv(p,sep=sep,index_col=0,chunksize=256):
  if ids is None:ids=chunk.columns.astype(str).to_numpy()
  else:assert np.array_equal(ids,chunk.columns.astype(str))
  v=chunk.to_numpy(dtype=np.float32);assert np.isfinite(v).all() and (v>=0).all()
  blocks.append(sparse.csr_matrix(v.T));genes.extend(chunk.index.astype(str))
 return sparse.hstack(blocks,format='csr'),np.array(genes),ids

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['sources','gene-info','roster','output']:p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
 symsets={};enssets={}
 for row in pd.read_csv(args.gene_info,sep='\t',dtype=str).itertuples(index=False):
  for v in row.dbXrefs.split('|'):
   if v.startswith('Ensembl:ENSG'):
    e=v.split(':')[1].split('.')[0];symsets.setdefault(row.Symbol,set()).add(e);enssets.setdefault(e,set()).add(row.Symbol)
 sm={s:next(iter(v)) for s,v in symsets.items() if len(v)==1};es={e:next(iter(v)) for e,v in enssets.items() if len(v)==1}
 arrays=[];qc=[];mapping_audit=[]
 def add(x,genes,ids,study,groups,cellline,train,replicates=None,provided_symbols=None,count_type='UMI'):
  ids=np.asarray(ids,dtype=str);genes=np.asarray(genes,dtype=str);assert len(set(ids))==len(ids) and x.shape==(len(ids),len(genes))
  x=sparse.csr_matrix(x,dtype=np.float32);x.sum_duplicates();x.eliminate_zeros();assert np.isfinite(x.data).all() and (x.data>=0).all()
  if count_type!='RSEM expected gene counts':assert np.allclose(x.data,np.round(x.data),rtol=0,atol=1e-6)
  mapped=[g.split('.')[0] if g.startswith('ENSG') else sm.get(g) for g in genes]
  valid=[i for i,g in enumerate(mapped) if g and g in es];unique=list(dict.fromkeys(mapped[i] for i in valid));lookup={g:i for i,g in enumerate(unique)}
  collapse=sparse.csr_matrix((np.ones(len(valid),dtype=np.float32),(valid,[lookup[mapped[i]] for i in valid])),shape=(len(genes),len(unique)))
  x=(x@collapse).tocsr();x.eliminate_zeros();x.sort_indices();symbols=[es[g] for g in unique]
  n=np.asarray(x.sum(1)).ravel();ng=x.getnnz(axis=1);mt=np.array([s.startswith('MT-') for s in symbols]);pct=np.asarray(x[:,mt].sum(1)).ravel()/np.maximum(n,1);keep=(n>=1000)&(ng>=200)&(pct<=.2)
  groups=np.asarray(groups,dtype=str);reps=np.asarray(replicates if replicates is not None else ['not_resolved']*len(ids),dtype=str)
  for g in np.unique(groups):
   ix=groups==g;qc.append(dict(study=study,cell_line=cellline,group=g,before=int(ix.sum()),after=int((ix&keep).sum()),count_type=count_type))
  cellids=np.array([study+'::'+cellline+'::'+s for s in ids]);obs=pd.DataFrame(dict(cell_id=cellids,source_barcode=ids,study=study,cell_line=cellline,group=groups,replicate=reps,train_key=train,n_counts=n,n_genes=ng,mito_fraction=pct,count_type=count_type),index=cellids)
  a=ad.AnnData(x[keep],obs=obs.loc[keep].copy(),var=pd.DataFrame({'human_ensembl_gene':unique,'gene_symbol':symbols},index=unique));arrays.append(a)
  mapping_audit.append(dict(study=study,cell_line=cellline,original_genes=len(genes),mapped_columns=len(valid),unique_ensembl=len(unique),unmapped=[genes[i] for i in range(len(genes)) if mapped[i] is None or mapped[i] not in es]));print('PREPARED',study,cellline,a.shape,flush=True)
 for gsm,group in [('GSM9125465','parental'),('GSM9125466','persister')]:
  p=next((args.sources).glob(gsm+'*.zip'))
  with zipfile.ZipFile(p) as z:
   def member(s):
    names=[n for n in z.namelist() if n.endswith(s) or n.endswith(s+'.gz')];assert len(names)==1,names
    b=z.read(names[0]);return gzip.decompress(b) if names[0].endswith('.gz') else b
   x=mmread(io.BytesIO(member('matrix.mtx'))).T.tocsr();g=pd.read_csv(io.BytesIO(member('features.tsv')),sep='\t',header=None);ids=pd.read_csv(io.BytesIO(member('barcodes.tsv')),sep='\t',header=None)[0].astype(str).to_numpy()
  add(x,g[0],np.array([gsm+'_'+s for s in ids]),'GSE303411',[group]*len(ids),'PC9','Data11')
 for gsm,group,rep in [('GSM8635332','DMSO','rep1'),('GSM8635333','DMSO','rep2'),('GSM8635334','gefitinib','rep1'),('GSM8635335','gefitinib','rep2'),('GSM8635336','osimertinib','rep1')]:
  x,g,ids=table_matrix(next((args.sources).glob(gsm+'*_lRNA_*.gz')),'\t');add(x,g,np.array([gsm+'_'+s for s in ids]),'GSE282074',[group]*len(ids),'PC9','Data11',[rep]*len(ids),count_type='RSEM expected gene counts')

 for gsm,arm,y in [('GSM5857972','parental',0),('GSM5857973','persister',1)]:
  x,g,ids=table_matrix(args.sources/f'{gsm}_PC9_WT_{arm}_counts_matrix.csv.gz',',')
  # Its original preparation retains unique symbol->Ensembl mappings, then sums duplicates.
  keep=np.array([v in sm for v in g]);g=g[keep];x=x[:,keep]
  ens=[sm[v] for v in g];unique=list(dict.fromkeys(ens));lookup={v:i for i,v in enumerate(unique)}
  collapse=sparse.csr_matrix((np.ones(len(ens)),(np.arange(len(ens)),[lookup[v] for v in ens])),shape=(len(ens),len(unique)))
  x=(x@collapse).astype(np.float32).tocsr();syms=dict(zip(ens,g));symbols=[syms[v] for v in unique]
  if not np.array_equal(x.data,np.round(x.data)):raise ValueError('GSE196018 expects integer counts')
  n=np.asarray(x.sum(1)).ravel();ng=np.asarray((x>0).sum(1)).ravel();mt=np.array([v.startswith('MT-') for v in symbols]);pct=np.asarray(x[:,mt].sum(1)).ravel()/np.maximum(n,1)
  keep=(n>=1000)&(ng>=200)&(pct<=.2);cellids=np.array(['GSE196018::'+gsm+'::'+v for v in ids])[keep]
  obs=pd.DataFrame({'cell_id':cellids,'study':'GSE196018','label':y},index=cellids)
  arrays.append(ad.AnnData(x[keep],obs=obs,var=pd.DataFrame({'human_ensembl_gene':unique,'gene_symbol':symbols},index=unique)))
  qc.append(dict(study='GSE196018',group=arm,before=len(ids),after=int(keep.sum()),count_type='UMI'))
 full=ad.concat(arrays,join='outer',merge='first',fill_value=0);full.X=full.X.tocsr();full.X.eliminate_zeros();full.X.sort_indices()
 symbols={g:s for a in arrays for g,s in zip(a.var_names,a.var.gene_symbol)}
 full.var['human_ensembl_gene']=full.var_names;full.var['gene_symbol']=[symbols[g] for g in full.var_names]
 roster=pd.read_csv(args.roster,sep='\t');pos=full.obs_names.get_indexer(roster.cell_id)
 if (pos<0).any() or len(full)!=len(roster):raise ValueError('Prepared cells differ from the fixed PC9 roster')
 full=full[pos].copy();full.obs=roster.set_index('cell_id',drop=False);full.write_h5ad(args.output,compression='lzf')
 pd.DataFrame(qc).to_csv(args.output.with_suffix('.qc.tsv'),sep='\t',index=False)

if __name__=='__main__':main()
