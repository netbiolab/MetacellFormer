"""Figure 6 response vectors, within/cross-screen retrieval and CORUM validation."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score
from benchmarks.shared._common import embeddings,write_table,write_json
MC_ID='MetacellFormer';SC_ID='MetacellFormer-SC'
def norm(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def aggregate_delta(x,ix,selected=None):
    if selected is not None:x=x[selected];ix=ix.iloc[selected].reset_index(drop=True)
    pert=ix.cell_role.eq('perturbed').to_numpy();control=ix.cell_role.eq('control').to_numpy()
    assert np.all(pert|control)
    key=ix.target_gene.astype(str)+'\t'+ix.batch_group.astype(str)
    codes,keys=pd.factorize(key[pert],sort=True)
    g=sparse.csr_matrix((np.ones(pert.sum(),np.float32),(codes,np.arange(pert.sum()))),shape=(len(keys),pert.sum()))
    counts=np.asarray(g.sum(1)).ravel();means=(g@x[pert])/counts[:,None]
    meta=pd.DataFrame([s.split('\t') for s in keys],columns=['target_gene','batch_group']);meta['n_cells']=counts.astype(int)
    controls={b:x[control & ix.batch_group.eq(b).to_numpy()].mean(0) for b in sorted(ix.batch_group.unique())}
    assert all(np.isfinite(v).all() for v in controls.values())
    delta=norm(means-np.stack([controls[b] for b in meta.batch_group]))
    return delta,meta

def gene_similarity(delta,meta,genes,batches=None,device='cuda:2'):
    """Equal query-batch mean of equal other-batch candidate means of cosine.

    Every gene pair excludes same-batch comparisons BEFORE aggregation.
    This is a gene-level diagnostic, not canonical batch-item mAP.
    """
    import torch
    if batches is None:batches=sorted(meta.batch_group.astype(str).unique())
    gi={g:i for i,g in enumerate(genes)};bi={str(b):i for i,b in enumerate(batches)}
    keep=meta.target_gene.isin(gi)&meta.batch_group.astype(str).isin(bi)
    mm=meta[keep];dd=np.asarray(delta[keep]);n=len(genes)
    cube=np.zeros((len(batches),n,dd.shape[1]),np.float32);present=np.zeros((len(batches),n),bool)
    ii=mm.target_gene.map(gi).to_numpy();bb=mm.batch_group.astype(str).map(bi).to_numpy()
    cube[bb,ii]=dd;present[bb,ii]=True
    assert (present.sum(0)>=2).all(),'Need >=2 batches per candidate gene'
    t=torch.as_tensor(cube,device=device);p=torch.as_tensor(present,device=device)
    sums=t.sum(0);counts=p.sum(0);scores=torch.zeros((n,n),device=device)
    for b in range(len(batches)):
        scores+=(t[b]@(sums-t[b]).T)/(counts-p[b].int()).clamp(min=1)[None,:]
    scores/=counts[:,None]
    s=scores.cpu().numpy();np.fill_diagonal(s,-np.inf)
    centroid=norm(cube.sum(0)/present.sum(0)[:,None])
    return s,centroid
def eligible_groups(meta,n=3,min_controls=20):
    pert=meta.cell_role.eq('perturbed').to_numpy();idx=np.flatnonzero(pert)
    control_batches=meta.loc[meta.cell_role.eq('control'),'batch_group'].value_counts()
    eligible_batches=set(control_batches[control_batches>=min_controls].index.astype(str))
    raw=meta.iloc[idx].groupby(['target_gene','batch_group'],sort=True).indices
    groups={key:idx[np.asarray(pos)] for key,pos in raw.items() if len(pos)>=n and str(key[1]) in eligible_batches}
    target_counts=pd.Series([k[0] for k in groups]).value_counts()
    return {key:value for key,value in groups.items() if target_counts[key[0]]>=2}

def select(groups,meta,targets,seed,n_controls=20):
    rng=np.random.default_rng(seed);chosen=[];items=[]
    for target in targets:
        keys=sorted(k for k in groups if k[0]==target)
        picked=sorted(rng.choice(len(keys),2,replace=False))
        for index in picked:
            key=keys[index];chosen.append(rng.choice(groups[key],3,replace=False));items.append(key)
    cells=np.stack(chosen)
    item=pd.DataFrame(items,columns=['target_gene','batch_group'])
    controls={b:np.flatnonzero(meta.cell_role.eq('control').to_numpy() & meta.batch_group.eq(b).to_numpy())
              for b in sorted(item.batch_group.unique())}
    if min(map(len,controls.values()))<n_controls:raise AssertionError(f'Fewer than {n_controls} same-batch controls')
    controls={b:rng.choice(v,n_controls,replace=False) for b,v in controls.items()}
    return cells,item,controls
def target_vectors(embeddings, cells, items, controls, targets):
    batches = {b: norm(np.asarray(embeddings[indices])).mean(0) for b, indices in controls.items()}
    flat = cells.ravel()
    x = norm(np.asarray(embeddings[flat])).reshape(cells.shape[0], cells.shape[1], -1)
    delta = norm(x.mean(1) - np.stack([batches[b] for b in items.batch_group]))
    assert np.array_equal(items.target_gene.to_numpy().reshape(-1, 2)[:, 0], targets)
    return norm(delta.reshape(len(targets), 2, -1).mean(1))

def score_matrix(q, r, targets, labels, model, direction, seed):
    similarity = q @ r.T
    rows = []
    for i, gene in enumerate(targets):
        order = np.argsort(-similarity[i], kind="stable")
        position = int(np.flatnonzero(order == i)[0]) + 1
        candidate = np.arange(len(targets)) != i
        truth = (labels[candidate] == labels[i])
        functional_ap = average_precision_score(truth, similarity[i, candidate]) if labels[i] != "Unassigned" and truth.any() else np.nan
        rows.append(dict(direction=direction, model=model, seed=seed, target_gene=gene,
                         exact_rank=position, reciprocal_rank=1 / position,
                         recall_at_1=position <= 1, recall_at_5=position <= 5,
                         recall_at_10=position <= 10, functional_ap=functional_ap))
    return rows
def ap_scores(v,meta,device):
    """Cross-batch item AP for exact target and annotated functional module."""
    target,_=pd.factorize(meta.target_gene,sort=True)
    batch,_=pd.factorize(meta.batch_group.astype(str),sort=True)
    module,_=pd.factorize(meta.module,sort=True)
    n=len(meta);t=torch.as_tensor(v,device=device,dtype=torch.float32)
    target=torch.as_tensor(target,device=device);batch=torch.as_tensor(batch,device=device)
    module=torch.as_tensor(module,device=device)
    annotated=torch.as_tensor(meta.module.ne('Unassigned').to_numpy(),device=device)
    rank=torch.arange(1,n+1,device=device,dtype=torch.float32)[None,:]
    exact=np.empty(n,np.float32);functional=np.full(n,np.nan,np.float32)
    with torch.inference_mode():
        for start in range(0,n,128):
            stop=min(start+128,n);q=t[start:stop]
            score=q@t.T
            score.masked_fill_(batch[start:stop,None]==batch[None,:],-1e9)
            order=torch.argsort(score,dim=1,descending=True,stable=True)
            candidate_batch=batch[order]
            valid=candidate_batch!=batch[start:stop,None]
            pe=(target[order]==target[start:stop,None]) & valid
            denom=pe.sum(1).clamp(min=1)
            exact[start:stop]=((pe.cumsum(1)/rank)*pe).sum(1).div(denom).cpu().numpy()
            pf=(module[order]==module[start:stop,None]) & valid & annotated[start:stop,None] & annotated[order]
            fd=pf.sum(1)
            fs=((pf.cumsum(1)/rank)*pf).sum(1).div(fd.clamp(min=1))
            arr=fs.cpu().numpy();arr[fd.cpu().numpy()==0]=np.nan
            functional[start:stop]=arr
    return exact,functional
def corum_sets(genes, corum):
    raw = pd.read_csv(corum, sep="\t")
    universe = set(genes)
    eligible = []
    seen = set()
    for row in raw.itertuples(index=False):
        members = frozenset(str(row.subunits_gene_name).split(";")) & universe
        if len(members) < 3 or members in seen:
            continue
        seen.add(members)
        eligible.append((int(row.complex_id), str(row.complex_name), members))
    membership = {g: set() for g in genes}
    for cid, _, members in eligible:
        for g in members:
            membership[g].add(cid)
    annotated = np.array([bool(membership[g]) for g in genes])
    positive = np.zeros((len(genes), len(genes)), dtype=bool)
    for _, _, members in eligible:
        idx = np.array([np.searchsorted(genes, g) for g in members])
        positive[np.ix_(idx, idx)] = True
    np.fill_diagonal(positive, False)
    return eligible, membership, annotated, positive
def _all_complexes(genes: np.ndarray, scores: dict[str, np.ndarray],
                   sets: list, annotated: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = {g: i for i, g in enumerate(genes)}
    query_rows = []
    for cid, name, members in sets:
        for gene in sorted(members):
            i = index[gene]
            candidates = np.flatnonzero(annotated & (genes != gene))
            truth = np.isin(genes[candidates], list(members))
            if int(truth.sum()) != len(members) - 1:
                raise ValueError(f'CORUM membership/candidate mismatch: {cid}/{gene}')
            for model, similarity in scores.items():
                rank = np.argsort(-similarity[i, candidates], kind='stable')
                query_rows.append(dict(complex_id=cid, complex_name=name,
                                       n_members=len(members), query=gene, model_id=model,
                                       n_candidates=len(candidates), n_positive=int(truth.sum()),
                                       ap=average_precision_score(truth, similarity[i, candidates]),
                                       recall_at_10=float(truth[rank[:10]].sum() / truth.sum())))
    queries = pd.DataFrame(query_rows)
    grouped = queries.groupby(['complex_id', 'complex_name', 'n_members', 'model_id'],
                               as_index=False).agg(mean_ap=('ap', 'mean'),
                                                   mean_recall_at_10=('recall_at_10', 'mean'))
    wide = grouped.pivot(index=['complex_id', 'complex_name', 'n_members'], columns='model_id',
                         values=['mean_ap', 'mean_recall_at_10'])
    complexes = pd.DataFrame({
        'complex_id': wide.index.get_level_values(0),
        'complex_name': wide.index.get_level_values(1),
        'n_members': wide.index.get_level_values(2),
        'mc_ap': wide[('mean_ap', MC_ID)].to_numpy(),
        'sc_ap': wide[('mean_ap', SC_ID)].to_numpy(),
        'mc_recall_at_10': wide[('mean_recall_at_10', MC_ID)].to_numpy(),
        'sc_recall_at_10': wide[('mean_recall_at_10', SC_ID)].to_numpy(),
    })
    complexes['mc_minus_sc_ap'] = complexes.mc_ap - complexes.sc_ap
    return queries, complexes
def cross_screen(args):
    root=args.contracts;targets=np.load(root/'targets.npy').astype(str)
    labels=pd.read_csv(root/'target_labels.tsv',sep='\t').set_index('target_gene').loc[targets,'module'].to_numpy()
    arrays={}
    for screen,path in [('K562',args.k562),('RPE1',args.rpe1)]:
        ids=np.load(root/f'{screen}_cell_ids.npy').astype(str)
        arrays[screen],_,valid=embeddings(path,ids)
        if not valid.all():raise ValueError('Incomplete response embeddings')
    old_arrays={}
    for screen,path in [('K562',args.k562_seed42),('RPE1',args.rpe1_seed42)]:
        ids=np.load(root/f'{screen}_seed42_cell_ids.npy').astype(str)
        old_arrays[screen],_,valid=embeddings(path,ids)
        if not valid.all():raise ValueError('Incomplete original seed-42 embeddings')
    rows=[];within=[]
    for seed in range(42,52):
        vectors={}
        for screen in arrays:
            x=old_arrays[screen] if seed==42 else arrays[screen]
            cells=np.load(root/f'{screen}_seed{seed}_cells.npy')
            items=pd.read_csv(root/f'{screen}_seed{seed}_items.tsv',sep='\t',dtype={'batch_group':str})
            controls=json.loads((root/f'{screen}_seed{seed}_controls.json').read_text())
            vectors[screen]=target_vectors(x,cells,items,controls,targets)
            delta=norm(norm(x[cells]).mean(1)-np.stack([norm(x[controls[b]]).mean(0) for b in items.batch_group]))
            items['module']=items.target_gene.map(dict(zip(targets,labels)))
            exact,functional=ap_scores(delta,items,args.device)
            items['exact_ap']=exact;items['functional_ap']=functional;items['screen']=screen;items['seed']=seed;items['model']=args.model
            within.append(items)
            args.output_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(args.output_dir/f'{screen}_target_vectors_s{seed}.npz',genes=targets,embedding=vectors[screen])
        for first,second in [('K562','RPE1'),('RPE1','K562')]:
            rows+=score_matrix(vectors[first],vectors[second],targets,labels,args.model,f'{first} → {second}',seed)
    write_table(pd.DataFrame(rows),args.output_dir/'cross_screen_queries.tsv.gz')
    write_table(pd.concat(within),args.output_dir/'within_screen_queries.tsv.gz')

def corum_run(args):
    meta=pd.read_csv(args.metadata,sep='\t',dtype=str)
    splits=pd.read_csv(args.batch_assignments,sep='\t',dtype=str).rename(columns={'Batch ID':'batch_group','Split':'split'})
    if 'split' not in splits:raise ValueError('batch assignments require batch_group,split')
    batches={s:splits.loc[splits.split.eq(s),'batch_group'].tolist() for s in ['discovery','validation']}
    if set(batches['discovery'])&set(batches['validation']):raise ValueError('Overlapping discovery and validation batches')
    if not all(batches.values()):raise ValueError('Empty split')
    graphs={s:{} for s in batches};genes=None
    for model,path in [(MC_ID,args.mc),(SC_ID,args.sc)]:
        x,_,valid=embeddings(path,meta.cell_id)
        if not valid.all():raise ValueError('Incomplete response embeddings')
        delta,items=aggregate_delta(norm(x),meta)
        if genes is None:genes=np.array(sorted(items.target_gene.unique()))
        for split,bs in batches.items():
            graphs[split][model],_=gene_similarity(delta,items,genes,bs,args.device)
    sets,membership,annotated,positive=corum_sets(genes,args.corum)
    discovery,complexes=_all_complexes(genes,graphs['discovery'],sets,annotated)
    eligible=complexes[complexes.n_members.ge(5)].sort_values(['mc_minus_sc_ap','complex_id'],ascending=[False,True],kind='stable')
    chosen=int(eligible.iloc[0].complex_id)
    # Selection uses discovery only, then evaluates the frozen complex on validation.
    members=next(m for cid,_,m in sets if cid==chosen)
    pair=discovery[discovery.complex_id.eq(chosen)].pivot(index='query',columns='model_id',values='ap')
    exemplar=(pair[MC_ID]-pair[SC_ID]).idxmax()
    val,_=_all_complexes(genes,graphs['validation'],sets,annotated)
    val=val[val.complex_id.eq(chosen)].copy();val['split']='validation';val['candidate_set']='all annotated'
    write_table(discovery,args.output_dir/'discovery_query_scores.tsv')
    write_table(complexes,args.output_dir/'discovery_complexes.tsv')
    write_table(val,args.output_dir/'validation_query_scores.tsv')
    union=[]
    for model,scores in graphs['discovery'].items():
        for i,g in enumerate(genes):
            candidate=annotated.copy();candidate[i]=False
            truth=positive[i,candidate]
            if truth.any() and (~truth).any():union.append(dict(model=model,target_gene=g,ap=average_precision_score(truth,scores[i,candidate])))
    write_table(pd.DataFrame(union),args.output_dir/'corum_query_scores.tsv')
    write_json(dict(selected_complex=chosen,selected_query=exemplar,members=sorted(members),selection='largest discovery MC minus SC complex AP, >=5 members'),args.output_dir/'selection.json')

def umap_run(args):
    import umap
    with np.load(args.vectors) as z:genes=z['genes'];x=z['embedding']
    xy=umap.UMAP(n_neighbors=15,min_dist=.3,metric='cosine',random_state=42,n_jobs=1).fit_transform(x)
    write_table(pd.DataFrame({'target_gene':genes,'UMAP1':xy[:,0],'UMAP2':xy[:,1]}),args.output)


def within_screen(args):
    meta={screen:pd.read_csv(path,sep='\t',dtype=str) for screen,path in [('K562',args.k562_metadata),('RPE1',args.rpe1_metadata)]}
    groups={screen:eligible_groups(m) for screen,m in meta.items()}
    targets=sorted({k[0] for k in groups['K562']} & {k[0] for k in groups['RPE1']})
    ann=pd.read_csv(args.annotations,sep='\t');lookup=ann[ann.primary_clean_eligible.eq(1)].set_index('target_gene').primary_functional_group.to_dict()
    result=[]
    for screen,path in [('K562',args.k562),('RPE1',args.rpe1)]:
        x,_,valid=embeddings(path,meta[screen].cell_id)
        if not valid.all():raise ValueError('Incomplete full-gene embeddings')
        x=norm(x)
        for seed in range(42,52):
            cells,items,controls=select(groups[screen],meta[screen],targets,seed)
            delta=norm(x[cells].mean(1)-np.stack([x[controls[b]].mean(0) for b in items.batch_group]))
            items['module']=items.target_gene.map(lookup).fillna('Unassigned')
            exact,functional=ap_scores(delta,items,args.device)
            items['exact_ap']=exact;items['functional_ap']=functional;items['screen']=screen;items['seed']=seed;items['model']=args.model;result.append(items)
    write_table(pd.concat(result),args.output_dir/'within_screen_queries.tsv.gz')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('cross-screen')
    for name in ['contracts','k562','rpe1','k562-seed42','rpe1-seed42','output-dir']:q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--model',choices=[MC_ID,SC_ID],required=True);q.add_argument('--device',default='cpu');q.set_defaults(run=cross_screen)
    q=sub.add_parser('within-screen')
    for name in ['k562','rpe1','k562-metadata','rpe1-metadata','annotations','output-dir']:q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--model',choices=[MC_ID,SC_ID],required=True);q.add_argument('--device',default='cpu');q.set_defaults(run=within_screen)
    q=sub.add_parser('corum')
    for name in ['mc','sc','metadata','batch-assignments','corum','output-dir']:q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--device',default='cpu');q.set_defaults(run=corum_run)
    q=sub.add_parser('umap');q.add_argument('--vectors',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.set_defaults(run=umap_run)
    args=p.parse_args();args.run(args)
if __name__=='__main__':main()
