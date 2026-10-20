"""Figure 2 numerical analyses on matched, fixed profiles (counts or token presence)."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from benchmarks.shared._common import write_table,write_json
N_PROFILES=1200;N_CONTROLS=5
def token_matrix(ids: np.ndarray, bins: np.ndarray, n_genes: int, weighted: bool) -> sp.csr_matrix:
    if ids.shape != bins.shape or ids.shape[1] != 1024:
        raise ValueError('token array shape mismatch')
    valid = ids[:, 1:] >= 4
    if np.any(ids[:, 1:][valid] >= n_genes + 4):
        raise ValueError('token outside vocab')
    row, col = np.nonzero(valid)
    gene = ids[:, 1:][row, col] - 4
    values = bins[:, 1:][row, col].astype(np.float32) if weighted else np.ones(len(row), np.float32)
    if weighted and (np.any(values < 1) or np.any(values > 64)):
        raise ValueError('gene value bins must be 1..64')
    matrix = sp.csr_matrix((values, (row, gene)), shape=(len(ids), n_genes), dtype=np.float32)
    if matrix.nnz != len(row):
        raise ValueError('duplicate gene token within a profile')
    return matrix

def score_matrix(matrix: sp.csr_matrix, labels: dict[str, dict[int, np.ndarray]],
                 device: torch.device, block: int = 128) -> list[dict]:
    if matrix.shape[0] != N_PROFILES:
        raise ValueError('profile count mismatch')
    query = np.array(sorted(set(labels['HuRI']) | set(labels['CORUM'])), dtype=np.int32)
    x = torch.from_numpy(matrix.toarray()).to(device)
    x -= x.mean(dim=0, keepdim=True)
    norms = torch.linalg.vector_norm(x, dim=0)
    variable = norms > 1e-7
    safe_norms = torch.where(variable, norms, 1)
    result = []
    for start in range(0, len(query), block):
        q = query[start:start + block]
        q_t = torch.as_tensor(q, device=device, dtype=torch.long)
        s = x[:, q_t].T @ x
        s /= safe_norms[q_t, None] * safe_norms[None, :]
        s[:, ~variable] = -float('inf')
        s[~variable[q_t], :] = -float('inf')
        s[torch.arange(len(q), device=device), q_t] = -float('inf')
        # Tie-aware AP: a whole equal-score group gets the same precision,
        # as in threshold-based average precision. No gene-ID tie advantage.
        sorted_score = torch.sort(s, dim=1, descending=True).values.cpu().numpy()
        score_cpu = s.cpu().numpy()
        variable_cpu = variable.cpu().numpy()
        for j, qi in enumerate(q):
            finite_score = sorted_score[j, np.isfinite(sorted_score[j])]
            inverted = -finite_score
            for annotation, lookup in labels.items():
                partners = lookup.get(int(qi))
                if partners is None:
                    continue
                total = len(partners)
                if not variable_cpu[qi]:
                    ap, recall = 0.0, 0.0
                    accessible = 0
                else:
                    pos_score = score_cpu[j, partners]
                    pos_score = pos_score[np.isfinite(pos_score)]
                    accessible = len(pos_score)
                    ap, recall, positive_seen = 0.0, 0.0, 0
                    unique, counts = np.unique(pos_score, return_counts=True)
                    for value, count in zip(unique[::-1], counts[::-1]):
                        before = int(np.searchsorted(inverted, -value, side='left'))
                        through = int(np.searchsorted(inverted, -value, side='right'))
                        positive_seen += int(count)
                        ap += (float(count) / total) * (positive_seen / through)
                        recall += (float(count) / total) * np.clip((100 - before) / (through - before), 0., 1.)
                result.append({'annotation': annotation, 'query_gene_idx': int(qi),
                               'n_partners': total, 'n_rankable_partners': accessible,
                               'query_rankable': bool(variable_cpu[qi]),
                               'ap': ap, 'recall_at_100': recall})
        del s, sorted_score, score_cpu
    return result
def control_lookup(labels: dict, mc_counts: np.ndarray, sc_counts: np.ndarray,
                   n_genes: int):
    """Five shared negatives/positive, nearest in two-arm marginal frequency.

    A tiny third KD-tree coordinate resolves exact frequency ties by gene ID.
    Known partners in either reference are excluded. Controls may be reused
    across different positives but are distinct within each positive's five.
    """
    point = np.column_stack((mc_counts / N_PROFILES, sc_counts / N_PROFILES,
                             np.arange(n_genes, dtype=np.float64) * 1e-10))
    tree = cKDTree(point)
    lookup = {}
    imbalance = []
    unique_controls = []
    query_list = sorted(set(labels['HuRI']) | set(labels['CORUM']))
    for qi in query_list:
        excluded = set(labels['HuRI'].get(qi, ())) | set(labels['CORUM'].get(qi, ())) | {qi}
        for annotation, partners in labels.items():
            pos = partners.get(qi)
            if pos is None:
                continue
            chosen = np.empty((len(pos), N_CONTROLS), dtype=np.int32)
            for j, gene in enumerate(pos):
                k = 32
                while True:
                    # Query coordinate 3=0 ensures deterministic lower-ID tie ordering.
                    target = point[int(gene)].copy()
                    target[2] = 0.
                    _, nearest = tree.query(target, k=k)
                    selected = [int(v) for v in np.atleast_1d(nearest) if int(v) not in excluded]
                    if len(selected) >= N_CONTROLS:
                        chosen[j] = selected[:N_CONTROLS]
                        break
                    if k >= n_genes:
                        raise RuntimeError(f'Insufficient controls: query={qi}')
                    k = min(k * 2, n_genes)
            lookup[(annotation, qi)] = (pos, chosen)
            ref = point[pos, :2]
            ctrl = point[chosen, :2].mean(axis=1)
            imbalance.append(np.abs(ref - ctrl))
            unique_controls.append(len(np.unique(chosen)) / chosen.size)
    errors = np.concatenate(imbalance)
    return lookup, {'n_control_sets': int(sum(len(x) for x in imbalance)),
                    'median_abs_frequency_error_mc': float(np.median(errors[:, 0])),
                    'p95_abs_frequency_error_mc': float(np.quantile(errors[:, 0], .95)),
                    'median_abs_frequency_error_sc': float(np.median(errors[:, 1])),
                    'p95_abs_frequency_error_sc': float(np.quantile(errors[:, 1], .95)),
                    'mean_distinct_control_fraction_per_query': float(np.mean(unique_controls)),
                    'n_controls_per_partner': N_CONTROLS}

def coinput_scores(presence: sp.csr_matrix, controls: dict,
                   device: torch.device, arm: str, seed: int, source: str):
    """P(partner|query) minus frequency-matched P(control|query)."""
    count = np.asarray(presence.sum(axis=0)).ravel().astype(np.int32)
    x = torch.from_numpy(presence.toarray().astype(np.float32, copy=False)).to(device)
    query_list = np.array(sorted({key[1] for key in controls}), dtype=np.int32)
    result = []
    for start in range(0, len(query_list), 128):
        q = query_list[start:start+128]
        joint = (x[:, torch.as_tensor(q, device=device)].T @ x).cpu().numpy()
        for j, qi in enumerate(q):
            nq = int(count[qi])
            for ann in ('HuRI', 'CORUM'):
                pair = controls.get((ann, int(qi)))
                if pair is None:
                    continue
                pos, neg = pair
                if nq:
                    positive = joint[j, pos] / nq
                    negative = joint[j, neg].mean(axis=1) / nq
                    mean_pos = float(positive.mean())
                    mean_neg = float(negative.mean())
                    excess = mean_pos - mean_neg
                else:
                    mean_pos = mean_neg = excess = np.nan
                result.append({'dataset_id': source, 'seed': seed, 'arm': arm,
                               'annotation': ann, 'query_gene_idx': int(qi),
                               'n_query_present': nq, 'n_partners': len(pos),
                               'partner_conditional_rate': mean_pos,
                               'control_conditional_rate': mean_neg,
                               'matched_excess': excess})
    return pd.DataFrame(result)
def normalize_log(x: sparse.csr_matrix) -> sparse.csr_matrix:
    x = x.copy().tocsr().astype(np.float32)
    if x.nnz and (np.min(x.data) < 0 or not np.allclose(x.data, np.rint(x.data), atol=1e-4)):
        raise ValueError("Expected nonnegative integer-valued raw counts")
    totals = np.asarray(x.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("At least one selected profile has zero protein-coding count")
    x = sparse.diags((1e4 / totals).astype(np.float32)) @ x
    x = x.tocsr()
    np.log1p(x.data, out=x.data)
    return x

def pca_view(x: sparse.csr_matrix, hvgs: np.ndarray) -> np.ndarray:
    z = x[:, hvgs].toarray()
    z = StandardScaler(copy=False).fit_transform(z)
    np.clip(z, -10, 10, out=z)
    n_components = min(50, z.shape[0] - 1, z.shape[1])
    if n_components < 2:
        raise ValueError("Too few samples/features for PCA")
    return PCA(n_components=n_components, svd_solver="randomized", random_state=42).fit_transform(z).astype(np.float32)
def context_weights(groups, labels, n_groups, n_types):
    valid = labels >= 0
    return sparse.csr_matrix((np.ones(valid.sum(), dtype='float64'),
        (groups[valid], labels[valid])), shape=(n_groups, n_types))

def context_rates(weights, detected, tokens):
    """SC-anchored probabilities: MC indicator weighted by constituent SC count."""
    totals = np.asarray(weights.sum(0)).ravel()
    d = np.asarray((weights.T @ detected).toarray())
    t = np.asarray((weights.T @ tokens).toarray())
    assert np.all(t <= d + 1e-9)
    return d, t, totals

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('partners');q.add_argument('--matrix',type=Path,required=True);q.add_argument('--labels',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--device',default='cpu')
    q=sub.add_parser('cooccurrence')
    for name in ['mc','sc','labels','output']:q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--dataset',required=True);q.add_argument('--seed',type=int,required=True);q.add_argument('--device',default='cpu')
    q=sub.add_parser('asw')
    for name in ['mc','sc','labels','output']:q.add_argument('--'+name,type=Path,required=True)
    q=sub.add_parser('markers')
    for name in ['weights','detected','tokens','output']:q.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.command in ['partners','cooccurrence']:
        raw=json.loads(a.labels.read_text());labels={ann:{int(k):np.array(v,dtype=np.int32) for k,v in values.items()} for ann,values in raw['partners'].items()}
        if a.command=='partners':write_table(pd.DataFrame(score_matrix(sp.load_npz(a.matrix),labels,torch.device(a.device))),a.output)
        else:
            mc=sp.load_npz(a.mc);sc=sp.load_npz(a.sc)
            if mc.shape!=sc.shape or mc.shape!=(1200,len(raw['genes'])):raise ValueError('Expected paired 1200-profile matrices in fixed vocabulary order')
            if not np.isin(mc.data,[0,1]).all() or not np.isin(sc.data,[0,1]).all():raise ValueError('Token presence must be binary')
            controls,audit=control_lookup(labels,np.asarray(mc.sum(0)).ravel(),np.asarray(sc.sum(0)).ravel(),mc.shape[1])
            rows=[coinput_scores(x,controls,torch.device(a.device),arm,a.seed,a.dataset) for arm,x in [('Metacell',mc),('Single cell',sc)]]
            write_table(pd.concat(rows),a.output);write_json(audit,a.output.with_suffix('.json'))
    elif a.command=='asw':
        import scanpy as sc
        import anndata as ad
        import scib_metrics
        mc=normalize_log(sp.load_npz(a.mc));archive=np.load(a.sc)
        # Six SC arrays are already matched to the same retained groups and labels.
        single={seed:normalize_log(sp.csr_matrix(archive[f'seed{seed}'])) for seed in range(42,48)}
        labels=np.load(a.labels);joint=ad.AnnData(sp.vstack([mc,single[42]],format='csr'))
        sc.pp.highly_variable_genes(joint,flavor='seurat',n_top_genes=min(2000,joint.n_vars));hvg=np.flatnonzero(joint.var.highly_variable)
        rows=[]
        for seed,x in [('metacell',mc),*single.items()]:
            if len(labels)!=x.shape[0]:raise ValueError('ASW labels and profiles differ')
            rows.append({'arm':'Metacell' if seed=='metacell' else 'Single cell','seed':seed,'asw_label':float(scib_metrics.silhouette_label(pca_view(x,hvg),labels,rescale=True))})
        write_table(pd.DataFrame(rows),a.output)
    else:
        d,t,n=context_rates(sp.load_npz(a.weights),sp.load_npz(a.detected),sp.load_npz(a.tokens))
        a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,detected=d,tokens=t,constituent_cells=n)
if __name__=='__main__':main()
