"""Scientific invariants that would change reported results if violated."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import average_precision_score
from benchmarks.tasks.robustness import gaussian,fit_clean,evaluate_noisy
from benchmarks.shared._common import embeddings
from benchmarks.tasks.pc9 import evaluate_comparisons
from benchmarks.tasks.huri import average_precision
from benchmarks.tasks.perturbation_retrieval import make_delta_items,evaluate_retrieval
from benchmarks.analysis.perturbation_biology import gene_similarity
from benchmarks.shared._retrieval_metrics import metrics_from_relevance


def test_gaussian_population_variance_zeros_and_float_counts():
    raw=sparse.csr_matrix(np.zeros((5000,3)))
    # A positive center avoids clipping and distinguishes variance r*v from (r*v)^2.
    raw=raw+sparse.csr_matrix(np.full((5000,3),100.))
    noise=gaussian(raw,np.array([4.,0.,9.]),.5,np.random.default_rng(42)).toarray()-100
    np.testing.assert_allclose(noise.var(0),[2,0,4.5],rtol=.06)
    z=gaussian(sparse.csr_matrix((1000,2)),np.ones(2),1,np.random.default_rng(42)).toarray()
    assert (z==0).any() and (z>0).any() and (z!=np.rint(z)).any()
    assert z.dtype==np.float32 and z.min()==0

def test_scaler_only_fits_training_and_abstentions_stay_in_denominator():
    x=np.array([[-2.,0],[-1,0],[1,0],[2,0]],np.float32);y=np.array(['a','a','b','b'])
    scaler,probes,clean=fit_clean(x,y,x,y,seeds=[1])
    np.testing.assert_array_equal(scaler.mean_,[0,0])
    rows,_=evaluate_noisy(x,np.array([True,False,True,True]),y,scaler,probes,clean)
    assert rows[0]['coverage']==.75 and rows[0]['macro_f1']<clean[1]
    np.testing.assert_array_equal(scaler.mean_,[0,0])

def test_embedding_identity_and_duplicate_rejection(tmp_path):
    p=tmp_path/'x.npz';np.savez(p,embeddings=np.ones((2,3)),cell_ids=['a','b'])
    with pytest.raises(ValueError):embeddings(p,['b','a'])
    np.savez(p,embeddings=np.ones((2,3)),cell_ids=['a','a'])
    with pytest.raises(ValueError):embeddings(p)

def test_pc9_uses_raw_positive_logit_and_shared_controls():
    logits=np.array([[100.,2.],[0.,1.],[0.,0.]])
    rows=pd.DataFrame({'cell_id':['p','c','p','d'],'label':[1,0,1,0],'comparison_id':['first','first','second','second']})
    out=evaluate_comparisons(logits,['p','c','d'],rows)
    np.testing.assert_array_equal(out.AP,[1.,1.])
    # Softmax would reverse p and c; the published protocol ranks raw logit 1.
    assert average_precision_score([1,0],[-98.,1.])==.5

def test_threshold_ap_and_stable_rank_ap_are_distinct():
    y=np.array([1,0,1,0]);s=np.ones(4)
    assert average_precision(y,s)==average_precision_score(y,s)==.5
    rank=metrics_from_relevance(y[None,:].astype(bool))['map'][0]
    assert rank==pytest.approx((1+2/3)/2) and rank!=.5

def test_functional_retrieval_excludes_same_target_and_batch():
    # Same-target twins are nearest but must be excluded from functional retrieval.
    x=np.array([[1,0],[1,0],[0,1],[0,1]],np.float32)
    meta=pd.DataFrame({'item_id':['a0','a1','b0','b1'],'target_gene':['a','a','b','b'],'label':['group']*4,'group':['0','1','0','1'],'n_cells':[3]*4})
    out=evaluate_retrieval(x,meta,'functional','cpu')
    assert out.n_reference_items.eq(1).all() and out.n_positive_reference_items.eq(1).all()
    assert out['map'].eq(1).all()

def test_gene_graph_excludes_same_batch_before_averaging():
    vectors=np.array([[1,0],[0,1],[1,0],[0,1]],np.float32)
    items=pd.DataFrame({'target_gene':['a','a','b','b'],'batch_group':['0','1','0','1']})
    scores,_=gene_similarity(vectors,items,np.array(['a','b']),device='cpu')
    assert scores[0,1]==0 and scores[1,0]==0 and np.isneginf(scores.diagonal()).all()


def test_pc9_external_labels_cannot_change_training_or_predictions():
    from types import SimpleNamespace
    import torch
    from benchmarks.shared._drug_head import train_mlp
    torch.set_num_threads(2)
    rng=np.random.default_rng(1);x=rng.normal(size=(20,5)).astype(np.float32);y=np.arange(20)%2
    args=SimpleNamespace(hidden_size=512,device='cpu',lr=1e-4,epochs=10,batch_size_train=128,batch_size_test=256)
    first=train_mlp(x,y,x[:6],np.zeros(6,dtype=np.int64),args,1)['logits']
    second=train_mlp(x,y,x[:6],np.ones(6,dtype=np.int64),args,1)['logits']
    np.testing.assert_array_equal(first,second)
