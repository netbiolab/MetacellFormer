"""Validate and aggregate saved Gaussian and PC9 result tables.

Usage: python -m benchmarks.analysis.aggregate_additional_analyses --root RESULT_DIRECTORY
       [--output-dir OUTPUT_DIRECTORY] [--figure5-gaussian]
"""
from pathlib import Path
import argparse, json
from itertools import product
import numpy as np
import pandas as pd
from metacellformer.names import read_table, model_directory

MAIN_MODELS = ['MetacellFormer', 'Geneformer-V2', 'scPRINT', 'scGPT', 'Geneformer-V1', 'CellFM']
DATASETS = ['Cell_Lines', 'DC', 'HumanPBMC', 'Immune', 'Liver_cross', 'MCA', 'MHSP', 'PBMC_368K', 'Pancrm']

def summarize_gaussian(scores,clean,models=None):
    if models is None:models=MAIN_MODELS+['MetacellFormer-SC']
    metrics=['macro_f1','clean_macro_f1','retention','f1_loss','coverage']
    ds=scores.groupby(['model','dataset','condition','level'],as_index=False)[metrics].mean()
    ds['n_test']=ds.set_index(['model','dataset']).index.map(scores.groupby(['model','dataset']).n_test.first())
    ds['n_repeats']=15
    anchors=clean.groupby(['model','dataset'],as_index=False).agg(macro_f1=('macro_f1','mean'),n_test=('n_test','first'))
    anchors['clean_macro_f1']=anchors.macro_f1;anchors['retention']=1.;anchors['f1_loss']=0.;anchors['coverage']=1.;anchors['condition']='clean';anchors['level']=0.;anchors['n_repeats']=3
    ds=pd.concat([ds,anchors],ignore_index=True).sort_values(['model','dataset','level'])
    idx=np.random.default_rng(2402).integers(0,9,(10000,9)); rows=[];paired=[];arrays={}
    for model in models:
        for r in [0.,.1,.5,.8,1.]:
            part=ds[ds.model.eq(model)&ds.level.eq(r)].set_index('dataset').loc[DATASETS]
            for metric in ['retention','macro_f1']:
                v=part[metric].to_numpy();arrays[model,r,metric]=v
                lo,hi=np.quantile(v[idx].mean(1),[.025,.975])
                rows.append([model,r,metric,v.mean(),lo,hi,9])
    for comparator in models[1:]:
        for r in [.1,.5,.8,1.]:
            for metric in ['retention','macro_f1']:
                a=arrays['MetacellFormer',r,metric];b=arrays[comparator,r,metric];d=a-b
                lo,hi=np.quantile(d[idx].mean(1),[.025,.975])
                paired.append([comparator,r,metric,a.mean(),b.mean(),d.mean(),lo,hi,int((d>0).sum()),9])
    return ds,pd.DataFrame(rows,columns=['model','r','metric','mean','ci_low','ci_high','n_datasets']),pd.DataFrame(paired,columns=['comparator','r','metric','mc_mean','comparator_mean','difference','ci_low','ci_high','mc_higher','n_datasets'])

def require(condition, message):
    if not condition: raise ValueError(message)

def validate_grid(frame, columns, expected, name):
    actual=set(map(tuple,frame[columns].itertuples(index=False,name=None)))
    require(len(frame)==len(expected) and actual==expected and not frame.duplicated(columns).any(),
            f"{name}: incomplete, unexpected or duplicated evaluation units")

def compare(actual, saved, keys, columns, name):
    require(not saved.duplicated(keys).any(), f"{name}: duplicate saved keys")
    merged=actual.merge(saved,on=keys,suffixes=("","_saved"),validate="one_to_one")
    require(len(merged)==len(actual)==len(saved), f"{name}: saved keys differ")
    for c in columns:
        require(np.allclose(merged[c],merged[c+"_saved"],rtol=0,atol=2e-12), f"{name}: {c} differs")

def gaussian_inputs(root,models):
    g=root/"tasks/cell_type_annotation_robustness"
    read=lambda p:read_table(p,sep="\t")
    scores=read(g/"all_scores.tsv");clean=read(g/"clean_scores.tsv")
    allowed=set(MAIN_MODELS+["MetacellFormer-SC"])
    require(set(scores.model).issubset(allowed) and set(clean.model).issubset(allowed),"Unknown Gaussian model")
    scores=scores[scores.model.isin(models)].copy();clean=clean[clean.model.isin(models)].copy()
    keys=["model","dataset","level","draw_seed","probe_seed"]
    expected=set(product(models,DATASETS,[.1,.5,.8,1.],range(42,47),range(1,4)))
    validate_grid(scores,keys,expected,"Gaussian noisy scores")
    validate_grid(clean,["model","dataset","probe_seed"],set(product(models,DATASETS,range(1,4))),"Gaussian clean scores")
    cols=["macro_f1","clean_macro_f1","retention","coverage"]
    require(np.isfinite(scores[cols].to_numpy()).all() and scores.coverage.eq(1).all(),"Gaussian nonfinite/incomplete scores")
    require(np.isfinite(clean.macro_f1).all() and (clean.macro_f1>0).all(),"Invalid clean macro-F1")
    require(np.allclose(scores.retention,scores.macro_f1/scores.clean_macro_f1,rtol=0,atol=2e-12),"Retention formula mismatch")
    aligned=scores.merge(clean[["model","dataset","probe_seed","macro_f1"]],on=["model","dataset","probe_seed"],suffixes=("","_reference"),validate="many_to_one")
    require(np.allclose(aligned.clean_macro_f1,aligned.macro_f1_reference,rtol=0,atol=2e-12),"Unmatched clean probe baseline")
    expected_conditions={.1:"gaussian_0.1",.5:"gaussian_0.5",.8:"gaussian_0.8",1.:"gaussian_1"}
    require(scores.condition.eq(scores.level.map(expected_conditions)).all(),"Gaussian condition/variance-ratio mismatch")
    return g,read,scores,clean

def calculate_figure5(root):
    _,_,scores,clean=gaussian_inputs(root,MAIN_MODELS)
    ds,means,_=summarize_gaussian(scores,clean,MAIN_MODELS)
    curves=means.sort_values(['model','r','metric']).reset_index(drop=True)
    r1=ds[ds.level.eq(1.)].sort_values(['model','dataset']).reset_index(drop=True)
    require(len(curves)==60 and len(r1)==54,"Fig. 5 Gaussian output grid differs")
    return {"figure5_gaussian_curves":curves,"figure5_gaussian_r1_datasets":r1}

def calculate(root):
    g,read,scores,clean=gaussian_inputs(root,MAIN_MODELS+["MetacellFormer-SC"])
    ds,means,pairs=summarize_gaussian(scores,clean)
    compare(ds,read(g/"dataset_summary.tsv"),["model","dataset","level"],["macro_f1","clean_macro_f1","retention"],"Gaussian dataset summary")
    compare(means,read(g/"mean_bootstrap_ci.tsv"),["model","r","metric"],["mean","ci_low","ci_high"],"Gaussian mean CI")
    compare(pairs,read(g/"paired_comparisons.tsv"),["comparator","r","metric"],["difference","ci_low","ci_high"],"Gaussian paired CI")
    p=root/"tasks/pc9_cross_study";runs=read(p/"seed_metrics.tsv");comparisons=read(p/"comparisons.tsv")
    require(len(comparisons)==4 and comparisons.comparison_id.nunique()==4 and comparisons.study.nunique()==3,"PC9 four comparisons / three studies expected")
    validate_grid(runs,["comparison_id","model","seed"],set(product(comparisons.comparison_id,MAIN_MODELS,[1,2,3])),"PC9 classifier runs")
    require(np.isfinite(runs.AP).all() and runs.AP.between(0,1).all(),"PC9 AP outside [0,1]")
    counts=runs.merge(comparisons[["comparison_id","n","pos"]],on="comparison_id",validate="many_to_one")
    require(counts.n_test.eq(counts.n).all() and counts.n_positive.eq(counts.pos).all(),"PC9 comparison cell-count mismatch")
    summary=runs.groupby(["comparison_id","model"],as_index=False).AP.mean()
    compare(summary,read(p/"summary.tsv"),["comparison_id","model"],["AP"],"PC9 mean AP")
    return {"gaussian_dataset_summary":ds,"gaussian_mean_bootstrap_ci":means,
            "gaussian_paired_comparisons":pairs,"pc9_mean_ap":summary}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--figure5-gaussian",action="store_true",help="Aggregate the six Fig. 5 Gaussian models without PC9 inputs")
    args=parser.parse_args();tables=(calculate_figure5(args.root.resolve()) if args.figure5_gaussian else calculate(args.root.resolve()))
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for name,d in tables.items():d.to_csv(args.output_dir/(name+".tsv"),sep="\t",index=False,float_format="%.17g")
    report={"status":"PASS","gaussian_noisy_runs":3240 if args.figure5_gaussian else 3780,
            "gaussian_clean_runs":162 if args.figure5_gaussian else 189,
            "output_rows":{k:len(v) for k,v in tables.items()}}
    if not args.figure5_gaussian:report["pc9_seed_runs"]=72
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
