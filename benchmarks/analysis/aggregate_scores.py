"""Aggregate fixed per-unit results for the six paper benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from metacellformer.names import read_table, model_directory


MAIN_MODELS = (
    'MetacellFormer', 'Geneformer-V2', 'scPRINT',
    'scGPT', 'Geneformer-V1', 'CellFM',
)
SC = 'MetacellFormer-SC'
NAMES = {
    'scGPT': 'scGPT', 'Geneformer-V1': 'Geneformer-V1',
    'Geneformer-V2': 'Geneformer-V2',
    'MetacellFormer': 'MetacellFormer', 'MetacellFormer-SC': 'MetacellFormer-SC',
    'scPRINT': 'scPRINT', 'CellFM': 'CellFM',
}
TASKS = (
    'cell_type_annotation', 'drug_response_classification',
    'exact_perturbation_state_retrieval', 'functional_perturbation_retrieval',
    'tissue_specificity_prediction', 'protein_interaction_prediction',
)
TASK_NAMES = {
    'cell_type_annotation': 'Cell-type annotation',
    'drug_response_classification': 'Drug-response classification',
    'exact_perturbation_state_retrieval': 'Exact perturbation-state retrieval',
    'functional_perturbation_retrieval': 'Functional perturbation retrieval',
    'tissue_specificity_prediction': 'Tissue-specificity prediction',
    'protein_interaction_prediction': 'Protein-interaction prediction',
}


def _expect_models(values: pd.Series, expected: int, task: str) -> None:
    counts = values.value_counts()
    if set(counts.index) != set((*MAIN_MODELS, SC)) or not counts.eq(expected).all():
        raise ValueError(f'{task}: unexpected model IDs or record counts: {counts.to_dict()}')


def calculate(root: Path) -> pd.DataFrame:
    units = root / 'tasks'
    rows = []
    cell = read_table(units / 'cell_type_annotation/per_dataset_scores.tsv', sep='\t')
    _expect_models(cell.model_id, 9, 'cell_type_annotation')
    if not all(d.dataset.nunique() == 9 for _, d in cell.groupby('model_id')):
        raise ValueError('Cell dataset roster mismatch')
    for model, value in cell.groupby('model_id').metric_primary.mean().items():
        rows.append((model, 'cell_type_annotation', float(value)))

    drug = read_table(units / 'drug_response_classification/pooled_fold_scores.tsv', sep='\t')
    _expect_models(drug.model_id, 450, 'drug_response_classification')
    if not all(d.groupby(['axis', 'category_value']).size().eq(10).all() and
               d.groupby('axis').category_value.nunique().sum() == 45
               for _, d in drug.groupby('model_id')):
        raise ValueError('Drug 4-axis/45-category/10-fold contract mismatch')
    drug_scores = drug.groupby(['model_id', 'axis', 'category_value']).test_f1.mean() \
        .groupby(['model_id', 'axis']).mean().groupby('model_id').mean()
    for model, value in drug_scores.items():
        rows.append((model, 'drug_response_classification', float(value)))

    retrieval_tasks = (
        ('exact_perturbation_state_retrieval', 59380, 1299),
        ('functional_perturbation_retrieval', 16504, 359),
    )
    for task, n_query, n_target in retrieval_tasks:
        query = read_table(units / task / 'query_results.tsv.gz', sep='\t')
        _expect_models(query.model_id, n_query, task)
        if not all(d.query_target_gene.nunique() == n_target and
                   not d.query_item_id.duplicated().any() for _, d in query.groupby('model_id')):
            raise ValueError(f'{task}: query/target contract mismatch')
        scores = query.groupby(['model_id', 'query_target_gene'])['map'].mean().groupby('model_id').mean()
        for model, value in scores.items():
            rows.append((model, task, float(value)))

    for model in (*MAIN_MODELS, SC):
        gt = read_table(model_directory(units / 'tissue_specificity_prediction', model) / 'fold_metrics.tsv', sep='\t')
        if len(gt) != 150 or gt.unit.nunique() != 30 or not gt.groupby('unit').size().eq(5).all():
            raise ValueError(f'{model}: GTEx 30 tissues x 5 folds mismatch')
        rows.append((model, 'tissue_specificity_prediction', float(gt.groupby('unit').metric_primary.mean().mean())))
        hu = read_table(model_directory(units / 'protein_interaction_prediction', model) / 'fold_metrics.tsv', sep='\t')
        if len(hu) != 20 or hu.fold.nunique() != 20 or hu.n_test_pairs.sum() != 4538:
            raise ValueError(f'{model}: HuRI 20-fold contract mismatch')
        rows.append((model, 'protein_interaction_prediction', float(hu.metric_primary.mean())))

    result = pd.DataFrame(rows, columns=['model_id', 'task_id', 'score'])
    if len(result) != 42 or result.duplicated(['model_id', 'task_id']).any() or not np.isfinite(result.score).all():
        raise ValueError('Expected 7 models x 6 finite task scores')
    return result


def write(root: Path, output_dir: Path | None = None) -> None:
    all_scores = calculate(root)
    main = all_scores[all_scores.model_id.isin(MAIN_MODELS)].copy()
    if len(main) != 36:
        raise ValueError('Expected exactly 6 models x 6 main scores')
    main['rank_within_six'] = main.groupby('task_id').score.rank(ascending=False, method='average')
    main['model_name'] = main.model_id.map(NAMES)
    main['task_name'] = main.task_id.map(TASK_NAMES)
    main['metric_direction'] = 'higher_is_better'
    main = main[['task_id', 'task_name', 'model_id', 'model_name', 'score',
                 'rank_within_six', 'metric_direction']]
    order = {model: i for i, model in enumerate(MAIN_MODELS)}
    task_order = {task: i for i, task in enumerate(TASKS)}
    main = main.sort_values(['task_id', 'model_id'], key=lambda s:
                            s.map(task_order) if s.name == 'task_id' else s.map(order))
    contrast = all_scores[all_scores.model_id.isin(('MetacellFormer', SC))].copy()
    contrast['model_name'] = contrast.model_id.map(NAMES)
    contrast = contrast.sort_values(['task_id', 'model_id'], key=lambda s:
                                    s.map(task_order) if s.name == 'task_id' else s)
    wide = main.pivot(index=['model_id', 'model_name'], columns='task_id', values='score') \
        .reset_index()[['model_id', 'model_name', *TASKS]]
    mean_rank = main.groupby(['model_id', 'model_name'], as_index=False).rank_within_six.mean() \
        .rename(columns={'rank_within_six': 'descriptive_mean_rank'})
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        main.to_csv(output_dir / 'six_model_scores.tsv', sep='\t', index=False, float_format='%.17g')
        wide.to_csv(output_dir / 'six_model_scores_wide.tsv', sep='\t', index=False, float_format='%.17g')
        mean_rank.to_csv(output_dir / 'six_model_descriptive_mean_rank.tsv', sep='\t', index=False, float_format='%.17g')
        contrast.to_csv(output_dir / 'metacellformer_sc_scores.tsv', sep='\t', index=False, float_format='%.17g')
    print(main.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='result directory containing tasks/')
    parser.add_argument('--output-dir', type=Path, help='Optional directory for aggregate TSV files')
    args = parser.parse_args()
    write(args.root.resolve(), args.output_dir)


if __name__ == '__main__':
    main()
