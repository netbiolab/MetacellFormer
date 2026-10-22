# Evaluation

Run from the repository root. `DATA` is the downloaded evaluation dataset; `MODEL_DIRECTORY` is a downloaded MetacellFormer checkpoint. Outputs should go to a new directory. Every command supports `--help`.

| Directory | Purpose |
|---|---|
| `embeddings/` | Native cell and gene embedding extraction; shared scPRINT adapter |
| `tasks/` | Benchmark scoring, robustness and PC9 evaluation |
| `analysis/` | Result summaries and additional figure calculations |
| `shared/` | Internal input/output, classifier and retrieval helpers; not run directly |

## Inputs

| Input | Format |
|---|---|
| Native input | AnnData H5AD, cells × genes, unique cell IDs in `obs_names`; nonnegative archived input in `X`; expression type recorded in the data manifest |
| Cell embeddings | NPZ: float32 `embeddings`, string `cell_ids`, optional boolean `valid`; exact fixed roster order |
| Original embedding cache | NPY with matching `.ids.npy` (or `cell_ids.npy`) and optional `.valid.npy` |
| Static gene embeddings | NPZ: string `genes` (symbols), float32 `embedding` |
| Metadata | Tab-separated; IDs must match embeddings exactly |

Count inputs retain all measured genes. Provide `var.human_ensembl_gene` or `var.ensembl_id` for rank models and `var.gene_symbol` for symbol models. Resolve ambiguous mappings using the recorded snapshots. The encoder command strips phenotype metadata. The CellFM/scGPT input path sums repeated gene symbols.

The evaluation repository's `input_manifest.tsv` lists included matrices and external-only inputs. Released cell-annotation files have frozen `var.metacellformer_gene_id` and `var.geneformer_ensembl_id` mappings, which the native adapters use directly. MetacellFormer retains historical alias features separately; Geneformer selects its original mapped feature subset. Prepare Gaussian inputs before model-specific feature selection. DC's upstream benchmark describes the archived input as TPM.

```bash
python -m benchmarks.embeddings.extract_embeddings --input counts.h5ad --output embeddings.npz --config runtime.json --model MetacellFormer --task cell --device cuda:0
python -m benchmarks.embeddings.extract_gene_embeddings --config runtime.json --model MetacellFormer --gene-info "$DATA/contracts/protein_interaction_prediction/Homo_sapiens.gene_info.gz" --output genes.npz
```

Choose the actual task (`cell`, `drug`, `robustness`, `pc9`, `perturbation`). Geneformer Gaussian/PC9/perturbation uses official `EmbExtractor(emb_layer=-1)` (penultimate layer); Gaussian runs in float32. Standard cell annotation uses the native final encoder layer. scPRINT also requires the recorded `--sampling-key`; use `cell/DATASET` for every Gaussian condition so sampling identity stays fixed. Do not infer seeds from display names.

## Six main tasks

```bash
python -m benchmarks.tasks.cell_annotation --train-embedding train.npz --test-embedding test.npz --roster "$DATA/contracts/cell_type_annotation/DC_roster.json" --model MetacellFormer --dataset DC --output-dir results/cell/DC
python -m benchmarks.tasks.drug_classification --embedding drug.npz --metadata "$DATA/contracts/drug_response_classification/cells.tsv.gz" --folds "$DATA/contracts/drug_response_classification/folds.npz" --model MetacellFormer --device cuda:0 --output-dir results/drug
python -m benchmarks.tasks.perturbation_retrieval --embedding perturbation.npz --metadata "$DATA/contracts/perturbation_retrieval/selected_cells.tsv" --annotations "$DATA/contracts/perturbation_retrieval/functional_annotations.tsv" --model MetacellFormer --output-dir results/retrieval
python -m benchmarks.tasks.gtex --data-root "$DATA" --embedding genes.npz --model MetacellFormer --output-dir results/gtex
python -m benchmarks.tasks.huri --data-root "$DATA" --embedding genes.npz --model MetacellFormer --output-dir results/huri
```

The retrieval command produces both exact-state and functional tasks. Functional retrieval excludes same-target and same-batch candidates. Drug classification uses 45 categories across four axes, ten folds, final epoch 10, and the original seed-24 training/evaluation shuffle sequence. GTEx requires CUDA and preserves its original PCA/L-BFGS seed identity. HuRI uses the fixed **20-fold** node split and 4,538 eligible held-out pairs; its AP is threshold/tie aware. Retrieval AP uses stable rank order.

## Gaussian robustness

```bash
python -m benchmarks.tasks.robustness prepare --input "$DATA/inputs/cell_annotation/DC.h5ad" --roster "$DATA/contracts/cell_type_annotation/DC_roster.json" --dataset DC --output-dir noise/DC
python -m benchmarks.embeddings.extract_embeddings --input noise/DC/train_clean.h5ad --output embeddings/DC/train_clean.npz --config runtime.json --model MetacellFormer --task robustness --device cuda:0
python -m benchmarks.tasks.robustness score --roster "$DATA/contracts/cell_type_annotation/DC_roster.json" --embedding-dir embeddings/DC --model MetacellFormer --dataset DC --output-dir results/robustness/DC
```

Before `score`, encode `clean.h5ad` and every `gaussian_R_sSEED.h5ad` using matching output stems. Repeat for the nine dataset rosters and six main models; include MetacellFormer-SC for the Fig. 3 comparison. Aggregate Fig. 5b,c with `python -m benchmarks.analysis.aggregate_additional_analyses --root results --output-dir figure5_source --figure5-gaussian`. Ratios are 0.1, 0.5, 0.8, 1; draw seeds are 42–46; probe seeds are 1–3. Noise variance is `ratio × clean-test gene variance` (population variance, including zeros). Negative values are clipped; counts are neither rounded nor renormalized. The scaler and classifier fit clean training data once per probe and are reused for every noisy condition. Failed embeddings count as abstentions. Retention divides by that probe's matched clean score.

## PC9 external evaluation

```bash
python -m benchmarks.embeddings.extract_embeddings --input "$DATA/inputs/pc9/GSE303411.h5ad" --output GSE303411_embeddings.npz --config runtime.json --model MetacellFormer --task pc9 --device cuda:0
python -m benchmarks.tasks.pc9 --train-embedding pc9_train.npz --test-embedding pc9_test.npz --train-metadata "$DATA/contracts/pc9/train_cells.tsv.gz" --test-metadata "$DATA/contracts/pc9/test_cells.tsv.gz" --comparisons "$DATA/contracts/pc9/comparisons.tsv.gz" --model MetacellFormer --device cuda:0 --output-dir results/pc9
```

The released study-specific H5AD files are the post-QC historical inputs. Repeat extraction for GSE282074 and GSE196018, retaining each study's gene axis, then join embeddings in the exact order of `contracts/pc9/test_cells.tsv.gz` before scoring. GSE282074 retains decimal RSEM expected counts. QC is mapped counts ≥1,000, detected genes ≥200 and mitochondrial fraction ≤0.20.

For source-level reconstruction, `scripts/prepare_pc9.py` accepts GEO files for GSM9125465–GSM9125466 (ZIP matrices), GSM8635332–GSM8635336 (`*_lRNA_*.gz`), and GSM5857972–GSM5857973 (`*_PC9_WT_*_counts_matrix.csv.gz`). This helper creates a combined gene union; align each study back to its released feature axis before native inference. The published study-specific files avoid that extra step.

Prepare and encode the 2,393 GSE149214 training cells from authorized scDrugMap inputs, aligned to `train_cells.tsv.gz`; encode external cells with `--task pc9`. The two-hidden-layer MLP uses 512 units per layer, two logits, BCE, AdamW, ten epochs and seeds 1–3. No scaler, PCA or external tuning is used. AP ranks the raw positive-class logit. GSE282074 controls occur in both drug comparisons. CellFM inference uses the official MindSpore encoder and its native input preparation. Preserve study-specific extraction settings before joining outputs in roster order.

## Paper analysis code

Figure 2 and Figure 6 calculations are retained in `analysis/`. Supply the study-specific inputs below when running these analyses.

| Script | Required inputs and calculation |
|---|---|
| `corpus_summary.py counts` | Corpus `source_manifest.tsv` → counts and compression (`--split train` by default) |
| `corpus_summary.py cost` | Timing TSV → preprocessing and training times |
| `corpus_summary.py sparsity` | Count H5AD and vocabulary → zero fraction |
| `corpus_summary.py umap` | Training embeddings → PCA50 and cosine UMAP, seed 42 |
| `metacell_properties.py partners` | Matched raw-log/token matrices and partner annotation JSON → HuRI/CORUM partner AP |
| `metacell_properties.py cooccurrence` | Paired binary token matrices and partner annotations → conditional co-occurrence |
| `metacell_properties.py asw` | Matched MC/SC counts and labels → shared HVGs, PCA and ASW |
| `metacell_properties.py markers` | Constituent weights and detection/token indicators → weighted counts |
| `perturbation_biology.py within-screen` | Full-gene K562/RPE1 embeddings and metadata → within-screen retrieval |
| `perturbation_biology.py cross-screen` | Common-gene embeddings and cross-screen contracts → bidirectional retrieval |
| `perturbation_biology.py corum` | MC/SC embeddings, cell metadata, batch assignments and CORUM annotations → discovery selection and held-out AP |
| `perturbation_biology.py umap` | K562 target-vector NPZ → cosine UMAP |

Figure 2 uses matched profiles in fixed gene order; ASW uses the separate `scib-metrics==0.6.1` environment. For Figure 6, `scripts/prepare_cross_screen.py` takes supplied K562/RPE1 matrices and cross-screen contracts, restricting to 7,226 common measured genes before normalization. Supply the seed-42 caches with `--k562-seed42` and `--rpe1-seed42` alongside the ten-seed union embeddings. CORUM selection uses discovery batches and scores disjoint validation batches. Each command's `--help` lists its file arguments.

## Summarizing result tables

`aggregate_scores.py`, `aggregate_additional_analyses.py` and `paper_statistics.py` read result directories with the `tasks/` layout specified in their source. Set `--root` to that result directory. Evaluation inputs are provided separately in MetacellFormer-evaluation. Figure Source Data and Supplementary Tables accompany the article.
