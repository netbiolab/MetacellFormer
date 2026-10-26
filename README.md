# MetacellFormer

MetacellFormer learns gene and cell representations from metacells. MetacellFormer-SC is the matched single-cell-pretrained control.

## Installation

Use Python 3.11 on Linux or WSL. Run commands from the repository root.

```bash
python3.11 -m venv MetacellFormer-env
source MetacellFormer-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs/evaluation-requirements.txt '.[evaluation,test]'
python -m pip check
python -m pytest -q tests
```

This setup uses CPU PyTorch. Install a CUDA build of PyTorch for GPU inference and training. Environment and external encoder settings are in [configs/README.md](configs/README.md).

## Data and models

| Repository | Contents |
|---|---|
| [MetacellFormer](https://huggingface.co/netbiolab/MetacellFormer) | Metacell-pretrained checkpoint, vocabulary and reference statistics |
| [MetacellFormer-SC](https://huggingface.co/netbiolab/MetacellFormer-SC) | Matched single-cell-pretrained checkpoint and resources |
| [MetacellFormer-corpus](https://huggingface.co/datasets/netbiolab/MetacellFormer-corpus) | 818,609 metacell count profiles, source-cell memberships and fixed splits from 1,334 datasets |
| [MetacellFormer-evaluation](https://huggingface.co/datasets/netbiolab/MetacellFormer-evaluation) | Inputs, cell rosters, labels and folds for six main benchmarks, Gaussian robustness and PC9 |

Download the files required by your notebook and set their local paths:

```bash
export METACELLFORMER_EVALUATION=/path/to/MetacellFormer-evaluation
export METACELLFORMER_MODEL=/path/to/MetacellFormer
```

Source access, expression types and terms are recorded in the dataset manifests. The corpus README explains reconstruction from the original single-cell providers.

## Examples

Start Jupyter from this repository root. Each notebook writes to `example_outputs/`.

| Notebook | Environment | Example |
|---|---|---|
| [01_metaq.ipynb](examples/01_metaq.ipynb) | MetaQ-env | Group 100 K562 cells and aggregate raw counts |
| [02_embeddings.ipynb](examples/02_embeddings.ipynb) | MetacellFormer-env | Extract embeddings for 96 fixed human PBMC_368K cells |
| [03_evaluation.ipynb](examples/03_evaluation.ipynb) | MetacellFormer-env | Score PBMC_368K annotation and one Gaussian noise condition |
| [04_perturbation_retrieval.ipynb](examples/04_perturbation_retrieval.ipynb) | MetacellFormer-env | Retrieve K562 perturbation states and functional groups |

Each notebook lists its minimal downloads, runtime, output files and completion checks, and includes a small executed result. To use the pretrained model, start at notebook 02 without installing MetaQ. Run notebook 03 after 02; notebook 04 is independent.

Cell-based downstream examples feed individual single cells to the encoder. Notebook 04 aggregates cell embeddings by perturbation and batch after inference. To register a notebook kernel, activate its environment and run:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name MetacellFormer-env --display-name "Python (MetacellFormer-env)"
```

For the MetaQ environment below, use `MetaQ-env` as the kernel name and display name.

## Evaluation

Copy [configs/runtime.example.json](configs/runtime.example.json) to `runtime.json` and set the checkpoint paths for the models you will run. Follow [benchmarks/README.md](benchmarks/README.md) for embedding extraction, all six main tasks, Gaussian robustness and PC9. Use `metacellformer.load_encoder` to load either MetacellFormer checkpoint.

The main comparison uses MetacellFormer, Geneformer-V2, Geneformer-V1, scPRINT, scGPT and CellFM; MetacellFormer-SC is the matched control. Published results are in the article's Figure Source Data and Supplementary Tables.

## Metacells and pretraining

Install [MetaQ](https://github.com/XLearning-SCU/MetaQ) in a separate Python 3.10 environment:

```bash
python3.10 -m venv MetaQ-env
source MetaQ-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r configs/metaq-requirements.txt
METAQ_DIR=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("MetaQ_sc").submodule_search_locations)))')
patch --dry-run -d "$METAQ_DIR" -p1 < patches/metaq.patch
patch -d "$METAQ_DIR" -p1 < patches/metaq.patch
python -m pip check
python tests/check_metaq.py
```

Apply the patch once to a clean installation. It preserves CP10K/log1p preprocessing, HVG selection and the original training loop, and exports memberships for raw-count aggregation. MetaQ uses seed 1, 300 epochs, Random initialization and `max(1, floor(n_cells/50))` groups. Inputs with fewer than 50 cells receive one group directly. For the 100-cell example, use `--device cpu --batch-size 16`.

```bash
python scripts/prepare_metaq.py --input counts.h5ad --output-dir metaq --device cpu
python scripts/aggregate_metacells.py counts.h5ad metaq/memberships.tsv metacells.h5ad --source-id SOURCE
# After aggregation, deactivate MetaQ and activate the main environment.
deactivate
source MetacellFormer-env/bin/activate
python -m pip install -c configs/evaluation-requirements.txt '.[training]'
python scripts/tokenize_training.py --input metacells.h5ad --model-dir MODEL_DIRECTORY --split split.tsv --dataset-id SOURCE --dataset-idx 0 --output-dir tokens
python scripts/build_training_shards.py --token_index_csv tokens/token_index.csv --out_train_dir train --out_eval_dir eval --out_prefix metacells --max_input_size 1024
python scripts/train_paper.py --help
```

Aggregation sums raw counts across all measured genes. Use `--counts-layer` when counts are stored in an AnnData layer. The split TSV contains `profile_id` and `split` (`train` or `eval`) in input row order. Combine source token-index CSVs before building corpus shards. The model repository supplies reference medians and fixed 64-bin edges; [configs/pretraining.json](configs/pretraining.json) specifies the training schedule. `train_paper.py --dry-run` prints launch commands.

## Repository layout

| Directory | Contents |
|---|---|
| `src/metacellformer/` | Encoder, tokenizer, value bins and training runtime |
| `scripts/` | MetaQ, aggregation and data preparation |
| `benchmarks/` | Embedding extraction, evaluation and paper analysis code |
| `configs/` | Environments, checkpoints and training schedules |
| `examples/`, `tests/` | Workflow notebooks and consistency checks |

[GitHub Actions](.github/workflows/tests.yml) runs the tests and a separate MetaQ check on pushes and pull requests.

## License

Code: [Apache License 2.0](LICENSE). The MetaQ patch retains its [MIT notice](patches/MetaQ-LICENSE.txt). Model weights: CC BY 4.0. Datasets and upstream model code retain their source terms.
