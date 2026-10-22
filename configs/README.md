# Environments and resources

Use separate environments for external model encoders; do not install every model into the evaluation environment.

| Component | Recorded configuration |
|---|---|
| MetacellFormer / MetacellFormer-SC | Python 3.11, torch 2.4.0, transformers 4.44.2, safetensors 0.4.4; released checkpoint and training resources |
| Geneformer-V2 / Geneformer-V1 | Official Geneformer tokenizer and matching token dictionary |
| scPRINT | Medium checkpoint v1.5, scPRINT code 2.3.8 at `e7c60766cbff58bb7280a1c8fe4b7461b1af1911`, scDataLoader v1.6.4; four official class tokens |
| scGPT | Native human checkpoint, 51-bin input, maximum length 1,200, seed 123 |
| CellFM | Native 800M checkpoint and matching gene-info order; MindSpore native environment |
| ASW | Original isolated Python 3.12 / scib-metrics 0.6.1 environment |

`runtime.example.json` contains paths to replace, not bundled third-party weights. scPRINT's original environment used lazy imports for optional scDataLoader tasks and scPRINT model-utils tasks; strict tensor loading and the checkpoint-key rename are retained in `_scprint.py`. Install those optional upstream dependencies or retain those import-only compatibility changes.

`gtex_seed_identifiers.json` preserves historical run identifiers because they are hashed into PCA random seeds. They are not model display names. `pretraining.json` records the paper's training schedule. Library/checkpoint versions belong here and in Methods; figure and result model labels use the short names.
