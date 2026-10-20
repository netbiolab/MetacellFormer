"""Canonical paper model names and compatibility with previous result IDs."""
from pathlib import Path
import pandas as pd

ALIASES = {'MetacellFormer (MC E3)': 'MetacellFormer', 'mcfm_mc_e3': 'MetacellFormer', 'mcfm_sc_e3': 'MetacellFormer-SC', 'mcfm_mc_e1': 'MetacellFormer', 'mcfm_mc_e2': 'MetacellFormer', 'mcfm_sc_e1': 'MetacellFormer-SC', 'mcfm_sc_e2': 'MetacellFormer-SC', 'Fixed-E3': 'Frozen-model', 'MC-E3': 'MetacellFormer', 'SC-E3': 'MetacellFormer-SC', 'MC E3': 'MetacellFormer', 'SC E3': 'MetacellFormer-SC', 'Geneformer-V2-104M': 'Geneformer-V2', 'Geneformer v2-104M': 'Geneformer-V2', 'geneformer_v2_104m': 'Geneformer-V2', 'geneformer_v2': 'Geneformer-V2', 'Geneformer v2': 'Geneformer-V2', 'Geneformer-V1-10M': 'Geneformer-V1', 'Geneformer v1-10M': 'Geneformer-V1', 'geneformer_v1_10m': 'Geneformer-V1', 'geneformer_v1': 'Geneformer-V1', 'Geneformer v1': 'Geneformer-V1', 'scPRINT Medium v1.5': 'scPRINT', 'scPRINT-Medium': 'scPRINT', 'scprint_medium_v1_5': 'scPRINT', 'scprint': 'scPRINT', 'scgpt_human_static': 'scGPT', 'scGPT human': 'scGPT', 'scgpt': 'scGPT', 'CellFM-800M CLS': 'CellFM', 'CellFM-800M': 'CellFM', 'cellfm_800m_cls': 'CellFM', 'cellfm': 'CellFM'}

def read_table(*args, **kwargs):
    data = pd.read_csv(*args, **kwargs)
    for column in data:
        if data[column].dtype == object:
            data[column] = data[column].replace(ALIASES)
    return data

def model_directory(root, model):
    root = Path(root)
    if (root / model).is_dir():
        return root / model
    for old, canonical in ALIASES.items():
        if canonical == model and (root / old).is_dir():
            return root / old
    return root / model
