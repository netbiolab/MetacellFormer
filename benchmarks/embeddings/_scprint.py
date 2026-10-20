"""Native scPRINT input and forward helpers used in the paper.
Used by the cell and gene embedding commands; requires the original scPRINT code.
"""
from pathlib import Path
import gzip,hashlib,json,sys
import numpy as np
import torch
from scipy import sparse
MAX_LEN=2000
PRED_EMBEDDING=['cell_type_ontology_term_id','disease_ontology_term_id','self_reported_ethnicity_ontology_term_id','sex_ontology_term_id']
CHECKPOINT=GENE_INFO=REPO=SCDL_REPO=None

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')

def checkpoint() -> dict:
    return torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

def ncbi_maps(valid_ensembl: set[str]):
    exact: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    ens_to_symbol: dict[str, str] = {}
    with gzip.open(GENE_INFO, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: header.index(name) for name in ["Symbol", "Synonyms", "dbXrefs"]}
        for line in f:
            row = line.rstrip("\n").split("\t")
            ens = [
                x.split(":", 1)[1].split(".", 1)[0]
                for x in row[col["dbXrefs"]].split("|")
                if x.startswith("Ensembl:ENSG")
            ]
            ens = [x for x in ens if x in valid_ensembl]
            if not ens:
                continue
            symbol = row[col["Symbol"]].strip()
            if symbol and symbol != "-":
                exact.setdefault(symbol.upper(), ens[0])
                for e in ens:
                    ens_to_symbol.setdefault(e, symbol)
            for alias in row[col["Synonyms"]].split("|"):
                alias = alias.strip().upper()
                if alias and alias != "-":
                    aliases.setdefault(alias, set()).update(ens)
    unique_alias = {k: next(iter(v)) for k, v in aliases.items() if len(v) == 1}
    return exact, unique_alias, ens_to_symbol

def source_to_model(var, genes: list[str]):
    human = [g for g in genes if g.startswith("ENSG")]
    valid = set(human)
    pos = {g: i for i, g in enumerate(genes)}
    exact, alias, _ = ncbi_maps(valid)
    preferred = None
    if "human_ensembl_gene" in var.columns:
        preferred = np.asarray(var["human_ensembl_gene"].astype(str))
    out = np.full(len(var), -1, dtype=np.int32)
    methods = {"direct_ensembl": 0, "var_human_ensembl": 0, "exact_symbol": 0, "unique_alias": 0, "unmapped": 0}
    for i, name in enumerate(var.index.astype(str)):
        candidates = []
        if preferred is not None:
            candidates.extend(preferred[i].replace(";", "|").replace(",", "|").split("|"))
        candidates.append(name)
        found = None
        method = None
        for raw in candidates:
            key = str(raw).strip().split(".", 1)[0]
            if key in valid:
                found, method = key, "direct_ensembl" if raw == name else "var_human_ensembl"
                break
        if found is None:
            key = name.upper()
            if key in exact:
                found, method = exact[key], "exact_symbol"
            elif key in alias:
                found, method = alias[key], "unique_alias"
        if found is None:
            methods["unmapped"] += 1
        else:
            out[i] = pos[found]
            methods[method] += 1
    return out, np.asarray([pos[g] for g in human], dtype=np.int32), methods

def load_model(device: torch.device, transformer: str):
    sys.path.insert(0, str(SCDL_REPO))
    sys.path.insert(0, str(REPO))
    from scprint import scPrint

    ckpt = checkpoint()
    h = dict(ckpt["hyper_parameters"])
    ignored_values = {"_instantiator": h.pop("_instantiator", None)}
    h["precpt_gene_emb"] = None
    h["transformer"] = transformer
    h["num_batch_labels"] = int(
        ckpt["state_dict"]["grad_reverse_discriminator_loss.out_layer.bias"].numel()
    )
    model = scPrint(**h)
    state = dict(ckpt["state_dict"])
    state["gene_encoder.embeddings.weight"] = state.pop("gene_encoder.embedding.weight")
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(incompat)
    model.eval().to(device)
    return model, ignored_values

def row_values(x, row: int, mapping: np.ndarray):
    if sparse.issparse(x):
        s, e = x.indptr[row], x.indptr[row + 1]
        src = x.indices[s:e]
        val = x.data[s:e].astype(np.float32, copy=False)
    else:
        dense = np.asarray(x[row]).reshape(-1)
        src = np.flatnonzero(dense)
        val = dense[src].astype(np.float32, copy=False)
    model_pos = mapping[src]
    keep = model_pos >= 0
    model_pos, val = model_pos[keep], val[keep]
    if not len(model_pos):
        return model_pos, val
    order = np.argsort(model_pos, kind="stable")
    model_pos, val = model_pos[order], val[order]
    uniq, first = np.unique(model_pos, return_index=True)
    return uniq.astype(np.int32), np.add.reduceat(val, first).astype(np.float32)

def collate_rows(x, depths, ids, rows, mapping, human_positions, job_key):
    exprs, positions = [], []
    for row in rows:
        pos, val = row_values(x, row, mapping)
        nz = val > 0
        pos, val = pos[nz], val[nz]
        seed_text = f"{job_key}\0{ids[row]}".encode()
        seed = int.from_bytes(hashlib.blake2b(seed_text, digest_size=4).digest(), "little")
        rng = np.random.RandomState(seed)
        if len(pos) > MAX_LEN:
            take = rng.choice(len(pos), MAX_LEN, replace=False)
            pos, val = pos[take], val[take]
        if len(pos) < MAX_LEN:
            zeros = np.setdiff1d(human_positions, pos, assume_unique=True)
            add = rng.choice(zeros, MAX_LEN - len(pos), replace=False)
            pos = np.concatenate([pos, add.astype(np.int32)])
            val = np.concatenate([val, np.zeros(len(add), dtype=np.float32)])
        positions.append(pos)
        exprs.append(val)
    return (
        torch.from_numpy(np.stack(positions).astype(np.int64)),
        torch.from_numpy(np.stack(exprs).astype(np.float32)),
        torch.from_numpy(np.asarray(depths[list(rows)], dtype=np.float32)),
    )

def export_static(target: Path):
    ckpt = checkpoint()
    genes = list(map(str, ckpt["hyper_parameters"]["genes"]))
    emb = ckpt["state_dict"]["gene_encoder.embedding.weight"].float().numpy()
    human = {g for g in genes if g.startswith("ENSG")}
    _, _, ens_to_symbol = ncbi_maps(human)
    seen, names, rows, unmapped = set(), [], [], []
    duplicates = 0
    for i, gene in enumerate(genes):
        if gene not in human:
            continue
        symbol = ens_to_symbol.get(gene)
        if symbol is None:
            unmapped.append(gene)
            continue
        if symbol in seen:
            duplicates += 1
            continue
        seen.add(symbol); names.append(symbol); rows.append(emb[i])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, genes=np.asarray(names, dtype=str), embedding=np.asarray(rows, dtype=np.float32))
    write_json(target.with_suffix(".manifest.json"), {
        "status": "PASS", "checkpoint": str(CHECKPOINT), "source_parameter": "gene_encoder.embedding.weight",
        "checkpoint_rows": len(genes), "human_checkpoint_rows": len(human), "mapped_unique_symbols": len(names),
        "unmapped_ensembl": unmapped, "duplicate_symbols_skipped": duplicates,
        "mapping_snapshot": str(GENE_INFO), "policy": "NCBI official Symbol from Ensembl dbXref; first checkpoint row per duplicate symbol",
    })
    print(json.dumps({"status": "PASS", "static_rows": len(names), "target": str(target)}), flush=True)
