"""Generate MetaQ memberships for subsequent raw-count aggregation."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--divisor", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    import anndata as ad
    import h5py
    from anndata.experimental import read_elem

    with h5py.File(args.input, "r") as handle:
        ids = read_elem(handle["obs"]).index.astype(str).to_numpy()
    if args.divisor <= 0 or len(ids) == 0:
        raise ValueError("Positive divisor and nonempty input required")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source cell IDs")
    if args.batch_size <= 0:
        raise ValueError("Positive batch size required")
    n = max(1, len(ids) // args.divisor)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(ids) < args.divisor:
        groups = np.zeros(len(ids), np.int64)
        print("Fewer cells than divisor: assigning one group without MetaQ training")
    else:
        from MetaQ_sc import run_metaq, data_utils

        if not getattr(data_utils, "PAPER_PREPROCESSING", False):
            raise RuntimeError("Apply patches/metaq.patch as described in README.md")
        batch = 4096 if n > 1000 and args.batch_size <= 512 else args.batch_size
        if batch > len(ids):
            raise ValueError("Training batch exceeds cell count; set --batch-size for small inputs")
        print(f"MetaQ: metacells={n}, seed=1, epochs=300, init=Random, convergence=10, batch={batch}")
        run_metaq(
            data_path=[str(args.input)], data_type=["RNA"],
            save_name=str(args.output_dir), metacell_num=n, type_key="cell_type",
            codebook_init="Random", train_epoch=300, batch_size=args.batch_size,
            random_seed=1, converge_threshold=10, device=args.device,
        )
        result = ad.read_h5ad(args.output_dir / "save" / f"{n}metacell_ids.h5ad")
        if not result.obs_names.is_unique:
            raise ValueError("Duplicate MetaQ IDs")
        groups = result.obs.loc[ids, "metacell"].to_numpy()
    pd.DataFrame({"source_cell_id": ids, "metacell_id": groups}).to_csv(
        args.output_dir / "memberships.tsv", sep="\t", index=False,
    )


if __name__ == "__main__":
    main()
