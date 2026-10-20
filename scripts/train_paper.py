"""Run the paper's three-epoch MetacellFormer or SC control schedule.

Input shards must be prepared separately. Use --dry-run to inspect commands.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    config = json.loads(args.config.read_text())
    unit = config["units"][args.unit]
    world = int(config["world_size"])
    if int(args.nproc_per_node) != world:
        raise ValueError(f"This paper configuration uses {world} processes")
    effective_batch = world * int(config["per_device_batch"]) * int(config["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(int(unit["train_examples"]) / effective_batch)
    model_dir = args.model_resources_dir
    resources = [model_dir / "gene_vocabulary.json", model_dir / "gene_reference_support.npy",
                 model_dir / "training_resources/rank_sampling_stats.npz"]
    missing = [str(path) for path in resources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model training resources: {missing}")
    commands = []
    previous_boundary = None
    for entry in config["epochs"]:
        epoch = int(entry["epoch"])
        cumulative = epoch * steps_per_epoch
        output = args.output_dir / f"epoch{epoch}"
        boundary_name = f"{args.unit}_epoch{epoch}_step{cumulative}"
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
                   "--nproc_per_node", str(world), "--module", "metacellformer.training_runtime",
                   "--train_dataset_dir", str(args.train_dataset_dir),
                   "--eval_dataset_dir", str(args.eval_dataset_dir),
                   "--train_lengths_pkl", str(args.train_lengths_pkl),
                   "--token_dictionary", str(resources[0]),
                   "--gene_support_npy", str(resources[1]),
                   "--sampling_stats_npz", str(resources[2]),
                   "--output_dir", str(output),
                   "--per_device_batch", str(config["per_device_batch"]),
                   "--grad_accum_steps", str(config["gradient_accumulation_steps"]),
                   "--learning_rate", str(config["epochs"][0]["learning_rate"]),
                   "--warmup_steps", str(config["warmup_steps"]),
                   "--max_steps", str(cumulative),
                   "--scheduler_total_steps_override", str(steps_per_epoch),
                   "--eval_steps", str(cumulative + 1),
                   "--embedding_lr_multiplier", str(entry["embedding_lr_multiplier"]),
                   "--boundary-save", f"{cumulative}:{boundary_name}",
                   *config["common_flags"]]
        if previous_boundary is not None:
            command += ["--init_model_dir", str(previous_boundary / "final_model"),
                        "--resume_from_checkpoint", str(previous_boundary / "checkpoint"),
                        "--force_embedding_param_group", "--reset_scheduler_for_epoch",
                        "--epoch_scheduler_steps", str(steps_per_epoch),
                        "--epoch_warmup_steps", str(config["warmup_steps"]),
                        "--epoch_start_learning_rate", str(entry["learning_rate"])]
        else:
            command += ["--resume_from_checkpoint", "none"]
        commands.append(command)
        previous_boundary = output / "pass_boundaries" / boundary_name
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", choices=["metacell", "singlecell"], required=True)
    parser.add_argument("--train-dataset-dir", type=Path, required=True)
    parser.add_argument("--eval-dataset-dir", type=Path, required=True)
    parser.add_argument("--train-lengths-pkl", type=Path, required=True)
    parser.add_argument("--model-resources-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/pretraining.json")
    parser.add_argument("--nproc-per-node", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for command in build_commands(args):
        if args.dry_run:
            import shlex
            print(shlex.join(command))
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
