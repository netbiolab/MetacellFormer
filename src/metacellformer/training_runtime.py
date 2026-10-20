#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import concatenate_datasets, disable_caching, load_from_disk
from torch.utils.data import Subset
from transformers import BertConfig, EvalPrediction, Trainer, TrainingArguments
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, MaskedLMOutput
from transformers.models.bert.modeling_bert import (
    BERT_SELF_ATTENTION_CLASSES,
    BertEmbeddings,
    BertForMaskedLM,
    BertIntermediate,
    BertModel,
    BertPooler,
)

_THIS_DIR = Path(__file__).resolve().parent
_RUNTIME_ROOT = _THIS_DIR.parent
for _path in (_RUNTIME_ROOT, _THIS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

try:
    from pretrain.stochasticset_utils import (
        FullObservedCandidateDataset,
        apply_value_masking,
        gene_token_universe_from_token_dictionary,
        sample_candidate_subset,
    )
except ModuleNotFoundError:
    from stochasticset_utils import (
        FullObservedCandidateDataset,
        apply_value_masking,
        gene_token_universe_from_token_dictionary,
        sample_candidate_subset,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dataset_dir", required=True)
    ap.add_argument("--eval_dataset_dir", required=True)
    ap.add_argument("--eval_max_examples", type=int, default=None)
    ap.add_argument("--train_lengths_pkl", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--init_model_dir", default=None)
    ap.add_argument("--init_optimizer_state_path", default=None)
    ap.add_argument("--init_scheduler_state_path", default=None)
    ap.add_argument("--scheduler_total_steps_override", type=int, default=None)
    ap.add_argument("--token_dictionary", required=True)
    ap.add_argument("--max_input_size", type=int, default=4096)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--num_hidden_layers", type=int, default=12)
    ap.add_argument("--num_attention_heads", type=int, default=12)
    ap.add_argument("--intermediate_size", type=int, default=3072)
    ap.add_argument("--attn_implementation", choices=["auto", "eager", "sdpa"], default="auto")
    ap.add_argument("--norm_position", choices=["post", "pre"], default="post")
    ap.add_argument("--position_embedding_alpha", type=float, default=1.0)
    ap.add_argument("--pre_ln_final_layer_norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--encoder_layout", choices=["unique", "looped"], default="unique")
    ap.add_argument("--num_loop_steps", type=int, default=None)
    ap.add_argument("--num_unique_layers", type=int, default=None)
    ap.add_argument("--use_loop_step_embedding", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--domain_token_type_id", type=int, default=None)
    ap.add_argument("--sc_token_type_id", type=int, default=0)
    ap.add_argument("--mc_token_type_id", type=int, default=1)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=24)
    ap.add_argument(
        "--exact_train_coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Group the train dataset into a sampler-divisible partition so one logical epoch "
            "visits every source example exactly once. The DataLoader batch size becomes one group."
        ),
    )
    ap.add_argument(
        "--exact_train_coverage_expected_examples",
        type=int,
        default=None,
        help="Required source-example count when --exact_train_coverage is enabled.",
    )
    ap.add_argument("--learning_rate", type=float, default=8e-5)
    ap.add_argument("--embedding_lr_multiplier", type=float, default=1.0)
    ap.add_argument("--embedding_lr_multiplier_after_step", type=int, default=None)
    ap.add_argument("--embedding_lr_multiplier_after", type=float, default=None)
    ap.add_argument("--force_embedding_param_group", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reset_scheduler_for_epoch", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--epoch_scheduler_steps", type=int, default=None)
    ap.add_argument("--epoch_warmup_steps", type=int, default=None)
    ap.add_argument("--epoch_start_learning_rate", type=float, default=None)
    ap.add_argument("--warmup_steps", type=int, default=5000)
    ap.add_argument("--weight_decay", type=float, default=0.001)
    ap.add_argument("--lr_scheduler_type", choices=["linear", "cosine_with_floor"], default="linear")
    ap.add_argument("--cosine_min_lr_ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_seed", type=int, default=None)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Checkpoint interval. Defaults to --eval_steps for backward compatibility.",
    )
    ap.add_argument(
        "--disable_in_training_eval",
        action="store_true",
        help="Disable Trainer evaluation during training without disabling periodic checkpoint saves.",
    )
    ap.add_argument("--eval_accum_steps", type=int, default=2)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=500000)
    ap.add_argument("--target_tokens", type=float, default=None)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    ap.add_argument(
        "--mask_sampling",
        choices=["uniform", "rank_frequency_balanced"],
        default="uniform",
    )
    ap.add_argument("--sampling_stats_npz", default=None)
    ap.add_argument("--mask_frequency_exponent", type=float, default=0.25)
    ap.add_argument("--mask_rank_bands", type=int, default=4)
    ap.add_argument("--gene_mlm_policy", choices=["bert_801010", "mask_only"], default="bert_801010")
    ap.add_argument("--gene_mlm_mask_replace_prob", type=float, default=None)
    ap.add_argument("--gene_mlm_random_replace_prob", type=float, default=None)
    ap.add_argument("--hard_random_replacement", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cosine_mlm_head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cosine_mlm_temperature", type=float, default=20.0)
    ap.add_argument("--dual_mask_views", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fixed_gene_feature_npy", default=None)
    ap.add_argument("--fixed_gene_feature_scale", type=float, default=0.5)
    ap.add_argument("--retrieval_feature_weight", type=float, default=0.4)
    ap.add_argument("--dataset_balance_exponent", type=float, default=0.0)
    ap.add_argument("--dataset_weight_min", type=float, default=0.5)
    ap.add_argument("--dataset_weight_max", type=float, default=2.0)
    ap.add_argument("--use_value_input", action="store_true")
    ap.add_argument("--use_value_loss", action="store_true")
    ap.add_argument("--value_loss_weight", type=float, default=0.1)
    ap.add_argument("--n_value_bins", type=int, default=64)
    ap.add_argument("--value_pad_id", type=int, default=0)
    ap.add_argument("--value_mask_id", type=int, default=65)
    ap.add_argument("--value_vocab_size", type=int, default=None)
    ap.add_argument("--program_targets_manifest", default=None)
    ap.add_argument("--program_loss_weight", type=float, default=0.0)
    ap.add_argument("--program_dim", type=int, default=64)
    ap.add_argument("--program_huber_delta", type=float, default=1.0)
    ap.add_argument("--program_target_required_for_train", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gene_loading_targets_manifest", default=None)
    ap.add_argument("--gene_loading_loss_weight", type=float, default=0.0)
    ap.add_argument("--gene_loading_dim", type=int, default=64)
    ap.add_argument("--gene_loading_huber_delta", type=float, default=1.0)
    ap.add_argument(
        "--gene_loading_gradient_scope",
        choices=["contextual", "native_embedding"],
        default="contextual",
        help="Apply gene-loading loss to contextual hidden states or native input gene embeddings.",
    )
    ap.add_argument(
        "--gene_loading_calibration_target_ratio",
        type=float,
        default=None,
        help="Calibrate the fixed gene-loading coefficient to this native-embedding gradient ratio.",
    )
    ap.add_argument("--fixed_coordinate", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gene_mlm_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--value_target_decoupled", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--value_target_probability", type=float, default=0.15)
    ap.add_argument(
        "--value_target_scope",
        choices=["observed_nonzero", "all_eligible"],
        default="observed_nonzero",
    )
    ap.add_argument("--gene_support_npy", default=None)
    ap.add_argument("--visible_gene_discrimination_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--visible_gene_disc_probability", type=float, default=0.15)
    ap.add_argument("--visible_gene_disc_negative_k", type=int, default=128)
    ap.add_argument("--visible_gene_disc_overlap_policy", choices=["disjoint", "allow"], default="disjoint")
    ap.add_argument("--visible_gene_disc_same_row_min_distance", type=int, default=20)
    ap.add_argument("--visible_gene_disc_same_row_max_distance", type=int, default=300)
    ap.add_argument("--visible_gene_disc_support_log_tolerance", type=float, default=0.25)
    ap.add_argument("--visible_gene_disc_support_log_tolerance_fallback", type=float, default=0.5)
    ap.add_argument("--visible_gene_disc_same_row_n", type=int, default=51)
    ap.add_argument("--visible_gene_disc_support_n", type=int, default=51)
    ap.add_argument("--visible_gene_disc_random_n", type=int, default=26)
    ap.add_argument("--visible_gene_loss_weight", type=float, default=0.03)
    ap.add_argument("--view_consistency_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--view_consistency_weight", type=float, default=0.01)
    ap.add_argument("--view_gene_dropout_prob", type=float, default=0.10)
    ap.add_argument("--view_value_jitter_prob", type=float, default=0.10)
    ap.add_argument("--view_value_jitter_delta", type=int, default=1)
    ap.add_argument("--masked_gene_discrimination_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--masked_gene_disc_probability", type=float, default=0.10)
    ap.add_argument("--masked_gene_loss_weight", type=float, default=0.03)
    ap.add_argument("--rtd_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--rtd_loss_weight", type=float, default=0.0)
    ap.add_argument("--rtd_visible_positive_policy", choices=["neg_count_match"], default="neg_count_match")
    ap.add_argument("--gene_disc_support_bucket_count", type=int, default=64)
    ap.add_argument("--value_mask_ratio", type=float, default=0.15)
    ap.add_argument("--value_nonzero_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ordinal_huber_weight", type=float, default=0.0)
    ap.add_argument("--ordinal_huber_delta", type=float, default=1.0)
    ap.add_argument("--varcov_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--varcov_lambda", type=float, default=0.0)
    ap.add_argument("--varcov_layers", default="")
    ap.add_argument("--varcov_positions", choices=["masked", "all"], default="masked")
    ap.add_argument(
        "--input_subset_mode",
        choices=["legacy_fixed_length", "static_top_nonzero_4096", "stochastic_mixed_4096"],
        default="legacy_fixed_length",
    )
    ap.add_argument(
        "--value_target_mode",
        choices=["legacy_all_masked", "observed_nonzero_only", "mixed_nonzero_lowzero"],
        default="legacy_all_masked",
    )
    ap.add_argument("--subset_top_count", type=int, default=2457)
    ap.add_argument("--subset_random_count", type=int, default=1024)
    ap.add_argument("--subset_fill_count", type=int, default=615)
    ap.add_argument("--target_nonzero_fraction", type=float, default=0.9)
    ap.add_argument("--randorder_seed", type=int, default=42)
    ap.add_argument("--fixed_length_randorder", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--paired_metacell_token_index_csv", default=None)
    ap.add_argument("--paired_mode", choices=["off", "one_to_one"], default="off")
    ap.add_argument("--paired_mc_loss_weight", type=float, default=1.0)
    ap.add_argument("--gene_pool_contrastive_enable", action="store_true")
    ap.add_argument("--gene_pool_projection_dim", type=int, default=256)
    ap.add_argument("--gene_pool_temperature", type=float, default=0.07)
    ap.add_argument("--gene_pool_stopgrad_anchor", action="store_true")
    ap.add_argument("--gene_pool_lambda_point", action="append", default=[])
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="MetacellFormer")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_anonymous", default="allow")
    ap.add_argument("--dataloader_num_workers", type=int, default=8)
    ap.add_argument("--disable_persistent_workers", action="store_true")
    ap.add_argument("--ignore_data_skip", action="store_true")
    ap.add_argument("--stream_eval_masked_acc_only", action="store_true")
    ap.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--training_metric_local_accumulation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--resume_from_checkpoint", default="none")
    ap.add_argument("--boundary-save", action="append", default=[])
    ap.add_argument("--time-boundary-save", action="append", default=[])
    ap.add_argument("--stop-after-wall-seconds", type=float, default=None)
    ap.add_argument("--skip-boundary-exports", action="store_true", default=False)
    ap.add_argument("--skip-final-export", action="store_true", default=False)
    return ap.parse_args()


def resolve_train_eval_save_policy(args: argparse.Namespace) -> tuple[str, int, int]:
    eval_steps = int(args.eval_steps)
    if eval_steps <= 0:
        raise ValueError(f"eval_steps must be positive, got {eval_steps}")
    raw_save_steps = getattr(args, "save_steps", None)
    save_steps = eval_steps if raw_save_steps is None else int(raw_save_steps)
    if save_steps <= 0:
        raise ValueError(f"save_steps must be positive, got {save_steps}")
    evaluation_strategy = "no" if bool(getattr(args, "disable_in_training_eval", False)) else "steps"
    return evaluation_strategy, eval_steps, save_steps


def parse_int_list(raw: str | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        items = [int(item) for item in raw]
    else:
        text = str(raw).strip()
        if not text:
            return ()
        items = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    seen: set[int] = set()
    ordered: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def default_gene_mlm_corruption_probs(gene_mlm_policy: str) -> tuple[float, float, float]:
    policy = str(gene_mlm_policy).strip().lower()
    if policy == "bert_801010":
        return 0.8, 0.1, 0.1
    if policy == "mask_only":
        return 1.0, 0.0, 0.0
    raise ValueError(f"unsupported gene_mlm_policy={gene_mlm_policy}")


def resolve_gene_mlm_corruption_probs(
    *,
    gene_mlm_policy: str,
    mask_replace_prob: float | None = None,
    random_replace_prob: float | None = None,
    unchanged_prob: float | None = None,
) -> tuple[float, float, float]:
    if mask_replace_prob is None and random_replace_prob is None and unchanged_prob is None:
        return default_gene_mlm_corruption_probs(gene_mlm_policy)

    if unchanged_prob is None:
        if mask_replace_prob is None or random_replace_prob is None:
            raise ValueError(
                "explicit gene MLM corruption requires both mask_replace_prob and random_replace_prob when unchanged_prob is omitted"
            )
        unchanged_prob = 1.0 - float(mask_replace_prob) - float(random_replace_prob)
    if mask_replace_prob is None or random_replace_prob is None:
        raise ValueError(
            "explicit gene MLM corruption requires mask_replace_prob and random_replace_prob"
        )

    mask_prob = float(mask_replace_prob)
    random_prob = float(random_replace_prob)
    unchanged = float(unchanged_prob)
    values = {
        "gene_mlm_mask_replace_prob": mask_prob,
        "gene_mlm_random_replace_prob": random_prob,
        "gene_mlm_unchanged_prob": unchanged,
    }
    for name, value in values.items():
        if value < -1e-8 or value > 1.0 + 1e-8:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    total = mask_prob + random_prob + unchanged
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "gene MLM corruption probabilities must sum to 1.0, "
            f"got mask={mask_prob}, random={random_prob}, unchanged={unchanged}, total={total}"
        )
    return mask_prob, random_prob, unchanged


def normalize_varcov_layers(raw: str | list[int] | tuple[int, ...] | None, num_hidden_layers: int) -> tuple[int, ...]:
    layers = parse_int_list(raw)
    normalized: list[int] = []
    max_layer = int(num_hidden_layers)
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx > max_layer:
            raise ValueError(f"varcov layer index must be in [0, {max_layer}], got {layer_idx}")
        normalized.append(int(layer_idx))
    return tuple(normalized)


def offdiag(mat: torch.Tensor) -> torch.Tensor:
    if mat.ndim != 2 or mat.size(0) != mat.size(1):
        raise ValueError(f"offdiag expects a square matrix, got shape={tuple(mat.shape)}")
    n = int(mat.size(0))
    if n <= 1:
        return mat.new_zeros((0,), dtype=mat.dtype)
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def load_dataset_any(dataset_dir: str):
    d = Path(dataset_dir)
    if (d / "candidate_cache_manifest.json").exists():
        return FullObservedCandidateDataset(d)
    try:
        return load_from_disk(str(d))
    except Exception:
        shard_dirs = sorted(d.glob("*.dataset"))
        if not shard_dirs:
            raise FileNotFoundError(f"No dataset found under {dataset_dir}")
        shards = [load_from_disk(str(sd)) for sd in shard_dirs]
        return concatenate_datasets(shards)


EXACT_COVERAGE_SAMPLE_IDS_KEY = "exact_coverage_sample_ids"
EXACT_COVERAGE_STATE_PREFIX = "exact_train_coverage_rank"


@dataclass(frozen=True)
class ExactTrainCoveragePlan:
    expected_examples: int
    target_group_size: int
    world_size: int
    grad_accum_steps: int
    group_count: int
    small_group_size: int
    small_group_count: int
    large_group_size: int
    large_group_count: int
    optimizer_steps_per_epoch: int
    dataloader_item_batch_size: int
    permutation_seed: int
    permutation_sha256: str
    structural_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "expected_examples": int(self.expected_examples),
            "target_group_size": int(self.target_group_size),
            "world_size": int(self.world_size),
            "grad_accum_steps": int(self.grad_accum_steps),
            "group_count": int(self.group_count),
            "small_group_size": int(self.small_group_size),
            "small_group_count": int(self.small_group_count),
            "large_group_size": int(self.large_group_size),
            "large_group_count": int(self.large_group_count),
            "optimizer_steps_per_epoch": int(self.optimizer_steps_per_epoch),
            "dataloader_item_batch_size": int(self.dataloader_item_batch_size),
            "permutation_seed": int(self.permutation_seed),
            "permutation_sha256": str(self.permutation_sha256),
            "mapping": "seeded_numpy_pcg64_permutation_group_slices_v1",
            "structural_sha256": str(self.structural_sha256),
        }


def exact_coverage_permutation(expected_examples: int, permutation_seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(int(permutation_seed)))
    return np.asarray(rng.permutation(int(expected_examples)), dtype=np.int64)


def build_exact_train_coverage_plan(
    *,
    expected_examples: int,
    target_group_size: int,
    world_size: int,
    grad_accum_steps: int,
    permutation_seed: int,
) -> ExactTrainCoveragePlan:
    expected_examples = int(expected_examples)
    target_group_size = int(target_group_size)
    world_size = int(world_size)
    grad_accum_steps = int(grad_accum_steps)
    permutation_seed = int(permutation_seed)
    values = {
        "expected_examples": expected_examples,
        "target_group_size": target_group_size,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
    }
    invalid = {name: value for name, value in values.items() if value <= 0}
    if invalid:
        raise ValueError(f"exact train coverage values must be positive: {invalid}")

    groups_per_optimizer_step = world_size * grad_accum_steps
    minimum_group_count = int(math.ceil(expected_examples / target_group_size))
    group_count = int(
        math.ceil(minimum_group_count / groups_per_optimizer_step) * groups_per_optimizer_step
    )
    if group_count > expected_examples:
        raise ValueError(
            "exact train coverage would require empty groups: "
            f"examples={expected_examples}, groups={group_count}"
        )

    small_group_size, large_group_count = divmod(expected_examples, group_count)
    large_group_size = small_group_size + (1 if large_group_count else 0)
    small_group_count = group_count - large_group_count
    if large_group_size > target_group_size:
        raise AssertionError("exact coverage grouping exceeded target_group_size")

    permutation = exact_coverage_permutation(expected_examples, permutation_seed)
    permutation_sha256 = hashlib.sha256(permutation.tobytes(order="C")).hexdigest()
    structural_payload = {
        "schema_version": 1,
        "expected_examples": expected_examples,
        "target_group_size": target_group_size,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
        "group_count": group_count,
        "small_group_size": small_group_size,
        "small_group_count": small_group_count,
        "large_group_size": large_group_size,
        "large_group_count": large_group_count,
        "permutation_seed": permutation_seed,
        "permutation_sha256": permutation_sha256,
        "mapping": "seeded_numpy_pcg64_permutation_group_slices_v1",
    }
    structural_sha256 = hashlib.sha256(
        json.dumps(structural_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return ExactTrainCoveragePlan(
        expected_examples=expected_examples,
        target_group_size=target_group_size,
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
        group_count=group_count,
        small_group_size=small_group_size,
        small_group_count=small_group_count,
        large_group_size=large_group_size,
        large_group_count=large_group_count,
        optimizer_steps_per_epoch=group_count // groups_per_optimizer_step,
        dataloader_item_batch_size=1,
        permutation_seed=permutation_seed,
        permutation_sha256=permutation_sha256,
        structural_sha256=structural_sha256,
    )


@dataclass(frozen=True)
class ExactCoverageGroup:
    features: tuple[dict[str, Any], ...]
    sample_ids: tuple[int, ...]


class ExactCoverageGroupedDataset:
    def __init__(self, dataset, plan: ExactTrainCoveragePlan):
        if len(dataset) != int(plan.expected_examples):
            raise ValueError(
                "exact train coverage dataset length mismatch: "
                f"dataset={len(dataset)}, expected={plan.expected_examples}"
            )
        self.dataset = dataset
        self.plan = plan
        self.permutation = exact_coverage_permutation(
            plan.expected_examples,
            plan.permutation_seed,
        )
        permutation_sha256 = hashlib.sha256(self.permutation.tobytes(order="C")).hexdigest()
        if permutation_sha256 != plan.permutation_sha256:
            raise RuntimeError(
                "exact train coverage permutation does not match its structural plan: "
                f"{permutation_sha256} != {plan.permutation_sha256}"
            )

    def __len__(self) -> int:
        return int(self.plan.group_count)

    def group_bounds(self, group_index: int) -> tuple[int, int]:
        group_index = int(group_index)
        if group_index < 0:
            group_index += len(self)
        if group_index < 0 or group_index >= len(self):
            raise IndexError(group_index)
        if group_index < self.plan.large_group_count:
            start = group_index * self.plan.large_group_size
            size = self.plan.large_group_size
        else:
            start = (
                self.plan.large_group_count * self.plan.large_group_size
                + (group_index - self.plan.large_group_count) * self.plan.small_group_size
            )
            size = self.plan.small_group_size
        return int(start), int(start + size)

    def __getitem__(self, group_index: int) -> ExactCoverageGroup:
        start, stop = self.group_bounds(group_index)
        sample_ids = tuple(int(value) for value in self.permutation[start:stop])
        features = tuple(dict(self.dataset[sample_id]) for sample_id in sample_ids)
        return ExactCoverageGroup(features=features, sample_ids=sample_ids)


class ExactCoverageCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __getattr__(self, name: str):
        return getattr(self.base_collator, name)

    def __call__(self, items):
        if not items or not isinstance(items[0], ExactCoverageGroup):
            return self.base_collator(items)
        if any(not isinstance(item, ExactCoverageGroup) for item in items):
            raise TypeError("exact coverage groups cannot be mixed with ordinary dataset rows")
        features = [feature for item in items for feature in item.features]
        sample_ids = [sample_id for item in items for sample_id in item.sample_ids]
        batch = self.base_collator(features)
        if EXACT_COVERAGE_SAMPLE_IDS_KEY in batch:
            raise KeyError(f"reserved collator key already exists: {EXACT_COVERAGE_SAMPLE_IDS_KEY}")
        batch[EXACT_COVERAGE_SAMPLE_IDS_KEY] = torch.as_tensor(sample_ids, dtype=torch.long)
        return batch


class ExactCoverageTracker:
    def __init__(self, plan: ExactTrainCoveragePlan):
        self.plan = plan
        self._seen = np.zeros((plan.expected_examples,), dtype=np.uint8)

    @property
    def local_seen_examples(self) -> int:
        return int(self._seen.sum(dtype=np.int64))

    def record(self, sample_ids: torch.Tensor | np.ndarray | list[int] | tuple[int, ...]) -> None:
        if torch.is_tensor(sample_ids):
            ids = sample_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        else:
            ids = np.asarray(sample_ids, dtype=np.int64)
        ids = ids.reshape(-1)
        if ids.size == 0:
            raise ValueError("exact coverage batch contained no sample IDs")
        if int(ids.min()) < 0 or int(ids.max()) >= self.plan.expected_examples:
            raise ValueError(
                "exact coverage sample ID out of range: "
                f"min={int(ids.min())}, max={int(ids.max())}, expected={self.plan.expected_examples}"
            )
        unique = np.unique(ids)
        if unique.size != ids.size:
            raise RuntimeError("exact coverage batch contains duplicate sample IDs")
        repeated = unique[self._seen[unique] != 0]
        if repeated.size:
            preview = repeated[:8].tolist()
            raise RuntimeError(f"exact coverage replay detected on local rank: {preview}")
        self._seen[unique] = 1

    def state_path(self, checkpoint_dir: str | Path, process_index: int) -> Path:
        return Path(checkpoint_dir) / f"{EXACT_COVERAGE_STATE_PREFIX}{int(process_index)}.npz"

    def save(self, checkpoint_dir: str | Path, *, process_index: int, global_step: int) -> Path:
        path = self.state_path(checkpoint_dir, process_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                packed_seen=np.packbits(self._seen, bitorder="little"),
                expected_examples=np.asarray([self.plan.expected_examples], dtype=np.int64),
                local_seen_examples=np.asarray([self.local_seen_examples], dtype=np.int64),
                global_step=np.asarray([int(global_step)], dtype=np.int64),
                structural_sha256=np.asarray([self.plan.structural_sha256]),
            )
        tmp.replace(path)
        return path

    def load(self, checkpoint_dir: str | Path, *, process_index: int, global_step: int) -> Path:
        path = self.state_path(checkpoint_dir, process_index)
        if not path.is_file():
            raise FileNotFoundError(f"missing exact coverage checkpoint state: {path}")
        with np.load(path, allow_pickle=False) as state:
            expected_examples = int(state["expected_examples"][0])
            saved_local_seen = int(state["local_seen_examples"][0])
            saved_global_step = int(state["global_step"][0])
            structural_sha256 = str(state["structural_sha256"][0])
            packed_seen = np.asarray(state["packed_seen"], dtype=np.uint8)
        checks = {
            "expected_examples": (expected_examples, self.plan.expected_examples),
            "global_step": (saved_global_step, int(global_step)),
            "structural_sha256": (structural_sha256, self.plan.structural_sha256),
        }
        mismatches = {key: pair for key, pair in checks.items() if pair[0] != pair[1]}
        if mismatches:
            raise RuntimeError(f"exact coverage checkpoint state mismatch at {path}: {mismatches}")
        seen = np.unpackbits(packed_seen, bitorder="little")[: self.plan.expected_examples]
        self._seen = np.asarray(seen, dtype=np.uint8)
        if self.local_seen_examples != saved_local_seen:
            raise RuntimeError(
                f"exact coverage checkpoint count mismatch at {path}: "
                f"bitmap={self.local_seen_examples}, saved={saved_local_seen}"
            )
        return path

    def summarize(self, *, device: torch.device, global_step: int) -> dict[str, Any]:
        # NCCL supports int32 reductions but not torch.int16/Short.
        global_counts = torch.from_numpy(self._seen.astype(np.int32, copy=False)).to(device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        counts = global_counts.to(device="cpu", dtype=torch.int32).numpy()
        actual_examples = int(counts.sum(dtype=np.int64))
        actual_unique_examples = int(np.count_nonzero(counts))
        missing_examples = int(np.count_nonzero(counts == 0))
        duplicate_examples = int(np.maximum(counts.astype(np.int64) - 1, 0).sum())
        complete = missing_examples == 0 and duplicate_examples == 0
        if duplicate_examples:
            raise RuntimeError(
                "exact train coverage validation found cross-rank replay: "
                f"duplicate_occurrences={duplicate_examples}"
            )
        if int(global_step) == self.plan.optimizer_steps_per_epoch and not complete:
            raise RuntimeError(
                "exact train coverage validation failed at the one-epoch boundary: "
                f"missing={missing_examples}, actual_unique={actual_unique_examples}"
            )
        observed_sha256 = hashlib.sha256(
            np.packbits(counts != 0, bitorder="little").tobytes()
        ).hexdigest()
        return {
            **self.plan.as_dict(),
            "actual_optimizer_steps": int(global_step),
            "actual_examples": actual_examples,
            "actual_unique_examples": actual_unique_examples,
            "missing_examples": missing_examples,
            "duplicate_examples": duplicate_examples,
            "complete": bool(complete),
            "validation_passed": True,
            "observed_sample_id_bitmap_sha256": observed_sha256,
        }


def load_token_dictionary_any(path: str | Path) -> dict[str, int]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        specials = {str(k): int(v) for k, v in dict(obj["special_tokens"]).items()}
        genes = [str(gene) for gene in obj["genes"]]
        out = dict(specials)
        next_id = (max(specials.values()) + 1) if specials else 0
        for idx, gene in enumerate(genes):
            out[gene] = next_id + idx
        return out
    with open(p, "rb") as handle:
        raw = pickle.load(handle)
    return {str(key): int(value) for key, value in raw.items()}


@dataclass(frozen=True)
class GeneDiscRowContext:
    row_gene_ids: np.ndarray
    row_source_rank: np.ndarray
    same_row_nonzero_gene_ids: np.ndarray
    same_row_nonzero_source_rank: np.ndarray
    same_row_gene_universe: np.ndarray


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlphaScaledBertEmbeddings(BertEmbeddings):
    """BERT embeddings with a configurable multiplier on absolute position embeddings."""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.position_embedding_alpha = float(getattr(config, "position_embedding_alpha", 1.0))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + (self.position_embedding_alpha * self.position_embeddings(position_ids))
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def resolved_encoder_layout(config: BertConfig) -> str:
    layout = str(getattr(config, "encoder_layout", "unique")).strip().lower()
    if layout not in {"unique", "looped"}:
        raise ValueError(f"unsupported encoder_layout={layout}")
    return layout


def resolved_num_loop_steps(config: BertConfig) -> int:
    raw = getattr(config, "num_loop_steps", None)
    if raw is None:
        raw = getattr(config, "num_hidden_layers", 1)
    steps = int(raw)
    if steps <= 0:
        raise ValueError(f"num_loop_steps must be > 0, got {steps}")
    return steps


def resolved_num_unique_layers(config: BertConfig) -> int:
    layout = resolved_encoder_layout(config)
    raw = getattr(config, "num_unique_layers", None)
    if raw is None:
        raw = 1 if layout == "looped" else getattr(config, "num_hidden_layers", 1)
    unique_layers = int(raw)
    if unique_layers <= 0:
        raise ValueError(f"num_unique_layers must be > 0, got {unique_layers}")
    if layout == "looped" and unique_layers != 1:
        raise ValueError(f"looped encoder currently supports num_unique_layers=1 only, got {unique_layers}")
    return unique_layers


class PreLayerNormBertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.dropout(hidden_states)


class PreLayerNormBertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.dropout(hidden_states)


class PreLayerNormBertAttention(nn.Module):
    def __init__(self, config: BertConfig, position_embedding_type: str | None = None):
        super().__init__()
        self.self = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config,
            position_embedding_type=position_embedding_type,
        )
        self.output = PreLayerNormBertSelfOutput(config)
        self.pruned_heads: set[int] = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0])
        return (attention_output,) + self_outputs[1:]


class PreLayerNormBertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.is_decoder or config.add_cross_attention:
            raise ValueError("PreLayerNormBertLayer is only implemented for encoder-only BERT")
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        if self.chunk_size_feed_forward:
            raise ValueError("PreLayerNormBertLayer does not support chunk_size_feed_forward != 0")
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = PreLayerNormBertAttention(config)
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = BertIntermediate(config)
        self.output = PreLayerNormBertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, ...]:
        if encoder_hidden_states is not None or encoder_attention_mask is not None or past_key_value is not None:
            raise ValueError("PreLayerNormBertLayer does not support decoder/cross-attention cache inputs")

        attention_inputs = self.attention_layer_norm(hidden_states)
        self_attention_outputs = self.attention(
            attention_inputs,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = hidden_states + self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        ffn_inputs = self.ffn_layer_norm(attention_output)
        intermediate_output = self.intermediate(ffn_inputs)
        layer_output = attention_output + self.output(intermediate_output)
        return (layer_output,) + outputs


class PreLayerNormBertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PreLayerNormBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if bool(getattr(config, "pre_ln_final_layer_norm", True))
            else None
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    None,
                ]
                if value is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


class LoopedPreLayerNormBertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.num_loop_steps = resolved_num_loop_steps(config)
        self.num_unique_layers = resolved_num_unique_layers(config)
        self.use_loop_step_embedding = bool(getattr(config, "use_loop_step_embedding", False))
        self.layer = nn.ModuleList([PreLayerNormBertLayer(config) for _ in range(self.num_unique_layers)])
        self.loop_step_embeddings = (
            nn.Embedding(self.num_loop_steps, config.hidden_size) if self.use_loop_step_embedding else None
        )
        self.final_layer_norm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            if bool(getattr(config, "pre_ln_final_layer_norm", True))
            else None
        )
        self.gradient_checkpointing = False

    def _layer_head_mask(
        self,
        head_mask: Optional[torch.FloatTensor],
        *,
        step_idx: int,
        layer_idx: int,
    ) -> Optional[torch.FloatTensor]:
        if head_mask is None:
            return None
        if isinstance(head_mask, (list, tuple)):
            if len(head_mask) == self.num_loop_steps:
                return head_mask[step_idx]
            if len(head_mask) == self.num_unique_layers:
                return head_mask[layer_idx]
            if len(head_mask) > 0:
                return head_mask[min(layer_idx, len(head_mask) - 1)]
            return None
        if hasattr(head_mask, "size") and head_mask.size(0) == self.num_loop_steps:
            return head_mask[step_idx]
        if hasattr(head_mask, "size") and head_mask.size(0) == self.num_unique_layers:
            return head_mask[layer_idx]
        if hasattr(head_mask, "size") and head_mask.size(0) > 0:
            return head_mask[min(layer_idx, int(head_mask.size(0)) - 1)]
        return head_mask

    def _past_key_value_for_step(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        *,
        step_idx: int,
        layer_idx: int,
    ) -> Optional[Tuple[torch.FloatTensor]]:
        if past_key_values is None:
            return None
        if len(past_key_values) == self.num_loop_steps:
            return past_key_values[step_idx]
        if len(past_key_values) == self.num_unique_layers:
            return past_key_values[layer_idx]
        if len(past_key_values) > 0:
            return past_key_values[min(layer_idx, len(past_key_values) - 1)]
        return None

    def _apply_step_embedding(self, hidden_states: torch.Tensor, step_idx: int) -> torch.Tensor:
        if self.loop_step_embeddings is None:
            return hidden_states
        step_ids = torch.full(
            (int(hidden_states.size(0)),),
            int(step_idx),
            dtype=torch.long,
            device=hidden_states.device,
        )
        step_emb = self.loop_step_embeddings(step_ids).unsqueeze(1)
        return hidden_states + step_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for step_idx in range(self.num_loop_steps):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_idx = step_idx % self.num_unique_layers
            layer_module = self.layer[layer_idx]
            layer_head_mask = self._layer_head_mask(head_mask, step_idx=step_idx, layer_idx=layer_idx)
            past_key_value = self._past_key_value_for_step(
                past_key_values,
                step_idx=step_idx,
                layer_idx=layer_idx,
            )
            step_hidden_states = self._apply_step_embedding(hidden_states, step_idx)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    step_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    step_hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    None,
                ]
                if value is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=None,
        )


class PreLayerNormBertModel(BertModel):
    _no_split_modules = [
        "AlphaScaledBertEmbeddings",
        "PreLayerNormBertLayer",
        "LoopedPreLayerNormBertEncoder",
    ]

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True):
        super(BertModel, self).__init__(config)
        self.config = config
        self.embeddings = AlphaScaledBertEmbeddings(config)
        layout = resolved_encoder_layout(config)
        self.encoder = LoopedPreLayerNormBertEncoder(config) if layout == "looped" else PreLayerNormBertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type
        self.post_init()


def query_to_anchor_contrastive_loss(
    query_z: torch.Tensor,
    anchor_z: torch.Tensor,
    positive_indices: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if query_z.size(0) == 0 or anchor_z.size(0) == 0:
        return query_z.new_tensor(0.0)
    if query_z.size(1) != anchor_z.size(1):
        raise ValueError(f"query/anchor contrastive dim mismatch: {query_z.shape} vs {anchor_z.shape}")
    if positive_indices.ndim != 1 or positive_indices.size(0) != query_z.size(0):
        raise ValueError(
            f"positive_indices must have shape ({query_z.size(0)},), got {tuple(positive_indices.shape)}"
        )

    labels = positive_indices.to(device=query_z.device, dtype=torch.long)
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= anchor_z.size(0):
        raise ValueError(f"positive_indices out of range for {anchor_z.size(0)} anchors")

    logits = (query_z @ anchor_z.t()) / float(temperature)
    return nn.functional.cross_entropy(logits, labels)


def _get_world_size_safe() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def ensure_lengths_pickle(ds, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    if hasattr(ds, "manifest") and isinstance(getattr(ds, "manifest", None), dict):
        max_input_size = int(ds.manifest.get("max_input_size", 4097))
        lengths = [max_input_size] * int(len(ds))
    else:
        lengths = list(map(int, ds["length"]))
    with open(out_path, "wb") as f:
        pickle.dump(lengths, f)
    return out_path


def load_lengths(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    arr = np.asarray(obj, dtype=np.int64)
    arr[arr <= 0] = 1
    return arr


def parse_schedule_points(items: list[str] | None) -> tuple[tuple[int, float], ...]:
    parsed: dict[int, float] = {}
    for item in items or []:
        raw = str(item).strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"--gene_pool_lambda_point must be STEP:VALUE, got: {raw}")
        step_raw, value_raw = raw.split(":", 1)
        step = int(step_raw)
        value = float(value_raw)
        if step < 0:
            raise ValueError(f"--gene_pool_lambda_point step must be >= 0, got: {raw}")
        prev = parsed.get(step)
        if prev is not None and prev != value:
            raise ValueError(f"conflicting --gene_pool_lambda_point for step={step}: {prev} vs {value}")
        parsed[step] = value
    return tuple((step, parsed[step]) for step in sorted(parsed))


def resolve_scheduled_value(points: tuple[tuple[int, float], ...], step: int) -> float:
    if not points:
        return 0.0
    cur_step = int(step)
    if cur_step <= points[0][0]:
        return float(points[0][1])
    if cur_step >= points[-1][0]:
        return float(points[-1][1])
    for (step0, value0), (step1, value1) in zip(points[:-1], points[1:]):
        if step0 <= cur_step <= step1:
            if step0 == step1:
                return float(value1)
            alpha = float(cur_step - step0) / float(step1 - step0)
            return float(value0 + (alpha * (value1 - value0)))
    return float(points[-1][1])


def resolve_resume_checkpoint(value: str | None, output_dir: Path) -> str | None:
    if value is None or value.lower() in {"none", "false", "off"}:
        return None
    if value != "auto":
        return value
    checkpoints = []
    for p in output_dir.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
        except Exception:
            continue
        checkpoints.append((step, p))
    if not checkpoints:
        return None
    checkpoints.sort()
    return str(checkpoints[-1][1])


def parse_boundary_saves(items: list[str] | None) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in items or []:
        raw = str(item).strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"--boundary-save must be STEP:NAME, got: {raw}")
        step_raw, name_raw = raw.split(":", 1)
        step = int(step_raw)
        name = str(name_raw).strip()
        if step <= 0:
            raise ValueError(f"--boundary-save step must be positive, got: {raw}")
        if not name:
            raise ValueError(f"--boundary-save name must be non-empty, got: {raw}")
        prev = out.get(step)
        if prev is not None and prev != name:
            raise ValueError(f"conflicting --boundary-save for step={step}: {prev} vs {name}")
        out[step] = name
    return dict(sorted(out.items()))


def parse_time_boundary_saves(items: list[str] | None) -> dict[float, str]:
    out: dict[float, str] = {}
    names: set[str] = set()
    for item in items or []:
        raw = str(item).strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"--time-boundary-save must be SECONDS:NAME, got: {raw}")
        seconds_raw, name_raw = raw.split(":", 1)
        seconds = float(seconds_raw)
        name = str(name_raw).strip()
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"--time-boundary-save seconds must be positive finite, got: {raw}")
        if not name:
            raise ValueError(f"--time-boundary-save name must be non-empty, got: {raw}")
        prev = out.get(seconds)
        if prev is not None and prev != name:
            raise ValueError(f"conflicting --time-boundary-save for seconds={seconds}: {prev} vs {name}")
        if name in names and prev is None:
            raise ValueError(f"duplicate --time-boundary-save name: {name}")
        out[seconds] = name
        names.add(name)
    return dict(sorted(out.items()))


def ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"refusing to replace existing non-symlink path: {link_path}")
    link_path.symlink_to(target)


@dataclass
class PairedMetacellShard:
    dataset_idx: int
    input_ids: np.ndarray
    value_bins: np.ndarray
    attention_mask: np.ndarray
    metacell_local_idx: np.ndarray


class PairedMetacellLookup:
    def __init__(self, token_index_csv: str | Path):
        token_index_path = Path(token_index_csv).expanduser().resolve()
        if not token_index_path.exists():
            raise FileNotFoundError(f"missing paired metacell token index: {token_index_path}")

        self.token_index_path = token_index_path
        self.shards: list[PairedMetacellShard] = []
        self._pair_lookup: dict[tuple[int, int], tuple[int, int]] = {}
        self._cached_mean_tokens: dict[int, float] = {}

        with token_index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows.sort(key=lambda row: int(row["dataset_idx"]))
        for shard_idx, row in enumerate(rows):
            dataset_idx = int(row["dataset_idx"])
            input_ids = np.load(str(row["input_ids_path"]), mmap_mode="r")
            value_bins = np.load(str(row["value_bins_path"]), mmap_mode="r")
            attention_mask = np.load(str(row["attention_mask_path"]), mmap_mode="r")
            metacell_local_idx = np.load(str(row["metacell_path"]), mmap_mode="r")
            if input_ids.shape != value_bins.shape or input_ids.shape != attention_mask.shape:
                raise ValueError(f"paired metacell shard shape mismatch under {token_index_path}")
            if input_ids.shape[0] != int(np.asarray(metacell_local_idx).shape[0]):
                raise ValueError(f"paired metacell pair-key shape mismatch under {token_index_path}")
            self.shards.append(
                PairedMetacellShard(
                    dataset_idx=dataset_idx,
                    input_ids=input_ids,
                    value_bins=value_bins,
                    attention_mask=attention_mask,
                    metacell_local_idx=metacell_local_idx,
                )
            )
            for local_row, metacell_local_idx_value in enumerate(np.asarray(metacell_local_idx, dtype=np.int64).tolist()):
                self._pair_lookup.setdefault(
                    (dataset_idx, int(metacell_local_idx_value)),
                    (shard_idx, int(local_row)),
                )

    def get_example(self, *, dataset_idx: int, metacell_local_idx: int) -> dict[str, np.ndarray | int]:
        key = (int(dataset_idx), int(metacell_local_idx))
        shard_local = self._pair_lookup.get(key)
        if shard_local is None:
            raise KeyError(f"missing paired metacell row for dataset_idx={dataset_idx}, metacell_local_idx={metacell_local_idx}")
        shard_idx, local_row = shard_local
        shard = self.shards[shard_idx]
        return {
            "input_ids": np.asarray(shard.input_ids[local_row], dtype=np.int64),
            "value_bins": np.asarray(shard.value_bins[local_row], dtype=np.int64),
            "attention_mask": np.asarray(shard.attention_mask[local_row], dtype=np.int64),
            "dataset_idx": int(dataset_idx),
            "metacell_local_idx": int(metacell_local_idx),
        }

    def mean_clipped_tokens(self, max_input_size: int) -> float:
        key = int(max_input_size)
        cached = self._cached_mean_tokens.get(key)
        if cached is not None:
            return float(cached)

        total = 0.0
        count = 0
        for shard in self.shards:
            lengths = np.asarray(shard.attention_mask.sum(axis=1), dtype=np.int64)
            total += float(np.minimum(lengths, key).sum())
            count += int(lengths.size)
        mean_tokens = float(total / max(1, count))
        self._cached_mean_tokens[key] = mean_tokens
        return mean_tokens


@dataclass(frozen=True)
class ProgramTargetShard:
    dataset_idx: int
    dataset_id: str
    targets: np.ndarray
    n_rows: int


class ProgramTargetStore:
    def __init__(self, manifest_path: str | Path):
        path = Path(manifest_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"missing program target manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        program_dim = int(payload.get("program_dim", 0))
        if program_dim <= 0:
            raise ValueError(f"program target manifest has invalid program_dim={program_dim}: {path}")
        self.manifest_path = path
        self.program_dim = program_dim
        self.key_fields = tuple(payload.get("key_fields") or ["dataset_idx", "source_row_idx"])
        self.shards: dict[int, ProgramTargetShard] = {}
        for raw in payload.get("targets", []):
            dataset_idx = int(raw["dataset_idx"])
            target_path = Path(str(raw["path"]))
            if not target_path.is_absolute():
                target_path = path.parent / target_path
            target_path = target_path.expanduser().resolve()
            if not target_path.exists():
                raise FileNotFoundError(f"missing program target shard for dataset_idx={dataset_idx}: {target_path}")
            arr = np.load(str(target_path), mmap_mode="r")
            if arr.ndim != 2 or int(arr.shape[1]) != self.program_dim:
                raise ValueError(
                    "program target shard shape mismatch for "
                    f"dataset_idx={dataset_idx}: got {tuple(arr.shape)}, expected [N,{self.program_dim}]"
                )
            self.shards[dataset_idx] = ProgramTargetShard(
                dataset_idx=dataset_idx,
                dataset_id=str(raw.get("dataset_id", "")),
                targets=arr,
                n_rows=int(arr.shape[0]),
            )
        if not self.shards:
            raise ValueError(f"program target manifest contains no target shards: {path}")

    def get_target(self, feature: dict[str, Any]) -> np.ndarray | None:
        if "dataset_idx" not in feature:
            return None
        dataset_idx = int(feature["dataset_idx"])
        shard = self.shards.get(dataset_idx)
        if shard is None:
            return None
        if "source_row_idx" in feature:
            row_idx = int(feature["source_row_idx"])
        elif "metacell_local_idx" in feature:
            row_idx = int(feature["metacell_local_idx"])
        else:
            return None
        if row_idx < 0 or row_idx >= shard.n_rows:
            return None
        target = np.array(shard.targets[row_idx], dtype=np.float32, copy=True)
        if target.shape != (self.program_dim,) or not np.isfinite(target).all():
            return None
        return target


class GeneLoadingTargetStore:
    def __init__(self, manifest_path: str | Path):
        path = Path(manifest_path).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.manifest_path = path
        self.manifest = manifest
        self.loading_dim = int(manifest.get("loading_dim", manifest.get("gene_loading_dim", 0)))
        if self.loading_dim <= 0:
            raise ValueError(f"gene loading manifest missing positive loading_dim: {path}")

        targets_path = Path(str(manifest["gene_loading_targets_npy"]))
        if not targets_path.is_absolute():
            targets_path = path.parent / targets_path
        targets_path = targets_path.expanduser().resolve()
        if not targets_path.exists():
            raise FileNotFoundError(f"missing gene loading target array: {targets_path}")
        mask_path = Path(str(manifest["gene_loading_target_mask_npy"]))
        if not mask_path.is_absolute():
            mask_path = path.parent / mask_path
        mask_path = mask_path.expanduser().resolve()
        if not mask_path.exists():
            raise FileNotFoundError(f"missing gene loading target mask: {mask_path}")

        self.targets = np.load(str(targets_path), mmap_mode="r")
        self.target_mask = np.asarray(np.load(str(mask_path), mmap_mode="r"), dtype=bool)
        if self.targets.ndim != 2 or int(self.targets.shape[1]) != self.loading_dim:
            raise ValueError(
                "gene loading target shape mismatch: "
                f"got {tuple(self.targets.shape)}, expected [token_lookup,{self.loading_dim}]"
            )
        if self.target_mask.ndim != 1 or int(self.target_mask.shape[0]) != int(self.targets.shape[0]):
            raise ValueError(
                "gene loading target mask shape mismatch: "
                f"mask={tuple(self.target_mask.shape)} targets={tuple(self.targets.shape)}"
            )
        self.token_lookup_size = int(self.targets.shape[0])

    def build_batch(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = input_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        attn = attention_mask.detach().cpu().numpy().astype(bool, copy=False)
        in_range = (ids >= 0) & (ids < self.token_lookup_size)
        valid = attn & in_range
        has_target = np.zeros(ids.shape, dtype=bool)
        if bool(valid.any()):
            has_target[valid] = self.target_mask[ids[valid]]
        out = np.zeros(ids.shape + (self.loading_dim,), dtype=np.float32)
        if bool(has_target.any()):
            out[has_target, :] = np.asarray(self.targets[ids[has_target]], dtype=np.float32)
        return torch.from_numpy(out), torch.from_numpy(has_target)


class ValuePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, value_vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, value_vocab_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.dense(h)
        x = nn.functional.gelu(x)
        x = self.norm(x)
        return self.decoder(x)


class ProgramPredictionHead(nn.Module):
    def __init__(self, hidden_size: int, program_dim: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, program_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.dense(h)
        x = nn.functional.gelu(x)
        x = self.norm(x)
        return self.decoder(x)


@dataclass
class ValueAugmentedMaskedLMOutput(MaskedLMOutput):
    value_logits: torch.FloatTensor | None = None
    program_pred: torch.FloatTensor | None = None
    gene_loading_pred: torch.FloatTensor | None = None
    mlm_loss: torch.FloatTensor | None = None
    value_loss: torch.FloatTensor | None = None
    value_correct_n: torch.FloatTensor | None = None
    value_masked_n: torch.FloatTensor | None = None
    program_loss: torch.FloatTensor | None = None
    program_target_n: torch.FloatTensor | None = None
    program_abs_error_sum: torch.FloatTensor | None = None
    program_squared_error_sum: torch.FloatTensor | None = None
    program_pred_norm_sum: torch.FloatTensor | None = None
    program_target_norm_sum: torch.FloatTensor | None = None
    gene_loading_loss: torch.FloatTensor | None = None
    gene_loading_target_n: torch.FloatTensor | None = None
    gene_loading_abs_error_sum: torch.FloatTensor | None = None
    gene_loading_squared_error_sum: torch.FloatTensor | None = None
    gene_loading_pred_norm_sum: torch.FloatTensor | None = None
    gene_loading_target_norm_sum: torch.FloatTensor | None = None
    ordinal_huber_loss: torch.FloatTensor | None = None
    ordinal_abs_error_sum: torch.FloatTensor | None = None
    ordinal_squared_error_sum: torch.FloatTensor | None = None
    visible_gene_loss: torch.FloatTensor | None = None
    visible_gene_target_n: torch.FloatTensor | None = None
    visible_gene_pos_score_sum: torch.FloatTensor | None = None
    visible_gene_neg_score_sum: torch.FloatTensor | None = None
    visible_gene_pos_correct_n: torch.FloatTensor | None = None
    visible_gene_neg_correct_n: torch.FloatTensor | None = None
    visible_gene_neg_n: torch.FloatTensor | None = None
    visible_gene_same_row_neg_correct_n: torch.FloatTensor | None = None
    visible_gene_same_row_neg_n: torch.FloatTensor | None = None
    visible_gene_support_neg_correct_n: torch.FloatTensor | None = None
    visible_gene_support_neg_n: torch.FloatTensor | None = None
    visible_gene_random_neg_correct_n: torch.FloatTensor | None = None
    visible_gene_random_neg_n: torch.FloatTensor | None = None
    visible_gene_auc_sum: torch.FloatTensor | None = None
    visible_gene_positive_rank_sum: torch.FloatTensor | None = None
    visible_gene_top1_n: torch.FloatTensor | None = None
    masked_gene_loss: torch.FloatTensor | None = None
    masked_gene_target_n: torch.FloatTensor | None = None
    masked_gene_pos_score_sum: torch.FloatTensor | None = None
    masked_gene_neg_score_sum: torch.FloatTensor | None = None
    masked_gene_pos_correct_n: torch.FloatTensor | None = None
    masked_gene_neg_correct_n: torch.FloatTensor | None = None
    masked_gene_neg_n: torch.FloatTensor | None = None
    masked_gene_same_row_neg_correct_n: torch.FloatTensor | None = None
    masked_gene_same_row_neg_n: torch.FloatTensor | None = None
    masked_gene_support_neg_correct_n: torch.FloatTensor | None = None
    masked_gene_support_neg_n: torch.FloatTensor | None = None
    masked_gene_random_neg_correct_n: torch.FloatTensor | None = None
    masked_gene_random_neg_n: torch.FloatTensor | None = None
    masked_gene_auc_sum: torch.FloatTensor | None = None
    masked_gene_positive_rank_sum: torch.FloatTensor | None = None
    masked_gene_top1_n: torch.FloatTensor | None = None
    rtd_logits: torch.FloatTensor | None = None
    rtd_loss: torch.FloatTensor | None = None
    rtd_correct_n: torch.FloatTensor | None = None
    rtd_target_n: torch.FloatTensor | None = None
    rtd_pos_n: torch.FloatTensor | None = None
    rtd_neg_n: torch.FloatTensor | None = None
    rtd_negative_random_correct_n: torch.FloatTensor | None = None
    rtd_negative_random_n: torch.FloatTensor | None = None
    rtd_positive_unchanged_correct_n: torch.FloatTensor | None = None
    rtd_positive_unchanged_n: torch.FloatTensor | None = None
    rtd_positive_visible_correct_n: torch.FloatTensor | None = None
    rtd_positive_visible_n: torch.FloatTensor | None = None
    rtd_auc_sum: torch.FloatTensor | None = None
    rtd_auc_pair_n: torch.FloatTensor | None = None
    varcov_loss: torch.FloatTensor | None = None
    varcov_metrics: dict[str, torch.FloatTensor] | None = None
    gene_pool_contrastive_loss: torch.FloatTensor | None = None
    view_consistency_loss: torch.FloatTensor | None = None
    view_cosine_sum: torch.FloatTensor | None = None
    view_pair_n: torch.FloatTensor | None = None


class ValueAugmentedBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config: BertConfig):
        saved_attn_impl = str(getattr(config, "attn_implementation", "")).strip().lower()
        if saved_attn_impl in {"eager", "sdpa"}:
            config._attn_implementation = saved_attn_impl
        super().__init__(config)
        self.norm_position = str(getattr(config, "norm_position", "post")).strip().lower()
        if self.norm_position not in {"post", "pre"}:
            raise ValueError(f"unsupported norm_position={self.norm_position}")
        self.position_embedding_alpha = float(getattr(config, "position_embedding_alpha", 1.0))
        self.pre_ln_final_layer_norm = bool(getattr(config, "pre_ln_final_layer_norm", True))
        self.encoder_layout = resolved_encoder_layout(config)
        self.num_loop_steps = resolved_num_loop_steps(config)
        self.num_unique_layers = resolved_num_unique_layers(config)
        self.use_loop_step_embedding = bool(getattr(config, "use_loop_step_embedding", False))
        if self.norm_position == "pre":
            self.bert = PreLayerNormBertModel(config, add_pooling_layer=False)
            self.tie_weights()
        elif self.position_embedding_alpha != 1.0:
            self.bert.embeddings = AlphaScaledBertEmbeddings(config)
            self.bert.embeddings.apply(self._init_weights)
            self.tie_weights()
        self.use_value_input = bool(getattr(config, "use_value_input", False))
        self.use_value_loss = bool(getattr(config, "use_value_loss", False))
        self.fixed_coordinate = bool(getattr(config, "fixed_coordinate", False))
        self.gene_mlm_enabled = bool(getattr(config, "gene_mlm_enabled", True))
        self.cosine_mlm_head = bool(getattr(config, "cosine_mlm_head", False))
        cosine_temperature = float(getattr(config, "cosine_mlm_temperature", 20.0))
        if not math.isfinite(cosine_temperature) or cosine_temperature <= 0.0:
            raise ValueError(f"cosine_mlm_temperature must be positive finite, got {cosine_temperature}")
        cosine_log_scale = torch.tensor(math.log(cosine_temperature), dtype=torch.float32)
        if self.cosine_mlm_head:
            self.cosine_mlm_logit_scale = nn.Parameter(cosine_log_scale)
        else:
            self.register_buffer("cosine_mlm_logit_scale", cosine_log_scale, persistent=False)
        self.fixed_gene_feature_dim = int(getattr(config, "fixed_gene_feature_dim", 0))
        self.fixed_gene_feature_scale = float(getattr(config, "fixed_gene_feature_scale", 0.5))
        self.retrieval_feature_weight = float(getattr(config, "retrieval_feature_weight", 0.4))
        if self.fixed_gene_feature_dim < 0:
            raise ValueError("fixed_gene_feature_dim cannot be negative")
        if not math.isfinite(self.fixed_gene_feature_scale) or self.fixed_gene_feature_scale < 0.0:
            raise ValueError("fixed_gene_feature_scale must be finite and non-negative")
        if not (0.0 <= self.retrieval_feature_weight < 1.0):
            raise ValueError("retrieval_feature_weight must be in [0,1)")
        if self.fixed_gene_feature_dim > 0:
            self.register_buffer(
                "fixed_gene_features",
                torch.zeros(int(config.vocab_size), self.fixed_gene_feature_dim, dtype=torch.float32),
                persistent=True,
            )
            self.fixed_gene_feature_projection = nn.Linear(
                self.fixed_gene_feature_dim,
                int(config.hidden_size),
                bias=False,
            )
            with torch.no_grad():
                self.fixed_gene_feature_projection.weight.zero_()
                identity_dim = min(self.fixed_gene_feature_dim, int(config.hidden_size))
                self.fixed_gene_feature_projection.weight[:identity_dim, :identity_dim].copy_(
                    torch.eye(identity_dim, dtype=self.fixed_gene_feature_projection.weight.dtype)
                )
        else:
            self.register_buffer("fixed_gene_features", torch.empty(0, 0), persistent=False)
            self.fixed_gene_feature_projection = None
        self.gene_mlm_policy = str(getattr(config, "gene_mlm_policy", "bert_801010")).strip().lower()
        (
            self.gene_mlm_mask_replace_prob,
            self.gene_mlm_random_replace_prob,
            self.gene_mlm_unchanged_prob,
        ) = resolve_gene_mlm_corruption_probs(
            gene_mlm_policy=self.gene_mlm_policy,
            mask_replace_prob=getattr(config, "gene_mlm_mask_replace_prob", None),
            random_replace_prob=getattr(config, "gene_mlm_random_replace_prob", None),
            unchanged_prob=getattr(config, "gene_mlm_unchanged_prob", None),
        )
        self.value_target_decoupled = bool(getattr(config, "value_target_decoupled", False))
        self.value_target_probability = float(getattr(config, "value_target_probability", 0.15))
        self.value_target_scope = str(getattr(config, "value_target_scope", "observed_nonzero")).strip().lower()
        self.visible_gene_discrimination_enable = bool(
            getattr(config, "visible_gene_discrimination_enable", False)
        )
        self.visible_gene_loss_weight = float(getattr(config, "visible_gene_loss_weight", 0.03))
        self.masked_gene_discrimination_enable = bool(
            getattr(config, "masked_gene_discrimination_enable", False)
        )
        self.masked_gene_loss_weight = float(getattr(config, "masked_gene_loss_weight", 0.03))
        self.rtd_enable = bool(getattr(config, "rtd_enable", False))
        self.rtd_loss_weight = float(getattr(config, "rtd_loss_weight", 0.0))
        self.rtd_visible_positive_policy = str(
            getattr(config, "rtd_visible_positive_policy", "neg_count_match")
        ).strip().lower()
        if self.rtd_visible_positive_policy not in {"neg_count_match"}:
            raise ValueError(f"unsupported rtd_visible_positive_policy={self.rtd_visible_positive_policy}")
        self.view_consistency_enable = bool(getattr(config, "view_consistency_enable", False))
        self.view_consistency_weight = float(getattr(config, "view_consistency_weight", 0.01))
        self.use_gene_pool_contrastive = bool(getattr(config, "use_gene_pool_contrastive", False))
        self.value_loss_weight = float(getattr(config, "value_loss_weight", 0.1))
        self.use_program_loss = bool(getattr(config, "use_program_loss", False))
        self.program_loss_weight = float(getattr(config, "program_loss_weight", 0.0))
        self.program_dim = int(getattr(config, "program_dim", 64))
        self.program_huber_delta = float(getattr(config, "program_huber_delta", 1.0))
        self.use_gene_loading_loss = bool(getattr(config, "use_gene_loading_loss", False))
        self.gene_loading_loss_weight = float(getattr(config, "gene_loading_loss_weight", 0.0))
        self.gene_loading_dim = int(getattr(config, "gene_loading_dim", 64))
        self.gene_loading_huber_delta = float(getattr(config, "gene_loading_huber_delta", 1.0))
        self.gene_loading_gradient_scope = str(
            getattr(config, "gene_loading_gradient_scope", "contextual")
        ).strip().lower()
        if self.gene_loading_gradient_scope not in {"contextual", "native_embedding"}:
            raise ValueError(
                "unsupported gene_loading_gradient_scope="
                f"{self.gene_loading_gradient_scope}"
            )
        if self.program_dim <= 0:
            raise ValueError(f"program_dim must be positive, got {self.program_dim}")
        if self.program_huber_delta <= 0:
            raise ValueError(f"program_huber_delta must be positive, got {self.program_huber_delta}")
        if self.gene_loading_dim <= 0:
            raise ValueError(f"gene_loading_dim must be positive, got {self.gene_loading_dim}")
        if self.gene_loading_huber_delta <= 0:
            raise ValueError(
                f"gene_loading_huber_delta must be positive, got {self.gene_loading_huber_delta}"
            )
        self.value_mask_ratio = float(getattr(config, "value_mask_ratio", 0.15))
        self.value_nonzero_only = bool(getattr(config, "value_nonzero_only", False))
        self.ordinal_huber_weight = float(getattr(config, "ordinal_huber_weight", 0.0))
        self.ordinal_huber_delta = float(getattr(config, "ordinal_huber_delta", 1.0))
        self.varcov_enable = bool(getattr(config, "varcov_enable", False))
        self.varcov_lambda = float(getattr(config, "varcov_lambda", 0.0))
        self.varcov_layers = normalize_varcov_layers(getattr(config, "varcov_layers", []), int(config.num_hidden_layers))
        self.varcov_positions = str(getattr(config, "varcov_positions", "masked")).strip().lower()
        if self.varcov_positions not in {"masked", "all"}:
            raise ValueError(f"unsupported varcov_positions={self.varcov_positions}")
        if self.varcov_enable and not self.varcov_layers:
            raise ValueError("varcov_enable requires at least one layer in varcov_layers")
        self.n_value_bins = int(getattr(config, "n_value_bins", 64))
        self.value_pad_id = int(getattr(config, "value_pad_id", 0))
        self.value_mask_id = int(getattr(config, "value_mask_id", self.n_value_bins + 1))
        self.value_vocab_size = int(getattr(config, "value_vocab_size", self.n_value_bins + 2))
        self.gene_pool_projection_dim = int(getattr(config, "gene_pool_projection_dim", config.hidden_size))
        self.gene_pool_temperature = float(getattr(config, "gene_pool_temperature", 0.07))
        self.gene_pool_stopgrad_anchor = bool(getattr(config, "gene_pool_stopgrad_anchor", False))
        self.gene_pool_excluded_token_ids = tuple(int(x) for x in getattr(config, "gene_pool_excluded_token_ids", [config.pad_token_id]))
        self.view_pool_excluded_token_ids = tuple(
            int(x)
            for x in getattr(
                config,
                "view_pool_excluded_token_ids",
                self.gene_pool_excluded_token_ids,
            )
        )

        self.value_embeddings = nn.Embedding(self.value_vocab_size, config.hidden_size)
        self.value_cls = ValuePredictionHead(hidden_size=config.hidden_size, value_vocab_size=self.n_value_bins + 1)
        self.rtd_head = nn.Linear(config.hidden_size, 1)
        self.program_head = ProgramPredictionHead(hidden_size=config.hidden_size, program_dim=self.program_dim)
        self.gene_loading_head = ProgramPredictionHead(
            hidden_size=config.hidden_size,
            program_dim=self.gene_loading_dim,
        )
        self.value_embeddings.apply(self._init_weights)
        self.value_cls.apply(self._init_weights)
        self.rtd_head.apply(self._init_weights)
        self.program_head.apply(self._init_weights)
        self.gene_loading_head.apply(self._init_weights)
        if not self.gene_mlm_enabled:
            # Keep tied word embeddings trainable, but disable the standalone MLM transform/bias stack.
            self.cls.predictions.transform.requires_grad_(False)
            self.cls.predictions.bias.requires_grad_(False)
        if not self.use_value_input:
            self.value_embeddings.requires_grad_(False)
        if not self.use_value_loss:
            self.value_cls.requires_grad_(False)
        if not self.rtd_enable or self.rtd_loss_weight == 0.0:
            self.rtd_head.requires_grad_(False)
        if not self.use_program_loss or self.program_loss_weight == 0.0:
            self.program_head.requires_grad_(False)
        if not self.use_gene_loading_loss or self.gene_loading_loss_weight == 0.0:
            self.gene_loading_head.requires_grad_(False)
        self.gene_pool_proj = None
        if self.use_gene_pool_contrastive:
            self.gene_pool_proj = ProjectionHead(config.hidden_size, self.gene_pool_projection_dim)
            self.gene_pool_proj.apply(self._init_weights)

        ignore = set(getattr(self.config, "keys_to_ignore_at_inference", []))
        ignore.update(
            [
                "value_logits",
                "program_pred",
                "gene_loading_pred",
                "mlm_loss",
                "value_loss",
                "value_correct_n",
                "value_masked_n",
                "program_loss",
                "program_target_n",
                "program_abs_error_sum",
                "program_squared_error_sum",
                "program_pred_norm_sum",
                "program_target_norm_sum",
                "gene_loading_loss",
                "gene_loading_target_n",
                "gene_loading_abs_error_sum",
                "gene_loading_squared_error_sum",
                "gene_loading_pred_norm_sum",
                "gene_loading_target_norm_sum",
                "ordinal_huber_loss",
                "ordinal_abs_error_sum",
                "ordinal_squared_error_sum",
                "visible_gene_loss",
                "visible_gene_target_n",
                "visible_gene_pos_score_sum",
                "visible_gene_neg_score_sum",
                "visible_gene_pos_correct_n",
                "visible_gene_neg_correct_n",
                "visible_gene_neg_n",
                "visible_gene_same_row_neg_correct_n",
                "visible_gene_same_row_neg_n",
                "visible_gene_support_neg_correct_n",
                "visible_gene_support_neg_n",
                "visible_gene_random_neg_correct_n",
                "visible_gene_random_neg_n",
                "visible_gene_auc_sum",
                "visible_gene_positive_rank_sum",
                "visible_gene_top1_n",
                "masked_gene_loss",
                "masked_gene_target_n",
                "masked_gene_pos_score_sum",
                "masked_gene_neg_score_sum",
                "masked_gene_pos_correct_n",
                "masked_gene_neg_correct_n",
                "masked_gene_neg_n",
                "masked_gene_same_row_neg_correct_n",
                "masked_gene_same_row_neg_n",
                "masked_gene_support_neg_correct_n",
                "masked_gene_support_neg_n",
                "masked_gene_random_neg_correct_n",
                "masked_gene_random_neg_n",
                "masked_gene_auc_sum",
                "masked_gene_positive_rank_sum",
                "masked_gene_top1_n",
                "rtd_logits",
                "rtd_loss",
                "rtd_correct_n",
                "rtd_target_n",
                "rtd_pos_n",
                "rtd_neg_n",
                "rtd_negative_random_correct_n",
                "rtd_negative_random_n",
                "rtd_positive_unchanged_correct_n",
                "rtd_positive_unchanged_n",
                "rtd_positive_visible_correct_n",
                "rtd_positive_visible_n",
                "rtd_auc_sum",
                "rtd_auc_pair_n",
                "varcov_loss",
                "varcov_metrics",
                "view_consistency_loss",
                "view_cosine_sum",
                "view_pair_n",
            ]
        )
        self.config.keys_to_ignore_at_inference = sorted(ignore)

    def get_token_embedding_layer(self) -> nn.Embedding:
        return self.bert.embeddings.word_embeddings

    def _compute_prediction_scores(self, sequence_output: torch.Tensor) -> torch.Tensor | None:
        if not self.gene_mlm_enabled:
            return None
        if not self.cosine_mlm_head:
            return self.cls(sequence_output)
        transformed = self.cls.predictions.transform(sequence_output)
        hidden_unit = nn.functional.normalize(transformed.float(), p=2.0, dim=-1, eps=1e-6)
        decoder_weight = self.cls.predictions.decoder.weight
        weight_unit = nn.functional.normalize(decoder_weight.float(), p=2.0, dim=-1, eps=1e-6)
        logit_scale = self.cosine_mlm_logit_scale.float().clamp(
            min=math.log(1.0),
            max=math.log(100.0),
        ).exp()
        scores = torch.matmul(hidden_unit, weight_unit.transpose(0, 1)) * logit_scale
        bias = self.cls.predictions.bias
        if bias is not None:
            scores = scores + bias.float()
        return scores.to(dtype=sequence_output.dtype)

    @staticmethod
    def _weighted_token_loss(
        token_loss: torch.Tensor,
        valid_mask: torch.Tensor,
        example_loss_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if token_loss.shape != valid_mask.shape:
            raise ValueError(
                f"token loss/mask shape mismatch: {tuple(token_loss.shape)} != {tuple(valid_mask.shape)}"
            )
        valid = valid_mask.to(dtype=token_loss.dtype)
        if example_loss_weights is None:
            return (token_loss * valid).sum() / valid.sum().clamp_min(1.0)
        weights = example_loss_weights.to(device=token_loss.device, dtype=token_loss.dtype).reshape(-1)
        if int(weights.numel()) != int(token_loss.size(0)):
            raise ValueError(
                f"example_loss_weights must have {token_loss.size(0)} rows, got {weights.numel()}"
            )
        per_row_den = valid.sum(dim=1)
        active = per_row_den.gt(0)
        if not bool(active.any()):
            return token_loss.sum() * 0.0
        per_row_loss = (token_loss * valid).sum(dim=1) / per_row_den.clamp_min(1.0)
        active_weights = weights[active].clamp_min(0.0)
        return (per_row_loss[active] * active_weights).sum() / active_weights.sum().clamp_min(1e-12)

    def _compute_sampled_gene_discrimination(
        self,
        *,
        sequence_output: torch.Tensor,
        row_idx: torch.Tensor | None,
        col_idx: torch.Tensor | None,
        pos_gene_ids: torch.Tensor | None,
        neg_gene_ids: torch.Tensor | None,
        neg_types: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        keys = [
            "loss",
            "target_n",
            "pos_score_sum",
            "neg_score_sum",
            "pos_correct_n",
            "neg_correct_n",
            "neg_n",
            "same_row_neg_correct_n",
            "same_row_neg_n",
            "support_neg_correct_n",
            "support_neg_n",
            "random_neg_correct_n",
            "random_neg_n",
            "auc_sum",
            "positive_rank_sum",
            "top1_n",
        ]
        empty = {key: None for key in keys}
        if (
            row_idx is None
            or col_idx is None
            or pos_gene_ids is None
            or neg_gene_ids is None
            or neg_types is None
        ):
            return empty

        target_n = int(pos_gene_ids.numel())
        if target_n <= 0:
            zero = sequence_output.new_zeros((), dtype=torch.float32)
            return {key: zero for key in keys}

        row_idx = row_idx.to(dtype=torch.long)
        col_idx = col_idx.to(dtype=torch.long)
        pos_gene_ids = pos_gene_ids.to(dtype=torch.long)
        neg_gene_ids = neg_gene_ids.to(dtype=torch.long)
        neg_types = neg_types.to(dtype=torch.long)
        hidden = sequence_output[row_idx, col_idx]
        token_embeddings = self.get_token_embedding_layer().weight
        pos_emb = token_embeddings[pos_gene_ids]
        neg_emb = token_embeddings[neg_gene_ids]
        pos_score = (hidden * pos_emb).sum(dim=-1)
        neg_scores = (hidden.unsqueeze(1) * neg_emb).sum(dim=-1)
        pos_targets = torch.ones_like(pos_score)
        neg_targets = torch.zeros_like(neg_scores)
        pos_bce = nn.functional.binary_cross_entropy_with_logits(pos_score, pos_targets, reduction="mean")
        neg_bce = nn.functional.binary_cross_entropy_with_logits(neg_scores, neg_targets, reduction="mean")
        loss = (0.5 * pos_bce) + (0.5 * neg_bce)

        pos_correct = pos_score.gt(0.0)
        neg_correct = neg_scores.lt(0.0)
        pos_score_expanded = pos_score.unsqueeze(1)
        better = neg_scores.gt(pos_score_expanded)
        ties = neg_scores.eq(pos_score_expanded)
        auc = better.logical_not().to(dtype=hidden.dtype).mean(dim=1) - (
            0.5 * ties.to(dtype=hidden.dtype).mean(dim=1)
        )
        positive_rank = 1.0 + better.sum(dim=1).to(dtype=hidden.dtype) + (
            0.5 * ties.sum(dim=1).to(dtype=hidden.dtype)
        )
        top1 = neg_scores.le(pos_score_expanded).all(dim=1)

        same_row_mask = neg_types.eq(0)
        support_mask = neg_types.eq(1)
        random_mask = neg_types.eq(2)

        def _masked_correct_sum(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mask_f = mask.to(dtype=torch.bool)
            if bool(mask_f.any()):
                corr = neg_correct[mask_f].sum().to(dtype=torch.float32)
                denom = mask_f.sum().to(dtype=torch.float32)
            else:
                zero = pos_score.new_zeros((), dtype=torch.float32)
                corr = zero
                denom = zero
            return corr, denom

        same_row_neg_correct_n, same_row_neg_n = _masked_correct_sum(same_row_mask)
        support_neg_correct_n, support_neg_n = _masked_correct_sum(support_mask)
        random_neg_correct_n, random_neg_n = _masked_correct_sum(random_mask)

        return {
            "loss": loss,
            "target_n": pos_score.new_tensor(float(target_n), dtype=torch.float32),
            "pos_score_sum": pos_score.sum().to(dtype=torch.float32),
            "neg_score_sum": neg_scores.sum().to(dtype=torch.float32),
            "pos_correct_n": pos_correct.sum().to(dtype=torch.float32),
            "neg_correct_n": neg_correct.sum().to(dtype=torch.float32),
            "neg_n": neg_correct.numel() * pos_score.new_ones((), dtype=torch.float32),
            "same_row_neg_correct_n": same_row_neg_correct_n,
            "same_row_neg_n": same_row_neg_n,
            "support_neg_correct_n": support_neg_correct_n,
            "support_neg_n": support_neg_n,
            "random_neg_correct_n": random_neg_correct_n,
            "random_neg_n": random_neg_n,
            "auc_sum": auc.sum().to(dtype=torch.float32),
            "positive_rank_sum": positive_rank.sum().to(dtype=torch.float32),
            "top1_n": top1.sum().to(dtype=torch.float32),
        }

    def _compute_varcov_regularizer(
        self,
        *,
        hidden_states: tuple[torch.Tensor, ...] | None,
        attention_mask: torch.Tensor | None,
        value_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor] | None]:
        if not self.varcov_enable:
            return None, None
        if hidden_states is None or len(hidden_states) == 0:
            return None, None

        zero = hidden_states[-1].new_zeros(())
        if self.varcov_positions == "masked":
            token_mask = (value_labels != -100) if value_labels is not None else None
        else:
            token_mask = (
                attention_mask > 0
                if attention_mask is not None
                else torch.ones(hidden_states[-1].shape[:2], device=hidden_states[-1].device, dtype=torch.bool)
            )

        metrics: dict[str, torch.Tensor] = {}
        layer_losses: list[torch.Tensor] = []
        for layer_idx in self.varcov_layers:
            var_loss = zero
            cov_loss = zero
            layer_loss = zero
            if token_mask is not None:
                active_mask = token_mask.to(dtype=torch.bool)
                if bool(active_mask.any()) and int(active_mask.sum().item()) >= 2:
                    z = hidden_states[layer_idx][active_mask]
                    z = z - z.mean(dim=0, keepdim=True)
                    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
                    var_loss = torch.relu(std.new_tensor(1.0) - std).pow(2).mean()
                    cov = (z.transpose(0, 1) @ z) / float(max(int(z.size(0)) - 1, 1))
                    cov_offdiag = offdiag(cov)
                    cov_loss = (
                        cov_offdiag.pow(2).sum() / float(max(int(cov.size(0)), 1))
                        if cov_offdiag.numel() > 0
                        else zero
                    )
                    layer_loss = var_loss + cov_loss
            layer_losses.append(layer_loss)
            metrics[f"varcov_layer_{layer_idx}"] = layer_loss
            metrics[f"varcov_layer_{layer_idx}_var"] = var_loss
            metrics[f"varcov_layer_{layer_idx}_cov"] = cov_loss

        raw_total = torch.stack(layer_losses).mean() if layer_losses else zero
        weighted_total = raw_total * float(self.varcov_lambda)
        metrics["varcov_total"] = raw_total
        metrics["varcov_weighted"] = weighted_total
        return weighted_total, metrics

    def project_gene_pool(self, pooled: torch.Tensor) -> torch.Tensor:
        if self.gene_pool_proj is None:
            raise ValueError("gene_pool projection head is not initialized")
        projected = self.gene_pool_proj(pooled)
        return nn.functional.normalize(projected, p=2, dim=-1)

    def _mean_pool_original_token_embeddings(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_emb = self.get_token_embedding_layer()(token_ids)
        keep = attention_mask > 0
        for token_id in self.gene_pool_excluded_token_ids:
            keep &= token_ids.ne(int(token_id))
        keep_f = keep.unsqueeze(-1).to(dtype=token_emb.dtype)
        denom = keep_f.sum(dim=1).clamp(min=1.0)
        return (token_emb * keep_f).sum(dim=1) / denom

    def _mean_pool_sequence_hidden(
        self,
        sequence_output: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            keep = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.bool)
        else:
            keep = attention_mask.to(device=input_ids.device).gt(0)
        for token_id in self.view_pool_excluded_token_ids:
            keep &= input_ids.ne(int(token_id))
        keep_f = keep.unsqueeze(-1).to(dtype=sequence_output.dtype)
        denom = keep_f.sum(dim=1).clamp(min=1.0)
        return (sequence_output * keep_f).sum(dim=1) / denom

    def _compute_view_consistency(
        self,
        *,
        sequence_output: torch.Tensor,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        view_input_ids: torch.Tensor | None,
        view_attention_mask: torch.Tensor | None,
        view_value_bins: torch.Tensor | None,
        view_token_type_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        head_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.view_consistency_enable or input_ids is None or view_input_ids is None:
            return None, None, None
        resolved_view_input_ids = view_input_ids
        resolved_view_embeds = None
        if self.use_value_input:
            resolved_view_embeds = self._build_inputs_embeds(
                input_ids=view_input_ids,
                value_bins=view_value_bins,
                token_type_ids=view_token_type_ids,
                position_ids=position_ids,
            )
            resolved_view_input_ids = None

        view_outputs = self.bert(
            input_ids=resolved_view_input_ids,
            attention_mask=view_attention_mask,
            token_type_ids=view_token_type_ids if not self.use_value_input else None,
            position_ids=position_ids if not self.use_value_input else None,
            head_mask=head_mask,
            inputs_embeds=resolved_view_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        pooled = self._mean_pool_sequence_hidden(sequence_output, input_ids, attention_mask)
        view_pooled = self._mean_pool_sequence_hidden(view_outputs[0], view_input_ids, view_attention_mask)
        cosine = nn.functional.cosine_similarity(pooled, view_pooled, dim=-1)
        loss = (1.0 - cosine).mean()
        cosine_sum = cosine.sum().to(dtype=torch.float32)
        pair_n = cosine.new_tensor(float(cosine.numel()), dtype=torch.float32)
        return loss, cosine_sum, pair_n

    def _build_inputs_embeds(
        self,
        *,
        input_ids: torch.Tensor,
        value_bins: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if value_bins is None:
            raise ValueError("value_bins is required when use_value_input=true")
        if input_ids is None:
            raise ValueError("input_ids is required when use_value_input=true")

        input_shape = input_ids.size()
        batch_size, seq_len = input_shape
        device = input_ids.device

        if position_ids is None:
            position_ids = self.bert.embeddings.position_ids[:, :seq_len].to(device)
        if token_type_ids is None:
            if hasattr(self.bert.embeddings, "token_type_ids"):
                buffered = self.bert.embeddings.token_type_ids[:, :seq_len].to(device)
                token_type_ids = buffered.expand(batch_size, seq_len)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embeddings = self.bert.embeddings.word_embeddings(input_ids)
        embeddings = embeddings + self.bert.embeddings.token_type_embeddings(token_type_ids)
        if self.bert.embeddings.position_embedding_type == "absolute":
            embeddings = embeddings + (
                self.position_embedding_alpha * self.bert.embeddings.position_embeddings(position_ids)
            )
        embeddings = embeddings + self.value_embeddings(value_bins)
        embeddings = self.bert.embeddings.LayerNorm(embeddings)
        embeddings = self.bert.embeddings.dropout(embeddings)
        return embeddings

    def _fixed_feature_word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.bert.embeddings.word_embeddings(input_ids)
        if self.fixed_gene_feature_projection is None:
            return embeddings
        fixed = self.fixed_gene_features[input_ids].to(dtype=embeddings.dtype)
        return embeddings + (
            self.fixed_gene_feature_projection(fixed) * float(self.fixed_gene_feature_scale)
        )

    def _effective_gene_embedding_weight(self) -> torch.Tensor:
        base = self.bert.embeddings.word_embeddings.weight
        if self.fixed_gene_feature_projection is None:
            return base
        fixed = self.fixed_gene_features.to(device=base.device, dtype=base.dtype)
        return base + (
            self.fixed_gene_feature_projection(fixed) * float(self.fixed_gene_feature_scale)
        )

    def _export_retrieval_embeddings(self, output_dir: Path) -> dict[str, Any] | None:
        if self.fixed_gene_feature_projection is None:
            return None
        weight = float(self.retrieval_feature_weight)
        base = self.bert.embeddings.word_embeddings.weight.detach().to(dtype=torch.float32)
        feature = self.fixed_gene_features.detach().to(device=base.device, dtype=torch.float32)
        base = nn.functional.normalize(base, p=2, dim=-1)
        feature = nn.functional.normalize(feature, p=2, dim=-1)
        retrieval = torch.cat(
            [math.sqrt(1.0 - weight) * base, math.sqrt(weight) * feature],
            dim=-1,
        ).cpu().numpy().astype(np.float32, copy=False)
        retrieval_path = output_dir / "retrieval_embeddings.npy"
        np.save(retrieval_path, retrieval)
        metadata = {
            "path": str(retrieval_path),
            "shape": list(retrieval.shape),
            "dtype": str(retrieval.dtype),
            "retrieval_feature_weight": weight,
            "uses_downstream_labels": False,
            "uses_public_weights": False,
        }
        write_json(output_dir / "retrieval_embeddings_manifest.json", metadata)
        return metadata

    def export_plain_mlm(self, output_dir: str | Path) -> dict[str, list[str]]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if (
            self.norm_position == "pre"
            or self.position_embedding_alpha != 1.0
            or self.encoder_layout != "unique"
            or self.use_value_input
            or not self.gene_mlm_enabled
        ):
            reasons = [
                f"norm_position={self.norm_position}",
                f"position_embedding_alpha={self.position_embedding_alpha}",
            ]
            if self.encoder_layout != "unique":
                reasons.extend(
                    [
                        f"encoder_layout={self.encoder_layout}",
                        f"num_loop_steps={self.num_loop_steps}",
                        f"num_unique_layers={self.num_unique_layers}",
                        f"use_loop_step_embedding={self.use_loop_step_embedding}",
                    ]
                )
            if self.use_value_input:
                reasons.append("use_value_input=True")
            if not self.gene_mlm_enabled:
                reasons.append("gene_mlm_enabled=False")
            self.save_pretrained(str(out_dir))
            self._export_retrieval_embeddings(out_dir)
            return {
                "exported_plain_mlm": False,
                "custom_model_saved": True,
                "reason": reasons,
                "missing_keys": [],
                "unexpected_keys": [],
            }

        plain_config = BertConfig.from_dict(self.config.to_dict())
        plain_config.architectures = ["BertForMaskedLM"]
        plain_config.cosine_mlm_head = False
        plain_model = BertForMaskedLM(plain_config)
        filtered_state = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("value_embeddings.")
            and not key.startswith("value_cls.")
            and not key.startswith("rtd_head.")
            and not key.startswith("program_head.")
            and not key.startswith("gene_loading_head.")
            and not key.startswith("gene_pool_proj.")
            and not key.startswith("fixed_gene_feature_projection.")
            and key != "fixed_gene_features"
            and key != "cosine_mlm_logit_scale"
        }
        incompatible = plain_model.load_state_dict(filtered_state, strict=False)
        if self.fixed_gene_feature_projection is not None:
            effective_weight = self._effective_gene_embedding_weight().detach().to(
                device=plain_model.get_input_embeddings().weight.device,
                dtype=plain_model.get_input_embeddings().weight.dtype,
            )
            with torch.no_grad():
                plain_model.get_input_embeddings().weight.copy_(effective_weight)
            plain_model.tie_weights()
        plain_model.save_pretrained(str(out_dir))
        self._export_retrieval_embeddings(out_dir)
        return {
            "exported_plain_mlm": True,
            "custom_model_saved": False,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        value_bins: torch.Tensor | None = None,
        value_labels: torch.Tensor | None = None,
        program_targets: torch.Tensor | None = None,
        program_target_mask: torch.Tensor | None = None,
        gene_loading_targets: torch.Tensor | None = None,
        gene_loading_target_mask: torch.Tensor | None = None,
        gene_loading_original_input_ids: torch.Tensor | None = None,
        gene_loading_original_attention_mask: torch.Tensor | None = None,
        visible_gene_disc_row_idx: torch.Tensor | None = None,
        visible_gene_disc_col_idx: torch.Tensor | None = None,
        visible_gene_disc_pos_gene_ids: torch.Tensor | None = None,
        visible_gene_disc_neg_gene_ids: torch.Tensor | None = None,
        visible_gene_disc_neg_types: torch.Tensor | None = None,
        masked_gene_disc_row_idx: torch.Tensor | None = None,
        masked_gene_disc_col_idx: torch.Tensor | None = None,
        masked_gene_disc_pos_gene_ids: torch.Tensor | None = None,
        masked_gene_disc_neg_gene_ids: torch.Tensor | None = None,
        masked_gene_disc_neg_types: torch.Tensor | None = None,
        rtd_labels: torch.Tensor | None = None,
        rtd_token_types: torch.Tensor | None = None,
        gene_pool_original_input_ids: torch.Tensor | None = None,
        gene_pool_original_attention_mask: torch.Tensor | None = None,
        gene_pool_anchor_input_ids: torch.Tensor | None = None,
        gene_pool_anchor_attention_mask: torch.Tensor | None = None,
        gene_pool_positive_indices: torch.Tensor | None = None,
        view_input_ids: torch.Tensor | None = None,
        view_attention_mask: torch.Tensor | None = None,
        view_token_type_ids: torch.Tensor | None = None,
        view_value_bins: torch.Tensor | None = None,
        example_loss_weights: torch.Tensor | None = None,
    ) -> ValueAugmentedMaskedLMOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        need_hidden_states = bool(output_hidden_states) or self.varcov_enable

        resolved_inputs_embeds = inputs_embeds
        resolved_input_ids = input_ids
        if self.fixed_gene_feature_projection is not None and not self.use_value_input:
            if input_ids is None:
                raise ValueError("input_ids is required when fixed gene features are enabled")
            resolved_inputs_embeds = self._fixed_feature_word_embeddings(input_ids)
            resolved_input_ids = None
        elif self.use_value_input:
            resolved_inputs_embeds = self._build_inputs_embeds(
                input_ids=input_ids,
                value_bins=value_bins,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            resolved_input_ids = None

        outputs = self.bert(
            input_ids=resolved_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if not self.use_value_input else None,
            position_ids=position_ids if not self.use_value_input else None,
            head_mask=head_mask,
            inputs_embeds=resolved_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=need_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self._compute_prediction_scores(sequence_output)

        mlm_loss = None
        if labels is not None and prediction_scores is not None:
            valid_mlm = labels != -100
            if bool(valid_mlm.any()):
                token_loss = nn.functional.cross_entropy(
                    prediction_scores.reshape(-1, self.config.vocab_size),
                    labels.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).reshape_as(labels)
                mlm_loss = self._weighted_token_loss(token_loss, valid_mlm, example_loss_weights)

        value_logits = None
        value_loss = None
        value_correct_n = None
        value_masked_n = None
        program_pred = None
        program_loss = None
        program_target_n = None
        program_abs_error_sum = None
        program_squared_error_sum = None
        program_pred_norm_sum = None
        program_target_norm_sum = None
        gene_loading_pred = None
        gene_loading_loss = None
        gene_loading_target_n = None
        gene_loading_abs_error_sum = None
        gene_loading_squared_error_sum = None
        gene_loading_pred_norm_sum = None
        gene_loading_target_norm_sum = None
        ordinal_huber_loss = None
        ordinal_abs_error_sum = None
        ordinal_squared_error_sum = None
        visible_gene_loss = None
        visible_gene_target_n = None
        visible_gene_pos_score_sum = None
        visible_gene_neg_score_sum = None
        visible_gene_pos_correct_n = None
        visible_gene_neg_correct_n = None
        visible_gene_neg_n = None
        visible_gene_same_row_neg_correct_n = None
        visible_gene_same_row_neg_n = None
        visible_gene_support_neg_correct_n = None
        visible_gene_support_neg_n = None
        visible_gene_random_neg_correct_n = None
        visible_gene_random_neg_n = None
        visible_gene_auc_sum = None
        visible_gene_positive_rank_sum = None
        visible_gene_top1_n = None
        masked_gene_loss = None
        masked_gene_target_n = None
        masked_gene_pos_score_sum = None
        masked_gene_neg_score_sum = None
        masked_gene_pos_correct_n = None
        masked_gene_neg_correct_n = None
        masked_gene_neg_n = None
        masked_gene_same_row_neg_correct_n = None
        masked_gene_same_row_neg_n = None
        masked_gene_support_neg_correct_n = None
        masked_gene_support_neg_n = None
        masked_gene_random_neg_correct_n = None
        masked_gene_random_neg_n = None
        masked_gene_auc_sum = None
        masked_gene_positive_rank_sum = None
        masked_gene_top1_n = None
        rtd_logits = None
        rtd_loss = None
        rtd_correct_n = None
        rtd_target_n = None
        rtd_pos_n = None
        rtd_neg_n = None
        rtd_negative_random_correct_n = None
        rtd_negative_random_n = None
        rtd_positive_unchanged_correct_n = None
        rtd_positive_unchanged_n = None
        rtd_positive_visible_correct_n = None
        rtd_positive_visible_n = None
        rtd_auc_sum = None
        rtd_auc_pair_n = None
        if self.use_value_loss:
            value_logits = self.value_cls(sequence_output)
            if value_labels is not None:
                valid = value_labels != -100
                if bool(valid.any()):
                    token_loss = nn.functional.cross_entropy(
                        value_logits.reshape(-1, value_logits.size(-1)),
                        value_labels.reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).reshape_as(value_labels)
                    value_loss = self._weighted_token_loss(token_loss, valid, example_loss_weights)
                    preds = value_logits.argmax(dim=-1)
                    value_correct_n = (preds[valid] == value_labels[valid]).sum().to(dtype=torch.float32)
                    value_masked_n = valid.sum().to(dtype=torch.float32)
                    valid_logits = value_logits[valid]
                    probs = torch.softmax(valid_logits, dim=-1)
                    class_ids = torch.arange(valid_logits.size(-1), device=valid_logits.device, dtype=probs.dtype)
                    expected_bin = (probs * class_ids.unsqueeze(0)).sum(dim=-1)
                    target_bin = value_labels[valid].to(dtype=expected_bin.dtype)
                    delta = float(max(self.ordinal_huber_delta, 1e-8))
                    ordinal_huber_loss = nn.functional.huber_loss(
                        expected_bin,
                        target_bin,
                        reduction="mean",
                        delta=delta,
                    )
                    abs_err = torch.abs(expected_bin - target_bin)
                    ordinal_abs_error_sum = abs_err.sum().to(dtype=torch.float32)
                    ordinal_squared_error_sum = (abs_err.square()).sum().to(dtype=torch.float32)
                else:
                    zero = sequence_output.new_zeros(())
                    value_loss = zero
                    value_correct_n = zero
                    value_masked_n = zero
                    ordinal_huber_loss = zero
                    ordinal_abs_error_sum = zero
                    ordinal_squared_error_sum = zero

        if self.use_program_loss and program_targets is not None:
            pooled = self._mean_pool_sequence_hidden(sequence_output, input_ids, attention_mask)
            program_pred = self.program_head(pooled)
            targets = program_targets.to(device=program_pred.device, dtype=program_pred.dtype)
            if targets.ndim != 2 or int(targets.size(-1)) != int(self.program_dim):
                raise ValueError(
                    f"program_targets must have shape [batch,{self.program_dim}], got {tuple(targets.shape)}"
                )
            if program_target_mask is None:
                target_mask = torch.isfinite(targets).all(dim=-1)
            else:
                target_mask = program_target_mask.to(device=program_pred.device).bool().reshape(-1)
                target_mask = target_mask & torch.isfinite(targets).all(dim=-1)
            if bool(target_mask.any()):
                pred_valid = program_pred[target_mask]
                target_valid = targets[target_mask]
                program_loss = nn.functional.huber_loss(
                    pred_valid,
                    target_valid,
                    reduction="mean",
                    delta=float(self.program_huber_delta),
                )
                abs_err = torch.abs(pred_valid - target_valid)
                program_abs_error_sum = abs_err.sum().to(dtype=torch.float32)
                program_squared_error_sum = abs_err.square().sum().to(dtype=torch.float32)
                program_pred_norm_sum = torch.linalg.vector_norm(pred_valid, dim=-1).sum().to(dtype=torch.float32)
                program_target_norm_sum = torch.linalg.vector_norm(target_valid, dim=-1).sum().to(dtype=torch.float32)
                program_target_n = target_mask.sum().to(dtype=torch.float32)
            else:
                zero = sequence_output.new_zeros(())
                program_loss = zero
                program_abs_error_sum = zero
                program_squared_error_sum = zero
                program_pred_norm_sum = zero
                program_target_norm_sum = zero
                program_target_n = zero

        if self.use_gene_loading_loss and gene_loading_targets is not None:
            if self.gene_loading_gradient_scope == "native_embedding":
                native_input_ids = gene_loading_original_input_ids
                if native_input_ids is None:
                    raise ValueError(
                        "native_embedding gene loading requires gene_loading_original_input_ids"
                    )
                native_embeddings = self.bert.embeddings.word_embeddings(native_input_ids)
                gene_loading_pred = self.gene_loading_head(native_embeddings)
                loading_attention_mask = (
                    gene_loading_original_attention_mask
                    if gene_loading_original_attention_mask is not None
                    else attention_mask
                )
            else:
                gene_loading_pred = self.gene_loading_head(sequence_output)
                loading_attention_mask = attention_mask
            targets = gene_loading_targets.to(device=gene_loading_pred.device, dtype=gene_loading_pred.dtype)
            if targets.ndim != 3 or int(targets.size(-1)) != int(self.gene_loading_dim):
                raise ValueError(
                    "gene_loading_targets must have shape "
                    f"[batch,seq,{self.gene_loading_dim}], got {tuple(targets.shape)}"
                )
            finite_mask = torch.isfinite(targets).all(dim=-1)
            if loading_attention_mask is not None:
                finite_mask = finite_mask & loading_attention_mask.to(device=finite_mask.device).bool()
            if gene_loading_target_mask is None:
                target_mask = finite_mask
            else:
                target_mask = gene_loading_target_mask.to(device=gene_loading_pred.device).bool()
                if target_mask.shape != targets.shape[:2]:
                    raise ValueError(
                        "gene_loading_target_mask must have shape "
                        f"{tuple(targets.shape[:2])}, got {tuple(target_mask.shape)}"
                    )
                target_mask = target_mask & finite_mask
            if bool(target_mask.any()):
                pred_valid = gene_loading_pred[target_mask]
                target_valid = targets[target_mask]
                gene_loading_loss = nn.functional.huber_loss(
                    pred_valid,
                    target_valid,
                    reduction="mean",
                    delta=float(self.gene_loading_huber_delta),
                )
                abs_err = torch.abs(pred_valid - target_valid)
                gene_loading_abs_error_sum = abs_err.sum().to(dtype=torch.float32)
                gene_loading_squared_error_sum = abs_err.square().sum().to(dtype=torch.float32)
                gene_loading_pred_norm_sum = torch.linalg.vector_norm(pred_valid, dim=-1).sum().to(dtype=torch.float32)
                gene_loading_target_norm_sum = torch.linalg.vector_norm(target_valid, dim=-1).sum().to(dtype=torch.float32)
                gene_loading_target_n = target_mask.sum().to(dtype=torch.float32)
            else:
                zero = sequence_output.new_zeros(())
                gene_loading_loss = zero
                gene_loading_abs_error_sum = zero
                gene_loading_squared_error_sum = zero
                gene_loading_pred_norm_sum = zero
                gene_loading_target_norm_sum = zero
                gene_loading_target_n = zero

        if (
            self.visible_gene_discrimination_enable
            and visible_gene_disc_row_idx is not None
            and visible_gene_disc_col_idx is not None
            and visible_gene_disc_pos_gene_ids is not None
            and visible_gene_disc_neg_gene_ids is not None
            and visible_gene_disc_neg_types is not None
        ):
            visible_stats = self._compute_sampled_gene_discrimination(
                sequence_output=sequence_output,
                row_idx=visible_gene_disc_row_idx,
                col_idx=visible_gene_disc_col_idx,
                pos_gene_ids=visible_gene_disc_pos_gene_ids,
                neg_gene_ids=visible_gene_disc_neg_gene_ids,
                neg_types=visible_gene_disc_neg_types,
            )
            visible_gene_loss = visible_stats["loss"]
            visible_gene_target_n = visible_stats["target_n"]
            visible_gene_pos_score_sum = visible_stats["pos_score_sum"]
            visible_gene_neg_score_sum = visible_stats["neg_score_sum"]
            visible_gene_pos_correct_n = visible_stats["pos_correct_n"]
            visible_gene_neg_correct_n = visible_stats["neg_correct_n"]
            visible_gene_neg_n = visible_stats["neg_n"]
            visible_gene_same_row_neg_correct_n = visible_stats["same_row_neg_correct_n"]
            visible_gene_same_row_neg_n = visible_stats["same_row_neg_n"]
            visible_gene_support_neg_correct_n = visible_stats["support_neg_correct_n"]
            visible_gene_support_neg_n = visible_stats["support_neg_n"]
            visible_gene_random_neg_correct_n = visible_stats["random_neg_correct_n"]
            visible_gene_random_neg_n = visible_stats["random_neg_n"]
            visible_gene_auc_sum = visible_stats["auc_sum"]
            visible_gene_positive_rank_sum = visible_stats["positive_rank_sum"]
            visible_gene_top1_n = visible_stats["top1_n"]

        if (
            self.masked_gene_discrimination_enable
            and masked_gene_disc_row_idx is not None
            and masked_gene_disc_col_idx is not None
            and masked_gene_disc_pos_gene_ids is not None
            and masked_gene_disc_neg_gene_ids is not None
            and masked_gene_disc_neg_types is not None
        ):
            masked_stats = self._compute_sampled_gene_discrimination(
                sequence_output=sequence_output,
                row_idx=masked_gene_disc_row_idx,
                col_idx=masked_gene_disc_col_idx,
                pos_gene_ids=masked_gene_disc_pos_gene_ids,
                neg_gene_ids=masked_gene_disc_neg_gene_ids,
                neg_types=masked_gene_disc_neg_types,
            )
            masked_gene_loss = masked_stats["loss"]
            masked_gene_target_n = masked_stats["target_n"]
            masked_gene_pos_score_sum = masked_stats["pos_score_sum"]
            masked_gene_neg_score_sum = masked_stats["neg_score_sum"]
            masked_gene_pos_correct_n = masked_stats["pos_correct_n"]
            masked_gene_neg_correct_n = masked_stats["neg_correct_n"]
            masked_gene_neg_n = masked_stats["neg_n"]
            masked_gene_same_row_neg_correct_n = masked_stats["same_row_neg_correct_n"]
            masked_gene_same_row_neg_n = masked_stats["same_row_neg_n"]
            masked_gene_support_neg_correct_n = masked_stats["support_neg_correct_n"]
            masked_gene_support_neg_n = masked_stats["support_neg_n"]
            masked_gene_random_neg_correct_n = masked_stats["random_neg_correct_n"]
            masked_gene_random_neg_n = masked_stats["random_neg_n"]
            masked_gene_auc_sum = masked_stats["auc_sum"]
            masked_gene_positive_rank_sum = masked_stats["positive_rank_sum"]
            masked_gene_top1_n = masked_stats["top1_n"]

        if self.rtd_enable:
            rtd_logits = self.rtd_head(sequence_output).squeeze(-1)
            if rtd_labels is not None:
                valid = rtd_labels != -100
                if bool(valid.any()):
                    scores = rtd_logits[valid]
                    targets = rtd_labels[valid].to(dtype=scores.dtype)
                    token_loss = nn.functional.binary_cross_entropy_with_logits(
                        rtd_logits,
                        rtd_labels.clamp_min(0).to(dtype=rtd_logits.dtype),
                        reduction="none",
                    )
                    rtd_loss = self._weighted_token_loss(token_loss, valid, example_loss_weights)
                    pred_is_real = scores.ge(0.0)
                    target_is_real = targets.ge(0.5)
                    correct = pred_is_real.eq(target_is_real)
                    pos_mask = target_is_real
                    neg_mask = ~target_is_real
                    rtd_correct_n = correct.sum().to(dtype=torch.float32)
                    rtd_target_n = valid.sum().to(dtype=torch.float32)
                    rtd_pos_n = pos_mask.sum().to(dtype=torch.float32)
                    rtd_neg_n = neg_mask.sum().to(dtype=torch.float32)

                    if rtd_token_types is not None:
                        token_types = rtd_token_types[valid].to(dtype=torch.long)
                    else:
                        token_types = torch.full_like(rtd_labels[valid], -100, dtype=torch.long)

                    negative_random_mask = token_types.eq(0)
                    positive_unchanged_mask = token_types.eq(1)
                    positive_visible_mask = token_types.eq(2)

                    zero = scores.new_zeros((), dtype=torch.float32)
                    if bool(negative_random_mask.any()):
                        rtd_negative_random_correct_n = correct[negative_random_mask].sum().to(dtype=torch.float32)
                        rtd_negative_random_n = negative_random_mask.sum().to(dtype=torch.float32)
                    else:
                        rtd_negative_random_correct_n = zero
                        rtd_negative_random_n = zero
                    if bool(positive_unchanged_mask.any()):
                        rtd_positive_unchanged_correct_n = correct[positive_unchanged_mask].sum().to(dtype=torch.float32)
                        rtd_positive_unchanged_n = positive_unchanged_mask.sum().to(dtype=torch.float32)
                    else:
                        rtd_positive_unchanged_correct_n = zero
                        rtd_positive_unchanged_n = zero
                    if bool(positive_visible_mask.any()):
                        rtd_positive_visible_correct_n = correct[positive_visible_mask].sum().to(dtype=torch.float32)
                        rtd_positive_visible_n = positive_visible_mask.sum().to(dtype=torch.float32)
                    else:
                        rtd_positive_visible_correct_n = zero
                        rtd_positive_visible_n = zero

                    pos_scores = scores[pos_mask]
                    neg_scores = scores[neg_mask]
                    if int(pos_scores.numel()) > 0 and int(neg_scores.numel()) > 0:
                        comparisons = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                        auc = comparisons.gt(0.0).to(dtype=scores.dtype).mean()
                        auc = auc + (0.5 * comparisons.eq(0.0).to(dtype=scores.dtype).mean())
                        pair_n = pos_scores.new_tensor(float(pos_scores.numel() * neg_scores.numel()), dtype=torch.float32)
                        rtd_auc_sum = (auc.to(dtype=torch.float32) * pair_n).to(dtype=torch.float32)
                        rtd_auc_pair_n = pair_n
                    else:
                        rtd_auc_sum = zero
                        rtd_auc_pair_n = zero
                else:
                    zero = rtd_logits.sum().to(dtype=torch.float32) * 0.0
                    rtd_loss = zero
                    rtd_correct_n = zero
                    rtd_target_n = zero
                    rtd_pos_n = zero
                    rtd_neg_n = zero
                    rtd_negative_random_correct_n = zero
                    rtd_negative_random_n = zero
                    rtd_positive_unchanged_correct_n = zero
                    rtd_positive_unchanged_n = zero
                    rtd_positive_visible_correct_n = zero
                    rtd_positive_visible_n = zero
                    rtd_auc_sum = zero
                    rtd_auc_pair_n = zero

        if self.rtd_enable and rtd_loss is None:
            rtd_loss = rtd_logits.sum().to(dtype=torch.float32) * 0.0

        hidden_state_tuple = getattr(outputs, "hidden_states", None)
        if hidden_state_tuple is None and need_hidden_states and isinstance(outputs, tuple) and len(outputs) > 1:
            candidate = outputs[1]
            if isinstance(candidate, tuple):
                hidden_state_tuple = candidate

        varcov_loss = None
        varcov_metrics = None
        if self.varcov_enable:
            varcov_loss, varcov_metrics = self._compute_varcov_regularizer(
                hidden_states=hidden_state_tuple,
                attention_mask=attention_mask,
                value_labels=value_labels,
            )

        view_consistency_loss = None
        view_cosine_sum = None
        view_pair_n = None
        if self.view_consistency_enable and view_input_ids is not None:
            view_consistency_loss, view_cosine_sum, view_pair_n = self._compute_view_consistency(
                sequence_output=sequence_output,
                input_ids=input_ids,
                attention_mask=attention_mask,
                view_input_ids=view_input_ids,
                view_attention_mask=view_attention_mask,
                view_value_bins=view_value_bins,
                view_token_type_ids=view_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
            )

        total_loss = mlm_loss
        if value_loss is not None:
            value_objective = value_loss
            if ordinal_huber_loss is not None and self.ordinal_huber_weight != 0.0:
                value_objective = value_objective + (
                    ordinal_huber_loss * float(self.ordinal_huber_weight)
                )
            weighted_value = value_objective * self.value_loss_weight
            total_loss = weighted_value if total_loss is None else total_loss + weighted_value
        elif ordinal_huber_loss is not None and self.ordinal_huber_weight != 0.0:
            weighted_ordinal = ordinal_huber_loss * (self.value_loss_weight * float(self.ordinal_huber_weight))
            total_loss = weighted_ordinal if total_loss is None else total_loss + weighted_ordinal
        if program_loss is not None:
            weighted_program = program_loss * float(self.program_loss_weight)
            total_loss = weighted_program if total_loss is None else total_loss + weighted_program
        if gene_loading_loss is not None:
            weighted_gene_loading = gene_loading_loss * float(self.gene_loading_loss_weight)
            total_loss = weighted_gene_loading if total_loss is None else total_loss + weighted_gene_loading
        if varcov_loss is not None:
            total_loss = varcov_loss if total_loss is None else total_loss + varcov_loss
        if visible_gene_loss is not None:
            weighted_visible_gene = visible_gene_loss * float(self.visible_gene_loss_weight)
            total_loss = weighted_visible_gene if total_loss is None else total_loss + weighted_visible_gene
        if masked_gene_loss is not None:
            weighted_masked_gene = masked_gene_loss * float(self.masked_gene_loss_weight)
            total_loss = weighted_masked_gene if total_loss is None else total_loss + weighted_masked_gene
        if rtd_loss is not None:
            weighted_rtd = rtd_loss * float(self.rtd_loss_weight)
            total_loss = weighted_rtd if total_loss is None else total_loss + weighted_rtd
        if view_consistency_loss is not None:
            weighted_view = view_consistency_loss * float(self.view_consistency_weight)
            total_loss = weighted_view if total_loss is None else total_loss + weighted_view
        if total_loss is None:
            total_loss = sequence_output.sum() * 0.0
        elif torch.is_tensor(total_loss) and not total_loss.requires_grad:
            total_loss = total_loss + (sequence_output.sum() * 0.0)

        gene_pool_contrastive_loss = None
        if (
            self.gene_pool_proj is not None
            and gene_pool_original_input_ids is not None
            and gene_pool_original_attention_mask is not None
            and gene_pool_anchor_input_ids is not None
            and gene_pool_anchor_attention_mask is not None
            and gene_pool_positive_indices is not None
        ):
            query_pooled = self._mean_pool_original_token_embeddings(
                gene_pool_original_input_ids,
                gene_pool_original_attention_mask,
            )
            anchor_pooled = self._mean_pool_original_token_embeddings(
                gene_pool_anchor_input_ids,
                gene_pool_anchor_attention_mask,
            )
            query_z = self.project_gene_pool(query_pooled)
            anchor_z = self.project_gene_pool(anchor_pooled)
            if self.gene_pool_stopgrad_anchor:
                anchor_z = anchor_z.detach()
            gene_pool_contrastive_loss = query_to_anchor_contrastive_loss(
                query_z=query_z,
                anchor_z=anchor_z,
                positive_indices=gene_pool_positive_indices,
                temperature=self.gene_pool_temperature,
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ValueAugmentedMaskedLMOutput(
            loss=total_loss,
            logits=prediction_scores,
            hidden_states=hidden_state_tuple,
            attentions=outputs.attentions,
            value_logits=value_logits,
            program_pred=program_pred,
            gene_loading_pred=gene_loading_pred,
            mlm_loss=mlm_loss,
            value_loss=value_loss,
            value_correct_n=value_correct_n,
            value_masked_n=value_masked_n,
            program_loss=program_loss,
            program_target_n=program_target_n,
            program_abs_error_sum=program_abs_error_sum,
            program_squared_error_sum=program_squared_error_sum,
            program_pred_norm_sum=program_pred_norm_sum,
            program_target_norm_sum=program_target_norm_sum,
            gene_loading_loss=gene_loading_loss,
            gene_loading_target_n=gene_loading_target_n,
            gene_loading_abs_error_sum=gene_loading_abs_error_sum,
            gene_loading_squared_error_sum=gene_loading_squared_error_sum,
            gene_loading_pred_norm_sum=gene_loading_pred_norm_sum,
            gene_loading_target_norm_sum=gene_loading_target_norm_sum,
            ordinal_huber_loss=ordinal_huber_loss,
            ordinal_abs_error_sum=ordinal_abs_error_sum,
            ordinal_squared_error_sum=ordinal_squared_error_sum,
            visible_gene_loss=visible_gene_loss,
            visible_gene_target_n=visible_gene_target_n,
            visible_gene_pos_score_sum=visible_gene_pos_score_sum,
            visible_gene_neg_score_sum=visible_gene_neg_score_sum,
            visible_gene_pos_correct_n=visible_gene_pos_correct_n,
            visible_gene_neg_correct_n=visible_gene_neg_correct_n,
            visible_gene_neg_n=visible_gene_neg_n,
            visible_gene_same_row_neg_correct_n=visible_gene_same_row_neg_correct_n,
            visible_gene_same_row_neg_n=visible_gene_same_row_neg_n,
            visible_gene_support_neg_correct_n=visible_gene_support_neg_correct_n,
            visible_gene_support_neg_n=visible_gene_support_neg_n,
            visible_gene_random_neg_correct_n=visible_gene_random_neg_correct_n,
            visible_gene_random_neg_n=visible_gene_random_neg_n,
            visible_gene_auc_sum=visible_gene_auc_sum,
            visible_gene_positive_rank_sum=visible_gene_positive_rank_sum,
            visible_gene_top1_n=visible_gene_top1_n,
            masked_gene_loss=masked_gene_loss,
            masked_gene_target_n=masked_gene_target_n,
            masked_gene_pos_score_sum=masked_gene_pos_score_sum,
            masked_gene_neg_score_sum=masked_gene_neg_score_sum,
            masked_gene_pos_correct_n=masked_gene_pos_correct_n,
            masked_gene_neg_correct_n=masked_gene_neg_correct_n,
            masked_gene_neg_n=masked_gene_neg_n,
            masked_gene_same_row_neg_correct_n=masked_gene_same_row_neg_correct_n,
            masked_gene_same_row_neg_n=masked_gene_same_row_neg_n,
            masked_gene_support_neg_correct_n=masked_gene_support_neg_correct_n,
            masked_gene_support_neg_n=masked_gene_support_neg_n,
            masked_gene_random_neg_correct_n=masked_gene_random_neg_correct_n,
            masked_gene_random_neg_n=masked_gene_random_neg_n,
            masked_gene_auc_sum=masked_gene_auc_sum,
            masked_gene_positive_rank_sum=masked_gene_positive_rank_sum,
            masked_gene_top1_n=masked_gene_top1_n,
            rtd_logits=rtd_logits,
            rtd_loss=rtd_loss,
            rtd_correct_n=rtd_correct_n,
            rtd_target_n=rtd_target_n,
            rtd_pos_n=rtd_pos_n,
            rtd_neg_n=rtd_neg_n,
            rtd_negative_random_correct_n=rtd_negative_random_correct_n,
            rtd_negative_random_n=rtd_negative_random_n,
            rtd_positive_unchanged_correct_n=rtd_positive_unchanged_correct_n,
            rtd_positive_unchanged_n=rtd_positive_unchanged_n,
            rtd_positive_visible_correct_n=rtd_positive_visible_correct_n,
            rtd_positive_visible_n=rtd_positive_visible_n,
            rtd_auc_sum=rtd_auc_sum,
            rtd_auc_pair_n=rtd_auc_pair_n,
            varcov_loss=varcov_loss,
            varcov_metrics=varcov_metrics,
            gene_pool_contrastive_loss=gene_pool_contrastive_loss,
            view_consistency_loss=view_consistency_loss,
            view_cosine_sum=view_cosine_sum,
            view_pair_n=view_pair_n,
        )


class FixedLengthMLMCollator:
    def __init__(
        self,
        token_dictionary: dict[str, int],
        max_len: int,
        mlm_probability: float = 0.15,
        *,
        mask_sampling: str = "uniform",
        sampling_stats_npz: str | Path | None = None,
        mask_frequency_exponent: float = 0.25,
        mask_rank_bands: int = 4,
        gene_mlm_policy: str = "bert_801010",
        gene_mlm_mask_replace_prob: float | None = None,
        gene_mlm_random_replace_prob: float | None = None,
        hard_random_replacement: bool = False,
        dual_mask_views: bool = False,
        dataset_balance_exponent: float = 0.0,
        dataset_weight_min: float = 0.5,
        dataset_weight_max: float = 2.0,
        gene_mlm_enabled: bool = True,
        use_value_input: bool = False,
        use_value_loss: bool = False,
        value_pad_id: int = 0,
        value_mask_id: int = 65,
        value_target_decoupled: bool = False,
        value_target_probability: float = 0.15,
        value_target_scope: str = "observed_nonzero",
        gene_support_npy: str | Path | None = None,
        visible_gene_discrimination_enable: bool = False,
        visible_gene_disc_probability: float = 0.15,
        visible_gene_disc_negative_k: int = 128,
        visible_gene_disc_overlap_policy: str = "disjoint",
        visible_gene_disc_same_row_min_distance: int = 20,
        visible_gene_disc_same_row_max_distance: int = 300,
        visible_gene_disc_support_log_tolerance: float = 0.25,
        visible_gene_disc_support_log_tolerance_fallback: float = 0.5,
        visible_gene_disc_same_row_n: int = 51,
        visible_gene_disc_support_n: int = 51,
        visible_gene_disc_random_n: int = 26,
        masked_gene_discrimination_enable: bool = False,
        masked_gene_disc_probability: float = 0.10,
        rtd_enable: bool = False,
        rtd_visible_positive_policy: str = "neg_count_match",
        gene_disc_support_bucket_count: int = 64,
        value_mask_ratio: float = 0.15,
        value_nonzero_only: bool = False,
        fixed_coordinate: bool = False,
        fixed_length_randorder: bool = False,
        randorder_seed: int = 42,
        paired_mode: str = "off",
        paired_lookup: PairedMetacellLookup | None = None,
        program_target_store: ProgramTargetStore | None = None,
        program_target_required_for_train: bool = False,
        gene_loading_target_store: GeneLoadingTargetStore | None = None,
        gene_pool_contrastive_enabled: bool = False,
        view_consistency_enable: bool = False,
        view_gene_dropout_prob: float = 0.10,
        view_value_jitter_prob: float = 0.10,
        view_value_jitter_delta: int = 1,
        domain_token_type_id: int | None = None,
        sc_token_type_id: int = 0,
        mc_token_type_id: int = 1,
    ):
        if "<pad>" not in token_dictionary or "<mask>" not in token_dictionary:
            raise KeyError("token dictionary must include <pad> and <mask>")
        self.token_dictionary = token_dictionary
        self.max_len = int(max_len)
        self.mlm_probability = float(mlm_probability)
        self.mask_sampling = str(mask_sampling).strip().lower()
        if self.mask_sampling not in {"uniform", "rank_frequency_balanced"}:
            raise ValueError(f"unsupported mask_sampling={mask_sampling}")
        self.mask_frequency_exponent = float(mask_frequency_exponent)
        self.mask_rank_bands = int(mask_rank_bands)
        self.hard_random_replacement = bool(hard_random_replacement)
        self.dual_mask_views = bool(dual_mask_views)
        self.dataset_balance_exponent = float(dataset_balance_exponent)
        self.dataset_weight_min = float(dataset_weight_min)
        self.dataset_weight_max = float(dataset_weight_max)
        if not 0.0 <= self.mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in [0, 1]")
        if self.mask_frequency_exponent < 0.0:
            raise ValueError("mask_frequency_exponent must be >= 0")
        if self.mask_rank_bands <= 0:
            raise ValueError("mask_rank_bands must be positive")
        if self.dataset_balance_exponent < 0.0:
            raise ValueError("dataset_balance_exponent must be >= 0")
        if not 0.0 < self.dataset_weight_min <= 1.0 <= self.dataset_weight_max:
            raise ValueError("dataset weight bounds must satisfy 0 < min <= 1 <= max")
        self.gene_mlm_enabled = bool(gene_mlm_enabled)
        self.use_value_input = bool(use_value_input)
        self.use_value_loss = bool(use_value_loss)
        self.value_pad_id = int(value_pad_id)
        self.value_mask_id = int(value_mask_id)
        self.value_mask_ratio = float(value_mask_ratio)
        self.value_nonzero_only = bool(value_nonzero_only)
        self.fixed_coordinate = bool(fixed_coordinate)
        self.pad_token_id = int(token_dictionary["<pad>"])
        self.mask_token_id = int(token_dictionary["<mask>"])
        self.special_token_ids = {
            int(token_id)
            for token, token_id in token_dictionary.items()
            if str(token).startswith("<")
        }
        self.vocab_size = len(token_dictionary)
        self.token_lookup_size = max(int(token_id) for token_id in token_dictionary.values()) + 1
        self.special_token_id_array = np.asarray(sorted(self.special_token_ids), dtype=np.int64)
        self.special_token_lookup = np.zeros((self.token_lookup_size,), dtype=bool)
        self.special_token_lookup[self.special_token_id_array] = True
        self._token_exclude_marks = np.zeros((self.token_lookup_size,), dtype=np.int32)
        self._token_exclude_generation = 0
        self.gene_mlm_policy = str(gene_mlm_policy).strip().lower()
        if self.gene_mlm_policy not in {"bert_801010", "mask_only"}:
            raise ValueError(f"unsupported gene_mlm_policy={gene_mlm_policy}")
        (
            self.gene_mlm_mask_replace_prob,
            self.gene_mlm_random_replace_prob,
            self.gene_mlm_unchanged_prob,
        ) = resolve_gene_mlm_corruption_probs(
            gene_mlm_policy=self.gene_mlm_policy,
            mask_replace_prob=gene_mlm_mask_replace_prob,
            random_replace_prob=gene_mlm_random_replace_prob,
        )
        self.paired_mode = str(paired_mode).strip().lower()
        self.paired_lookup = paired_lookup
        self.program_target_store = program_target_store
        self.program_target_required_for_train = bool(program_target_required_for_train)
        self.gene_loading_target_store = gene_loading_target_store
        self.gene_pool_contrastive_enabled = bool(gene_pool_contrastive_enabled)
        self.view_consistency_enable = bool(view_consistency_enable)
        self.view_gene_dropout_prob = float(view_gene_dropout_prob)
        self.view_value_jitter_prob = float(view_value_jitter_prob)
        self.view_value_jitter_delta = int(view_value_jitter_delta)
        self.domain_token_type_id = int(domain_token_type_id) if domain_token_type_id is not None else None
        self.sc_token_type_id = int(sc_token_type_id)
        self.mc_token_type_id = int(mc_token_type_id)
        self.value_target_decoupled = bool(value_target_decoupled)
        self.value_target_probability = float(value_target_probability)
        self.value_target_scope = str(value_target_scope).strip().lower()
        if self.value_target_scope not in {"observed_nonzero", "all_eligible"}:
            raise ValueError(f"unsupported value_target_scope={value_target_scope}")
        self.visible_gene_discrimination_enable = bool(visible_gene_discrimination_enable)
        self.visible_gene_disc_probability = float(visible_gene_disc_probability)
        self.visible_gene_disc_negative_k = int(visible_gene_disc_negative_k)
        self.visible_gene_disc_overlap_policy = str(visible_gene_disc_overlap_policy).strip().lower()
        if self.visible_gene_disc_overlap_policy not in {"disjoint", "allow"}:
            raise ValueError(
                f"unsupported visible_gene_disc_overlap_policy={visible_gene_disc_overlap_policy}"
            )
        self.visible_gene_disc_same_row_min_distance = int(visible_gene_disc_same_row_min_distance)
        self.visible_gene_disc_same_row_max_distance = int(visible_gene_disc_same_row_max_distance)
        self.visible_gene_disc_support_log_tolerance = float(visible_gene_disc_support_log_tolerance)
        self.visible_gene_disc_support_log_tolerance_fallback = float(
            visible_gene_disc_support_log_tolerance_fallback
        )
        self.visible_gene_disc_same_row_n = int(visible_gene_disc_same_row_n)
        self.visible_gene_disc_support_n = int(visible_gene_disc_support_n)
        self.visible_gene_disc_random_n = int(visible_gene_disc_random_n)
        self.masked_gene_discrimination_enable = bool(masked_gene_discrimination_enable)
        self.masked_gene_disc_probability = float(masked_gene_disc_probability)
        self.rtd_enable = bool(rtd_enable)
        self.rtd_visible_positive_policy = str(rtd_visible_positive_policy).strip().lower()
        if self.rtd_visible_positive_policy not in {"neg_count_match"}:
            raise ValueError(f"unsupported rtd_visible_positive_policy={rtd_visible_positive_policy}")
        self.gene_disc_support_bucket_count = int(gene_disc_support_bucket_count)
        if (
            self.visible_gene_disc_same_row_n
            + self.visible_gene_disc_support_n
            + self.visible_gene_disc_random_n
            != self.visible_gene_disc_negative_k
        ):
            raise ValueError("visible gene negative split must sum to visible_gene_disc_negative_k")
        self.fixed_length_randorder = bool(fixed_length_randorder)
        self.randorder_seed = int(randorder_seed)
        if self.paired_mode not in {"off", "one_to_one"}:
            raise ValueError(f"unsupported paired_mode={paired_mode}")
        if self.dual_mask_views and self.paired_mode != "off":
            raise ValueError("dual_mask_views currently requires paired_mode=off")
        if self.paired_mode != "off" and self.paired_lookup is None:
            raise ValueError("paired_lookup is required when paired_mode is enabled")
        if self.program_target_required_for_train and self.program_target_store is None:
            raise ValueError("program_target_store is required when program targets are required")
        if self.value_target_decoupled and not self.gene_mlm_enabled:
            raise ValueError("value_target_decoupled requires gene_mlm_enabled=True")
        if self.value_target_decoupled and not self.use_value_loss:
            raise ValueError("value_target_decoupled requires use_value_loss=True")
        if self.value_target_decoupled and self.use_value_input:
            raise ValueError("value_target_decoupled requires use_value_input=False")
        if self.visible_gene_discrimination_enable and not self.use_value_loss:
            raise ValueError("visible_gene_discrimination requires use_value_loss=True")
        if self.masked_gene_discrimination_enable and not self.use_value_input:
            raise ValueError("masked_gene_discrimination requires use_value_input=True")
        if self.masked_gene_discrimination_enable and not self.use_value_loss:
            raise ValueError("masked_gene_discrimination requires use_value_loss=True")
        if self.masked_gene_discrimination_enable and self.gene_mlm_enabled:
            raise ValueError("masked_gene_discrimination currently requires gene_mlm_enabled=False")
        if self.rtd_enable and not self.gene_mlm_enabled:
            raise ValueError("rtd_enable currently requires gene_mlm_enabled=True")
        if not 0.0 <= self.view_gene_dropout_prob <= 1.0:
            raise ValueError("view_gene_dropout_prob must be in [0, 1]")
        if not 0.0 <= self.view_value_jitter_prob <= 1.0:
            raise ValueError("view_value_jitter_prob must be in [0, 1]")
        if self.view_value_jitter_delta < 0:
            raise ValueError("view_value_jitter_delta must be >= 0")
        self.is_paired = self.paired_mode != "off"
        self.gene_token_ids = np.asarray(
            sorted(int(token_id) for token_id in token_dictionary.values() if int(token_id) not in self.special_token_ids),
            dtype=np.int64,
        )
        self.token_frequency = np.ones((self.token_lookup_size,), dtype=np.float64)
        self.token_median_rank = np.full((self.token_lookup_size,), np.nan, dtype=np.float64)
        self.dataset_loss_weight = np.ones((1,), dtype=np.float64)
        needs_sampling_stats = (
            self.mask_sampling == "rank_frequency_balanced"
            or self.hard_random_replacement
            or self.dataset_balance_exponent > 0.0
        )
        if needs_sampling_stats:
            if sampling_stats_npz is None:
                raise ValueError("sampling_stats_npz is required for balanced masking, hard replacement, or dataset weighting")
            stats_path = Path(sampling_stats_npz).expanduser().resolve()
            if not stats_path.exists():
                raise FileNotFoundError(f"sampling stats not found: {stats_path}")
            with np.load(stats_path, allow_pickle=False) as stats:
                token_frequency = np.asarray(stats["token_frequency"], dtype=np.float64).reshape(-1)
                token_median_rank = np.asarray(stats["token_median_rank"], dtype=np.float64).reshape(-1)
                dataset_count = np.asarray(stats["dataset_count"], dtype=np.float64).reshape(-1)
            if token_frequency.shape[0] < self.token_lookup_size:
                raise ValueError(
                    f"token_frequency too short: {token_frequency.shape[0]} < {self.token_lookup_size}"
                )
            if token_median_rank.shape[0] < self.token_lookup_size:
                raise ValueError(
                    f"token_median_rank too short: {token_median_rank.shape[0]} < {self.token_lookup_size}"
                )
            self.token_frequency = np.maximum(token_frequency[: self.token_lookup_size], 1.0)
            self.token_median_rank = token_median_rank[: self.token_lookup_size]
            self.dataset_loss_weight = self._build_dataset_loss_weights(dataset_count)
        self.hard_bucket_by_token = np.full((self.token_lookup_size,), -1, dtype=np.int64)
        self.hard_bucket_tokens: list[np.ndarray] = []
        self.hard_frequency_bucket_by_token = np.full((self.token_lookup_size,), -1, dtype=np.int64)
        self.hard_frequency_bucket_tokens: list[np.ndarray] = []
        if self.hard_random_replacement:
            self._build_hard_replacement_buckets()
        self.token_support_log: np.ndarray | None = None
        self.token_support_bucket_by_token: np.ndarray | None = None
        self.token_support_bucket_tokens: list[np.ndarray] = []
        self.token_support_bucket_orders: list[np.ndarray] = []
        if self.visible_gene_discrimination_enable or self.masked_gene_discrimination_enable:
            if gene_support_npy is None:
                raise ValueError("gene discrimination requires gene_support_npy")
            self.token_support_log = self._load_token_support_log(Path(gene_support_npy))
            (
                self.token_support_bucket_by_token,
                self.token_support_bucket_tokens,
            ) = self._build_support_buckets(self.token_support_log)
            self.token_support_bucket_orders = self._build_support_bucket_orders(len(self.token_support_bucket_tokens))

    def _build_dataset_loss_weights(self, dataset_count: np.ndarray) -> np.ndarray:
        counts = np.asarray(dataset_count, dtype=np.float64).reshape(-1)
        if counts.size == 0:
            return np.ones((1,), dtype=np.float64)
        positive = counts > 0
        if not bool(np.any(positive)) or self.dataset_balance_exponent == 0.0:
            return np.ones_like(counts, dtype=np.float64)
        raw = np.ones_like(counts, dtype=np.float64)
        raw[positive] = np.power(counts[positive], -self.dataset_balance_exponent)
        total_count = float(counts[positive].sum())

        def weighted_mean(scale: float) -> float:
            candidate = np.clip(
                raw[positive] * float(scale),
                self.dataset_weight_min,
                self.dataset_weight_max,
            )
            return float(np.dot(counts[positive], candidate) / total_count)

        lo = 0.0
        hi = 1.0
        while weighted_mean(hi) < 1.0:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if weighted_mean(mid) < 1.0:
                lo = mid
            else:
                hi = mid
        weights = np.ones_like(counts, dtype=np.float64)
        weights[positive] = np.clip(
            raw[positive] * (0.5 * (lo + hi)),
            self.dataset_weight_min,
            self.dataset_weight_max,
        )
        return weights

    @staticmethod
    def _quantile_assignments(values: np.ndarray, bucket_count: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0 or bucket_count <= 1:
            return np.zeros((arr.size,), dtype=np.int64)
        edges = np.quantile(arr, np.linspace(0.0, 1.0, int(bucket_count) + 1, dtype=np.float64))
        return np.searchsorted(edges[1:-1], arr, side="right").astype(np.int64, copy=False)

    def _build_hard_replacement_buckets(self) -> None:
        genes = np.asarray(self.gene_token_ids, dtype=np.int64)
        freq_assignment = self._quantile_assignments(np.log1p(self.token_frequency[genes]), 10)
        rank_values = self.token_median_rank[genes]
        finite_rank = np.isfinite(rank_values)
        rank_assignment = np.full((genes.size,), 3, dtype=np.int64)
        if bool(np.any(finite_rank)):
            rank_assignment[finite_rank] = self._quantile_assignments(rank_values[finite_rank], 4)
        self.hard_frequency_bucket_tokens = []
        for freq_bucket in range(10):
            tokens = genes[freq_assignment == freq_bucket]
            self.hard_frequency_bucket_tokens.append(tokens.astype(np.int64, copy=False))
            self.hard_frequency_bucket_by_token[tokens] = int(freq_bucket)
        self.hard_bucket_tokens = []
        for freq_bucket in range(10):
            for rank_bucket in range(4):
                bucket_idx = (freq_bucket * 4) + rank_bucket
                tokens = genes[(freq_assignment == freq_bucket) & (rank_assignment == rank_bucket)]
                self.hard_bucket_tokens.append(tokens.astype(np.int64, copy=False))
                self.hard_bucket_by_token[tokens] = int(bucket_idx)

    def _sample_masked_indices(self, input_ids: torch.Tensor, special_mask: torch.Tensor) -> torch.Tensor:
        if self.mask_sampling == "uniform":
            probability_matrix = torch.full(input_ids.shape, self.mlm_probability, dtype=torch.float32)
            probability_matrix.masked_fill_(special_mask, 0.0)
            return torch.bernoulli(probability_matrix).bool()

        selected = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_idx in range(int(input_ids.size(0))):
            eligible = torch.nonzero(~special_mask[row_idx], as_tuple=False).flatten()
            eligible_n = int(eligible.numel())
            if eligible_n == 0:
                continue
            total_target = int(round(self.mlm_probability * eligible_n))
            if self.mlm_probability > 0.0:
                total_target = max(1, total_target)
            total_target = min(eligible_n, total_target)
            for band_idx in range(self.mask_rank_bands):
                start = (eligible_n * band_idx) // self.mask_rank_bands
                end = (eligible_n * (band_idx + 1)) // self.mask_rank_bands
                if end <= start:
                    continue
                target_before = int(round(total_target * (start / eligible_n)))
                target_after = int(round(total_target * (end / eligible_n)))
                band_target = min(end - start, max(0, target_after - target_before))
                if band_target <= 0:
                    continue
                band_positions = eligible[start:end]
                token_ids = input_ids[row_idx, band_positions].detach().cpu().numpy().astype(np.int64, copy=False)
                frequency = np.maximum(self.token_frequency[token_ids], 1.0)
                weights = np.power(frequency, -self.mask_frequency_exponent)
                weights_t = torch.as_tensor(weights, dtype=torch.float64)
                chosen_local = torch.multinomial(weights_t, num_samples=band_target, replacement=False)
                selected[row_idx, band_positions[chosen_local]] = True
        return selected

    def _sample_hard_random_words(
        self,
        *,
        features: list[dict[str, Any]],
        original_input_ids: torch.Tensor,
        indices_random: torch.Tensor,
    ) -> torch.Tensor:
        random_words = original_input_ids.clone()
        for row_idx in range(int(original_input_ids.size(0))):
            cols = torch.nonzero(indices_random[row_idx], as_tuple=False).flatten()
            if int(cols.numel()) == 0:
                continue
            row_ids = original_input_ids[row_idx].detach().cpu().numpy().astype(np.int64, copy=False)
            excluded = set(int(token_id) for token_id in row_ids.tolist())
            rng = self._rng_for_feature(features[row_idx], row_offset=(row_idx + 4000))
            for col_idx in cols.tolist():
                original_gene = int(original_input_ids[row_idx, int(col_idx)].item())
                bucket_idx = int(self.hard_bucket_by_token[original_gene])
                candidates = (
                    self.hard_bucket_tokens[bucket_idx]
                    if 0 <= bucket_idx < len(self.hard_bucket_tokens)
                    else np.empty((0,), dtype=np.int64)
                )
                candidates = self._filter_excluded_token_ids(candidates, excluded)
                if candidates.size == 0:
                    freq_bucket = int(self.hard_frequency_bucket_by_token[original_gene])
                    candidates = (
                        self.hard_frequency_bucket_tokens[freq_bucket]
                        if 0 <= freq_bucket < len(self.hard_frequency_bucket_tokens)
                        else np.empty((0,), dtype=np.int64)
                    )
                    candidates = self._filter_excluded_token_ids(candidates, excluded)
                if candidates.size == 0:
                    candidates = self._filter_excluded_token_ids(self.gene_token_ids, excluded)
                if candidates.size == 0:
                    raise ValueError("hard replacement has no valid candidate")
                replacement = int(rng.choice(candidates))
                random_words[row_idx, int(col_idx)] = replacement
        return random_words

    def _example_loss_weights(self, features: list[dict[str, Any]]) -> torch.Tensor | None:
        if self.dataset_balance_exponent == 0.0:
            return None
        values: list[float] = []
        for feature in features:
            dataset_idx = int(feature.get("dataset_idx", -1))
            if 0 <= dataset_idx < self.dataset_loss_weight.shape[0]:
                values.append(float(self.dataset_loss_weight[dataset_idx]))
            else:
                values.append(1.0)
        return torch.as_tensor(values, dtype=torch.float32)

    def _load_token_support_log(self, support_path: Path) -> np.ndarray:
        support = np.asarray(np.load(support_path), dtype=np.float64).reshape(-1)
        gene_items = [
            (str(token), int(token_id))
            for token, token_id in self.token_dictionary.items()
            if int(token_id) not in self.special_token_ids
        ]
        gene_items.sort(key=lambda item: item[1])
        if support.shape[0] != len(gene_items):
            raise ValueError(
                f"gene_support_npy length mismatch: support={support.shape[0]} gene_tokens={len(gene_items)}"
            )
        token_support = np.zeros((self.token_lookup_size,), dtype=np.float64)
        for support_idx, (_, token_id) in enumerate(gene_items):
            token_support[int(token_id)] = float(support[support_idx])
        return np.log1p(token_support)

    def _build_support_buckets(self, support_log: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        global_pool = np.asarray(self.gene_token_ids, dtype=np.int64)
        if global_pool.size == 0:
            raise ValueError("gene_token_ids is empty")
        bucket_count = max(1, min(int(self.gene_disc_support_bucket_count), int(global_pool.size)))
        support_values = np.asarray(support_log[global_pool], dtype=np.float64)
        if bucket_count == 1:
            assignments = np.zeros((global_pool.size,), dtype=np.int64)
        else:
            quantiles = np.linspace(0.0, 1.0, bucket_count + 1, dtype=np.float64)
            edges = np.quantile(support_values, quantiles)
            assignments = np.searchsorted(edges[1:-1], support_values, side="right").astype(np.int64, copy=False)
        bucket_by_token = np.full((self.token_lookup_size,), -1, dtype=np.int64)
        bucket_tokens: list[np.ndarray] = []
        for bucket_idx in range(bucket_count):
            tokens = global_pool[assignments == bucket_idx].astype(np.int64, copy=False)
            bucket_tokens.append(tokens)
            if tokens.size > 0:
                bucket_by_token[tokens] = int(bucket_idx)
        return bucket_by_token, bucket_tokens

    def _build_support_bucket_orders(self, bucket_count: int) -> list[np.ndarray]:
        orders: list[np.ndarray] = []
        for pos_bucket in range(int(bucket_count)):
            order: list[int] = []
            for offset in range(int(bucket_count)):
                if offset == 0:
                    order.append(pos_bucket)
                    continue
                left = pos_bucket - offset
                right = pos_bucket + offset
                if left >= 0:
                    order.append(left)
                if right < bucket_count:
                    order.append(right)
            orders.append(np.asarray(order, dtype=np.int64))
        return orders

    def _normalize_exclude_ids(self, exclude_ids: Any) -> np.ndarray:
        if exclude_ids is None:
            return np.empty((0,), dtype=np.int64)
        if isinstance(exclude_ids, np.ndarray):
            return np.asarray(exclude_ids, dtype=np.int64).reshape(-1)
        if isinstance(exclude_ids, set):
            if not exclude_ids:
                return np.empty((0,), dtype=np.int64)
            return np.fromiter((int(item) for item in exclude_ids), dtype=np.int64, count=len(exclude_ids))
        if isinstance(exclude_ids, (list, tuple)):
            if len(exclude_ids) == 0:
                return np.empty((0,), dtype=np.int64)
            return np.asarray(exclude_ids, dtype=np.int64).reshape(-1)
        return np.asarray([int(exclude_ids)], dtype=np.int64)

    def _combine_exclude_ids(self, *exclude_groups: Any) -> np.ndarray:
        parts: list[np.ndarray] = []
        for exclude_ids in exclude_groups:
            arr = self._normalize_exclude_ids(exclude_ids)
            if arr.size > 0:
                parts.append(arr)
        if not parts:
            return np.empty((0,), dtype=np.int64)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    def _filter_excluded_token_ids(self, candidates: np.ndarray, exclude_ids: Any) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
        if candidates.size == 0:
            return candidates
        exclude_arr = self._normalize_exclude_ids(exclude_ids)
        if exclude_arr.size == 0:
            return candidates
        next_generation = int(self._token_exclude_generation) + 1
        if next_generation >= np.iinfo(self._token_exclude_marks.dtype).max:
            self._token_exclude_marks.fill(0)
            next_generation = 1
        self._token_exclude_generation = next_generation
        valid = exclude_arr[(exclude_arr >= 0) & (exclude_arr < self.token_lookup_size)]
        if valid.size == 0:
            return candidates
        self._token_exclude_marks[valid] = next_generation
        return candidates[self._token_exclude_marks[candidates] != next_generation]

    def _sample_without_replacement_from_pool(
        self,
        pool: np.ndarray,
        *,
        n: int,
        rng: np.random.Generator,
        exclude_ids: set[int] | None = None,
        assume_unique: bool = False,
    ) -> np.ndarray:
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        candidates = np.asarray(pool, dtype=np.int64).reshape(-1)
        if candidates.size == 0:
            return np.empty((0,), dtype=np.int64)
        if not assume_unique:
            candidates = np.unique(candidates)
            if candidates.size == 0:
                return np.empty((0,), dtype=np.int64)
        candidates = self._filter_excluded_token_ids(candidates, exclude_ids)
        if candidates.size == 0:
            return np.empty((0,), dtype=np.int64)
        if candidates.size >= n:
            choice = rng.choice(candidates, size=n, replace=False)
            return np.asarray(choice, dtype=np.int64)
        return np.asarray(candidates, dtype=np.int64)

    def _pad_sampled_ids(
        self,
        sampled: np.ndarray,
        *,
        n: int,
        fallback_pool: np.ndarray,
        rng: np.random.Generator,
        exclude_ids: set[int] | None = None,
        assume_unique_fallback: bool = False,
    ) -> np.ndarray:
        sampled = np.asarray(sampled, dtype=np.int64).reshape(-1)
        if sampled.size >= n:
            return sampled[:n]
        fallback = np.asarray(fallback_pool, dtype=np.int64).reshape(-1)
        if fallback.size == 0:
            fallback = np.asarray(self.gene_token_ids, dtype=np.int64)
            assume_unique_fallback = True
        if not assume_unique_fallback:
            fallback = np.unique(fallback)
        fallback = self._filter_excluded_token_ids(fallback, exclude_ids)
        if fallback.size == 0:
            fallback = self._filter_excluded_token_ids(np.asarray(self.gene_token_ids, dtype=np.int64), exclude_ids)
        if fallback.size == 0:
            raise ValueError("no fallback pool available for visible gene negative sampling")
        need = int(n - sampled.size)
        extra = rng.choice(fallback, size=need, replace=(fallback.size < need))
        return np.concatenate([sampled, np.asarray(extra, dtype=np.int64)], axis=0)

    def _sample_support_matched_gene_ids(
        self,
        *,
        pos_gene_id: int,
        n: int,
        rng: np.random.Generator,
        exclude_ids: set[int],
    ) -> np.ndarray:
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        bucket_by_token = self.token_support_bucket_by_token
        bucket_tokens = self.token_support_bucket_tokens
        if bucket_by_token is None or not bucket_tokens:
            raise ValueError("support buckets must be initialized for gene discrimination")
        pos_bucket = int(bucket_by_token[int(pos_gene_id)])
        if pos_bucket < 0:
            return self._pad_sampled_ids(
                np.empty((0,), dtype=np.int64),
                n=n,
                fallback_pool=np.asarray(self.gene_token_ids, dtype=np.int64),
                rng=rng,
                exclude_ids=exclude_ids,
                assume_unique_fallback=True,
            )
        selected_parts: list[np.ndarray] = []
        selected_ids = self._normalize_exclude_ids(exclude_ids)
        selected_n = 0
        for bucket_idx in self.token_support_bucket_orders[pos_bucket]:
            remaining = int(n - selected_n)
            if remaining <= 0:
                break
            sampled = self._sample_without_replacement_from_pool(
                bucket_tokens[int(bucket_idx)],
                n=remaining,
                rng=rng,
                exclude_ids=selected_ids,
                assume_unique=True,
            )
            if sampled.size > 0:
                selected_parts.append(sampled)
                selected_ids = self._combine_exclude_ids(selected_ids, sampled)
                selected_n += int(sampled.shape[0])
        selected_arr = (
            np.concatenate(selected_parts, axis=0).astype(np.int64, copy=False)
            if selected_parts
            else np.empty((0,), dtype=np.int64)
        )
        if selected_arr.size < n:
            selected_arr = self._pad_sampled_ids(
                selected_arr,
                n=n,
                fallback_pool=np.asarray(self.gene_token_ids, dtype=np.int64),
                rng=rng,
                exclude_ids=selected_ids,
                assume_unique_fallback=True,
            )
        return selected_arr

    def _prepare_gene_disc_row_context(
        self,
        *,
        row_gene_ids: np.ndarray,
        row_source_rank: np.ndarray,
        row_value_bins: np.ndarray,
    ) -> GeneDiscRowContext:
        row_gene_ids = np.asarray(row_gene_ids, dtype=np.int64).reshape(-1)
        row_source_rank = np.asarray(row_source_rank, dtype=np.int64).reshape(-1)
        row_value_bins = np.asarray(row_value_bins, dtype=np.int64).reshape(-1)
        same_row_nonzero_mask = row_value_bins > 0
        same_row_nonzero_gene_ids = row_gene_ids[same_row_nonzero_mask]
        same_row_nonzero_source_rank = row_source_rank[same_row_nonzero_mask]
        same_row_gene_universe = np.unique(same_row_nonzero_gene_ids)
        return GeneDiscRowContext(
            row_gene_ids=row_gene_ids,
            row_source_rank=row_source_rank,
            same_row_nonzero_gene_ids=same_row_nonzero_gene_ids,
            same_row_nonzero_source_rank=same_row_nonzero_source_rank,
            same_row_gene_universe=same_row_gene_universe,
        )

    def _sample_gene_disc_negatives(
        self,
        *,
        row_ctx: GeneDiscRowContext,
        pos_source_rank: int,
        pos_gene_id: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_rank = int(pos_source_rank)
        row_distance = np.abs(row_ctx.same_row_nonzero_source_rank - pos_rank)
        pos_exclude = np.asarray([int(pos_gene_id)], dtype=np.int64)

        same_row_selected = np.empty((0,), dtype=np.int64)
        for min_distance, max_distance in (
            (int(self.visible_gene_disc_same_row_min_distance), int(self.visible_gene_disc_same_row_max_distance)),
            (6, 600),
            (6, 1024),
        ):
            remaining = int(self.visible_gene_disc_same_row_n - same_row_selected.size)
            if remaining <= 0:
                break
            band_mask = (
                (row_distance >= int(min_distance))
                & (row_distance <= int(max_distance))
            )
            band_sample = self._sample_without_replacement_from_pool(
                row_ctx.same_row_nonzero_gene_ids[band_mask],
                n=remaining,
                rng=rng,
                exclude_ids=self._combine_exclude_ids(pos_exclude, same_row_selected),
            )
            if band_sample.size > 0:
                same_row_selected = np.concatenate([same_row_selected, band_sample.astype(np.int64, copy=False)], axis=0)
        if same_row_selected.size < self.visible_gene_disc_same_row_n:
            fallback_same_row = self._sample_without_replacement_from_pool(
                row_ctx.same_row_nonzero_gene_ids,
                n=int(self.visible_gene_disc_same_row_n - same_row_selected.size),
                rng=rng,
                exclude_ids=self._combine_exclude_ids(pos_exclude, same_row_selected),
            )
            if fallback_same_row.size > 0:
                same_row_selected = np.concatenate(
                    [same_row_selected, fallback_same_row.astype(np.int64, copy=False)],
                    axis=0,
                )

        same_row_deficit = int(self.visible_gene_disc_same_row_n - same_row_selected.size)
        support_target_n = int(self.visible_gene_disc_support_n + max(0, same_row_deficit))
        same_row_gene_universe = row_ctx.same_row_gene_universe[row_ctx.same_row_gene_universe != int(pos_gene_id)]
        support_exclude = self._combine_exclude_ids(pos_exclude, same_row_gene_universe, same_row_selected)
        support_sample = self._sample_support_matched_gene_ids(
            pos_gene_id=int(pos_gene_id),
            n=support_target_n,
            rng=rng,
            exclude_ids=support_exclude,
        )

        support_deficit = int(support_target_n - support_sample.size)
        random_target_n = int(self.visible_gene_disc_random_n + max(0, support_deficit))
        global_pool = np.asarray(self.gene_token_ids, dtype=np.int64)
        random_exclude = self._combine_exclude_ids(support_exclude, support_sample)
        random_sample = self._sample_without_replacement_from_pool(
            global_pool,
            n=random_target_n,
            rng=rng,
            exclude_ids=random_exclude,
            assume_unique=True,
        )
        if random_sample.size < random_target_n:
            random_sample = self._pad_sampled_ids(
                random_sample,
                n=random_target_n,
                fallback_pool=global_pool,
                rng=rng,
                exclude_ids=self._combine_exclude_ids(random_exclude, random_sample),
                assume_unique_fallback=True,
            )

        neg_gene_ids = np.concatenate([same_row_selected, support_sample, random_sample], axis=0).astype(
            np.int64,
            copy=False,
        )
        neg_types = np.concatenate(
            [
                np.zeros((same_row_selected.shape[0],), dtype=np.int64),
                np.ones((support_sample.shape[0],), dtype=np.int64),
                np.full((random_sample.shape[0],), 2, dtype=np.int64),
            ],
            axis=0,
        )
        if neg_gene_ids.shape[0] != self.visible_gene_disc_negative_k:
            raise ValueError("gene discrimination negative sampler produced unexpected candidate count")
        return neg_gene_ids, neg_types

    def _tensorize_1d(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=torch.long).flatten()
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
        return torch.from_numpy(arr)

    def _rng_for_feature(self, feature: dict[str, Any], *, row_offset: int) -> np.random.Generator:
        dataset_idx = int(feature.get("dataset_idx", 0))
        metacell_local_idx = int(feature.get("metacell_local_idx", row_offset))
        seed = (
            self.randorder_seed
            + (dataset_idx * 1_000_003)
            + (metacell_local_idx * 10_007)
            + (row_offset * 97)
        ) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def _build_program_target_batch(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        store = self.program_target_store
        if store is None:
            return {}
        batch_size = len(features)
        targets = torch.zeros((batch_size, store.program_dim), dtype=torch.float32)
        mask = torch.zeros((batch_size,), dtype=torch.bool)
        for row_idx, feature in enumerate(features):
            target = store.get_target(feature)
            if target is None:
                continue
            targets[row_idx] = torch.from_numpy(np.asarray(target, dtype=np.float32))
            mask[row_idx] = True
        return {
            "program_targets": targets,
            "program_target_mask": mask,
        }

    def _build_gene_loading_target_batch(
        self,
        *,
        original_input_ids: torch.Tensor,
        original_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        store = self.gene_loading_target_store
        if store is None:
            return {}
        targets, mask = store.build_batch(
            input_ids=original_input_ids,
            attention_mask=original_attention_mask,
        )
        return {
            "gene_loading_targets": targets,
            "gene_loading_target_mask": mask,
        }

    def _sample_source_positions(
        self,
        source_ranks: np.ndarray,
        *,
        probability: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidates = np.asarray(source_ranks, dtype=np.int64).reshape(-1)
        if candidates.size == 0 or probability <= 0.0:
            return np.empty((0,), dtype=np.int64)
        select = rng.random(candidates.shape[0]) < float(probability)
        if not bool(np.any(select)):
            return np.empty((0,), dtype=np.int64)
        return candidates[select].astype(np.int64, copy=False)

    def _build_perturbed_view(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        value_bins: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        view_input_ids = input_ids.clone()
        view_attention_mask = attention_mask.clone()
        view_value_bins = value_bins.clone() if value_bins is not None else None
        view_token_type_ids = token_type_ids.clone() if token_type_ids is not None else None

        gene_mask = view_attention_mask.gt(0)
        for token_id in self.special_token_ids:
            gene_mask &= view_input_ids.ne(int(token_id))

        if self.view_gene_dropout_prob > 0.0 and bool(gene_mask.any()):
            drop_prob = torch.full(view_input_ids.shape, float(self.view_gene_dropout_prob), dtype=torch.float32)
            drop_mask = torch.bernoulli(drop_prob).bool() & gene_mask
            view_input_ids[drop_mask] = self.pad_token_id
            view_attention_mask[drop_mask] = 0
            if view_value_bins is not None:
                view_value_bins[drop_mask] = self.value_pad_id
            if view_token_type_ids is not None:
                view_token_type_ids[drop_mask] = 0
            gene_mask &= ~drop_mask

        if (
            view_value_bins is not None
            and self.view_value_jitter_prob > 0.0
            and self.view_value_jitter_delta > 0
            and bool(gene_mask.any())
        ):
            jitter_candidate = gene_mask & view_value_bins.gt(self.value_pad_id) & view_value_bins.lt(self.value_mask_id)
            if bool(jitter_candidate.any()):
                jitter_prob = torch.full(view_value_bins.shape, float(self.view_value_jitter_prob), dtype=torch.float32)
                jitter_mask = torch.bernoulli(jitter_prob).bool() & jitter_candidate
                if bool(jitter_mask.any()):
                    sign = torch.randint(0, 2, view_value_bins.shape, dtype=torch.long)
                    delta = torch.where(sign.eq(0), -self.view_value_jitter_delta, self.view_value_jitter_delta)
                    view_value_bins[jitter_mask] = (
                        view_value_bins[jitter_mask] + delta[jitter_mask]
                    ).clamp(min=1, max=self.value_mask_id - 1)

        batch = {
            "view_input_ids": view_input_ids,
            "view_attention_mask": view_attention_mask,
        }
        if view_token_type_ids is not None:
            batch["view_token_type_ids"] = view_token_type_ids
        if view_value_bins is not None:
            batch["view_value_bins"] = view_value_bins
        return batch

    def _apply_fixed_length_randorder(
        self,
        *,
        input_ids_row: torch.Tensor,
        attention_mask_row: torch.Tensor,
        value_bins_row: torch.Tensor | None,
        source_rank_row: torch.Tensor | None,
        rng: np.random.Generator,
    ) -> None:
        eligible = attention_mask_row.gt(0)
        for token_id in self.special_token_ids:
            eligible &= input_ids_row.ne(int(token_id))
        eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()
        if int(eligible_idx.numel()) < 2:
            return
        perm = torch.as_tensor(rng.permutation(int(eligible_idx.numel())), dtype=torch.long)
        source_idx = eligible_idx[perm]
        ids_clone = input_ids_row.clone()
        input_ids_row[eligible_idx] = ids_clone[source_idx]
        if value_bins_row is not None:
            value_clone = value_bins_row.clone()
            value_bins_row[eligible_idx] = value_clone[source_idx]
        if source_rank_row is not None:
            rank_clone = source_rank_row.clone()
            source_rank_row[eligible_idx] = rank_clone[source_idx]

    def _build_view_batch(
        self,
        features: list[dict[str, Any]],
        *,
        prefix: str | None = None,
        token_type_id: int | None = None,
        build_dual: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        input_ids = torch.full((batch_size, self.max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        token_type_ids = (
            torch.full((batch_size, self.max_len), int(token_type_id), dtype=torch.long)
            if token_type_id is not None
            else None
        )
        value_bins = None
        if self.use_value_input or self.use_value_loss:
            value_bins = torch.full((batch_size, self.max_len), self.value_pad_id, dtype=torch.long)
        source_rank = None
        if (
            self.fixed_length_randorder
            or self.visible_gene_discrimination_enable
            or self.masked_gene_discrimination_enable
        ):
            source_rank = torch.full((batch_size, self.max_len), -1, dtype=torch.long)

        for row_idx, feature in enumerate(features):
            ids = self._tensorize_1d(feature["input_ids"])[: self.max_len]
            n = int(ids.numel())
            if n > 0:
                input_ids[row_idx, :n] = ids
                if source_rank is not None:
                    source_rank[row_idx, :n] = torch.arange(n, dtype=torch.long)

            if value_bins is not None:
                if "value_bins" not in feature:
                    raise KeyError("value_bins column is required when value input/loss is enabled")
                vals = self._tensorize_1d(feature["value_bins"])[: self.max_len]
                value_bins[row_idx, : int(vals.numel())] = vals

            if "attention_mask" in feature:
                att = self._tensorize_1d(feature["attention_mask"])[: self.max_len]
                attention_mask[row_idx, : int(att.numel())] = att
            elif "length" in feature:
                length = min(int(feature["length"]), self.max_len)
                attention_mask[row_idx, :length] = 1
            else:
                attention_mask[row_idx] = input_ids[row_idx].ne(self.pad_token_id).long()

            if self.fixed_length_randorder:
                self._apply_fixed_length_randorder(
                    input_ids_row=input_ids[row_idx],
                    attention_mask_row=attention_mask[row_idx],
                    value_bins_row=(value_bins[row_idx] if value_bins is not None else None),
                    source_rank_row=(source_rank[row_idx] if source_rank is not None else None),
                    rng=self._rng_for_feature(feature, row_offset=row_idx),
                )

        original_input_ids = input_ids.clone()
        original_attention_mask = attention_mask.clone()
        original_value_bins = value_bins.clone() if value_bins is not None else None
        original_token_type_ids = token_type_ids.clone() if token_type_ids is not None else None

        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_mask |= input_ids.eq(token_id)
        special_mask |= attention_mask.eq(0)
        structured_gene_disc_sampling = (
            (not self.gene_mlm_enabled)
            and (self.visible_gene_discrimination_enable or self.masked_gene_discrimination_enable)
        )

        if self.gene_mlm_enabled:
            labels = torch.full_like(input_ids, -100)
            masked_indices = self._sample_masked_indices(input_ids, special_mask)
            labels[masked_indices] = input_ids[masked_indices]
            indices_replaced = torch.zeros_like(masked_indices)
            indices_random = torch.zeros_like(masked_indices)

            value_labels = None
            if value_bins is not None:
                value_labels = torch.full_like(value_bins, -100)
                if self.value_target_decoupled:
                    value_probability = torch.full(value_bins.shape, self.value_target_probability, dtype=torch.float32)
                    value_mask_exclude = special_mask | masked_indices
                    if self.value_target_scope == "observed_nonzero":
                        value_mask_exclude |= value_bins.le(0)
                    else:
                        value_mask_exclude |= value_bins.eq(self.value_pad_id)
                    value_probability.masked_fill_(value_mask_exclude, 0.0)
                    value_masked_indices = torch.bernoulli(value_probability).bool()
                    value_labels[value_masked_indices] = value_bins[value_masked_indices]
                else:
                    value_labels[masked_indices] = value_bins[masked_indices]

            if masked_indices.any():
                replace_prob = torch.full(
                    labels.shape,
                    float(self.gene_mlm_mask_replace_prob),
                    dtype=torch.float32,
                )
                indices_replaced = torch.bernoulli(replace_prob).bool() & masked_indices
                input_ids[indices_replaced] = self.mask_token_id

                random_after_mask = 0.0
                if self.gene_mlm_random_replace_prob > 0.0:
                    remaining_prob = 1.0 - float(self.gene_mlm_mask_replace_prob)
                    random_after_mask = float(self.gene_mlm_random_replace_prob) / max(remaining_prob, 1e-12)
                random_prob = torch.full(labels.shape, random_after_mask, dtype=torch.float32)
                indices_random = (
                    torch.bernoulli(random_prob).bool() & masked_indices & ~indices_replaced
                )
                if indices_random.any():
                    if self.hard_random_replacement:
                        random_words = self._sample_hard_random_words(
                            features=features,
                            original_input_ids=original_input_ids,
                            indices_random=indices_random,
                        )
                    else:
                        gene_pool = torch.as_tensor(self.gene_token_ids, dtype=torch.long)
                        sampled_idx = torch.randint(low=0, high=int(gene_pool.numel()), size=labels.shape, dtype=torch.long)
                        random_words = gene_pool[sampled_idx]
                    input_ids[indices_random] = random_words[indices_random]

                if value_bins is not None and self.use_value_input and not self.value_target_decoupled:
                    value_bins[masked_indices] = self.value_mask_id
            indices_unchanged = masked_indices & ~indices_replaced & ~indices_random
        else:
            labels = torch.full_like(input_ids, -100)
            masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
            indices_replaced = torch.zeros_like(input_ids, dtype=torch.bool)
            indices_random = torch.zeros_like(input_ids, dtype=torch.bool)
            indices_unchanged = torch.zeros_like(input_ids, dtype=torch.bool)
            value_labels = None
            if value_bins is not None:
                value_labels = torch.full_like(value_bins, -100)
                if structured_gene_disc_sampling:
                    if source_rank is None:
                        raise ValueError("structured gene discrimination sampling requires source_rank")
                    for row_idx in range(batch_size):
                        valid_pos = (
                            attention_mask[row_idx].gt(0)
                            & source_rank[row_idx].ge(0)
                        )
                        if not bool(valid_pos.any()):
                            continue
                        current_positions = torch.nonzero(valid_pos, as_tuple=False).flatten()
                        row_current = current_positions.detach().cpu().numpy().astype(np.int64, copy=False)
                        row_source = source_rank[row_idx, current_positions].detach().cpu().numpy().astype(np.int64, copy=False)
                        row_gene_ids = input_ids[row_idx, current_positions].detach().cpu().numpy().astype(np.int64, copy=False)
                        row_value_bins = value_bins[row_idx, current_positions].detach().cpu().numpy().astype(np.int64, copy=False)
                        row_special = np.isin(row_gene_ids, np.asarray(sorted(self.special_token_ids), dtype=np.int64))
                        source_to_current = np.full((int(row_source.max()) + 1,), -1, dtype=np.int64)
                        source_to_current[row_source] = row_current
                        eligible_source = row_source[~row_special]
                        observed_nonzero_source = row_source[(~row_special) & (row_value_bins > 0)]
                        if self.value_target_scope == "observed_nonzero":
                            value_candidate_source = observed_nonzero_source
                        else:
                            value_candidate_source = row_source[(~row_special) & (row_value_bins != int(self.value_pad_id))]
                        rng_row = self._rng_for_feature(features[row_idx], row_offset=row_idx)
                        value_source_selected = self._sample_source_positions(
                            value_candidate_source,
                            probability=self.value_target_probability,
                            rng=rng_row,
                        )
                        if value_source_selected.size > 0:
                            value_cols = source_to_current[value_source_selected]
                            value_cols_t = torch.as_tensor(value_cols, dtype=torch.long)
                            masked_indices[row_idx, value_cols_t] = True
                            value_labels[row_idx, value_cols_t] = value_bins[row_idx, value_cols_t]
                            value_bins[row_idx, value_cols_t] = self.value_mask_id
                else:
                    value_prob = (
                        self.value_target_probability if self.visible_gene_discrimination_enable else self.value_mask_ratio
                    )
                    value_probability = torch.full(value_bins.shape, value_prob, dtype=torch.float32)
                    value_mask_exclude = special_mask | value_bins.eq(self.value_pad_id)
                    if self.visible_gene_discrimination_enable:
                        if self.value_target_scope == "observed_nonzero":
                            value_mask_exclude |= value_bins.le(0)
                    elif self.value_nonzero_only:
                        value_mask_exclude |= value_bins.le(0)
                    value_probability.masked_fill_(value_mask_exclude, 0.0)
                    masked_indices = torch.bernoulli(value_probability).bool()
                    value_labels[masked_indices] = value_bins[masked_indices]
                    value_bins[masked_indices] = self.value_mask_id

        rtd_labels = None
        rtd_token_types = None
        if self.rtd_enable:
            rtd_labels = torch.full_like(input_ids, -100)
            rtd_token_types = torch.full_like(input_ids, -100)
            current_special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in self.special_token_ids:
                current_special_mask |= input_ids.eq(int(token_id))
            current_special_mask |= attention_mask.eq(0)
            rtd_eligible = (~special_mask) & (~current_special_mask) & input_ids.ne(self.mask_token_id)
            negative_random = indices_random & rtd_eligible & labels.ne(-100) & input_ids.ne(labels)
            positive_unchanged = indices_unchanged & rtd_eligible & labels.ne(-100)
            visible_candidate = (~masked_indices) & rtd_eligible

            rtd_labels[negative_random] = 0
            rtd_token_types[negative_random] = 0
            rtd_labels[positive_unchanged] = 1
            rtd_token_types[positive_unchanged] = 1

            neg_n = int(negative_random.sum().item())
            if neg_n > 0 and bool(visible_candidate.any()):
                flat_candidates = torch.nonzero(visible_candidate.reshape(-1), as_tuple=False).flatten()
                if int(flat_candidates.numel()) > 0:
                    take_n = min(neg_n, int(flat_candidates.numel()))
                    order = torch.randperm(int(flat_candidates.numel()))[:take_n]
                    chosen = flat_candidates[order]
                    rows = torch.div(chosen, self.max_len, rounding_mode="floor")
                    cols = chosen.remainder(self.max_len)
                    rtd_labels[rows, cols] = 1
                    rtd_token_types[rows, cols] = 2

        visible_gene_disc_row_idx = torch.empty((0,), dtype=torch.long)
        visible_gene_disc_col_idx = torch.empty((0,), dtype=torch.long)
        visible_gene_disc_pos_gene_ids = torch.empty((0,), dtype=torch.long)
        visible_gene_disc_neg_gene_ids = torch.empty((0, self.visible_gene_disc_negative_k), dtype=torch.long)
        visible_gene_disc_neg_types = torch.empty((0, self.visible_gene_disc_negative_k), dtype=torch.long)
        masked_gene_disc_row_idx = torch.empty((0,), dtype=torch.long)
        masked_gene_disc_col_idx = torch.empty((0,), dtype=torch.long)
        masked_gene_disc_pos_gene_ids = torch.empty((0,), dtype=torch.long)
        masked_gene_disc_neg_gene_ids = torch.empty((0, self.visible_gene_disc_negative_k), dtype=torch.long)
        masked_gene_disc_neg_types = torch.empty((0, self.visible_gene_disc_negative_k), dtype=torch.long)
        masked_gene_candidate_n = torch.zeros((), dtype=torch.float32)
        masked_gene_visible_overlap_n = torch.zeros((), dtype=torch.float32)
        visible_gene_corruption_overlap_n = torch.zeros((), dtype=torch.float32)
        if self.visible_gene_discrimination_enable or self.masked_gene_discrimination_enable:
            if value_bins is None or source_rank is None:
                raise ValueError("gene discrimination requires value_bins and source_rank")
            row_items: list[int] = []
            col_items: list[int] = []
            pos_items: list[int] = []
            neg_items: list[np.ndarray] = []
            neg_type_items: list[np.ndarray] = []
            masked_row_items: list[int] = []
            masked_col_items: list[int] = []
            masked_pos_items: list[int] = []
            masked_neg_items: list[np.ndarray] = []
            masked_neg_type_items: list[np.ndarray] = []
            masked_candidate_total = 0.0
            masked_visible_overlap_total = 0.0
            visible_corruption_overlap_total = 0.0
            for row_idx in range(batch_size):
                valid_pos = (
                    attention_mask[row_idx].gt(0)
                    & source_rank[row_idx].ge(0)
                )
                if not bool(valid_pos.any()):
                    continue
                current_positions = torch.nonzero(valid_pos, as_tuple=False).flatten()
                row_current = current_positions.detach().cpu().numpy().astype(np.int64, copy=False)
                row_source = source_rank[row_idx, current_positions].detach().cpu().numpy().astype(np.int64, copy=False)
                row_gene_ids = input_ids[row_idx, current_positions].detach().cpu().numpy().astype(np.int64, copy=False)
                row_value_bins = value_bins[row_idx, current_positions].detach().cpu().numpy().astype(np.int64, copy=False)
                row_special = self.special_token_lookup[row_gene_ids]
                source_to_current = np.full((int(row_source.max()) + 1,), -1, dtype=np.int64)
                source_to_current[row_source] = row_current
                eligible_source = row_source[~row_special]
                observed_nonzero_source = row_source[(~row_special) & (row_value_bins > 0)]
                value_source_selected = source_rank[row_idx, masked_indices[row_idx]].detach().cpu().numpy().astype(np.int64, copy=False)
                source_selected_mask = np.zeros((source_to_current.shape[0],), dtype=bool)
                if value_source_selected.size > 0:
                    source_selected_mask[value_source_selected] = True
                masked_source_selected = np.empty((0,), dtype=np.int64)
                if self.masked_gene_discrimination_enable:
                    masked_candidate_source = observed_nonzero_source[~source_selected_mask[observed_nonzero_source]]
                    masked_candidate_total += float(masked_candidate_source.size)
                    masked_source_selected = self._sample_source_positions(
                        masked_candidate_source,
                        probability=self.masked_gene_disc_probability,
                        rng=self._rng_for_feature(features[row_idx], row_offset=(row_idx + 2000)),
                    )
                visible_source_selected = np.empty((0,), dtype=np.int64)
                if self.visible_gene_discrimination_enable:
                    visible_blocked_mask = np.zeros((source_to_current.shape[0],), dtype=bool)
                    if masked_source_selected.size > 0:
                        visible_blocked_mask[masked_source_selected] = True
                    if self.visible_gene_disc_overlap_policy == "disjoint" and value_source_selected.size > 0:
                        visible_blocked_mask[value_source_selected] = True
                    visible_candidate_source = eligible_source[~visible_blocked_mask[eligible_source]]
                    visible_source_selected = self._sample_source_positions(
                        visible_candidate_source,
                        probability=self.visible_gene_disc_probability,
                        rng=self._rng_for_feature(features[row_idx], row_offset=(row_idx + 1000)),
                    )
                    if value_source_selected.size > 0 and visible_source_selected.size > 0:
                        visible_corruption_overlap_total += float(
                            np.intersect1d(value_source_selected, visible_source_selected).shape[0]
                        )
                if masked_source_selected.size > 0 and visible_source_selected.size > 0:
                    masked_visible_overlap_total += float(
                        np.intersect1d(masked_source_selected, visible_source_selected).shape[0]
                    )
                eligible_mask_np = ~row_special
                row_ctx = self._prepare_gene_disc_row_context(
                    row_gene_ids=row_gene_ids[eligible_mask_np],
                    row_source_rank=row_source[eligible_mask_np],
                    row_value_bins=row_value_bins[eligible_mask_np],
                )
                rng_row = self._rng_for_feature(features[row_idx], row_offset=(row_idx + 1000))
                for source_rank_value in visible_source_selected.tolist():
                    col_idx = int(source_to_current[int(source_rank_value)])
                    pos_gene_id = int(input_ids[row_idx, col_idx].item())
                    neg_gene_ids, neg_types = self._sample_gene_disc_negatives(
                        row_ctx=row_ctx,
                        pos_source_rank=int(source_rank_value),
                        pos_gene_id=pos_gene_id,
                        rng=rng_row,
                    )
                    row_items.append(int(row_idx))
                    col_items.append(int(col_idx))
                    pos_items.append(pos_gene_id)
                    neg_items.append(neg_gene_ids)
                    neg_type_items.append(neg_types)
                for source_rank_value in masked_source_selected.tolist():
                    col_idx = int(source_to_current[int(source_rank_value)])
                    pos_gene_id = int(input_ids[row_idx, col_idx].item())
                    neg_gene_ids, neg_types = self._sample_gene_disc_negatives(
                        row_ctx=row_ctx,
                        pos_source_rank=int(source_rank_value),
                        pos_gene_id=pos_gene_id,
                        rng=rng_row,
                    )
                    masked_row_items.append(int(row_idx))
                    masked_col_items.append(int(col_idx))
                    masked_pos_items.append(pos_gene_id)
                    masked_neg_items.append(neg_gene_ids)
                    masked_neg_type_items.append(neg_types)
                if masked_source_selected.size > 0:
                    masked_cols = torch.as_tensor(source_to_current[masked_source_selected], dtype=torch.long)
                    input_ids[row_idx, masked_cols] = self.mask_token_id
                    value_bins[row_idx, masked_cols] = self.value_mask_id
            masked_gene_candidate_n = torch.tensor(masked_candidate_total, dtype=torch.float32)
            masked_gene_visible_overlap_n = torch.tensor(masked_visible_overlap_total, dtype=torch.float32)
            visible_gene_corruption_overlap_n = torch.tensor(
                visible_corruption_overlap_total,
                dtype=torch.float32,
            )
            if row_items:
                visible_gene_disc_row_idx = torch.as_tensor(row_items, dtype=torch.long)
                visible_gene_disc_col_idx = torch.as_tensor(col_items, dtype=torch.long)
                visible_gene_disc_pos_gene_ids = torch.as_tensor(pos_items, dtype=torch.long)
                visible_gene_disc_neg_gene_ids = torch.as_tensor(np.stack(neg_items, axis=0), dtype=torch.long)
                visible_gene_disc_neg_types = torch.as_tensor(np.stack(neg_type_items, axis=0), dtype=torch.long)
            if masked_row_items:
                masked_gene_disc_row_idx = torch.as_tensor(masked_row_items, dtype=torch.long)
                masked_gene_disc_col_idx = torch.as_tensor(masked_col_items, dtype=torch.long)
                masked_gene_disc_pos_gene_ids = torch.as_tensor(masked_pos_items, dtype=torch.long)
                masked_gene_disc_neg_gene_ids = torch.as_tensor(np.stack(masked_neg_items, axis=0), dtype=torch.long)
                masked_gene_disc_neg_types = torch.as_tensor(np.stack(masked_neg_type_items, axis=0), dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if token_type_ids is not None:
            batch["token_type_ids"] = token_type_ids
        if value_bins is not None:
            batch["value_bins"] = value_bins
            batch["value_labels"] = value_labels
        if self.visible_gene_discrimination_enable:
            batch["visible_gene_disc_row_idx"] = visible_gene_disc_row_idx
            batch["visible_gene_disc_col_idx"] = visible_gene_disc_col_idx
            batch["visible_gene_disc_pos_gene_ids"] = visible_gene_disc_pos_gene_ids
            batch["visible_gene_disc_neg_gene_ids"] = visible_gene_disc_neg_gene_ids
            batch["visible_gene_disc_neg_types"] = visible_gene_disc_neg_types
            batch["visible_gene_corruption_overlap_n"] = visible_gene_corruption_overlap_n
        if self.masked_gene_discrimination_enable:
            batch["masked_gene_disc_row_idx"] = masked_gene_disc_row_idx
            batch["masked_gene_disc_col_idx"] = masked_gene_disc_col_idx
            batch["masked_gene_disc_pos_gene_ids"] = masked_gene_disc_pos_gene_ids
            batch["masked_gene_disc_neg_gene_ids"] = masked_gene_disc_neg_gene_ids
            batch["masked_gene_disc_neg_types"] = masked_gene_disc_neg_types
            batch["masked_gene_candidate_n"] = masked_gene_candidate_n
            batch["masked_gene_visible_overlap_n"] = masked_gene_visible_overlap_n
        if self.rtd_enable:
            batch["rtd_labels"] = rtd_labels
            batch["rtd_token_types"] = rtd_token_types
        example_loss_weights = self._example_loss_weights(features)
        if example_loss_weights is not None:
            batch["example_loss_weights"] = example_loss_weights
        if self.gene_pool_contrastive_enabled:
            batch["original_input_ids"] = labels.masked_fill(labels.eq(-100), 0) + input_ids.masked_fill(labels.ne(-100), 0)
            batch["original_attention_mask"] = attention_mask.clone()
        if self.gene_loading_target_store is not None:
            batch["gene_loading_original_input_ids"] = original_input_ids
            batch["gene_loading_original_attention_mask"] = original_attention_mask
        if self.view_consistency_enable:
            batch.update(
                self._build_perturbed_view(
                    input_ids=original_input_ids,
                    attention_mask=original_attention_mask,
                    value_bins=original_value_bins,
                    token_type_ids=original_token_type_ids,
                )
            )
        batch.update(self._build_program_target_batch(features))
        batch.update(
            self._build_gene_loading_target_batch(
                original_input_ids=original_input_ids,
                original_attention_mask=original_attention_mask,
            )
        )
        if any("raw_cell_count" in feature for feature in features):
            batch["raw_cell_count"] = torch.as_tensor(
                [int(feature.get("raw_cell_count", 0)) for feature in features],
                dtype=torch.long,
            )
        if prefix is None:
            if self.dual_mask_views and build_dual:
                dual_batch = self._build_view_batch(
                    features,
                    token_type_id=token_type_id,
                    build_dual=False,
                )
                batch.update({f"dual_{key}": value for key, value in dual_batch.items()})
            return batch
        return {f"{prefix}_{key}": value for key, value in batch.items()}

    def _build_paired_features(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.paired_lookup is None:
            raise ValueError("paired_lookup is required when paired_mode is enabled")
        paired_features: list[dict[str, Any]] = []
        for feature in features:
            if "dataset_idx" not in feature or "metacell_local_idx" not in feature:
                raise KeyError("dataset_idx and metacell_local_idx columns are required when paired_mode is enabled")
            paired_features.append(
                self.paired_lookup.get_example(
                    dataset_idx=int(feature["dataset_idx"]),
                    metacell_local_idx=int(feature["metacell_local_idx"]),
                )
            )
        return paired_features

    def _build_anchor_features(self, features: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], torch.Tensor]:
        if self.paired_lookup is None:
            raise ValueError("paired_lookup is required when paired_mode is enabled")
        key_to_anchor_idx: dict[tuple[int, int], int] = {}
        anchor_features: list[dict[str, Any]] = []
        sc_anchor_indices = torch.empty((len(features),), dtype=torch.long)
        for row_idx, feature in enumerate(features):
            if "dataset_idx" not in feature or "metacell_local_idx" not in feature:
                raise KeyError("dataset_idx and metacell_local_idx columns are required when paired_mode is enabled")
            key = (int(feature["dataset_idx"]), int(feature["metacell_local_idx"]))
            anchor_idx = key_to_anchor_idx.get(key)
            if anchor_idx is None:
                anchor_idx = len(anchor_features)
                key_to_anchor_idx[key] = anchor_idx
                anchor_features.append(
                    self.paired_lookup.get_example(
                        dataset_idx=key[0],
                        metacell_local_idx=key[1],
                    )
                )
            sc_anchor_indices[row_idx] = int(anchor_idx)
        return anchor_features, sc_anchor_indices

    def _build_original_only_batch(self, features: list[dict[str, Any]], *, prefix: str) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        input_ids = torch.full((batch_size, self.max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, self.max_len), dtype=torch.long)
        for row_idx, feature in enumerate(features):
            ids = self._tensorize_1d(feature["input_ids"])[: self.max_len]
            n = int(ids.numel())
            if n > 0:
                input_ids[row_idx, :n] = ids
            if "attention_mask" in feature:
                att = self._tensorize_1d(feature["attention_mask"])[: self.max_len]
                attention_mask[row_idx, : int(att.numel())] = att
            elif "length" in feature:
                length = min(int(feature["length"]), self.max_len)
                attention_mask[row_idx, :length] = 1
            else:
                attention_mask[row_idx] = input_ids[row_idx].ne(self.pad_token_id).long()
        return {
            f"{prefix}_original_input_ids": input_ids,
            f"{prefix}_original_attention_mask": attention_mask,
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not self.is_paired:
            return self._build_view_batch(features, token_type_id=self.domain_token_type_id)

        sc_batch = self._build_view_batch(features, prefix="sc", token_type_id=self.sc_token_type_id)
        paired_features = self._build_paired_features(features)
        mc_batch = self._build_view_batch(paired_features, prefix="mc", token_type_id=self.mc_token_type_id)
        out = dict(sc_batch)
        out.update(mc_batch)
        if self.gene_pool_contrastive_enabled:
            anchor_features, sc_anchor_indices = self._build_anchor_features(features)
            out["sc_anchor_indices"] = sc_anchor_indices
            out.update(self._build_original_only_batch(anchor_features, prefix="anchor"))
        return out


class StochasticSetValueCollator:
    def __init__(
        self,
        *,
        token_dictionary: dict[str, int],
        max_len: int,
        value_pad_id: int,
        value_mask_id: int,
        value_mask_ratio: float,
        input_subset_mode: str,
        value_target_mode: str,
        subset_top_count: int,
        subset_random_count: int,
        subset_fill_count: int,
        target_nonzero_fraction: float,
        randorder_seed: int,
        gene_mlm_enabled: bool,
        use_value_input: bool,
        use_value_loss: bool,
        paired_mode: str = "off",
    ):
        if bool(gene_mlm_enabled):
            raise ValueError("StochasticSetValueCollator requires gene_mlm_enabled=False")
        if not bool(use_value_input) or not bool(use_value_loss):
            raise ValueError("StochasticSetValueCollator requires use_value_input/use_value_loss enabled")
        if str(paired_mode).strip().lower() != "off":
            raise ValueError("StochasticSetValueCollator does not support paired_mode")
        self.token_dictionary = token_dictionary
        self.max_len = int(max_len)
        self.value_pad_id = int(value_pad_id)
        self.value_mask_id = int(value_mask_id)
        self.value_mask_ratio = float(value_mask_ratio)
        self.input_subset_mode = str(input_subset_mode).strip().lower()
        self.value_target_mode = str(value_target_mode).strip().lower()
        self.subset_top_count = int(subset_top_count)
        self.subset_random_count = int(subset_random_count)
        self.subset_fill_count = int(subset_fill_count)
        self.target_nonzero_fraction = float(target_nonzero_fraction)
        self.randorder_seed = int(randorder_seed)
        self.cls_token_id = int(token_dictionary["<cls>"])
        self.gene_universe = gene_token_universe_from_token_dictionary(token_dictionary)
        self.is_paired = False

    def _rng_for_feature(self, feature: dict[str, Any], *, row_offset: int) -> np.random.Generator:
        dataset_idx = int(feature.get("dataset_idx", 0))
        metacell_local_idx = int(feature.get("metacell_local_idx", row_offset))
        stochastic_jitter = 0 if self.input_subset_mode == "static_top_nonzero_4096" else random.randint(0, 2_147_483_647)
        seed = (
            self.randorder_seed
            + (dataset_idx * 1_000_003)
            + (metacell_local_idx * 10_007)
            + (row_offset * 97)
            + stochastic_jitter
        ) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        input_ids = torch.empty((batch_size, self.max_len), dtype=torch.long)
        attention_mask = torch.empty((batch_size, self.max_len), dtype=torch.long)
        value_bins = torch.empty((batch_size, self.max_len), dtype=torch.long)
        value_labels = torch.full((batch_size, self.max_len), -100, dtype=torch.long)
        labels = torch.full((batch_size, self.max_len), -100, dtype=torch.long)
        subset_source_ids = torch.full((batch_size, self.max_len), -1, dtype=torch.long)
        pre_shuffle_rank_bucket = torch.full((batch_size, self.max_len), -1, dtype=torch.long)

        for row_idx, feature in enumerate(features):
            rng = self._rng_for_feature(feature, row_offset=row_idx)
            materialized = sample_candidate_subset(
                feature,
                cls_token_id=self.cls_token_id,
                max_input_size=self.max_len,
                value_pad_id=self.value_pad_id,
                gene_universe=self.gene_universe,
                input_subset_mode=self.input_subset_mode,
                subset_top_count=self.subset_top_count,
                subset_random_count=self.subset_random_count,
                subset_fill_count=self.subset_fill_count,
                rng=rng,
            )
            masked = apply_value_masking(
                materialized,
                value_mask_ratio=self.value_mask_ratio,
                value_mask_id=self.value_mask_id,
                value_target_mode=self.value_target_mode,
                target_nonzero_fraction=self.target_nonzero_fraction,
                rng=rng,
            )

            input_ids[row_idx] = torch.from_numpy(np.asarray(materialized["input_ids"], dtype=np.int64))
            attention_mask[row_idx] = torch.from_numpy(np.asarray(materialized["attention_mask"], dtype=np.int64))
            value_bins[row_idx] = torch.from_numpy(np.asarray(masked["value_bins"], dtype=np.int64))
            value_labels[row_idx] = torch.from_numpy(np.asarray(masked["value_labels"], dtype=np.int64))
            subset_source_ids[row_idx] = torch.from_numpy(np.asarray(materialized["subset_source_ids"], dtype=np.int64))
            pre_shuffle_rank_bucket[row_idx] = torch.from_numpy(
                np.asarray(materialized["pre_shuffle_rank_bucket"], dtype=np.int64)
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "value_bins": value_bins,
            "value_labels": value_labels,
            "subset_source_ids": subset_source_ids,
            "pre_shuffle_rank_bucket": pre_shuffle_rank_bucket,
        }


def _make_metric_bucket() -> dict[str, Any]:
    keys = [
        "steps",
        "total_loss_sum",
        "total_loss_steps",
        "mlm_loss_sum",
        "value_loss_sum",
        "value_loss_steps",
        "value_correct_n",
        "value_masked_n",
        "program_loss_sum",
        "program_loss_steps",
        "program_target_n",
        "program_target_candidate_n",
        "program_abs_error_sum",
        "program_squared_error_sum",
        "program_pred_norm_sum",
        "program_target_norm_sum",
        "gene_loading_loss_sum",
        "gene_loading_loss_steps",
        "gene_loading_target_n",
        "gene_loading_target_candidate_n",
        "gene_loading_abs_error_sum",
        "gene_loading_squared_error_sum",
        "gene_loading_pred_norm_sum",
        "gene_loading_target_norm_sum",
        "ordinal_huber_sum",
        "ordinal_huber_steps",
        "ordinal_abs_error_sum",
        "ordinal_squared_error_sum",
        "masked_correct_n",
        "masked_top5_correct_n",
        "masked_top10_correct_n",
        "masked_n",
        "gene_eligible_n",
        "value_candidate_n",
        "visible_gene_candidate_n",
        "visible_gene_corruption_overlap_n",
        "visible_gene_loss_sum",
        "visible_gene_loss_steps",
        "visible_gene_target_n",
        "visible_gene_pos_score_sum",
        "visible_gene_neg_score_sum",
        "visible_gene_pos_correct_n",
        "visible_gene_neg_correct_n",
        "visible_gene_neg_n",
        "visible_gene_same_row_neg_correct_n",
        "visible_gene_same_row_neg_n",
        "visible_gene_support_neg_correct_n",
        "visible_gene_support_neg_n",
        "visible_gene_random_neg_correct_n",
        "visible_gene_random_neg_n",
        "visible_gene_auc_sum",
        "visible_gene_positive_rank_sum",
        "visible_gene_top1_n",
        "masked_gene_candidate_n",
        "masked_gene_visible_overlap_n",
        "masked_gene_loss_sum",
        "masked_gene_loss_steps",
        "masked_gene_target_n",
        "masked_gene_pos_score_sum",
        "masked_gene_neg_score_sum",
        "masked_gene_pos_correct_n",
        "masked_gene_neg_correct_n",
        "masked_gene_neg_n",
        "masked_gene_same_row_neg_correct_n",
        "masked_gene_same_row_neg_n",
        "masked_gene_support_neg_correct_n",
        "masked_gene_support_neg_n",
        "masked_gene_random_neg_correct_n",
        "masked_gene_random_neg_n",
        "masked_gene_auc_sum",
        "masked_gene_positive_rank_sum",
        "masked_gene_top1_n",
        "view_consistency_loss_sum",
        "view_consistency_loss_steps",
        "view_cosine_sum",
        "view_pair_n",
        "sc_steps",
        "sc_total_loss_sum",
        "sc_total_loss_steps",
        "sc_mlm_loss_sum",
        "sc_value_loss_sum",
        "sc_value_loss_steps",
        "sc_value_correct_n",
        "sc_value_masked_n",
        "sc_program_loss_sum",
        "sc_program_loss_steps",
        "sc_program_target_n",
        "sc_program_target_candidate_n",
        "sc_program_abs_error_sum",
        "sc_program_squared_error_sum",
        "sc_program_pred_norm_sum",
        "sc_program_target_norm_sum",
        "sc_gene_loading_loss_sum",
        "sc_gene_loading_loss_steps",
        "sc_gene_loading_target_n",
        "sc_gene_loading_target_candidate_n",
        "sc_gene_loading_abs_error_sum",
        "sc_gene_loading_squared_error_sum",
        "sc_gene_loading_pred_norm_sum",
        "sc_gene_loading_target_norm_sum",
        "sc_ordinal_huber_sum",
        "sc_ordinal_huber_steps",
        "sc_ordinal_abs_error_sum",
        "sc_ordinal_squared_error_sum",
        "sc_masked_correct_n",
        "sc_masked_top5_correct_n",
        "sc_masked_top10_correct_n",
        "sc_masked_n",
        "sc_gene_eligible_n",
        "sc_value_candidate_n",
        "sc_visible_gene_candidate_n",
        "sc_visible_gene_corruption_overlap_n",
        "sc_visible_gene_loss_sum",
        "sc_visible_gene_loss_steps",
        "sc_visible_gene_target_n",
        "sc_visible_gene_pos_score_sum",
        "sc_visible_gene_neg_score_sum",
        "sc_visible_gene_pos_correct_n",
        "sc_visible_gene_neg_correct_n",
        "sc_visible_gene_neg_n",
        "sc_visible_gene_same_row_neg_correct_n",
        "sc_visible_gene_same_row_neg_n",
        "sc_visible_gene_support_neg_correct_n",
        "sc_visible_gene_support_neg_n",
        "sc_visible_gene_random_neg_correct_n",
        "sc_visible_gene_random_neg_n",
        "sc_visible_gene_auc_sum",
        "sc_visible_gene_positive_rank_sum",
        "sc_visible_gene_top1_n",
        "sc_masked_gene_candidate_n",
        "sc_masked_gene_visible_overlap_n",
        "sc_masked_gene_loss_sum",
        "sc_masked_gene_loss_steps",
        "sc_masked_gene_target_n",
        "sc_masked_gene_pos_score_sum",
        "sc_masked_gene_neg_score_sum",
        "sc_masked_gene_pos_correct_n",
        "sc_masked_gene_neg_correct_n",
        "sc_masked_gene_neg_n",
        "sc_masked_gene_same_row_neg_correct_n",
        "sc_masked_gene_same_row_neg_n",
        "sc_masked_gene_support_neg_correct_n",
        "sc_masked_gene_support_neg_n",
        "sc_masked_gene_random_neg_correct_n",
        "sc_masked_gene_random_neg_n",
        "sc_masked_gene_auc_sum",
        "sc_masked_gene_positive_rank_sum",
        "sc_masked_gene_top1_n",
        "sc_view_consistency_loss_sum",
        "sc_view_consistency_loss_steps",
        "sc_view_cosine_sum",
        "sc_view_pair_n",
        "mc_steps",
        "mc_total_loss_sum",
        "mc_total_loss_steps",
        "mc_mlm_loss_sum",
        "mc_value_loss_sum",
        "mc_value_loss_steps",
        "mc_value_correct_n",
        "mc_value_masked_n",
        "mc_program_loss_sum",
        "mc_program_loss_steps",
        "mc_program_target_n",
        "mc_program_target_candidate_n",
        "mc_program_abs_error_sum",
        "mc_program_squared_error_sum",
        "mc_program_pred_norm_sum",
        "mc_program_target_norm_sum",
        "mc_gene_loading_loss_sum",
        "mc_gene_loading_loss_steps",
        "mc_gene_loading_target_n",
        "mc_gene_loading_target_candidate_n",
        "mc_gene_loading_abs_error_sum",
        "mc_gene_loading_squared_error_sum",
        "mc_gene_loading_pred_norm_sum",
        "mc_gene_loading_target_norm_sum",
        "mc_ordinal_huber_sum",
        "mc_ordinal_huber_steps",
        "mc_ordinal_abs_error_sum",
        "mc_ordinal_squared_error_sum",
        "mc_masked_correct_n",
        "mc_masked_top5_correct_n",
        "mc_masked_top10_correct_n",
        "mc_masked_n",
        "mc_gene_eligible_n",
        "mc_value_candidate_n",
        "mc_visible_gene_candidate_n",
        "mc_visible_gene_corruption_overlap_n",
        "mc_visible_gene_loss_sum",
        "mc_visible_gene_loss_steps",
        "mc_visible_gene_target_n",
        "mc_visible_gene_pos_score_sum",
        "mc_visible_gene_neg_score_sum",
        "mc_visible_gene_pos_correct_n",
        "mc_visible_gene_neg_correct_n",
        "mc_visible_gene_neg_n",
        "mc_visible_gene_same_row_neg_correct_n",
        "mc_visible_gene_same_row_neg_n",
        "mc_visible_gene_support_neg_correct_n",
        "mc_visible_gene_support_neg_n",
        "mc_visible_gene_random_neg_correct_n",
        "mc_visible_gene_random_neg_n",
        "mc_visible_gene_auc_sum",
        "mc_visible_gene_positive_rank_sum",
        "mc_visible_gene_top1_n",
        "mc_masked_gene_candidate_n",
        "mc_masked_gene_visible_overlap_n",
        "mc_masked_gene_loss_sum",
        "mc_masked_gene_loss_steps",
        "mc_masked_gene_target_n",
        "mc_masked_gene_pos_score_sum",
        "mc_masked_gene_neg_score_sum",
        "mc_masked_gene_pos_correct_n",
        "mc_masked_gene_neg_correct_n",
        "mc_masked_gene_neg_n",
        "mc_masked_gene_same_row_neg_correct_n",
        "mc_masked_gene_same_row_neg_n",
        "mc_masked_gene_support_neg_correct_n",
        "mc_masked_gene_support_neg_n",
        "mc_masked_gene_random_neg_correct_n",
        "mc_masked_gene_random_neg_n",
        "mc_masked_gene_auc_sum",
        "mc_masked_gene_positive_rank_sum",
        "mc_masked_gene_top1_n",
        "mc_view_consistency_loss_sum",
        "mc_view_consistency_loss_steps",
        "mc_view_cosine_sum",
        "mc_view_pair_n",
        "gene_pool_contrastive_loss_sum",
        "gene_pool_contrastive_steps",
        "effective_gene_pool_lambda_sum",
        "effective_gene_pool_lambda_steps",
        "eligible_coord_n",
        "observed_nonzero_correct_n",
        "observed_nonzero_n",
        "observed_nonzero_abs_error_sum",
        "observed_nonzero_squared_error_sum",
        "within_1_bin_correct_n",
        "within_2_bin_correct_n",
        "bin0_correct_n",
        "bin0_target_n",
        "bin0_pred_n",
    ]
    bucket: dict[str, Any] = {key: 0.0 for key in keys}
    bucket["target_hist"] = np.zeros((65,), dtype=np.float64)
    bucket["pred_hist"] = np.zeros((65,), dtype=np.float64)
    bucket["correct_hist"] = np.zeros((65,), dtype=np.float64)
    bucket["dynamic_scalar_sums"] = {}
    bucket["dynamic_scalar_steps"] = {}
    rtd_keys = [
        "rtd_loss_sum",
        "rtd_loss_steps",
        "rtd_correct_n",
        "rtd_target_n",
        "rtd_pos_n",
        "rtd_neg_n",
        "rtd_negative_random_correct_n",
        "rtd_negative_random_n",
        "rtd_positive_unchanged_correct_n",
        "rtd_positive_unchanged_n",
        "rtd_positive_visible_correct_n",
        "rtd_positive_visible_n",
        "rtd_auc_sum",
        "rtd_auc_pair_n",
    ]
    for key in rtd_keys:
        bucket.setdefault(key, 0.0)
        bucket.setdefault(f"sc_{key}", 0.0)
        bucket.setdefault(f"mc_{key}", 0.0)
    return bucket


class HFMLMTrainer(Trainer):
    _SINGLE_MODEL_INPUT_KEYS = {
        "input_ids",
        "attention_mask",
        "labels",
        "token_type_ids",
        "value_bins",
        "value_labels",
        "program_targets",
        "program_target_mask",
        "gene_loading_targets",
        "gene_loading_target_mask",
        "gene_loading_original_input_ids",
        "gene_loading_original_attention_mask",
        "visible_gene_disc_row_idx",
        "visible_gene_disc_col_idx",
        "visible_gene_disc_pos_gene_ids",
        "visible_gene_disc_neg_gene_ids",
        "visible_gene_disc_neg_types",
        "masked_gene_disc_row_idx",
        "masked_gene_disc_col_idx",
        "masked_gene_disc_pos_gene_ids",
        "masked_gene_disc_neg_gene_ids",
        "masked_gene_disc_neg_types",
        "rtd_labels",
        "rtd_token_types",
        "gene_pool_original_input_ids",
        "gene_pool_original_attention_mask",
        "gene_pool_anchor_input_ids",
        "gene_pool_anchor_attention_mask",
        "gene_pool_positive_indices",
        "view_input_ids",
        "view_attention_mask",
        "view_token_type_ids",
        "view_value_bins",
        "example_loss_weights",
    }

    def __init__(
        self,
        *args,
        boundary_saves: dict[int, str] | None = None,
        time_boundary_saves: dict[float, str] | None = None,
        stop_after_wall_seconds: float | None = None,
        init_optimizer_state_path: str | Path | None = None,
        init_scheduler_state_path: str | Path | None = None,
        scheduler_total_steps_override: int | None = None,
        stream_eval_masked_acc_only: bool = False,
        paired_mc_loss_weight: float = 1.0,
        gene_pool_contrastive_enabled: bool = False,
        gene_pool_temperature: float = 0.07,
        gene_pool_stopgrad_anchor: bool = False,
        gene_pool_lambda_points: tuple[tuple[int, float], ...] | None = None,
        embedding_lr_multiplier: float = 1.0,
        embedding_lr_multiplier_after_step: int | None = None,
        embedding_lr_multiplier_after: float | None = None,
        force_embedding_param_group: bool = False,
        reset_scheduler_for_epoch: bool = False,
        epoch_scheduler_steps: int | None = None,
        epoch_warmup_steps: int | None = None,
        epoch_start_learning_rate: float | None = None,
        lr_scheduler_type: str = "linear",
        cosine_min_lr_ratio: float = 0.0,
        exact_train_coverage_tracker: ExactCoverageTracker | None = None,
        training_metric_local_accumulation: bool = False,
        **kwargs,
    ):
        self._boundary_saves = dict(boundary_saves or {})
        self._time_boundary_saves = dict(time_boundary_saves or {})
        self._saved_boundary_steps: set[int] = set()
        self._saved_time_boundary_names: set[str] = set()
        self._train_start_wall = time.time()
        self._seen_non_pad_tokens = 0.0
        self._seen_sc_non_pad_tokens = 0.0
        self._seen_mc_non_pad_tokens = 0.0
        self._seen_examples = 0.0
        self._seen_raw_cells = 0.0
        self._exact_train_coverage_tracker = exact_train_coverage_tracker
        self._training_metric_local_accumulation = bool(training_metric_local_accumulation)
        self._exact_train_coverage_resume_summary: dict[str, Any] | None = None
        if stop_after_wall_seconds is not None:
            if not math.isfinite(float(stop_after_wall_seconds)) or float(stop_after_wall_seconds) <= 0.0:
                raise ValueError("stop_after_wall_seconds must be positive finite")
            self._stop_after_wall_seconds = float(stop_after_wall_seconds)
        else:
            self._stop_after_wall_seconds = None
        self._init_optimizer_state_path = (
            Path(init_optimizer_state_path).expanduser().resolve() if init_optimizer_state_path else None
        )
        self._init_scheduler_state_path = (
            Path(init_scheduler_state_path).expanduser().resolve() if init_scheduler_state_path else None
        )
        self._scheduler_total_steps_override = (
            int(scheduler_total_steps_override) if scheduler_total_steps_override else None
        )
        self._stream_eval_masked_acc_only = bool(stream_eval_masked_acc_only)
        self._paired_mc_loss_weight = float(paired_mc_loss_weight)
        self._gene_pool_contrastive_enabled = bool(gene_pool_contrastive_enabled)
        self._gene_pool_temperature = float(gene_pool_temperature)
        self._gene_pool_stopgrad_anchor = bool(gene_pool_stopgrad_anchor)
        self._gene_pool_lambda_points = tuple(gene_pool_lambda_points or ())
        self._embedding_lr_multiplier = float(embedding_lr_multiplier)
        if not math.isfinite(self._embedding_lr_multiplier) or self._embedding_lr_multiplier <= 0.0:
            raise ValueError("embedding_lr_multiplier must be positive finite")
        self._embedding_lr_group_applied = False
        self._embedding_lr_multiplier_after_step = (
            int(embedding_lr_multiplier_after_step) if embedding_lr_multiplier_after_step is not None else None
        )
        self._embedding_lr_multiplier_after = (
            float(embedding_lr_multiplier_after) if embedding_lr_multiplier_after is not None else None
        )
        if self._embedding_lr_multiplier_after_step is not None:
            if self._embedding_lr_multiplier_after is None or self._embedding_lr_multiplier_after <= 0.0:
                raise ValueError("embedding_lr_multiplier_after_step requires positive embedding_lr_multiplier_after")
        self._embedding_lr_schedule_switched = False
        self._force_embedding_param_group = bool(force_embedding_param_group)
        self._reset_scheduler_for_epoch = bool(reset_scheduler_for_epoch)
        self._epoch_scheduler_steps = int(epoch_scheduler_steps) if epoch_scheduler_steps else None
        self._epoch_warmup_steps = int(epoch_warmup_steps) if epoch_warmup_steps is not None else None
        self._epoch_start_learning_rate = (
            float(epoch_start_learning_rate) if epoch_start_learning_rate is not None else None
        )
        self._runtime_lr_scheduler_type = str(lr_scheduler_type).strip().lower()
        self._cosine_min_lr_ratio = float(cosine_min_lr_ratio)
        if self._runtime_lr_scheduler_type not in {"linear", "cosine_with_floor"}:
            raise ValueError(f"unsupported lr_scheduler_type={lr_scheduler_type}")
        if not 0.0 <= self._cosine_min_lr_ratio <= 1.0:
            raise ValueError("cosine_min_lr_ratio must be in [0, 1]")
        self._fresh_init_states_loaded = False
        self._reset_train_metric_sums()
        self._reset_eval_metric_sums()
        super().__init__(*args, **kwargs)
        self._paired_collator_enabled = bool(getattr(self.data_collator, "is_paired", False))
        self._saved_boundary_steps = self._discover_saved_boundary_steps()
        self._saved_time_boundary_names = self._discover_saved_time_boundary_names()

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)
        tracker = self._exact_train_coverage_tracker
        if tracker is None:
            return
        checkpoint_dir = Path(self._get_output_dir(trial=trial)) / f"checkpoint-{self.state.global_step}"
        tracker.save(
            checkpoint_dir,
            process_index=int(self.args.process_index),
            global_step=int(self.state.global_step),
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def load_exact_train_coverage_state(self, checkpoint_dir: str | Path | None) -> None:
        tracker = self._exact_train_coverage_tracker
        if tracker is None or checkpoint_dir is None:
            return
        checkpoint = Path(checkpoint_dir)
        try:
            checkpoint_step = int(checkpoint.name.rsplit("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid checkpoint directory for exact coverage: {checkpoint}") from exc
        tracker.load(
            checkpoint,
            process_index=int(self.args.process_index),
            global_step=checkpoint_step,
        )
        summary = tracker.summarize(device=self.args.device, global_step=checkpoint_step)
        self._seen_examples = float(summary["actual_examples"])
        self._exact_train_coverage_resume_summary = summary

    def summarize_exact_train_coverage(self) -> dict[str, Any] | None:
        tracker = self._exact_train_coverage_tracker
        if tracker is None:
            return None
        return tracker.summarize(device=self.args.device, global_step=int(self.state.global_step))

    def _record_exact_train_coverage(self, inputs: dict[str, Any]) -> None:
        sample_ids = inputs.pop(EXACT_COVERAGE_SAMPLE_IDS_KEY, None)
        tracker = self._exact_train_coverage_tracker
        if tracker is None:
            if sample_ids is not None:
                raise RuntimeError("received exact coverage sample IDs without an enabled tracker")
            return
        if sample_ids is None:
            raise RuntimeError("exact coverage training batch is missing sample IDs")
        tracker.record(sample_ids)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self._embedding_lr_group_applied or (
            math.isclose(self._embedding_lr_multiplier, 1.0)
            and not self._force_embedding_param_group
        ):
            return optimizer
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        embedding_param = model_obj.get_token_embedding_layer().weight
        source_group = None
        for group in optimizer.param_groups:
            if any(param is embedding_param for param in group["params"]):
                source_group = group
                break
        if source_group is None:
            raise ValueError("tied gene embedding parameter was not found in optimizer groups")
        remaining = [param for param in source_group["params"] if param is not embedding_param]
        source_group["params"] = remaining
        new_group = {key: value for key, value in source_group.items() if key != "params"}
        new_group["params"] = [embedding_param]
        new_group["lr"] = float(source_group["lr"]) * self._embedding_lr_multiplier
        if "initial_lr" in new_group:
            new_group["initial_lr"] = float(source_group["initial_lr"]) * self._embedding_lr_multiplier
        optimizer.add_param_group(new_group)
        self._embedding_lr_group_applied = True
        return optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None):
        if self._runtime_lr_scheduler_type != "cosine_with_floor":
            return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        optimizer = self.optimizer if optimizer is None else optimizer
        warmup_steps = int(self.args.get_warmup_steps(int(num_training_steps)))
        floor = float(self._cosine_min_lr_ratio)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, int(num_training_steps) - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + ((1.0 - floor) * cosine)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        effective_num_training_steps = (
            self._scheduler_total_steps_override
            if self._scheduler_total_steps_override is not None
            else int(num_training_steps)
        )
        return super().create_optimizer_and_scheduler(num_training_steps=effective_num_training_steps)

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            rng_file = os.path.join(checkpoint, f"rng_state_{self.args.process_index}.pth")
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            return

        # PyTorch 2.6 changed torch.load default to weights_only=True, but trainer RNG state
        # checkpoints contain Python/Numpy objects and must be loaded as full pickles.
        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        if torch.cuda.is_available() and checkpoint_rng_state.get("cuda") is not None:
            try:
                if self.args.world_size > 1:
                    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
                else:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
            except Exception as exc:
                print(
                    "[warn] failed to restore CUDA RNG state; resume will continue without exact RNG replay: "
                    f"{exc}",
                    flush=True,
                )

    def _reset_train_metric_sums(self) -> None:
        self._train_metric_sums = _make_metric_bucket()

    def _reset_eval_metric_sums(self) -> None:
        self._eval_metric_sums = _make_metric_bucket()

    def _reduce_scalar(self, value: torch.Tensor | float | None, *, average: bool) -> float | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            scalar = value.detach()
            if scalar.ndim > 0:
                scalar = scalar.sum()
            device = scalar.device
            scalar = scalar.to(dtype=torch.float64)
        else:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            scalar = torch.tensor(float(value), device=device, dtype=torch.float64)
        local_training_metric = bool(
            self._training_metric_local_accumulation
            and getattr(self, "model", None) is not None
            and self.model.training
        )
        if dist.is_available() and dist.is_initialized() and not local_training_metric:
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
            if average:
                scalar /= dist.get_world_size()
        elif dist.is_available() and dist.is_initialized() and local_training_metric and not average:
            scalar *= dist.get_world_size()
        return float(scalar.item())

    def _reduce_histogram(self, value: torch.Tensor | None) -> np.ndarray | None:
        if value is None:
            return None
        hist = value.detach()
        if hist.ndim != 1:
            hist = hist.reshape(-1)
        hist = hist.to(dtype=torch.float64)
        local_training_metric = bool(
            self._training_metric_local_accumulation
            and getattr(self, "model", None) is not None
            and self.model.training
        )
        if dist.is_available() and dist.is_initialized() and not local_training_metric:
            dist.all_reduce(hist, op=dist.ReduceOp.SUM)
        elif dist.is_available() and dist.is_initialized() and local_training_metric:
            hist *= dist.get_world_size()
        return hist.cpu().numpy()

    def _accumulate_dynamic_scalar_metrics(
        self,
        bucket: dict[str, Any],
        metrics: dict[str, torch.Tensor] | None,
        *,
        view: str | None = None,
    ) -> None:
        if not metrics:
            return
        metric_prefix = "" if view is None else f"{view}_"
        scalar_sums = bucket.setdefault("dynamic_scalar_sums", {})
        scalar_steps = bucket.setdefault("dynamic_scalar_steps", {})
        for key, value in metrics.items():
            reduced = self._reduce_scalar(value, average=True)
            if reduced is None:
                continue
            full_key = f"{metric_prefix}{key}" if view is not None else key
            scalar_sums[full_key] = float(scalar_sums.get(full_key, 0.0)) + float(reduced)
            scalar_steps[full_key] = float(scalar_steps.get(full_key, 0.0)) + 1.0
            if view is not None:
                scalar_sums[key] = float(scalar_sums.get(key, 0.0)) + float(reduced)
                scalar_steps[key] = float(scalar_steps.get(key, 0.0)) + 1.0

    def _filter_model_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in inputs.items() if key in self._SINGLE_MODEL_INPUT_KEYS}

    def _check_required_program_targets(self, inputs: dict[str, Any], *, prefix: str | None = None) -> None:
        if not bool(getattr(self.data_collator, "program_target_required_for_train", False)):
            return
        key = "program_target_mask" if prefix is None else f"{prefix}_program_target_mask"
        mask = inputs.get(key)
        if mask is None:
            raise KeyError(f"program targets are required for train batches, but {key} is missing")
        mask_bool = mask.to(dtype=torch.bool)
        if bool(mask_bool.all()):
            return
        missing_n = int((~mask_bool).sum().detach().cpu().item())
        batch_n = int(mask_bool.numel())
        label = "batch" if prefix is None else f"{prefix} batch"
        raise KeyError(f"missing required program targets in {label}: {missing_n}/{batch_n} rows")

    def _paired_batch_enabled(self, inputs: dict[str, Any] | None = None) -> bool:
        if inputs is not None and "sc_input_ids" in inputs and "mc_input_ids" in inputs:
            return True
        return bool(getattr(self, "_paired_collator_enabled", False))

    @staticmethod
    def _dual_batch_enabled(inputs: dict[str, Any] | None = None) -> bool:
        return bool(inputs is not None and "dual_input_ids" in inputs and "dual_labels" in inputs)

    def _extract_loss_tensor(self, outputs: Any) -> torch.Tensor | None:
        if outputs is None:
            return None
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        else:
            loss = getattr(outputs, "loss", None)
        if loss is None:
            return None
        return loss.mean() if torch.is_tensor(loss) and loss.ndim > 0 else loss

    def _extract_view_inputs(self, inputs: dict[str, Any], prefix: str) -> dict[str, torch.Tensor]:
        out = {
            "input_ids": inputs[f"{prefix}_input_ids"],
            "attention_mask": inputs[f"{prefix}_attention_mask"],
            "labels": inputs[f"{prefix}_labels"],
        }
        if f"{prefix}_token_type_ids" in inputs:
            out["token_type_ids"] = inputs[f"{prefix}_token_type_ids"]
        if f"{prefix}_value_bins" in inputs:
            out["value_bins"] = inputs[f"{prefix}_value_bins"]
        if f"{prefix}_value_labels" in inputs:
            out["value_labels"] = inputs[f"{prefix}_value_labels"]
        for key in (
            "visible_gene_disc_row_idx",
            "visible_gene_disc_col_idx",
            "visible_gene_disc_pos_gene_ids",
            "visible_gene_disc_neg_gene_ids",
            "visible_gene_disc_neg_types",
            "masked_gene_disc_row_idx",
            "masked_gene_disc_col_idx",
            "masked_gene_disc_pos_gene_ids",
            "masked_gene_disc_neg_gene_ids",
            "masked_gene_disc_neg_types",
            "rtd_labels",
            "rtd_token_types",
            "program_targets",
            "program_target_mask",
            "gene_loading_targets",
            "gene_loading_target_mask",
            "gene_loading_original_input_ids",
            "gene_loading_original_attention_mask",
            "view_input_ids",
            "view_attention_mask",
            "view_token_type_ids",
            "view_value_bins",
            "example_loss_weights",
        ):
            prefixed = f"{prefix}_{key}"
            if prefixed in inputs:
                out[key] = inputs[prefixed]
        if prefix in {"sc", "mc"}:
            if "sc_original_input_ids" in inputs:
                out["gene_pool_original_input_ids"] = inputs["sc_original_input_ids"]
            if "sc_original_attention_mask" in inputs:
                out["gene_pool_original_attention_mask"] = inputs["sc_original_attention_mask"]
            if "anchor_original_input_ids" in inputs:
                out["gene_pool_anchor_input_ids"] = inputs["anchor_original_input_ids"]
            if "anchor_original_attention_mask" in inputs:
                out["gene_pool_anchor_attention_mask"] = inputs["anchor_original_attention_mask"]
            if "sc_anchor_indices" in inputs:
                out["gene_pool_positive_indices"] = inputs["sc_anchor_indices"]
        return out

    def _extract_view_metric_inputs(self, inputs: dict[str, Any], prefix: str) -> dict[str, torch.Tensor]:
        out = self._extract_view_inputs(inputs, prefix)
        for key in (
            "visible_gene_corruption_overlap_n",
            "masked_gene_candidate_n",
            "masked_gene_visible_overlap_n",
        ):
            prefixed = f"{prefix}_{key}"
            if prefixed in inputs:
                out[key] = inputs[prefixed]
        return out

    def _accumulate_metric_sums_view(self, bucket: dict[str, float], outputs: Any, *, view: str | None = None) -> None:
        if outputs is None:
            return
        metric_prefix = "" if view is None else f"{view}_"
        total_loss = self._extract_loss_tensor(outputs)
        mlm_loss = getattr(outputs, "mlm_loss", None)
        value_loss = getattr(outputs, "value_loss", None)
        value_correct_n = getattr(outputs, "value_correct_n", None)
        value_masked_n = getattr(outputs, "value_masked_n", None)
        program_loss = getattr(outputs, "program_loss", None)
        program_target_n = getattr(outputs, "program_target_n", None)
        program_abs_error_sum = getattr(outputs, "program_abs_error_sum", None)
        program_squared_error_sum = getattr(outputs, "program_squared_error_sum", None)
        program_pred_norm_sum = getattr(outputs, "program_pred_norm_sum", None)
        program_target_norm_sum = getattr(outputs, "program_target_norm_sum", None)
        gene_loading_loss = getattr(outputs, "gene_loading_loss", None)
        gene_loading_target_n = getattr(outputs, "gene_loading_target_n", None)
        gene_loading_abs_error_sum = getattr(outputs, "gene_loading_abs_error_sum", None)
        gene_loading_squared_error_sum = getattr(outputs, "gene_loading_squared_error_sum", None)
        gene_loading_pred_norm_sum = getattr(outputs, "gene_loading_pred_norm_sum", None)
        gene_loading_target_norm_sum = getattr(outputs, "gene_loading_target_norm_sum", None)
        ordinal_huber_loss = getattr(outputs, "ordinal_huber_loss", None)
        ordinal_abs_error_sum = getattr(outputs, "ordinal_abs_error_sum", None)
        ordinal_squared_error_sum = getattr(outputs, "ordinal_squared_error_sum", None)
        visible_gene_loss = getattr(outputs, "visible_gene_loss", None)
        visible_gene_target_n = getattr(outputs, "visible_gene_target_n", None)
        visible_gene_pos_score_sum = getattr(outputs, "visible_gene_pos_score_sum", None)
        visible_gene_neg_score_sum = getattr(outputs, "visible_gene_neg_score_sum", None)
        visible_gene_pos_correct_n = getattr(outputs, "visible_gene_pos_correct_n", None)
        visible_gene_neg_correct_n = getattr(outputs, "visible_gene_neg_correct_n", None)
        visible_gene_neg_n = getattr(outputs, "visible_gene_neg_n", None)
        visible_gene_same_row_neg_correct_n = getattr(outputs, "visible_gene_same_row_neg_correct_n", None)
        visible_gene_same_row_neg_n = getattr(outputs, "visible_gene_same_row_neg_n", None)
        visible_gene_support_neg_correct_n = getattr(outputs, "visible_gene_support_neg_correct_n", None)
        visible_gene_support_neg_n = getattr(outputs, "visible_gene_support_neg_n", None)
        visible_gene_random_neg_correct_n = getattr(outputs, "visible_gene_random_neg_correct_n", None)
        visible_gene_random_neg_n = getattr(outputs, "visible_gene_random_neg_n", None)
        visible_gene_auc_sum = getattr(outputs, "visible_gene_auc_sum", None)
        visible_gene_positive_rank_sum = getattr(outputs, "visible_gene_positive_rank_sum", None)
        visible_gene_top1_n = getattr(outputs, "visible_gene_top1_n", None)
        masked_gene_loss = getattr(outputs, "masked_gene_loss", None)
        masked_gene_target_n = getattr(outputs, "masked_gene_target_n", None)
        masked_gene_pos_score_sum = getattr(outputs, "masked_gene_pos_score_sum", None)
        masked_gene_neg_score_sum = getattr(outputs, "masked_gene_neg_score_sum", None)
        masked_gene_pos_correct_n = getattr(outputs, "masked_gene_pos_correct_n", None)
        masked_gene_neg_correct_n = getattr(outputs, "masked_gene_neg_correct_n", None)
        masked_gene_neg_n = getattr(outputs, "masked_gene_neg_n", None)
        masked_gene_same_row_neg_correct_n = getattr(outputs, "masked_gene_same_row_neg_correct_n", None)
        masked_gene_same_row_neg_n = getattr(outputs, "masked_gene_same_row_neg_n", None)
        masked_gene_support_neg_correct_n = getattr(outputs, "masked_gene_support_neg_correct_n", None)
        masked_gene_support_neg_n = getattr(outputs, "masked_gene_support_neg_n", None)
        masked_gene_random_neg_correct_n = getattr(outputs, "masked_gene_random_neg_correct_n", None)
        masked_gene_random_neg_n = getattr(outputs, "masked_gene_random_neg_n", None)
        masked_gene_auc_sum = getattr(outputs, "masked_gene_auc_sum", None)
        masked_gene_positive_rank_sum = getattr(outputs, "masked_gene_positive_rank_sum", None)
        masked_gene_top1_n = getattr(outputs, "masked_gene_top1_n", None)
        rtd_loss = getattr(outputs, "rtd_loss", None)
        rtd_correct_n = getattr(outputs, "rtd_correct_n", None)
        rtd_target_n = getattr(outputs, "rtd_target_n", None)
        rtd_pos_n = getattr(outputs, "rtd_pos_n", None)
        rtd_neg_n = getattr(outputs, "rtd_neg_n", None)
        rtd_negative_random_correct_n = getattr(outputs, "rtd_negative_random_correct_n", None)
        rtd_negative_random_n = getattr(outputs, "rtd_negative_random_n", None)
        rtd_positive_unchanged_correct_n = getattr(outputs, "rtd_positive_unchanged_correct_n", None)
        rtd_positive_unchanged_n = getattr(outputs, "rtd_positive_unchanged_n", None)
        rtd_positive_visible_correct_n = getattr(outputs, "rtd_positive_visible_correct_n", None)
        rtd_positive_visible_n = getattr(outputs, "rtd_positive_visible_n", None)
        rtd_auc_sum = getattr(outputs, "rtd_auc_sum", None)
        rtd_auc_pair_n = getattr(outputs, "rtd_auc_pair_n", None)
        varcov_metrics = getattr(outputs, "varcov_metrics", None)
        view_consistency_loss = getattr(outputs, "view_consistency_loss", None)
        view_cosine_sum = getattr(outputs, "view_cosine_sum", None)
        view_pair_n = getattr(outputs, "view_pair_n", None)

        if total_loss is not None:
            reduced = float(self._reduce_scalar(total_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}total_loss_sum"] += reduced
            bucket[f"{metric_prefix}total_loss_steps"] += 1.0
            if view is not None:
                bucket["total_loss_sum"] += reduced
                bucket["total_loss_steps"] += 1.0
        if mlm_loss is not None:
            reduced = float(self._reduce_scalar(mlm_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}mlm_loss_sum"] += reduced
            bucket[f"{metric_prefix}steps"] += 1.0
            if view is not None:
                bucket["mlm_loss_sum"] += reduced
                bucket["steps"] += 1.0
        if value_loss is not None:
            reduced = float(self._reduce_scalar(value_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}value_loss_sum"] += reduced
            bucket[f"{metric_prefix}value_loss_steps"] += 1.0
            if view is not None:
                bucket["value_loss_sum"] += reduced
                bucket["value_loss_steps"] += 1.0
        if value_correct_n is not None and value_masked_n is not None:
            reduced_correct = float(self._reduce_scalar(value_correct_n, average=False) or 0.0)
            reduced_masked = float(self._reduce_scalar(value_masked_n, average=False) or 0.0)
            bucket[f"{metric_prefix}value_correct_n"] += reduced_correct
            bucket[f"{metric_prefix}value_masked_n"] += reduced_masked
            if view is not None:
                bucket["value_correct_n"] += reduced_correct
                bucket["value_masked_n"] += reduced_masked
        if program_loss is not None:
            reduced = float(self._reduce_scalar(program_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}program_loss_sum"] += reduced
            bucket[f"{metric_prefix}program_loss_steps"] += 1.0
            if view is not None:
                bucket["program_loss_sum"] += reduced
                bucket["program_loss_steps"] += 1.0
        program_pairs = [
            ("program_target_n", program_target_n),
            ("program_abs_error_sum", program_abs_error_sum),
            ("program_squared_error_sum", program_squared_error_sum),
            ("program_pred_norm_sum", program_pred_norm_sum),
            ("program_target_norm_sum", program_target_norm_sum),
        ]
        for key, value in program_pairs:
            if value is None:
                continue
            reduced = float(self._reduce_scalar(value, average=False) or 0.0)
            bucket[f"{metric_prefix}{key}"] += reduced
            if view is not None:
                bucket[key] += reduced
        if gene_loading_loss is not None:
            reduced = float(self._reduce_scalar(gene_loading_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}gene_loading_loss_sum"] += reduced
            bucket[f"{metric_prefix}gene_loading_loss_steps"] += 1.0
            if view is not None:
                bucket["gene_loading_loss_sum"] += reduced
                bucket["gene_loading_loss_steps"] += 1.0
        gene_loading_pairs = [
            ("gene_loading_target_n", gene_loading_target_n),
            ("gene_loading_abs_error_sum", gene_loading_abs_error_sum),
            ("gene_loading_squared_error_sum", gene_loading_squared_error_sum),
            ("gene_loading_pred_norm_sum", gene_loading_pred_norm_sum),
            ("gene_loading_target_norm_sum", gene_loading_target_norm_sum),
        ]
        for key, value in gene_loading_pairs:
            if value is None:
                continue
            reduced = float(self._reduce_scalar(value, average=False) or 0.0)
            bucket[f"{metric_prefix}{key}"] += reduced
            if view is not None:
                bucket[key] += reduced
        if ordinal_huber_loss is not None:
            reduced = float(self._reduce_scalar(ordinal_huber_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}ordinal_huber_sum"] += reduced
            bucket[f"{metric_prefix}ordinal_huber_steps"] += 1.0
            if view is not None:
                bucket["ordinal_huber_sum"] += reduced
                bucket["ordinal_huber_steps"] += 1.0
        if ordinal_abs_error_sum is not None and ordinal_squared_error_sum is not None and value_masked_n is not None:
            reduced_abs = float(self._reduce_scalar(ordinal_abs_error_sum, average=False) or 0.0)
            reduced_sq = float(self._reduce_scalar(ordinal_squared_error_sum, average=False) or 0.0)
            bucket[f"{metric_prefix}ordinal_abs_error_sum"] += reduced_abs
            bucket[f"{metric_prefix}ordinal_squared_error_sum"] += reduced_sq
            if view is not None:
                bucket["ordinal_abs_error_sum"] += reduced_abs
                bucket["ordinal_squared_error_sum"] += reduced_sq
        if visible_gene_loss is not None:
            reduced = float(self._reduce_scalar(visible_gene_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}visible_gene_loss_sum"] += reduced
            bucket[f"{metric_prefix}visible_gene_loss_steps"] += 1.0
            if view is not None:
                bucket["visible_gene_loss_sum"] += reduced
                bucket["visible_gene_loss_steps"] += 1.0
        visible_pairs = [
            ("visible_gene_target_n", visible_gene_target_n),
            ("visible_gene_pos_score_sum", visible_gene_pos_score_sum),
            ("visible_gene_neg_score_sum", visible_gene_neg_score_sum),
            ("visible_gene_pos_correct_n", visible_gene_pos_correct_n),
            ("visible_gene_neg_correct_n", visible_gene_neg_correct_n),
            ("visible_gene_neg_n", visible_gene_neg_n),
            ("visible_gene_same_row_neg_correct_n", visible_gene_same_row_neg_correct_n),
            ("visible_gene_same_row_neg_n", visible_gene_same_row_neg_n),
            ("visible_gene_support_neg_correct_n", visible_gene_support_neg_correct_n),
            ("visible_gene_support_neg_n", visible_gene_support_neg_n),
            ("visible_gene_random_neg_correct_n", visible_gene_random_neg_correct_n),
            ("visible_gene_random_neg_n", visible_gene_random_neg_n),
            ("visible_gene_auc_sum", visible_gene_auc_sum),
            ("visible_gene_positive_rank_sum", visible_gene_positive_rank_sum),
            ("visible_gene_top1_n", visible_gene_top1_n),
        ]
        for key, value in visible_pairs:
            if value is None:
                continue
            reduced = float(self._reduce_scalar(value, average=False) or 0.0)
            bucket[f"{metric_prefix}{key}"] += reduced
            if view is not None:
                bucket[key] += reduced
        if masked_gene_loss is not None:
            reduced = float(self._reduce_scalar(masked_gene_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}masked_gene_loss_sum"] += reduced
            bucket[f"{metric_prefix}masked_gene_loss_steps"] += 1.0
            if view is not None:
                bucket["masked_gene_loss_sum"] += reduced
                bucket["masked_gene_loss_steps"] += 1.0
        masked_pairs = [
            ("masked_gene_target_n", masked_gene_target_n),
            ("masked_gene_pos_score_sum", masked_gene_pos_score_sum),
            ("masked_gene_neg_score_sum", masked_gene_neg_score_sum),
            ("masked_gene_pos_correct_n", masked_gene_pos_correct_n),
            ("masked_gene_neg_correct_n", masked_gene_neg_correct_n),
            ("masked_gene_neg_n", masked_gene_neg_n),
            ("masked_gene_same_row_neg_correct_n", masked_gene_same_row_neg_correct_n),
            ("masked_gene_same_row_neg_n", masked_gene_same_row_neg_n),
            ("masked_gene_support_neg_correct_n", masked_gene_support_neg_correct_n),
            ("masked_gene_support_neg_n", masked_gene_support_neg_n),
            ("masked_gene_random_neg_correct_n", masked_gene_random_neg_correct_n),
            ("masked_gene_random_neg_n", masked_gene_random_neg_n),
            ("masked_gene_auc_sum", masked_gene_auc_sum),
            ("masked_gene_positive_rank_sum", masked_gene_positive_rank_sum),
            ("masked_gene_top1_n", masked_gene_top1_n),
        ]
        for key, value in masked_pairs:
            if value is None:
                continue
            reduced = float(self._reduce_scalar(value, average=False) or 0.0)
            bucket[f"{metric_prefix}{key}"] += reduced
            if view is not None:
                bucket[key] += reduced
        if rtd_loss is not None:
            reduced = float(self._reduce_scalar(rtd_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}rtd_loss_sum"] += reduced
            bucket[f"{metric_prefix}rtd_loss_steps"] += 1.0
            if view is not None:
                bucket["rtd_loss_sum"] += reduced
                bucket["rtd_loss_steps"] += 1.0
        rtd_pairs = [
            ("rtd_correct_n", rtd_correct_n),
            ("rtd_target_n", rtd_target_n),
            ("rtd_pos_n", rtd_pos_n),
            ("rtd_neg_n", rtd_neg_n),
            ("rtd_negative_random_correct_n", rtd_negative_random_correct_n),
            ("rtd_negative_random_n", rtd_negative_random_n),
            ("rtd_positive_unchanged_correct_n", rtd_positive_unchanged_correct_n),
            ("rtd_positive_unchanged_n", rtd_positive_unchanged_n),
            ("rtd_positive_visible_correct_n", rtd_positive_visible_correct_n),
            ("rtd_positive_visible_n", rtd_positive_visible_n),
            ("rtd_auc_sum", rtd_auc_sum),
            ("rtd_auc_pair_n", rtd_auc_pair_n),
        ]
        for key, value in rtd_pairs:
            if value is None:
                continue
            reduced = float(self._reduce_scalar(value, average=False) or 0.0)
            bucket[f"{metric_prefix}{key}"] += reduced
            if view is not None:
                bucket[key] += reduced
        if view_consistency_loss is not None:
            reduced = float(self._reduce_scalar(view_consistency_loss, average=True) or 0.0)
            bucket[f"{metric_prefix}view_consistency_loss_sum"] += reduced
            bucket[f"{metric_prefix}view_consistency_loss_steps"] += 1.0
            if view is not None:
                bucket["view_consistency_loss_sum"] += reduced
                bucket["view_consistency_loss_steps"] += 1.0
        if view_cosine_sum is not None and view_pair_n is not None:
            reduced_cosine = float(self._reduce_scalar(view_cosine_sum, average=False) or 0.0)
            reduced_pair_n = float(self._reduce_scalar(view_pair_n, average=False) or 0.0)
            bucket[f"{metric_prefix}view_cosine_sum"] += reduced_cosine
            bucket[f"{metric_prefix}view_pair_n"] += reduced_pair_n
            if view is not None:
                bucket["view_cosine_sum"] += reduced_cosine
                bucket["view_pair_n"] += reduced_pair_n
        self._accumulate_dynamic_scalar_metrics(bucket, varcov_metrics, view=view)

    def _accumulate_masked_accuracy(
        self,
        bucket: dict[str, float],
        logits: torch.Tensor | None,
        labels: torch.Tensor | None,
        *,
        view: str | None = None,
    ) -> None:
        if logits is None or labels is None:
            return
        mask = labels != -100
        if not bool(mask.any()):
            return
        preds = logits.argmax(dim=-1)
        top5 = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
        top10 = torch.topk(logits, k=min(10, logits.size(-1)), dim=-1).indices
        label_exp = labels.unsqueeze(-1)
        masked_correct_n = (preds[mask] == labels[mask]).sum().to(dtype=torch.float32)
        masked_top5_correct_n = (top5.eq(label_exp).any(dim=-1) & mask).sum().to(dtype=torch.float32)
        masked_top10_correct_n = (top10.eq(label_exp).any(dim=-1) & mask).sum().to(dtype=torch.float32)
        masked_n = mask.sum().to(dtype=torch.float32)
        metric_prefix = "" if view is None else f"{view}_"
        reduced_correct = float(self._reduce_scalar(masked_correct_n, average=False) or 0.0)
        reduced_top5 = float(self._reduce_scalar(masked_top5_correct_n, average=False) or 0.0)
        reduced_top10 = float(self._reduce_scalar(masked_top10_correct_n, average=False) or 0.0)
        reduced_n = float(self._reduce_scalar(masked_n, average=False) or 0.0)
        bucket[f"{metric_prefix}masked_correct_n"] += reduced_correct
        bucket[f"{metric_prefix}masked_top5_correct_n"] += reduced_top5
        bucket[f"{metric_prefix}masked_top10_correct_n"] += reduced_top10
        bucket[f"{metric_prefix}masked_n"] += reduced_n
        if view is not None:
            bucket["masked_correct_n"] += reduced_correct
            bucket["masked_top5_correct_n"] += reduced_top5
            bucket["masked_top10_correct_n"] += reduced_top10
            bucket["masked_n"] += reduced_n

    def _accumulate_masking_contract_metrics(
        self,
        bucket: dict[str, Any],
        inputs: dict[str, torch.Tensor],
        *,
        view: str | None = None,
    ) -> None:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        value_bins = inputs.get("value_bins")
        if input_ids is None or attention_mask is None:
            return
        metric_prefix = "" if view is None else f"{view}_"
        restored_ids = input_ids
        if labels is not None:
            restored_ids = torch.where(labels != -100, labels, input_ids)
        eligible = attention_mask > 0
        for token_id in getattr(self.data_collator, "special_token_ids", set()):
            eligible &= restored_ids.ne(int(token_id))
        gene_eligible_n = eligible.sum().to(dtype=torch.float32)
        reduced_gene_eligible = float(self._reduce_scalar(gene_eligible_n, average=False) or 0.0)
        bucket[f"{metric_prefix}gene_eligible_n"] += reduced_gene_eligible
        if view is not None:
            bucket["gene_eligible_n"] += reduced_gene_eligible

        if value_bins is None:
            return
        value_candidate = eligible.clone()
        if bool(getattr(self.data_collator, "value_target_decoupled", False)) and labels is not None:
            value_candidate &= labels.eq(-100)
        value_scope = str(getattr(self.data_collator, "value_target_scope", "all_eligible")).strip().lower()
        if value_scope == "observed_nonzero":
            value_candidate &= value_bins.gt(0)
        else:
            value_candidate &= value_bins.ne(int(getattr(self.data_collator, "value_pad_id", 0)))
        value_candidate_n = value_candidate.sum().to(dtype=torch.float32)
        reduced_value_candidate = float(self._reduce_scalar(value_candidate_n, average=False) or 0.0)
        bucket[f"{metric_prefix}value_candidate_n"] += reduced_value_candidate
        if view is not None:
            bucket["value_candidate_n"] += reduced_value_candidate

        program_mask = inputs.get("program_target_mask")
        if program_mask is not None:
            program_mask_bool = program_mask.to(dtype=torch.bool)
            program_candidate_n = torch.tensor(
                float(program_mask_bool.numel()),
                dtype=torch.float32,
                device=program_mask_bool.device,
            )
            reduced_program_candidate = float(self._reduce_scalar(program_candidate_n, average=False) or 0.0)
            bucket[f"{metric_prefix}program_target_candidate_n"] += reduced_program_candidate
            if view is not None:
                bucket["program_target_candidate_n"] += reduced_program_candidate

        gene_loading_mask = inputs.get("gene_loading_target_mask")
        if gene_loading_mask is not None:
            gene_loading_candidate_n = eligible.sum().to(dtype=torch.float32)
            reduced_gene_loading_candidate = float(
                self._reduce_scalar(gene_loading_candidate_n, average=False) or 0.0
            )
            bucket[f"{metric_prefix}gene_loading_target_candidate_n"] += reduced_gene_loading_candidate
            if view is not None:
                bucket["gene_loading_target_candidate_n"] += reduced_gene_loading_candidate

        if bool(getattr(self.data_collator, "visible_gene_discrimination_enable", False)):
            visible_candidate = eligible.clone()
            overlap_policy = str(
                getattr(self.data_collator, "visible_gene_disc_overlap_policy", "disjoint")
            ).strip().lower()
            if overlap_policy == "disjoint":
                labels = inputs.get("labels")
                if labels is not None and bool((labels != -100).any()):
                    visible_candidate &= labels.eq(-100)
                elif "value_labels" in inputs and inputs["value_labels"] is not None:
                    visible_candidate &= inputs["value_labels"].eq(-100)
            visible_gene_candidate_n = visible_candidate.sum().to(dtype=torch.float32)
            reduced_visible_candidate = float(self._reduce_scalar(visible_gene_candidate_n, average=False) or 0.0)
            bucket[f"{metric_prefix}visible_gene_candidate_n"] += reduced_visible_candidate
            if view is not None:
                bucket["visible_gene_candidate_n"] += reduced_visible_candidate
            visible_gene_corruption_overlap = inputs.get("visible_gene_corruption_overlap_n")
            if visible_gene_corruption_overlap is not None:
                reduced_overlap = float(self._reduce_scalar(visible_gene_corruption_overlap, average=False) or 0.0)
                bucket[f"{metric_prefix}visible_gene_corruption_overlap_n"] += reduced_overlap
                if view is not None:
                    bucket["visible_gene_corruption_overlap_n"] += reduced_overlap
        if bool(getattr(self.data_collator, "masked_gene_discrimination_enable", False)):
            masked_candidate = inputs.get("masked_gene_candidate_n")
            if masked_candidate is not None:
                reduced_masked_candidate = float(self._reduce_scalar(masked_candidate, average=False) or 0.0)
                bucket[f"{metric_prefix}masked_gene_candidate_n"] += reduced_masked_candidate
                if view is not None:
                    bucket["masked_gene_candidate_n"] += reduced_masked_candidate
            masked_visible_overlap = inputs.get("masked_gene_visible_overlap_n")
            if masked_visible_overlap is not None:
                reduced_overlap = float(self._reduce_scalar(masked_visible_overlap, average=False) or 0.0)
                bucket[f"{metric_prefix}masked_gene_visible_overlap_n"] += reduced_overlap
                if view is not None:
                    bucket["masked_gene_visible_overlap_n"] += reduced_overlap

    def _accumulate_value_target_breakdown(
        self,
        bucket: dict[str, Any],
        outputs: Any,
        inputs: dict[str, torch.Tensor],
    ) -> None:
        value_logits = getattr(outputs, "value_logits", None)
        value_labels = inputs.get("value_labels")
        if value_logits is None or value_labels is None:
            return
        mask = value_labels != -100
        if not bool(mask.any()):
            return

        preds = value_logits.argmax(dim=-1)
        target = value_labels[mask]
        pred = preds[mask]
        probs = torch.softmax(value_logits[mask], dim=-1)
        class_ids = torch.arange(value_logits.size(-1), device=probs.device, dtype=probs.dtype)
        expected_bin = (probs * class_ids.unsqueeze(0)).sum(dim=-1)
        target_float = target.to(dtype=expected_bin.dtype)

        observed_nonzero = target > 0
        obs_target = target[observed_nonzero]
        obs_pred = pred[observed_nonzero]
        obs_expected = expected_bin[observed_nonzero]
        obs_target_float = target_float[observed_nonzero]

        eligible_coords = None
        attn = inputs.get("attention_mask")
        if attn is not None:
            eligible_coords = attn[:, 1:].sum().to(dtype=torch.float32)

        target_hist = torch.bincount(target.to(dtype=torch.long), minlength=value_logits.size(-1)).to(dtype=torch.float32)
        pred_hist = torch.bincount(pred.to(dtype=torch.long), minlength=value_logits.size(-1)).to(dtype=torch.float32)
        correct_hist = torch.bincount(
            target[(pred == target)].to(dtype=torch.long),
            minlength=value_logits.size(-1),
        ).to(dtype=torch.float32)

        if eligible_coords is not None:
            bucket["eligible_coord_n"] += float(self._reduce_scalar(eligible_coords, average=False) or 0.0)

        obs_correct_n = (obs_pred == obs_target).sum().to(dtype=torch.float32)
        obs_n = observed_nonzero.sum().to(dtype=torch.float32)
        obs_abs_error = torch.abs(obs_expected - obs_target_float)
        within_1 = (torch.abs(obs_pred - obs_target) <= 1).sum().to(dtype=torch.float32)
        within_2 = (torch.abs(obs_pred - obs_target) <= 2).sum().to(dtype=torch.float32)
        bin0_target_n = (target == 0).sum().to(dtype=torch.float32)
        bin0_correct_n = ((target == 0) & (pred == 0)).sum().to(dtype=torch.float32)
        bin0_pred_n = (pred == 0).sum().to(dtype=torch.float32)

        bucket["observed_nonzero_correct_n"] += float(self._reduce_scalar(obs_correct_n, average=False) or 0.0)
        bucket["observed_nonzero_n"] += float(self._reduce_scalar(obs_n, average=False) or 0.0)
        bucket["observed_nonzero_abs_error_sum"] += float(
            self._reduce_scalar(obs_abs_error.sum().to(dtype=torch.float32), average=False) or 0.0
        )
        bucket["observed_nonzero_squared_error_sum"] += float(
            self._reduce_scalar(obs_abs_error.square().sum().to(dtype=torch.float32), average=False) or 0.0
        )
        bucket["within_1_bin_correct_n"] += float(self._reduce_scalar(within_1, average=False) or 0.0)
        bucket["within_2_bin_correct_n"] += float(self._reduce_scalar(within_2, average=False) or 0.0)
        bucket["bin0_target_n"] += float(self._reduce_scalar(bin0_target_n, average=False) or 0.0)
        bucket["bin0_correct_n"] += float(self._reduce_scalar(bin0_correct_n, average=False) or 0.0)
        bucket["bin0_pred_n"] += float(self._reduce_scalar(bin0_pred_n, average=False) or 0.0)

        reduced_target_hist = self._reduce_histogram(target_hist)
        reduced_pred_hist = self._reduce_histogram(pred_hist)
        reduced_correct_hist = self._reduce_histogram(correct_hist)
        if reduced_target_hist is not None:
            bucket["target_hist"] += reduced_target_hist
        if reduced_pred_hist is not None:
            bucket["pred_hist"] += reduced_pred_hist
        if reduced_correct_hist is not None:
            bucket["correct_hist"] += reduced_correct_hist

    def _effective_gene_pool_lambda(self) -> float:
        if not self._gene_pool_contrastive_enabled:
            return 0.0
        return float(resolve_scheduled_value(self._gene_pool_lambda_points, int(self.state.global_step)))

    def _mean_pool_token_embeddings(
        self,
        model,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        model_obj = unwrap_model(model)
        if not isinstance(model_obj, ValueAugmentedBertForMaskedLM):
            raise TypeError(f"unsupported model type for gene_pool contrastive: {type(model_obj)!r}")
        token_emb = model_obj.get_token_embedding_layer()(token_ids)
        keep = attention_mask > 0
        for token_id in getattr(self.data_collator, "special_token_ids", set()):
            keep &= token_ids.ne(int(token_id))
        keep_f = keep.unsqueeze(-1).to(dtype=token_emb.dtype)
        denom = keep_f.sum(dim=1).clamp(min=1.0)
        return (token_emb * keep_f).sum(dim=1) / denom

    def _compute_gene_pool_contrastive_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, float]:
        effective_lambda = self._effective_gene_pool_lambda()
        if not self._gene_pool_contrastive_enabled:
            return None, float(effective_lambda)
        required = [
            "sc_original_input_ids",
            "sc_original_attention_mask",
            "anchor_original_input_ids",
            "anchor_original_attention_mask",
            "sc_anchor_indices",
        ]
        missing = [key for key in required if key not in inputs]
        if missing:
            raise KeyError(f"gene_pool contrastive requires batch keys: {missing}")

        model_obj = unwrap_model(model)
        if not isinstance(model_obj, ValueAugmentedBertForMaskedLM):
            raise TypeError(f"unsupported model type for gene_pool contrastive: {type(model_obj)!r}")

        sc_pooled = self._mean_pool_token_embeddings(
            model,
            inputs["sc_original_input_ids"],
            inputs["sc_original_attention_mask"],
        )
        anchor_pooled = self._mean_pool_token_embeddings(
            model,
            inputs["anchor_original_input_ids"],
            inputs["anchor_original_attention_mask"],
        )
        query_z = model_obj.project_gene_pool(sc_pooled)
        anchor_z = model_obj.project_gene_pool(anchor_pooled)
        if self._gene_pool_stopgrad_anchor:
            anchor_z = anchor_z.detach()
        loss = query_to_anchor_contrastive_loss(
            query_z=query_z,
            anchor_z=anchor_z,
            positive_indices=inputs["sc_anchor_indices"],
            temperature=self._gene_pool_temperature,
        )
        return loss, float(effective_lambda)

    def _accumulate_gene_pool_metrics(
        self,
        bucket: dict[str, float],
        gene_pool_loss: torch.Tensor | None,
        effective_lambda: float,
    ) -> None:
        if gene_pool_loss is None:
            return
        reduced_loss = float(self._reduce_scalar(gene_pool_loss, average=True) or 0.0)
        bucket["gene_pool_contrastive_loss_sum"] += reduced_loss
        bucket["gene_pool_contrastive_steps"] += 1.0
        bucket["effective_gene_pool_lambda_sum"] += float(effective_lambda)
        bucket["effective_gene_pool_lambda_steps"] += 1.0

    def _add_metric_logs(
        self,
        out: dict[str, float],
        bucket: dict[str, Any],
        *,
        prefix: str,
        metric_prefix: str = "",
    ) -> None:
        total_loss_steps = float(bucket.get(f"{metric_prefix}total_loss_steps", 0.0))
        if total_loss_steps > 0:
            out[f"{prefix}total_loss"] = float(bucket[f"{metric_prefix}total_loss_sum"] / total_loss_steps)
        steps = float(bucket.get(f"{metric_prefix}steps", 0.0))
        if steps > 0:
            out[f"{prefix}mlm_loss"] = float(bucket[f"{metric_prefix}mlm_loss_sum"] / steps)
        value_loss_steps = float(bucket.get(f"{metric_prefix}value_loss_steps", 0.0))
        if value_loss_steps > 0:
            out[f"{prefix}value_loss"] = float(bucket[f"{metric_prefix}value_loss_sum"] / value_loss_steps)
            out[f"{prefix}value_ce"] = float(bucket[f"{metric_prefix}value_loss_sum"] / value_loss_steps)
            out[f"{prefix}masked_value_ce"] = float(bucket[f"{metric_prefix}value_loss_sum"] / value_loss_steps)
        value_masked_n = float(bucket.get(f"{metric_prefix}value_masked_n", 0.0))
        if value_masked_n > 0:
            out[f"{prefix}value_bin_acc"] = float(bucket.get(f"{metric_prefix}value_correct_n", 0.0) / value_masked_n)
            out[f"{prefix}masked_value_acc"] = float(bucket.get(f"{metric_prefix}value_correct_n", 0.0) / value_masked_n)
            out[f"{prefix}value_masked_n"] = value_masked_n
            out[f"{prefix}value_target_n"] = value_masked_n
            out[f"{prefix}ordinal_mae"] = float(bucket.get(f"{metric_prefix}ordinal_abs_error_sum", 0.0) / value_masked_n)
            out[f"{prefix}ordinal_rmse"] = float(
                math.sqrt(max(bucket.get(f"{metric_prefix}ordinal_squared_error_sum", 0.0) / value_masked_n, 0.0))
            )
            out[f"{prefix}value_expected_bin_mae"] = out[f"{prefix}ordinal_mae"]
            out[f"{prefix}value_expected_bin_rmse"] = out[f"{prefix}ordinal_rmse"]
        program_loss_steps = float(bucket.get(f"{metric_prefix}program_loss_steps", 0.0))
        if program_loss_steps > 0:
            out[f"{prefix}program_loss"] = float(
                bucket.get(f"{metric_prefix}program_loss_sum", 0.0) / program_loss_steps
            )
        program_target_n = float(bucket.get(f"{metric_prefix}program_target_n", 0.0))
        if program_target_n > 0:
            program_dim = int(getattr(getattr(self.model, "config", None), "program_dim", 1))
            program_value_n = max(program_target_n * float(program_dim), 1.0)
            out[f"{prefix}program_target_n"] = program_target_n
            out[f"{prefix}program_mae"] = float(
                bucket.get(f"{metric_prefix}program_abs_error_sum", 0.0) / program_value_n
            )
            out[f"{prefix}program_rmse"] = float(
                math.sqrt(max(bucket.get(f"{metric_prefix}program_squared_error_sum", 0.0) / program_value_n, 0.0))
            )
            out[f"{prefix}program_pred_norm_mean"] = float(
                bucket.get(f"{metric_prefix}program_pred_norm_sum", 0.0) / program_target_n
            )
            out[f"{prefix}program_target_norm_mean"] = float(
                bucket.get(f"{metric_prefix}program_target_norm_sum", 0.0) / program_target_n
            )
        program_candidate_n = float(bucket.get(f"{metric_prefix}program_target_candidate_n", 0.0))
        if program_candidate_n > 0:
            out[f"{prefix}program_target_candidate_n"] = program_candidate_n
            out[f"{prefix}program_coverage"] = float(program_target_n / program_candidate_n)
        gene_loading_loss_steps = float(bucket.get(f"{metric_prefix}gene_loading_loss_steps", 0.0))
        if gene_loading_loss_steps > 0:
            out[f"{prefix}gene_loading_loss"] = float(
                bucket.get(f"{metric_prefix}gene_loading_loss_sum", 0.0) / gene_loading_loss_steps
            )
        gene_loading_target_n = float(bucket.get(f"{metric_prefix}gene_loading_target_n", 0.0))
        if gene_loading_target_n > 0:
            gene_loading_dim = int(getattr(getattr(self.model, "config", None), "gene_loading_dim", 1))
            gene_loading_value_n = max(gene_loading_target_n * float(gene_loading_dim), 1.0)
            out[f"{prefix}gene_loading_target_n"] = gene_loading_target_n
            out[f"{prefix}gene_loading_mae"] = float(
                bucket.get(f"{metric_prefix}gene_loading_abs_error_sum", 0.0) / gene_loading_value_n
            )
            out[f"{prefix}gene_loading_rmse"] = float(
                math.sqrt(
                    max(bucket.get(f"{metric_prefix}gene_loading_squared_error_sum", 0.0) / gene_loading_value_n, 0.0)
                )
            )
            out[f"{prefix}gene_loading_pred_norm_mean"] = float(
                bucket.get(f"{metric_prefix}gene_loading_pred_norm_sum", 0.0) / gene_loading_target_n
            )
            out[f"{prefix}gene_loading_target_norm_mean"] = float(
                bucket.get(f"{metric_prefix}gene_loading_target_norm_sum", 0.0) / gene_loading_target_n
            )
        gene_loading_candidate_n = float(bucket.get(f"{metric_prefix}gene_loading_target_candidate_n", 0.0))
        if gene_loading_candidate_n > 0:
            out[f"{prefix}gene_loading_target_candidate_n"] = gene_loading_candidate_n
            out[f"{prefix}gene_loading_coverage"] = float(gene_loading_target_n / gene_loading_candidate_n)
        masked_n = float(bucket.get(f"{metric_prefix}masked_n", 0.0))
        if masked_n > 0:
            out[f"{prefix}masked_acc"] = float(bucket.get(f"{metric_prefix}masked_correct_n", 0.0) / masked_n)
            out[f"{prefix}masked_top5_acc"] = float(bucket.get(f"{metric_prefix}masked_top5_correct_n", 0.0) / masked_n)
            out[f"{prefix}masked_top10_acc"] = float(bucket.get(f"{metric_prefix}masked_top10_correct_n", 0.0) / masked_n)
            out[f"{prefix}masked_n"] = masked_n
            out[f"{prefix}gene_masked_n"] = masked_n
            out[f"{prefix}mask_only_top1_acc"] = float(bucket.get(f"{metric_prefix}masked_correct_n", 0.0) / masked_n)
            out[f"{prefix}mask_only_top5_acc"] = float(bucket.get(f"{metric_prefix}masked_top5_correct_n", 0.0) / masked_n)
            out[f"{prefix}mask_only_top10_acc"] = float(bucket.get(f"{metric_prefix}masked_top10_correct_n", 0.0) / masked_n)
            out[f"{prefix}gene_mask_only_top1_acc"] = float(bucket.get(f"{metric_prefix}masked_correct_n", 0.0) / masked_n)
            out[f"{prefix}gene_mask_only_top5_acc"] = float(bucket.get(f"{metric_prefix}masked_top5_correct_n", 0.0) / masked_n)
            out[f"{prefix}gene_mask_only_top10_acc"] = float(bucket.get(f"{metric_prefix}masked_top10_correct_n", 0.0) / masked_n)
        ordinal_huber_steps = float(bucket.get(f"{metric_prefix}ordinal_huber_steps", 0.0))
        if ordinal_huber_steps > 0:
            out[f"{prefix}ordinal_huber"] = float(bucket[f"{metric_prefix}ordinal_huber_sum"] / ordinal_huber_steps)
            out[f"{prefix}value_ordinal_huber"] = out[f"{prefix}ordinal_huber"]
        rtd_loss_steps = float(bucket.get(f"{metric_prefix}rtd_loss_steps", 0.0))
        if rtd_loss_steps > 0:
            out[f"{prefix}rtd_loss"] = float(bucket[f"{metric_prefix}rtd_loss_sum"] / rtd_loss_steps)
        rtd_target_n = float(bucket.get(f"{metric_prefix}rtd_target_n", 0.0))
        if rtd_target_n > 0:
            rtd_pos_n = float(bucket.get(f"{metric_prefix}rtd_pos_n", 0.0))
            rtd_neg_n = float(bucket.get(f"{metric_prefix}rtd_neg_n", 0.0))
            out[f"{prefix}rtd_acc"] = float(bucket.get(f"{metric_prefix}rtd_correct_n", 0.0) / rtd_target_n)
            out[f"{prefix}rtd_target_n"] = rtd_target_n
            out[f"{prefix}rtd_pos_frac"] = float(rtd_pos_n / rtd_target_n)
            out[f"{prefix}rtd_neg_frac"] = float(rtd_neg_n / rtd_target_n)
            negative_random_n = float(bucket.get(f"{metric_prefix}rtd_negative_random_n", 0.0))
            if negative_random_n > 0:
                out[f"{prefix}rtd_acc_negative_random"] = float(
                    bucket.get(f"{metric_prefix}rtd_negative_random_correct_n", 0.0) / negative_random_n
                )
            positive_unchanged_n = float(bucket.get(f"{metric_prefix}rtd_positive_unchanged_n", 0.0))
            if positive_unchanged_n > 0:
                out[f"{prefix}rtd_acc_positive_unchanged"] = float(
                    bucket.get(f"{metric_prefix}rtd_positive_unchanged_correct_n", 0.0) / positive_unchanged_n
                )
            positive_visible_n = float(bucket.get(f"{metric_prefix}rtd_positive_visible_n", 0.0))
            if positive_visible_n > 0:
                out[f"{prefix}rtd_acc_positive_visible"] = float(
                    bucket.get(f"{metric_prefix}rtd_positive_visible_correct_n", 0.0) / positive_visible_n
                )
            auc_pair_n = float(bucket.get(f"{metric_prefix}rtd_auc_pair_n", 0.0))
            if auc_pair_n > 0:
                out[f"{prefix}rtd_auc"] = float(bucket.get(f"{metric_prefix}rtd_auc_sum", 0.0) / auc_pair_n)
        observed_nonzero_n = float(bucket.get("observed_nonzero_n", 0.0))
        if metric_prefix == "" and observed_nonzero_n > 0:
            out[f"{prefix}observed_nonzero_acc"] = float(bucket.get("observed_nonzero_correct_n", 0.0) / observed_nonzero_n)
            out[f"{prefix}observed_nonzero_value_acc"] = float(
                bucket.get("observed_nonzero_correct_n", 0.0) / observed_nonzero_n
            )
            out[f"{prefix}observed_nonzero_target_n"] = observed_nonzero_n
            out[f"{prefix}observed_nonzero_ordinal_mae"] = float(
                bucket.get("observed_nonzero_abs_error_sum", 0.0) / observed_nonzero_n
            )
            out[f"{prefix}observed_nonzero_ordinal_rmse"] = float(
                math.sqrt(max(bucket.get("observed_nonzero_squared_error_sum", 0.0) / observed_nonzero_n, 0.0))
            )
            out[f"{prefix}within_1_bin_acc"] = float(bucket.get("within_1_bin_correct_n", 0.0) / observed_nonzero_n)
            out[f"{prefix}within_2_bin_acc"] = float(bucket.get("within_2_bin_correct_n", 0.0) / observed_nonzero_n)
            out[f"{prefix}value_expected_bin_within1"] = out[f"{prefix}within_1_bin_acc"]
            out[f"{prefix}value_expected_bin_within2"] = out[f"{prefix}within_2_bin_acc"]
            out[f"{prefix}within1"] = out[f"{prefix}within_1_bin_acc"]
            out[f"{prefix}within2"] = out[f"{prefix}within_2_bin_acc"]
        bin0_target_n = float(bucket.get("bin0_target_n", 0.0))
        if metric_prefix == "" and bin0_target_n > 0:
            out[f"{prefix}bin_0_acc"] = float(bucket.get("bin0_correct_n", 0.0) / bin0_target_n)
        if metric_prefix == "" and value_masked_n > 0:
            out[f"{prefix}bin_0_prediction_rate"] = float(bucket.get("bin0_pred_n", 0.0) / value_masked_n)
            out[f"{prefix}realized_nonzero_target_fraction"] = float(observed_nonzero_n / value_masked_n)
        eligible_coord_n = float(bucket.get("eligible_coord_n", 0.0))
        if metric_prefix == "" and eligible_coord_n > 0:
            out[f"{prefix}realized_value_mask_ratio"] = float(value_masked_n / eligible_coord_n)
        gene_eligible_n = float(bucket.get(f"{metric_prefix}gene_eligible_n", 0.0))
        if gene_eligible_n > 0:
            out[f"{prefix}realized_mlm_probability"] = float(masked_n / gene_eligible_n)
            out[f"{prefix}gene_eligible_n"] = gene_eligible_n
        value_candidate_n = float(bucket.get(f"{metric_prefix}value_candidate_n", 0.0))
        if value_candidate_n > 0:
            out[f"{prefix}realized_value_target_probability"] = float(value_masked_n / value_candidate_n)
            out[f"{prefix}value_candidate_n"] = value_candidate_n
        visible_gene_loss_steps = float(bucket.get(f"{metric_prefix}visible_gene_loss_steps", 0.0))
        if visible_gene_loss_steps > 0:
            out[f"{prefix}visible_gene_loss"] = float(
                bucket.get(f"{metric_prefix}visible_gene_loss_sum", 0.0) / visible_gene_loss_steps
            )
        visible_gene_target_n = float(bucket.get(f"{metric_prefix}visible_gene_target_n", 0.0))
        if visible_gene_target_n > 0:
            out[f"{prefix}visible_gene_target_n"] = visible_gene_target_n
            out[f"{prefix}visible_gene_pos_score_mean"] = float(
                bucket.get(f"{metric_prefix}visible_gene_pos_score_sum", 0.0) / visible_gene_target_n
            )
            neg_n = float(bucket.get(f"{metric_prefix}visible_gene_neg_n", 0.0))
            if neg_n > 0:
                out[f"{prefix}visible_gene_neg_score_mean"] = float(
                    bucket.get(f"{metric_prefix}visible_gene_neg_score_sum", 0.0) / neg_n
                )
                out[f"{prefix}visible_gene_binary_acc"] = float(
                    (
                        bucket.get(f"{metric_prefix}visible_gene_pos_correct_n", 0.0)
                        + bucket.get(f"{metric_prefix}visible_gene_neg_correct_n", 0.0)
                    )
                    / (visible_gene_target_n + neg_n)
                )
            out[f"{prefix}visible_gene_pos_acc"] = float(
                bucket.get(f"{metric_prefix}visible_gene_pos_correct_n", 0.0) / visible_gene_target_n
            )
            same_row_neg_n = float(bucket.get(f"{metric_prefix}visible_gene_same_row_neg_n", 0.0))
            if same_row_neg_n > 0:
                same_row_acc = float(
                    bucket.get(f"{metric_prefix}visible_gene_same_row_neg_correct_n", 0.0) / same_row_neg_n
                )
                out[f"{prefix}visible_gene_neg_acc_same_row"] = same_row_acc
                out[f"{prefix}visible_gene_same_row_binary_acc"] = same_row_acc
            support_neg_n = float(bucket.get(f"{metric_prefix}visible_gene_support_neg_n", 0.0))
            if support_neg_n > 0:
                support_acc = float(
                    bucket.get(f"{metric_prefix}visible_gene_support_neg_correct_n", 0.0) / support_neg_n
                )
                out[f"{prefix}visible_gene_neg_acc_support"] = support_acc
                out[f"{prefix}visible_gene_support_matched_binary_acc"] = support_acc
            random_neg_n = float(bucket.get(f"{metric_prefix}visible_gene_random_neg_n", 0.0))
            if random_neg_n > 0:
                random_acc = float(
                    bucket.get(f"{metric_prefix}visible_gene_random_neg_correct_n", 0.0) / random_neg_n
                )
                out[f"{prefix}visible_gene_neg_acc_random"] = random_acc
                out[f"{prefix}visible_gene_random_global_binary_acc"] = random_acc
            out[f"{prefix}visible_gene_auc"] = float(
                bucket.get(f"{metric_prefix}visible_gene_auc_sum", 0.0) / visible_gene_target_n
            )
            out[f"{prefix}visible_gene_sampled_top1_acc"] = float(
                bucket.get(f"{metric_prefix}visible_gene_top1_n", 0.0) / visible_gene_target_n
            )
            visible_rank_mean = float(
                bucket.get(f"{metric_prefix}visible_gene_positive_rank_sum", 0.0) / visible_gene_target_n
            )
            out[f"{prefix}visible_gene_positive_rank_mean"] = visible_rank_mean
            out[f"{prefix}visible_gene_positive_rank_among_candidates"] = visible_rank_mean
        visible_gene_candidate_n = float(bucket.get(f"{metric_prefix}visible_gene_candidate_n", 0.0))
        if visible_gene_candidate_n > 0:
            out[f"{prefix}visible_gene_candidate_n"] = visible_gene_candidate_n
            out[f"{prefix}realized_visible_gene_disc_probability"] = float(
                visible_gene_target_n / visible_gene_candidate_n
            )
        visible_gene_corruption_overlap_n = float(bucket.get(f"{metric_prefix}visible_gene_corruption_overlap_n", 0.0))
        if visible_gene_target_n > 0 or visible_gene_candidate_n > 0 or visible_gene_corruption_overlap_n > 0:
            out[f"{prefix}visible_gene_corruption_overlap_n"] = visible_gene_corruption_overlap_n
            overlap_denom = visible_gene_target_n if visible_gene_target_n > 0 else max(visible_gene_candidate_n, 1.0)
            out[f"{prefix}visible_gene_corruption_overlap_rate"] = float(
                visible_gene_corruption_overlap_n / overlap_denom
            )
        masked_gene_loss_steps = float(bucket.get(f"{metric_prefix}masked_gene_loss_steps", 0.0))
        if masked_gene_loss_steps > 0:
            out[f"{prefix}masked_gene_loss"] = float(
                bucket.get(f"{metric_prefix}masked_gene_loss_sum", 0.0) / masked_gene_loss_steps
            )
        masked_gene_target_n = float(bucket.get(f"{metric_prefix}masked_gene_target_n", 0.0))
        if masked_gene_target_n > 0:
            out[f"{prefix}masked_gene_target_n"] = masked_gene_target_n
            out[f"{prefix}masked_gene_pos_score_mean"] = float(
                bucket.get(f"{metric_prefix}masked_gene_pos_score_sum", 0.0) / masked_gene_target_n
            )
            masked_neg_n = float(bucket.get(f"{metric_prefix}masked_gene_neg_n", 0.0))
            if masked_neg_n > 0:
                out[f"{prefix}masked_gene_neg_score_mean"] = float(
                    bucket.get(f"{metric_prefix}masked_gene_neg_score_sum", 0.0) / masked_neg_n
                )
                out[f"{prefix}masked_gene_binary_acc"] = float(
                    (
                        bucket.get(f"{metric_prefix}masked_gene_pos_correct_n", 0.0)
                        + bucket.get(f"{metric_prefix}masked_gene_neg_correct_n", 0.0)
                    )
                    / (masked_gene_target_n + masked_neg_n)
                )
            same_row_neg_n = float(bucket.get(f"{metric_prefix}masked_gene_same_row_neg_n", 0.0))
            if same_row_neg_n > 0:
                same_row_acc = float(
                    bucket.get(f"{metric_prefix}masked_gene_same_row_neg_correct_n", 0.0) / same_row_neg_n
                )
                out[f"{prefix}masked_gene_same_row_binary_acc"] = same_row_acc
            support_neg_n = float(bucket.get(f"{metric_prefix}masked_gene_support_neg_n", 0.0))
            if support_neg_n > 0:
                support_acc = float(
                    bucket.get(f"{metric_prefix}masked_gene_support_neg_correct_n", 0.0) / support_neg_n
                )
                out[f"{prefix}masked_gene_support_matched_binary_acc"] = support_acc
            random_neg_n = float(bucket.get(f"{metric_prefix}masked_gene_random_neg_n", 0.0))
            if random_neg_n > 0:
                random_acc = float(
                    bucket.get(f"{metric_prefix}masked_gene_random_neg_correct_n", 0.0) / random_neg_n
                )
                out[f"{prefix}masked_gene_random_global_binary_acc"] = random_acc
            out[f"{prefix}masked_gene_positive_rank_among_candidates"] = float(
                bucket.get(f"{metric_prefix}masked_gene_positive_rank_sum", 0.0) / masked_gene_target_n
            )
        masked_gene_candidate_n = float(bucket.get(f"{metric_prefix}masked_gene_candidate_n", 0.0))
        if masked_gene_candidate_n > 0:
            out[f"{prefix}masked_gene_candidate_n"] = masked_gene_candidate_n
            out[f"{prefix}realized_masked_gene_probability"] = float(
                masked_gene_target_n / masked_gene_candidate_n
            )
        masked_gene_visible_overlap_n = float(bucket.get(f"{metric_prefix}masked_gene_visible_overlap_n", 0.0))
        if masked_gene_target_n > 0:
            out[f"{prefix}masked_gene_visible_overlap_n"] = masked_gene_visible_overlap_n
            out[f"{prefix}masked_gene_visible_overlap_rate"] = float(
                masked_gene_visible_overlap_n / masked_gene_target_n
            )
        view_consistency_steps = float(bucket.get(f"{metric_prefix}view_consistency_loss_steps", 0.0))
        if view_consistency_steps > 0:
            out[f"{prefix}view_consistency_loss"] = float(
                bucket.get(f"{metric_prefix}view_consistency_loss_sum", 0.0) / view_consistency_steps
            )
        view_pair_n = float(bucket.get(f"{metric_prefix}view_pair_n", 0.0))
        if view_pair_n > 0:
            out[f"{prefix}view_cosine_mean"] = float(
                bucket.get(f"{metric_prefix}view_cosine_sum", 0.0) / view_pair_n
            )
            out[f"{prefix}view_pair_n"] = view_pair_n
        if metric_prefix == "":
            target_hist = np.asarray(bucket.get("target_hist", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
            pred_hist = np.asarray(bucket.get("pred_hist", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
            correct_hist = np.asarray(bucket.get("correct_hist", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
            if value_masked_n > 0 and target_hist.size > 0:
                per_bin_acc = np.divide(
                    correct_hist,
                    np.maximum(target_hist, 1.0),
                    out=np.zeros_like(correct_hist),
                    where=target_hist > 0,
                )
                present = target_hist > 0
                if bool(np.any(present)):
                    macro_bin_acc = float(per_bin_acc[present].mean())
                    out[f"{prefix}macro_bin_acc"] = macro_bin_acc
                    out[f"{prefix}balanced_acc"] = macro_bin_acc
                for bin_idx in range(int(target_hist.size)):
                    out[f"{prefix}target_bin_frac_{bin_idx}"] = float(target_hist[bin_idx] / value_masked_n)
                    out[f"{prefix}pred_bin_frac_{bin_idx}"] = float(pred_hist[bin_idx] / value_masked_n)
        gene_pool_steps = float(bucket.get("gene_pool_contrastive_steps", 0.0))
        if metric_prefix == "" and gene_pool_steps > 0:
            out[f"{prefix}gene_pool_contrastive_loss"] = float(bucket["gene_pool_contrastive_loss_sum"] / gene_pool_steps)
        gene_pool_lambda_steps = float(bucket.get("effective_gene_pool_lambda_steps", 0.0))
        if metric_prefix == "" and gene_pool_lambda_steps > 0:
            out[f"{prefix}effective_gene_pool_contrastive_lambda"] = float(
                bucket["effective_gene_pool_lambda_sum"] / gene_pool_lambda_steps
            )
        scalar_sums = dict(bucket.get("dynamic_scalar_sums", {}))
        scalar_steps = dict(bucket.get("dynamic_scalar_steps", {}))
        for key in sorted(scalar_sums):
            if metric_prefix:
                if not key.startswith(metric_prefix):
                    continue
                metric_name = key[len(metric_prefix) :]
            else:
                if key.startswith("sc_") or key.startswith("mc_"):
                    continue
                metric_name = key
            step_n = float(scalar_steps.get(key, 0.0))
            if step_n <= 0:
                continue
            out[f"{prefix}{metric_name}"] = float(scalar_sums[key] / step_n)

    def _format_metric_logs(self, bucket: dict[str, Any], *, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        self._add_metric_logs(out, bucket, prefix=prefix)
        if float(bucket.get("sc_steps", 0.0)) > 0:
            self._add_metric_logs(out, bucket, prefix=f"{prefix}sc_", metric_prefix="sc_")
        if float(bucket.get("mc_steps", 0.0)) > 0:
            self._add_metric_logs(out, bucket, prefix=f"{prefix}mc_", metric_prefix="mc_")
        return out

    def _boundary_root(self, name: str) -> Path:
        return Path(self.args.output_dir) / "pass_boundaries" / name

    def _discover_saved_boundary_steps(self) -> set[int]:
        saved: set[int] = set()
        for step, name in self._boundary_saves.items():
            manifest = self._boundary_root(name) / "boundary_manifest.json"
            if manifest.exists():
                saved.add(int(step))
        return saved

    def _discover_saved_time_boundary_names(self) -> set[str]:
        saved: set[str] = set()
        for _, name in self._time_boundary_saves.items():
            manifest = self._boundary_root(name) / "boundary_manifest.json"
            if manifest.exists():
                saved.add(str(name))
        return saved

    def _save_boundary_exports(
        self,
        *,
        model,
        step: int,
        name: str,
        trial=None,
        boundary_kind: str = "step",
        target_wallclock_seconds: float | None = None,
        mark_step_saved: bool = True,
    ) -> None:
        if bool(getattr(self.args, "skip_boundary_exports", False)):
            if mark_step_saved:
                self._saved_boundary_steps.add(int(step))
            return
        save_started_wall = time.time()
        elapsed_before_export = save_started_wall - self._train_start_wall
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{step}"
        if not checkpoint_dir.exists():
            self._save_checkpoint(model, trial, metrics=None)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.is_world_process_zero():
            boundary_root = self._boundary_root(name)
            plain_model_dir = boundary_root / "final_model"
            full_model_dir = boundary_root / "last_model_aux"

            unwrapped = unwrap_model(self.model)
            self.save_model(str(full_model_dir))
            if isinstance(unwrapped, ValueAugmentedBertForMaskedLM):
                incompatible = unwrapped.export_plain_mlm(plain_model_dir)
                export_meta = {
                    "exported_plain_mlm": bool(incompatible.get("exported_plain_mlm", True)),
                    "plain_model_dir": str(plain_model_dir),
                    "full_model_dir": str(full_model_dir),
                    **incompatible,
                }
            else:
                self.save_model(str(plain_model_dir))
                export_meta = {
                    "exported_plain_mlm": False,
                    "plain_model_dir": str(plain_model_dir),
                    "full_model_dir": str(full_model_dir),
                }

            ensure_symlink(boundary_root / "checkpoint", checkpoint_dir)
            write_json(boundary_root / "plain_export_meta.json", export_meta)
            save_finished_wall = time.time()
            elapsed_after_export = save_finished_wall - self._train_start_wall
            wallclock_overshoot = (
                elapsed_after_export - float(target_wallclock_seconds)
                if target_wallclock_seconds is not None
                else None
            )
            write_json(
                boundary_root / "boundary_manifest.json",
                {
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "boundary_name": str(name),
                    "boundary_kind": str(boundary_kind),
                    "global_step": int(step),
                    "epoch": float(self.state.epoch) if self.state.epoch is not None else None,
                    "checkpoint_dir": str(checkpoint_dir),
                    "full_model_dir": str(full_model_dir),
                    "plain_model_dir": str(plain_model_dir),
                    "target_wallclock_seconds": (
                        float(target_wallclock_seconds) if target_wallclock_seconds is not None else None
                    ),
                    "elapsed_wallclock_seconds_before_export": float(elapsed_before_export),
                    "elapsed_wallclock_seconds": float(elapsed_after_export),
                    "wallclock_overshoot_seconds": (
                        float(wallclock_overshoot) if wallclock_overshoot is not None else None
                    ),
                    "boundary_export_duration_seconds": float(save_finished_wall - save_started_wall),
                    "train_seen_examples": float(self._seen_examples),
                    "train_seen_raw_cells": float(self._seen_raw_cells),
                    "train_seen_non_pad_tokens": float(self._seen_non_pad_tokens),
                    "train_seen_sc_non_pad_tokens": float(self._seen_sc_non_pad_tokens),
                    "train_seen_mc_non_pad_tokens": float(self._seen_mc_non_pad_tokens),
                },
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if mark_step_saved:
            self._saved_boundary_steps.add(int(step))

    def _maybe_save_exact_boundaries(self, *, model, trial=None) -> None:
        step = int(self.state.global_step)
        name = self._boundary_saves.get(step)
        if name is None or step in self._saved_boundary_steps:
            return
        self._save_boundary_exports(model=model, step=step, name=name, trial=trial)

    def _maybe_save_time_boundaries(self, *, model, trial=None) -> bool:
        if not self._time_boundary_saves:
            return False
        step = int(self.state.global_step)
        elapsed = time.time() - self._train_start_wall
        saved_any = False
        for target_seconds, name in self._time_boundary_saves.items():
            if str(name) in self._saved_time_boundary_names:
                continue
            if elapsed < float(target_seconds):
                continue
            self._save_boundary_exports(
                model=model,
                step=step,
                name=str(name),
                trial=trial,
                boundary_kind="wallclock",
                target_wallclock_seconds=float(target_seconds),
                mark_step_saved=False,
            )
            self._saved_time_boundary_names.add(str(name))
            saved_any = True
        if self._stop_after_wall_seconds is None:
            return False
        elapsed = time.time() - self._train_start_wall
        if elapsed < float(self._stop_after_wall_seconds):
            return False
        required = {
            str(name)
            for target_seconds, name in self._time_boundary_saves.items()
            if float(target_seconds) <= float(self._stop_after_wall_seconds)
        }
        return required.issubset(self._saved_time_boundary_names) or not required

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is not None and self._reset_scheduler_for_epoch:
            if self._epoch_scheduler_steps is None or self._epoch_start_learning_rate is None:
                raise ValueError(
                    "reset_scheduler_for_epoch requires epoch_scheduler_steps and epoch_start_learning_rate"
                )
            base_lr = float(self._epoch_start_learning_rate)
            for group in self.optimizer.param_groups:
                group["lr"] = base_lr
                group["initial_lr"] = base_lr
            total_steps = int(self._epoch_scheduler_steps)
            warmup_steps = (
                int(self._epoch_warmup_steps)
                if self._epoch_warmup_steps is not None
                else int(self.args.get_warmup_steps(total_steps))
            )
            if self._runtime_lr_scheduler_type == "cosine_with_floor":
                floor = float(self._cosine_min_lr_ratio)

                def lr_lambda(current_step: int) -> float:
                    if warmup_steps > 0 and current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    progress = min(1.0, max(0.0, progress))
                    return floor + ((1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            else:

                def lr_lambda(current_step: int) -> float:
                    if warmup_steps > 0 and current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return max(0.0, 1.0 - min(1.0, max(0.0, progress)))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            if self.is_world_process_zero():
                print(
                    "[continuation_scheduler_reset] "
                    f"steps={total_steps} warmup={warmup_steps} start_lr={base_lr:.10g}",
                    flush=True,
                )
        if checkpoint is not None or self._fresh_init_states_loaded:
            return
        if self._init_optimizer_state_path is None and self._init_scheduler_state_path is None:
            return
        if self._init_optimizer_state_path is None or self._init_scheduler_state_path is None:
            raise ValueError("fresh init requires both init_optimizer_state_path and init_scheduler_state_path")
        if not self._init_optimizer_state_path.exists():
            raise FileNotFoundError(f"missing optimizer state: {self._init_optimizer_state_path}")
        if not self._init_scheduler_state_path.exists():
            raise FileNotFoundError(f"missing scheduler state: {self._init_scheduler_state_path}")

        map_location = self.args.device if self.args.world_size > 1 else "cpu"
        optimizer_state = torch.load(self._init_optimizer_state_path, map_location=map_location, weights_only=False)
        scheduler_state = torch.load(self._init_scheduler_state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(optimizer_state)
        self.lr_scheduler.load_state_dict(scheduler_state)
        self._fresh_init_states_loaded = True

    def _forward_paired_outputs(self, model, inputs: dict[str, torch.Tensor]) -> tuple[Any, Any]:
        sc_outputs = model(**self._extract_view_inputs(inputs, "sc"))
        mc_outputs = model(**self._extract_view_inputs(inputs, "mc"))
        return sc_outputs, mc_outputs

    def _paired_total_loss(
        self,
        sc_outputs: Any,
        mc_outputs: Any,
        *,
        gene_pool_loss: torch.Tensor | None = None,
        effective_gene_pool_lambda: float = 0.0,
    ) -> torch.Tensor:
        sc_loss = self._extract_loss_tensor(sc_outputs)
        mc_loss = self._extract_loss_tensor(mc_outputs)
        if sc_loss is None and mc_loss is None:
            raise ValueError("paired mode requires at least one non-null loss")
        if sc_loss is None:
            total_loss = mc_loss * self._paired_mc_loss_weight
        elif mc_loss is None:
            total_loss = sc_loss
        else:
            total_loss = sc_loss + (mc_loss * self._paired_mc_loss_weight)
        if gene_pool_loss is not None:
            total_loss = total_loss + (gene_pool_loss * float(effective_gene_pool_lambda))
        return total_loss

    def _accumulate_paired_metrics(self, bucket: dict[str, float], sc_outputs: Any, mc_outputs: Any, inputs: dict[str, torch.Tensor]) -> None:
        self._accumulate_metric_sums_view(bucket, sc_outputs, view="sc")
        self._accumulate_metric_sums_view(bucket, mc_outputs, view="mc")
        self._accumulate_masked_accuracy(bucket, getattr(sc_outputs, "logits", None), inputs.get("sc_labels"), view="sc")
        self._accumulate_masked_accuracy(bucket, getattr(mc_outputs, "logits", None), inputs.get("mc_labels"), view="mc")
        self._accumulate_masking_contract_metrics(
            bucket,
            self._extract_view_metric_inputs(inputs, "sc"),
            view="sc",
        )
        self._accumulate_masking_contract_metrics(
            bucket,
            self._extract_view_metric_inputs(inputs, "mc"),
            view="mc",
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self._paired_batch_enabled(inputs):
            if model.training:
                self._check_required_program_targets(inputs)
            model_inputs = self._filter_model_inputs(inputs)
            outputs = model(**model_inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            dual_outputs = None
            if self._dual_batch_enabled(inputs):
                dual_outputs = model(**self._extract_view_inputs(inputs, "dual"))
                dual_loss = dual_outputs["loss"] if isinstance(dual_outputs, dict) else dual_outputs.loss
                loss = 0.5 * (loss + dual_loss)
            if model.training:
                self._accumulate_metric_sums_view(self._train_metric_sums, outputs, view=None)
                self._accumulate_masked_accuracy(self._train_metric_sums, getattr(outputs, "logits", None), inputs.get("labels"), view=None)
                self._accumulate_masking_contract_metrics(self._train_metric_sums, inputs, view=None)
                self._accumulate_value_target_breakdown(self._train_metric_sums, outputs, inputs)
                if dual_outputs is not None:
                    dual_metric_inputs = self._extract_view_metric_inputs(inputs, "dual")
                    self._accumulate_metric_sums_view(self._train_metric_sums, dual_outputs, view=None)
                    self._accumulate_masked_accuracy(
                        self._train_metric_sums,
                        getattr(dual_outputs, "logits", None),
                        inputs.get("dual_labels"),
                        view=None,
                    )
                    self._accumulate_masking_contract_metrics(self._train_metric_sums, dual_metric_inputs, view=None)
                    self._accumulate_value_target_breakdown(self._train_metric_sums, dual_outputs, dual_metric_inputs)
            returned_outputs = {"main": outputs, "dual": dual_outputs} if dual_outputs is not None else outputs
            return (loss, returned_outputs) if return_outputs else loss

        if model.training:
            self._check_required_program_targets(inputs, prefix="sc")
            self._check_required_program_targets(inputs, prefix="mc")
        sc_outputs, mc_outputs = self._forward_paired_outputs(model, inputs)
        gene_pool_loss = getattr(sc_outputs, "gene_pool_contrastive_loss", None)
        effective_gene_pool_lambda = self._effective_gene_pool_lambda()
        loss = self._paired_total_loss(
            sc_outputs,
            mc_outputs,
            gene_pool_loss=gene_pool_loss,
            effective_gene_pool_lambda=effective_gene_pool_lambda,
        )
        if model.training:
            self._accumulate_paired_metrics(self._train_metric_sums, sc_outputs, mc_outputs, inputs)
            self._accumulate_gene_pool_metrics(self._train_metric_sums, gene_pool_loss, effective_gene_pool_lambda)
        outputs = {
            "sc": sc_outputs,
            "mc": mc_outputs,
            "gene_pool_contrastive_loss": gene_pool_loss,
            "effective_gene_pool_contrastive_lambda": float(effective_gene_pool_lambda),
        }
        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        control = super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
        )
        self._maybe_save_exact_boundaries(model=model, trial=trial)
        if self._maybe_save_time_boundaries(model=model, trial=trial):
            if control is None:
                control = self.control
            control.should_training_stop = True
            self.control.should_training_stop = True
        return control

    def _accumulate_seen_tokens(self, attention_mask: torch.Tensor, *, view: str | None = None) -> None:
        token_sum = float(attention_mask.detach().sum().item())
        if dist.is_available() and dist.is_initialized() and not self._training_metric_local_accumulation:
            reduced = torch.tensor(token_sum, device=attention_mask.device, dtype=torch.float64)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            token_sum = float(reduced.item())
        elif dist.is_available() and dist.is_initialized():
            token_sum *= dist.get_world_size()
        self._seen_non_pad_tokens += token_sum
        if view == "sc":
            self._seen_sc_non_pad_tokens += token_sum
        elif view == "mc":
            self._seen_mc_non_pad_tokens += token_sum

    def _accumulate_seen_examples_and_raw_cells(
        self,
        inputs: dict[str, Any],
        *,
        attention_key: str = "attention_mask",
        raw_cell_key: str = "raw_cell_count",
    ) -> None:
        attention_mask = inputs.get(attention_key)
        if attention_mask is None:
            return
        examples = float(int(attention_mask.shape[0]))
        raw_cells = 0.0
        raw_cell_count = inputs.get(raw_cell_key)
        if raw_cell_count is not None:
            if torch.is_tensor(raw_cell_count):
                raw_cells = float(raw_cell_count.detach().to(dtype=torch.float64).sum().item())
            else:
                raw_cells = float(np.asarray(raw_cell_count, dtype=np.float64).sum())
        if dist.is_available() and dist.is_initialized() and not self._training_metric_local_accumulation:
            device = attention_mask.device if torch.is_tensor(attention_mask) else self.args.device
            reduced = torch.tensor([examples, raw_cells], device=device, dtype=torch.float64)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            examples = float(reduced[0].item())
            raw_cells = float(reduced[1].item())
        elif dist.is_available() and dist.is_initialized():
            examples *= dist.get_world_size()
            raw_cells *= dist.get_world_size()
        self._seen_examples += examples
        self._seen_raw_cells += raw_cells

    def _zero_loss_connected_to_model(self, model: nn.Module, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.new_zeros(()) if torch.is_tensor(reference) else torch.tensor(0.0, device=self.args.device)
        for param in model.parameters():
            if param.requires_grad and param.numel() > 0:
                zero = zero + (param.reshape(-1)[0].to(dtype=zero.dtype) * 0.0)
        return zero

    def _backward_single_loss(self, loss: torch.Tensor) -> torch.Tensor:
        loss_for_backward = loss.mean() if torch.is_tensor(loss) and loss.ndim > 0 else loss
        if self.args.n_gpu > 1:
            loss_for_backward = loss_for_backward.mean()
        if torch.is_tensor(loss_for_backward) and not loss_for_backward.requires_grad:
            loss_for_backward = self._zero_loss_connected_to_model(self.model, loss_for_backward)
        if self.use_apex:
            from apex import amp as apex_amp

            with apex_amp.scale_loss(loss_for_backward, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss_for_backward)
        return loss_for_backward.detach()

    def _maybe_update_embedding_lr_schedule(self) -> None:
        step = int(self.state.global_step)
        if self._embedding_lr_schedule_switched:
            return
        if self._embedding_lr_multiplier_after_step is None or step < self._embedding_lr_multiplier_after_step:
            return
        if self.optimizer is None:
            return
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        embedding_param = model_obj.get_token_embedding_layer().weight
        embedding_index = None
        transformer_index = None
        for index, group in enumerate(self.optimizer.param_groups):
            if any(param is embedding_param for param in group["params"]):
                embedding_index = index
            elif transformer_index is None:
                transformer_index = index
        if embedding_index is None or transformer_index is None:
            raise RuntimeError("embedding LR schedule requires separate embedding and transformer parameter groups")
        target_multiplier = float(self._embedding_lr_multiplier_after)
        transformer_group = self.optimizer.param_groups[transformer_index]
        embedding_group = self.optimizer.param_groups[embedding_index]
        transformer_initial_lr = float(transformer_group.get("initial_lr", transformer_group["lr"]))
        embedding_group["initial_lr"] = transformer_initial_lr * target_multiplier
        embedding_group["lr"] = float(transformer_group["lr"]) * target_multiplier
        scheduler = self.lr_scheduler
        if scheduler is not None and hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs[embedding_index] = embedding_group["initial_lr"]
        self._embedding_lr_schedule_switched = True
        if self.is_world_process_zero():
            print(
                "[embedding_lr_schedule_switch] "
                f"step={step} multiplier={target_multiplier:.8g} embedding_lr={embedding_group['lr']:.10g}",
                flush=True,
            )

    def training_step(self, model, inputs, *args, **kwargs):
        self._maybe_update_embedding_lr_schedule()
        if not self._paired_batch_enabled(inputs):
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            self._record_exact_train_coverage(inputs)
            attn = inputs.get("attention_mask")
            if attn is not None:
                self._accumulate_seen_tokens(attn, view=None)
                self._accumulate_seen_examples_and_raw_cells(inputs)
            num_items_in_batch = kwargs.get("num_items_in_batch")
            if num_items_in_batch is None and args:
                num_items_in_batch = args[0]
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            del inputs
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if torch.is_tensor(loss) and not loss.requires_grad:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                print(
                    f"[rank{rank}] no-grad loss encountered; applying zero-gradient surrogate",
                    flush=True,
                )
                loss = self._zero_loss_connected_to_model(model, loss)
            if self.use_apex:
                from apex import amp as apex_amp

                with apex_amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss = loss * self.args.gradient_accumulation_steps
                self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

        model.train()
        inputs = self._prepare_inputs(inputs)
        self._record_exact_train_coverage(inputs)
        if "sc_attention_mask" in inputs:
            self._accumulate_seen_tokens(inputs["sc_attention_mask"], view="sc")
            self._accumulate_seen_examples_and_raw_cells(
                inputs,
                attention_key="sc_attention_mask",
                raw_cell_key="sc_raw_cell_count",
            )
        if "mc_attention_mask" in inputs:
            self._accumulate_seen_tokens(inputs["mc_attention_mask"], view="mc")

        total_loss: torch.Tensor | None = None
        effective_gene_pool_lambda = self._effective_gene_pool_lambda()

        with self.compute_loss_context_manager():
            self._check_required_program_targets(inputs, prefix="sc")
            sc_outputs = model(**self._extract_view_inputs(inputs, "sc"))
        gene_pool_loss = getattr(sc_outputs, "gene_pool_contrastive_loss", None)
        self._accumulate_gene_pool_metrics(self._train_metric_sums, gene_pool_loss, effective_gene_pool_lambda)
        self._accumulate_metric_sums_view(self._train_metric_sums, sc_outputs, view="sc")
        self._accumulate_masked_accuracy(
            self._train_metric_sums,
            getattr(sc_outputs, "logits", None),
            inputs.get("sc_labels"),
            view="sc",
        )
        sc_loss = self._extract_loss_tensor(sc_outputs)
        sc_weighted_loss = sc_loss
        if gene_pool_loss is not None:
            gene_pool_weighted_loss = gene_pool_loss * float(effective_gene_pool_lambda)
            sc_weighted_loss = gene_pool_weighted_loss if sc_weighted_loss is None else (sc_weighted_loss + gene_pool_weighted_loss)
        if sc_weighted_loss is not None:
            total_loss = self._backward_single_loss(sc_weighted_loss)
        del sc_weighted_loss
        del sc_outputs, sc_loss

        with self.compute_loss_context_manager():
            self._check_required_program_targets(inputs, prefix="mc")
            mc_outputs = model(**self._extract_view_inputs(inputs, "mc"))
        self._accumulate_metric_sums_view(self._train_metric_sums, mc_outputs, view="mc")
        self._accumulate_masked_accuracy(
            self._train_metric_sums,
            getattr(mc_outputs, "logits", None),
            inputs.get("mc_labels"),
            view="mc",
        )
        mc_loss = self._extract_loss_tensor(mc_outputs)
        mc_gene_pool_loss = getattr(mc_outputs, "gene_pool_contrastive_loss", None)
        weighted_mc_loss = mc_loss * self._paired_mc_loss_weight if mc_loss is not None else None
        if mc_gene_pool_loss is not None:
            mc_gene_pool_surrogate = mc_gene_pool_loss * 0.0
            weighted_mc_loss = (
                mc_gene_pool_surrogate if weighted_mc_loss is None else (weighted_mc_loss + mc_gene_pool_surrogate)
            )
        if weighted_mc_loss is not None:
            mc_detached = self._backward_single_loss(weighted_mc_loss)
            total_loss = mc_detached if total_loss is None else total_loss + mc_detached
            del weighted_mc_loss, mc_detached
        del mc_outputs, mc_loss, mc_gene_pool_loss
        del gene_pool_loss, inputs

        if total_loss is None:
            raise ValueError("paired mode requires at least one non-null loss")
        return total_loss / self.args.gradient_accumulation_steps

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs or ("sc_labels" in inputs and "mc_labels" in inputs)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self._paired_batch_enabled(inputs):
                sc_outputs, mc_outputs = self._forward_paired_outputs(model, inputs)
                gene_pool_loss = getattr(sc_outputs, "gene_pool_contrastive_loss", None)
                effective_gene_pool_lambda = self._effective_gene_pool_lambda()
                loss = (
                    self._paired_total_loss(
                        sc_outputs,
                        mc_outputs,
                        gene_pool_loss=gene_pool_loss,
                        effective_gene_pool_lambda=effective_gene_pool_lambda,
                    ).detach()
                    if has_labels
                    else None
                )
                self._accumulate_paired_metrics(self._eval_metric_sums, sc_outputs, mc_outputs, inputs)
                self._accumulate_gene_pool_metrics(self._eval_metric_sums, gene_pool_loss, effective_gene_pool_lambda)
                return loss, None, None

            model_inputs = self._filter_model_inputs(inputs)
            outputs = model(**model_inputs)
            dual_outputs = None
            loss_tensor = outputs.loss
            if self._dual_batch_enabled(inputs):
                dual_outputs = model(**self._extract_view_inputs(inputs, "dual"))
                if loss_tensor is not None and dual_outputs.loss is not None:
                    loss_tensor = 0.5 * (loss_tensor + dual_outputs.loss)
            loss = loss_tensor.mean().detach() if has_labels and loss_tensor is not None else None
            self._accumulate_metric_sums_view(self._eval_metric_sums, outputs, view=None)
            if has_labels:
                self._accumulate_masked_accuracy(self._eval_metric_sums, getattr(outputs, "logits", None), inputs["labels"], view=None)
                self._accumulate_masking_contract_metrics(self._eval_metric_sums, inputs, view=None)
                self._accumulate_value_target_breakdown(self._eval_metric_sums, outputs, inputs)
                if dual_outputs is not None:
                    dual_metric_inputs = self._extract_view_metric_inputs(inputs, "dual")
                    self._accumulate_metric_sums_view(self._eval_metric_sums, dual_outputs, view=None)
                    self._accumulate_masked_accuracy(
                        self._eval_metric_sums,
                        getattr(dual_outputs, "logits", None),
                        inputs["dual_labels"],
                        view=None,
                    )
                    self._accumulate_masking_contract_metrics(self._eval_metric_sums, dual_metric_inputs, view=None)
                    self._accumulate_value_target_breakdown(self._eval_metric_sums, dual_outputs, dual_metric_inputs)
            if self._stream_eval_masked_acc_only:
                return loss, None, None
            logits = outputs.logits.detach() if (not prediction_loss_only and outputs.logits is not None) else None
        labels = inputs["labels"].detach() if has_labels and not prediction_loss_only else None
        if prediction_loss_only:
            return loss, None, None
        return loss, logits, labels

    def evaluation_loop(self, *args, **kwargs):
        metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
        self._reset_eval_metric_sums()
        output = super().evaluation_loop(*args, **kwargs)
        output.metrics.update(self._format_metric_logs(self._eval_metric_sums, prefix=f"{metric_key_prefix}_"))
        return output

    def log(self, logs):
        if logs is not None:
            world_size = _get_world_size_safe()
            elapsed_hours = (time.time() - self._train_start_wall) / 3600.0
            logs.setdefault("train_seen_examples", float(self._seen_examples))
            logs.setdefault("train_seen_raw_cells", float(self._seen_raw_cells))
            logs.setdefault("train_seen_non_pad_tokens", float(self._seen_non_pad_tokens))
            logs.setdefault("train_seen_sc_non_pad_tokens", float(self._seen_sc_non_pad_tokens))
            logs.setdefault("train_seen_mc_non_pad_tokens", float(self._seen_mc_non_pad_tokens))
            logs.setdefault("train_optimizer_steps", float(self.state.global_step))
            logs.setdefault("train_cumulative_gpu_hours", float(elapsed_hours * world_size))
            logs.update(self._format_metric_logs(self._train_metric_sums, prefix="train_"))
            self._reset_train_metric_sums()
        return super().log(logs)


def preprocess_logits_for_metrics(logits, labels):
    pred1 = logits.argmax(dim=-1)
    top5 = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
    top10 = torch.topk(logits, k=min(10, logits.size(-1)), dim=-1).indices
    return (pred1, top5, top10)


def compute_metrics(eval_pred: EvalPrediction):
    pred1, top5, top10 = eval_pred.predictions
    labels = eval_pred.label_ids
    mask = labels != -100
    denom = int(mask.sum())
    out = {}
    if denom > 0:
        out["masked_acc"] = float(((pred1 == labels) & mask).sum() / denom)
        lab_exp = labels[..., None]
        out["masked_top5_acc"] = float((np.any(top5 == lab_exp, axis=-1) & mask).sum() / denom)
        out["masked_top10_acc"] = float((np.any(top10 == lab_exp, axis=-1) & mask).sum() / denom)
        out["masked_n"] = float(denom)
        out["mask_only_top1_acc"] = out["masked_acc"]
        out["mask_only_top5_acc"] = out["masked_top5_acc"]
        out["mask_only_top10_acc"] = out["masked_top10_acc"]
        out["gene_mask_only_top1_acc"] = out["masked_acc"]
        out["gene_mask_only_top5_acc"] = out["masked_top5_acc"]
        out["gene_mask_only_top10_acc"] = out["masked_top10_acc"]
        out["gene_masked_n"] = float(denom)
    return out


def write_training_done(output_dir: Path, payload: dict[str, Any]):
    marker = output_dir / "TRAINING_DONE.json"
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def cuda_peak_memory_stats_mb() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }

    device = torch.cuda.current_device()
    stats = torch.tensor(
        [
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
            float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
        ],
        dtype=torch.float32,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return {
        "cuda_max_memory_allocated_mb": float(stats[0].item()),
        "cuda_max_memory_reserved_mb": float(stats[1].item()),
    }


def unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_or_load_model(args: argparse.Namespace, token_dictionary: dict[str, int]) -> ValueAugmentedBertForMaskedLM:
    expected_vocab_size = int(len(token_dictionary))
    fixed_gene_feature_path = (
        Path(args.fixed_gene_feature_npy).expanduser().resolve()
        if args.fixed_gene_feature_npy
        else None
    )
    fixed_gene_feature_array: np.ndarray | None = None
    expected_fixed_gene_feature_dim = 0
    if fixed_gene_feature_path is not None:
        if not fixed_gene_feature_path.exists():
            raise FileNotFoundError(f"fixed gene feature matrix not found: {fixed_gene_feature_path}")
        fixed_gene_feature_array = np.asarray(np.load(fixed_gene_feature_path), dtype=np.float32)
        if fixed_gene_feature_array.ndim != 2 or fixed_gene_feature_array.shape[0] != expected_vocab_size:
            raise ValueError(
                "fixed gene feature matrix must have shape "
                f"[{expected_vocab_size},D], got {fixed_gene_feature_array.shape}"
            )
        if not np.isfinite(fixed_gene_feature_array).all():
            raise ValueError("fixed gene feature matrix contains non-finite values")
        expected_fixed_gene_feature_dim = int(fixed_gene_feature_array.shape[1])
    expected_value_vocab_size = (
        int(args.value_vocab_size)
        if args.value_vocab_size is not None
        else int(max(args.n_value_bins + 2, args.value_mask_id + 1, args.value_pad_id + 1))
    )
    expected_encoder_layout = str(args.encoder_layout).strip().lower()
    expected_num_loop_steps = int(args.num_loop_steps) if args.num_loop_steps is not None else int(args.num_hidden_layers)
    expected_num_unique_layers = (
        int(args.num_unique_layers)
        if args.num_unique_layers is not None
        else (1 if expected_encoder_layout == "looped" else int(args.num_hidden_layers))
    )
    (
        expected_gene_mlm_mask_replace_prob,
        expected_gene_mlm_random_replace_prob,
        expected_gene_mlm_unchanged_prob,
    ) = resolve_gene_mlm_corruption_probs(
        gene_mlm_policy=str(args.gene_mlm_policy),
        mask_replace_prob=args.gene_mlm_mask_replace_prob,
        random_replace_prob=args.gene_mlm_random_replace_prob,
    )
    expected_use_loop_step_embedding = bool(args.use_loop_step_embedding)
    expected_varcov_layers = normalize_varcov_layers(args.varcov_layers, int(args.num_hidden_layers))
    expected_varcov_positions = str(args.varcov_positions).strip().lower()
    expected_view_pool_excluded_token_ids = sorted(
        int(token_id)
        for token, token_id in token_dictionary.items()
        if str(token).startswith("<")
    )
    if int(args.hidden_size) % int(args.num_attention_heads) != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads: "
            f"{args.hidden_size} vs {args.num_attention_heads}"
        )
    init_model_dir = Path(args.init_model_dir).expanduser().resolve() if args.init_model_dir else None
    if init_model_dir is None:
        config = BertConfig(
            hidden_size=int(args.hidden_size),
            num_hidden_layers=int(args.num_hidden_layers),
            num_attention_heads=int(args.num_attention_heads),
            intermediate_size=int(args.intermediate_size),
            max_position_embeddings=args.max_input_size,
            vocab_size=expected_vocab_size,
            pad_token_id=token_dictionary["<pad>"],
            norm_position=str(args.norm_position),
            position_embedding_alpha=float(args.position_embedding_alpha),
            pre_ln_final_layer_norm=bool(args.pre_ln_final_layer_norm),
            encoder_layout=expected_encoder_layout,
            num_loop_steps=expected_num_loop_steps,
            num_unique_layers=expected_num_unique_layers,
            use_loop_step_embedding=expected_use_loop_step_embedding,
            domain_token_type_id=(
                int(args.domain_token_type_id) if args.domain_token_type_id is not None else None
            ),
            sc_token_type_id=int(args.sc_token_type_id),
            mc_token_type_id=int(args.mc_token_type_id),
            use_value_input=bool(args.use_value_input),
            use_value_loss=bool(args.use_value_loss),
            fixed_coordinate=bool(args.fixed_coordinate),
            gene_mlm_enabled=bool(args.gene_mlm_enabled),
            cosine_mlm_head=bool(args.cosine_mlm_head),
            cosine_mlm_temperature=float(args.cosine_mlm_temperature),
            fixed_gene_feature_dim=expected_fixed_gene_feature_dim,
            fixed_gene_feature_scale=float(args.fixed_gene_feature_scale),
            retrieval_feature_weight=float(args.retrieval_feature_weight),
            gene_mlm_policy=str(args.gene_mlm_policy),
            gene_mlm_mask_replace_prob=float(expected_gene_mlm_mask_replace_prob),
            gene_mlm_random_replace_prob=float(expected_gene_mlm_random_replace_prob),
            gene_mlm_unchanged_prob=float(expected_gene_mlm_unchanged_prob),
            value_target_decoupled=bool(args.value_target_decoupled),
            value_target_probability=float(args.value_target_probability),
            value_target_scope=str(args.value_target_scope),
            visible_gene_discrimination_enable=bool(args.visible_gene_discrimination_enable),
            masked_gene_discrimination_enable=bool(args.masked_gene_discrimination_enable),
            masked_gene_disc_probability=float(args.masked_gene_disc_probability),
            visible_gene_loss_weight=float(args.visible_gene_loss_weight),
            masked_gene_loss_weight=float(args.masked_gene_loss_weight),
            rtd_enable=bool(args.rtd_enable),
            rtd_loss_weight=float(args.rtd_loss_weight),
            rtd_visible_positive_policy=str(args.rtd_visible_positive_policy),
            view_consistency_enable=bool(args.view_consistency_enable),
            view_consistency_weight=float(args.view_consistency_weight),
            view_pool_excluded_token_ids=expected_view_pool_excluded_token_ids,
            gene_disc_support_bucket_count=int(args.gene_disc_support_bucket_count),
            value_loss_weight=float(args.value_loss_weight),
            use_program_loss=bool(args.program_targets_manifest and float(args.program_loss_weight) != 0.0),
            program_loss_weight=float(args.program_loss_weight),
            program_dim=int(args.program_dim),
            program_huber_delta=float(args.program_huber_delta),
            use_gene_loading_loss=bool(
                args.gene_loading_targets_manifest and float(args.gene_loading_loss_weight) != 0.0
            ),
            gene_loading_loss_weight=float(args.gene_loading_loss_weight),
            gene_loading_dim=int(args.gene_loading_dim),
            gene_loading_huber_delta=float(args.gene_loading_huber_delta),
            gene_loading_gradient_scope=str(args.gene_loading_gradient_scope),
            value_mask_ratio=float(args.value_mask_ratio),
            value_nonzero_only=bool(args.value_nonzero_only),
            ordinal_huber_weight=float(args.ordinal_huber_weight),
            ordinal_huber_delta=float(args.ordinal_huber_delta),
            varcov_enable=bool(args.varcov_enable),
            varcov_lambda=float(args.varcov_lambda),
            varcov_layers=list(expected_varcov_layers),
            varcov_positions=expected_varcov_positions,
            n_value_bins=int(args.n_value_bins),
            value_pad_id=int(args.value_pad_id),
            value_mask_id=int(args.value_mask_id),
            value_vocab_size=expected_value_vocab_size,
            input_subset_mode=str(args.input_subset_mode),
            value_target_mode=str(args.value_target_mode),
            subset_top_count=int(args.subset_top_count),
            subset_random_count=int(args.subset_random_count),
            subset_fill_count=int(args.subset_fill_count),
            target_nonzero_fraction=float(args.target_nonzero_fraction),
            randorder_seed=int(args.randorder_seed),
            fixed_length_randorder=bool(args.fixed_length_randorder),
            use_gene_pool_contrastive=bool(args.gene_pool_contrastive_enable),
            gene_pool_projection_dim=int(args.gene_pool_projection_dim),
            gene_pool_temperature=float(args.gene_pool_temperature),
            gene_pool_stopgrad_anchor=bool(args.gene_pool_stopgrad_anchor),
            gene_pool_excluded_token_ids=sorted({int(token_dictionary["<pad>"]), int(token_dictionary["<mask>"])}),
        )
        if str(args.attn_implementation) != "auto":
            config._attn_implementation = str(args.attn_implementation)
            config.attn_implementation = str(args.attn_implementation)
        config.architectures = ["ValueAugmentedBertForMaskedLM"]
        model = ValueAugmentedBertForMaskedLM(config)
        if fixed_gene_feature_array is not None:
            with torch.no_grad():
                model.fixed_gene_features.copy_(torch.from_numpy(fixed_gene_feature_array))
        return model

    config = BertConfig.from_pretrained(str(init_model_dir))
    if str(args.attn_implementation) != "auto":
        config._attn_implementation = str(args.attn_implementation)
        config.attn_implementation = str(args.attn_implementation)
    config.norm_position = str(args.norm_position)
    config.position_embedding_alpha = float(args.position_embedding_alpha)
    config.pre_ln_final_layer_norm = bool(args.pre_ln_final_layer_norm)
    config.encoder_layout = expected_encoder_layout
    config.num_loop_steps = expected_num_loop_steps
    config.num_unique_layers = expected_num_unique_layers
    config.use_loop_step_embedding = expected_use_loop_step_embedding
    config.domain_token_type_id = (
        int(args.domain_token_type_id) if args.domain_token_type_id is not None else None
    )
    config.sc_token_type_id = int(args.sc_token_type_id)
    config.mc_token_type_id = int(args.mc_token_type_id)
    config.use_value_input = bool(args.use_value_input)
    config.use_value_loss = bool(args.use_value_loss)
    config.fixed_coordinate = bool(args.fixed_coordinate)
    config.gene_mlm_enabled = bool(args.gene_mlm_enabled)
    config.cosine_mlm_head = bool(args.cosine_mlm_head)
    config.cosine_mlm_temperature = float(args.cosine_mlm_temperature)
    config.gene_mlm_policy = str(args.gene_mlm_policy)
    config.gene_mlm_mask_replace_prob = float(expected_gene_mlm_mask_replace_prob)
    config.gene_mlm_random_replace_prob = float(expected_gene_mlm_random_replace_prob)
    config.gene_mlm_unchanged_prob = float(expected_gene_mlm_unchanged_prob)
    config.value_target_decoupled = bool(args.value_target_decoupled)
    config.value_target_probability = float(args.value_target_probability)
    config.value_target_scope = str(args.value_target_scope)
    config.visible_gene_discrimination_enable = bool(args.visible_gene_discrimination_enable)
    config.masked_gene_discrimination_enable = bool(args.masked_gene_discrimination_enable)
    config.masked_gene_disc_probability = float(args.masked_gene_disc_probability)
    config.visible_gene_loss_weight = float(args.visible_gene_loss_weight)
    config.masked_gene_loss_weight = float(args.masked_gene_loss_weight)
    config.rtd_enable = bool(args.rtd_enable)
    config.rtd_loss_weight = float(args.rtd_loss_weight)
    config.rtd_visible_positive_policy = str(args.rtd_visible_positive_policy)
    config.view_consistency_enable = bool(args.view_consistency_enable)
    config.view_consistency_weight = float(args.view_consistency_weight)
    config.view_pool_excluded_token_ids = list(expected_view_pool_excluded_token_ids)
    config.gene_disc_support_bucket_count = int(args.gene_disc_support_bucket_count)
    config.value_loss_weight = float(args.value_loss_weight)
    config.use_program_loss = bool(args.program_targets_manifest and float(args.program_loss_weight) != 0.0)
    config.program_loss_weight = float(args.program_loss_weight)
    config.program_dim = int(args.program_dim)
    config.program_huber_delta = float(args.program_huber_delta)
    config.use_gene_loading_loss = bool(
        args.gene_loading_targets_manifest and float(args.gene_loading_loss_weight) != 0.0
    )
    config.gene_loading_loss_weight = float(args.gene_loading_loss_weight)
    config.gene_loading_dim = int(args.gene_loading_dim)
    config.gene_loading_huber_delta = float(args.gene_loading_huber_delta)
    config.gene_loading_gradient_scope = str(args.gene_loading_gradient_scope)
    config.value_mask_ratio = float(args.value_mask_ratio)
    config.value_nonzero_only = bool(args.value_nonzero_only)
    config.ordinal_huber_weight = float(args.ordinal_huber_weight)
    config.ordinal_huber_delta = float(args.ordinal_huber_delta)
    config.varcov_enable = bool(args.varcov_enable)
    config.varcov_lambda = float(args.varcov_lambda)
    config.varcov_layers = list(expected_varcov_layers)
    config.varcov_positions = expected_varcov_positions
    config.n_value_bins = int(args.n_value_bins)
    config.value_pad_id = int(args.value_pad_id)
    config.value_mask_id = int(args.value_mask_id)
    config.value_vocab_size = expected_value_vocab_size
    config.input_subset_mode = str(args.input_subset_mode)
    config.value_target_mode = str(args.value_target_mode)
    config.subset_top_count = int(args.subset_top_count)
    config.subset_random_count = int(args.subset_random_count)
    config.subset_fill_count = int(args.subset_fill_count)
    config.target_nonzero_fraction = float(args.target_nonzero_fraction)
    config.randorder_seed = int(args.randorder_seed)
    config.fixed_length_randorder = bool(args.fixed_length_randorder)
    config.use_gene_pool_contrastive = bool(args.gene_pool_contrastive_enable)
    config.gene_pool_projection_dim = int(args.gene_pool_projection_dim)
    config.gene_pool_temperature = float(args.gene_pool_temperature)
    config.gene_pool_stopgrad_anchor = bool(args.gene_pool_stopgrad_anchor)
    config.gene_pool_excluded_token_ids = sorted({int(token_dictionary["<pad>"]), int(token_dictionary["<mask>"])})
    config.fixed_gene_feature_dim = expected_fixed_gene_feature_dim
    config.fixed_gene_feature_scale = float(args.fixed_gene_feature_scale)
    config.retrieval_feature_weight = float(args.retrieval_feature_weight)
    config.architectures = ["ValueAugmentedBertForMaskedLM"]
    model = ValueAugmentedBertForMaskedLM.from_pretrained(str(init_model_dir), config=config)
    if fixed_gene_feature_array is not None:
        with torch.no_grad():
            model.fixed_gene_features.copy_(torch.from_numpy(fixed_gene_feature_array))
    cfg = model.config
    checks = {
        "vocab_size": (int(getattr(cfg, "vocab_size", -1)), expected_vocab_size),
        "hidden_size": (int(getattr(cfg, "hidden_size", -1)), int(args.hidden_size)),
        "num_hidden_layers": (int(getattr(cfg, "num_hidden_layers", -1)), int(args.num_hidden_layers)),
        "num_attention_heads": (int(getattr(cfg, "num_attention_heads", -1)), int(args.num_attention_heads)),
        "intermediate_size": (int(getattr(cfg, "intermediate_size", -1)), int(args.intermediate_size)),
        "max_position_embeddings": (int(getattr(cfg, "max_position_embeddings", -1)), int(args.max_input_size)),
        "norm_position": (str(getattr(cfg, "norm_position", "post")), str(args.norm_position)),
        "position_embedding_alpha": (
            float(getattr(cfg, "position_embedding_alpha", 1.0)),
            float(args.position_embedding_alpha),
        ),
        "pre_ln_final_layer_norm": (
            bool(getattr(cfg, "pre_ln_final_layer_norm", True)),
            bool(args.pre_ln_final_layer_norm),
        ),
        "encoder_layout": (resolved_encoder_layout(cfg), expected_encoder_layout),
        "num_loop_steps": (resolved_num_loop_steps(cfg), expected_num_loop_steps),
        "num_unique_layers": (resolved_num_unique_layers(cfg), expected_num_unique_layers),
        "use_loop_step_embedding": (
            bool(getattr(cfg, "use_loop_step_embedding", False)),
            expected_use_loop_step_embedding,
        ),
        "use_value_input": (bool(getattr(cfg, "use_value_input", False)), bool(args.use_value_input)),
        "use_value_loss": (bool(getattr(cfg, "use_value_loss", False)), bool(args.use_value_loss)),
        "fixed_coordinate": (bool(getattr(cfg, "fixed_coordinate", False)), bool(args.fixed_coordinate)),
        "gene_mlm_enabled": (bool(getattr(cfg, "gene_mlm_enabled", True)), bool(args.gene_mlm_enabled)),
        "cosine_mlm_head": (bool(getattr(cfg, "cosine_mlm_head", False)), bool(args.cosine_mlm_head)),
        "cosine_mlm_temperature": (
            float(getattr(cfg, "cosine_mlm_temperature", 20.0)),
            float(args.cosine_mlm_temperature),
        ),
        "gene_mlm_policy": (str(getattr(cfg, "gene_mlm_policy", "bert_801010")), str(args.gene_mlm_policy)),
        "gene_mlm_corruption_probs": (
            resolve_gene_mlm_corruption_probs(
                gene_mlm_policy=str(getattr(cfg, "gene_mlm_policy", "bert_801010")),
                mask_replace_prob=getattr(cfg, "gene_mlm_mask_replace_prob", None),
                random_replace_prob=getattr(cfg, "gene_mlm_random_replace_prob", None),
                unchanged_prob=getattr(cfg, "gene_mlm_unchanged_prob", None),
            ),
            (
                float(expected_gene_mlm_mask_replace_prob),
                float(expected_gene_mlm_random_replace_prob),
                float(expected_gene_mlm_unchanged_prob),
            ),
        ),
        "value_target_decoupled": (
            bool(getattr(cfg, "value_target_decoupled", False)),
            bool(args.value_target_decoupled),
        ),
        "value_target_probability": (
            float(getattr(cfg, "value_target_probability", 0.15)),
            float(args.value_target_probability),
        ),
        "value_target_scope": (str(getattr(cfg, "value_target_scope", "observed_nonzero")), str(args.value_target_scope)),
        "visible_gene_discrimination_enable": (
            bool(getattr(cfg, "visible_gene_discrimination_enable", False)),
            bool(args.visible_gene_discrimination_enable),
        ),
        "masked_gene_discrimination_enable": (
            bool(getattr(cfg, "masked_gene_discrimination_enable", False)),
            bool(args.masked_gene_discrimination_enable),
        ),
        "masked_gene_disc_probability": (
            float(getattr(cfg, "masked_gene_disc_probability", 0.10)),
            float(args.masked_gene_disc_probability),
        ),
        "visible_gene_loss_weight": (
            float(getattr(cfg, "visible_gene_loss_weight", 0.03)),
            float(args.visible_gene_loss_weight),
        ),
        "masked_gene_loss_weight": (
            float(getattr(cfg, "masked_gene_loss_weight", 0.03)),
            float(args.masked_gene_loss_weight),
        ),
        "rtd_enable": (bool(getattr(cfg, "rtd_enable", False)), bool(args.rtd_enable)),
        "rtd_visible_positive_policy": (
            str(getattr(cfg, "rtd_visible_positive_policy", "neg_count_match")),
            str(args.rtd_visible_positive_policy),
        ),
        "view_consistency_enable": (
            bool(getattr(cfg, "view_consistency_enable", False)),
            bool(args.view_consistency_enable),
        ),
        "view_consistency_weight": (
            float(getattr(cfg, "view_consistency_weight", 0.01)),
            float(args.view_consistency_weight),
        ),
        "gene_disc_support_bucket_count": (
            int(getattr(cfg, "gene_disc_support_bucket_count", 64)),
            int(args.gene_disc_support_bucket_count),
        ),
        "program_dim": (
            int(getattr(cfg, "program_dim", int(args.program_dim))),
            int(args.program_dim),
        ),
        "gene_loading_dim": (
            int(getattr(cfg, "gene_loading_dim", int(args.gene_loading_dim))),
            int(args.gene_loading_dim),
        ),
        "gene_loading_gradient_scope": (
            str(getattr(cfg, "gene_loading_gradient_scope", "contextual")),
            str(args.gene_loading_gradient_scope),
        ),
        "n_value_bins": (int(getattr(cfg, "n_value_bins", -1)), int(args.n_value_bins)),
        "value_pad_id": (int(getattr(cfg, "value_pad_id", -1)), int(args.value_pad_id)),
        "value_mask_id": (int(getattr(cfg, "value_mask_id", -1)), int(args.value_mask_id)),
        "value_vocab_size": (int(getattr(cfg, "value_vocab_size", -1)), expected_value_vocab_size),
        "value_nonzero_only": (
            bool(getattr(cfg, "value_nonzero_only", False)),
            bool(args.value_nonzero_only),
        ),
        "fixed_length_randorder": (
            bool(getattr(cfg, "fixed_length_randorder", False)),
            bool(args.fixed_length_randorder),
        ),
        "varcov_enable": (bool(getattr(cfg, "varcov_enable", False)), bool(args.varcov_enable)),
        "varcov_layers": (
            normalize_varcov_layers(getattr(cfg, "varcov_layers", []), int(args.num_hidden_layers)),
            expected_varcov_layers,
        ),
        "varcov_positions": (
            str(getattr(cfg, "varcov_positions", "masked")).strip().lower(),
            expected_varcov_positions,
        ),
        "input_subset_mode": (str(getattr(cfg, "input_subset_mode", "legacy_fixed_length")), str(args.input_subset_mode)),
        "value_target_mode": (str(getattr(cfg, "value_target_mode", "legacy_all_masked")), str(args.value_target_mode)),
    }
    mismatches = [
        f"{name}: init={got} expected={want}"
        for name, (got, want) in checks.items()
        if got != want
    ]
    if mismatches:
        raise ValueError("init_model_dir is incompatible with requested run settings:\n" + "\n".join(mismatches))
    cfg.use_value_input = bool(args.use_value_input)
    cfg.use_value_loss = bool(args.use_value_loss)
    cfg.fixed_coordinate = bool(args.fixed_coordinate)
    cfg.gene_mlm_enabled = bool(args.gene_mlm_enabled)
    cfg.cosine_mlm_head = bool(args.cosine_mlm_head)
    cfg.cosine_mlm_temperature = float(args.cosine_mlm_temperature)
    cfg.gene_mlm_policy = str(args.gene_mlm_policy)
    cfg.gene_mlm_mask_replace_prob = float(expected_gene_mlm_mask_replace_prob)
    cfg.gene_mlm_random_replace_prob = float(expected_gene_mlm_random_replace_prob)
    cfg.gene_mlm_unchanged_prob = float(expected_gene_mlm_unchanged_prob)
    cfg.value_target_decoupled = bool(args.value_target_decoupled)
    cfg.value_target_probability = float(args.value_target_probability)
    cfg.value_target_scope = str(args.value_target_scope)
    cfg.visible_gene_discrimination_enable = bool(args.visible_gene_discrimination_enable)
    cfg.masked_gene_discrimination_enable = bool(args.masked_gene_discrimination_enable)
    cfg.masked_gene_disc_probability = float(args.masked_gene_disc_probability)
    cfg.visible_gene_loss_weight = float(args.visible_gene_loss_weight)
    cfg.masked_gene_loss_weight = float(args.masked_gene_loss_weight)
    cfg.rtd_enable = bool(args.rtd_enable)
    cfg.rtd_loss_weight = float(args.rtd_loss_weight)
    cfg.rtd_visible_positive_policy = str(args.rtd_visible_positive_policy)
    cfg.view_consistency_enable = bool(args.view_consistency_enable)
    cfg.view_consistency_weight = float(args.view_consistency_weight)
    cfg.view_pool_excluded_token_ids = list(expected_view_pool_excluded_token_ids)
    cfg.gene_disc_support_bucket_count = int(args.gene_disc_support_bucket_count)
    cfg.value_loss_weight = float(args.value_loss_weight)
    cfg.use_program_loss = bool(args.program_targets_manifest and float(args.program_loss_weight) != 0.0)
    cfg.program_loss_weight = float(args.program_loss_weight)
    cfg.program_dim = int(args.program_dim)
    cfg.program_huber_delta = float(args.program_huber_delta)
    cfg.use_gene_loading_loss = bool(
        args.gene_loading_targets_manifest and float(args.gene_loading_loss_weight) != 0.0
    )
    cfg.gene_loading_loss_weight = float(args.gene_loading_loss_weight)
    cfg.gene_loading_dim = int(args.gene_loading_dim)
    cfg.gene_loading_huber_delta = float(args.gene_loading_huber_delta)
    cfg.gene_loading_gradient_scope = str(args.gene_loading_gradient_scope)
    cfg.value_mask_ratio = float(args.value_mask_ratio)
    cfg.value_nonzero_only = bool(args.value_nonzero_only)
    cfg.ordinal_huber_weight = float(args.ordinal_huber_weight)
    cfg.ordinal_huber_delta = float(args.ordinal_huber_delta)
    cfg.varcov_enable = bool(args.varcov_enable)
    cfg.varcov_lambda = float(args.varcov_lambda)
    cfg.varcov_layers = list(expected_varcov_layers)
    cfg.varcov_positions = expected_varcov_positions
    cfg.input_subset_mode = str(args.input_subset_mode)
    cfg.value_target_mode = str(args.value_target_mode)
    cfg.subset_top_count = int(args.subset_top_count)
    cfg.subset_random_count = int(args.subset_random_count)
    cfg.subset_fill_count = int(args.subset_fill_count)
    cfg.target_nonzero_fraction = float(args.target_nonzero_fraction)
    cfg.randorder_seed = int(args.randorder_seed)
    cfg.fixed_length_randorder = bool(args.fixed_length_randorder)
    cfg.norm_position = str(args.norm_position)
    cfg.position_embedding_alpha = float(args.position_embedding_alpha)
    cfg.pre_ln_final_layer_norm = bool(args.pre_ln_final_layer_norm)
    cfg.encoder_layout = expected_encoder_layout
    cfg.num_loop_steps = expected_num_loop_steps
    cfg.num_unique_layers = expected_num_unique_layers
    cfg.use_loop_step_embedding = expected_use_loop_step_embedding
    cfg.domain_token_type_id = (
        int(args.domain_token_type_id) if args.domain_token_type_id is not None else None
    )
    cfg.sc_token_type_id = int(args.sc_token_type_id)
    cfg.mc_token_type_id = int(args.mc_token_type_id)
    cfg.use_gene_pool_contrastive = bool(args.gene_pool_contrastive_enable)
    cfg.gene_pool_projection_dim = int(args.gene_pool_projection_dim)
    cfg.gene_pool_temperature = float(args.gene_pool_temperature)
    cfg.gene_pool_stopgrad_anchor = bool(args.gene_pool_stopgrad_anchor)
    cfg.gene_pool_excluded_token_ids = sorted({int(token_dictionary["<pad>"]), int(token_dictionary["<mask>"])})
    model.value_loss_weight = float(args.value_loss_weight)
    model.use_program_loss = bool(args.program_targets_manifest and float(args.program_loss_weight) != 0.0)
    model.program_loss_weight = float(args.program_loss_weight)
    model.program_huber_delta = float(args.program_huber_delta)
    model.use_gene_loading_loss = bool(
        args.gene_loading_targets_manifest and float(args.gene_loading_loss_weight) != 0.0
    )
    model.gene_loading_loss_weight = float(args.gene_loading_loss_weight)
    model.gene_loading_dim = int(args.gene_loading_dim)
    model.gene_loading_huber_delta = float(args.gene_loading_huber_delta)
    model.gene_loading_gradient_scope = str(args.gene_loading_gradient_scope)
    model.ordinal_huber_weight = float(args.ordinal_huber_weight)
    model.ordinal_huber_delta = float(args.ordinal_huber_delta)
    model.cosine_mlm_head = bool(args.cosine_mlm_head)
    model.gene_mlm_policy = str(args.gene_mlm_policy)
    model.gene_mlm_mask_replace_prob = float(expected_gene_mlm_mask_replace_prob)
    model.gene_mlm_random_replace_prob = float(expected_gene_mlm_random_replace_prob)
    model.gene_mlm_unchanged_prob = float(expected_gene_mlm_unchanged_prob)
    model.value_target_decoupled = bool(args.value_target_decoupled)
    model.value_target_probability = float(args.value_target_probability)
    model.value_target_scope = str(args.value_target_scope)
    model.visible_gene_discrimination_enable = bool(args.visible_gene_discrimination_enable)
    model.masked_gene_discrimination_enable = bool(args.masked_gene_discrimination_enable)
    model.visible_gene_loss_weight = float(args.visible_gene_loss_weight)
    model.masked_gene_loss_weight = float(args.masked_gene_loss_weight)
    model.rtd_enable = bool(args.rtd_enable)
    model.rtd_loss_weight = float(args.rtd_loss_weight)
    model.rtd_visible_positive_policy = str(args.rtd_visible_positive_policy)
    model.rtd_head.requires_grad_(bool(args.rtd_enable) and float(args.rtd_loss_weight) != 0.0)
    model.view_consistency_enable = bool(args.view_consistency_enable)
    model.view_consistency_weight = float(args.view_consistency_weight)
    model.view_pool_excluded_token_ids = tuple(expected_view_pool_excluded_token_ids)
    model.varcov_enable = bool(args.varcov_enable)
    model.varcov_lambda = float(args.varcov_lambda)
    model.varcov_layers = expected_varcov_layers
    model.varcov_positions = expected_varcov_positions
    return model


def calibrate_gene_loading_loss_weight(
    *,
    model: ValueAugmentedBertForMaskedLM,
    collator: Any,
    train_dataset: Any,
    batch_size: int,
    target_ratio: float,
    output_dir: Path,
    device: torch.device,
    is_main: bool,
) -> dict[str, float]:
    """Calibrate a fixed SVD coefficient from native gene-embedding gradients.

    The calibration uses the first deterministic train microbatch, performs no
    optimizer update, and restores all RNG streams before returning.  The SVD
    branch is measured independently from the main MLM/Value/RTD objective.
    """
    if str(model.gene_loading_gradient_scope) != "native_embedding":
        raise ValueError("gradient calibration requires native_embedding gene-loading scope")
    if not 0.0 < float(target_ratio) < 1.0:
        raise ValueError(f"target_ratio must be in (0,1), got {target_ratio}")
    if len(train_dataset) < int(batch_size):
        raise ValueError("train dataset is smaller than calibration batch")

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    previous_weight = float(model.gene_loading_loss_weight)
    previous_training = bool(model.training)
    try:
        features = [train_dataset[index] for index in range(int(batch_size))]
        batch = collator(features)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        main_inputs = {
            key: value
            for key, value in batch.items()
            if key in HFMLMTrainer._SINGLE_MODEL_INPUT_KEYS
        }
        main_inputs.pop("gene_loading_targets", None)
        main_inputs.pop("gene_loading_target_mask", None)
        main_inputs.pop("gene_loading_original_input_ids", None)
        main_inputs.pop("gene_loading_original_attention_mask", None)

        model.to(device)
        model.train()
        embedding_weight = model.get_input_embeddings().weight
        main_outputs = model(**main_inputs)
        main_loss = getattr(main_outputs, "loss", None)
        if main_loss is None or not torch.isfinite(main_loss):
            raise RuntimeError("main calibration loss is missing or non-finite")
        main_grad = torch.autograd.grad(
            main_loss,
            embedding_weight,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if main_grad is None:
            raise RuntimeError("main calibration produced no native gene-embedding gradient")
        main_norm_sq = torch.sum(main_grad.detach().float().square())

        svd_inputs = {
            key: value
            for key, value in batch.items()
            if key in HFMLMTrainer._SINGLE_MODEL_INPUT_KEYS
        }
        for key in (
            "labels",
            "value_labels",
            "rtd_labels",
            "rtd_token_types",
            "program_targets",
            "program_target_mask",
            "visible_gene_disc_row_idx",
            "visible_gene_disc_col_idx",
            "visible_gene_disc_pos_gene_ids",
            "visible_gene_disc_neg_gene_ids",
            "visible_gene_disc_neg_types",
            "masked_gene_disc_row_idx",
            "masked_gene_disc_col_idx",
            "masked_gene_disc_pos_gene_ids",
            "masked_gene_disc_neg_gene_ids",
            "masked_gene_disc_neg_types",
        ):
            svd_inputs.pop(key, None)
        svd_inputs.pop("gene_pool_original_input_ids", None)
        svd_inputs.pop("gene_pool_original_attention_mask", None)
        svd_inputs.pop("gene_pool_anchor_input_ids", None)
        svd_inputs.pop("gene_pool_anchor_attention_mask", None)
        svd_inputs.pop("gene_pool_positive_indices", None)

        model.gene_loading_loss_weight = 1.0
        svd_outputs = model(**svd_inputs)
        svd_loss = getattr(svd_outputs, "gene_loading_loss", None)
        if svd_loss is None or not torch.isfinite(svd_loss):
            raise RuntimeError("SVD calibration loss is missing or non-finite")
        svd_grad = torch.autograd.grad(
            svd_loss,
            embedding_weight,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if svd_grad is None:
            raise RuntimeError("SVD calibration produced no native gene-embedding gradient")
        svd_norm_sq = torch.sum(svd_grad.detach().float().square())

        norms = torch.stack([main_norm_sq, svd_norm_sq]).to(device=device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(norms, op=dist.ReduceOp.SUM)
            norms /= float(dist.get_world_size())
        main_norm = float(torch.sqrt(norms[0].clamp_min(0.0)).item())
        svd_norm = float(torch.sqrt(norms[1].clamp_min(0.0)).item())
        if not math.isfinite(main_norm) or main_norm <= 0.0:
            raise RuntimeError(f"invalid main gene-embedding gradient norm: {main_norm}")
        if not math.isfinite(svd_norm) or svd_norm <= 0.0:
            raise RuntimeError(f"invalid SVD gene-embedding gradient norm: {svd_norm}")
        coefficient = float(target_ratio) * main_norm / svd_norm
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise RuntimeError(f"invalid calibrated gene-loading coefficient: {coefficient}")
        model.gene_loading_loss_weight = coefficient
        model.config.gene_loading_loss_weight = coefficient
        result = {
            "target_ratio": float(target_ratio),
            "main_gene_embedding_grad_norm": main_norm,
            "svd_unit_gene_embedding_grad_norm": svd_norm,
            "calibrated_coefficient": coefficient,
            "observed_ratio": coefficient * svd_norm / main_norm,
        }
        if is_main:
            write_json(
                output_dir / "gene_loading_calibration.json",
                {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "device": str(device),
                    "batch_size": int(batch_size),
                    **result,
                },
            )
        return result
    finally:
        model.gene_loading_loss_weight = (
            float(result["calibrated_coefficient"])
            if "result" in locals()
            else previous_weight
        )
        model.config.gene_loading_loss_weight = float(model.gene_loading_loss_weight)
        if previous_training:
            model.train()
        else:
            model.eval()
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def main():
    args = parse_args()
    evaluation_strategy, effective_eval_steps, effective_save_steps = resolve_train_eval_save_policy(args)

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    disable_caching()

    training_seed = int(args.seed)
    data_seed = int(args.data_seed) if args.data_seed is not None else training_seed
    hf_data_seed = data_seed
    try:
        import accelerate
        from packaging.version import Version

        if Version(str(accelerate.__version__)) < Version("1.1.0"):
            hf_data_seed = None
    except Exception:
        hf_data_seed = None

    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    is_main = rank == 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_dictionary = load_token_dictionary_any(args.token_dictionary)
    (
        gene_mlm_mask_replace_prob,
        gene_mlm_random_replace_prob,
        gene_mlm_unchanged_prob,
    ) = resolve_gene_mlm_corruption_probs(
        gene_mlm_policy=str(args.gene_mlm_policy),
        mask_replace_prob=args.gene_mlm_mask_replace_prob,
        random_replace_prob=args.gene_mlm_random_replace_prob,
    )
    train_ds = load_dataset_any(args.train_dataset_dir)
    eval_ds = load_dataset_any(args.eval_dataset_dir)
    if args.eval_max_examples is not None and len(eval_ds) > int(args.eval_max_examples):
        keep_indices = range(int(args.eval_max_examples))
        eval_ds = eval_ds.select(keep_indices) if hasattr(eval_ds, "select") else Subset(eval_ds, keep_indices)
    paired_mode = str(args.paired_mode).strip().lower()
    gene_pool_lambda_points = parse_schedule_points(args.gene_pool_lambda_point)
    if paired_mode != "off" and not args.stream_eval_masked_acc_only:
        raise ValueError("paired_mode requires --stream_eval_masked_acc_only")
    if args.gene_pool_contrastive_enable and paired_mode == "off":
        raise ValueError("gene_pool contrastive requires paired_mode != off")
    if args.gene_pool_contrastive_enable and not gene_pool_lambda_points:
        raise ValueError("gene_pool contrastive requires at least one --gene_pool_lambda_point")
    if args.dual_mask_views and paired_mode != "off":
        raise ValueError("dual_mask_views currently requires paired_mode=off")
    if args.dual_mask_views and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("dual_mask_views currently requires input_subset_mode=legacy_fixed_length")
    if args.mask_sampling == "rank_frequency_balanced" and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("rank_frequency_balanced masking requires input_subset_mode=legacy_fixed_length")
    if args.hard_random_replacement and float(gene_mlm_random_replace_prob) <= 0.0:
        raise ValueError("hard_random_replacement requires gene_mlm_random_replace_prob > 0")
    if float(args.embedding_lr_multiplier) <= 0.0:
        raise ValueError("embedding_lr_multiplier must be positive")
    if not math.isfinite(float(args.fixed_gene_feature_scale)) or float(args.fixed_gene_feature_scale) < 0.0:
        raise ValueError("fixed_gene_feature_scale must be finite and non-negative")
    if not 0.0 <= float(args.retrieval_feature_weight) < 1.0:
        raise ValueError("retrieval_feature_weight must be in [0,1)")
    if args.fixed_gene_feature_npy and not Path(args.fixed_gene_feature_npy).expanduser().resolve().exists():
        raise FileNotFoundError(
            f"missing fixed_gene_feature_npy: {Path(args.fixed_gene_feature_npy).expanduser().resolve()}"
        )
    if bool(args.exact_train_coverage) and args.exact_train_coverage_expected_examples is None:
        raise ValueError(
            "--exact_train_coverage_expected_examples is required with --exact_train_coverage"
        )
    if not bool(args.exact_train_coverage) and args.exact_train_coverage_expected_examples is not None:
        raise ValueError(
            "--exact_train_coverage_expected_examples requires --exact_train_coverage"
        )
    if float(args.cosine_mlm_temperature) <= 0.0:
        raise ValueError("cosine_mlm_temperature must be positive")
    if not 0.0 <= float(args.cosine_min_lr_ratio) <= 1.0:
        raise ValueError("cosine_min_lr_ratio must be in [0, 1]")
    needs_sampling_stats = (
        args.mask_sampling == "rank_frequency_balanced"
        or bool(args.hard_random_replacement)
        or float(args.dataset_balance_exponent) > 0.0
    )
    if needs_sampling_stats and not args.sampling_stats_npz:
        raise ValueError("sampling_stats_npz is required by the requested sampling/weighting mode")
    if args.sampling_stats_npz and not Path(args.sampling_stats_npz).expanduser().resolve().exists():
        raise FileNotFoundError(f"missing sampling_stats_npz: {Path(args.sampling_stats_npz).expanduser().resolve()}")
    if args.value_target_decoupled and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("value_target_decoupled currently requires input_subset_mode=legacy_fixed_length")
    if args.value_target_decoupled and not bool(args.gene_mlm_enabled):
        raise ValueError("value_target_decoupled requires gene_mlm_enabled=True")
    if args.value_target_decoupled and not bool(args.use_value_loss):
        raise ValueError("value_target_decoupled requires use_value_loss=True")
    if args.value_target_decoupled and bool(args.use_value_input):
        raise ValueError("value_target_decoupled requires use_value_input=False")
    if bool(args.fixed_length_randorder) and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("fixed_length_randorder currently requires input_subset_mode=legacy_fixed_length")
    if args.visible_gene_discrimination_enable and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("visible_gene_discrimination currently requires input_subset_mode=legacy_fixed_length")
    if args.view_consistency_enable and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("view_consistency currently requires input_subset_mode=legacy_fixed_length")
    if float(args.view_consistency_weight) < 0.0:
        raise ValueError("view_consistency_weight must be >= 0")
    if not 0.0 <= float(args.view_gene_dropout_prob) <= 1.0:
        raise ValueError("view_gene_dropout_prob must be in [0, 1]")
    if not 0.0 <= float(args.view_value_jitter_prob) <= 1.0:
        raise ValueError("view_value_jitter_prob must be in [0, 1]")
    if int(args.view_value_jitter_delta) < 0:
        raise ValueError("view_value_jitter_delta must be >= 0")
    if args.masked_gene_discrimination_enable and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("masked_gene_discrimination currently requires input_subset_mode=legacy_fixed_length")
    if args.masked_gene_discrimination_enable and bool(args.gene_mlm_enabled):
        raise ValueError("masked_gene_discrimination currently requires gene_mlm_enabled=False")
    if args.masked_gene_discrimination_enable and not bool(args.use_value_input):
        raise ValueError("masked_gene_discrimination requires use_value_input=True")
    if args.visible_gene_discrimination_enable and not bool(args.use_value_loss):
        raise ValueError("visible_gene_discrimination requires use_value_loss=True")
    if args.masked_gene_discrimination_enable and not bool(args.use_value_loss):
        raise ValueError("masked_gene_discrimination requires use_value_loss=True")
    if args.rtd_enable and not bool(args.gene_mlm_enabled):
        raise ValueError("rtd_enable requires gene_mlm_enabled=True")
    if float(args.rtd_loss_weight) < 0.0:
        raise ValueError("rtd_loss_weight must be >= 0")
    if (args.visible_gene_discrimination_enable or args.masked_gene_discrimination_enable) and (
        int(args.visible_gene_disc_same_row_n)
        + int(args.visible_gene_disc_support_n)
        + int(args.visible_gene_disc_random_n)
        != int(args.visible_gene_disc_negative_k)
    ):
        raise ValueError("gene discrimination negative split must sum to visible_gene_disc_negative_k")
    if int(args.gene_disc_support_bucket_count) < 1:
        raise ValueError("gene_disc_support_bucket_count must be >= 1")
    gene_support_npy = Path(args.gene_support_npy).expanduser().resolve() if args.gene_support_npy else None
    if (args.visible_gene_discrimination_enable or args.masked_gene_discrimination_enable) and gene_support_npy is None:
        raise ValueError("gene discrimination requires --gene_support_npy")
    if gene_support_npy is not None and not gene_support_npy.exists():
        raise FileNotFoundError(f"missing gene_support_npy: {gene_support_npy}")
    if int(args.program_dim) <= 0:
        raise ValueError("program_dim must be positive")
    if float(args.program_huber_delta) <= 0.0:
        raise ValueError("program_huber_delta must be positive")
    if float(args.program_loss_weight) < 0.0:
        raise ValueError("program_loss_weight must be >= 0")
    if float(args.program_loss_weight) > 0.0 and not args.program_targets_manifest:
        raise ValueError("--program_targets_manifest is required when program_loss_weight > 0")
    if float(args.program_loss_weight) > 0.0 and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("program_loss currently requires input_subset_mode=legacy_fixed_length")
    if int(args.gene_loading_dim) <= 0:
        raise ValueError("gene_loading_dim must be positive")
    if float(args.gene_loading_huber_delta) <= 0.0:
        raise ValueError("gene_loading_huber_delta must be positive")
    if float(args.gene_loading_loss_weight) < 0.0:
        raise ValueError("gene_loading_loss_weight must be >= 0")
    if float(args.gene_loading_loss_weight) > 0.0 and not args.gene_loading_targets_manifest:
        raise ValueError("--gene_loading_targets_manifest is required when gene_loading_loss_weight > 0")
    if float(args.gene_loading_loss_weight) > 0.0 and str(args.input_subset_mode).strip().lower() != "legacy_fixed_length":
        raise ValueError("gene_loading_loss currently requires input_subset_mode=legacy_fixed_length")
    if args.gene_loading_calibration_target_ratio is not None:
        ratio = float(args.gene_loading_calibration_target_ratio)
        if not math.isfinite(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError("gene_loading_calibration_target_ratio must be finite and in (0,1)")
        if float(args.gene_loading_loss_weight) <= 0.0:
            raise ValueError(
                "gene_loading_calibration_target_ratio requires gene_loading_loss_weight > 0"
            )
    program_target_store = None
    program_targets_manifest = (
        Path(args.program_targets_manifest).expanduser().resolve()
        if args.program_targets_manifest
        else None
    )
    if program_targets_manifest is not None:
        program_target_store = ProgramTargetStore(program_targets_manifest)
        if int(program_target_store.program_dim) != int(args.program_dim):
            raise ValueError(
                "program target manifest dimension mismatch: "
                f"manifest={program_target_store.program_dim}, args={args.program_dim}"
            )

    gene_loading_target_store = None
    gene_loading_targets_manifest = (
        Path(args.gene_loading_targets_manifest).expanduser().resolve()
        if args.gene_loading_targets_manifest
        else None
    )
    if gene_loading_targets_manifest is not None:
        gene_loading_target_store = GeneLoadingTargetStore(gene_loading_targets_manifest)
        if int(gene_loading_target_store.loading_dim) != int(args.gene_loading_dim):
            raise ValueError(
                "gene loading target manifest dimension mismatch: "
                f"manifest={gene_loading_target_store.loading_dim}, args={args.gene_loading_dim}"
            )

    paired_lookup = None
    paired_avg_tokens_per_sample = 0.0
    if paired_mode != "off":
        if not args.paired_metacell_token_index_csv:
            raise ValueError("--paired_metacell_token_index_csv is required when paired_mode is enabled")
        paired_lookup = PairedMetacellLookup(args.paired_metacell_token_index_csv)
        paired_avg_tokens_per_sample = float(paired_lookup.mean_clipped_tokens(args.max_input_size))

    lengths_path = Path(args.train_lengths_pkl) if args.train_lengths_pkl else (output_dir / "train_lengths.pkl")
    lengths_path = ensure_lengths_pickle(train_ds, lengths_path)
    train_lengths = load_lengths(lengths_path)
    boundary_saves = parse_boundary_saves(args.boundary_save)
    time_boundary_saves = parse_time_boundary_saves(args.time_boundary_save)
    if args.stop_after_wall_seconds is not None:
        if not math.isfinite(float(args.stop_after_wall_seconds)) or float(args.stop_after_wall_seconds) <= 0.0:
            raise ValueError("--stop-after-wall-seconds must be positive finite")

    world_size = _get_world_size_safe()
    avg_tokens_per_sample = float(np.minimum(train_lengths, args.max_input_size).mean())
    if paired_mode != "off":
        avg_tokens_per_sample += paired_avg_tokens_per_sample
    tokens_per_update = avg_tokens_per_sample * args.per_device_batch * world_size * args.grad_accum_steps
    max_steps = args.max_steps
    if args.target_tokens is not None:
        max_steps = max(1, int(math.ceil(args.target_tokens / tokens_per_update)))

    exact_train_coverage_plan: ExactTrainCoveragePlan | None = None
    exact_train_coverage_tracker: ExactCoverageTracker | None = None
    effective_per_device_train_batch = int(args.per_device_batch)
    if bool(args.exact_train_coverage):
        expected_examples = int(args.exact_train_coverage_expected_examples)
        if len(train_ds) != expected_examples:
            raise ValueError(
                "exact train coverage source count mismatch: "
                f"dataset={len(train_ds)}, expected={expected_examples}"
            )
        exact_train_coverage_plan = build_exact_train_coverage_plan(
            expected_examples=expected_examples,
            target_group_size=int(args.per_device_batch),
            world_size=world_size,
            grad_accum_steps=int(args.grad_accum_steps),
            permutation_seed=data_seed,
        )
        if int(max_steps) > exact_train_coverage_plan.optimizer_steps_per_epoch:
            raise ValueError(
                "exact train coverage forbids training past its one-epoch boundary: "
                f"max_steps={max_steps}, epoch_steps={exact_train_coverage_plan.optimizer_steps_per_epoch}"
            )
        train_ds = ExactCoverageGroupedDataset(train_ds, exact_train_coverage_plan)
        exact_train_coverage_tracker = ExactCoverageTracker(exact_train_coverage_plan)
        effective_per_device_train_batch = exact_train_coverage_plan.dataloader_item_batch_size

    init_model_dir = str(Path(args.init_model_dir).expanduser().resolve()) if args.init_model_dir else None
    init_optimizer_state_path = (
        str(Path(args.init_optimizer_state_path).expanduser().resolve()) if args.init_optimizer_state_path else None
    )
    init_scheduler_state_path = (
        str(Path(args.init_scheduler_state_path).expanduser().resolve()) if args.init_scheduler_state_path else None
    )
    model = build_or_load_model(args, token_dictionary)

    if args.use_wandb and is_main:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name,
            anonymous=args.wandb_anonymous,
            config={
                "max_input_size": args.max_input_size,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "num_attention_heads": args.num_attention_heads,
                "intermediate_size": args.intermediate_size,
                "attn_implementation": args.attn_implementation,
                "norm_position": args.norm_position,
                "position_embedding_alpha": float(args.position_embedding_alpha),
                "pre_ln_final_layer_norm": bool(args.pre_ln_final_layer_norm),
                "encoder_layout": str(args.encoder_layout),
                "num_loop_steps": (
                    int(args.num_loop_steps) if args.num_loop_steps is not None else int(args.num_hidden_layers)
                ),
                "num_unique_layers": (
                    int(args.num_unique_layers)
                    if args.num_unique_layers is not None
                    else (1 if str(args.encoder_layout).strip().lower() == "looped" else int(args.num_hidden_layers))
                ),
                "use_loop_step_embedding": bool(args.use_loop_step_embedding),
                "domain_token_type_id": args.domain_token_type_id,
                "sc_token_type_id": int(args.sc_token_type_id),
                "mc_token_type_id": int(args.mc_token_type_id),
                "per_device_batch": args.per_device_batch,
                "grad_accum_steps": args.grad_accum_steps,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "learning_rate": float(args.learning_rate),
                "embedding_lr_multiplier": float(args.embedding_lr_multiplier),
                "embedding_lr_multiplier_after_step": args.embedding_lr_multiplier_after_step,
                "embedding_lr_multiplier_after": args.embedding_lr_multiplier_after,
                "force_embedding_param_group": bool(args.force_embedding_param_group),
                "reset_scheduler_for_epoch": bool(args.reset_scheduler_for_epoch),
                "epoch_scheduler_steps": args.epoch_scheduler_steps,
                "epoch_warmup_steps": args.epoch_warmup_steps,
                "epoch_start_learning_rate": args.epoch_start_learning_rate,
                "fixed_gene_feature_npy": (
                    str(Path(args.fixed_gene_feature_npy).expanduser().resolve())
                    if args.fixed_gene_feature_npy
                    else None
                ),
                "fixed_gene_feature_scale": float(args.fixed_gene_feature_scale),
                "retrieval_feature_weight": float(args.retrieval_feature_weight),
                "warmup_steps": int(args.warmup_steps),
                "weight_decay": float(args.weight_decay),
                "lr_scheduler_type": str(args.lr_scheduler_type),
                "cosine_min_lr_ratio": float(args.cosine_min_lr_ratio),
                "learning_rate": args.learning_rate,
                "embedding_lr_multiplier": float(args.embedding_lr_multiplier),
                "weight_decay": args.weight_decay,
                "lr_scheduler_type": str(args.lr_scheduler_type),
                "cosine_min_lr_ratio": float(args.cosine_min_lr_ratio),
                "target_tokens": args.target_tokens,
                "max_steps": max_steps,
                "exact_train_coverage_enabled": bool(args.exact_train_coverage),
                "exact_train_coverage": (
                    exact_train_coverage_plan.as_dict()
                    if exact_train_coverage_plan is not None
                    else None
                ),
                "mlm_probability": args.mlm_probability,
                "mask_sampling": str(args.mask_sampling),
                "sampling_stats_npz": str(args.sampling_stats_npz) if args.sampling_stats_npz else None,
                "mask_frequency_exponent": float(args.mask_frequency_exponent),
                "mask_rank_bands": int(args.mask_rank_bands),
                "gene_mlm_policy": str(args.gene_mlm_policy),
                "gene_mlm_mask_replace_prob": float(gene_mlm_mask_replace_prob),
                "gene_mlm_random_replace_prob": float(gene_mlm_random_replace_prob),
                "gene_mlm_unchanged_prob": float(gene_mlm_unchanged_prob),
                "hard_random_replacement": bool(args.hard_random_replacement),
                "cosine_mlm_head": bool(args.cosine_mlm_head),
                "cosine_mlm_temperature": float(args.cosine_mlm_temperature),
                "dual_mask_views": bool(args.dual_mask_views),
                "dataset_balance_exponent": float(args.dataset_balance_exponent),
                "dataset_weight_min": float(args.dataset_weight_min),
                "dataset_weight_max": float(args.dataset_weight_max),
                "init_model_dir": init_model_dir,
                "init_optimizer_state_path": init_optimizer_state_path,
                "init_scheduler_state_path": init_scheduler_state_path,
                "scheduler_total_steps_override": args.scheduler_total_steps_override,
                "stream_eval_masked_acc_only": bool(args.stream_eval_masked_acc_only),
                "token_dictionary": str(args.token_dictionary),
                "eval_max_examples": args.eval_max_examples,
                "use_value_input": bool(args.use_value_input),
                "use_value_loss": bool(args.use_value_loss),
                "fixed_coordinate": bool(args.fixed_coordinate),
                "gene_mlm_enabled": bool(args.gene_mlm_enabled),
                "value_target_decoupled": bool(args.value_target_decoupled),
                "value_target_probability": float(args.value_target_probability),
                "value_target_scope": str(args.value_target_scope),
                "visible_gene_discrimination_enable": bool(args.visible_gene_discrimination_enable),
                "visible_gene_disc_probability": float(args.visible_gene_disc_probability),
                "visible_gene_disc_negative_k": int(args.visible_gene_disc_negative_k),
                "visible_gene_disc_overlap_policy": str(args.visible_gene_disc_overlap_policy),
                "visible_gene_disc_same_row_min_distance": int(args.visible_gene_disc_same_row_min_distance),
                "visible_gene_disc_same_row_max_distance": int(args.visible_gene_disc_same_row_max_distance),
                "visible_gene_disc_support_log_tolerance": float(args.visible_gene_disc_support_log_tolerance),
                "visible_gene_disc_support_log_tolerance_fallback": float(
                    args.visible_gene_disc_support_log_tolerance_fallback
                ),
                "visible_gene_disc_same_row_n": int(args.visible_gene_disc_same_row_n),
                "visible_gene_disc_support_n": int(args.visible_gene_disc_support_n),
                "visible_gene_disc_random_n": int(args.visible_gene_disc_random_n),
                "visible_gene_loss_weight": float(args.visible_gene_loss_weight),
                "view_consistency_enable": bool(args.view_consistency_enable),
                "view_consistency_weight": float(args.view_consistency_weight),
                "view_gene_dropout_prob": float(args.view_gene_dropout_prob),
                "view_value_jitter_prob": float(args.view_value_jitter_prob),
                "view_value_jitter_delta": int(args.view_value_jitter_delta),
                "masked_gene_discrimination_enable": bool(args.masked_gene_discrimination_enable),
                "masked_gene_disc_probability": float(args.masked_gene_disc_probability),
                "masked_gene_loss_weight": float(args.masked_gene_loss_weight),
                "rtd_enable": bool(args.rtd_enable),
                "rtd_loss_weight": float(args.rtd_loss_weight),
                "rtd_visible_positive_policy": str(args.rtd_visible_positive_policy),
                "gene_disc_support_bucket_count": int(args.gene_disc_support_bucket_count),
                "gene_support_npy": (str(gene_support_npy) if gene_support_npy is not None else None),
                "value_loss_weight": float(args.value_loss_weight),
                "program_targets_manifest": (str(program_targets_manifest) if program_targets_manifest is not None else None),
                "program_loss_weight": float(args.program_loss_weight),
                "program_dim": int(args.program_dim),
                "program_huber_delta": float(args.program_huber_delta),
                "program_target_required_for_train": bool(args.program_target_required_for_train),
                "gene_loading_targets_manifest": (
                    str(gene_loading_targets_manifest) if gene_loading_targets_manifest is not None else None
                ),
                "gene_loading_loss_weight": float(args.gene_loading_loss_weight),
                "gene_loading_dim": int(args.gene_loading_dim),
                "gene_loading_huber_delta": float(args.gene_loading_huber_delta),
                "gene_loading_gradient_scope": str(args.gene_loading_gradient_scope),
                "gene_loading_calibration_target_ratio": (
                    float(args.gene_loading_calibration_target_ratio)
                    if args.gene_loading_calibration_target_ratio is not None
                    else None
                ),
                "value_mask_ratio": float(args.value_mask_ratio),
                "value_nonzero_only": bool(args.value_nonzero_only),
                "ordinal_huber_weight": float(args.ordinal_huber_weight),
                "ordinal_huber_delta": float(args.ordinal_huber_delta),
                "view_consistency_enable": bool(args.view_consistency_enable),
                "view_consistency_weight": float(args.view_consistency_weight),
                "view_gene_dropout_prob": float(args.view_gene_dropout_prob),
                "view_value_jitter_prob": float(args.view_value_jitter_prob),
                "view_value_jitter_delta": int(args.view_value_jitter_delta),
                "varcov_enable": bool(args.varcov_enable),
                "varcov_lambda": float(args.varcov_lambda),
                "varcov_layers": list(normalize_varcov_layers(args.varcov_layers, int(args.num_hidden_layers))),
                "varcov_positions": str(args.varcov_positions),
                "input_subset_mode": str(args.input_subset_mode),
                "value_target_mode": str(args.value_target_mode),
                "subset_top_count": int(args.subset_top_count),
                "subset_random_count": int(args.subset_random_count),
                "subset_fill_count": int(args.subset_fill_count),
                "target_nonzero_fraction": float(args.target_nonzero_fraction),
                "evaluation_strategy": evaluation_strategy,
                "eval_steps": int(effective_eval_steps),
                "save_steps": int(effective_save_steps),
                "disable_in_training_eval": bool(args.disable_in_training_eval),
                "seed": training_seed,
                "data_seed": data_seed,
                "hf_training_args_data_seed": hf_data_seed,
                "randorder_seed": int(args.randorder_seed),
                "fixed_length_randorder": bool(args.fixed_length_randorder),
                "n_value_bins": int(args.n_value_bins),
                "value_pad_id": int(args.value_pad_id),
                "value_mask_id": int(args.value_mask_id),
                "value_vocab_size": int(expected_value_vocab_size := (args.value_vocab_size if args.value_vocab_size is not None else max(args.n_value_bins + 2, args.value_mask_id + 1, args.value_pad_id + 1))),
                "paired_mode": paired_mode,
                "paired_metacell_token_index_csv": args.paired_metacell_token_index_csv,
                "paired_mc_loss_weight": float(args.paired_mc_loss_weight),
                "paired_avg_tokens_per_sample": paired_avg_tokens_per_sample,
                "gene_pool_contrastive_enable": bool(args.gene_pool_contrastive_enable),
                "gene_pool_projection_dim": int(args.gene_pool_projection_dim),
                "gene_pool_temperature": float(args.gene_pool_temperature),
                "gene_pool_stopgrad_anchor": bool(args.gene_pool_stopgrad_anchor),
                "gene_pool_lambda_points": [f"{step}:{value}" for step, value in gene_pool_lambda_points],
            },
        )

    training_args_seed_kwargs = {"seed": training_seed}
    if hf_data_seed is not None:
        training_args_seed_kwargs["data_seed"] = hf_data_seed

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=effective_per_device_train_batch,
        per_device_eval_batch_size=max(1, args.per_device_batch // 4),
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy=evaluation_strategy,
        eval_steps=effective_eval_steps,
        save_steps=effective_save_steps,
        save_total_limit=5,
        logging_steps=args.logging_steps,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        dataloader_drop_last=True,
        load_best_model_at_end=False,
        report_to="wandb" if args.use_wandb and is_main else "none",
        run_name=args.wandb_run_name or output_dir.name,
        eval_accumulation_steps=args.eval_accum_steps,
        ddp_find_unused_parameters=bool(args.ddp_find_unused_parameters),
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0 and not args.disable_persistent_workers,
        ignore_data_skip=args.ignore_data_skip,
        remove_unused_columns=False,
        **training_args_seed_kwargs,
    )

    if str(args.input_subset_mode).strip().lower() == "legacy_fixed_length":
        collator = FixedLengthMLMCollator(
            token_dictionary=token_dictionary,
            max_len=args.max_input_size,
            mlm_probability=args.mlm_probability,
            mask_sampling=args.mask_sampling,
            sampling_stats_npz=args.sampling_stats_npz,
            mask_frequency_exponent=args.mask_frequency_exponent,
            mask_rank_bands=args.mask_rank_bands,
            gene_mlm_policy=args.gene_mlm_policy,
            gene_mlm_mask_replace_prob=args.gene_mlm_mask_replace_prob,
            gene_mlm_random_replace_prob=args.gene_mlm_random_replace_prob,
            hard_random_replacement=args.hard_random_replacement,
            dual_mask_views=args.dual_mask_views,
            dataset_balance_exponent=args.dataset_balance_exponent,
            dataset_weight_min=args.dataset_weight_min,
            dataset_weight_max=args.dataset_weight_max,
            gene_mlm_enabled=args.gene_mlm_enabled,
            use_value_input=args.use_value_input,
            use_value_loss=args.use_value_loss,
            value_pad_id=args.value_pad_id,
            value_mask_id=args.value_mask_id,
            value_target_decoupled=args.value_target_decoupled,
            value_target_probability=args.value_target_probability,
            value_target_scope=args.value_target_scope,
            gene_support_npy=gene_support_npy,
            visible_gene_discrimination_enable=args.visible_gene_discrimination_enable,
            visible_gene_disc_probability=args.visible_gene_disc_probability,
            visible_gene_disc_negative_k=args.visible_gene_disc_negative_k,
            visible_gene_disc_overlap_policy=args.visible_gene_disc_overlap_policy,
            visible_gene_disc_same_row_min_distance=args.visible_gene_disc_same_row_min_distance,
            visible_gene_disc_same_row_max_distance=args.visible_gene_disc_same_row_max_distance,
            visible_gene_disc_support_log_tolerance=args.visible_gene_disc_support_log_tolerance,
            visible_gene_disc_support_log_tolerance_fallback=args.visible_gene_disc_support_log_tolerance_fallback,
            visible_gene_disc_same_row_n=args.visible_gene_disc_same_row_n,
            visible_gene_disc_support_n=args.visible_gene_disc_support_n,
            visible_gene_disc_random_n=args.visible_gene_disc_random_n,
            masked_gene_discrimination_enable=args.masked_gene_discrimination_enable,
            masked_gene_disc_probability=args.masked_gene_disc_probability,
            rtd_enable=args.rtd_enable,
            rtd_visible_positive_policy=args.rtd_visible_positive_policy,
            gene_disc_support_bucket_count=args.gene_disc_support_bucket_count,
            value_mask_ratio=args.value_mask_ratio,
            value_nonzero_only=args.value_nonzero_only,
            fixed_coordinate=args.fixed_coordinate,
            fixed_length_randorder=args.fixed_length_randorder,
            randorder_seed=args.randorder_seed,
            paired_mode=paired_mode,
            paired_lookup=paired_lookup,
            program_target_store=program_target_store,
            program_target_required_for_train=bool(args.program_target_required_for_train),
            gene_loading_target_store=gene_loading_target_store,
            gene_pool_contrastive_enabled=args.gene_pool_contrastive_enable,
            view_consistency_enable=args.view_consistency_enable,
            view_gene_dropout_prob=args.view_gene_dropout_prob,
            view_value_jitter_prob=args.view_value_jitter_prob,
            view_value_jitter_delta=args.view_value_jitter_delta,
            domain_token_type_id=args.domain_token_type_id,
            sc_token_type_id=args.sc_token_type_id,
            mc_token_type_id=args.mc_token_type_id,
        )
    else:
        collator = StochasticSetValueCollator(
            token_dictionary=token_dictionary,
            max_len=args.max_input_size,
            value_pad_id=args.value_pad_id,
            value_mask_id=args.value_mask_id,
            value_mask_ratio=args.value_mask_ratio,
            input_subset_mode=args.input_subset_mode,
            value_target_mode=args.value_target_mode,
            subset_top_count=args.subset_top_count,
            subset_random_count=args.subset_random_count,
            subset_fill_count=args.subset_fill_count,
            target_nonzero_fraction=args.target_nonzero_fraction,
            randorder_seed=args.randorder_seed,
            gene_mlm_enabled=args.gene_mlm_enabled,
            use_value_input=args.use_value_input,
            use_value_loss=args.use_value_loss,
            paired_mode=paired_mode,
        )
    if exact_train_coverage_plan is not None:
        collator = ExactCoverageCollator(collator)
    if args.gene_loading_calibration_target_ratio is not None:
        calibration_device = torch.device(
            "cuda",
            local_rank if torch.cuda.is_available() and local_rank >= 0 else 0,
        ) if torch.cuda.is_available() else torch.device("cpu")
        calibration = calibrate_gene_loading_loss_weight(
            model=model,
            collator=collator,
            train_dataset=train_ds,
            batch_size=int(args.per_device_batch),
            target_ratio=float(args.gene_loading_calibration_target_ratio),
            output_dir=output_dir,
            device=calibration_device,
            is_main=is_main,
        )
        args.gene_loading_loss_weight = float(calibration["calibrated_coefficient"])
        model.gene_loading_loss_weight = float(args.gene_loading_loss_weight)
        model.config.gene_loading_loss_weight = float(args.gene_loading_loss_weight)
    trainer = HFMLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=(None if args.stream_eval_masked_acc_only or not args.gene_mlm_enabled else compute_metrics),
        preprocess_logits_for_metrics=(
            None if args.stream_eval_masked_acc_only or not args.gene_mlm_enabled else preprocess_logits_for_metrics
        ),
        boundary_saves=boundary_saves,
        time_boundary_saves=time_boundary_saves,
        stop_after_wall_seconds=args.stop_after_wall_seconds,
        init_optimizer_state_path=init_optimizer_state_path,
        init_scheduler_state_path=init_scheduler_state_path,
        scheduler_total_steps_override=args.scheduler_total_steps_override,
        stream_eval_masked_acc_only=args.stream_eval_masked_acc_only,
        paired_mc_loss_weight=args.paired_mc_loss_weight,
        gene_pool_contrastive_enabled=args.gene_pool_contrastive_enable,
        gene_pool_temperature=args.gene_pool_temperature,
        gene_pool_stopgrad_anchor=args.gene_pool_stopgrad_anchor,
        gene_pool_lambda_points=gene_pool_lambda_points,
        embedding_lr_multiplier=args.embedding_lr_multiplier,
        embedding_lr_multiplier_after_step=args.embedding_lr_multiplier_after_step,
        embedding_lr_multiplier_after=args.embedding_lr_multiplier_after,
        force_embedding_param_group=bool(args.force_embedding_param_group),
        reset_scheduler_for_epoch=bool(args.reset_scheduler_for_epoch),
        epoch_scheduler_steps=args.epoch_scheduler_steps,
        epoch_warmup_steps=args.epoch_warmup_steps,
        epoch_start_learning_rate=args.epoch_start_learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        cosine_min_lr_ratio=args.cosine_min_lr_ratio,
        exact_train_coverage_tracker=exact_train_coverage_tracker,
        training_metric_local_accumulation=bool(args.training_metric_local_accumulation),
    )

    resume_path = resolve_resume_checkpoint(args.resume_from_checkpoint, output_dir)
    trainer.load_exact_train_coverage_state(resume_path)
    if init_model_dir is not None and is_main:
        print(f"[init_model_dir] {init_model_dir}")
    if init_optimizer_state_path is not None and is_main:
        print(f"[init_optimizer_state_path] {init_optimizer_state_path}")
    if init_scheduler_state_path is not None and is_main:
        print(f"[init_scheduler_state_path] {init_scheduler_state_path}")
    if args.scheduler_total_steps_override is not None and is_main:
        print(f"[scheduler_total_steps_override] {int(args.scheduler_total_steps_override)}")
    if paired_mode != "off" and is_main:
        print(f"[paired_mode] {paired_mode}")
        print(f"[paired_metacell_token_index_csv] {Path(args.paired_metacell_token_index_csv).expanduser().resolve()}")
        print(f"[paired_mc_loss_weight] {float(args.paired_mc_loss_weight)}")
        print(f"[paired_avg_tokens_per_sample] {paired_avg_tokens_per_sample:.6f}")
    if args.gene_pool_contrastive_enable and is_main:
        print(f"[gene_pool_contrastive_enable] {bool(args.gene_pool_contrastive_enable)}")
        print(f"[gene_pool_projection_dim] {int(args.gene_pool_projection_dim)}")
        print(f"[gene_pool_temperature] {float(args.gene_pool_temperature)}")
        print(f"[gene_pool_stopgrad_anchor] {bool(args.gene_pool_stopgrad_anchor)}")
        print(f"[gene_pool_lambda_points] {[f'{step}:{value}' for step, value in gene_pool_lambda_points]}")
    if args.view_consistency_enable and is_main:
        print(f"[view_consistency_enable] {bool(args.view_consistency_enable)}")
        print(f"[view_consistency_weight] {float(args.view_consistency_weight)}")
        print(f"[view_gene_dropout_prob] {float(args.view_gene_dropout_prob)}")
        print(f"[view_value_jitter_prob] {float(args.view_value_jitter_prob)}")
        print(f"[view_value_jitter_delta] {int(args.view_value_jitter_delta)}")
    if program_targets_manifest is not None and is_main:
        print(f"[program_targets_manifest] {program_targets_manifest}")
        print(f"[program_loss_weight] {float(args.program_loss_weight)}")
        print(f"[program_dim] {int(args.program_dim)}")
        print(f"[program_huber_delta] {float(args.program_huber_delta)}")
        print(f"[program_target_required_for_train] {bool(args.program_target_required_for_train)}")
    if gene_loading_targets_manifest is not None and is_main:
        print(f"[gene_loading_targets_manifest] {gene_loading_targets_manifest}")
        print(f"[gene_loading_loss_weight] {float(args.gene_loading_loss_weight)}")
        print(f"[gene_loading_dim] {int(args.gene_loading_dim)}")
        print(f"[gene_loading_huber_delta] {float(args.gene_loading_huber_delta)}")
        print(f"[gene_loading_gradient_scope] {str(args.gene_loading_gradient_scope)}")
        print(
            "[gene_loading_calibration_target_ratio] "
            f"{args.gene_loading_calibration_target_ratio}"
        )
    if resume_path is not None and is_main:
        print(f"[resume_from_checkpoint] {resume_path}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    train_metrics = json_safe_metrics(dict(train_result.metrics))
    exact_train_coverage_summary = trainer.summarize_exact_train_coverage()
    cuda_peak_stats = cuda_peak_memory_stats_mb()
    if not args.skip_boundary_exports:
        trainer._maybe_save_exact_boundaries(model=trainer.model, trial=None)
        trainer._maybe_save_time_boundaries(model=trainer.model, trial=None)
    if not args.skip_final_export and dist.is_available() and dist.is_initialized():
        dist.barrier()

    if is_main:
        trainer.save_metrics("train", train_metrics)
        full_model_dir: Path | None = None
        plain_model_dir: Path | None = None
        export_meta = {"exported_plain_mlm": False, "skipped_final_export": bool(args.skip_final_export)}
        if not args.skip_final_export:
            full_model_dir = output_dir / "last_model_aux"
            plain_model_dir = output_dir / "last_model"
            trainer.save_model(str(full_model_dir))

            unwrapped = unwrap_model(trainer.model)
            if isinstance(unwrapped, ValueAugmentedBertForMaskedLM):
                incompatible = unwrapped.export_plain_mlm(plain_model_dir)
                export_meta = {
                    "exported_plain_mlm": bool(incompatible.get("exported_plain_mlm", True)),
                    "skipped_final_export": False,
                    "plain_model_dir": str(plain_model_dir),
                    "full_model_dir": str(full_model_dir),
                    **incompatible,
                }
            else:
                trainer.save_model(str(plain_model_dir))
                export_meta = {
                    "exported_plain_mlm": False,
                    "skipped_final_export": False,
                    "plain_model_dir": str(plain_model_dir),
                    "full_model_dir": str(full_model_dir),
                }

            write_json(output_dir / "plain_export_meta.json", export_meta)
        write_training_done(
            output_dir,
            {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "world_size": int(world_size),
                "max_steps": int(max_steps),
                "max_input_size": int(args.max_input_size),
                "hidden_size": int(args.hidden_size),
                "num_hidden_layers": int(args.num_hidden_layers),
                "num_attention_heads": int(args.num_attention_heads),
                "intermediate_size": int(args.intermediate_size),
                "attn_implementation": str(args.attn_implementation),
                "norm_position": str(args.norm_position),
                "position_embedding_alpha": float(args.position_embedding_alpha),
                "pre_ln_final_layer_norm": bool(args.pre_ln_final_layer_norm),
                "encoder_layout": str(args.encoder_layout),
                "num_loop_steps": (
                    int(args.num_loop_steps) if args.num_loop_steps is not None else int(args.num_hidden_layers)
                ),
                "num_unique_layers": (
                    int(args.num_unique_layers)
                    if args.num_unique_layers is not None
                    else (1 if str(args.encoder_layout).strip().lower() == "looped" else int(args.num_hidden_layers))
                ),
                "use_loop_step_embedding": bool(args.use_loop_step_embedding),
                "domain_token_type_id": (
                    int(args.domain_token_type_id) if args.domain_token_type_id is not None else None
                ),
                "sc_token_type_id": int(args.sc_token_type_id),
                "mc_token_type_id": int(args.mc_token_type_id),
                "per_device_batch": int(args.per_device_batch),
                "grad_accum_steps": int(args.grad_accum_steps),
                "dataloader_item_batch_size": int(effective_per_device_train_batch),
                "exact_train_coverage_enabled": bool(args.exact_train_coverage),
                "exact_train_coverage": exact_train_coverage_summary,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "learning_rate": float(args.learning_rate),
                "embedding_lr_multiplier": float(args.embedding_lr_multiplier),
                "fixed_gene_feature_npy": (
                    str(Path(args.fixed_gene_feature_npy).expanduser().resolve())
                    if args.fixed_gene_feature_npy
                    else None
                ),
                "fixed_gene_feature_scale": float(args.fixed_gene_feature_scale),
                "retrieval_feature_weight": float(args.retrieval_feature_weight),
                "warmup_steps": int(args.warmup_steps),
                "weight_decay": float(args.weight_decay),
                "lr_scheduler_type": str(args.lr_scheduler_type),
                "cosine_min_lr_ratio": float(args.cosine_min_lr_ratio),
                **train_metrics,
                **cuda_peak_stats,
                "eval_dataset_dir": str(args.eval_dataset_dir),
                "eval_max_examples": args.eval_max_examples,
                "init_model_dir": init_model_dir,
                "init_optimizer_state_path": init_optimizer_state_path,
                "init_scheduler_state_path": init_scheduler_state_path,
                "scheduler_total_steps_override": args.scheduler_total_steps_override,
                "evaluation_strategy": evaluation_strategy,
                "eval_steps": int(effective_eval_steps),
                "save_steps": int(effective_save_steps),
                "disable_in_training_eval": bool(args.disable_in_training_eval),
                "stream_eval_masked_acc_only": bool(args.stream_eval_masked_acc_only),
                "boundary_saves": {str(step): name for step, name in boundary_saves.items()},
                "time_boundary_saves": {str(seconds): name for seconds, name in time_boundary_saves.items()},
                "stop_after_wall_seconds": (
                    float(args.stop_after_wall_seconds) if args.stop_after_wall_seconds is not None else None
                ),
                "skip_boundary_exports": bool(args.skip_boundary_exports),
                "skip_final_export": bool(args.skip_final_export),
                "last_model_dir": (str(plain_model_dir) if plain_model_dir is not None else None),
                "last_model_aux_dir": (str(full_model_dir) if full_model_dir is not None else None),
                "use_value_input": bool(args.use_value_input),
                "use_value_loss": bool(args.use_value_loss),
                "value_aux_disabled": not bool(args.use_value_loss),
                "fixed_coordinate": bool(args.fixed_coordinate),
                "gene_mlm_enabled": bool(args.gene_mlm_enabled),
                "mask_sampling": str(args.mask_sampling),
                "sampling_stats_npz": (
                    str(Path(args.sampling_stats_npz).expanduser().resolve()) if args.sampling_stats_npz else None
                ),
                "mask_frequency_exponent": float(args.mask_frequency_exponent),
                "mask_rank_bands": int(args.mask_rank_bands),
                "gene_mlm_policy": str(args.gene_mlm_policy),
                "gene_mlm_mask_replace_prob": float(gene_mlm_mask_replace_prob),
                "gene_mlm_random_replace_prob": float(gene_mlm_random_replace_prob),
                "gene_mlm_unchanged_prob": float(gene_mlm_unchanged_prob),
                "hard_random_replacement": bool(args.hard_random_replacement),
                "cosine_mlm_head": bool(args.cosine_mlm_head),
                "cosine_mlm_temperature": float(args.cosine_mlm_temperature),
                "dual_mask_views": bool(args.dual_mask_views),
                "dataset_balance_exponent": float(args.dataset_balance_exponent),
                "dataset_weight_min": float(args.dataset_weight_min),
                "dataset_weight_max": float(args.dataset_weight_max),
                "value_target_decoupled": bool(args.value_target_decoupled),
                "value_target_probability": float(args.value_target_probability),
                "value_target_scope": str(args.value_target_scope),
                "value_loss_weight": float(args.value_loss_weight),
                "rtd_enable": bool(args.rtd_enable),
                "rtd_loss_weight": float(args.rtd_loss_weight),
                "rtd_visible_positive_policy": str(args.rtd_visible_positive_policy),
                "program_targets_manifest": (str(program_targets_manifest) if program_targets_manifest is not None else None),
                "program_loss_weight": float(args.program_loss_weight),
                "program_dim": int(args.program_dim),
                "program_huber_delta": float(args.program_huber_delta),
                "program_target_required_for_train": bool(args.program_target_required_for_train),
                "value_mask_ratio": float(args.value_mask_ratio),
                "value_nonzero_only": bool(args.value_nonzero_only),
                "ordinal_huber_weight": float(args.ordinal_huber_weight),
                "ordinal_huber_delta": float(args.ordinal_huber_delta),
                "varcov_enable": bool(args.varcov_enable),
                "varcov_lambda": float(args.varcov_lambda),
                "varcov_layers": list(normalize_varcov_layers(args.varcov_layers, int(args.num_hidden_layers))),
                "varcov_positions": str(args.varcov_positions),
                "input_subset_mode": str(args.input_subset_mode),
                "value_target_mode": str(args.value_target_mode),
                "subset_top_count": int(args.subset_top_count),
                "subset_random_count": int(args.subset_random_count),
                "subset_fill_count": int(args.subset_fill_count),
                "target_nonzero_fraction": float(args.target_nonzero_fraction),
                "seed": training_seed,
                "data_seed": data_seed,
                "hf_training_args_data_seed": hf_data_seed,
                "randorder_seed": int(args.randorder_seed),
                "fixed_length_randorder": bool(args.fixed_length_randorder),
                "n_value_bins": int(args.n_value_bins),
                "value_pad_id": int(args.value_pad_id),
                "value_mask_id": int(args.value_mask_id),
                "value_vocab_size": (
                    int(args.value_vocab_size)
                    if args.value_vocab_size is not None
                    else int(max(args.n_value_bins + 2, args.value_mask_id + 1, args.value_pad_id + 1))
                ),
                "paired_mode": paired_mode,
                "paired_metacell_token_index_csv": (
                    str(Path(args.paired_metacell_token_index_csv).expanduser().resolve())
                    if args.paired_metacell_token_index_csv
                    else None
                ),
                "paired_mc_loss_weight": float(args.paired_mc_loss_weight),
                "paired_avg_tokens_per_sample": paired_avg_tokens_per_sample,
                "gene_pool_contrastive_enable": bool(args.gene_pool_contrastive_enable),
                "gene_pool_projection_dim": int(args.gene_pool_projection_dim),
                "gene_pool_temperature": float(args.gene_pool_temperature),
                "gene_pool_stopgrad_anchor": bool(args.gene_pool_stopgrad_anchor),
                "gene_pool_lambda_points": [f"{step}:{value}" for step, value in gene_pool_lambda_points],
                "train_optimizer_steps": float(trainer.state.global_step),
                "train_seen_examples": float(trainer._seen_examples),
                "train_seen_raw_cells": float(trainer._seen_raw_cells),
                "train_seen_non_pad_tokens": float(trainer._seen_non_pad_tokens),
                "train_seen_sc_non_pad_tokens": float(trainer._seen_sc_non_pad_tokens),
                "train_seen_mc_non_pad_tokens": float(trainer._seen_mc_non_pad_tokens),
            },
        )
        if args.use_wandb:
            import wandb

            wandb.finish()
    if args.skip_final_export:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
