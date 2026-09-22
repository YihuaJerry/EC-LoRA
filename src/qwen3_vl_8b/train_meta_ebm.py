#!/usr/bin/env python3
from __future__ import annotations

import _bootstrap  # noqa: F401 - configure the grouped source layout

import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from checkpointing import (
    capture_rng_state,
    load_task_checkpoint,
    make_out_prefix,
    restore_rng_state,
    save_task_checkpoint,
    task_checkpoint_path,
)
from cl_logger import CLLogger
from output_guard import OutputDirGuard
from qwenvl_adapter_data import (
    CheckpointSpec,
    FullAdapterDataset,
    build_adapter_template,
    discover_task_checkpoints,
    first_checkpoint,
    parse_task_order,
)
from models.meta_ebm import MetaEBM
from task_conditioning import build_task_condition_embeddings

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


ALLOWED_QWENVL_TASKS = {
    "scienceqa",
    "vizwiz",
    "imagenet",
    "vqav2",
    "iconqa",
    "flickr30k",
    "grounding",
    "ocr_vqa",
}
UNSEEN_BENCHMARK_TASKS = (
    "aokvqa",
    "imagenet_r",
    "screen2words",
    "tabmwp",
)
UNSEEN_ADAPTER_TASKS = {
    "aokvqa": "iconqa",
    "imagenet_r": "imagenet",
    "screen2words": "flickr30k",
    "tabmwp": "ocr_vqa",
}


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


class ReservoirMemory:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        self.num_seen = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, x: torch.Tensor, cond: torch.Tensor, task_name: str) -> None:
        for row in x.detach().cpu():
            self.num_seen += 1
            item = (row.clone(), cond.detach().cpu().view(-1).clone(), task_name)
            if len(self.samples) < self.capacity:
                self.samples.append(item)
                continue
            replace_idx = random.randint(0, self.num_seen - 1)
            if replace_idx < self.capacity:
                self.samples[replace_idx] = item

    def sample(
        self,
        k: int,
        device: torch.device,
        query: Optional[torch.Tensor] = None,
        sampling_mode: str = "random",
        importance_topk: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        if not self.samples or k <= 0:
            return None
        k = min(int(k), len(self.samples))

        if sampling_mode == "cosine_topk" and query is not None:
            q = query.detach().cpu().float().view(1, -1)
            scored = []
            for idx, (x, _, _) in enumerate(self.samples):
                score = F.cosine_similarity(q, x.float().view(1, -1), dim=1).item()
                scored.append((score, idx))
            scored.sort(key=lambda item: item[0], reverse=True)
            topk = int(importance_topk) if int(importance_topk) > 0 else min(len(scored), max(k, 4 * k))
            topk = max(k, min(topk, len(scored)))
            pool = [idx for _, idx in scored[:topk]]
            chosen = random.sample(pool, k)
        else:
            chosen = random.sample(range(len(self.samples)), k)

        xs, conds, names = [], [], []
        for idx in chosen:
            x, cond, name = self.samples[idx]
            xs.append(x)
            conds.append(cond)
            names.append(name)
        return torch.stack(xs).to(device), torch.stack(conds).to(device), names

    def state_dict(self) -> Dict:
        return {
            "capacity": self.capacity,
            "samples": self.samples,
            "num_seen": self.num_seen,
        }

    def load_state_dict(self, state: Optional[Dict]) -> None:
        if not state:
            return
        self.capacity = int(state.get("capacity", self.capacity))
        self.samples = list(state.get("samples", []))
        self.num_seen = int(state.get("num_seen", len(self.samples)))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Dict:
    if not path:
        return {}
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required for YAML configs.")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-EBM on Qwen3-VL PEFT LoRA adapter checkpoints")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--checkpoint_root", type=str, default="")
    parser.add_argument(
        "--task_order",
        type=str,
        default="scienceqa,vizwiz,imagenet,vqav2,iconqa,flickr30k,grounding,ocr_vqa",
    )
    parser.add_argument("--task_limit", type=int, default=0)
    parser.add_argument("--topk_checkpoints_per_task", type=int, default=10)
    parser.add_argument("--normalize_mode", type=str, default="per_tensor", choices=["per_tensor", "none"])
    parser.add_argument("--normalization_eps", type=float, default=1e-6)
    parser.add_argument("--include_non_lora", "--include-non-lora", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--adapter_key_regex", type=str, default="")
    parser.add_argument("--pooled_cache_dir", type=str, default="")
    parser.add_argument("--disable_pooled_cache", action="store_true")
    parser.add_argument("--task_condition_mode", type=str, default="dense_hidden", choices=["dense_hidden", "one_hot"])
    parser.add_argument("--task_condition_model_path", type=str, default="")
    parser.add_argument("--task_condition_cache_dir", type=str, default="")
    parser.add_argument("--task_condition_pooling", type=str, default="mean", choices=["mean", "last"])
    parser.add_argument("--task_condition_max_length", type=int, default=256)
    parser.add_argument("--task_condition_precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--task_condition_normalize", type=str2bool, default=True)
    parser.add_argument("--task_condition_trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--task_condition_device", type=str, default="")

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs_per_task", type=int, default=100)
    parser.add_argument("--max_steps_per_task", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--pool_length", type=int, default=2048)
    parser.add_argument("--vector_hidden_dim", type=int, default=384)
    parser.add_argument("--cond_hidden_dim", type=int, default=256)
    parser.add_argument("--score_matching_mode", type=str, default="hutchinson", choices=["denoising", "hutchinson"])
    parser.add_argument("--dsm_noise_std", type=float, default=0.1)
    parser.add_argument("--hutchinson_samples", type=int, default=1)
    parser.add_argument("--trace_weight", type=float, default=0.2)
    parser.add_argument("--score_loss_scale", type=float, default=1.0)

    parser.add_argument("--memories", type=int, default=160)
    parser.add_argument("--replay_batch_size", type=int, default=8)
    parser.add_argument("--replay_sampling_mode", type=str, default="random", choices=["random", "cosine_topk"])
    parser.add_argument("--replay_importance_topk", type=int, default=0)
    parser.add_argument("--replay_loss_weight", type=float, default=1.0)
    parser.add_argument("--align_weight", type=float, default=1e-4)
    parser.add_argument("--align_min_grad_norm", type=float, default=1e-8)
    parser.add_argument("--align_target_cosine", type=float, default=0.2)
    parser.add_argument("--comp_weight", type=float, default=0.0)
    parser.add_argument("--comp_batch_pairs", type=int, default=2)

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="meta_ebm_qwenvl")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--disable_task_checkpoint", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--auto_eval_after_train", type=str2bool, default=True)
    parser.add_argument("--auto_eval_on_dry_run", type=str2bool, default=False)
    parser.add_argument("--eval_after_each_task", type=str2bool, default=False)
    parser.add_argument("--eval_after_each_task_current_only", type=str2bool, default=False)
    parser.add_argument("--eval_after_each_task_seen_only", type=str2bool, default=True)
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument("--eval_benchmark_tasks", "--eval-benchmark-tasks", type=str, default="seen_so_far_plus_unseen")
    parser.add_argument(
        "--eval_pairing_mode",
        "--eval-pairing-mode",
        type=str,
        default="seen_one_to_one_unseen_mapped",
        choices=["cartesian", "seen_one_to_one_unseen_current", "seen_one_to_one_unseen_mapped"],
    )
    parser.add_argument("--eval_output_dir", type=str, default="")
    parser.add_argument("--eval_run_name", type=str, default="")
    parser.add_argument("--eval_lora_project_dir", type=str, default="")
    parser.add_argument("--eval_lora_config_dir", type=str, default="")
    parser.add_argument("--eval_source_index", type=int, default=0)
    parser.add_argument("--eval_refine_steps", type=int, default=5)
    parser.add_argument("--eval_refine_lr", type=float, default=1e-4)
    parser.add_argument("--eval_refine_beta", type=float, default=1.0)
    parser.add_argument("--eval_refine_grad_clip", type=float, default=0.0)
    parser.add_argument("--eval_langevin_std", type=float, default=0.0)
    parser.add_argument("--eval_num_samples", type=int, default=1)
    parser.add_argument("--eval_init_noise_std", type=float, default=0.0)
    parser.add_argument("--eval_save_safetensors", type=str2bool, default=False)
    parser.add_argument("--eval_skip_model_eval", type=str2bool, default=False)
    parser.add_argument("--eval_split", choices=("validation", "test"), default="validation")
    parser.add_argument("--eval_dataset_source", choices=("auto", "official", "local"), default="local")
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--eval_precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--eval_per_device_batch_size", type=int, default=8)
    parser.add_argument("--eval_dataloader_num_workers", type=int, default=0)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_cuda_device", type=str, default=None)
    parser.add_argument("--eval_python_bin", type=str, default="")
    parser.add_argument("--eval_attn_impl", type=str, default="flash_attention_2", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--eval_qwen_device", type=str, default="cuda")
    parser.add_argument("--eval_num_chunks", type=int, default=1)
    parser.add_argument("--eval_trust_remote_code", type=str2bool, default=False)
    parser.add_argument("--eval_fail_fast", type=str2bool, default=False)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    known, _ = config_parser.parse_known_args()
    if known.config:
        cfg = load_yaml_config(known.config)
        parser.set_defaults(**cfg)
    args = parser.parse_args()
    if args.config:
        args.config = os.path.abspath(os.path.expanduser(args.config))
    return args


def maybe_apply_dry_run(args: argparse.Namespace) -> None:
    if not args.dry_run:
        return
    args.task_limit = min(args.task_limit or 2, 2)
    args.topk_checkpoints_per_task = min(args.topk_checkpoints_per_task, 2)
    args.num_epochs_per_task = 1
    args.max_steps_per_task = 2
    args.batch_size = min(args.batch_size, 1)
    args.replay_batch_size = min(args.replay_batch_size, 2)
    print("[dry_run] lightweight smoke settings enabled")


def validate_qwenvl_tasks(tasks: Sequence[str]) -> None:
    invalid = [task for task in tasks if task not in ALLOWED_QWENVL_TASKS]
    if invalid:
        raise ValueError(
            "Meta_EBM_QwenVL received non-QwenVL tasks "
            f"{invalid}. Expected only: {', '.join(sorted(ALLOWED_QWENVL_TASKS))}. "
            "This usually means the Qwen text/GLUE config was used by mistake; "
            "check configs/meta_ebm_qwenvl.yaml, --task_order, and --checkpoint_root."
        )


def score_matching_loss(
    model: MetaEBM,
    x_vec: torch.Tensor,
    cond: torch.Tensor,
    mode: str = "hutchinson",
    dsm_noise_std: float = 0.1,
    hutchinson_samples: int = 1,
    trace_weight: float = 0.2,
    loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mode = str(mode).lower()
    scale = float(loss_scale)
    if mode == "denoising":
        clean = x_vec.detach()
        sigma = max(float(dsm_noise_std), 1e-8)
        noise = torch.randn_like(clean) * sigma
        x = (clean + noise).detach().requires_grad_(True)
        energy = model.energy(x, cond, already_pooled=False)
        grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
        target_grad = noise / (sigma * sigma)
        per_sample = 0.5 * (grad - target_grad).pow(2).flatten(1).mean(dim=1)
        raw_loss = per_sample.mean()
        loss = raw_loss * scale
        stats = {
            "energy": float(energy.mean().detach().item()),
            "raw_loss": float(raw_loss.detach().item()),
            "loss_scale": scale,
            "grad_sq": float(grad.pow(2).flatten(1).mean(dim=1).mean().detach().item()),
            "target_grad_sq": float(target_grad.pow(2).flatten(1).mean(dim=1).mean().detach().item()),
            "dsm_noise_std": float(sigma),
        }
        return loss, stats

    if mode != "hutchinson":
        raise ValueError(f"Unsupported score_matching_mode: {mode}")

    x = x_vec.detach().requires_grad_(True)
    energy = model.energy(x, cond, already_pooled=False)
    grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]

    grad_sq = 0.5 * grad.pow(2).flatten(1).mean(dim=1)
    trace_est = torch.zeros_like(grad_sq)
    num_probes = max(1, int(hutchinson_samples))
    for _ in range(num_probes):
        eps = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        hvp = torch.autograd.grad((grad * eps).sum(), x, create_graph=True)[0]
        trace_est = trace_est + (hvp * eps).flatten(1).mean(dim=1)
    trace_est = trace_est / float(num_probes)

    raw_loss = (grad_sq - float(trace_weight) * trace_est).mean()
    loss = raw_loss * scale
    stats = {
        "energy": float(energy.mean().detach().item()),
        "raw_loss": float(raw_loss.detach().item()),
        "loss_scale": scale,
        "grad_sq": float(grad_sq.mean().detach().item()),
        "trace": float(trace_est.mean().detach().item()),
    }
    return loss, stats


def _flatten_param_grads(
    params: List[torch.nn.Parameter],
    grads: Tuple[Optional[torch.Tensor], ...],
) -> torch.Tensor:
    chunks = []
    for param, grad in zip(params, grads):
        if grad is None:
            chunks.append(torch.zeros_like(param).reshape(-1))
        else:
            chunks.append(grad.reshape(-1))
    if not chunks:
        raise ValueError("No trainable parameters found for gradient alignment.")
    return torch.cat(chunks)


def gradient_alignment_loss(
    model: MetaEBM,
    loss_cur: torch.Tensor,
    loss_mem: torch.Tensor,
    min_grad_norm: float = 1e-8,
    target_cosine: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    params = [param for param in model.parameters() if param.requires_grad]
    grads_cur = torch.autograd.grad(loss_cur, params, retain_graph=True, create_graph=True, allow_unused=True)
    grads_mem = torch.autograd.grad(loss_mem, params, retain_graph=True, create_graph=True, allow_unused=True)
    flat_cur = _flatten_param_grads(params, grads_cur)
    flat_mem = _flatten_param_grads(params, grads_mem)
    norm_cur = flat_cur.norm()
    norm_mem = flat_mem.norm()
    if float(norm_cur.detach()) < min_grad_norm or float(norm_mem.detach()) < min_grad_norm:
        zero = loss_cur.new_zeros(())
        return zero, {
            "align_cosine": 0.0,
            "align_penalty": 0.0,
            "align_skipped": 1.0,
            "grad_norm_cur": float(norm_cur.detach()),
            "grad_norm_mem": float(norm_mem.detach()),
        }
    cosine = torch.dot(flat_cur, flat_mem) / (norm_cur * norm_mem + 1e-12)
    penalty = F.relu(float(target_cosine) - cosine).pow(2)
    return penalty, {
        "align_cosine": float(cosine.detach()),
        "align_penalty": float(penalty.detach()),
        "align_skipped": 0.0,
        "grad_norm_cur": float(norm_cur.detach()),
        "grad_norm_mem": float(norm_mem.detach()),
    }


def _input_score_pooled(model: MetaEBM, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().requires_grad_(True)
    energy = model.energy(x_req, cond, already_pooled=False)
    return torch.autograd.grad(energy.sum(), x_req, create_graph=True)[0]


def composition_consistency_loss(
    model: MetaEBM,
    x1: torch.Tensor,
    c1: torch.Tensor,
    x2: torch.Tensor,
    c2: torch.Tensor,
    max_pairs: int = 2,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    num_pairs = min(int(max_pairs), x1.size(0), x2.size(0))
    if num_pairs <= 0:
        raise ValueError("composition_consistency_loss requires at least one valid pair.")
    x1 = x1[:num_pairs]
    x2 = x2[:num_pairs]
    c1 = c1[:num_pairs]
    c2 = c2[:num_pairs]
    fused_x = 0.5 * (x1 + x2)
    fused_cond = F.normalize(c1 + c2, p=2, dim=1, eps=eps)
    score_fused = _input_score_pooled(model, fused_x, fused_cond)
    score_avg = 0.5 * (_input_score_pooled(model, x1, c1) + _input_score_pooled(model, x2, c2))
    per_sample_l2 = (score_fused - score_avg).pow(2).flatten(1).mean(dim=1)
    loss = per_sample_l2.mean()
    return loss, {
        "comp_pairs": float(num_pairs),
        "score_diff_rms": float(per_sample_l2.detach().mean().sqrt().item()),
    }


def compute_task_mean(dataset: FullAdapterDataset) -> torch.Tensor:
    accumulator = None
    for idx in range(len(dataset)):
        vec = dataset[idx].float()
        accumulator = vec if accumulator is None else accumulator + vec
    if accumulator is None:
        raise ValueError("Cannot compute task prototype from an empty dataset.")
    return accumulator / float(len(dataset))


def jsonable_template(template: Dict) -> Dict:
    out = {}
    for key, value in template.items():
        if isinstance(value, dict):
            out[key] = {str(k): list(v) if isinstance(v, tuple) else v for k, v in value.items()}
        else:
            out[key] = value
    return out


def build_auto_eval_args(
    args: argparse.Namespace,
    final_checkpoint: str,
    tasks: Sequence[str],
    project_root: str,
    checkpoint_root: str,
    run_dir: str,
    eval_tasks: Optional[Sequence[str]] = None,
    benchmark_tasks: Optional[Sequence[str]] = None,
    run_name: Optional[str] = None,
    pairing_mode: Optional[str] = None,
    current_task: str = "",
) -> argparse.Namespace:
    if eval_tasks is not None:
        eval_task_text = ",".join(eval_tasks)
    else:
        eval_task_text = args.eval_tasks or ",".join(tasks)
    if benchmark_tasks is not None:
        benchmark_task_text = ",".join(benchmark_tasks)
    else:
        benchmark_task_text = args.eval_benchmark_tasks or "all"
        if str(benchmark_task_text).strip().lower() == "seen_so_far_plus_unseen":
            benchmark_task_text = "all"
    return argparse.Namespace(
        config="",
        ebm_checkpoint=final_checkpoint,
        tasks=eval_task_text,
        benchmark_tasks=benchmark_task_text,
        pairing_mode=pairing_mode or args.eval_pairing_mode,
        current_task=current_task,
        checkpoint_root=checkpoint_root,
        lora_project_dir=args.eval_lora_project_dir or os.path.join(project_root, "LoRA_Qwen_VL"),
        lora_config_dir=args.eval_lora_config_dir or os.path.join(project_root, "LoRA_Qwen_VL", "config"),
        output_dir=args.eval_output_dir or os.path.join(run_dir, "eval"),
        run_name=run_name or args.eval_run_name or "after_train",
        source_index=args.eval_source_index,
        device=args.device,
        refine_steps=args.eval_refine_steps,
        refine_lr=args.eval_refine_lr,
        refine_beta=args.eval_refine_beta,
        refine_grad_clip=args.eval_refine_grad_clip,
        langevin_std=args.eval_langevin_std,
        num_samples=args.eval_num_samples,
        init_noise_std=args.eval_init_noise_std,
        save_safetensors=args.eval_save_safetensors,
        skip_eval=args.eval_skip_model_eval,
        eval_split=args.eval_split,
        dataset_source=args.eval_dataset_source,
        model_path=args.eval_model_path,
        precision=args.eval_precision,
        per_device_batch_size=args.eval_per_device_batch_size,
        dataloader_num_workers=args.eval_dataloader_num_workers,
        max_samples=args.eval_max_samples,
        cuda_device=args.eval_cuda_device,
        python_bin=args.eval_python_bin,
        attn_impl=args.eval_attn_impl,
        qwen_device=args.eval_qwen_device,
        num_chunks=args.eval_num_chunks,
        model_base=args.eval_model_path,
        trust_remote_code=args.eval_trust_remote_code,
        fail_fast=args.eval_fail_fast,
        parent_result_root=run_dir,
    )


def select_after_task_eval_tasks(args: argparse.Namespace, tasks: Sequence[str], task_idx: int) -> List[str]:
    if args.eval_tasks:
        requested = parse_task_order(args.eval_tasks)
    elif args.eval_after_each_task_current_only:
        requested = [tasks[task_idx]]
    else:
        requested = list(tasks)
    if not args.eval_after_each_task_seen_only:
        return requested
    seen = set(tasks[: task_idx + 1])
    return [task for task in requested if task in seen]


def select_after_task_benchmark_tasks(args: argparse.Namespace, tasks: Sequence[str], task_idx: int) -> Optional[List[str]]:
    benchmark_text = str(args.eval_benchmark_tasks or "").strip().lower()
    progressive_aliases = {"", "all", "seen_so_far_plus_unseen", "seen+unseen", "progressive"}
    if args.eval_pairing_mode not in {"seen_one_to_one_unseen_current", "seen_one_to_one_unseen_mapped"} or benchmark_text not in progressive_aliases:
        return None
    ordered: List[str] = []
    for task in list(tasks[: task_idx + 1]) + list(UNSEEN_BENCHMARK_TASKS):
        if task not in ordered:
            ordered.append(task)
    return ordered


def select_after_task_adapter_tasks(
    args: argparse.Namespace,
    tasks: Sequence[str],
    task_idx: int,
    benchmark_tasks: Optional[Sequence[str]],
) -> List[str]:
    selected = select_after_task_eval_tasks(args, tasks, task_idx)
    if args.eval_pairing_mode != "seen_one_to_one_unseen_mapped":
        return selected
    for eval_task in benchmark_tasks or UNSEEN_BENCHMARK_TASKS:
        mapped_task = UNSEEN_ADAPTER_TASKS.get(eval_task)
        if mapped_task in tasks and mapped_task not in selected:
            selected.append(mapped_task)
    return selected


def train(args: argparse.Namespace) -> str:
    maybe_apply_dry_run(args)
    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(args.project_root or os.path.join(script_dir, ".."))
    checkpoint_root = os.path.abspath(
        os.path.expanduser(
            args.checkpoint_root
            or os.environ.get("CHECKPOINT_ROOT", "")
            or os.path.join(project_root, "CMA_ES", "QwenVL", "checkpoints")
        )
    )
    save_dir = os.path.abspath(args.save_dir or os.path.join(script_dir, "results"))
    log_dir = os.path.abspath(args.log_dir or os.path.join(script_dir, "logs"))
    pooled_cache_dir = ""
    if not args.disable_pooled_cache:
        pooled_cache_dir = os.path.abspath(args.pooled_cache_dir or os.path.join(script_dir, ".cache", "pooled"))

    tasks = parse_task_order(args.task_order)
    if args.task_limit and args.task_limit > 0:
        tasks = tasks[: int(args.task_limit)]
    validate_qwenvl_tasks(tasks)
    print(f"[info] tasks={tasks}")
    print(f"[info] checkpoint_root={checkpoint_root}")

    task_to_specs = discover_task_checkpoints(
        tasks,
        checkpoint_root=checkpoint_root,
        topk_per_task=args.topk_checkpoints_per_task,
    )
    reference_checkpoint = first_checkpoint(task_to_specs)
    template = build_adapter_template(
        reference_checkpoint,
        include_non_lora=bool(args.include_non_lora),
        key_regex=args.adapter_key_regex,
    )
    print(f"[info] adapter_dim={template['input_dim']} pooled_dim={args.pool_length} keys={len(template['keys'])}")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    resume_payload = None
    if args.resume_from:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_from))
        print(f"[resume] loading {resume_path}")
        resume_payload = load_task_checkpoint(resume_path)
    if resume_payload is not None and resume_payload.get("task_embeddings"):
        task_embeddings = {
            key: value.detach().float().cpu() for key, value in resume_payload.get("task_embeddings", {}).items()
        }
        task_condition_metadata = dict(resume_payload.get("task_condition_metadata", {"mode": "resume"}))
    else:
        task_embeddings, task_condition_metadata = build_task_condition_embeddings(
            args=args,
            tasks=tasks,
            project_root=project_root,
            script_dir=script_dir,
            default_lora_config_name="LoRA_Qwen_VL",
        )
    condition_dim = int(next(iter(task_embeddings.values())).numel())
    print(f"[info] task_condition={task_condition_metadata.get('mode')} dim={condition_dim}")

    out_prefix = make_out_prefix(save_dir, args.exp_name, resume_payload=resume_payload)
    run_dir = os.path.dirname(out_prefix)
    os.makedirs(run_dir, exist_ok=True)
    result_guard = OutputDirGuard(run_dir, "training result directory", initialize=True)
    result_guard.assert_alive()
    with open(os.path.join(run_dir, "config_resolved.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "adapter_template.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable_template(template), handle, ensure_ascii=False, indent=2)

    logger = CLLogger(log_dir, args.exp_name, comp_weight=args.comp_weight)
    model = MetaEBM(
        input_dim=int(template["input_dim"]),
        task_embed_dim=condition_dim,
        pool_length=args.pool_length,
        hidden_dim=args.vector_hidden_dim,
        task_hidden_dim=args.cond_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    memory = ReservoirMemory(args.memories)
    task_prototypes: Dict[str, torch.Tensor] = {}
    start_task_idx = 0

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        if resume_payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        memory.load_state_dict(resume_payload.get("memory_state"))
        task_prototypes = {
            key: value.float().cpu() for key, value in resume_payload.get("task_prototypes", {}).items()
        }
        start_task_idx = int(resume_payload.get("next_task_idx", 0))
        restore_rng_state(resume_payload.get("rng_state"))

    global_step = int(resume_payload.get("global_step", 0)) if resume_payload is not None else 0
    spent_start = time.time()
    after_task_eval_summaries: List[Dict[str, str]] = []

    for task_idx, task_name in enumerate(tasks[start_task_idx:], start=start_task_idx):
        result_guard.assert_alive()
        specs = task_to_specs[task_name]
        dataset = FullAdapterDataset(
            checkpoint_specs=specs,
            template=template,
            normalize_mode=args.normalize_mode,
            eps=args.normalization_eps,
        )
        if task_name not in task_prototypes:
            task_prototypes[task_name] = compute_task_mean(dataset).cpu()

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        cond = task_embeddings[task_name].to(device).view(1, -1)
        print(
            f"[task {task_idx + 1}/{len(tasks)}] {task_name}: "
            f"{len(specs)} checkpoints, {len(loader)} batches/epoch"
        )
        model.train()
        pbar = range(args.num_epochs_per_task)
        if not args.disable_tqdm:
            pbar = tqdm(pbar, desc=f"train:{task_name}", dynamic_ncols=True)

        for epoch in pbar:
            result_guard.assert_alive()
            n_batches = 0
            epoch_total = epoch_task = epoch_replay = epoch_align = epoch_comp = 0.0
            epoch_align_cos = epoch_align_skip = epoch_grad_cur = epoch_grad_mem = 0.0
            epoch_comp_pairs = epoch_comp_rms = 0.0
            align_count = comp_count = 0
            for step, x_cpu in enumerate(loader):
                result_guard.assert_alive()
                if args.max_steps_per_task > 0 and step >= args.max_steps_per_task:
                    break
                x = x_cpu.float().to(device, non_blocking=device.type == "cuda")
                cond_batch = cond.repeat(x.size(0), 1)

                optimizer.zero_grad(set_to_none=True)
                loss_cur, _ = score_matching_loss(
                    model,
                    x,
                    cond_batch,
                    mode=args.score_matching_mode,
                    dsm_noise_std=args.dsm_noise_std,
                    hutchinson_samples=args.hutchinson_samples,
                    trace_weight=args.trace_weight,
                    loss_scale=args.score_loss_scale,
                )
                total_loss = loss_cur
                replay_loss_scalar = align_loss_scalar = comp_loss_scalar = 0.0
                align_cos = align_skip = grad_cur = grad_mem = 0.0
                comp_pairs = comp_rms = 0.0

                replay = None
                if args.replay_batch_size > 0 and len(memory) > 0:
                    replay = memory.sample(
                        args.replay_batch_size,
                        device=device,
                        query=x.detach().mean(dim=0).cpu(),
                        sampling_mode=args.replay_sampling_mode,
                        importance_topk=args.replay_importance_topk,
                    )
                if replay is not None:
                    mem_x, mem_cond, _ = replay
                    loss_mem, _ = score_matching_loss(
                        model,
                        mem_x,
                        mem_cond,
                        mode=args.score_matching_mode,
                        dsm_noise_std=args.dsm_noise_std,
                        hutchinson_samples=args.hutchinson_samples,
                        trace_weight=args.trace_weight,
                        loss_scale=args.score_loss_scale,
                    )
                    replay_loss_scalar = float(loss_mem.detach())
                    total_loss = total_loss + args.replay_loss_weight * loss_mem
                    if args.align_weight > 0:
                        loss_align, align_stats = gradient_alignment_loss(
                            model,
                            loss_cur,
                            loss_mem,
                            min_grad_norm=args.align_min_grad_norm,
                            target_cosine=args.align_target_cosine,
                        )
                        align_loss_scalar = float(loss_align.detach())
                        align_cos = float(align_stats["align_cosine"])
                        align_skip = float(align_stats["align_skipped"])
                        grad_cur = float(align_stats["grad_norm_cur"])
                        grad_mem = float(align_stats["grad_norm_mem"])
                        total_loss = total_loss + args.align_weight * loss_align
                        epoch_align_cos += align_cos
                        epoch_align_skip += align_skip
                        epoch_grad_cur += grad_cur
                        epoch_grad_mem += grad_mem
                        align_count += 1
                    if args.comp_weight > 0 and args.comp_batch_pairs > 0:
                        loss_comp, comp_stats = composition_consistency_loss(
                            model,
                            x,
                            cond_batch,
                            mem_x,
                            mem_cond,
                            max_pairs=args.comp_batch_pairs,
                        )
                        comp_loss_scalar = float(loss_comp.detach())
                        comp_pairs = float(comp_stats["comp_pairs"])
                        comp_rms = float(comp_stats["score_diff_rms"])
                        total_loss = total_loss + args.comp_weight * loss_comp
                        epoch_comp_pairs += comp_pairs
                        epoch_comp_rms += comp_rms
                        comp_count += 1

                if not torch.isfinite(total_loss):
                    raise RuntimeError(
                        f"Non-finite loss at task={task_name} epoch={epoch + 1} step={step + 1}. "
                        "Try lowering learning_rate or trace_weight."
                    )
                total_loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                memory.add(x.detach(), cond.view(-1).detach(), task_name)

                total_scalar = float(total_loss.detach())
                cur_scalar = float(loss_cur.detach())
                epoch_total += total_scalar
                epoch_task += cur_scalar
                epoch_replay += replay_loss_scalar
                epoch_align += align_loss_scalar
                epoch_comp += comp_loss_scalar
                n_batches += 1
                global_step += 1

                logger.log_step(
                    task_name,
                    epoch + 1,
                    step + 1,
                    total_scalar,
                    cur_scalar,
                    replay_loss_scalar,
                    align_loss_scalar,
                    align_cos,
                    align_skip,
                    grad_cur,
                    grad_mem,
                    comp_loss_scalar,
                    comp_pairs,
                    comp_rms,
                )

                if args.log_every > 0 and global_step % args.log_every == 0:
                    logger.save_loss_plot()

            if n_batches == 0:
                continue
            avg_total = epoch_total / n_batches
            avg_task = epoch_task / n_batches
            avg_replay = epoch_replay / n_batches
            avg_align = epoch_align / n_batches
            avg_comp = epoch_comp / n_batches
            avg_align_cos = epoch_align_cos / max(1, align_count)
            avg_align_skip = epoch_align_skip / max(1, align_count)
            avg_grad_cur = epoch_grad_cur / max(1, align_count)
            avg_grad_mem = epoch_grad_mem / max(1, align_count)
            avg_comp_pairs = epoch_comp_pairs / max(1, comp_count)
            avg_comp_rms = epoch_comp_rms / max(1, comp_count)
            if not args.disable_tqdm and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{avg_total:.4f}",
                    task=f"{avg_task:.4f}",
                    replay=f"{avg_replay:.4f}",
                    align=f"{avg_align:.4f}",
                    comp=f"{avg_comp:.4f}",
                    memory=len(memory),
                )
            print(
                f"[epoch] task={task_name} epoch={epoch + 1} "
                f"loss={avg_total:.6f} task={avg_task:.6f} replay={avg_replay:.6f} "
                f"align={avg_align:.6f} comp={avg_comp:.6f} memory={len(memory)} "
                f"align_cos={avg_align_cos:.6f} "
                f"align_skip={avg_align_skip:.6f} "
                f"grad_cur={avg_grad_cur:.6f} "
                f"grad_mem={avg_grad_mem:.6f} "
                f"comp_pairs={avg_comp_pairs:.6f} "
                f"comp_rms={avg_comp_rms:.6f}"
            )
            logger.log_epoch(
                {
                    "task": task_name,
                    "task_index": task_idx,
                    "epoch": epoch + 1,
                    "num_batches": n_batches,
                    "total_loss": avg_total,
                    "task_loss": avg_task,
                    "replay_loss": avg_replay,
                    "align_loss": avg_align,
                    "align_cosine": avg_align_cos,
                    "align_skipped": avg_align_skip,
                    "grad_norm_cur": avg_grad_cur,
                    "grad_norm_mem": avg_grad_mem,
                    "comp_loss": avg_comp,
                    "weighted_comp_loss": args.comp_weight * avg_comp,
                    "comp_pairs": avg_comp_pairs,
                    "comp_score_rms": avg_comp_rms,
                    "memory": len(memory),
                    "progress_loss": avg_total,
                    "progress_task": avg_task,
                    "progress_replay": avg_replay,
                    "progress_align": avg_align,
                    "progress_comp": avg_comp,
                    "progress_memory": len(memory),
                }
            )

        logger.mark_task_boundary(task_name)
        result_guard.assert_alive()
        logger.save_loss_plot()
        task_payload = {
            "out_prefix": out_prefix,
            "global_step": global_step,
            "next_task_idx": task_idx + 1,
            "tasks": tasks,
            "template": template,
            "normalize_mode": args.normalize_mode,
            "normalization_eps": args.normalization_eps,
            "model_config": {
                "input_dim": int(template["input_dim"]),
                "task_embed_dim": condition_dim,
                "pool_length": int(args.pool_length),
                "hidden_dim": int(args.vector_hidden_dim),
                "task_hidden_dim": int(args.cond_hidden_dim),
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "memory_state": memory.state_dict(),
            "task_embeddings": task_embeddings,
            "task_condition_metadata": task_condition_metadata,
            "task_prototypes": task_prototypes,
            "rng_state": capture_rng_state(),
            "checkpoint_root": checkpoint_root,
            "task_to_checkpoints": {
                task: [spec.__dict__ for spec in specs] for task, specs in task_to_specs.items()
            },
            "args": vars(args),
        }
        ckpt_path = task_checkpoint_path(out_prefix, task_idx, task_name)
        result_guard.assert_alive()
        if not args.disable_task_checkpoint:
            save_task_checkpoint(ckpt_path, task_payload)
            print(f"[checkpoint] {ckpt_path}")

        if args.eval_after_each_task and (not args.dry_run or args.auto_eval_on_dry_run):
            result_guard.assert_alive()
            if args.disable_task_checkpoint:
                save_task_checkpoint(ckpt_path, task_payload)
                print(f"[checkpoint] saved for after-task eval: {ckpt_path}")
            eval_benchmark_tasks_for_stage = select_after_task_benchmark_tasks(args, tasks, task_idx)
            eval_tasks_for_stage = select_after_task_adapter_tasks(
                args,
                tasks,
                task_idx,
                eval_benchmark_tasks_for_stage,
            )
            if eval_tasks_for_stage:
                print(
                    f"[auto-eval] task {task_idx + 1}/{len(tasks)} finished; "
                    f"adapter_tasks={eval_tasks_for_stage} "
                    f"benchmark_tasks={eval_benchmark_tasks_for_stage or args.eval_benchmark_tasks}"
                )
                model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                from eval_meta_ebm import run_evaluation

                eval_args = build_auto_eval_args(
                    args=args,
                    final_checkpoint=ckpt_path,
                    tasks=tasks,
                    project_root=project_root,
                    checkpoint_root=checkpoint_root,
                    run_dir=run_dir,
                    eval_tasks=eval_tasks_for_stage,
                    benchmark_tasks=eval_benchmark_tasks_for_stage,
                    run_name=f"after_task_{task_idx + 1:02d}_{task_name}",
                    pairing_mode=args.eval_pairing_mode,
                    current_task=task_name,
                )
                eval_summary = run_evaluation(eval_args)
                result_guard.assert_alive()
                after_task_eval_summaries.append(eval_summary)
                model.to(device)
                model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"[auto-eval] no tasks selected after task={task_name}; skipped")
        elif args.eval_after_each_task and args.dry_run:
            print("[auto-eval] after-task eval skipped during dry_run; set auto_eval_on_dry_run=true to enable it")

    final_path = f"{out_prefix}_final.pt"
    result_guard.assert_alive()
    save_task_checkpoint(
        final_path,
        {
            "out_prefix": out_prefix,
            "global_step": global_step,
            "next_task_idx": len(tasks),
            "tasks": tasks,
            "template": template,
            "normalize_mode": args.normalize_mode,
            "normalization_eps": args.normalization_eps,
            "model_config": {
                "input_dim": int(template["input_dim"]),
                "task_embed_dim": condition_dim,
                "pool_length": int(args.pool_length),
                "hidden_dim": int(args.vector_hidden_dim),
                "task_hidden_dim": int(args.cond_hidden_dim),
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "memory_state": memory.state_dict(),
            "task_embeddings": task_embeddings,
            "task_condition_metadata": task_condition_metadata,
            "task_prototypes": task_prototypes,
            "rng_state": capture_rng_state(),
            "checkpoint_root": checkpoint_root,
            "task_to_checkpoints": {task: [spec.__dict__ for spec in specs] for task, specs in task_to_specs.items()},
            "args": vars(args),
        },
    )
    summary = {
        "final_checkpoint": final_path,
        "run_dir": run_dir,
        "loss_plot": logger.plot_path,
        "train_log": logger.log_path,
        "train_metrics_jsonl": logger.metrics_path,
        "elapsed_seconds": time.time() - spent_start,
        "tasks": tasks,
        "adapter_dim": int(template["input_dim"]),
        "pool_length": int(args.pool_length),
        "normalize_mode": args.normalize_mode,
        "task_condition_metadata": task_condition_metadata,
        "after_task_eval_summaries": after_task_eval_summaries,
    }
    result_guard.assert_alive()
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.auto_eval_after_train and args.eval_after_each_task:
        print("[auto-eval] final after_train eval skipped because eval_after_each_task already evaluates the final task")
    elif args.auto_eval_after_train and (not args.dry_run or args.auto_eval_on_dry_run):
        print("[auto-eval] training finished, running Meta-EBM refinement + QwenVL generation evaluation")
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from eval_meta_ebm import run_evaluation

        eval_args = build_auto_eval_args(
            args=args,
            final_checkpoint=final_path,
            tasks=tasks,
            project_root=project_root,
            checkpoint_root=checkpoint_root,
            run_dir=run_dir,
        )
        eval_summary = run_evaluation(eval_args)
        result_guard.assert_alive()
        summary["auto_eval"] = eval_summary
        result_guard.assert_alive()
        with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"[auto-eval] summary_json={eval_summary['summary_json']}")
    elif args.auto_eval_after_train and args.dry_run:
        print("[auto-eval] skipped during dry_run; set auto_eval_on_dry_run=true to enable it")
    print(f"[done] final_checkpoint={final_path}")
    print(f"[done] loss_plot={logger.plot_path}")
    return final_path


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
