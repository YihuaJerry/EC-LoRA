#!/usr/bin/env python3
import _bootstrap  # noqa: F401 - configure the grouped source layout
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from checkpointing import (
    capture_rng_state,
    load_task_checkpoint,
    make_out_prefix,
    restore_rng_state,
    save_task_checkpoint,
    task_checkpoint_path,
)
from cl_logger import CLLogger
from data_pipeline import (
    NormalizedFileDataset,
    compute_task_prototype,
    list_task_files,
    load_task_vectors,
    pad_vector,
    validate_input_dim,
)
from eval_meta_ebm import MultiTaskEBMEvaluator, dump_eval_details
from metrics import confusion_matrix
from models.meta_ebm import MetaEBM
from task_setup import (
    DEFAULT_TASK_ORDER,
    default_log_dir,
    default_normalized_root,
    default_project_root,
    default_save_dir,
    default_task_vectors_dir,
    parse_task_order,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReservoirMemory:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[torch.Tensor] = []
        self.tasks: List[int] = []
        self.age = 0

    def __len__(self):
        return len(self.data)

    def add(self, x_cpu: torch.Tensor, task_id: int):
        if self.capacity <= 0:
            return
        self.age += 1
        x = x_cpu.detach().clone()
        if len(self.data) < self.capacity:
            self.data.append(x)
            self.tasks.append(int(task_id))
            return

        idx = random.randint(0, self.age - 1)
        if idx < self.capacity:
            self.data[idx] = x
            self.tasks[idx] = int(task_id)

    def add_batch(self, x_batch_cpu: torch.Tensor, task_id: int):
        for idx in range(x_batch_cpu.size(0)):
            self.add(x_batch_cpu[idx], task_id)

    def sample(
        self,
        batch_size: int,
        max_task_id: Optional[int] = None,
        query: Optional[torch.Tensor] = None,
        sampling_mode: str = "random",
        importance_topk: int = 0,
        eps: float = 1e-12,
    ) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
        if len(self.data) == 0:
            return None, None
        candidate_idxs = list(range(len(self.data)))
        if max_task_id is not None:
            candidate_idxs = [idx for idx, task_id in enumerate(self.tasks) if int(task_id) <= int(max_task_id)]
        if len(candidate_idxs) == 0:
            return None, None

        k = min(int(batch_size), len(candidate_idxs))
        if sampling_mode == "cosine_topk" and query is not None:
            query_vec = query.detach().float().cpu().view(-1)
            query_norm = float(query_vec.norm(p=2).item())
            if query_norm > 0.0:
                scored_candidates = []
                for idx in candidate_idxs:
                    x_i = self.data[idx].float().view(-1)
                    denom = float(x_i.norm(p=2).item()) * query_norm + float(eps)
                    sim = float(torch.dot(x_i, query_vec).item()) / denom
                    scored_candidates.append((sim, idx))
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                topk = int(importance_topk) if int(importance_topk) > 0 else min(len(scored_candidates), max(k, 4 * k))
                topk = max(k, min(topk, len(scored_candidates)))
                candidate_pool = [idx for _, idx in scored_candidates[:topk]]
                idxs = random.sample(candidate_pool, k) if len(candidate_pool) > k else candidate_pool[:k]
            else:
                idxs = random.sample(candidate_idxs, k)
        else:
            idxs = random.sample(candidate_idxs, k)
        xs = torch.stack([self.data[i] for i in idxs], dim=0)
        ts = [self.tasks[i] for i in idxs]
        return xs, ts

    def state_dict(self) -> Dict:
        return {
            "capacity": self.capacity,
            "age": self.age,
            "tasks": [int(t) for t in self.tasks],
            "data": [x.detach().clone() for x in self.data],
        }

    def load_state_dict(self, state: Optional[Dict]):
        if not state:
            return
        self.capacity = int(state.get("capacity", self.capacity))
        self.age = int(state.get("age", 0))
        self.tasks = [int(t) for t in state.get("tasks", [])]
        self.data = [x.detach().clone().cpu() for x in state.get("data", [])]

def parse_args():
    parser = argparse.ArgumentParser(description="Meta-EBM continual learning on normalized task vectors")
    parser.add_argument("--config", type=str, default="", help="Path to a YAML config file.")
    parser.add_argument("--project_root", type=str, default=default_project_root(__file__))
    parser.add_argument("--task_order", type=str, default=",".join(DEFAULT_TASK_ORDER))
    parser.add_argument("--task_limit", type=int, default=0)
    parser.add_argument("--normalized_root", type=str, default="")
    parser.add_argument("--normalized_all_dir", type=str, default="")
    parser.add_argument("--topk_checkpoints_per_task", type=int, default=10)
    parser.add_argument("--task_vectors_dir", type=str, default="")

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs_per_task", type=int, default=100)
    parser.add_argument("--max_steps_per_task", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument("--input_dim", type=int, default=-1)
    parser.add_argument("--condition_dim", type=int, default=-1)
    parser.add_argument("--pool_length", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--task_hidden_dim", type=int, default=128)
    parser.add_argument("--hutchinson_samples", type=int, default=1)
    parser.add_argument("--vpfb_trace_weight", type=float, default=1.0)

    parser.add_argument("--memories", type=int, default=160)
    parser.add_argument("--replay_batch_size", type=int, default=8)
    parser.add_argument("--replay_sampling_mode", type=str, default="random", choices=["random", "cosine_topk"])
    parser.add_argument("--replay_importance_topk", type=int, default=0)
    parser.add_argument("--replay_loss_weight", type=float, default=1.0)
    parser.add_argument("--align_weight", type=float, default=0.0)
    parser.add_argument("--align_min_grad_norm", type=float, default=1e-8)
    parser.add_argument("--align_target_cosine", type=float, default=0.2)
    parser.add_argument("--task_lr_warmup_steps", type=int, default=0)
    parser.add_argument("--task_lr_decay", type=str, default="none", choices=["none", "linear", "cosine"])
    parser.add_argument("--task_lr_min_scale", type=float, default=0.1)
    parser.add_argument("--comp_weight", type=float, default=0.0)
    parser.add_argument("--comp_batch_pairs", type=int, default=2)

    parser.add_argument("--eval_num_samples", type=int, default=1)
    parser.add_argument("--eval_aggregate", type=str, default="best", choices=["best", "mean"])
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--eval_refine_steps", type=int, default=5)
    parser.add_argument("--eval_refine_lr", type=float, default=1e-2)
    parser.add_argument("--eval_refine_beta", type=float, default=1.0)
    parser.add_argument("--eval_init_noise_std", type=float, default=0.0)
    parser.add_argument("--eval_refine_grad_clip", type=float, default=0.0)
    parser.add_argument("--eval_refine_langevin_std", type=float, default=0.0)
    parser.add_argument("--seen_init_mode", type=str, default="prototype", choices=["prototype", "zero"])
    parser.add_argument("--unseen_init_mode", type=str, default="zero", choices=["prototype", "zero"])
    parser.add_argument("--skip_initial_eval", action="store_true")

    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="meta_ebm")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--disable_task_checkpoint", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return _parse_with_yaml(parser)


def _load_yaml_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a dict: {path}")
    return data


def _parse_with_yaml(parser: argparse.ArgumentParser):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.config:
        cfg_path = os.path.abspath(pre_args.config)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
        cfg = _load_yaml_config(cfg_path)
        known_keys = {action.dest for action in parser._actions}
        unknown_keys = [key for key in cfg.keys() if key not in known_keys]
        if unknown_keys:
            raise ValueError("Unknown config keys: " + ", ".join(sorted(unknown_keys)))
        parser.set_defaults(**cfg)

    return parser.parse_args()


def _maybe_apply_dry_run(args):
    if not args.dry_run:
        return
    if args.task_limit <= 0:
        args.task_limit = 2
    if args.max_steps_per_task <= 0:
        args.max_steps_per_task = 1
    if args.eval_max_batches <= 0:
        args.eval_max_batches = 1
    args.num_epochs_per_task = 1
    args.eval_num_samples = 1
    args.eval_refine_steps = min(args.eval_refine_steps, 1)
    args.batch_size = min(args.batch_size, 2)
    args.replay_batch_size = min(args.replay_batch_size, 2)
    print("[dry_run] Running a lightweight smoke configuration.")


def score_matching_loss(
    model: MetaEBM,
    x: torch.Tensor,
    cond: torch.Tensor,
    hutchinson_samples: int = 1,
    trace_weight: float = 1.0,
):
    x = x.detach().requires_grad_(True)
    energy = model(x, cond)
    grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]

    grad_sq = 0.5 * grad.pow(2).flatten(1).mean(dim=1)
    trace_est = torch.zeros_like(grad_sq)
    num_probes = max(1, int(hutchinson_samples))
    for _ in range(num_probes):
        eps = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        hvp = torch.autograd.grad((grad * eps).sum(), x, create_graph=True)[0]
        trace_est = trace_est + (hvp * eps).flatten(1).mean(dim=1)
    trace_est = trace_est / float(num_probes)

    loss = (grad_sq - float(trace_weight) * trace_est).mean()
    stats = {
        "energy": float(energy.mean().detach().item()),
        "grad_sq": float(grad_sq.mean().detach().item()),
        "trace": float(trace_est.mean().detach().item()),
        "trace_weight": float(trace_weight),
    }
    return loss, stats


def _flatten_param_grads(
    params: List[torch.nn.Parameter],
    grads: Tuple[Optional[torch.Tensor], ...],
) -> torch.Tensor:
    pieces = []
    for param, grad in zip(params, grads):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    if not pieces:
        raise ValueError("No trainable parameters found for gradient alignment.")
    return torch.cat(pieces, dim=0)


def compute_task_lr(
    base_lr: float,
    step_idx: int,
    total_steps: int,
    warmup_steps: int = 0,
    decay: str = "none",
    min_scale: float = 0.1,
) -> float:
    step_idx = max(0, int(step_idx))
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    min_scale = float(min_scale)

    if warmup_steps > 0 and step_idx < warmup_steps:
        return float(base_lr) * float(step_idx + 1) / float(max(1, warmup_steps))

    if decay == "none":
        return float(base_lr)

    decay_start = min(warmup_steps, total_steps - 1)
    decay_steps = max(1, total_steps - decay_start)
    progress = min(1.0, max(0.0, float(step_idx - decay_start) / float(decay_steps)))
    if decay == "linear":
        scale = min_scale + (1.0 - min_scale) * (1.0 - progress)
    elif decay == "cosine":
        scale = min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        scale = 1.0
    return float(base_lr) * float(scale)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


def gradient_alignment_loss(
    model: MetaEBM,
    loss_cur: torch.Tensor,
    loss_mem: torch.Tensor,
    min_grad_norm: float = 1e-8,
    target_cosine: float = 0.2,
    eps: float = 1e-12,
):
    params = [param for param in model.parameters() if param.requires_grad]
    grads_cur = torch.autograd.grad(
        loss_cur,
        params,
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )
    grads_mem = torch.autograd.grad(
        loss_mem,
        params,
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )

    flat_cur = _flatten_param_grads(params, grads_cur)
    flat_mem = _flatten_param_grads(params, grads_mem)
    norm_cur = flat_cur.norm(p=2)
    norm_mem = flat_mem.norm(p=2)
    if norm_cur.detach().item() < float(min_grad_norm) or norm_mem.detach().item() < float(min_grad_norm):
        zero = loss_cur.new_zeros(())
        stats = {
            "align_cosine": 0.0,
            "align_penalty": 0.0,
            "align_skipped": 1.0,
            "grad_norm_cur": float(norm_cur.detach().item()),
            "grad_norm_mem": float(norm_mem.detach().item()),
        }
        return zero, stats

    cur_unit = flat_cur / norm_cur.detach().clamp_min(eps)
    mem_unit = flat_mem / norm_mem.detach().clamp_min(eps)
    cosine = torch.dot(cur_unit, mem_unit)
    if not torch.isfinite(cosine):
        zero = loss_cur.new_zeros(())
        stats = {
            "align_cosine": 0.0,
            "align_penalty": 0.0,
            "align_skipped": 1.0,
            "grad_norm_cur": float(norm_cur.detach().item()),
            "grad_norm_mem": float(norm_mem.detach().item()),
        }
        return zero, stats

    gap = torch.relu(float(target_cosine) - cosine)
    penalty = gap.pow(2)
    stats = {
        "align_cosine": float(cosine.detach().item()),
        "align_penalty": float(penalty.detach().item()),
        "align_skipped": 0.0,
        "grad_norm_cur": float(norm_cur.detach().item()),
        "grad_norm_mem": float(norm_mem.detach().item()),
    }
    return penalty, stats


def _input_score(model: MetaEBM, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    energy = model(x, cond)
    return torch.autograd.grad(energy.sum(), x, create_graph=True)[0]


def composition_consistency_loss(
    model: MetaEBM,
    x_cur: torch.Tensor,
    c_cur: torch.Tensor,
    x_mem: torch.Tensor,
    c_mem: torch.Tensor,
    max_pairs: int = 1,
    eps: float = 1e-12,
):
    num_pairs = min(int(max_pairs), x_cur.size(0), x_mem.size(0))
    if num_pairs <= 0:
        raise ValueError("composition_consistency_loss requires at least one valid pair.")

    x1 = x_cur[:num_pairs]
    c1 = c_cur[:num_pairs]
    x2 = x_mem[:num_pairs]
    c2 = c_mem[:num_pairs]
    fused_x = 0.5 * (x1 + x2)
    fused_cond = F.normalize(c1 + c2, p=2, dim=1, eps=eps)
    score_1 = _input_score(model, x1, c1)
    score_2 = _input_score(model, x2, c2)
    score_fused = _input_score(model, fused_x, fused_cond)
    diff = score_fused - 0.5 * (score_1 + score_2)
    diff_flat = diff.flatten(1)
    per_sample_l2 = diff_flat.norm(p=2, dim=1)
    per_sample_rms = diff_flat.pow(2).mean(dim=1).sqrt()
    loss = diff_flat.pow(2).mean(dim=1).mean()
    stats = {
        "comp_pairs": float(num_pairs),
        "score_diff_rms": float(per_sample_rms.mean().detach().item()),
        "score_diff_l2": float(per_sample_l2.mean().detach().item()),
    }
    return loss, stats


def train(args):
    _maybe_apply_dry_run(args)
    set_seed(args.seed)

    project_root = os.path.abspath(args.project_root)
    task_order = parse_task_order(args.task_order)
    if args.task_limit > 0:
        task_order = task_order[: args.task_limit]
    if not task_order:
        raise ValueError("task_order is empty.")

    resume_payload = None
    if args.resume_from:
        resume_path = os.path.abspath(args.resume_from)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"resume_from does not exist: {resume_path}")
        resume_payload = load_task_checkpoint(resume_path)
        print(f"[resume] loading checkpoint: {resume_path}")

    normalized_root = os.path.abspath(args.normalized_root) if args.normalized_root else default_normalized_root(project_root)
    normalized_all_dir = os.path.abspath(args.normalized_all_dir) if args.normalized_all_dir else ""
    task_vectors_dir = os.path.abspath(args.task_vectors_dir) if args.task_vectors_dir else default_task_vectors_dir(project_root)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else default_save_dir(project_root)
    log_dir = os.path.abspath(args.log_dir) if args.log_dir else default_log_dir(project_root)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    method_name = args.exp_name or "meta_ebm"
    logger = CLLogger(log_dir, method_name, comp_weight=args.comp_weight)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[info] project_root={project_root}")
    print(f"[info] tasks={task_order}")
    print(f"[info] device={device}")
    if device.type == "cuda":
        gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"[info] gpu_index={gpu_idx} gpu_name={torch.cuda.get_device_name(gpu_idx)}")

    task_to_files = list_task_files(
        tasks=task_order,
        normalized_root=normalized_root,
        normalized_all_dir=normalized_all_dir,
        topk_per_task=args.topk_checkpoints_per_task,
    )
    input_dim, per_task_dim = validate_input_dim(task_to_files)
    print(f"[info] input_dim={input_dim}")
    print(f"[info] per_task_dim={per_task_dim}")
    print(f"[info] topk_checkpoints_per_task={args.topk_checkpoints_per_task}")
    files_per_task = {task: len(files) for task, files in task_to_files.items()}
    print(f"[info] files_per_task={files_per_task}")
    print(
        "[info] sharpen_settings="
        f"trace_weight:{args.vpfb_trace_weight}, "
        f"eval_refine_beta:{args.eval_refine_beta}"
    )
    print(
        "[info] strategy_settings="
        f"replay_sampling:{args.replay_sampling_mode}, "
        f"lr_decay:{args.task_lr_decay}, "
        f"langevin_std:{args.eval_refine_langevin_std}"
    )

    task_vectors = load_task_vectors(task_vectors_dir=task_vectors_dir, tasks=task_order)
    cond_dim = int(next(iter(task_vectors.values())).numel())

    if args.input_dim > 0 and args.input_dim != input_dim:
        raise ValueError(f"input_dim mismatch: args={args.input_dim} vs data={input_dim}")
    if args.condition_dim > 0 and args.condition_dim != cond_dim:
        raise ValueError(f"condition_dim mismatch: args={args.condition_dim} vs data={cond_dim}")

    model = MetaEBM(
        input_dim=input_dim,
        task_embed_dim=cond_dim,
        pool_length=args.pool_length,
        hidden_dim=args.hidden_dim,
        task_hidden_dim=args.task_hidden_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    memory = ReservoirMemory(args.memories)

    evaluator = MultiTaskEBMEvaluator(
        project_root=project_root,
        tasks=task_order,
        task_to_files=task_to_files,
        model_input_dim=input_dim,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers,
        eval_num_samples=args.eval_num_samples,
        eval_aggregate=args.eval_aggregate,
        eval_max_batches=args.eval_max_batches,
        eval_refine_steps=args.eval_refine_steps,
        eval_refine_lr=args.eval_refine_lr,
        eval_refine_beta=args.eval_refine_beta,
        eval_init_noise_std=args.eval_init_noise_std,
        eval_refine_grad_clip=args.eval_refine_grad_clip,
        eval_refine_langevin_std=args.eval_refine_langevin_std,
        seen_init_mode=args.seen_init_mode,
        unseen_init_mode=args.unseen_init_mode,
        device=args.device,
        show_progress=not args.disable_tqdm,
    )

    result_t: List[int] = []
    result_a: List[List[float]] = []
    eval_records: List[dict] = []
    task_prototypes: Dict[str, torch.Tensor] = {}
    start_task_idx = 0
    spent_before = 0.0
    out_prefix = make_out_prefix(save_dir, method_name, resume_payload)
    time_start = time.time()

    if resume_payload is not None:
        ckpt_task_order = resume_payload.get("task_order")
        if ckpt_task_order is not None and list(ckpt_task_order) != list(task_order):
            raise ValueError("resume checkpoint task_order does not match current config.")
        model.load_state_dict(resume_payload["model_state_dict"])
        if resume_payload.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            except ValueError as exc:
                print(f"[resume] optimizer state skipped due to mismatch: {exc}")
        memory.load_state_dict(resume_payload.get("memory_state"))
        memory.data = [pad_vector(x, input_dim).cpu() for x in memory.data]
        result_t = [int(x) for x in resume_payload.get("result_t", [])]
        result_a = [[float(v) for v in row] for row in resume_payload.get("result_a", [])]
        eval_records = list(resume_payload.get("eval_records", []))
        saved_task_prototypes = resume_payload.get("task_prototypes", {})
        task_prototypes = {task: pad_vector(tensor, input_dim).cpu() for task, tensor in saved_task_prototypes.items()}
        start_task_idx = int(resume_payload.get("next_task_idx", 0))
        spent_before = float(resume_payload.get("spent_time_sec", 0.0))
        restore_rng_state(resume_payload.get("rng_state"))
        print(f"[resume] start_task_idx={start_task_idx} out_prefix={out_prefix}")

    if start_task_idx == 0 and len(result_t) == 0 and not args.skip_initial_eval:
        print("[eval] running initial zero/prototype evaluation")
        baseline_accs, baseline_detail = evaluator.evaluate_all(
            ebm_model=model,
            task_vectors=task_vectors,
            task_prototypes=task_prototypes,
            max_seen_task_idx=-1,
        )
        result_t.append(0)
        result_a.append(baseline_accs)
        eval_records.append(
            {
                "stage": "initial",
                "task_idx": 0,
                "accs": baseline_accs,
                "detail": baseline_detail,
            }
        )
        print("[eval] baseline accs:", [round(x, 4) for x in baseline_accs])
        logger.log_baseline_eval(baseline_accs, task_order)

    remaining_task_indices = list(range(start_task_idx, len(task_order)))
    task_iter = tqdm(remaining_task_indices, desc="train-tasks", leave=True, dynamic_ncols=True, disable=args.disable_tqdm)
    for task_idx in task_iter:
        task_name = task_order[task_idx]
        task_iter.set_postfix(task=task_name, memory=len(memory))
        print(f"[train] starting task {task_idx + 1}/{len(task_order)}: {task_name}")

        if task_name not in task_prototypes:
            task_prototypes[task_name] = compute_task_prototype(task_to_files[task_name], target_dim=input_dim).cpu()

        dataset = NormalizedFileDataset(task_to_files[task_name], target_dim=input_dim)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        total_task_steps = len(loader) * max(1, int(args.num_epochs_per_task))
        if args.max_steps_per_task > 0:
            total_task_steps = min(total_task_steps, int(args.max_steps_per_task))
        total_task_steps = max(1, total_task_steps)

        task_steps = 0
        epoch_iter = range(1, args.num_epochs_per_task + 1)
        if not args.disable_tqdm and args.num_epochs_per_task > 1:
            epoch_iter = tqdm(epoch_iter, desc=f"epochs:{task_name}", leave=False, dynamic_ncols=True)

        for epoch in epoch_iter:
            model.train()
            epoch_total = 0.0
            epoch_task = 0.0
            epoch_replay = 0.0
            epoch_align = 0.0
            epoch_align_cosine = 0.0
            epoch_align_skipped = 0.0
            epoch_grad_norm_cur = 0.0
            epoch_grad_norm_mem = 0.0
            epoch_align_count = 0
            epoch_comp = 0.0
            epoch_comp_pairs = 0.0
            epoch_comp_score_rms = 0.0
            epoch_comp_count = 0
            n_batches = 0

            batch_iter = loader
            if not args.disable_tqdm:
                batch_iter = tqdm(loader, desc=f"{task_name}-e{epoch}", leave=False, dynamic_ncols=True)

            for step, x in enumerate(batch_iter, start=1):
                x = x.to(device, non_blocking=True)
                c_cur = task_vectors[task_name].to(device, non_blocking=True).view(1, -1).repeat(x.size(0), 1)
                current_lr = compute_task_lr(
                    base_lr=args.learning_rate,
                    step_idx=task_steps,
                    total_steps=total_task_steps,
                    warmup_steps=args.task_lr_warmup_steps,
                    decay=args.task_lr_decay,
                    min_scale=args.task_lr_min_scale,
                )
                set_optimizer_lr(optimizer, current_lr)

                optimizer.zero_grad(set_to_none=True)
                loss_cur, _ = score_matching_loss(
                    model,
                    x,
                    c_cur,
                    hutchinson_samples=args.hutchinson_samples,
                    trace_weight=args.vpfb_trace_weight,
                )
                total_loss = loss_cur
                replay_loss_scalar = 0.0
                align_loss_scalar = 0.0
                comp_loss_scalar = 0.0
                align_cosine_scalar = 0.0
                align_skipped_scalar = 0.0
                grad_norm_cur_scalar = 0.0
                grad_norm_mem_scalar = 0.0
                comp_pairs_scalar = 0.0
                comp_score_rms_scalar = 0.0

                if args.replay_batch_size > 0 and len(memory) > 0:
                    replay_query = x.detach().mean(dim=0).cpu()
                    x_mem, t_mem = memory.sample(
                        args.replay_batch_size,
                        max_task_id=task_idx - 1,
                        query=replay_query,
                        sampling_mode=args.replay_sampling_mode,
                        importance_topk=args.replay_importance_topk,
                    )
                    if x_mem is not None and t_mem is not None:
                        x_mem = x_mem.to(device, non_blocking=True)
                        c_mem = torch.stack([task_vectors[task_order[t]] for t in t_mem], dim=0).to(device, non_blocking=True)
                        loss_mem, _ = score_matching_loss(
                            model,
                            x_mem,
                            c_mem,
                            hutchinson_samples=args.hutchinson_samples,
                            trace_weight=args.vpfb_trace_weight,
                        )
                        replay_loss_scalar = float(loss_mem.item())
                        total_loss = total_loss + args.replay_loss_weight * loss_mem

                        if args.align_weight > 0:
                            loss_align, align_stats = gradient_alignment_loss(
                                model,
                                loss_cur,
                                loss_mem,
                                min_grad_norm=args.align_min_grad_norm,
                                target_cosine=args.align_target_cosine,
                            )
                            align_loss_scalar = float(loss_align.item())
                            align_cosine_scalar = float(align_stats["align_cosine"])
                            align_skipped_scalar = float(align_stats["align_skipped"])
                            grad_norm_cur_scalar = float(align_stats["grad_norm_cur"])
                            grad_norm_mem_scalar = float(align_stats["grad_norm_mem"])
                            epoch_align_cosine += align_cosine_scalar
                            epoch_align_skipped += align_skipped_scalar
                            epoch_grad_norm_cur += grad_norm_cur_scalar
                            epoch_grad_norm_mem += grad_norm_mem_scalar
                            epoch_align_count += 1
                            total_loss = total_loss + args.align_weight * loss_align

                        if args.comp_weight > 0 and args.comp_batch_pairs > 0:
                            loss_comp, comp_stats = composition_consistency_loss(
                                model=model,
                                x_cur=x,
                                c_cur=c_cur,
                                x_mem=x_mem,
                                c_mem=c_mem,
                                max_pairs=args.comp_batch_pairs,
                            )
                            comp_loss_scalar = float(loss_comp.item())
                            comp_pairs_scalar = float(comp_stats["comp_pairs"])
                            comp_score_rms_scalar = float(comp_stats["score_diff_rms"])
                            epoch_comp_pairs += comp_pairs_scalar
                            epoch_comp_score_rms += comp_score_rms_scalar
                            epoch_comp_count += 1
                            total_loss = total_loss + args.comp_weight * loss_comp

                total_loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                memory.add_batch(x.detach().cpu(), task_idx)
                n_batches += 1
                task_steps += 1
                epoch_total += float(total_loss.item())
                epoch_task += float(loss_cur.item())
                epoch_replay += replay_loss_scalar
                epoch_align += align_loss_scalar
                epoch_comp += comp_loss_scalar

                if args.log_every > 0 and task_steps % args.log_every == 0:
                    avg_total = epoch_total / n_batches
                    avg_task = epoch_task / n_batches
                    avg_replay = epoch_replay / max(1, n_batches)
                    avg_align = epoch_align / max(1, n_batches)
                    avg_align_cosine = epoch_align_cosine / max(1, epoch_align_count)
                    avg_align_skipped = epoch_align_skipped / max(1, epoch_align_count)
                    avg_grad_norm_cur = epoch_grad_norm_cur / max(1, epoch_align_count)
                    avg_grad_norm_mem = epoch_grad_norm_mem / max(1, epoch_align_count)
                    avg_comp = epoch_comp / max(1, n_batches)
                    avg_comp_pairs = epoch_comp_pairs / max(1, epoch_comp_count)
                    avg_comp_score_rms = epoch_comp_score_rms / max(1, epoch_comp_count)
                    if not args.disable_tqdm:
                        batch_iter.set_postfix(
                            loss=f"{avg_total:.4f}",
                            replay=f"{avg_replay:.4f}",
                            align=f"{avg_align:.4f}",
                            cos=f"{avg_align_cosine:.4f}",
                            skip=f"{avg_align_skipped:.2f}",
                            comp=f"{avg_comp:.4f}",
                            comp_rms=f"{avg_comp_score_rms:.4f}",
                            mem=len(memory),
                        )
                    print(
                        f"[train] task={task_name} epoch={epoch} step={step} "
                        f"loss={avg_total:.6f} task={avg_task:.6f} replay={avg_replay:.6f} "
                        f"align={avg_align:.6f} align_cos={avg_align_cosine:.6f} "
                        f"align_skipped={avg_align_skipped:.6f} "
                        f"grad_norm_cur={avg_grad_norm_cur:.6f} grad_norm_mem={avg_grad_norm_mem:.6f} "
                        f"comp={avg_comp:.6f} comp_pairs={avg_comp_pairs:.6f} "
                        f"comp_score_rms={avg_comp_score_rms:.6f} last_cos={align_cosine_scalar:.6f} "
                        f"last_skip={align_skipped_scalar:.6f} memory={len(memory)}"
                    )
                    logger.log_step(
                        task_name,
                        epoch,
                        task_steps,
                        avg_total,
                        avg_task,
                        avg_replay,
                        avg_align,
                        avg_align_cosine,
                        avg_align_skipped,
                        avg_grad_norm_cur,
                        avg_grad_norm_mem,
                        avg_comp,
                        avg_comp_pairs,
                        avg_comp_score_rms,
                    )
                    logger.save_loss_plot()

                if args.max_steps_per_task > 0 and task_steps >= args.max_steps_per_task:
                    break

            if n_batches > 0:
                print(
                    f"[train] task={task_name} epoch={epoch} done "
                    f"loss={epoch_total / n_batches:.6f} task={epoch_task / n_batches:.6f} "
                    f"replay={epoch_replay / max(1, n_batches):.6f} "
                    f"align={epoch_align / max(1, n_batches):.6f} "
                    f"align_cos={epoch_align_cosine / max(1, epoch_align_count):.6f} "
                    f"align_skipped={epoch_align_skipped / max(1, epoch_align_count):.6f} "
                    f"grad_norm_cur={epoch_grad_norm_cur / max(1, epoch_align_count):.6f} "
                    f"grad_norm_mem={epoch_grad_norm_mem / max(1, epoch_align_count):.6f} "
                    f"comp={epoch_comp / max(1, n_batches):.6f} "
                    f"comp_pairs={epoch_comp_pairs / max(1, epoch_comp_count):.6f} "
                    f"comp_score_rms={epoch_comp_score_rms / max(1, epoch_comp_count):.6f}"
                )

            if args.max_steps_per_task > 0 and task_steps >= args.max_steps_per_task:
                break

        print(f"[eval] task {task_name} finished, running continual evaluation")
        task_accs, detail = evaluator.evaluate_all(
            ebm_model=model,
            task_vectors=task_vectors,
            task_prototypes=task_prototypes,
            max_seen_task_idx=task_idx,
        )
        result_t.append(task_idx)
        result_a.append(task_accs)
        eval_records.append(
            {
                "stage": "after_task",
                "task_idx": task_idx,
                "task_name": task_name,
                "accs": task_accs,
                "detail": detail,
            }
        )
        print("[eval] accs:", [round(x, 4) for x in task_accs])
        logger.mark_task_boundary(task_name)
        logger.log_task_eval(task_idx, task_name, task_accs, task_order)
        task_stats = confusion_matrix(torch.LongTensor(result_t), torch.tensor(result_a, dtype=torch.float32))
        logger.log_task_stats(task_idx, [float(x) for x in task_stats])
        logger.save_loss_plot()

        if not args.disable_task_checkpoint:
            ckpt_path = task_checkpoint_path(out_prefix, task_idx, task_name)
            save_task_checkpoint(
                ckpt_path,
                {
                    "method": "meta_ebm",
                    "out_prefix": out_prefix,
                    "args": vars(args),
                    "task_order": task_order,
                    "input_dim": input_dim,
                    "condition_dim": cond_dim,
                    "next_task_idx": task_idx + 1,
                    "result_t": result_t,
                    "result_a": result_a,
                    "eval_records": eval_records,
                    "task_prototypes": task_prototypes,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "memory_state": memory.state_dict(),
                    "rng_state": capture_rng_state(),
                    "spent_time_sec": spent_before + (time.time() - time_start),
                },
            )
            print(f"[ckpt] saved: {ckpt_path}")

    spent = spent_before + (time.time() - time_start)
    if not result_t or not result_a:
        raise RuntimeError("No evaluation results available for statistics.")

    result_t_tensor = torch.LongTensor(result_t)
    result_a_tensor = torch.tensor(result_a, dtype=torch.float32)
    stats = confusion_matrix(result_t_tensor, result_a_tensor, out_prefix + ".txt")
    one_liner = {
        "args": vars(args),
        "stats": [float(x) for x in stats],
        "spent_time_sec": spent,
    }

    torch.save(
        {
            "result_t": result_t_tensor,
            "result_a": result_a_tensor,
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "one_liner": one_liner,
            "args": vars(args),
            "task_order": task_order,
            "input_dim": input_dim,
            "condition_dim": cond_dim,
        },
        out_prefix + ".pt",
    )

    dump_eval_details(
        out_prefix + "_eval.json",
        {
            "task_order": task_order,
            "result_t": result_t,
            "result_a": result_a,
            "eval_records": eval_records,
            "stats": [float(x) for x in stats],
            "spent_time_sec": spent,
            "args": vars(args),
        },
    )

    with open(out_prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(one_liner, f, ensure_ascii=False, indent=2)

    logger.log_final([float(x) for x in stats], spent)

    print("[done] experiment finished")
    print("[done] output_prefix:", out_prefix)
    print("[done] stats Final/BWT/FWT =", " ".join([f"{float(x):.4f}" for x in stats]))
    print(f"[done] spent: {spent:.2f} sec")
    print(f"[done] log: {logger.log_path}")
    print(f"[done] loss_plot: {logger.plot_path}")


if __name__ == "__main__":
    train(parse_args())
