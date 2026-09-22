#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401 - configure the grouped source layout
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from checkpointing import load_task_checkpoint
from deepseek_adapter_data import (
    CheckpointSpec,
    discover_task_checkpoints,
    load_adapter_state,
    load_adapter_vector,
    parse_task_order,
    pool_vector_cpu,
    reconstruct_lora_state,
    save_generated_adapter_checkpoint,
)
from models.meta_ebm import MetaEBM
from output_guard import OutputDirGuard, assert_result_dir_alive

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def load_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required for YAML configs.")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine DeepSeek LoRA adapters with a trained Meta-EBM and run GLUE eval")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ebm-checkpoint", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--checkpoint-root", type=str, default="")
    parser.add_argument("--lora-project-dir", type=str, default="")
    parser.add_argument("--lora-config-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--source-index", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--refine-steps", type=int, default=5)
    parser.add_argument("--refine-lr", type=float, default=1e-4)
    parser.add_argument("--refine-beta", type=float, default=1.0)
    parser.add_argument("--refine-grad-clip", type=float, default=0.0)
    parser.add_argument("--langevin-std", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--init-mode", choices=("prototype", "random"), default="prototype")
    parser.add_argument("--random-init-std", type=float, default=1.0)
    parser.add_argument("--random-init-seed", type=int, default=-1)
    parser.add_argument("--init-noise-std", type=float, default=0.0)
    parser.add_argument("--save-safetensors", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--eval-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--dataset-source", choices=("auto", "official", "local"), default="local")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cuda-device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    known, _ = config_parser.parse_known_args()
    if known.config:
        parser.set_defaults(**load_yaml_config(known.config))
    args = parser.parse_args()
    if not args.ebm_checkpoint:
        parser.error("--ebm-checkpoint is required, either as a CLI flag or in --config YAML.")
    return args


def import_lora_eval_helpers(lora_project_dir: str) -> Dict[str, Any]:
    lora_project_dir = os.path.abspath(os.path.expanduser(lora_project_dir))
    if not os.path.isdir(lora_project_dir):
        raise FileNotFoundError(f"LoRA DeepSeek project dir does not exist: {lora_project_dir}")
    if lora_project_dir not in sys.path:
        sys.path.insert(0, lora_project_dir)

    from eval_untrained_glue import (  # type: ignore
        build_fallback_compute_metrics,
        build_model_and_tokenizer,
        build_runtime_args,
        determine_precision,
        load_eval_dataset,
        preprocess_dataset,
        run_manual_prediction,
    )
    from glue_lora_common import (  # type: ignore
        configure_cuda_device,
        extract_record_ids,
        load_config,
        save_predictions,
        split_has_real_labels,
    )
    from peft import PeftModel  # type: ignore
    from transformers import DataCollatorWithPadding, default_data_collator  # type: ignore

    return {
        "build_fallback_compute_metrics": build_fallback_compute_metrics,
        "build_model_and_tokenizer": build_model_and_tokenizer,
        "build_runtime_args": build_runtime_args,
        "determine_precision": determine_precision,
        "load_eval_dataset": load_eval_dataset,
        "preprocess_dataset": preprocess_dataset,
        "run_manual_prediction": run_manual_prediction,
        "configure_cuda_device": configure_cuda_device,
        "extract_record_ids": extract_record_ids,
        "load_config": load_config,
        "save_predictions": save_predictions,
        "split_has_real_labels": split_has_real_labels,
        "PeftModel": PeftModel,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "default_data_collator": default_data_collator,
    }


def load_ebm(path: str, device: torch.device) -> Tuple[MetaEBM, Dict[str, Any]]:
    payload = load_task_checkpoint(os.path.abspath(os.path.expanduser(path)))
    model = MetaEBM(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def choose_source_checkpoint(specs: Sequence[CheckpointSpec], source_index: int) -> CheckpointSpec:
    if not specs:
        raise ValueError("No source checkpoints are available.")
    index = max(0, min(int(source_index), len(specs) - 1))
    return specs[index]


def expand_pooled_delta(delta_pooled: torch.Tensor, target_dim: int) -> torch.Tensor:
    return expand_pooled_vector(delta_pooled, target_dim)


def expand_pooled_vector(pooled: torch.Tensor, target_dim: int) -> torch.Tensor:
    delta = pooled.detach().cpu().float().view(-1)
    pool_length = int(delta.numel())
    target_dim = int(target_dim)
    if target_dim == pool_length:
        return delta
    padded_dim = target_dim
    if padded_dim % pool_length != 0:
        padded_dim += pool_length - (padded_dim % pool_length)
    chunk_size = padded_dim // pool_length
    return delta.repeat_interleave(chunk_size)[:target_dim].contiguous()


def task_prototype_pooled(payload: Dict[str, Any], task: str, pool_length: int) -> torch.Tensor:
    prototypes = payload.get("pooled_task_prototypes") or {}
    proto = prototypes.get(task) if isinstance(prototypes, dict) else None
    if proto is None:
        return torch.zeros(int(pool_length), dtype=torch.float32)
    proto = proto.detach().cpu().float().view(-1)
    if proto.numel() != int(pool_length):
        raise ValueError(
            f"Task prototype for {task} has length {proto.numel()}, expected {int(pool_length)}."
        )
    return proto


def task_prototype_vector(payload: Dict[str, Any], task: str, target_dim: int) -> torch.Tensor:
    prototypes = payload.get("task_prototypes") or {}
    proto = prototypes.get(task) if isinstance(prototypes, dict) else None
    if proto is None:
        raise KeyError(
            f"Task {task} is missing full task_prototypes in the EBM checkpoint. "
            "Please retrain with the ViT-style full-vector Meta-EBM pipeline."
        )
    proto = proto.detach().cpu().float().view(-1)
    if proto.numel() != int(target_dim):
        raise ValueError(
            f"Task prototype for {task} has length {proto.numel()}, expected {int(target_dim)}."
        )
    return proto


def _stable_task_seed(task: str) -> int:
    value = 0
    for index, char in enumerate(str(task)):
        value = (value + (index + 1) * ord(char)) % (2**31 - 1)
    return value


def initial_adapter_vector(
    payload: Dict[str, Any],
    task: str,
    target_dim: int,
    init_mode: str,
    random_init_std: float,
    init_noise_std: float,
    random_init_seed: int = -1,
    sample_idx: int = 0,
) -> torch.Tensor:
    mode = str(init_mode or "prototype").lower()
    target_dim = int(target_dim)
    if mode == "prototype":
        init_vec = task_prototype_vector(payload, task, target_dim).view(1, -1)
    elif mode == "random":
        generator = None
        if int(random_init_seed) >= 0:
            seed = int(random_init_seed) + _stable_task_seed(task) + int(sample_idx) * 1000003
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed % (2**31 - 1))
        init_vec = torch.randn((1, target_dim), generator=generator, dtype=torch.float32)
        init_vec = init_vec * float(random_init_std)
    else:
        raise ValueError(f"Unsupported init_mode: {init_mode}")

    if init_noise_std > 0:
        init_vec = init_vec + torch.randn_like(init_vec) * float(init_noise_std)
    return init_vec


def generate_adapter_vector_with_pooled_energy(
    model: MetaEBM,
    init_pooled: torch.Tensor,
    cond: torch.Tensor,
    target_dim: int,
    device: torch.device,
    refine_steps: int,
    refine_lr: float,
    refine_beta: float,
    refine_grad_clip: float,
    langevin_std: float,
) -> torch.Tensor:
    init_pooled = init_pooled.detach().cpu().float().view(1, -1)
    refined_pooled = model.refine(
        init_pooled.to(device),
        cond,
        steps=refine_steps,
        step_size=refine_lr,
        grad_clip=refine_grad_clip,
        beta=refine_beta,
        langevin_noise_std=langevin_std,
    ).detach().cpu().view(-1)
    return expand_pooled_vector(refined_pooled, target_dim)


def refine_adapter_vector_with_pooled_energy(
    model: MetaEBM,
    init_vec: torch.Tensor,
    cond: torch.Tensor,
    device: torch.device,
    refine_steps: int,
    refine_lr: float,
    refine_beta: float,
    refine_grad_clip: float,
    langevin_std: float,
) -> torch.Tensor:
    init_flat = init_vec.detach().cpu().float().view(-1)
    init_pooled = pool_vector_cpu(init_flat, int(model.pool_length)).view(1, -1)
    refined_pooled = model.refine(
        init_pooled.to(device),
        cond,
        steps=refine_steps,
        step_size=refine_lr,
        grad_clip=refine_grad_clip,
        beta=refine_beta,
        langevin_noise_std=langevin_std,
    ).detach().cpu().view(-1)
    pooled_delta = refined_pooled - init_pooled.view(-1)
    return init_flat + expand_pooled_delta(pooled_delta, init_flat.numel())


def refine_and_export_adapter(
    model: MetaEBM,
    payload: Dict[str, Any],
    task: str,
    source_checkpoint: str,
    output_dir: str,
    device: torch.device,
    refine_steps: int,
    refine_lr: float,
    refine_beta: float,
    refine_grad_clip: float,
    langevin_std: float,
    init_mode: str,
    random_init_std: float,
    random_init_seed: int,
    sample_idx: int,
    init_noise_std: float,
    save_safetensors: bool,
    guard_root: Optional[str] = None,
) -> str:
    template = payload["template"]
    normalize_mode = payload.get("normalize_mode", "per_tensor")
    eps = float(payload.get("normalization_eps", 1e-6))
    task_embeddings = payload["task_embeddings"]
    if task not in task_embeddings:
        raise KeyError(f"Task {task} is not in the EBM checkpoint task embeddings.")

    init_vec = initial_adapter_vector(
        payload=payload,
        task=task,
        target_dim=int(template["input_dim"]),
        init_mode=init_mode,
        random_init_std=random_init_std,
        init_noise_std=init_noise_std,
        random_init_seed=random_init_seed,
        sample_idx=sample_idx,
    )
    cond = task_embeddings[task].float().view(1, -1).to(device)
    refined = model.refine(
        init_vec.to(device),
        cond,
        steps=refine_steps,
        step_size=refine_lr,
        grad_clip=refine_grad_clip,
        beta=refine_beta,
        langevin_noise_std=langevin_std,
    ).detach().cpu().view(-1)

    source_state = load_adapter_state(source_checkpoint)
    generated_lora = reconstruct_lora_state(
        refined,
        source_state=source_state,
        template=template,
        normalize_mode=normalize_mode,
        eps=eps,
    )
    return save_generated_adapter_checkpoint(
        output_dir=output_dir,
        source_checkpoint_dir=source_checkpoint,
        generated_lora_state=generated_lora,
        save_safetensors=save_safetensors,
        guard_root=guard_root,
    )


def resolve_config_path(lora_config_dir: str, task: str) -> str:
    path = os.path.join(lora_config_dir, f"{task}_lora_config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LoRA config not found for task={task}: {path}")
    return path


def evaluate_generated_adapter(
    helpers: Dict[str, Any],
    task: str,
    adapter_dir: str,
    task_output_dir: str,
    lora_config_dir: str,
    args: argparse.Namespace,
    guard_root: Optional[str] = None,
) -> Dict[str, Any]:
    if guard_root:
        OutputDirGuard(guard_root).makedirs(task_output_dir)
    else:
        os.makedirs(task_output_dir, exist_ok=True)
    config_path = resolve_config_path(lora_config_dir, task)
    config = helpers["load_config"](config_path)
    model_args, data_args, metrics_cfg, training_cfg, _ = helpers["build_runtime_args"](
        config=config,
        task_name=task,
        model_path_override=args.model_path,
        requested_precision=args.precision,
        force_trust_remote_code=args.trust_remote_code,
    )

    raw_dataset, dataset_source = helpers["load_eval_dataset"](
        task_name=task,
        data_args=data_args,
        model_args=model_args,
        dataset_source=args.dataset_source,
        eval_split=args.eval_split,
    )
    if args.max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), args.max_samples)))

    record_ids = list(helpers["extract_record_ids"](raw_dataset, data_args.id_column))
    has_real_labels = helpers["split_has_real_labels"](
        raw_dataset,
        data_args.label_column,
        model_args.problem_type == "regression",
    )

    model, tokenizer = helpers["build_model_and_tokenizer"](model_args, data_args)
    model = helpers["PeftModel"].from_pretrained(model, adapter_dir, is_trainable=False)
    processed_dataset = helpers["preprocess_dataset"](
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        overwrite_cache=data_args.overwrite_cache,
    )

    _, use_fp16, use_bf16 = helpers["determine_precision"](config, args.precision)
    if not torch.cuda.is_available():
        use_fp16 = False
        use_bf16 = False

    if data_args.pad_to_max_length:
        data_collator = helpers["default_data_collator"]
    elif use_fp16 or use_bf16:
        data_collator = helpers["DataCollatorWithPadding"](tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = helpers["DataCollatorWithPadding"](tokenizer)

    compute_metrics_fn = helpers["build_fallback_compute_metrics"](model_args, metrics_cfg) if has_real_labels else None
    predictions, metrics = helpers["run_manual_prediction"](
        model=model,
        dataset=processed_dataset,
        data_collator=data_collator,
        batch_size=args.per_device_batch_size,
        num_workers=args.dataloader_num_workers,
        metric_prefix=args.eval_split,
        compute_metrics_fn=compute_metrics_fn,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        progress_desc=f"{task}-{args.eval_split}",
    )

    if guard_root:
        assert_result_dir_alive(guard_root)
    predictions_path = helpers["save_predictions"](
        task_output_dir,
        f"{args.eval_split}_predictions.tsv",
        record_ids,
        np.asarray(predictions),
        model_args.problem_type == "regression",
    )
    if guard_root:
        assert_result_dir_alive(guard_root)
    metrics_path = os.path.join(task_output_dir, f"{args.eval_split}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    del processed_dataset
    del raw_dataset
    del tokenizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset_source": dataset_source,
        "config_path": config_path,
        "predictions_file": predictions_path,
        "metrics_file": metrics_path,
        "metrics": metrics,
        "has_real_labels": has_real_labels,
    }


def save_aggregate(output_root: str, rows: Sequence[Dict[str, Any]], guard_root: Optional[str] = None) -> Tuple[str, str]:
    if guard_root:
        assert_result_dir_alive(guard_root)
    json_path = os.path.join(output_root, "summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)

    metric_keys: List[str] = []
    for row in rows:
        for key in row.get("metrics", {}).keys():
            if key not in metric_keys:
                metric_keys.append(key)
    header = [
        "task",
        "sample",
        "status",
        "split",
        "source_checkpoint",
        "adapter_dir",
        "output_dir",
    ] + metric_keys + ["error"]
    tsv_path = os.path.join(output_root, "summary.tsv")
    with open(tsv_path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            values = [
                str(row.get("task", "")),
                str(row.get("sample", "")),
                str(row.get("status", "")),
                str(row.get("split", "")),
                str(row.get("source_checkpoint", "")),
                str(row.get("adapter_dir", "")),
                str(row.get("output_dir", "")),
            ]
            values.extend(str(row.get("metrics", {}).get(key, "")) for key in metric_keys)
            values.append(str(row.get("error", "")))
            handle.write("\t".join(values) + "\n")
    return json_path, tsv_path


def run_evaluation(args: argparse.Namespace) -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pami_root = os.path.abspath(os.path.join(script_dir, ".."))
    checkpoint_path = os.path.abspath(os.path.expanduser(args.ebm_checkpoint))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model, payload = load_ebm(checkpoint_path, device=device)

    tasks = parse_task_order(args.tasks) if args.tasks else list(payload["tasks"])
    checkpoint_root = os.path.abspath(
        os.path.expanduser(
            args.checkpoint_root
            or os.environ.get("CHECKPOINT_ROOT", "")
            or payload.get("checkpoint_root")
            or os.path.join(pami_root, "CMA_ES", "checkpoints")
        )
    )
    lora_project_dir = os.path.abspath(
        os.path.expanduser(args.lora_project_dir or os.path.join(pami_root, "models", "LoRA_deepseek"))
    )
    lora_config_dir = os.path.abspath(
        os.path.expanduser(args.lora_config_dir or os.path.join(pami_root, "models", "config", "LoRA_deepseek"))
    )
    output_root = os.path.abspath(args.output_dir or os.path.join(script_dir, "results", "eval"))
    parent_result_root = getattr(args, "parent_result_root", "")
    if parent_result_root:
        assert_result_dir_alive(parent_result_root, "parent training result directory")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(output_root, run_name)
    os.makedirs(run_root, exist_ok=True)
    result_guard = OutputDirGuard(run_root, "evaluation result directory", initialize=True)
    result_guard.assert_alive()

    helpers = None
    if not args.skip_eval:
        helpers = import_lora_eval_helpers(lora_project_dir)
        if args.cuda_device:
            helpers["configure_cuda_device"]({"cuda_device": args.cuda_device})

    task_to_specs = discover_task_checkpoints(tasks, checkpoint_root=checkpoint_root, topk_per_task=max(1, args.source_index + 1))
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        result_guard.assert_alive()
        try:
            source_spec = choose_source_checkpoint(task_to_specs[task], args.source_index)
            for sample_idx in range(max(1, int(args.num_samples))):
                result_guard.assert_alive()
                sample_dir = os.path.join(run_root, task, f"sample_{sample_idx:02d}")
                adapter_dir = os.path.join(sample_dir, "adapter")
                exported_dir = refine_and_export_adapter(
                    model=model,
                    payload=payload,
                    task=task,
                    source_checkpoint=source_spec.path,
                    output_dir=adapter_dir,
                    device=device,
                    refine_steps=args.refine_steps,
                    refine_lr=args.refine_lr,
                    refine_beta=args.refine_beta,
                    refine_grad_clip=args.refine_grad_clip,
                    langevin_std=args.langevin_std,
                    init_mode=args.init_mode,
                    random_init_std=args.random_init_std,
                    random_init_seed=args.random_init_seed,
                    sample_idx=sample_idx,
                    init_noise_std=args.init_noise_std if sample_idx > 0 else 0.0,
                    save_safetensors=args.save_safetensors,
                    guard_root=result_guard.root,
                )
                row = {
                    "task": task,
                    "sample": sample_idx,
                    "status": "exported",
                    "split": args.eval_split,
                    "source_checkpoint": source_spec.path,
                    "source_score": source_spec.score,
                    "source_metric": source_spec.metric_name,
                    "init_mode": args.init_mode,
                    "random_init_std": args.random_init_std,
                    "random_init_seed": args.random_init_seed,
                    "adapter_dir": exported_dir,
                    "output_dir": sample_dir,
                    "metrics": {},
                }
                if not args.skip_eval and helpers is not None:
                    eval_payload = evaluate_generated_adapter(
                        helpers=helpers,
                        task=task,
                        adapter_dir=exported_dir,
                        task_output_dir=sample_dir,
                        lora_config_dir=lora_config_dir,
                        args=args,
                        guard_root=result_guard.root,
                    )
                    row.update(eval_payload)
                    row["status"] = "ok"
                result_guard.assert_alive()
                with open(os.path.join(sample_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
                    json.dump(row, handle, ensure_ascii=False, indent=2)
                rows.append(row)
                print(f"[{row['status']}] task={task} sample={sample_idx} adapter={exported_dir}")
        except Exception as exc:
            failure = {
                "task": task,
                "sample": "",
                "status": "failed",
                "split": args.eval_split,
                "source_checkpoint": "",
                "init_mode": args.init_mode,
                "random_init_std": args.random_init_std,
                "random_init_seed": args.random_init_seed,
                "adapter_dir": "",
                "output_dir": os.path.join(run_root, task),
                "metrics": {},
                "error": str(exc),
            }
            result_guard.makedirs(failure["output_dir"])
            with open(os.path.join(failure["output_dir"], "run_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False, indent=2)
            rows.append(failure)
            print(f"[failed] task={task}: {exc}")
            if args.fail_fast:
                raise

    result_guard.assert_alive()
    summary_json, summary_tsv = save_aggregate(run_root, rows, guard_root=result_guard.root)
    print(f"[done] summary_json={summary_json}")
    print(f"[done] summary_tsv={summary_tsv}")
    return {
        "run_root": run_root,
        "summary_json": summary_json,
        "summary_tsv": summary_tsv,
    }


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
