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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from checkpointing import load_task_checkpoint
from qwen_adapter_data import (
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
    parser = argparse.ArgumentParser(description="Refine Qwen LoRA adapters with a trained Meta-EBM and run GLUE eval")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ebm-checkpoint", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--checkpoint-root", type=str, default="")
    parser.add_argument("--lora-project-dir", type=str, default="")
    parser.add_argument("--lora-config-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--checkpoint_output_suffix", "--checkpoint-output-suffix", type=str, default="qwen3_8b")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--refine-steps", type=int, default=5)
    parser.add_argument("--refine-lr", type=float, default=1e-4)
    parser.add_argument("--refine-beta", type=float, default=1.0)
    parser.add_argument("--refine-grad-clip", type=float, default=0.0)
    parser.add_argument("--langevin-std", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1)
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
        raise FileNotFoundError(f"LoRA Qwen project dir does not exist: {lora_project_dir}")
    if lora_project_dir not in sys.path:
        sys.path.insert(0, lora_project_dir)

    from glue_lora_common import (  # type: ignore
        canonical_glue_task_name,
        compute_fallback_metrics,
        configure_cuda_device,
        create_training_arguments,
        extract_record_ids,
        load_preferred_raw_datasets,
        load_config,
        normalize_text_batch,
        normalize_metrics_config,
        resolve_pretrained_checkpoint,
        resolve_torch_dtype,
        save_predictions,
        split_has_real_labels,
        task_to_keys,
    )
    from peft import PeftModel  # type: ignore
    from transformers import (  # type: ignore
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        default_data_collator,
    )

    return {
        "AutoConfig": AutoConfig,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "canonical_glue_task_name": canonical_glue_task_name,
        "compute_fallback_metrics": compute_fallback_metrics,
        "configure_cuda_device": configure_cuda_device,
        "create_training_arguments": create_training_arguments,
        "load_preferred_raw_datasets": load_preferred_raw_datasets,
        "extract_record_ids": extract_record_ids,
        "load_config": load_config,
        "normalize_text_batch": normalize_text_batch,
        "normalize_metrics_config": normalize_metrics_config,
        "resolve_pretrained_checkpoint": resolve_pretrained_checkpoint,
        "resolve_torch_dtype": resolve_torch_dtype,
        "save_predictions": save_predictions,
        "split_has_real_labels": split_has_real_labels,
        "task_to_keys": task_to_keys,
        "PeftModel": PeftModel,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
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

    init_vec = task_prototype_vector(payload, task, int(template["input_dim"])).view(1, -1)
    if init_noise_std > 0:
        init_vec = init_vec + torch.randn_like(init_vec) * float(init_noise_std)
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


def precision_to_dtype_name(precision: str, default_dtype: str) -> str:
    if precision == "fp32":
        return "float32"
    if precision == "fp16":
        return "float16"
    if precision == "bf16":
        return "bfloat16"
    return default_dtype


def ensure_cuda_build_env() -> None:
    if "CUDA_HOME" not in os.environ:
        for candidate in ("/usr/local/cuda-12.1", "/usr/local/cuda"):
            if os.path.isdir(candidate):
                os.environ["CUDA_HOME"] = candidate
                break
    compat_dir = "/usr/local/cuda-12.1/compat"
    if os.path.isdir(compat_dir):
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [item for item in current.split(":") if item]
        if compat_dir not in parts:
            os.environ["LD_LIBRARY_PATH"] = compat_dir + (f":{current}" if current else "")


def build_eval_arguments_from_config(
    helpers: Dict[str, Any],
    config_path: str,
    default_task_name: str,
) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    config = helpers["load_config"](config_path)
    experiment_cfg = config.get("experiment", {})
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = dict(config["training"])
    metrics_cfg = helpers["normalize_metrics_config"](config["metrics"], data_cfg.get("task_name"))
    output_cfg = dict(config["output"])
    text_columns = list(data_cfg.get("text_columns", []))

    model_args = SimpleNamespace(
        model_name_or_path=helpers["resolve_pretrained_checkpoint"](str(model_cfg["pretrained_checkpoint"])),
        config_name=model_cfg.get("config_name"),
        tokenizer_name=model_cfg.get("tokenizer_name"),
        cache_dir=model_cfg.get("cache_dir"),
        use_fast_tokenizer=bool(model_cfg.get("use_fast_tokenizer", True)),
        model_revision=model_cfg.get("model_revision", "main"),
        token=model_cfg.get("token"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        ignore_mismatched_sizes=bool(model_cfg.get("ignore_mismatched_sizes", True)),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        low_cpu_mem_usage=bool(model_cfg.get("low_cpu_mem_usage", True)),
        problem_type=model_cfg.get("problem_type", "single_label_classification"),
        num_labels=int(model_cfg["num_labels"]),
    )
    data_args = SimpleNamespace(
        task_name=(data_cfg.get("task_name") or default_task_name).lower(),
        dataset_name=data_cfg.get("dataset_name"),
        dataset_config_name=data_cfg.get("dataset_config_name"),
        prefer_official_dataset=bool(data_cfg.get("prefer_official_dataset", False)),
        predict_test_split=False,
        max_seq_length=int(data_cfg["max_length"]),
        overwrite_cache=bool(data_cfg.get("overwrite_cache", False)),
        pad_to_max_length=bool(data_cfg.get("pad_to_max_length", True)),
        max_train_samples=data_cfg.get("max_train_samples"),
        max_eval_samples=data_cfg.get("max_eval_samples"),
        max_predict_samples=data_cfg.get("max_predict_samples"),
        train_file=data_cfg.get("train_file"),
        validation_file=data_cfg.get("dev_file"),
        test_file=data_cfg.get("test_file"),
        sentence1_key=text_columns[0] if len(text_columns) >= 1 else None,
        sentence2_key=text_columns[1] if len(text_columns) == 2 else None,
        label_column=data_cfg.get("label_column", "label"),
        id_column=data_cfg.get("id_column", "idx"),
        display_name=data_cfg.get("display_name") or data_cfg.get("task_name") or default_task_name,
    )
    runtime_cfg = {
        "config": config,
        "config_path": config_path,
        "data_cfg": data_cfg,
        "train_cfg": train_cfg,
        "metrics_cfg": metrics_cfg,
        "output_cfg": output_cfg,
        "task_display_name": data_args.display_name,
        "experiment_name": experiment_cfg.get("name", str(data_args.display_name).lower().replace("-", "_")),
    }
    return model_args, data_args, metrics_cfg, train_cfg, runtime_cfg


def prepare_eval_runtime(
    helpers: Dict[str, Any],
    config_path: str,
    task: str,
    task_output_dir: str,
    args: argparse.Namespace,
) -> Tuple[Any, Any, Dict[str, Any], Any, Dict[str, Any]]:
    model_args, data_args, metrics_cfg, train_cfg, runtime_cfg = build_eval_arguments_from_config(
        helpers=helpers,
        config_path=config_path,
        default_task_name=task,
    )
    if args.model_path:
        model_args.model_name_or_path = os.path.abspath(os.path.expanduser(args.model_path))
    if args.trust_remote_code:
        model_args.trust_remote_code = True
    model_args.torch_dtype = precision_to_dtype_name(args.precision, str(model_args.torch_dtype))

    if args.dataset_source == "official":
        data_args.prefer_official_dataset = True
        if args.eval_split == "validation":
            data_args.validation_file = None
        elif args.eval_split == "test":
            data_args.test_file = None
    elif args.dataset_source == "local":
        data_args.prefer_official_dataset = False

    train_cfg["eval_batch_size"] = int(args.per_device_batch_size)
    train_cfg["dataloader_num_workers"] = int(args.dataloader_num_workers)
    if args.precision == "fp32":
        train_cfg["fp16"] = False
        train_cfg["bf16"] = False
    elif args.precision == "fp16":
        train_cfg["fp16"] = True
        train_cfg["bf16"] = False
    elif args.precision == "bf16":
        train_cfg["fp16"] = False
        train_cfg["bf16"] = True

    output_cfg = dict(runtime_cfg["output_cfg"])
    output_cfg["save_dir"] = task_output_dir
    training_args = helpers["create_training_arguments"](
        train_cfg=train_cfg,
        metrics_cfg=metrics_cfg,
        output_cfg=output_cfg,
        run_mode="train",
        do_train=False,
        do_eval=args.eval_split == "validation",
        do_predict=args.eval_split == "test",
    )
    return model_args, data_args, metrics_cfg, training_args, runtime_cfg


def build_qwen_model_and_tokenizer(helpers: Dict[str, Any], model_args: Any, data_args: Any) -> Tuple[Any, Any]:
    canonical_task_name = helpers["canonical_glue_task_name"](data_args.task_name)
    config = helpers["AutoConfig"].from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=int(model_args.num_labels),
        finetuning_task=canonical_task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = helpers["AutoTokenizer"].from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = helpers["AutoModelForSequenceClassification"].from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        torch_dtype=helpers["resolve_torch_dtype"](model_args.torch_dtype),
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = model_args.problem_type
    if model_args.problem_type != "regression":
        model.config.label2id = {str(index): index for index in range(int(model_args.num_labels))}
        model.config.id2label = {index: str(index) for index in range(int(model_args.num_labels))}
    return model, tokenizer


def preprocess_eval_dataset(
    helpers: Dict[str, Any],
    raw_dataset: Any,
    tokenizer: Any,
    data_args: Any,
    model_args: Any,
) -> Any:
    if data_args.sentence1_key is None:
        task_to_keys = helpers["task_to_keys"]
        if data_args.task_name in task_to_keys:
            data_args.sentence1_key, data_args.sentence2_key = task_to_keys[data_args.task_name]
        else:
            raise ValueError("Could not infer sentence columns.")

    max_seq_length = min(int(data_args.max_seq_length), tokenizer.model_max_length)
    padding = "max_length" if data_args.pad_to_max_length else False
    is_regression = model_args.problem_type == "regression"

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        sentence1_batch = helpers["normalize_text_batch"](examples[data_args.sentence1_key])
        if data_args.sentence2_key is None:
            text_args = (sentence1_batch,)
        else:
            text_args = (sentence1_batch, helpers["normalize_text_batch"](examples[data_args.sentence2_key]))
        result = tokenizer(*text_args, padding=padding, max_length=max_seq_length, truncation=True)
        if data_args.label_column in examples:
            raw_labels = examples[data_args.label_column]
            if is_regression:
                result["label"] = [float(label) for label in raw_labels]
            else:
                result["label"] = [-1 if label is None else int(label) for label in raw_labels]
        return result

    return raw_dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=raw_dataset.column_names,
        desc="Running tokenizer on eval dataset",
    )


def build_local_compute_metrics(helpers: Dict[str, Any], model_args: Any, metrics_cfg: Dict[str, Any]) -> Any:
    metric_names = list(metrics_cfg.get("names", []))
    num_labels = int(model_args.num_labels)
    f1_average = metrics_cfg.get("f1_average", "binary" if num_labels == 2 else "macro")
    positive_label = int(metrics_cfg.get("positive_label", 1))

    def compute(eval_pred: Any) -> Dict[str, float]:
        predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        labels = eval_pred.label_ids
        if not isinstance(predictions, np.ndarray):
            predictions = np.asarray(predictions)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        return helpers["compute_fallback_metrics"](
            predictions=predictions,
            labels=labels,
            metric_names=metric_names,
            problem_type=model_args.problem_type,
            f1_average=f1_average,
            positive_label=positive_label,
        )

    return compute


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
    model_args, data_args, metrics_cfg, training_args, _ = prepare_eval_runtime(
        helpers=helpers,
        config_path=config_path,
        task=task,
        task_output_dir=task_output_dir,
        args=args,
    )
    raw_datasets, dataset_source = helpers["load_preferred_raw_datasets"](
        training_args,
        data_args,
        model_args,
    )
    raw_dataset = raw_datasets[args.eval_split]
    if args.max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), args.max_samples)))

    record_ids = list(helpers["extract_record_ids"](raw_dataset, data_args.id_column))
    has_real_labels = helpers["split_has_real_labels"](
        raw_dataset,
        data_args.label_column,
        model_args.problem_type == "regression",
    )

    model, tokenizer = build_qwen_model_and_tokenizer(helpers, model_args, data_args)
    model = helpers["PeftModel"].from_pretrained(model, adapter_dir, is_trainable=False)
    processed_dataset = preprocess_eval_dataset(
        helpers=helpers,
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
    )

    if data_args.pad_to_max_length:
        data_collator = helpers["default_data_collator"]
    elif getattr(training_args, "fp16", False) or getattr(training_args, "bf16", False):
        data_collator = helpers["DataCollatorWithPadding"](tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = helpers["DataCollatorWithPadding"](tokenizer)

    compute_metrics_fn = build_local_compute_metrics(helpers, model_args, metrics_cfg) if has_real_labels else None
    dataset_for_prediction = processed_dataset
    if not has_real_labels and "label" in processed_dataset.column_names:
        dataset_for_prediction = processed_dataset.remove_columns("label")
    trainer = helpers["Trainer"](
        model=model,
        args=training_args,
        eval_dataset=processed_dataset,
        compute_metrics=compute_metrics_fn,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    prediction_output = trainer.predict(dataset_for_prediction, metric_key_prefix=args.eval_split)
    predictions = prediction_output.predictions[0] if isinstance(prediction_output.predictions, tuple) else prediction_output.predictions
    predictions = np.asarray(predictions)
    metrics = dict(prediction_output.metrics)

    if guard_root:
        assert_result_dir_alive(guard_root)
    predictions_path = helpers["save_predictions"](
        task_output_dir,
        f"{args.eval_split}_predictions.tsv",
        record_ids,
        predictions,
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
    ensure_cuda_build_env()
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
            or os.path.join(pami_root, "CMA_ES", "Qwen", "checkpoints")
        )
    )
    lora_project_dir = os.path.abspath(
        os.path.expanduser(args.lora_project_dir or os.path.join(pami_root, "models", "LoRA_Qwen"))
    )
    lora_config_dir = os.path.abspath(
        os.path.expanduser(args.lora_config_dir or os.path.join(pami_root, "models", "config", "LoRA_Qwen"))
    )
    output_root = os.path.abspath(args.output_dir or os.path.join(script_dir, "results", "eval"))
    parent_result_root = getattr(args, "parent_result_root", "")
    if parent_result_root:
        assert_result_dir_alive(parent_result_root, "parent training result directory")
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(output_root, run_name)
    os.makedirs(run_root, exist_ok=True)
    result_guard = OutputDirGuard(run_root, "evaluation result directory")
    result_guard.assert_alive()

    helpers = None
    if not args.skip_eval:
        helpers = import_lora_eval_helpers(lora_project_dir)
        if args.cuda_device:
            helpers["configure_cuda_device"]({"cuda_device": args.cuda_device})

    output_suffix = getattr(args, "checkpoint_output_suffix", None) or payload.get(
        "checkpoint_output_suffix",
        "qwen3_8b",
    )
    task_to_specs = discover_task_checkpoints(
        tasks,
        checkpoint_root=checkpoint_root,
        topk_per_task=max(1, args.source_index + 1),
        output_suffix=output_suffix,
    )
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
