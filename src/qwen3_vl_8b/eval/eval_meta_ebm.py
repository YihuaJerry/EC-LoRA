#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401 - configure the grouped source layout
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from checkpointing import load_task_checkpoint
from models.meta_ebm import MetaEBM
from output_guard import OutputDirGuard, assert_result_dir_alive
from qwenvl_adapter_data import (
    CheckpointSpec,
    discover_task_checkpoints,
    load_adapter_state,
    parse_task_order,
    reconstruct_lora_state,
    save_generated_adapter_checkpoint,
)

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


TASK_EVAL_SPECS: Dict[str, Dict[str, str]] = {
    "scienceqa": {
        "postprocess_task": "science_qa",
        "score_module": "qwen_eval.scoring.eval_science_qa",
        "score_data_arg": "--annotation-file",
    },
    "vizwiz": {
        "postprocess_task": "vizwiz_caption",
        "score_module": "qwen_eval.scoring.eval_vizwiz_caption",
        "score_data_arg": "--annotation-file",
    },
    "imagenet": {
        "postprocess_task": "imagenet",
        "score_module": "qwen_eval.scoring.eval_ImagetNet",
        "score_data_arg": "--test-file",
    },
    "vqav2": {
        "postprocess_task": "vqav2",
        "score_module": "qwen_eval.scoring.eval_vqav2",
        "score_data_arg": "--annotation-file",
    },
    "iconqa": {
        "postprocess_task": "iconqa",
        "score_module": "qwen_eval.scoring.eval_iconqa",
        "score_data_arg": "--annotation-file",
    },
    "flickr30k": {
        "postprocess_task": "flickr30k",
        "score_module": "qwen_eval.scoring.eval_flickr30k",
        "score_data_arg": "--annotation-file",
    },
    "grounding": {
        "postprocess_task": "grounding",
        "score_module": "qwen_eval.scoring.eval_grounding",
        "score_data_arg": "--test-file",
    },
    "ocr_vqa": {
        "postprocess_task": "ocrvqa",
        "score_module": "qwen_eval.scoring.eval_ocrvqa",
        "score_data_arg": "--annotation-file",
    },
}

SEEN_BENCHMARK_TASKS = (
    "scienceqa",
    "vizwiz",
    "imagenet",
    "vqav2",
    "iconqa",
    "flickr30k",
    "grounding",
    "ocr_vqa",
)
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
ALL_BENCHMARK_TASKS = SEEN_BENCHMARK_TASKS + UNSEEN_BENCHMARK_TASKS
IN_PROCESS_SCORE_TASKS = {
    "scienceqa",
    "imagenet",
    "vqav2",
    "iconqa",
    "grounding",
    "ocr_vqa",
    "aokvqa",
    "imagenet_r",
    "tabmwp",
}

BENCHMARK_TASK_ALIASES = {
    "science_qa": "scienceqa",
    "image_net": "imagenet",
    "ocrvqa": "ocr_vqa",
    "ocr-vqa": "ocr_vqa",
    "aok_vqa": "aokvqa",
    "aok-vqa": "aokvqa",
    "imagenet-r": "imagenet_r",
    "imagenet_r": "imagenet_r",
    "screen_2_words": "screen2words",
    "screen-2-words": "screen2words",
    "tab_mwp": "tabmwp",
    "tab-mwp": "tabmwp",
}

MM_MERGEBENCH_ROOT = os.path.expanduser(os.environ.get("MM_MERGEBENCH_ROOT", "data/MM-MergeBench"))

UNSEEN_EVAL_SPECS: Dict[str, Dict[str, Any]] = {
    "aokvqa": {
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "AOKVQA", "val.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "AOKVQA", "val.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data", "AOKVQA"),
        "postprocess_task": "iconqa",
        "score_kind": "aokvqa",
        "max_new_tokens": 32,
    },
    "imagenet_r": {
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "ImageNet-R", "test.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "ImageNet-R", "test.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data"),
        "postprocess_task": "imagenet",
        "score_kind": "imagenet_r",
        "max_new_tokens": 32,
    },
    "screen2words": {
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "Screen2words", "test.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "Screen2words", "test_coco_type.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data"),
        "postprocess_task": "flickr30k",
        "score_kind": "caption",
        "score_module": "qwen_eval.scoring.eval_flickr30k",
        "score_data_arg": "--annotation-file",
        "max_new_tokens": 64,
    },
    "tabmwp": {
        "question_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "TabMWP", "test.json"),
        "annotation_file": os.path.join(MM_MERGEBENCH_ROOT, "instructions", "Unseen_data", "TabMWP", "test.json"),
        "image_folder": os.path.join(MM_MERGEBENCH_ROOT, "Unseen_data"),
        "postprocess_task": "ocrvqa",
        "score_kind": "tabmwp",
        "max_new_tokens": 32,
    },
}


def load_yaml_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required for YAML configs.")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine QwenVL LoRA adapters with a trained Meta-EBM and run MM-MergeBench eval")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ebm-checkpoint", type=str, default="")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--benchmark-tasks", type=str, default="")
    parser.add_argument(
        "--pairing-mode",
        type=str,
        default="cartesian",
        choices=["cartesian", "seen_one_to_one_unseen_current", "seen_one_to_one_unseen_mapped"],
    )
    parser.add_argument("--current-task", type=str, default="")
    parser.add_argument("--checkpoint-root", type=str, default="")
    parser.add_argument("--qwen-project-dir", "--lora-project-dir", dest="qwen_project_dir", type=str, default="")
    parser.add_argument("--qwen-config-dir", "--lora-config-dir", dest="qwen_config_dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--parent-result-root", type=str, default="")
    parser.add_argument("--source-index", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--refine-steps", type=int, default=5)
    parser.add_argument("--refine-lr", type=float, default=1e-4)
    parser.add_argument("--refine-beta", type=float, default=1.0)
    parser.add_argument("--refine-grad-clip", type=float, default=0.0)
    parser.add_argument("--langevin-std", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--init-noise-std", type=float, default=0.0)
    parser.add_argument("--save-safetensors", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--python-bin", type=str, default="")
    parser.add_argument("--model-base", "--model-path", dest="model_base", type=str, default="")
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument("--precision", type=str, default="")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--qwen-device", type=str, default="cuda")
    parser.add_argument("--cuda-device", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
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


def parse_benchmark_tasks(task_text: str, default: Sequence[str] = ALL_BENCHMARK_TASKS) -> List[str]:
    raw = str(task_text or "").strip()
    if not raw:
        tokens = list(default)
    elif raw.lower() in {"all", "all_datasets", "seen+unseen", "seen_so_far_plus_unseen", "progressive"}:
        tokens = list(ALL_BENCHMARK_TASKS)
    elif raw.lower() == "seen":
        tokens = list(SEEN_BENCHMARK_TASKS)
    elif raw.lower() == "unseen":
        tokens = list(UNSEEN_BENCHMARK_TASKS)
    else:
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
    out: List[str] = []
    seen = set()
    for token in tokens:
        normalized = BENCHMARK_TASK_ALIASES.get(str(token).strip().lower(), str(token).strip().lower())
        if normalized not in ALL_BENCHMARK_TASKS:
            raise ValueError(
                f"Unsupported benchmark task={token!r}. "
                f"Expected one of: {', '.join(ALL_BENCHMARK_TASKS)}"
            )
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    if not out:
        raise ValueError("benchmark_tasks is empty.")
    return out


def adapter_task_for_benchmark(eval_task: str, adapter_tasks: Sequence[str]) -> str:
    mapped_task = UNSEEN_ADAPTER_TASKS.get(eval_task)
    if mapped_task in adapter_tasks:
        return mapped_task
    return eval_task if eval_task in adapter_tasks else ""


def normalize_current_adapter_task(current_task: str, adapter_tasks: Sequence[str]) -> str:
    if current_task in adapter_tasks:
        return current_task
    mapped = adapter_task_for_benchmark(current_task, adapter_tasks)
    return mapped or (adapter_tasks[-1] if adapter_tasks else "")


def unique_ordered(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_eval_plan(
    adapter_tasks: Sequence[str],
    benchmark_tasks: Sequence[str],
    pairing_mode: str,
    current_task: str = "",
) -> Dict[str, List[str]]:
    plan: Dict[str, List[str]] = {task: [] for task in adapter_tasks}
    if pairing_mode == "cartesian":
        for task in adapter_tasks:
            plan[task] = list(benchmark_tasks)
        return plan
    if pairing_mode not in {"seen_one_to_one_unseen_current", "seen_one_to_one_unseen_mapped"}:
        raise ValueError(f"Unsupported pairing_mode={pairing_mode!r}")

    current_adapter_task = normalize_current_adapter_task(current_task, adapter_tasks)
    for eval_task in benchmark_tasks:
        if pairing_mode == "seen_one_to_one_unseen_current" and eval_task in UNSEEN_BENCHMARK_TASKS:
            if current_adapter_task:
                plan.setdefault(current_adapter_task, []).append(eval_task)
            continue
        adapter_task = adapter_task_for_benchmark(eval_task, adapter_tasks)
        if adapter_task:
            plan.setdefault(adapter_task, []).append(eval_task)
    return {task: unique_ordered(eval_tasks) for task, eval_tasks in plan.items()}


def normalize_dtype(args: argparse.Namespace) -> str:
    raw = str(getattr(args, "dtype", "") or getattr(args, "precision", "") or "bfloat16").lower()
    mapping = {
        "auto": "bfloat16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported Qwen dtype/precision: {raw}")
    return mapping[raw]


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


def task_prototype_vector(payload: Dict[str, Any], task: str, target_dim: int) -> torch.Tensor:
    prototypes = payload.get("task_prototypes") or {}
    proto = prototypes.get(task) if isinstance(prototypes, dict) else None
    if proto is None:
        raise KeyError(
            f"Task {task} is missing full task_prototypes in the EBM checkpoint. "
            "Please retrain with the Deepseek-style full-vector Meta-EBM pipeline."
        )
    proto = proto.detach().cpu().float().view(-1)
    if proto.numel() != int(target_dim):
        raise ValueError(
            f"Task prototype for {task} has length {proto.numel()}, expected {int(target_dim)}."
        )
    return proto


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


def resolve_project_paths(args: argparse.Namespace, payload: Dict[str, Any]) -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pami_code_root = os.path.abspath(os.path.join(script_dir, ".."))
    checkpoint_root = os.path.abspath(
        os.path.expanduser(
            getattr(args, "checkpoint_root", "")
            or os.environ.get("CHECKPOINT_ROOT", "")
            or payload.get("checkpoint_root")
            or os.path.join(pami_code_root, "CMA_ES", "QwenVL", "checkpoints")
        )
    )
    qwen_project_dir = os.path.abspath(
        os.path.expanduser(
            getattr(args, "qwen_project_dir", "")
            or getattr(args, "lora_project_dir", "")
            or os.path.join(pami_code_root, "LoRA_Qwen_VL")
        )
    )
    qwen_config_dir = os.path.abspath(
        os.path.expanduser(
            getattr(args, "qwen_config_dir", "")
            or getattr(args, "lora_config_dir", "")
            or os.path.join(qwen_project_dir, "config")
        )
    )
    return checkpoint_root, qwen_project_dir, qwen_config_dir


def qwen_config_path(qwen_config_dir: str, task: str) -> str:
    path = os.path.join(qwen_config_dir, f"{task}_lora_config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"QwenVL task config not found for task={task}: {path}")
    return path


def load_qwen_eval_config(qwen_config_dir: str, task: str) -> Dict[str, Any]:
    path = qwen_config_path(qwen_config_dir, task)
    payload = load_yaml_config(path)
    eval_cfg = dict(payload.get("checkpoint_eval") or {})
    if not eval_cfg:
        raise ValueError(f"{path} does not contain a checkpoint_eval block.")
    eval_cfg["config_path"] = path
    return eval_cfg


def load_benchmark_eval_config(qwen_config_dir: str, task: str) -> Dict[str, Any]:
    task = parse_benchmark_tasks(task, default=[])[0]
    if task in UNSEEN_EVAL_SPECS:
        cfg = dict(UNSEEN_EVAL_SPECS[task])
        cfg["config_path"] = ""
        return cfg
    eval_cfg = load_qwen_eval_config(qwen_config_dir, task)
    spec = TASK_EVAL_SPECS[task]
    eval_cfg["postprocess_task"] = spec["postprocess_task"]
    eval_cfg["score_module"] = spec["score_module"]
    eval_cfg["score_data_arg"] = spec["score_data_arg"]
    if task in IN_PROCESS_SCORE_TASKS:
        eval_cfg["score_kind"] = task
    return eval_cfg


def find_java_home(java_root: str) -> Optional[Tuple[str, str]]:
    root = os.path.abspath(os.path.expanduser(str(java_root or "")))
    if not os.path.isdir(root):
        return None
    names = ("java.exe", "java") if os.name == "nt" else ("java", "java.exe")
    for name in names:
        direct = os.path.join(root, "bin", name)
        if os.path.isfile(direct):
            return root, os.path.join(root, "bin")
    for current_root, _, files in os.walk(root):
        if os.path.basename(current_root) != "bin":
            continue
        for name in names:
            if name in files:
                return os.path.dirname(current_root), current_root
    return None


def add_java_to_env(env: Dict[str, str], java_roots: Sequence[str]) -> None:
    for java_root in java_roots:
        found = find_java_home(java_root)
        if found is None:
            continue
        java_home, java_bin = found
        env["JAVA_HOME"] = java_home
        env["PATH"] = java_bin + os.pathsep + env.get("PATH", "")
        return


def command_env(qwen_project_dir: str, cuda_device: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = qwen_project_dir + os.pathsep + env.get("PYTHONPATH", "")
    compat = "/usr/local/cuda-12.1/compat"
    env["LD_LIBRARY_PATH"] = compat + ":" + env.get("LD_LIBRARY_PATH", "")
    requested_cuda = str(cuda_device or "").strip()
    inherited_cuda = str(env.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if requested_cuda:
        if inherited_cuda and requested_cuda in {"0", "cuda:0"}:
            env["CUDA_VISIBLE_DEVICES"] = inherited_cuda
        else:
            env["CUDA_VISIBLE_DEVICES"] = requested_cuda

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    add_java_to_env(
        env,
        (
            env.get("JAVA_HOME", ""),
            os.path.join(script_dir, "resources", "java"),
            os.path.join(qwen_project_dir, "resources", "java"),
        ),
    )
    return env


def run_logged_command(
    command: Sequence[str],
    *,
    cwd: str,
    env: Dict[str, str],
    log_path: str,
    guard_root: Optional[str] = None,
) -> None:
    if guard_root:
        OutputDirGuard(guard_root).makedirs(os.path.dirname(log_path))
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("[cmd] " + " ".join(command) + "\n\n")
        handle.write(f"[env] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        handle.write(f"[env] PYTHONPATH={env.get('PYTHONPATH', '')}\n\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if guard_root:
        assert_result_dir_alive(guard_root)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}; see {log_path}")


def merge_answer_chunks(eval_dir: str, num_chunks: int) -> str:
    merged = os.path.join(eval_dir, "merge.jsonl")
    with open(merged, "w", encoding="utf-8") as out:
        for idx in range(max(1, int(num_chunks))):
            chunk = os.path.join(eval_dir, f"{num_chunks}_{idx}.jsonl")
            if not os.path.isfile(chunk):
                raise FileNotFoundError(f"Missing answer chunk: {chunk}")
            with open(chunk, "r", encoding="utf-8") as handle:
                for line in handle:
                    out.write(line)
    return merged


def parse_result_text(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    metrics: Dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z0-9_-]+):\s*([0-9]+(?:\.[0-9]+)?)%?", text):
        norm_key = key.strip().lower().replace("-", "_")
        try:
            metrics[norm_key] = float(value)
        except ValueError:
            continue
    return metrics


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("annotations"), list):
        return [item for item in payload["annotations"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported annotation format: {path}")


def annotation_answer_for_task(task: str, ann: Mapping[str, Any]) -> Any:
    if task == "grounding" and "answer_bbox" in ann:
        return ann.get("answer_bbox")
    for key in ("answer", "multiple_choice_answer", "label", "caption", "text"):
        if key in ann:
            return ann.get(key)
    answers = ann.get("answers")
    if isinstance(answers, list):
        return [item.get("answer", item) if isinstance(item, dict) else item for item in answers]
    if "answer_bbox" in ann:
        return ann.get("answer_bbox")
    return ""


def coco_caption_ground_truths(annotation_file: str) -> Optional[Tuple[List[Any], Dict[str, List[str]]]]:
    with open(annotation_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        return None
    image_ids = [image.get("id") for image in payload.get("images", []) if isinstance(image, dict) and "id" in image]
    captions: Dict[str, List[str]] = {}
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or "image_id" not in ann:
            continue
        captions.setdefault(str(ann.get("image_id")), []).append(str(ann.get("caption", "")))
    return image_ids, captions


def write_ans_gt_json(task: str, predictions_file: str, annotation_file: str, output_dir: str) -> str:
    predictions = load_json_records(predictions_file)
    output_path = os.path.join(output_dir, "ans_gt.json")
    records: List[Dict[str, Any]] = []
    caption_gt = None
    try:
        caption_gt = coco_caption_ground_truths(annotation_file)
    except Exception:
        caption_gt = None

    if caption_gt is not None:
        image_ids, captions = caption_gt
        for idx, pred in enumerate(predictions):
            image_id = pred.get("image_id")
            if image_id is None and idx < len(image_ids):
                image_id = image_ids[idx]
            if image_id is None:
                image_id = idx + 1
            records.append(
                {
                    "question_id": pred.get("question_id"),
                    "image_id": image_id,
                    "pred": pred.get("text", ""),
                    "ground_truth": captions.get(str(image_id), []),
                }
            )
    else:
        annotation_records = load_json_records(annotation_file)
        annotation_by_id = {str(item.get("question_id")): item for item in annotation_records}
        pair_by_order = task == "imagenet_r"
        for idx, pred in enumerate(predictions):
            ann = annotation_records[idx] if pair_by_order and idx < len(annotation_records) else annotation_by_id.get(str(pred.get("question_id")))
            records.append(
                {
                    "question_id": pred.get("question_id"),
                    "pred": pred.get("text", ""),
                    "ground_truth": annotation_answer_for_task(task, ann or {}),
                }
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    return output_path


def image_field_aliases(image_file: Any) -> List[str]:
    aliases: List[str] = []
    seen = set()

    def add(candidate: Any) -> None:
        text = str(candidate or "").strip()
        if text and text not in seen:
            seen.add(text)
            aliases.append(text)

    raw = str(image_file or "").strip()
    add(raw)
    normalized = raw.replace("\\", "/")
    if "/" in normalized:
        prefix, filename = normalized.rsplit("/", 1)
        prefix = prefix + "/"
    else:
        prefix, filename = "", normalized
    stem, ext = os.path.splitext(filename)
    if stem:
        if stem.lower().startswith("test_") and len(stem) > 5:
            add(prefix + stem[5:] + ext)
        else:
            add(prefix + "test_" + filename)
    return aliases


def image_root_candidates(image_folder: str, data_path: str = "") -> List[Path]:
    roots: List[Path] = []
    seen = set()

    def add_root(root: Optional[Path]) -> None:
        if root is None:
            return
        try:
            resolved = root.resolve()
        except FileNotFoundError:
            resolved = root
        key = str(resolved)
        if key and key not in seen:
            seen.add(key)
            roots.append(resolved)

    if image_folder:
        image_root = Path(str(image_folder))
        add_root(image_root)
        add_root(image_root / "process")
        for parent in image_root.parents:
            add_root(parent)
    if data_path:
        data_root = Path(str(data_path)).parent
        add_root(data_root)
        for parent in data_root.parents:
            add_root(parent)
    return roots


def resolve_image_path_alias(image_file: Any, image_folder: str, data_path: str = "") -> Optional[str]:
    aliases = image_field_aliases(image_file)
    for alias in aliases:
        alias_path = Path(alias)
        if alias_path.is_absolute() and alias_path.is_file():
            return str(alias_path)
    for root in image_root_candidates(image_folder, data_path):
        for alias in aliases:
            candidate = root / alias
            if candidate.is_file():
                return str(candidate)
    return None


def prepare_question_file_with_image_aliases(question_file: str, image_folder: str, output_dir: str) -> Tuple[str, int, int]:
    questions = load_json_records(question_file)
    if not questions:
        return question_file, 0, 0
    prepared: List[Dict[str, Any]] = []
    patched = 0
    unresolved = 0
    for item in questions:
        row = dict(item)
        image_file = row.get("image")
        if image_file:
            resolved = resolve_image_path_alias(image_file, image_folder, question_file)
            if resolved:
                if str(image_file) != resolved:
                    row["image"] = resolved
                    patched += 1
            else:
                unresolved += 1
        prepared.append(row)
    if patched <= 0:
        return question_file, patched, unresolved
    output_path = os.path.join(output_dir, "questions_resolved_images.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(prepared, handle, ensure_ascii=False, indent=2)
    return output_path, patched, unresolved


def load_json_payload(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_coco_caption_annotation_payload(annotation_file: str) -> Any:
    payload = load_json_payload(annotation_file)
    if not isinstance(payload, dict):
        return payload

    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        return payload

    normalized_payload = dict(payload)
    normalized_annotations = []
    seen_annotation_ids = set()
    has_missing_category_id = False
    has_missing_id = False
    has_duplicate_id = False
    categories = payload.get("categories")
    default_category_id = 1
    if isinstance(categories, list) and categories:
        first_category = categories[0]
        if isinstance(first_category, dict) and "id" in first_category:
            try:
                default_category_id = int(first_category["id"])
            except (TypeError, ValueError):
                default_category_id = 1

    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            normalized_annotations.append(annotation)
            continue

        normalized_annotation = dict(annotation)
        if "category_id" not in normalized_annotation:
            normalized_annotation["category_id"] = default_category_id
            has_missing_category_id = True
        annotation_id = normalized_annotation.get("id")
        if annotation_id is None:
            normalized_annotation["id"] = index
            has_missing_id = True
        elif annotation_id in seen_annotation_ids:
            normalized_annotation["id"] = index
            has_duplicate_id = True
        seen_annotation_ids.add(normalized_annotation["id"])
        normalized_annotations.append(normalized_annotation)

    if has_missing_category_id or not isinstance(categories, list) or not categories:
        normalized_payload["categories"] = [{"id": default_category_id, "name": "captioning"}]
    if has_missing_id or has_missing_category_id or has_duplicate_id:
        normalized_payload["annotations"] = normalized_annotations
    return normalized_payload


def normalize_coco_caption_annotation_file(annotation_file: str, output_dir: str) -> str:
    payload = normalize_coco_caption_annotation_payload(annotation_file)
    if not isinstance(payload, dict):
        return annotation_file
    output_path = os.path.join(output_dir, "annotation_coco_normalized.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
    return output_path


def subset_coco_caption_annotation_file(annotation_file: str, output_dir: str, num_predictions: int) -> str:
    payload = load_json_payload(annotation_file)
    if not isinstance(payload, dict) or num_predictions <= 0:
        return annotation_file
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return annotation_file
    if num_predictions >= len(images):
        return annotation_file

    subset_payload = dict(payload)
    subset_images = images[:num_predictions]
    image_ids = {item.get("id") for item in subset_images if isinstance(item, dict)}
    subset_payload["images"] = subset_images
    subset_payload["annotations"] = [
        item
        for item in annotations
        if isinstance(item, dict) and item.get("image_id") in image_ids
    ]
    output_path = os.path.join(output_dir, "annotation_coco_scoring.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(subset_payload, handle, ensure_ascii=True, indent=2)
    return output_path


def normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"^(the answer is|answer is|answer:|response:)\s*", "", text)
    text = text.strip(" \t\r\n.?!,:;\"'`*")
    text = re.sub(r"\s+", " ", text)
    return text


def first_choice_token(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = normalize_match_text(text)
    match = re.search(r"\boption\s*([A-Za-z]|\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*[\(\[]?\s*([A-Za-z]|\d+)(?:\s*[\)\]\.:\-]|\s*$|\b)", cleaned)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b(?:the\s+)?answer\s*(?:is|:|=|-)?\s*([A-Za-z]|\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r"^\s*[\(\[]?\s*([A-Za-z]|\d+)(?:\s*[\)\]\.:\-]|\s*$)", text)
    return match.group(1).upper() if match else ""


def choice_map(prompt: Any) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    for line in str(prompt or "").splitlines():
        match = re.match(r"^\s*([A-Za-z]|\d+)\s*[\.\)]\s*(.+?)\s*$", line)
        if match:
            choices[match.group(1).upper()] = normalize_match_text(match.group(2))
    return choices


def option_answer_match(pred_text: Any, ann: Mapping[str, Any]) -> bool:
    gt = str(ann.get("answer", "")).strip()
    pred_norm = normalize_match_text(pred_text)
    gt_norm = normalize_match_text(gt)
    if pred_norm == gt_norm:
        return True
    token = first_choice_token(pred_text)
    labels = {gt.upper()}
    if gt.isdigit():
        idx = int(gt)
        if 0 <= idx < 26:
            labels.add(chr(ord("A") + idx))
    if token and token in labels:
        return True
    choices = choice_map(ann.get("text", ""))
    option_text = choices.get(gt.upper())
    if option_text and (pred_norm == option_text or option_text in pred_norm):
        return True
    if gt.isdigit():
        label = chr(ord("A") + int(gt)) if int(gt) < 26 else ""
        option_text = choices.get(label)
        if option_text and (pred_norm == option_text or option_text in pred_norm):
            return True
    return False


def numbers_equal(pred_text: Any, answer_text: Any) -> bool:
    pred_numbers = re.findall(r"-?\d+(?:\.\d+)?", str(pred_text or "").replace(",", ""))
    answer_numbers = re.findall(r"-?\d+(?:\.\d+)?", str(answer_text or "").replace(",", ""))
    if not pred_numbers or not answer_numbers:
        return False
    try:
        return abs(float(pred_numbers[0]) - float(answer_numbers[0])) <= 1.0e-6
    except Exception:
        return False


def short_answer_match(pred_text: Any, answer_text: Any) -> bool:
    return numbers_equal(pred_text, answer_text) or normalize_match_text(pred_text) == normalize_match_text(answer_text)


_BBOX_RE = re.compile(
    r"[\[\(]\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*"
    r"[\]\)]"
)


def parse_bbox_numbers(value: Any) -> Optional[List[float]]:
    text = str(value or "")
    match = _BBOX_RE.search(text)
    if match:
        try:
            return [float(item) for item in match.groups()]
        except Exception:
            return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) < 4:
        return None
    try:
        return [float(item) for item in numbers[:4]]
    except Exception:
        return None


def normalize_xyxy_box(box: Sequence[float]) -> Optional[List[float]]:
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return None
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = min(1.0, max(0.0, left))
    top = min(1.0, max(0.0, top))
    right = min(1.0, max(0.0, right))
    bottom = min(1.0, max(0.0, bottom))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def annotation_size(ann: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    try:
        size = [float(x) for x in ann.get("size", [])]
    except Exception:
        return None
    if len(size) < 2 or size[0] <= 0 or size[1] <= 0:
        return None
    return size[0], size[1]


def square_padded_pixel_xyxy_to_unit(box: Sequence[float], size: Optional[Tuple[float, float]]) -> Optional[List[float]]:
    if size is None:
        return None
    raw = normalize_pixel_xyxy_box(box)
    if raw is None:
        return None
    width, height = size
    max_wh = max(width, height)
    x1, y1, x2, y2 = raw
    if width > height:
        pad = (width - height) / 2.0
        y1 += pad
        y2 += pad
    elif height > width:
        pad = (height - width) / 2.0
        x1 += pad
        x2 += pad
    return normalize_xyxy_box([x1 / max_wh, y1 / max_wh, x2 / max_wh, y2 / max_wh])


def normalize_pixel_xyxy_box(box: Sequence[float]) -> Optional[List[float]]:
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return None
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def add_grounding_candidate(candidates: List[List[float]], box: Optional[Sequence[float]]) -> None:
    if box is None:
        return
    normalized = normalize_xyxy_box(box)
    if normalized is None:
        return
    for existing in candidates:
        if all(abs(a - b) <= 1.0e-6 for a, b in zip(existing, normalized)):
            return
    candidates.append(normalized)


def grounding_box_candidates(box: Sequence[float], ann: Mapping[str, Any]) -> List[List[float]]:
    size = annotation_size(ann)
    candidates: List[List[float]] = []
    try:
        max_abs = max(abs(float(x)) for x in box)
    except Exception:
        return candidates

    if max_abs <= 1.5:
        add_grounding_candidate(candidates, box)
        return candidates

    # Qwen-VL commonly emits boxes in a 0-1000 square coordinate system.
    add_grounding_candidate(candidates, [float(x) / 1000.0 for x in box])

    if size is not None:
        max_wh = max(size)
        add_grounding_candidate(candidates, [float(x) / max_wh for x in box])
        add_grounding_candidate(candidates, square_padded_pixel_xyxy_to_unit(box, size))

    return candidates


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in box_a]
    bx1, by1, bx2, by2 = [float(x) for x in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def grounding_iou(pred_text: Any, ann: Mapping[str, Any]) -> float:
    pred_box = parse_bbox_numbers(pred_text)
    gt_box = parse_bbox_numbers(ann.get("answer_bbox", ""))
    if pred_box is None or gt_box is None:
        return 0.0
    gt_candidates = grounding_box_candidates(gt_box, ann)
    pred_candidates = grounding_box_candidates(pred_box, ann)
    if not gt_candidates or not pred_candidates:
        return 0.0
    return max(bbox_iou(pred, gt) for pred in pred_candidates for gt in gt_candidates)


def grounding_match(pred_text: Any, ann: Mapping[str, Any]) -> bool:
    return grounding_iou(pred_text, ann) > 0.5


def score_benchmark_in_process(
    task: str,
    answers_file: str,
    annotation_file: str,
    output_dir: str,
    guard_root: Optional[str] = None,
) -> Dict[str, float]:
    annotation_records = load_json_records(annotation_file)
    annotations = {str(item.get("question_id")): item for item in annotation_records}
    predictions = load_json_records(answers_file)
    total = 0
    correct = 0
    false_answers: List[Dict[str, Any]] = []
    if task == "imagenet_r":
        # ImageNet-R filenames repeat across class directories, so question_id is not unique.
        paired_records = zip(predictions, annotation_records)
    else:
        paired_records = ((pred, annotations.get(str(pred.get("question_id")))) for pred in predictions)
    for pred, ann in paired_records:
        if ann is None:
            continue
        total += 1
        pred_text = str(pred.get("text", ""))
        answer = ann.get("answer", "")
        grounding_iou_value: Optional[float] = None
        if task in {"scienceqa", "aokvqa"}:
            ok = option_answer_match(pred_text, ann)
        elif task in {"imagenet", "imagenet_r"}:
            gt = str(answer)
            ok = bool(gt and pred_text) and ((gt.upper() in pred_text.upper()) or (pred_text.upper() in gt.upper()))
        elif task == "tabmwp":
            ok = short_answer_match(pred_text, answer)
        elif task == "grounding":
            grounding_iou_value = grounding_iou(pred_text, ann)
            ok = grounding_iou_value > 0.5
        elif task in {"iconqa", "vqav2"}:
            ok = pred_text.upper() == str(answer).upper()
        elif task == "ocr_vqa":
            ok = "Unanswerable" not in pred_text and pred_text.lower() == str(answer).lower()
        else:
            raise ValueError(f"No in-process scorer for task={task}")
        if ok:
            correct += 1
        else:
            row = dict(pred)
            row["ground_truth"] = ann.get("answer_bbox", "") if task == "grounding" else answer
            if grounding_iou_value is not None:
                row["grounding_iou"] = round(grounding_iou_value, 6)
            false_answers.append(row)
    if guard_root:
        OutputDirGuard(guard_root).makedirs(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)
    accuracy = 0.0 if total == 0 else 100.0 * correct / total
    with open(os.path.join(output_dir, "Result.text"), "w", encoding="utf-8") as handle:
        handle.write(f"Samples: {total}\nAccuracy: {accuracy:.2f}%\n")
    with open(os.path.join(output_dir, "false_answers.json"), "w", encoding="utf-8") as handle:
        json.dump({"items": false_answers[:1000], "num_false": len(false_answers)}, handle, ensure_ascii=False, indent=2)
    return {"samples": float(total), "accuracy": float(accuracy)}


def evaluate_generated_adapter(
    adapter_task: str,
    eval_task: str,
    adapter_dir: str,
    task_output_dir: str,
    qwen_project_dir: str,
    qwen_config_dir: str,
    args: argparse.Namespace,
    guard_root: Optional[str] = None,
) -> Dict[str, Any]:
    eval_task = parse_benchmark_tasks(eval_task, default=[])[0]
    eval_cfg = load_benchmark_eval_config(qwen_config_dir, eval_task)
    eval_dir = os.path.join(task_output_dir, "eval", eval_task)
    if guard_root:
        OutputDirGuard(guard_root).makedirs(eval_dir)
    else:
        os.makedirs(eval_dir, exist_ok=True)

    question_file = str(eval_cfg["question_file"])
    annotation_file = str(eval_cfg.get("annotation_file") or question_file)
    image_folder = str(eval_cfg["image_folder"])
    question_file_for_inference, patched_images, unresolved_images = prepare_question_file_with_image_aliases(
        question_file,
        image_folder,
        eval_dir,
    )
    if patched_images:
        print(f"[info] resolved {patched_images} image paths with Qwen-style aliases for {eval_task}")
    if unresolved_images:
        print(f"[warn] {unresolved_images} image paths were not pre-resolved for {eval_task}; falling back to Qwen resolver")
    max_samples = getattr(args, "max_samples", None)
    if max_samples is None:
        cfg_max = int(eval_cfg.get("max_samples") or 0)
        max_samples = cfg_max if cfg_max > 0 else None

    model_base = (
        getattr(args, "model_base", "")
        or getattr(args, "model_path", "")
        or os.environ.get("QWEN3_VL_BASE", "")
        or str((load_yaml_config(qwen_config_path(qwen_config_dir, adapter_task)).get("model") or {}).get("model_name_or_path", ""))
    )
    if not model_base:
        raise ValueError("Qwen base model path is empty. Set --model-base or QWEN3_VL_BASE.")

    python_bin = getattr(args, "python_bin", "") or sys.executable
    dtype = normalize_dtype(args)
    num_chunks = max(1, int(getattr(args, "num_chunks", 1) or 1))
    env = command_env(qwen_project_dir, str(getattr(args, "cuda_device", "") or ""))
    wrapper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwenvl_eval_adapter.py")
    for idx in range(num_chunks):
        answers = os.path.join(eval_dir, f"{num_chunks}_{idx}.jsonl")
        cmd = [
            python_bin,
            wrapper,
            "--model-path",
            adapter_dir,
            "--model-base",
            model_base,
            "--question-file",
            question_file_for_inference,
            "--image-folder",
            image_folder,
            "--answers-file",
            answers,
            "--num-chunks",
            str(num_chunks),
            "--chunk-idx",
            str(idx),
            "--temperature",
            str(eval_cfg.get("temperature", 0.0)),
            "--num-beams",
            str(eval_cfg.get("num_beams", 1)),
            "--max-new-tokens",
            str(eval_cfg.get("max_new_tokens", 128)),
            "--dtype",
            dtype,
            "--attn-impl",
            str(getattr(args, "attn_impl", "") or "flash_attention_2"),
            "--device",
            str(getattr(args, "qwen_device", "") or "cuda"),
            "--task",
            str(eval_cfg["postprocess_task"]),
        ]
        if max_samples is not None:
            cmd.extend(["--max-samples", str(int(max_samples))])
        run_logged_command(
            cmd,
            cwd=qwen_project_dir,
            env=env,
            log_path=os.path.join(eval_dir, f"inference_chunk_{idx}.log"),
            guard_root=guard_root,
        )

    merged = merge_answer_chunks(eval_dir, num_chunks)
    ans_gt_file = write_ans_gt_json(eval_task, merged, annotation_file, eval_dir)
    score_kind = str(eval_cfg.get("score_kind", "") or "")
    if score_kind in IN_PROCESS_SCORE_TASKS:
        metrics = score_benchmark_in_process(eval_task, merged, annotation_file, eval_dir, guard_root=guard_root)
    else:
        score_annotation_file = annotation_file
        if score_kind == "caption":
            score_annotation_file = normalize_coco_caption_annotation_file(annotation_file, eval_dir)
            score_annotation_file = subset_coco_caption_annotation_file(
                score_annotation_file,
                eval_dir,
                len(load_json_records(merged)),
            )
        score_cmd = [
            python_bin,
            "-m",
            str(eval_cfg["score_module"]),
            str(eval_cfg["score_data_arg"]),
            score_annotation_file,
            "--result-file",
            merged,
            "--output-dir",
            eval_dir,
        ]
        run_logged_command(
            score_cmd,
            cwd=qwen_project_dir,
            env=env,
            log_path=os.path.join(eval_dir, "scoring.log"),
            guard_root=guard_root,
        )
        metrics = parse_result_text(os.path.join(eval_dir, "Result.text"))

    result_text = os.path.join(eval_dir, "Result.text")
    return {
        "adapter_task": adapter_task,
        "eval_task": eval_task,
        "dataset_group": "unseen" if eval_task in UNSEEN_BENCHMARK_TASKS else "seen",
        "config_path": eval_cfg["config_path"],
        "question_file": question_file,
        "annotation_file": annotation_file,
        "image_folder": image_folder,
        "answers_file": merged,
        "answers_gt_file": ans_gt_file,
        "result_text": result_text,
        "output_dir": eval_dir,
        "metrics": metrics,
    }


def save_aggregate(output_root: str, rows: Sequence[Dict[str, Any]], guard_root: Optional[str] = None) -> Tuple[str, str]:
    if guard_root:
        OutputDirGuard(guard_root).makedirs(output_root)
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
        "adapter_task",
        "eval_task",
        "dataset_group",
        "sample",
        "status",
        "source_checkpoint",
        "source_score",
        "adapter_dir",
        "sample_dir",
        "output_dir",
    ] + metric_keys + ["error"]
    tsv_path = os.path.join(output_root, "summary.tsv")
    with open(tsv_path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            values = [
                str(row.get("task", "")),
                str(row.get("adapter_task", "")),
                str(row.get("eval_task", "")),
                str(row.get("dataset_group", "")),
                str(row.get("sample", "")),
                str(row.get("status", "")),
                str(row.get("source_checkpoint", "")),
                str(row.get("source_score", "")),
                str(row.get("adapter_dir", "")),
                str(row.get("sample_dir", "")),
                str(row.get("output_dir", "")),
            ]
            values.extend(str(row.get("metrics", {}).get(key, "")) for key in metric_keys)
            values.append(str(row.get("error", "")))
            handle.write("\t".join(values) + "\n")
    return json_path, tsv_path


def run_evaluation(args: argparse.Namespace) -> Dict[str, str]:
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.abspath(os.path.expanduser(args.ebm_checkpoint))
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model, payload = load_ebm(checkpoint_path, device=device)

    tasks = parse_task_order(args.tasks) if getattr(args, "tasks", "") else list(payload["tasks"])
    benchmark_tasks = parse_benchmark_tasks(getattr(args, "benchmark_tasks", ""))
    pairing_mode = str(getattr(args, "pairing_mode", "cartesian") or "cartesian")
    current_task = str(getattr(args, "current_task", "") or "")
    eval_plan = build_eval_plan(tasks, benchmark_tasks, pairing_mode=pairing_mode, current_task=current_task)
    checkpoint_root, qwen_project_dir, qwen_config_dir = resolve_project_paths(args, payload)
    output_root = os.path.abspath(getattr(args, "output_dir", "") or os.path.join(script_dir, "results", "eval"))
    parent_result_root = getattr(args, "parent_result_root", "")
    if parent_result_root:
        assert_result_dir_alive(parent_result_root, "parent training result directory")
    run_name = getattr(args, "run_name", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(output_root, run_name)
    os.makedirs(run_root, exist_ok=True)
    result_guard = OutputDirGuard(run_root, "evaluation result directory", initialize=True)
    result_guard.assert_alive()
    print(f"[info] adapter_tasks={tasks}")
    print(f"[info] benchmark_tasks={benchmark_tasks}")
    print(f"[info] pairing_mode={pairing_mode} current_task={current_task or (tasks[-1] if tasks else '')}")
    print(f"[info] eval_plan={eval_plan}")

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
                    init_noise_std=args.init_noise_std if sample_idx > 0 else 0.0,
                    save_safetensors=bool(args.save_safetensors),
                    guard_root=result_guard.root,
                )
                adapter_row = {
                    "task": task,
                    "adapter_task": task,
                    "eval_task": "",
                    "dataset_group": "",
                    "sample": sample_idx,
                    "status": "exported",
                    "source_checkpoint": source_spec.path,
                    "source_score": source_spec.score,
                    "source_metric": source_spec.metric_name,
                    "adapter_dir": exported_dir,
                    "sample_dir": sample_dir,
                    "output_dir": sample_dir,
                    "metrics": {},
                }
                eval_rows: List[Dict[str, Any]] = []
                if getattr(args, "skip_eval", False):
                    rows.append(adapter_row)
                else:
                    for eval_task in eval_plan.get(task, []):
                        result_guard.assert_alive()
                        row = dict(adapter_row)
                        row.update(
                            {
                                "eval_task": eval_task,
                                "dataset_group": "unseen" if eval_task in UNSEEN_BENCHMARK_TASKS else "seen",
                                "output_dir": os.path.join(sample_dir, "eval", eval_task),
                            }
                        )
                        try:
                            eval_payload = evaluate_generated_adapter(
                                adapter_task=task,
                                eval_task=eval_task,
                                adapter_dir=exported_dir,
                                task_output_dir=sample_dir,
                                qwen_project_dir=qwen_project_dir,
                                qwen_config_dir=qwen_config_dir,
                                args=args,
                                guard_root=result_guard.root,
                            )
                            row.update(eval_payload)
                            row["status"] = "ok"
                        except Exception as exc:
                            row["status"] = "failed"
                            row["error"] = str(exc)
                            print(f"[failed] adapter_task={task} eval_task={eval_task}: {exc}")
                            if getattr(args, "fail_fast", False):
                                raise
                        rows.append(row)
                        eval_rows.append(row)
                        print(
                            f"[{row['status']}] adapter_task={task} eval_task={eval_task} "
                            f"sample={sample_idx} adapter={exported_dir}"
                        )
                sample_summary = dict(adapter_row)
                sample_summary["eval_results"] = eval_rows
                result_guard.assert_alive()
                with open(os.path.join(sample_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
                    json.dump(sample_summary, handle, ensure_ascii=False, indent=2)
                if getattr(args, "skip_eval", False):
                    print(f"[exported] adapter_task={task} sample={sample_idx} adapter={exported_dir}")
        except Exception as exc:
            failure = {
                "task": task,
                "adapter_task": task,
                "eval_task": "",
                "dataset_group": "",
                "sample": "",
                "status": "failed",
                "source_checkpoint": "",
                "adapter_dir": "",
                "sample_dir": "",
                "output_dir": os.path.join(run_root, task),
                "metrics": {},
                "error": str(exc),
            }
            result_guard.makedirs(failure["output_dir"])
            with open(os.path.join(failure["output_dir"], "run_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False, indent=2)
            rows.append(failure)
            print(f"[failed] task={task}: {exc}")
            if getattr(args, "fail_fast", False):
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
