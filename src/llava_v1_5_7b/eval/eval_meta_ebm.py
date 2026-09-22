#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401 - configure the grouped source layout
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from checkpointing import load_task_checkpoint
from llava_adapter_data import (
    CheckpointSpec,
    discover_task_checkpoints,
    load_adapter_state,
    parse_task_order,
    reconstruct_lora_state,
    save_generated_adapter_checkpoint,
)
from models.meta_ebm import MetaEBM

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


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

BENCHMARK_TASK_ALIASES = {
    "science_qa": "scienceqa",
    "image_net": "imagenet",
    "ocrvqa": "ocr_vqa",
    "ocr-vqa": "ocr_vqa",
    "rec_coco": "grounding",
    "rec-coco": "grounding",
    "aok_vqa": "aokvqa",
    "aok-vqa": "aokvqa",
    "imagenet-r": "imagenet_r",
    "imagenet_r": "imagenet_r",
    "screen_2_words": "screen2words",
    "screen-2-words": "screen2words",
    "tab_mwp": "tabmwp",
    "tab-mwp": "tabmwp",
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
    parser = argparse.ArgumentParser(description="Refine LLaVA LoRA adapters with a trained Meta-EBM and run MM-MergeBench eval")
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
    parser.add_argument("--llava-project-dir", "--lora-project-dir", dest="llava_project_dir", type=str, default="")
    parser.add_argument("--llava-config-dir", "--lora-config-dir", dest="llava_config_dir", type=str, default="")
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
    parser.add_argument("--init-noise-std", type=float, default=0.0)
    parser.add_argument("--save-safetensors", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument("--python-bin", type=str, default="")
    parser.add_argument("--model-base", "--model-path", dest="model_base", type=str, default="")
    parser.add_argument("--llava-device", type=str, default="cuda")
    parser.add_argument("--cuda-device", type=str, default="")
    parser.add_argument("--use-flash-attn", type=str2bool, default=False)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")

    # Accepted for compatibility with train_meta_ebm.py auto-eval args.
    parser.add_argument("--precision", type=str, default="")
    parser.add_argument("--eval-split", type=str, default="")
    parser.add_argument("--dataset-source", type=str, default="")
    parser.add_argument("--per-device-batch-size", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--trust-remote-code", type=str2bool, default=False)

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
    if eval_task in adapter_tasks:
        return eval_task
    if eval_task == "grounding" and "rec_coco" in adapter_tasks:
        return "rec_coco"
    return ""


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
) -> str:
    template = payload["template"]
    normalize_mode = payload.get("normalize_mode", "per_tensor")
    eps = float(payload.get("normalization_eps", 1e-6))
    task_embeddings = payload["task_embeddings"]
    if task not in task_embeddings:
        raise KeyError(f"Task {task} is not in the EBM checkpoint task embeddings.")

    prototypes = payload.get("task_prototypes") or {}
    if task not in prototypes:
        raise KeyError(f"Task {task} is missing a full task prototype; retrain the EBM with the main method.")
    init_vec = prototypes[task].detach().float().cpu().view(1, -1)
    if init_vec.size(1) != int(template["input_dim"]):
        raise ValueError(f"Prototype dimension mismatch for task {task}: {init_vec.size(1)} vs {template['input_dim']}")
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
    )


def resolve_project_paths(args: argparse.Namespace, payload: Dict[str, Any]) -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pami_root = os.path.abspath(os.path.join(script_dir, ".."))
    checkpoint_root = os.path.abspath(
        os.path.expanduser(
            getattr(args, "checkpoint_root", "")
            or os.environ.get("CHECKPOINT_ROOT", "")
            or payload.get("checkpoint_root")
            or os.path.join(pami_root, "CMA_ES", "LLAVA", "checkpoints")
        )
    )
    llava_project_dir = os.path.abspath(
        os.path.expanduser(
            getattr(args, "llava_project_dir", "")
            or getattr(args, "lora_project_dir", "")
            or os.path.join(pami_root, "LoRA_LLAVA")
        )
    )
    llava_config_dir = os.path.abspath(
        os.path.expanduser(
            getattr(args, "llava_config_dir", "")
            or getattr(args, "lora_config_dir", "")
            or os.path.join(llava_project_dir, "config")
        )
    )
    return checkpoint_root, llava_project_dir, llava_config_dir


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


def command_env(llava_project_dir: str, cuda_device: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = llava_project_dir + os.pathsep + env.get("PYTHONPATH", "")
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
            os.path.join(llava_project_dir, "resources", "java"),
        ),
    )
    return env


def run_logged_command(command: Sequence[str], *, cwd: str, env: Dict[str, str], log_path: str) -> None:
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
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}; see {log_path}")


def evaluate_generated_adapter(
    adapter_task: str,
    eval_task: str,
    adapter_dir: str,
    task_output_dir: str,
    llava_project_dir: str,
    llava_config_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    eval_task = parse_benchmark_tasks(eval_task, default=[])[0]
    eval_dir = os.path.join(task_output_dir, "eval", eval_task)
    os.makedirs(eval_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner = os.path.join(script_dir, "eval", "llava_eval_adapter.py")
    python_bin = getattr(args, "python_bin", "") or os.environ.get("PYTHON_BIN", "") or sys.executable
    metrics_path = os.path.join(eval_dir, "metrics.json")
    predictions_path = os.path.join(eval_dir, "merge.jsonl")

    cmd = [
        python_bin,
        runner,
        "--task",
        eval_task,
        "--adapter-dir",
        adapter_dir,
        "--llava-project-dir",
        llava_project_dir,
        "--llava-config-dir",
        llava_config_dir,
        "--output-dir",
        eval_dir,
        "--metrics-file",
        metrics_path,
        "--predictions-file",
        predictions_path,
        "--device",
        str(getattr(args, "llava_device", "") or "cuda"),
        "--num-chunks",
        str(max(1, int(getattr(args, "num_chunks", 1) or 1))),
    ]
    model_base = getattr(args, "model_base", "") or getattr(args, "model_path", "")
    if model_base:
        cmd.extend(["--model-base", model_base])
    if getattr(args, "max_samples", None) is not None:
        cmd.extend(["--max-samples", str(int(args.max_samples))])
    if getattr(args, "use_flash_attn", False):
        cmd.append("--use-flash-attn")

    env = command_env(llava_project_dir, str(getattr(args, "cuda_device", "") or ""))
    run_logged_command(cmd, cwd=llava_project_dir, env=env, log_path=os.path.join(eval_dir, "llava_eval.log"))

    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {
        "adapter_task": adapter_task,
        "eval_task": eval_task,
        "dataset_group": "unseen" if eval_task in UNSEEN_BENCHMARK_TASKS else "seen",
        "metrics": metrics,
        "metrics_file": metrics_path,
        "predictions_file": predictions_path,
        "answers_gt_file": os.path.join(eval_dir, "ans_gt.json"),
        "result_text": os.path.join(eval_dir, "Result.text"),
        "output_dir": eval_dir,
    }


def save_aggregate(output_root: str, rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
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
    checkpoint_root, llava_project_dir, llava_config_dir = resolve_project_paths(args, payload)
    output_root = os.path.abspath(getattr(args, "output_dir", "") or os.path.join(script_dir, "results", "eval"))
    run_name = getattr(args, "run_name", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(output_root, run_name)
    os.makedirs(run_root, exist_ok=True)
    print(f"[info] adapter_tasks={tasks}")
    print(f"[info] benchmark_tasks={benchmark_tasks}")
    print(f"[info] pairing_mode={pairing_mode} current_task={current_task or (tasks[-1] if tasks else '')}")
    print(f"[info] eval_plan={eval_plan}")

    task_to_specs = discover_task_checkpoints(tasks, checkpoint_root=checkpoint_root, topk_per_task=max(1, args.source_index + 1))
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        try:
            source_spec = choose_source_checkpoint(task_to_specs[task], args.source_index)
            for sample_idx in range(max(1, int(args.num_samples))):
                sample_dir = os.path.join(run_root, task, f"sample_{sample_idx:02d}")
                adapter_dir = os.path.join(sample_dir, "llava_lora_adapter")
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
                                llava_project_dir=llava_project_dir,
                                llava_config_dir=llava_config_dir,
                                args=args,
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
            os.makedirs(failure["output_dir"], exist_ok=True)
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

    summary_json, summary_tsv = save_aggregate(run_root, rows)
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
