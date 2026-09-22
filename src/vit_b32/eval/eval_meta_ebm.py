import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401 - configure the grouped source layout
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml
from tqdm.auto import tqdm

from data_pipeline import pad_vector, safe_torch_load
from task_setup import default_source_checkpoint, default_task_config


def _ensure_lora_module_importable(project_root: str):
    lora_root = os.path.join(project_root, "models", "LoRA_ViT")
    if lora_root not in sys.path:
        sys.path.insert(0, lora_root)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_trainable_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("trainable_state_dict", "state_dict", "model_state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        maybe = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if maybe:
            return maybe
    raise ValueError("Unable to extract a trainable state dict from checkpoint.")


def build_template_and_dims(normalized_file: str):
    template = safe_torch_load(normalized_file)
    if "data" not in template:
        raise KeyError(f"{normalized_file} is missing the data field.")

    keys = [
        key
        for key, value in template.items()
        if isinstance(value, dict) and {"mean", "std", "length", "shape"}.issubset(value.keys())
    ]
    keys = sorted(keys)
    input_dim = int(template["data"].numel())
    return template, keys, input_dim


def reconstruct_state_dict(
    generated_vec: torch.Tensor,
    template: dict,
    keys: List[str],
    source_state: Dict[str, torch.Tensor],
):
    lengths = [int(template[key]["length"]) for key in keys]
    total = sum(lengths)

    vec = generated_vec.view(-1)
    if vec.numel() < total:
        raise ValueError(f"Generated vector length {vec.numel()} is shorter than expected {total}")
    if vec.numel() > total:
        vec = vec[:total]

    chunks = torch.split(vec, lengths)
    out = {}
    for key, chunk in zip(keys, chunks):
        if key not in source_state:
            raise KeyError(f"Source checkpoint is missing LoRA tensor {key}")
        source_tensor = source_state[key].detach().float()
        mean = float(source_tensor.mean())
        std = float(source_tensor.std(unbiased=False).clamp_min(1e-6))
        shape = tuple(template[key]["shape"])
        restored = chunk * std + mean
        out[key] = restored.reshape(shape).float()
    return out


def merge_with_source_checkpoint(generated_state: dict, source_checkpoint: str):
    if not source_checkpoint or not os.path.exists(source_checkpoint):
        return generated_state
    src = safe_torch_load(source_checkpoint)
    src_state = extract_trainable_state_dict(src)
    merged = dict(src_state)
    merged.update(generated_state)
    return merged


def inject_trainable_params(model, trainable_state: dict):
    model_sd = model.state_dict()
    loaded = 0
    skipped = []
    for key, value in trainable_state.items():
        if key in model_sd and tuple(model_sd[key].shape) == tuple(value.shape):
            model_sd[key] = value.to(device=model_sd[key].device, dtype=model_sd[key].dtype)
            loaded += 1
        else:
            skipped.append(key)
    model.load_state_dict(model_sd, strict=False)
    return loaded, skipped


@torch.no_grad()
def evaluate_task_accuracy(
    model,
    test_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_batches: int = 0,
    show_progress: bool = False,
    desc: str = "",
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    ds = datasets.ImageFolder(root=test_dir, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    model.eval()

    total = 0
    total_loss = 0.0
    total_correct = 0
    iterator = dl
    if show_progress:
        iterator = tqdm(dl, desc=desc or f"eval:{os.path.basename(test_dir)}", leave=False, dynamic_ncols=True)

    for batch_idx, (images, labels) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        pred = torch.argmax(logits, dim=1)
        total_correct += int((pred == labels).sum().item())

        if max_batches > 0 and batch_idx >= max_batches:
            break

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc, total


class MultiTaskEBMEvaluator:
    def __init__(
        self,
        project_root: str,
        tasks: List[str],
        task_to_files: Dict[str, List[str]],
        model_input_dim: int = 0,
        eval_batch_size: int = 256,
        eval_num_workers: int = 4,
        eval_num_samples: int = 1,
        eval_aggregate: str = "best",
        eval_max_batches: int = 0,
        eval_refine_steps: int = 5,
        eval_refine_lr: float = 1e-3,
        eval_refine_beta: float = 1.0,
        eval_init_noise_std: float = 0.0,
        eval_refine_grad_clip: float = 0.0,
        eval_refine_langevin_std: float = 0.0,
        seen_init_mode: str = "prototype",
        unseen_init_mode: str = "zero",
        device: str = "cuda",
        show_progress: bool = True,
    ):
        self.project_root = project_root
        self.tasks = tasks
        self.task_to_files = task_to_files
        self.model_input_dim = int(model_input_dim)
        self.eval_batch_size = eval_batch_size
        self.eval_num_workers = eval_num_workers
        self.eval_num_samples = eval_num_samples
        self.eval_aggregate = eval_aggregate
        self.eval_max_batches = eval_max_batches
        self.eval_refine_steps = eval_refine_steps
        self.eval_refine_lr = eval_refine_lr
        self.eval_refine_beta = float(eval_refine_beta)
        self.eval_init_noise_std = eval_init_noise_std
        self.eval_refine_grad_clip = eval_refine_grad_clip
        self.eval_refine_langevin_std = float(eval_refine_langevin_std)
        self.seen_init_mode = seen_init_mode
        self.unseen_init_mode = unseen_init_mode
        self.show_progress = show_progress
        self.task_to_index = {task: idx for idx, task in enumerate(self.tasks)}
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

        _ensure_lora_module_importable(project_root)
        from lora_vit_model import LoRAViTClassifier

        self.LoRAViTClassifier = LoRAViTClassifier
        self.task_runtime: Dict[str, dict] = {}
        self._prepare_task_runtime()
        if self.model_input_dim <= 0:
            self.model_input_dim = max(int(rt["input_dim"]) for rt in self.task_runtime.values())

    def _prepare_task_runtime(self):
        for task in self.tasks:
            sample_file = self.task_to_files[task][0]
            template, keys, input_dim = build_template_and_dims(sample_file)

            cfg_path = default_task_config(self.project_root, task)
            src_ckpt = default_source_checkpoint(self.project_root, task)
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Task config does not exist: {cfg_path}")
            if not os.path.exists(src_ckpt):
                raise FileNotFoundError(f"Task source checkpoint does not exist: {src_ckpt}")

            cfg = load_yaml(cfg_path)
            model = self.LoRAViTClassifier(cfg).to(self.device)
            self.task_runtime[task] = {
                "template": template,
                "keys": keys,
                "input_dim": input_dim,
                "sample_file": sample_file,
                "cfg": cfg,
                "source_ckpt": src_ckpt,
                "source_state": extract_trainable_state_dict(safe_torch_load(src_ckpt)),
                "model": model,
            }

    def _build_init_batch(
        self,
        task: str,
        task_prototypes: Dict[str, torch.Tensor],
        max_seen_task_idx: int,
    ) -> Tuple[torch.Tensor, str]:
        task_idx = self.task_to_index[task]
        init_mode = self.unseen_init_mode
        task_input_dim = int(self.task_runtime[task]["input_dim"])
        if self.model_input_dim < task_input_dim:
            raise ValueError(
                f"model_input_dim={self.model_input_dim} is smaller than task_input_dim={task_input_dim} for task={task}"
            )
        base = torch.zeros(self.model_input_dim, dtype=torch.float32)

        if task_idx <= max_seen_task_idx:
            if self.seen_init_mode == "prototype" and task in task_prototypes:
                init_mode = "prototype"
                base = pad_vector(task_prototypes[task], self.model_input_dim)
            elif self.seen_init_mode == "zero":
                init_mode = "zero"
        elif self.unseen_init_mode == "prototype" and task in task_prototypes:
            init_mode = "prototype"
            base = pad_vector(task_prototypes[task], self.model_input_dim)

        batch = base.view(1, -1).repeat(self.eval_num_samples, 1)
        if self.eval_init_noise_std > 0.0:
            batch = batch + torch.randn_like(batch) * self.eval_init_noise_std
        return batch, init_mode

    @torch.no_grad()
    def _evaluate_one_task(
        self,
        ebm_model,
        task: str,
        condition_vector: torch.Tensor,
        task_prototypes: Dict[str, torch.Tensor],
        max_seen_task_idx: int,
    ) -> Tuple[float, Dict]:
        rt = self.task_runtime[task]
        cfg = rt["cfg"]
        model = rt["model"]
        template = rt["template"]
        keys = rt["keys"]
        source_ckpt = rt["source_ckpt"]

        init_batch, init_mode = self._build_init_batch(task, task_prototypes, max_seen_task_idx)
        cond = condition_vector.view(1, -1).repeat(self.eval_num_samples, 1).to(self.device)
        refined = ebm_model.refine(
            init_batch.to(self.device),
            cond,
            steps=self.eval_refine_steps,
            step_size=self.eval_refine_lr,
            grad_clip=self.eval_refine_grad_clip,
            beta=self.eval_refine_beta,
            langevin_noise_std=self.eval_refine_langevin_std,
        ).cpu()

        sample_metrics = []
        sample_iter = range(self.eval_num_samples)
        if self.show_progress and self.eval_num_samples > 1:
            sample_iter = tqdm(sample_iter, desc=f"refine-eval:{task}", leave=False, dynamic_ncols=True)

        for sample_idx in sample_iter:
            gen_vec = refined[sample_idx]
            gen_state = reconstruct_state_dict(gen_vec, template, keys, rt["source_state"])
            merged_state = merge_with_source_checkpoint(gen_state, source_ckpt)
            loaded, skipped = inject_trainable_params(model, merged_state)

            test_dir = cfg["data"]["test_dir"]
            image_size = int(cfg["data"]["image_size"])
            loss, acc, num_test = evaluate_task_accuracy(
                model=model,
                test_dir=test_dir,
                image_size=image_size,
                batch_size=self.eval_batch_size,
                num_workers=self.eval_num_workers,
                device=self.device,
                max_batches=self.eval_max_batches,
                show_progress=self.show_progress,
                desc=f"test:{task}",
            )
            sample_metrics.append(
                {
                    "sample": sample_idx + 1,
                    "loss": loss,
                    "acc": acc,
                    "num_test": num_test,
                    "loaded_params": loaded,
                    "skipped_params": len(skipped),
                    "init_mode": init_mode,
                    "refine_steps": self.eval_refine_steps,
                    "refine_beta": self.eval_refine_beta,
                    "refine_langevin_std": self.eval_refine_langevin_std,
                }
            )

        if self.eval_aggregate == "mean":
            agg_acc = sum(item["acc"] for item in sample_metrics) / len(sample_metrics)
        else:
            agg_acc = max(item["acc"] for item in sample_metrics)

        detail = {
            "aggregate": self.eval_aggregate,
            "task": task,
            "aggregate_acc": agg_acc,
            "init_mode": init_mode,
            "samples": sample_metrics,
        }
        return agg_acc, detail

    @torch.no_grad()
    def evaluate_all(
        self,
        ebm_model,
        task_vectors: Dict[str, torch.Tensor],
        task_prototypes: Dict[str, torch.Tensor],
        max_seen_task_idx: int,
    ) -> Tuple[List[float], Dict[str, Dict]]:
        ebm_model.eval()
        accs: List[float] = []
        details: Dict[str, Dict] = {}
        task_iter = self.tasks
        if self.show_progress:
            task_iter = tqdm(self.tasks, desc="eval-tasks", leave=False, dynamic_ncols=True)

        for task in task_iter:
            acc, detail = self._evaluate_one_task(
                ebm_model=ebm_model,
                task=task,
                condition_vector=task_vectors[task],
                task_prototypes=task_prototypes,
                max_seen_task_idx=max_seen_task_idx,
            )
            accs.append(float(acc))
            details[task] = detail
        return accs, details


def dump_eval_details(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
