from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from checkpoint_pool import select_validation_topk
import torch.nn.functional as F
from torch.utils.data import Dataset


ADAPTER_WEIGHT_FILENAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def safe_torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading adapter_model.safetensors requires safetensors. "
            "Install it in the LoRA conda env or use checkpoints with adapter_model.bin."
        ) from exc
    return load_file(path, device="cpu")


def _save_safetensors(path: str, state: Dict[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Writing adapter_model.safetensors requires safetensors. "
            "Use save_safetensors=False to write adapter_model.bin instead."
        ) from exc
    save_file(state, path)


def find_adapter_weight_file(checkpoint_dir: str) -> str:
    for file_name in ADAPTER_WEIGHT_FILENAMES:
        path = os.path.join(checkpoint_dir, file_name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No adapter weights found under {checkpoint_dir}")


def load_adapter_state(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    weight_path = find_adapter_weight_file(checkpoint_dir)
    if weight_path.endswith(".safetensors"):
        return _load_safetensors(weight_path)
    obj = safe_torch_load(weight_path)
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "trainable_state_dict"):
            nested = obj.get(key)
            if isinstance(nested, dict) and all(isinstance(v, torch.Tensor) for v in nested.values()):
                return nested
    raise ValueError(f"Unable to read an adapter state dict from {weight_path}")


def is_lora_key(key: str) -> bool:
    lower = key.lower()
    return "lora_" in lower or ".lora" in lower


def _tensor_shape(value: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _numel(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def build_adapter_template(
    checkpoint_dir: str,
    include_non_lora: bool = False,
    key_regex: str = "",
) -> Dict[str, Any]:
    state = load_adapter_state(checkpoint_dir)
    regex = re.compile(key_regex) if key_regex else None
    keys: List[str] = []
    for key in sorted(state.keys()):
        if regex is not None and regex.search(key) is None:
            continue
        if include_non_lora or is_lora_key(key):
            keys.append(key)
    if not keys:
        raise ValueError(f"No LoRA keys matched in {checkpoint_dir}")

    shapes = {key: _tensor_shape(state[key]) for key in keys}
    lengths = {key: _numel(shapes[key]) for key in keys}
    offsets: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for key in keys:
        start = cursor
        cursor += lengths[key]
        offsets[key] = (start, cursor)

    return {
        "keys": keys,
        "shapes": shapes,
        "lengths": lengths,
        "offsets": offsets,
        "input_dim": int(cursor),
        "include_non_lora": bool(include_non_lora),
        "key_regex": key_regex,
        "reference_checkpoint": os.path.abspath(checkpoint_dir),
    }


def validate_adapter_template(checkpoint_dir: str, template: Dict[str, Any]) -> None:
    state = load_adapter_state(checkpoint_dir)
    for key in template["keys"]:
        if key not in state:
            raise KeyError(f"{checkpoint_dir} is missing template key: {key}")
        expected = tuple(template["shapes"][key])
        actual = _tensor_shape(state[key])
        if actual != expected:
            raise ValueError(f"Shape mismatch for {key} in {checkpoint_dir}: {actual} vs {expected}")


def tensor_stats(tensor: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = tensor.detach().float().view(-1)
    mean = flat.mean()
    std = flat.std(unbiased=False).clamp_min(float(eps))
    return mean, std


def flatten_adapter_state(
    state: Dict[str, torch.Tensor],
    template: Dict[str, Any],
    normalize_mode: str = "per_tensor",
    eps: float = 1e-6,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for key in template["keys"]:
        value = state[key].detach().float().view(-1)
        if normalize_mode == "per_tensor":
            mean, std = tensor_stats(value, eps=eps)
            value = (value - mean) / std
        elif normalize_mode != "none":
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
        chunks.append(value)
    return torch.cat(chunks, dim=0)


def load_adapter_vector(
    checkpoint_dir: str,
    template: Dict[str, Any],
    normalize_mode: str = "per_tensor",
    eps: float = 1e-6,
) -> torch.Tensor:
    state = load_adapter_state(checkpoint_dir)
    return flatten_adapter_state(state, template, normalize_mode=normalize_mode, eps=eps)


def pool_vector_cpu(vector: torch.Tensor, pool_length: int) -> torch.Tensor:
    vector = vector.float().view(1, -1)
    pool_length = int(pool_length)
    if vector.size(1) == pool_length:
        return vector.view(-1)
    if vector.size(1) % pool_length != 0:
        pad = pool_length - (vector.size(1) % pool_length)
        vector = F.pad(vector, (0, pad), mode="constant", value=0.0)
    chunk_size = vector.size(1) // pool_length
    return vector.view(1, pool_length, chunk_size).mean(dim=-1).view(-1).contiguous()


def _cache_path(
    cache_dir: str,
    checkpoint_dir: str,
    template: Dict[str, Any],
    normalize_mode: str,
    pool_length: int,
) -> str:
    fingerprint = "|".join(
        [
            os.path.abspath(checkpoint_dir),
            str(template.get("input_dim")),
            normalize_mode,
            str(pool_length),
            str(os.path.getmtime(find_adapter_weight_file(checkpoint_dir))),
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{digest}.pt")


def load_adapter_pooled_vector(
    checkpoint_dir: str,
    template: Dict[str, Any],
    normalize_mode: str,
    pool_length: int,
    cache_dir: str = "",
    eps: float = 1e-6,
) -> torch.Tensor:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, checkpoint_dir, template, normalize_mode, pool_length)
        if os.path.isfile(path):
            payload = safe_torch_load(path)
            pooled = payload.get("pooled") if isinstance(payload, dict) else payload
            if isinstance(pooled, torch.Tensor) and int(pooled.numel()) == int(pool_length):
                return pooled.float().view(-1)

    vector = load_adapter_vector(checkpoint_dir, template, normalize_mode=normalize_mode, eps=eps)
    pooled = pool_vector_cpu(vector, pool_length)
    if cache_dir:
        torch.save({"pooled": pooled, "checkpoint_dir": checkpoint_dir}, path)
    return pooled


def reconstruct_lora_state(
    vector: torch.Tensor,
    source_state: Dict[str, torch.Tensor],
    template: Dict[str, Any],
    normalize_mode: str = "per_tensor",
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    flat = vector.detach().cpu().float().view(-1)
    out: Dict[str, torch.Tensor] = {}
    for key in template["keys"]:
        start, end = template["offsets"][key]
        chunk = flat[int(start) : int(end)]
        source_tensor = source_state[key]
        if normalize_mode == "per_tensor":
            mean, std = tensor_stats(source_tensor, eps=eps)
            chunk = chunk * std + mean
        elif normalize_mode != "none":
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")
        out[key] = chunk.reshape(tuple(template["shapes"][key])).to(dtype=source_tensor.dtype)
    return out


def save_generated_adapter_checkpoint(
    output_dir: str,
    source_checkpoint_dir: str,
    generated_lora_state: Dict[str, torch.Tensor],
    save_safetensors: bool = False,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    source_config = os.path.join(source_checkpoint_dir, "adapter_config.json")
    if not os.path.isfile(source_config):
        raise FileNotFoundError(f"Source adapter_config.json not found: {source_config}")
    shutil.copy2(source_config, os.path.join(output_dir, "adapter_config.json"))
    for file_name in ("config.json", "non_lora_trainables.bin", "README.md"):
        source_extra = os.path.join(source_checkpoint_dir, file_name)
        if os.path.isfile(source_extra):
            shutil.copy2(source_extra, os.path.join(output_dir, file_name))

    merged_state = load_adapter_state(source_checkpoint_dir)
    merged_state.update(generated_lora_state)
    merged_state = {key: value.detach().cpu().contiguous() for key, value in merged_state.items()}
    if save_safetensors:
        _save_safetensors(os.path.join(output_dir, "adapter_model.safetensors"), merged_state)
    else:
        torch.save(merged_state, os.path.join(output_dir, "adapter_model.bin"))
    return output_dir


def canonical_task_name(task: str) -> str:
    aliases = {
        "science_qa": "scienceqa",
        "image_net": "imagenet",
        "ocrvqa": "ocr_vqa",
        "ocr-vqa": "ocr_vqa",
        "grounding": "rec_coco",
        "rec-coco": "rec_coco",
    }
    normalized = str(task).strip().lower().replace(" ", "_")
    return aliases.get(normalized, normalized)


def task_output_dir(checkpoint_root: str, task: str) -> str:
    return os.path.join(checkpoint_root, f"output_{canonical_task_name(task)}")


def _checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", os.path.normpath(path))
    if match is None:
        return -1
    return int(match.group(1))


def _reference_parts(reference_path: str) -> List[str]:
    normalized = str(reference_path).replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def _reference_basename(reference_path: str) -> str:
    parts = _reference_parts(reference_path)
    return parts[-1] if parts else ""


def resolve_checkpoint_path(reference_path: str, checkpoint_root: str, task: str) -> str:
    candidates: List[str] = []
    if reference_path:
        name = _reference_basename(reference_path)
        if name.startswith("checkpoint-"):
            candidates.append(os.path.join(task_output_dir(checkpoint_root, task), name))

        marker = f"output_{canonical_task_name(task)}"
        parts = _reference_parts(reference_path)
        if marker in parts:
            idx = parts.index(marker)
            suffix = os.path.join(*parts[idx + 1 :]) if idx + 1 < len(parts) else ""
            candidates.append(os.path.join(task_output_dir(checkpoint_root, task), suffix))

        candidates.append(os.path.abspath(os.path.expanduser(reference_path)))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            return candidate
    return candidates[-1] if candidates else ""


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_from_entry(entry: Dict[str, Any], metric_name: str = "") -> Optional[float]:
    for key in ("score", "metric", metric_name, f"eval_{metric_name}" if metric_name else ""):
        if key:
            value = _float_or_none(entry.get(key))
            if value is not None:
                return value
    for container_key in ("metrics", "dev_metrics", "eval_metrics"):
        metrics = entry.get(container_key)
        if not isinstance(metrics, dict):
            continue
        for key in (metric_name, f"eval_{metric_name}" if metric_name else "", "accuracy", "acc", "average", "eval_accuracy", "eval_acc"):
            if key:
                value = _float_or_none(metrics.get(key))
                if value is not None:
                    return value
    return None


def _load_eval_history_scores(out_dir: str) -> Tuple[str, Dict[str, float], Dict[int, float]]:
    history_path = os.path.join(out_dir, "eval_history.json")
    if not os.path.isfile(history_path):
        return "", {}, {}
    try:
        with open(history_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return "", {}, {}
    if not isinstance(payload, list):
        return "", {}, {}
    metric_name = ""
    by_checkpoint: Dict[str, float] = {}
    by_step: Dict[int, float] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        metric_name = metric_name or str(entry.get("metric_name", "") or "")
        score = _score_from_entry(entry, metric_name)
        if score is None:
            continue
        checkpoint_name = str(entry.get("checkpoint", "") or "")
        if checkpoint_name:
            by_checkpoint[checkpoint_name] = score
        try:
            by_step[int(entry.get("step"))] = score
        except (TypeError, ValueError):
            pass
    return metric_name, by_checkpoint, by_step


@dataclass
class CheckpointSpec:
    task: str
    path: str
    step: int = -1
    score: Optional[float] = None
    metric_name: str = ""


def discover_task_checkpoints(
    tasks: Sequence[str],
    checkpoint_root: str,
    topk_per_task: int = 10,
) -> Dict[str, List[CheckpointSpec]]:
    task_to_specs: Dict[str, List[CheckpointSpec]] = {}
    for task in tasks:
        out_dir = task_output_dir(checkpoint_root, task)
        top_path = os.path.join(out_dir, "top_checkpoints.json")
        history_metric_name, history_by_checkpoint, history_by_step = _load_eval_history_scores(out_dir)
        specs: List[CheckpointSpec] = []
        if os.path.isfile(top_path):
            with open(top_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            metric_name = str(payload.get("metric_name", "") or history_metric_name)
            for entry in payload.get("checkpoints", []):
                if not isinstance(entry, dict):
                    continue
                path = resolve_checkpoint_path(str(entry.get("path", "")), checkpoint_root, task)
                if not os.path.isdir(path):
                    continue
                step = int(entry.get("step", _checkpoint_step(path)))
                checkpoint_name = _reference_basename(path)
                score = _score_from_entry(entry, metric_name)
                if score is None:
                    score = history_by_checkpoint.get(checkpoint_name, history_by_step.get(step))
                specs.append(
                    CheckpointSpec(
                        task=task,
                        path=path,
                        step=step,
                        score=score,
                        metric_name=metric_name,
                    )
                )
        else:
            raise FileNotFoundError(f"Validation-ranked checkpoint manifest is required: {top_path}")

        task_to_specs[task] = select_validation_topk(
            specs, int(topk_per_task), metric_name, source=top_path
        )
    return task_to_specs


def build_one_hot_task_embeddings(tasks: Sequence[str]) -> Dict[str, torch.Tensor]:
    embeddings: Dict[str, torch.Tensor] = {}
    num_tasks = len(tasks)
    for idx, task in enumerate(tasks):
        vec = torch.zeros(num_tasks, dtype=torch.float32)
        vec[idx] = 1.0
        embeddings[task] = vec
    return embeddings


class PooledAdapterDataset(Dataset):
    def __init__(
        self,
        checkpoint_specs: Sequence[CheckpointSpec],
        template: Dict[str, Any],
        normalize_mode: str,
        pool_length: int,
        cache_dir: str = "",
        eps: float = 1e-6,
    ):
        self.checkpoint_specs = list(checkpoint_specs)
        self.template = template
        self.normalize_mode = normalize_mode
        self.pool_length = int(pool_length)
        self.cache_dir = cache_dir
        self.eps = float(eps)
        if not self.checkpoint_specs:
            raise ValueError("PooledAdapterDataset received an empty checkpoint list.")

    def __len__(self) -> int:
        return len(self.checkpoint_specs)

    def __getitem__(self, index: int) -> torch.Tensor:
        spec = self.checkpoint_specs[index]
        return load_adapter_pooled_vector(
            spec.path,
            self.template,
            normalize_mode=self.normalize_mode,
            pool_length=self.pool_length,
            cache_dir=self.cache_dir,
            eps=self.eps,
        )


class FullAdapterDataset(Dataset):
    """Normalized full LoRA-factor vectors for score matching and prototypes."""

    def __init__(
        self,
        checkpoint_specs: Sequence[CheckpointSpec],
        template: Dict[str, Any],
        normalize_mode: str,
        eps: float = 1e-6,
    ):
        self.checkpoint_specs = list(checkpoint_specs)
        self.template = template
        self.normalize_mode = normalize_mode
        self.eps = float(eps)
        if not self.checkpoint_specs:
            raise ValueError("FullAdapterDataset received an empty checkpoint list.")

    def __len__(self) -> int:
        return len(self.checkpoint_specs)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_adapter_vector(
            self.checkpoint_specs[index].path,
            self.template,
            normalize_mode=self.normalize_mode,
            eps=self.eps,
        )


def flatten_specs(task_to_specs: Dict[str, Sequence[CheckpointSpec]]) -> List[CheckpointSpec]:
    specs: List[CheckpointSpec] = []
    for task in task_to_specs:
        specs.extend(task_to_specs[task])
    return specs


def parse_task_order(task_order: str) -> List[str]:
    aliases = {
        "science_qa": "scienceqa",
        "image_net": "imagenet",
        "ocrvqa": "ocr_vqa",
        "ocr-vqa": "ocr_vqa",
        "grounding": "rec_coco",
        "rec-coco": "rec_coco",
    }
    tasks = [aliases.get(token.strip().lower(), token.strip().lower()) for token in str(task_order).split(",") if token.strip()]
    if not tasks:
        raise ValueError("task_order is empty.")
    seen = set()
    deduped: List[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        deduped.append(task)
    return deduped


def first_checkpoint(task_to_specs: Dict[str, Sequence[CheckpointSpec]]) -> str:
    for specs in task_to_specs.values():
        if specs:
            return specs[0].path
    raise ValueError("No checkpoint specs are available.")
