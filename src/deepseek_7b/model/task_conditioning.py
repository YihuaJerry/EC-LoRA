from __future__ import annotations

import gc
import hashlib
import json
import os
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


TASK_DESCRIPTIONS: Dict[str, str] = {
    "cola": (
        "Task: CoLA. Input: one English sentence. Output: whether the sentence is "
        "grammatically acceptable. Metric: Matthews correlation."
    ),
    "mnli_m": (
        "Task: MNLI matched. Input: a premise and a hypothesis from matched domains. "
        "Output: entailment, contradiction, or neutral."
    ),
    "mnli_mm": (
        "Task: MNLI mismatched. Input: a premise and a hypothesis from mismatched domains. "
        "Output: entailment, contradiction, or neutral."
    ),
    "mrpc": (
        "Task: MRPC. Input: two sentences. Output: whether the two sentences are "
        "semantically equivalent paraphrases."
    ),
    "qnli": (
        "Task: QNLI. Input: a question and a sentence. Output: whether the sentence "
        "contains the answer to the question."
    ),
    "qqp": (
        "Task: QQP. Input: two questions. Output: whether the two questions ask the "
        "same thing."
    ),
    "rte": (
        "Task: RTE. Input: a premise and a hypothesis. Output: whether the premise "
        "entails the hypothesis."
    ),
    "sst2": (
        "Task: SST-2. Input: one sentence. Output: whether the sentiment is positive "
        "or negative."
    ),
    "stsb": (
        "Task: STS-B. Input: two sentences. Output: a continuous semantic similarity "
        "score between the sentences."
    ),
    "wnli": (
        "Task: WNLI. Input: sentence pairs involving pronoun or coreference reasoning. "
        "Output: whether the second sentence follows from the first."
    ),
}


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _task_description(task: str) -> str:
    key = str(task).strip().lower()
    return TASK_DESCRIPTIONS.get(
        key,
        f"Task: {key}. Input: task-specific text fields. Output: the correct label or score for this task.",
    )


def _one_hot_embeddings(tasks: Sequence[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for idx, task in enumerate(tasks):
        vec = torch.zeros(len(tasks), dtype=torch.float32)
        vec[idx] = 1.0
        out[task] = vec
    return out


def _dtype_from_name(name: str, device: torch.device):
    value = str(name or "auto").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    return None


def _condition_device(args) -> torch.device:
    device_text = str(getattr(args, "task_condition_device", "") or getattr(args, "device", "cpu"))
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def _resolve_model_path_and_dtype(
    args,
    tasks: Sequence[str],
    project_root: str,
    default_lora_config_name: str,
) -> Tuple[str, str]:
    explicit_path = str(getattr(args, "task_condition_model_path", "") or "").strip()
    if explicit_path:
        return os.path.abspath(os.path.expanduser(explicit_path)), str(getattr(args, "task_condition_precision", "auto"))

    eval_model_path = str(getattr(args, "eval_model_path", "") or "").strip()
    if eval_model_path:
        return os.path.abspath(os.path.expanduser(eval_model_path)), str(getattr(args, "task_condition_precision", "auto"))

    config_dir = str(getattr(args, "eval_lora_config_dir", "") or "").strip()
    if not config_dir:
        config_dir = os.path.join(project_root, "models", "config", default_lora_config_name)
    config_dir = os.path.abspath(os.path.expanduser(config_dir))
    first_task = str(tasks[0])
    config = _load_yaml(os.path.join(config_dir, f"{first_task}_lora_config.yaml"))
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    model_path = str(model_cfg.get("pretrained_checkpoint", "") or "").strip()
    if not model_path:
        raise ValueError(
            "task_condition_model_path is empty and no pretrained_checkpoint could be read "
            f"from {config_dir}/{first_task}_lora_config.yaml"
        )
    dtype_name = str(getattr(args, "task_condition_precision", "auto") or "auto")
    if dtype_name == "auto":
        dtype_name = str(model_cfg.get("torch_dtype", "auto") or "auto")
    return os.path.abspath(os.path.expanduser(model_path)), dtype_name


def _cache_path(
    cache_dir: str,
    task: str,
    model_path: str,
    description: str,
    pooling: str,
    max_length: int,
    normalize: bool,
) -> str:
    payload = {
        "task": task,
        "model_path": model_path,
        "description": description,
        "pooling": pooling,
        "max_length": int(max_length),
        "normalize": bool(normalize),
        "version": 2,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    safe_task = str(task).replace("/", "_").replace(" ", "_")
    return os.path.join(cache_dir, f"{safe_task}_{digest}.pt")


def _load_cached_embedding(path: str) -> torch.Tensor | None:
    if not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    embedding = payload.get("embedding") if isinstance(payload, dict) else payload
    if isinstance(embedding, torch.Tensor) and embedding.numel() > 0:
        return embedding.detach().cpu().float().view(-1)
    return None


def _pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    pooling = str(pooling or "mean").lower()
    if pooling == "last":
        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        return last_hidden[batch_idx, lengths]
    mask = attention_mask.to(dtype=last_hidden.dtype).unsqueeze(-1)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def build_task_condition_embeddings(
    args,
    tasks: Sequence[str],
    project_root: str,
    script_dir: str,
    default_lora_config_name: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    mode = str(getattr(args, "task_condition_mode", "dense_hidden") or "dense_hidden").lower()
    if mode in {"one_hot", "onehot"}:
        embeddings = _one_hot_embeddings(tasks)
        return embeddings, {"mode": "one_hot", "dim": len(tasks)}
    if mode not in {"dense_hidden", "dense"}:
        raise ValueError(f"Unsupported task_condition_mode: {mode}")

    cache_dir = str(getattr(args, "task_condition_cache_dir", "") or "").strip()
    if not cache_dir:
        cache_dir = os.path.join(script_dir, ".cache", "task_conditions")
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    model_path, dtype_name = _resolve_model_path_and_dtype(args, tasks, project_root, default_lora_config_name)
    pooling = str(getattr(args, "task_condition_pooling", "mean") or "mean").lower()
    max_length = int(getattr(args, "task_condition_max_length", 256) or 256)
    normalize = bool(getattr(args, "task_condition_normalize", True))

    embeddings: Dict[str, torch.Tensor] = {}
    missing = []
    cache_paths: Dict[str, str] = {}
    for task in tasks:
        description = _task_description(task)
        path = _cache_path(cache_dir, task, model_path, description, pooling, max_length, normalize)
        cache_paths[task] = path
        cached = _load_cached_embedding(path)
        if cached is None:
            missing.append(task)
        else:
            embeddings[task] = cached

    if missing:
        from transformers import AutoModel, AutoTokenizer

        device = _condition_device(args)
        dtype = _dtype_from_name(dtype_name, device)
        trust_remote_code = bool(getattr(args, "task_condition_trust_remote_code", True))
        print(
            f"[task-condition] building dense hidden states for {len(missing)} tasks "
            f"with model={model_path} device={device} dtype={dtype_name}"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        model.to(device)
        model.eval()

        for task in missing:
            description = _task_description(task)
            inputs = tokenizer(
                description,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                pooled = _pool_hidden(outputs.last_hidden_state, inputs["attention_mask"], pooling)
            embedding = pooled[0].detach().float().cpu()
            if normalize:
                embedding = F.normalize(embedding.view(1, -1), p=2, dim=1).view(-1)
            torch.save(
                {
                    "task": task,
                    "description": description,
                    "model_path": model_path,
                    "pooling": pooling,
                    "max_length": max_length,
                    "normalize": normalize,
                    "embedding": embedding,
                },
                cache_paths[task],
            )
            embeddings[task] = embedding

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dims = {int(value.numel()) for value in embeddings.values()}
    if len(dims) != 1:
        raise ValueError(f"Task condition embedding dimensions are inconsistent: {sorted(dims)}")
    dim = next(iter(dims))
    return embeddings, {
        "mode": "dense_hidden",
        "dim": dim,
        "model_path": model_path,
        "pooling": pooling,
        "max_length": max_length,
        "normalize": normalize,
        "cache_dir": cache_dir,
    }
