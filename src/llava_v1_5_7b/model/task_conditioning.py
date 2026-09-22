"""Frozen text-backbone embeddings for LLaVA task descriptions."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


TASK_DESCRIPTIONS = {
    "scienceqa": "Task: ScienceQA. Input: an image and a science question. Output: the correct multiple-choice answer.",
    "vizwiz": "Task: VizWiz. Input: a user-taken image and a question. Output: a short answer grounded in the image.",
    "imagenet": "Task: ImageNet classification. Input: an image. Output: its object category.",
    "vqav2": "Task: VQAv2. Input: an image and a visual question. Output: a concise answer.",
    "iconqa": "Task: IconQA. Input: an icon-based image and a question. Output: the correct answer option.",
    "flickr30k": "Task: Flickr30k captioning. Input: an image. Output: a descriptive caption.",
    "rec_coco": "Task: referring expression comprehension. Input: an image and a referring expression. Output: the referred object's location.",
    "ocr_vqa": "Task: OCR-VQA. Input: an image containing text and a question. Output: the answer using visual text recognition.",
}


def build_task_condition_embeddings(args: Any, tasks: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    mode = str(args.task_condition_mode).lower()
    if mode != "dense_hidden":
        raise ValueError("The main EC-LoRA method requires task_condition_mode=dense_hidden.")
    model_path = str(args.task_condition_model_path or "").strip()
    if not model_path:
        raise ValueError(
            "Set task_condition_model_path to the frozen text backbone used by LLaVA; "
            "the task encoder cannot be inferred reliably from a multimodal checkpoint."
        )

    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    encoder.to(device).eval().requires_grad_(False)

    pooling = str(args.task_condition_pooling).lower()
    if pooling not in {"mean", "last"}:
        raise ValueError(f"Unsupported task_condition_pooling: {pooling}")
    embeddings: Dict[str, torch.Tensor] = {}
    for task in tasks:
        if task not in TASK_DESCRIPTIONS:
            raise KeyError(f"No public task description is defined for {task}")
        encoded = tokenizer(
            TASK_DESCRIPTIONS[task],
            return_tensors="pt",
            truncation=True,
            max_length=int(args.task_condition_max_length),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].to(hidden.dtype)
            if pooling == "last":
                pooled = hidden[0, int(mask[0].sum().item()) - 1]
            else:
                pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = pooled[0]
        embedding = pooled.detach().float().cpu().view(-1)
        if args.task_condition_normalize:
            embedding = F.normalize(embedding, p=2, dim=0)
        embeddings[task] = embedding

    dimension = next(iter(embeddings.values())).numel()
    return embeddings, {
        "mode": mode,
        "model_path": model_path,
        "pooling": pooling,
        "normalize": bool(args.task_condition_normalize),
        "dim": dimension,
    }
