"""Validation-ranked checkpoint selection shared by EC-LoRA backbones."""

from __future__ import annotations

import math
from typing import Sequence, TypeVar


T = TypeVar("T")


def select_validation_topk(
    candidates: Sequence[T],
    count: int,
    metric_name: str,
    *,
    source: str,
) -> list[T]:
    """Select the fixed Top-K by validation score, never by file/epoch order.

    Candidates expose ``path`` and ``score`` attributes. A manifest must specify
    which validation metric its scores represent so that loss-like metrics are
    ranked in the correct direction.
    """
    if count <= 0:
        raise ValueError(f"Top-K must be positive for {source}; got {count}.")
    if len(candidates) < count:
        raise ValueError(f"{source} contains {len(candidates)} checkpoints; Top-{count} requires at least {count}.")
    metric = str(metric_name or "").strip().lower()
    if not metric:
        raise ValueError(f"{source} must declare metric_name for validation-ranked Top-K selection.")
    lower_is_better = any(token in metric for token in ("loss", "error", "perplexity", "ppl", "mse", "mae", "wer"))
    unique: dict[str, T] = {}
    for item in candidates:
        path = str(getattr(item, "path"))
        score = getattr(item, "score", None)
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f"Missing or non-finite validation {metric_name} for {path} in {source}.")
        if path in unique:
            raise ValueError(f"Duplicate checkpoint {path} in {source}.")
        unique[path] = item
    return sorted(
        unique.values(),
        key=lambda item: (float(item.score) if lower_is_better else -float(item.score), str(item.path)),
    )[:count]
