import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from checkpoint_pool import select_validation_topk
from task_setup import normalized_task_dir


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def pad_vector(vec: torch.Tensor, target_dim: int = 0) -> torch.Tensor:
    out = vec.float().view(-1)
    target_dim = int(target_dim)
    if target_dim <= 0 or out.numel() == target_dim:
        return out
    if out.numel() > target_dim:
        return out[:target_dim]
    return F.pad(out, (0, target_dim - out.numel()), mode="constant", value=0.0)


class NormalizedFileDataset(Dataset):
    def __init__(self, file_paths: Sequence[str], target_dim: int = 0):
        self.file_paths = list(file_paths)
        self.target_dim = int(target_dim)
        if not self.file_paths:
            raise ValueError("NormalizedFileDataset received an empty file list.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        path = self.file_paths[index]
        item = safe_torch_load(path)
        if "data" not in item:
            raise KeyError(f"{path} is missing the data field.")
        return pad_vector(item["data"], self.target_dim)


def _list_files_in_dir(data_dir: str, task: str) -> List[str]:
    pattern = os.path.join(data_dir, f"normalized_{task}_*.pth")
    return sorted(glob(pattern))


@dataclass(frozen=True)
class _ScoredFile:
    path: str
    score: float


def _select_topk_files(files: Sequence[str], task: str, data_dir: str, topk: int) -> List[str]:
    manifests = (
        os.path.join(data_dir, f"top_checkpoints_{task}.json"),
        os.path.join(data_dir, "top_checkpoints.json"),
    )
    manifest = next((path for path in manifests if os.path.isfile(path)), None)
    if manifest is None:
        raise FileNotFoundError(
            f"Validation-ranked normalized checkpoint manifest is required for {task} in {data_dir}. "
            f"Expected {manifests[0]} or {manifests[1]}."
        )
    with open(manifest, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("task") not in (None, task):
        raise ValueError(f"Manifest {manifest} belongs to task {payload['task']}, not {task}.")
    available = {os.path.normcase(os.path.abspath(path)): path for path in files}
    candidates: List[_ScoredFile] = []
    for entry in payload.get("checkpoints", []):
        reference = str(entry.get("normalized_path") or entry.get("path") or "").strip()
        possible = (
            reference if os.path.isabs(reference) else os.path.join(data_dir, reference),
            os.path.join(data_dir, os.path.basename(reference)),
        )
        selected = next((available[os.path.normcase(os.path.abspath(path))]
                         for path in possible if os.path.normcase(os.path.abspath(path)) in available), None)
        if selected is None:
            raise FileNotFoundError(f"Manifest {manifest} refers to a missing normalized vector: {reference}")
        candidates.append(_ScoredFile(path=selected, score=entry.get("score")))
    count = int(topk) if int(topk) > 0 else len(candidates)
    ranked = select_validation_topk(candidates, count, payload.get("metric_name", ""), source=manifest)
    return [item.path for item in ranked]


def list_task_files(
    tasks: Sequence[str],
    normalized_root: str,
    normalized_all_dir: str = "",
    topk_per_task: int = 0,
) -> Dict[str, List[str]]:
    task_to_files: Dict[str, List[str]] = {}
    for task in tasks:
        attempted_dirs: List[str] = []
        files: List[str] = []

        task_dir = normalized_task_dir(normalized_root, task)
        data_dir = normalized_all_dir or task_dir
        if normalized_all_dir:
            attempted_dirs.append(normalized_all_dir)
            files = _list_files_in_dir(normalized_all_dir, task)

        if not files:
            data_dir = task_dir
            attempted_dirs.append(task_dir)
            files = _list_files_in_dir(task_dir, task)

        if not files:
            raise FileNotFoundError(
                f"Missing normalized files for task={task}. "
                f"attempted_dirs={attempted_dirs} normalized_root={normalized_root}"
            )
        task_to_files[task] = _select_topk_files(files, task=task, data_dir=data_dir, topk=topk_per_task)
    return task_to_files


def validate_input_dim(task_to_files: Dict[str, List[str]]) -> Tuple[int, Dict[str, int]]:
    dim = None
    per_task_dim: Dict[str, int] = {}
    for task, files in task_to_files.items():
        ref_item = safe_torch_load(files[0])
        if "data" not in ref_item:
            raise KeyError(f"{files[0]} is missing the data field.")
        task_dim = int(ref_item["data"].numel())
        per_task_dim[task] = task_dim
        if dim is None:
            dim = task_dim
        else:
            dim = max(dim, task_dim)

    if dim is None:
        raise ValueError("Unable to infer input_dim from task files.")
    return dim, per_task_dim


def load_task_vectors(task_vectors_dir: str, tasks: Sequence[str]) -> Dict[str, torch.Tensor]:
    vectors: Dict[str, torch.Tensor] = {}
    for task in tasks:
        path = os.path.join(task_vectors_dir, f"{task}_hidden_state.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Task vector not found: {path}")
        vec = safe_torch_load(path)
        if not isinstance(vec, torch.Tensor):
            raise TypeError(f"Task vector is not a tensor: {path}")
        vectors[task] = vec.float().view(-1)

    cond_dim = None
    for task, vec in vectors.items():
        if cond_dim is None:
            cond_dim = int(vec.numel())
        elif cond_dim != int(vec.numel()):
            raise ValueError(f"Condition dimension mismatch for task {task}: {vec.numel()} vs {cond_dim}")
    return vectors


def compute_task_prototype(file_paths: Sequence[str], target_dim: int = 0) -> torch.Tensor:
    accumulator = None
    count = 0
    for path in file_paths:
        item = safe_torch_load(path)
        if "data" not in item:
            raise KeyError(f"{path} is missing the data field.")
        vec = pad_vector(item["data"], target_dim)
        accumulator = vec if accumulator is None else accumulator + vec
        count += 1
    if accumulator is None or count == 0:
        raise ValueError("Cannot compute a prototype from an empty file list.")
    return accumulator / float(count)
