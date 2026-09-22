import os
from typing import Dict, List


DEFAULT_TASK_ORDER: List[str] = [
    "mnist",
    "svhn",
    "gtsrb",
    "eurosat",
    "resisc45",
    "dtd",
    "sun397",
    "stanfordcars",
]


def parse_task_order(task_order: str) -> List[str]:
    items = [x.strip().lower() for x in task_order.split(",") if x.strip()]
    if not items:
        raise ValueError("task_order is empty.")
    return items


def build_task_id_map(task_order: List[str]) -> Dict[str, int]:
    return {task: idx for idx, task in enumerate(task_order)}


def default_project_root(current_file: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))


def default_task_vectors_dir(project_root: str) -> str:
    return os.path.join(project_root, "ICM-LoRA-ViT", "all_tasks_vectors")


def default_normalized_root(project_root: str) -> str:
    return os.path.join(project_root, "ICM-LoRA-ViT", "checkpoints")


def normalized_task_dir(normalized_root: str, task: str) -> str:
    return os.path.join(normalized_root, f"normalized_data_{task}")


def default_source_checkpoint(project_root: str, task: str) -> str:
    return os.path.join(
        project_root,
        "ICM-LoRA-ViT",
        "checkpoints",
        f"output_{task}",
        f"best_{task}_lora_vit.pt",
    )


def default_task_config(project_root: str, task: str) -> str:
    return os.path.join(
        project_root,
        "models",
        "config",
        "LoRA_ViT",
        f"{task}_lora_config.yaml",
    )


def default_save_dir(project_root: str) -> str:
    return os.path.join(project_root, "Meta_EBM", "results")


def default_log_dir(project_root: str) -> str:
    return os.path.join(project_root, "Meta_EBM", "logs")
