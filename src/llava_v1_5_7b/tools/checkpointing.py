import datetime
import os
import random
import uuid
from typing import Any, Dict, Optional

import numpy as np
import torch


def make_out_prefix(save_dir: str, exp_name: str, resume_payload: Optional[Dict[str, Any]] = None) -> str:
    if resume_payload is not None:
        out_prefix = str(resume_payload.get("out_prefix", "")).strip()
        if out_prefix:
            return out_prefix
    uid = uuid.uuid4().hex[:8]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{exp_name}_{stamp}_{uid}"
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, run_name)


def task_checkpoint_path(out_prefix: str, task_idx: int, task_name: str) -> str:
    safe_name = str(task_name).replace("/", "_").replace(" ", "_")
    return f"{out_prefix}_task{task_idx + 1:02d}_{safe_name}.ckpt.pt"


def save_task_checkpoint(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_task_checkpoint(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
