#!/usr/bin/env python3
import datetime
import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CLLogger:
    def __init__(self, log_dir: str, method_name: str, comp_weight: float = 1.0):
        self.method_name = method_name
        self.comp_weight = float(comp_weight)
        self.out_dir = os.path.join(log_dir, method_name)
        os.makedirs(self.out_dir, exist_ok=True)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.out_dir, f"train_log_{stamp}.txt")
        self.metrics_path = os.path.join(self.out_dir, f"train_metrics_{stamp}.jsonl")
        self.plot_path = os.path.join(self.out_dir, f"loss_curve_{stamp}.png")

        self.global_steps: List[int] = []
        self.total_values: List[float] = []
        self.task_values: List[float] = []
        self.replay_values: List[float] = []
        self.align_values: List[float] = []
        self.align_cosine_values: List[float] = []
        self.align_skipped_values: List[float] = []
        self.grad_norm_cur_values: List[float] = []
        self.grad_norm_mem_values: List[float] = []
        self.comp_values: List[float] = []
        self.comp_pairs_values: List[float] = []
        self.comp_score_rms_values: List[float] = []

        self.task_boundaries: List[int] = []
        self.task_labels: List[str] = []
        self._global_step = 0

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# {method_name} training log\n")
            f.write(f"# start_time: {stamp}\n")
            f.write(f"# metrics_jsonl: {self.metrics_path}\n")
            f.write("=" * 60 + "\n\n")
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            f.write("")

    def _append_metrics(self, record: dict):
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_step(
        self,
        task_name: str,
        epoch: int,
        step: int,
        total_loss: float,
        task_loss: float = 0.0,
        replay_loss: float = 0.0,
        align_loss: float = 0.0,
        align_cosine: float = 0.0,
        align_skipped: float = 0.0,
        grad_norm_cur: float = 0.0,
        grad_norm_mem: float = 0.0,
        comp_loss: float = 0.0,
        comp_pairs: float = 0.0,
        comp_score_rms: float = 0.0,
    ):
        self._global_step += 1
        self.global_steps.append(self._global_step)
        self.total_values.append(total_loss)
        self.task_values.append(task_loss)
        self.replay_values.append(replay_loss)
        self.align_values.append(align_loss)
        self.align_cosine_values.append(align_cosine)
        self.align_skipped_values.append(align_skipped)
        self.grad_norm_cur_values.append(grad_norm_cur)
        self.grad_norm_mem_values.append(grad_norm_mem)
        self.comp_values.append(comp_loss)
        self.comp_pairs_values.append(comp_pairs)
        self.comp_score_rms_values.append(comp_score_rms)
        self._append_metrics(
            {
                "stage": "step",
                "global_step": self._global_step,
                "task": task_name,
                "epoch": int(epoch),
                "step": int(step),
                "total_loss": float(total_loss),
                "task_loss": float(task_loss),
                "replay_loss": float(replay_loss),
                "align_loss": float(align_loss),
                "align_cosine": float(align_cosine),
                "align_skipped": float(align_skipped),
                "grad_norm_cur": float(grad_norm_cur),
                "grad_norm_mem": float(grad_norm_mem),
                "comp_loss": float(comp_loss),
                "weighted_comp_loss": float(self.comp_weight * comp_loss),
                "comp_pairs": float(comp_pairs),
                "comp_score_rms": float(comp_score_rms),
            }
        )

    def log_epoch(self, record: dict):
        payload = {"stage": "epoch", "global_step": self._global_step}
        payload.update(record)
        self._append_metrics(payload)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                "[epoch] "
                f"global_step={payload.get('global_step')} "
                f"task={payload.get('task')} "
                f"epoch={payload.get('epoch')} "
                f"loss={float(payload.get('total_loss', 0.0)):.6f} "
                f"task_loss={float(payload.get('task_loss', 0.0)):.6f} "
                f"replay_loss={float(payload.get('replay_loss', 0.0)):.6f} "
                f"align_loss={float(payload.get('align_loss', 0.0)):.6f} "
                f"comp_loss={float(payload.get('comp_loss', 0.0)):.6f} "
                f"memory={payload.get('memory')} "
                f"align_cos={float(payload.get('align_cosine', 0.0)):.6f} "
                f"align_skip={float(payload.get('align_skipped', 0.0)):.6f} "
                f"grad_cur={float(payload.get('grad_norm_cur', 0.0)):.6f} "
                f"grad_mem={float(payload.get('grad_norm_mem', 0.0)):.6f} "
                f"comp_pairs={float(payload.get('comp_pairs', 0.0)):.6f} "
                f"comp_rms={float(payload.get('comp_score_rms', 0.0)):.6f}\n"
            )

    def mark_task_boundary(self, task_name: str):
        self.task_boundaries.append(self._global_step)
        self.task_labels.append(task_name)

    def save_loss_plot(self):
        if len(self.global_steps) < 2:
            return

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(self.global_steps, self.total_values, linewidth=0.9, alpha=0.9, label="total_loss")
        if any(v != 0.0 for v in self.task_values):
            ax.plot(self.global_steps, self.task_values, linewidth=0.7, alpha=0.7, label="task_loss")
        if any(v != 0.0 for v in self.replay_values):
            ax.plot(self.global_steps, self.replay_values, linewidth=0.7, alpha=0.7, label="replay_loss")
        if any(v != 0.0 for v in self.align_values):
            ax.plot(self.global_steps, self.align_values, linewidth=0.7, alpha=0.7, label="align_loss")
        weighted_comp_values = [self.comp_weight * v for v in self.comp_values]
        if any(v != 0.0 for v in weighted_comp_values):
            ax.plot(
                self.global_steps,
                weighted_comp_values,
                linewidth=0.7,
                alpha=0.7,
                label="weighted_comp_loss",
            )

        colors = plt.cm.tab10.colors
        for idx, (boundary, label) in enumerate(zip(self.task_boundaries, self.task_labels)):
            color = colors[idx % len(colors)]
            ax.axvline(x=boundary, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(boundary, ax.get_ylim()[1] * 0.95, f" {label}", fontsize=7, color=color, rotation=90, va="top")

        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.method_name} - Training Loss")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)

    def log_task_eval(self, task_idx: int, task_name: str, accs: List[float], task_order: List[str]):
        n_tasks = len(task_order)
        mean_acc = sum(accs) / len(accs) if accs else 0.0
        parts = []
        for idx, task in enumerate(task_order):
            val = accs[idx] if idx < len(accs) else 0.0
            parts.append(f"{task} : {val:.4f}")
        parts.append(f"mean_acc : {mean_acc:.4f}")
        line = f"Task {task_idx + 1}/{n_tasks} {task_name}: " + "; ".join(parts)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_task_stats(self, task_idx: int, stats: List[float]):
        return

    def log_final(self, stats: List[float], spent: float):
        return

    def log_baseline_eval(self, accs: List[float], task_order: List[str]):
        return
