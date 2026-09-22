from typing import List, Optional

import torch


def task_changes(result_t: torch.Tensor):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for idx, task_id in enumerate(result_t):
        if task_id != current:
            changes.append(idx)
            current = task_id
    return n_tasks, changes


def confusion_matrix(
    result_t: torch.Tensor,
    result_a: torch.Tensor,
    fname: Optional[str] = None,
) -> List[torch.Tensor]:
    nt, changes = task_changes(result_t)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a.index_select(0, torch.LongTensor(changes))

    diag_idx = torch.arange(nt)
    acc = result[diag_idx, diag_idx]
    fin = result[nt - 1, :nt]
    bwt = fin - acc

    fwt = torch.zeros(nt)
    for task_idx in range(1, nt):
        fwt[task_idx] = result[task_idx - 1, task_idx] - baseline[task_idx]

    if fname is not None:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(" ".join([f"{r:.4f}" for r in baseline.tolist()]) + "\n")
            f.write("|\n")
            for row_idx in range(result.size(0)):
                f.write(" ".join([f"{r:.4f}" for r in result[row_idx].tolist()]) + "\n")
            f.write("\n")
            f.write(f"Final Accuracy: {fin.mean():.4f}\n")
            f.write(f"Backward: {bwt.mean():.4f}\n")
            f.write(f"Forward: {fwt.mean():.4f}\n")

    return [fin.mean(), bwt.mean(), fwt.mean()]
