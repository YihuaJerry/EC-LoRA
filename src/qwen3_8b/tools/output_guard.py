from __future__ import annotations

import os


class OutputDirectoryDeletedError(RuntimeError):
    pass


def assert_result_dir_alive(path: str, label: str = "result directory") -> None:
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise OutputDirectoryDeletedError(
            f"{label} was removed during execution: {abs_path}. "
            "Abort instead of recreating it."
        )


def guarded_makedirs(path: str, guard_root: str) -> str:
    guard_root_abs = os.path.abspath(guard_root)
    path_abs = os.path.abspath(path)
    assert_result_dir_alive(guard_root_abs)

    rel = os.path.relpath(path_abs, guard_root_abs)
    if rel == ".":
        return path_abs
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        raise ValueError(f"Refusing to create output outside guarded result root: {path_abs}")

    current = guard_root_abs
    for part in rel.split(os.sep):
        current = os.path.join(current, part)
        if os.path.isdir(current):
            continue
        if os.path.exists(current):
            raise FileExistsError(f"Output path exists but is not a directory: {current}")
        assert_result_dir_alive(guard_root_abs)
        os.mkdir(current)
    assert_result_dir_alive(guard_root_abs)
    return path_abs


class OutputDirGuard:
    def __init__(self, root: str, label: str = "result directory"):
        self.root = os.path.abspath(root)
        self.label = label

    def assert_alive(self) -> None:
        assert_result_dir_alive(self.root, self.label)

    def makedirs(self, path: str) -> str:
        return guarded_makedirs(path, self.root)
