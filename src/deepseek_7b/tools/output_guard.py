from __future__ import annotations

import os
import uuid


SENTINEL_NAME = ".meta_ebm_result_guard"


class OutputDirectoryDeletedError(RuntimeError):
    pass


def sentinel_path(path: str) -> str:
    return os.path.join(os.path.abspath(path), SENTINEL_NAME)


def initialize_result_dir_guard(path: str, label: str = "result directory") -> None:
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise OutputDirectoryDeletedError(
            f"{label} is missing at startup: {abs_path}. "
            "Create it before installing the deletion guard."
        )
    marker = sentinel_path(abs_path)
    if not os.path.exists(marker):
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(uuid.uuid4().hex + "\n")


def assert_result_dir_alive(path: str, label: str = "result directory") -> None:
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise OutputDirectoryDeletedError(
            f"{label} was removed during execution: {abs_path}. "
            "Abort instead of recreating it."
        )
    if not os.path.isfile(sentinel_path(abs_path)):
        raise OutputDirectoryDeletedError(
            f"{label} guard marker is missing: {sentinel_path(abs_path)}. "
            "The directory was likely removed and recreated; abort instead of continuing."
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
    def __init__(self, root: str, label: str = "result directory", initialize: bool = False):
        self.root = os.path.abspath(root)
        self.label = label
        if initialize:
            initialize_result_dir_guard(self.root, self.label)

    def assert_alive(self) -> None:
        assert_result_dir_alive(self.root, self.label)

    def makedirs(self, path: str) -> str:
        return guarded_makedirs(path, self.root)
