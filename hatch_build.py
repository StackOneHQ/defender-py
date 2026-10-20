"""Build guard: never package a Git LFS pointer, a truncated model, or a missing one.

Runs on every wheel and sdist build (CI, the release job, and local ``uv build``).
"""

from __future__ import annotations

import os
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"
# Bundled model files and the smallest plausible real size for each.
MODEL_FILES = {
    "src/stackone_defender/models/minilm-multihead-v5/model_quantized.onnx": 1_000_000,
    "src/stackone_defender/sfe/model.ftz": 100_000,
}


def check_model_file(path: str, min_bytes: int) -> None:
    """Raise if ``path`` is missing, a Git LFS pointer, or smaller than ``min_bytes``."""
    if not os.path.isfile(path):
        raise RuntimeError(f"model file missing: {path}")
    with open(path, "rb") as f:
        head = f.read(len(LFS_POINTER_MAGIC))
    if head == LFS_POINTER_MAGIC:
        raise RuntimeError(f"{path} is a Git LFS pointer, not the model. Run `git lfs pull` before building.")
    size = os.path.getsize(path)
    if size < min_bytes:
        raise RuntimeError(f"{path} is {size} bytes, below the {min_bytes}-byte floor; the model looks truncated.")


class ModelGuardHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        for rel, min_bytes in MODEL_FILES.items():
            check_model_file(os.path.join(self.root, rel), min_bytes)
