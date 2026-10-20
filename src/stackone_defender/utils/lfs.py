"""Git LFS pointer detection for bundled model files.

Models are tracked with Git LFS. A checkout without ``git lfs pull`` (or without git-lfs
installed) leaves a ~130-byte text pointer in place of the real binary, and loading that as a
model fails with a cryptic parse error. Detect it and fail with an actionable message instead.
"""

from __future__ import annotations

LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path: str) -> bool:
    """True if ``path`` starts with the Git LFS pointer header (not the real binary)."""
    try:
        with open(path, "rb") as f:
            return f.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC
    except OSError:
        return False  # unreadable files aren't pointers; let the caller surface the error


def assert_not_lfs_pointer(path: str) -> None:
    """Raise a clear, actionable error if ``path`` is an unmaterialized Git LFS pointer."""
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"model at {path} is a Git LFS pointer, not the real file. "
            "Run `git lfs pull` (install git-lfs first if needed)."
        )
