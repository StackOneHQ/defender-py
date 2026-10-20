"""Git LFS guards: the build hook and the model loaders reject unmaterialized pointers."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pytest

from stackone_defender import get_default_model_path
from stackone_defender.utils.lfs import assert_not_lfs_pointer, is_lfs_pointer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import hatch_build  # noqa: E402

POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:3f23821d74d23b39a832f6fa30615e473f23821d74d23b39a832f6fa30615e4\n"
    "size 22984052\n"
)


def _pointer(path: Path) -> Path:
    path.write_text(POINTER)
    return path


class TestPointerDetection:
    def test_detects_pointer(self, tmp_path):
        assert is_lfs_pointer(str(_pointer(tmp_path / "m.onnx")))

    def test_real_and_missing_files_are_not_pointers(self, tmp_path):
        real = tmp_path / "m.onnx"
        real.write_bytes(b"\x08\x07" * 100)
        assert not is_lfs_pointer(str(real))
        assert not is_lfs_pointer(str(tmp_path / "missing.onnx"))

    def test_assert_names_the_fix(self, tmp_path):
        with pytest.raises(RuntimeError, match="git lfs pull"):
            assert_not_lfs_pointer(str(_pointer(tmp_path / "m.onnx")))

    def test_bundled_models_are_materialized(self):
        for rel in hatch_build.MODEL_FILES:
            assert not is_lfs_pointer(str(ROOT / rel)), f"{rel} is a pointer: run `git lfs pull`"


class TestBuildHook:
    def test_rejects_pointer(self, tmp_path):
        with pytest.raises(RuntimeError, match="Git LFS pointer"):
            hatch_build.check_model_file(str(_pointer(tmp_path / "m.onnx")), 1_000)

    def test_rejects_truncated(self, tmp_path):
        small = tmp_path / "m.onnx"
        small.write_bytes(b"\x00" * 10)
        with pytest.raises(RuntimeError, match="floor"):
            hatch_build.check_model_file(str(small), 1_000)

    def test_rejects_missing(self, tmp_path):
        with pytest.raises(RuntimeError, match="missing"):
            hatch_build.check_model_file(str(tmp_path / "nope.onnx"), 1_000)

    def test_accepts_bundled_models(self):
        for rel, floor in hatch_build.MODEL_FILES.items():
            hatch_build.check_model_file(str(ROOT / rel), floor)


class TestLoaderWiring:
    @pytest.mark.model
    def test_onnx_load_reports_pointer(self, tmp_path):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("tokenizers")
        from stackone_defender.classifiers.onnx_classifier import OnnxClassifier

        model_dir = tmp_path / "model"
        shutil.copytree(get_default_model_path(), model_dir)
        _pointer(model_dir / "model_quantized.onnx")
        with pytest.raises(RuntimeError, match="git lfs pull"):
            OnnxClassifier(model_path=str(model_dir)).load_model()

    @pytest.mark.sfe
    def test_sfe_load_fails_open_with_clear_warning(self, tmp_path, caplog):
        pytest.importorskip("fasttext")
        from stackone_defender.sfe.preprocess import get_default_predictor

        with caplog.at_level(logging.WARNING):
            assert get_default_predictor(str(_pointer(tmp_path / "model.ftz"))) is None
        assert any("git lfs pull" in r.getMessage() for r in caplog.records)
