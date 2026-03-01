# Unit tests for src/utils/logging.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.logging import NoOpLogger, TensorboardLogger, make_logger


# ---------------------------------------------------------------------------
# make_logger dispatch
# ---------------------------------------------------------------------------


class TestMakeLogger:
    def test_none_returns_noop(self, tmp_path):
        logger = make_logger(None, tmp_path)
        assert isinstance(logger, NoOpLogger)

    def test_tensorboard_returns_tb(self, tmp_path):
        logger = make_logger("tensorboard", tmp_path)
        assert isinstance(logger, TensorboardLogger)
        logger.close()

    def test_unknown_backend_raises(self, tmp_path):
        with pytest.raises(ValueError):
            make_logger("wandb", tmp_path)


# ---------------------------------------------------------------------------
# NoOpLogger
# ---------------------------------------------------------------------------


class TestNoOpLogger:
    @pytest.fixture
    def noop(self):
        return NoOpLogger()

    def test_log_scalar_returns_none(self, noop):
        assert noop.log_scalar("loss", 0.5, 1) is None

    def test_log_scalars_returns_none(self, noop):
        assert noop.log_scalars({"a": 1.0, "b": 2.0}, step=1) is None

    def test_log_text_returns_none(self, noop):
        assert noop.log_text("note", "hello", 1) is None

    def test_log_config_returns_none(self, noop):
        assert noop.log_config({"key": "value"}) is None

    def test_log_artifact_returns_none(self, noop, tmp_path):
        f = tmp_path / "dummy.txt"
        f.write_text("hi")
        assert noop.log_artifact(f) is None

    def test_close_returns_none(self, noop):
        assert noop.close() is None

    def test_no_method_raises(self, noop, tmp_path):
        """All NoOpLogger methods should be callable without raising."""
        f = tmp_path / "artifact.txt"
        f.write_text("data")
        noop.log_scalar("x", 1.0, 0)
        noop.log_scalars({"x": 1.0}, 0, prefix="train")
        noop.log_text("t", "text", 0)
        noop.log_config({"a": 1})
        noop.log_artifact(f)
        noop.close()


# ---------------------------------------------------------------------------
# TensorboardLogger
# ---------------------------------------------------------------------------


class TestTensorboardLogger:
    @pytest.fixture
    def tb(self, tmp_path):
        logger = TensorboardLogger(tmp_path)
        yield logger
        logger.close()

    def test_log_scalar_no_error(self, tb):
        tb.log_scalar("loss", 0.5, step=1)

    def test_log_scalars_no_error(self, tb):
        tb.log_scalars({"a": 1.0, "b": 2.0}, step=1)

    def test_log_scalars_with_prefix_no_slash(self, tb):
        """Prefix without trailing slash should be handled gracefully."""
        tb.log_scalars({"loss": 0.1}, step=1, prefix="train")

    def test_log_scalars_with_prefix_with_slash(self, tb):
        tb.log_scalars({"loss": 0.1}, step=1, prefix="train/")

    def test_log_text_no_error(self, tb):
        tb.log_text("note", "hello world", step=0)

    def test_log_config_creates_json_file(self, tb, tmp_path):
        cfg = {"seed": 42, "lr": 3e-4}
        tb.log_config(cfg)
        config_file = tmp_path / "config.json"
        assert config_file.exists()
        loaded = json.loads(config_file.read_text())
        assert loaded["seed"] == 42

    def test_log_config_json_is_valid(self, tb, tmp_path):
        tb.log_config({"nested": {"a": 1}})
        content = (tmp_path / "config.json").read_text()
        parsed = json.loads(content)
        assert parsed["nested"]["a"] == 1

    def test_log_artifact_copies_file(self, tb, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("artifact content")
        tb.log_artifact(src)
        dst = tmp_path / "source.txt"
        assert dst.exists()
        assert dst.read_text() == "artifact content"

    def test_log_artifact_custom_name(self, tb, tmp_path):
        src = tmp_path / "original.txt"
        src.write_text("data")
        tb.log_artifact(src, name="renamed.txt")
        assert (tmp_path / "renamed.txt").exists()

    def test_log_artifact_same_src_dst_no_error(self, tb, tmp_path):
        """Copying a file to itself (src == dst) should not raise."""
        src = tmp_path / "same.txt"
        src.write_text("content")
        # log_artifact with no name → dst = run_dir / src.name = src itself
        tb.log_artifact(src)  # should not raise

    def test_close_no_error(self, tmp_path):
        logger = TensorboardLogger(tmp_path)
        logger.close()  # should not raise
