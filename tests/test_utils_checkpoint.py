# Unit tests for src/utils/checkpoint.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    resume_from_checkpoint,
    init_from_checkpoint,
)
from src.models.mlp_actor_critic import MLPActorCritic
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = spaces.Discrete(2)
    return MLPActorCritic(
        single_obs_space=obs_space,
        single_act_space=act_space,
        hidden_sizes=(8,),
        orthogonal_init=False,
    )


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def dummy_cfg():
    return OmegaConf.create({"seed": 42, "device": "cpu", "train": {"total_env_steps": 1000}})


@pytest.fixture
def saved_ckpt_path(tmp_path, model, optimizer, dummy_cfg):
    """Save a checkpoint and return its path."""
    save_checkpoint(
        run_dir=tmp_path,
        file_name="test_ckpt",
        cfg=dummy_cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        global_step=500,
        update_idx=10,
        save_rng=False,
    )
    return tmp_path / "checkpoints" / "test_ckpt.pt"


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_checkpoint_file_created(self, saved_ckpt_path):
        assert saved_ckpt_path.exists()

    def test_load_returns_dict(self, saved_ckpt_path):
        ckpt = load_checkpoint(saved_ckpt_path)
        assert isinstance(ckpt, dict)

    def test_load_has_required_keys(self, saved_ckpt_path):
        ckpt = load_checkpoint(saved_ckpt_path)
        required = {"format_version", "cfg", "model", "optimizer", "lr_scheduler", "global_step", "update_idx"}
        assert required <= ckpt.keys()

    def test_load_global_step(self, saved_ckpt_path):
        ckpt = load_checkpoint(saved_ckpt_path)
        assert ckpt["global_step"] == 500

    def test_load_update_idx(self, saved_ckpt_path):
        ckpt = load_checkpoint(saved_ckpt_path)
        assert ckpt["update_idx"] == 10

    def test_load_format_version(self, saved_ckpt_path):
        ckpt = load_checkpoint(saved_ckpt_path)
        assert ckpt["format_version"] == 1

    def test_model_weights_roundtrip(self, tmp_path, model, optimizer, dummy_cfg):
        """Saved model weights should match loaded model weights exactly."""
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(tmp_path, "w_test", dummy_cfg, model, optimizer, None, 0, 0, False)
        ckpt = load_checkpoint(tmp_path / "checkpoints" / "w_test.pt")
        for k, v in ckpt["model"].items():
            assert torch.allclose(v, original_sd[k])

    def test_default_filename_uses_step(self, tmp_path, model, optimizer, dummy_cfg):
        """When file_name=None, the file should be named step_{global_step}.pt."""
        save_checkpoint(tmp_path, None, dummy_cfg, model, optimizer, None, 1234, 0, False)
        assert (tmp_path / "checkpoints" / "step_1234.pt").exists()

    def test_save_rng_state_included(self, tmp_path, model, optimizer, dummy_cfg):
        save_checkpoint(tmp_path, "rng_test", dummy_cfg, model, optimizer, None, 0, 0, save_rng=True)
        ckpt = load_checkpoint(tmp_path / "checkpoints" / "rng_test.pt")
        assert "rng_states" in ckpt
        assert "python_random" in ckpt["rng_states"]
        assert "numpy" in ckpt["rng_states"]
        assert "torch_cpu" in ckpt["rng_states"]

    def test_save_no_rng_state_excluded(self, saved_ckpt_path):
        ckpt = load_checkpoint(saved_ckpt_path)
        assert "rng_states" not in ckpt

    def test_wrong_format_version_raises(self, tmp_path):
        bad_ckpt = {
            "format_version": 99,
            "model": {},
            "optimizer": {},
            "lr_scheduler": None,
            "global_step": 0,
            "update_idx": 0,
            "cfg": {},
        }
        path = tmp_path / "bad.pt"
        torch.save(bad_ckpt, path)
        ckpt = load_checkpoint(path)
        with pytest.raises(AssertionError):
            resume_from_checkpoint(ckpt, torch.nn.Linear(1, 1), torch.optim.SGD([torch.zeros(1)], lr=0.01), None, False)


# ---------------------------------------------------------------------------
# resume_from_checkpoint
# ---------------------------------------------------------------------------


class TestResume:
    def test_restores_model_weights(self, tmp_path, model, optimizer, dummy_cfg):
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(tmp_path, "resume_test", dummy_cfg, model, optimizer, None, 100, 5, False)
        # Corrupt model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)
        ckpt = load_checkpoint(tmp_path / "checkpoints" / "resume_test.pt")
        resume_from_checkpoint(ckpt, model, optimizer, lr_scheduler=None, resume_rng_state=False)
        for k, v in model.state_dict().items():
            assert torch.allclose(v, original_sd[k])

    def test_returns_correct_counters(self, tmp_path, model, optimizer, dummy_cfg):
        save_checkpoint(tmp_path, "counters_test", dummy_cfg, model, optimizer, None, 777, 33, False)
        ckpt = load_checkpoint(tmp_path / "checkpoints" / "counters_test.pt")
        global_step, update_idx = resume_from_checkpoint(ckpt, model, optimizer, None, False)
        assert global_step == 777
        assert update_idx == 33

    def test_missing_keys_raises(self, model, optimizer):
        bad_ckpt = {"format_version": 1, "model": model.state_dict()}
        with pytest.raises(AssertionError):
            resume_from_checkpoint(bad_ckpt, model, optimizer, None, False)

    def test_resume_rng_state_restores_torch(self, tmp_path, model, optimizer, dummy_cfg):
        """After restoring RNG state, torch.rand should reproduce the same sequence."""
        torch.manual_seed(123)
        ref = torch.rand(5).clone()
        torch.manual_seed(123)  # reset to same state before saving
        save_checkpoint(tmp_path, "rng_resume", dummy_cfg, model, optimizer, None, 0, 0, save_rng=True)
        # Advance RNG
        _ = torch.rand(100)
        ckpt = load_checkpoint(tmp_path / "checkpoints" / "rng_resume.pt")
        resume_from_checkpoint(ckpt, model, optimizer, lr_scheduler=None, resume_rng_state=True)
        restored = torch.rand(5)
        assert torch.allclose(ref, restored)

    def test_resume_rng_requested_but_missing_raises(self, model, optimizer):
        ckpt = {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": None,
            "global_step": 0,
            "update_idx": 0,
        }
        with pytest.raises(AssertionError):
            resume_from_checkpoint(ckpt, model, optimizer, lr_scheduler=None, resume_rng_state=True)


# ---------------------------------------------------------------------------
# init_from_checkpoint
# ---------------------------------------------------------------------------


class TestInitFromCheckpoint:
    def test_loads_model_weights(self, tmp_path, model, optimizer, dummy_cfg):
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(tmp_path, "init_test", dummy_cfg, model, optimizer, None, 0, 0, False)
        # Corrupt model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)
        ckpt = load_checkpoint(tmp_path / "checkpoints" / "init_test.pt")
        init_from_checkpoint(ckpt, model)
        for k, v in model.state_dict().items():
            assert torch.allclose(v, original_sd[k])

    def test_does_not_restore_optimizer(self, tmp_path, model, optimizer, dummy_cfg):
        """init_from_checkpoint should only touch the model, not the optimizer."""
        # Do a gradient step to give the optimizer non-trivial state
        obs = torch.randn(4, 4)
        loss = model.get_value(obs).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optim_sd_before = {k: v for k, v in optimizer.state_dict().items()}

        save_checkpoint(tmp_path, "init_optim_test", dummy_cfg, model, optimizer, None, 0, 0, False)
        ckpt = load_checkpoint(tmp_path / "checkpoints" / "init_optim_test.pt")

        # Create a fresh optimizer (no state)
        fresh_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        init_from_checkpoint(ckpt, model)
        # Fresh optimizer state should still be empty (no step taken)
        assert len(fresh_optim.state) == 0

    def test_missing_keys_raises(self, model):
        with pytest.raises(AssertionError):
            init_from_checkpoint({"format_version": 1}, model)

    def test_wrong_format_version_raises(self, model):
        ckpt = {"format_version": 2, "model": model.state_dict()}
        with pytest.raises(AssertionError):
            init_from_checkpoint(ckpt, model)
