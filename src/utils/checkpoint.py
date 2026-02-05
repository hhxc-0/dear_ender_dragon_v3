# Save/load model+optimizer(+optional RNG) and resume training

from __future__ import annotations

import random
from typing import Union, Optional
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import torch


def save_checkpoint(
    run_dir: Union[str, Path],
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    update_idx: int,
    save_rng: bool,
):
    # make dir
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    # construct dictionary
    ckpt = {
        "format_version": 1,
        "cfg": cfg,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "update_idx": update_idx,
    }
    if save_rng:
        rng_states = {
            "python_random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["torch_cuda_all"] = torch.cuda.random.get_rng_state_all()
        ckpt["rng_states"] = rng_states
    # save
    file_name = f"step_{global_step}.pt"
    torch.save(ckpt, checkpoint_dir / file_name)


def load_checkpoint(path: Union[str, Path]):
    # load
    ckpt = torch.load(path)


def restore_from_checkpoint(
    checkpoint_dict: dict, model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    pass


def load_last_checkpoint(run_dir: Union[str, Path]):
    pass
