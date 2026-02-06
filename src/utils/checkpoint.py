# Save/load model+optimizer(+optional RNG) and resume training

from __future__ import annotations

import random
from typing import Union, Optional, Tuple
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch


def save_checkpoint(
    run_dir: Union[str, Path],
    file_name: Optional[str],
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # construct dictionary
    ckpt = {
        "format_version": 1,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
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
    if file_name is None:
        file_name = f"step_{global_step}"
    file_name += ".pt"
    torch.save(ckpt, checkpoint_dir / file_name)


def load_checkpoint(path: Union[str, Path]) -> dict:
    """Returns checkpoint dictionary. Can result in arbitrary code execution. Do it only if you got the file from a trusted source."""
    # load
    ckpt = torch.load(
        path, map_location="cpu", weights_only=False
    )  # Can result in arbitrary code execution. Do it only if you got the file from a trusted source.
    return ckpt


def resume_from_checkpoint(
    checkpoint_dict: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_rng_state: bool,
) -> Tuple[int, int]:
    """Restore states, returns (global_step, update_idx). Overrides optimizer LR. Will not restore run_dir."""

    def optimizer_to_device(optimizer: torch.optim.Optimizer) -> None:
        """
        Move optimizer state tensors to the device of their associated parameters.
        Call this AFTER optimizer.load_state_dict(...).
        """
        for param, state in optimizer.state.items():
            if not torch.is_tensor(param):
                continue  # very old/odd cases
            device = param.device
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=device)

    assert (
        set(("format_version", "model", "optimizer", "global_step", "update_idx"))
        <= checkpoint_dict.keys()
    ), "Keys missing in the checkpoint dictionary."
    assert checkpoint_dict["format_version"] == 1, "Unknown format version."
    model.load_state_dict(checkpoint_dict["model"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    optimizer_to_device(optimizer=optimizer)
    if resume_rng_state:
        assert (
            "rng_states" in checkpoint_dict.keys()
        ), "Resume RNG state requested, but rng_states not in checkpoint."
        rng_states = checkpoint_dict["rng_states"]
        assert isinstance(rng_states, dict)
        assert (
            set(("python_random", "numpy", "torch_cpu")) <= rng_states.keys()
        ), "Keys missing in the rng_states dictionary."
        random.setstate(rng_states["python_random"])
        np.random.set_state(rng_states["numpy"])
        torch.random.set_rng_state(rng_states["torch_cpu"])
        if "torch_cuda_all" in rng_states.keys() and torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(rng_states["torch_cuda_all"])
    assert isinstance(checkpoint_dict["global_step"], int)
    assert isinstance(checkpoint_dict["update_idx"], int)
    return checkpoint_dict["global_step"], checkpoint_dict["update_idx"]


def init_from_checkpoint(
    checkpoint_dict: dict, model: torch.nn.Module, strict: bool = True
) -> None:
    assert (
        set(("format_version", "model")) <= checkpoint_dict.keys()
    ), "Keys missing in the checkpoint dictionary."
    assert checkpoint_dict["format_version"] == 1, "Unknown format version."
    model.load_state_dict(checkpoint_dict["model"], strict=strict)
