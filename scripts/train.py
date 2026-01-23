# Only training entrypoint (loads config, wires pieces, runs loop)
"""
Phase 0 PPO training entrypoint (structure-only template).

Fill in the TODO blocks with your implementation.
Keep this file as orchestration/wiring; put math-heavy code in src/*.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional

import numpy as np
import torch


# ----------------------------
# CLI + config
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)          # path to yaml
    p.add_argument("--seed", type=int, default=None)            # override config seed
    p.add_argument("--device", type=str, default=None)          # cpu/cuda override
    p.add_argument("--resume", type=str, default=None)          # checkpoint path
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    # load yaml into a dict
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in [".yaml", ".yml"]:
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config extension: {suffix} (use .yaml/.yml)")
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a mapping/dict, got {type(cfg)}")
    return cfg


def save_config(cfg: Dict[str, Any], out_path: str) -> None:
    # save a dict into yaml
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def select_device(cfg: Dict[str, Any], cli_device: Optional[str]) -> torch.device:
    # choose device based on cfg + availability + cli override
    requested = (cli_device or cfg.get("device", "auto")).lower()
    def mps_available() -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    # allow "cuda", "cuda:0", "cpu", "mps"
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"{requested} is requested, but CUDA is not available")
    if dev.type == "mps" and not mps_available():
        raise ValueError(f"{requested} is requested, but MPS is not available")
    return dev


# ----------------------------
# Small helpers (keep minimal)
# ----------------------------
def make_run_name(cfg: Dict[str, Any]) -> str:
    return f"{cfg.get("run")}{env}__{algo}{variant}__{obs}__{arch}__lr{lr}_g{gamma}_bs{bs}__steps{steps}__seed{seed}__{tag}"


def maybe_log_episode_info(logger, global_step: int, info: Dict[str, Any]) -> None:
    # TODO: if using RecordEpisodeStatistics, log episodic return/len when present
    pass


@torch.no_grad()
def policy_step(model, obs_np: np.ndarray, device: torch.device):
    """
    TODO: convert obs -> torch, call model to get:
      - action (for env.step)
      - logprob (for PPO)
      - value estimate (for GAE bootstrap)
    Return whatever your buffer expects.
    """
    raise NotImplementedError


# ----------------------------
# Main
# ----------------------------
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    args = parse_args()

    # --- apply CLI overrides ---
    # TODO: override cfg["seed"] etc. if provided
    # TODO: pick device
    device = select_device(cfg, args.device)

    # --- setup run dir + logger ---
    # TODO: create run_dir, dump resolved config, init TensorBoard/W&B logger
    run_name = make_run_name(cfg)
    run_dir = os.path.join(cfg.get("runs_dir", "runs"), run_name)
    os.makedirs(run_dir, exist_ok=True)
    # logger = ...

    # --- seeding / determinism ---
    # TODO: set python/numpy/torch seeds
    # TODO: optional best-effort determinism toggles

    # --- create env ---
    # TODO: env = make_env(env_id=..., seed=..., render_mode=None)
    env = None

    # --- build model + optimizer ---
    # TODO: model = ActorCritic(obs_space, act_space, model_cfg).to(device)
    # TODO: optimizer = Adam(model.parameters(), lr=...)
    model = None
    optimizer = None

    # --- resume (optional) ---
    # TODO: if args.resume: load model/optim + counters (+ RNG state optionally)
    global_step = 0
    update_idx = 0

    # --- rollout buffer ---
    # TODO: buf = RolloutBuffer(n_steps=..., obs_space=..., act_space=..., device=..., cfg=...)
    buf = None

    # --- reset env ---
    # TODO: obs, info = env.reset(seed=cfg["seed"])
    obs = None
    info = None

    # ----------------------------
    # Training loop (by updates)
    # ----------------------------
    # TODO: define total_env_steps, n_steps, logging cadence, checkpoint cadence
    total_env_steps = cfg["train"]["total_env_steps"]
    n_steps = cfg["rollout"]["n_steps"]

    while global_step < total_env_steps:
        # --- collect rollout ---
        # TODO: buf.reset()
        for t in range(n_steps):
            # 1) get action/logp/value from policy
            # action, logp, value = policy_step(model, obs, device)

            # 2) step env
            # next_obs, reward, terminated, truncated, info = env.step(action)

            # 3) store transition in buffer
            # buf.add(obs, action, reward, terminated, truncated, logp, value)

            # 4) update counters + episodic logging
            # global_step += 1 (or += n_envs)
            # maybe_log_episode_info(logger, global_step, info)

            # 5) handle episode end (for single env, call reset)
            # if terminated or truncated: next_obs, info = env.reset()

            # 6) advance obs
            # obs = next_obs
            pass

        # --- bootstrap value for final obs ---
        # TODO: last_value = model.value(obs)
        last_value = None

        # --- compute advantages/returns ---
        # TODO: buf.compute_returns_and_advantages(last_value, gamma, gae_lambda)
        # TODO: log advantage mean/std

        # --- PPO update (multiple epochs/minibatches) ---
        # for epoch in range(update_epochs):
        #   for batch in buf.iter_minibatches(minibatch_size, shuffle=True):
        #       metrics = ppo_update(model, optimizer, batch, ...)
        #       accumulate metrics
        #
        # update_idx += 1

        # --- log update metrics ---
        # TODO: policy loss, value loss, entropy, approx_kl, clipfrac, explained_variance, lr, grad_norm, fps

        # --- checkpoint ---
        # TODO: periodically save model/optim/counters (+ RNG state optionally)

        pass

    # --- final save + cleanup ---
    # TODO: save final checkpoint
    # TODO: close env, close logger
    return


if __name__ == "__main__":
    main()
