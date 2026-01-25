# Only training entrypoint (loads config, wires pieces, runs loop)
"""
Phase 0 PPO training entrypoint (structure-only template).

Fill in the TODO blocks with your implementation.
Keep this file as orchestration/wiring; put math-heavy code in src/*.
"""

from __future__ import annotations

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, Optional
import tensorboard

import numpy as np
import torch
from gymnasium import spaces

from src.envs.make_env import make_env
from src.utils.logging import make_logger
from src.utils.seed import seed_all
from src.models.mlp_actor_critic import MLPActorCritic


# ----------------------------
# CLI + config
# ----------------------------
def select_device(requested: str) -> torch.device:
    # choose device based on cfg + availability + cli override
    requested = requested.lower()

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
@hydra.main(
    version_base=None, config_path="../configs", config_name="ppo_cartpole.yaml"
)
def main(cfg: DictConfig) -> None:
    device = select_device(cfg.device)

    # --- setup run dir + logger ---
    # Hydra changes working dir into outputs/... by default
    run_dir = os.getcwd()
    # optional: save resolved config
    with open(os.path.join(run_dir, "config_resolved.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    device = select_device(cfg.device)
    logger = make_logger(cfg.logging.backend, run_dir)

    # --- seeding / determinism ---
    seed_all(cfg.seed, cfg.run.deterministic)

    # --- create env ---
    envs = make_env(cfg.env.id, cfg.rollout.n_envs, cfg.seed)

    # --- build model + optimizer ---
    assert isinstance(envs.single_observation_space, spaces.Box)
    assert isinstance(envs.single_action_space, spaces.Discrete)
    assert len(envs.single_observation_space.shape) == 1
    assert isinstance(envs.single_action_space.n, int)
    model = MLPActorCritic(
        envs.single_observation_space.shape[0],
        envs.single_action_space.n,
        hidden_sizes=cfg.model.hidden_sizes,
        activation=cfg.model.activation,
        shared_backbone=cfg.model.shared_backbone,
        orthogonal_init=cfg.model.orthogonal_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, eps=cfg.optim.eps)

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
