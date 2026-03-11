# Loads checkpoint, runs fixed-episode eval (+ optional video)

from __future__ import annotations

from pathlib import Path
from typing import cast, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import tensorboard

import numpy as np
import torch

from src.envs.make_env import make_env
from src.utils.logging import make_logger, log_episode_info
from src.utils.seed import seed_all
from src.utils.device import select_device
from src.utils.checkpoint import load_checkpoint, init_from_checkpoint
from src.models.factory import make_model
from src.utils.tensor import to_torch


# ----------------------------
# Small helpers (keep minimal)
# ----------------------------


# ----------------------------
# Main
# ----------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # --- setup run dir + logger ---
    run_dir = Path(cfg.run.run_dir)
    debug = cfg.debug
    logger = make_logger(cfg.logging.backend, run_dir)

    # --- select device ---
    device = select_device(cfg.device)

    # --- aliasing frequently used configs ---
    N = cfg.n_envs

    # --- seeding / determinism ---
    seed_all(cfg.seed, cfg.run.deterministic)

    # --- create env ---
    env_kwargs = (
        OmegaConf.to_container(cfg.env.kwargs, resolve=True)
        if "kwargs" in cfg.env
        else None
    )
    assert env_kwargs is None or isinstance(env_kwargs, dict)
    env_kwargs = cast(dict[str, Any] | None, env_kwargs)
    envs = make_env(
        env_id=cfg.env.id,
        env_kwargs=env_kwargs,
        n_envs=N,
        seed=cfg.seed,
        flatten_observation=cfg.env.flatten_observation,
        normalize_pixel_observation=cfg.env.normalize_pixel_observation,
        capture_video=cfg.env.capture_video,
        video_folder=run_dir / "videos",
        human_render=cfg.env.human_render,
    )

    # --- build model ---
    model = make_model(
        cfg.model, envs.single_observation_space, envs.single_action_space
    ).to(device)
    model.eval()

    # --- load checkpoint ---
    ckpt = load_checkpoint(cfg.checkpoint_path)
    init_from_checkpoint(checkpoint_dict=ckpt, model=model)

    # ----------------------------
    # Eval loop (fixed episodes)
    # ----------------------------
    with torch.no_grad():
        # --- reset env ---
        obs_np, info = envs.reset(seed=cfg.seed)
        done = torch.empty(N, device=device, dtype=torch.bool)
        done[:] = False
        n_finished_episodes: int = 0
        episodic_returns: list[float] = []
        episodic_lengths: list[int] = []

        while n_finished_episodes < cfg.n_episodes:
            # get action/logp/value from policy
            obs = to_torch(obs_np, device=device, dtype=torch.float32)
            action, logp, v, _ = model.get_action_and_value(obs)

            # step env
            next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(
                action.detach().cpu().numpy()
            )
            terminated = to_torch(terminated_np, device=device, dtype=torch.bool)
            truncated = to_torch(truncated_np, device=device, dtype=torch.bool)
            done = terminated | truncated

            # log episodic return/len when present
            episode_results = log_episode_info(
                logger=logger, step=n_finished_episodes, info=info, print_return=True
            )
            if episode_results is not None:
                rets, lens = episode_results
                if debug:
                    assert len(rets) == len(lens)
                n_finished_episodes += len(rets)
                episodic_returns.extend(rets.tolist())
                episodic_lengths.extend(lens.tolist())

            # advance obs
            obs_np = next_obs_np

    # --- aggregate + report ---
    episodic_returns_np = np.asarray(
        episodic_returns[: cfg.n_episodes], dtype=np.float32
    )
    episodic_lengths_np = np.asarray(episodic_lengths[: cfg.n_episodes], dtype=np.int32)
    eval_result = {
        "episodic_return_mean": episodic_returns_np.mean(),
        "episodic_return_std": episodic_returns_np.std(),
        "episodic_return_min": episodic_returns_np.min(),
        "episodic_return_max": episodic_returns_np.max(),
        "episodic_length_mean": episodic_lengths_np.mean(),
        "episodic_length_std": episodic_lengths_np.std(),
    }
    print(f"\n----- evaluation result -----")
    for key, value in eval_result.items():
        print(f"{key}: {value}")

    # --- cleanup ---
    envs.close()


if __name__ == "__main__":
    main()
