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
from typing import Any, Dict, Optional, Union
import tensorboard
from collections import defaultdict

import numpy as np
import torch
from gymnasium import spaces

from src.envs.make_env import make_env
from src.utils.logging import make_logger, Logger
from src.utils.seed import seed_all
from src.models.factory import make_model
from src.buffers.rollout_buffer import RolloutBuffer
from src.algo.factory import make_learner


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
def log_episode_info(logger: Logger, global_step: int, info: Dict[str, Any]) -> None:
    # log episodic return/len when present
    if "episode" not in info or "_episode" not in info:
        return
    idx = np.flatnonzero(info["_episode"])
    if idx.size == 0:
        return
    rets = info["episode"]["r"][idx].astype(np.float32)
    lens = info["episode"]["l"][idx].astype(np.int32)
    print(f"got episodic return: {float(rets.mean())} at global step: {global_step}")
    logger.log_scalar("episodic_return_mean", float(rets.mean()), global_step)
    logger.log_scalar("episodic_length_mean", float(lens.mean()), global_step)


def log_advantage_mean_std(
    logger: Logger, buf: RolloutBuffer, global_step: int
) -> None:
    with torch.no_grad():
        if buf.advantages is None or buf.advantages_norm is None:
            raise RuntimeError(
                "Must call compute_returns_and_advantages() before log_advantage_mean_std()"
            )
        adv = buf.advantages.reshape(-1)
        adv_n = buf.advantages_norm.reshape(-1)
        assert torch.isfinite(adv).all()
        assert torch.isfinite(adv_n).all()
        logger.log_scalar("adv_mean_raw", adv.mean().item(), global_step)
        logger.log_scalar("adv_std_raw", adv.std(unbiased=False).item(), global_step)
        logger.log_scalar("adv_mean_norm", adv_n.mean().item(), global_step)
        logger.log_scalar("adv_std_norm", adv_n.std(unbiased=False).item(), global_step)


def log_update_metrics(logger: Logger, global_step:int,  metrics: dict, early_stop: bool) -> None:
    logger.log_scalar("early_stop", int(early_stop), global_step)
    for k, v in metrics.items():
        logger.log_scalar(k, v, global_step)



def to_torch(
    x: Any, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    # torch.as_tensor avoids an extra copy on CPU when possible;
    # specifying device will move/copy to GPU if needed.
    t = torch.as_tensor(x, device=device)
    return t if dtype is None else t.to(dtype)


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
    # --- setup run dir + logger ---
    # Hydra changes working dir into outputs/... by default
    run_dir = Path(cfg.run.run_dir)
    # optional: save resolved config (don't need since hydra will save config)
    # with open(os.path.join(run_dir, "config_resolved.yaml"), "w") as f:
    #     f.write(OmegaConf.to_yaml(cfg))
    debug = cfg.debug
    device = select_device(cfg.device)
    logger = make_logger(cfg.logging.backend, run_dir)

    # --- aliasing frequently used configs ---
    T = cfg.rollout.n_steps
    N = cfg.rollout.n_envs

    # --- seeding / determinism ---
    seed_all(cfg.seed, cfg.run.deterministic)

    # --- create env ---
    envs = make_env(id=cfg.env.id, n_envs=N, seed=cfg.seed, capture_video=cfg.env.capture_video, video_folder=run_dir / "videos", human_render=cfg.env.human_render)

    # --- build model + optimizer + learner ---
    model = make_model(
        cfg.model, envs.single_observation_space, envs.single_action_space
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, eps=cfg.optim.eps)
    learner = make_learner(cfg=cfg, model=model, optim=optimizer)

    # --- resume (optional) ---
    # TODO: if args.resume: load model/optim + counters (+ RNG state optionally)
    global_step = 0
    update_idx = 0

    # --- rollout buffer ---
    assert isinstance(envs.single_observation_space, spaces.Box)
    buf = RolloutBuffer(T, N, envs.single_observation_space.shape)

    # --- reset env ---
    obs_np, info = envs.reset(seed=cfg.seed)
    episode_start = torch.tensor([True] * N, dtype=torch.bool, device=device)
    next_value = torch.empty(
        N, device=device, dtype=torch.float32
    )  # will be finalized later
    next_value[:] = torch.nan  # optional (helps catch bugs)
    done = torch.empty(N, device=device, dtype=torch.bool)
    done[:] = False

    # ----------------------------
    # Training loop (by updates)
    # ----------------------------
    total_env_steps = cfg.train.total_env_steps
    logging_cadence = None
    checkpoint_cadence = None

    while global_step < total_env_steps:
        # --- collect rollout ---
        with torch.no_grad():
            buf.reset()
            for t in range(T):
                # 1) get action/logp/value from policy
                obs = to_torch(obs_np, device=device, dtype=torch.float32)
                action, logp, v, _ = model.get_action_and_value(obs)

                # 2) step env
                next_obs_np, reward_np, terminated_np, truncated_np, info = envs.step(
                    action.detach().cpu().numpy()
                )
                reward = to_torch(reward_np, device=device, dtype=torch.float32)
                terminated = to_torch(terminated_np, device=device, dtype=torch.bool)
                truncated = to_torch(truncated_np, device=device, dtype=torch.bool)
                timeout = truncated & (
                    ~terminated
                )  # move to info["TimeLimit.truncated"] when available
                done = terminated | truncated

                # 2.1) get next_value when VectorEnv auto-reset at same step
                next_value[:] = torch.nan  # optional (helps catch bugs)
                if done.any():
                    if debug:
                        assert "final_obs" in info and "_final_obs" in info
                    mask = info["_final_obs"].astype(
                        bool
                    )  # should match done for same_step
                    final_obs_np = np.stack(info["final_obs"][mask])
                    final_obs = to_torch(
                        final_obs_np, device=device, dtype=torch.float32
                    )
                    final_value = model.get_value(final_obs)
                    next_value[mask] = final_value

                # 3) store transition in buffer
                buf.add(
                    obs,
                    action,
                    logp,
                    v,
                    next_value,
                    reward,
                    terminated,
                    truncated,
                    done,
                    timeout,
                    episode_start,
                )

                # 4) update counters + episodic logging
                global_step += N
                # TODO: log terminated_frac, truncated_frac (counts / total transitions)
                log_episode_info(logger=logger, global_step=global_step, info=info)

                # 5) handle episode end

                # 6) advance obs
                obs_np = next_obs_np

                # 7) set episode_start
                episode_start = done

            # --- bootstrap value for final obs ---
            last_obs = to_torch(obs_np, device=device, dtype=torch.float32)
            last_value = model.get_value(last_obs)
            mask = ~done
            buf.finalize(last_value=last_value)

            # --- compute advantages/returns ---
            if debug:
                assert buf.t == T  # make sure rollout buffer is full
            buf.compute_returns_and_advantages(
                gamma=cfg.ppo.gamma,
                gae_lambda=cfg.ppo.gae_lambda,
                normalize_advantages=cfg.ppo.normalize_advantages,
                eps=cfg.ppo.eps,
            )
            # log advantage mean/std
            log_advantage_mean_std(logger=logger, buf=buf, global_step=global_step)

        # --- PPO update (multiple epochs/minibatches) ---
        metrics_weighted_sum = defaultdict(float)
        metrics_count = 0
        early_stop = False
        for epoch in range(cfg.ppo.update_epochs):
            kl_sum = 0
            kl_count = 0
            for mini_batch in buf.iter_minibatches(
                cfg.ppo.minibatch_size, shuffle=True, device=device
            ):
                metrics = learner.update(mini_batch=mini_batch)
                # accumulate metrics
                for k, v in metrics.items():
                    metrics_weighted_sum[k] += v * float(mini_batch.batch_size)
                metrics_count += mini_batch.batch_size
                kl_sum += metrics["approx_kl"]
                kl_count += 1
            # early-stop by KL
            if cfg.ppo.target_kl is not None and kl_count / kl_sum > cfg.ppo.target_kl:
                early_stop = True
                break
        metrics_weighted_mean = {
            k: v / metrics_count for k, v in metrics_weighted_sum.items()
        }
        update_idx += 1

        # --- log update metrics ---
        log_update_metrics(logger=logger, global_step=global_step, metrics=metrics_weighted_mean, early_stop=early_stop)
        # TODO: fps

        # --- checkpoint ---
        # TODO: periodically save model/optim/counters (+ RNG state optionally)

        pass

    # --- final save + cleanup ---
    # TODO: save final checkpoint

    # close env, close logger
    envs.close()
    logger.close()
    return


if __name__ == "__main__":
    main()
