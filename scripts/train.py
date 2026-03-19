# Only training entrypoint (loads config, wires pieces, runs loop)
"""
Phase 0 PPO training entrypoint.

Keep this file as orchestration/wiring; put math-heavy code in src/*.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast, Any
import hydra
from omegaconf import DictConfig, OmegaConf
import tensorboard
from collections import defaultdict

import numpy as np
import torch
from gymnasium import spaces

from src.envs.make_env import make_env
from src.utils.logging import make_logger, Logger, log_episode_info
from src.utils.seed import seed_all
from src.utils.device import select_device
from src.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    resume_from_checkpoint,
    init_from_checkpoint,
)
from src.models.factory import make_model
from src.buffers.rollout_buffer import RolloutBuffer
from src.algo.factory import make_learner
from src.utils.tensor import to_torch


# ----------------------------
# Small helpers (keep minimal)
# ----------------------------
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
        logger.log_scalar("advantages/mean_raw", adv.mean().item(), global_step)
        logger.log_scalar(
            "advantages/std_raw", adv.std(unbiased=False).item(), global_step
        )
        logger.log_scalar("advantages/mean_norm", adv_n.mean().item(), global_step)
        logger.log_scalar(
            "advantages/std_norm", adv_n.std(unbiased=False).item(), global_step
        )


def log_sps(
    logger: Logger,
    global_step: int,
    starting_step: int,
    starting_time: float,
    rollout_time: float,
    ending_time: float,
    print_sps: bool,
) -> None:
    delta_step = global_step - starting_step
    rollout_delta_time = rollout_time - starting_time
    total_delta_time = ending_time - starting_time
    logger.log_scalar("timing/sps_env", delta_step / rollout_delta_time, global_step)
    logger.log_scalar("timing/sps_total", delta_step / total_delta_time, global_step)
    logger.log_scalar("timing/rollout_sec", rollout_delta_time, global_step)
    logger.log_scalar("timing/update_sec", ending_time - rollout_time, global_step)
    if print_sps:
        print(f"SPS: {delta_step / rollout_delta_time}")


def log_episode_end_metrics(
    logger: Logger,
    global_step: int,
    completed_episodes: int,
    terminated_episodes: int,
    truncated_episodes: int,
    timeout_episodes: int,
) -> None:
    logger.log_scalar("episodes/completed", completed_episodes, global_step)
    logger.log_scalar("episodes/terminated", terminated_episodes, global_step)
    logger.log_scalar("episodes/truncated", truncated_episodes, global_step)
    logger.log_scalar("episodes/timeout", timeout_episodes, global_step)

    if completed_episodes > 0:
        logger.log_scalar(
            "episodes/truncation_rate",
            truncated_episodes / completed_episodes,
            global_step,
        )
        logger.log_scalar(
            "episodes/timeout_rate",
            timeout_episodes / completed_episodes,
            global_step,
        )
        logger.log_scalar(
            "episodes/termination_rate",
            terminated_episodes / completed_episodes,
            global_step,
        )


def log_update_metrics(
    logger: Logger, global_step: int, metrics: dict, early_stop: bool
) -> None:
    logger.log_scalar("update/early_stop", int(early_stop), global_step)
    for k, v in metrics.items():
        logger.log_scalar(k, v, global_step)


# ----------------------------
# Main
# ----------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # --- setup run dir + logger ---
    run_dir = Path(cfg.run.run_dir)
    debug = cfg.debug
    logger = make_logger(cfg.logging.backend, run_dir)

    # --- select device ---
    device = select_device(cfg.device)

    # --- aliasing frequently used configs ---
    T = cfg.rollout.n_steps
    N = cfg.rollout.n_envs
    total_env_steps = cfg.train.total_env_steps
    num_updates = np.ceil(total_env_steps / (N * T))

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

    # --- build model + optimizer + learner + LR scheduler ---
    model = make_model(
        cfg.model, envs.single_observation_space, envs.single_action_space
    ).to(device)
    if cfg.optim.name == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=cfg.optim.lr, eps=cfg.optim.eps
        )
    else:
        raise NotImplementedError("Unknown optimizer")
    learner = make_learner(cfg=cfg, model=model, optim=optimizer)
    if not cfg.lr_scheduler.name:
        lr_scheduler = None
    elif cfg.lr_scheduler.name == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=cfg.lr_scheduler.start_factor,
            end_factor=cfg.lr_scheduler.end_factor,
            total_iters=num_updates,
        )
    else:
        raise NotImplementedError("Unknown LR scheduler")

    # --- resume (optional) ---
    global_step = 0
    update_idx = 0
    next_checkpoint_step = cfg.checkpoint.save_every_steps
    if cfg.runtime.resume.mode is not None:
        assert cfg.runtime.resume.path is not None
        ckpt = load_checkpoint(cfg.runtime.resume.path)
        if cfg.runtime.resume.mode == "resume":
            global_step, update_idx = resume_from_checkpoint(
                checkpoint_dict=ckpt,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                resume_rng_state=cfg.runtime.resume.resume_rng_state,
            )
            next_checkpoint_step = (
                global_step // cfg.checkpoint.save_every_steps + 1
            ) * cfg.checkpoint.save_every_steps
        elif cfg.runtime.resume.mode == "init_from":
            init_from_checkpoint(checkpoint_dict=ckpt, model=model)
        else:
            raise ValueError("Unknown resume mode.")

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

    while global_step < total_env_steps:
        # --- reset SPS counters ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        starting_step = global_step
        starting_time = time.perf_counter()

        # --- reset episode counters ---
        completed_episodes = 0
        terminated_episodes = 0
        truncated_episodes = 0
        timeout_episodes = 0

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
                completed_episodes += int(done.sum().item())
                terminated_episodes += int(terminated.sum().item())
                truncated_episodes += int(truncated.sum().item())
                timeout_episodes += int(timeout.sum().item())

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
                log_episode_info(
                    logger=logger,
                    step=global_step,
                    info=info,
                    print_return=False,
                )

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
                gamma=cfg.algo.gamma,
                gae_lambda=cfg.algo.gae_lambda,
                normalize_advantages=cfg.algo.normalize_advantages,
                eps=cfg.algo.eps,
            )
            # log advantage mean/std
            log_advantage_mean_std(logger=logger, buf=buf, global_step=global_step)

        # --- log timer ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        rollout_time = time.perf_counter()

        # --- PPO update (multiple epochs/minibatches) ---
        metrics_weighted_sum = defaultdict(float)
        metrics_count = 0
        early_stop = False
        for epoch in range(cfg.algo.update_epochs):
            kl_sum = 0
            kl_count = 0
            for mini_batch in buf.iter_minibatches(
                cfg.algo.minibatch_size, shuffle=True, device=device
            ):
                single_update_metrics = learner.update(mini_batch=mini_batch)
                # accumulate metrics
                for k, v in single_update_metrics.items():
                    metrics_weighted_sum[k] += v * float(mini_batch.batch_size)
                metrics_count += mini_batch.batch_size
                kl_sum += single_update_metrics["ppo/approx_kl"]
                kl_count += 1
            # early-stop by KL
            if (
                cfg.algo.target_kl is not None
                and kl_sum / kl_count > cfg.algo.target_kl
            ):
                early_stop = True
                break
        # weighted mean
        metrics_weighted_mean = {
            k: v / metrics_count for k, v in metrics_weighted_sum.items()
        }
        update_idx += 1

        # --- step LR scheduler
        if lr_scheduler:
            lr_scheduler.step()

        # --- log update metrics ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        log_sps(
            logger=logger,
            global_step=global_step,
            starting_step=starting_step,
            starting_time=starting_time,
            rollout_time=rollout_time,
            ending_time=time.perf_counter(),
            print_sps=True,
        )
        log_episode_end_metrics(
            logger=logger,
            global_step=global_step,
            completed_episodes=completed_episodes,
            terminated_episodes=terminated_episodes,
            truncated_episodes=truncated_episodes,
            timeout_episodes=timeout_episodes,
        )
        log_update_metrics(
            logger=logger,
            global_step=global_step,
            metrics=metrics_weighted_mean,
            early_stop=early_stop,
        )

        # --- checkpoint ---
        # periodically save model/optim/counters (+ RNG state optionally)
        if cfg.checkpoint.save_every_steps and global_step >= next_checkpoint_step:
            save_checkpoint(
                run_dir=run_dir,
                file_name=None,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                global_step=global_step,
                update_idx=update_idx,
                save_rng=cfg.checkpoint.save_rng_state,
            )
            next_checkpoint_step += cfg.checkpoint.save_every_steps

    # --- final save + cleanup ---
    if cfg.checkpoint.final_save:
        save_checkpoint(
            run_dir=run_dir,
            file_name=f"final_step_{global_step}",
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            global_step=global_step,
            update_idx=update_idx,
            save_rng=cfg.checkpoint.save_rng_state,
        )

    # close env, close logger
    envs.close()
    logger.close()
    return


if __name__ == "__main__":
    main()
