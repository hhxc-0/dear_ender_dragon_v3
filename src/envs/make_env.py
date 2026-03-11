# Create env, seed reset(), optionally record stats/video in eval

from __future__ import annotations

from typing import Optional, Union, Mapping, Any, cast
from pathlib import Path
import gymnasium as gym

import highway_env

from .wrappers import NormalizePixelObservation


def make_env(
    env_id: str,
    env_kwargs: Optional[Mapping[str, Any]] = None,
    n_envs: int = 1,
    seed: int = 42,
    flatten_observation: bool = False,
    normalize_pixel_observation: bool = False,
    capture_video: bool = False,
    video_folder: Optional[Union[str, Path]] = None,
    human_render: bool = False,
):
    assert not (
        capture_video and human_render
    ), "capture_video and human_render cannot both be set to True."
    env_kwargs = dict(env_kwargs or {})
    assert "render_mode" not in env_kwargs.keys()

    def convert_config_value(x: Any) -> Any:
        if isinstance(x, dict):
            if x.get("__type__") == "tuple":
                return tuple(convert_config_value(v) for v in x["items"])
            return {k: convert_config_value(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert_config_value(v) for v in x]
        return x

    env_kwargs = convert_config_value(env_kwargs)
    env_kwargs = cast(Mapping[str, Any], env_kwargs)

    def make_thunk(rank: int):
        def thunk():
            if rank == 0 and capture_video:
                assert (
                    video_folder is not None
                ), "video_folder is required for capture_video."
                env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
                env = gym.wrappers.RecordVideo(env, video_folder=str(video_folder))
            elif rank == 0 and human_render:
                env = gym.make(env_id, render_mode="human", **env_kwargs)
            else:
                env = gym.make(env_id, **env_kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if flatten_observation:
                env = gym.wrappers.FlattenObservation(env)
            if normalize_pixel_observation:
                env = NormalizePixelObservation(env)
            # Seed spaces (optional but good practice)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            # Note: actual env RNG is seeded on reset()
            env.reset(seed=seed + rank)
            return env

        return thunk

    return gym.vector.SyncVectorEnv(
        [make_thunk(i) for i in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
