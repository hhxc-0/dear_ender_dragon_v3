# Create env, seed reset(), optionally record stats/video in eval

from typing import Optional, Union
from pathlib import Path
import gymnasium as gym


def make_env(id: str, n_envs: int = 1, seed: int = 42, capture_video: bool = False, video_folder: Optional[Union[str, Path]] = None, human_render: bool = False):
    assert not (capture_video and human_render), "capture_video and human_render cannot both be set to True."
    def make_thunk(rank:int):
        def thunk():
            if rank == 0 and capture_video:
                assert video_folder is not None, "video_folder is required for capture_video."
                env = gym.make(id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, video_folder=str(video_folder))
            elif rank == 0 and human_render:
                env = gym.make(id, render_mode="human")
            else:
                env = gym.make(id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # Seed spaces (optional but good practice)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            # Note: actual env RNG is seeded on reset()
            env.reset(seed=seed + rank)
            return env
        return thunk

    return gym.vector.SyncVectorEnv([make_thunk(i) for i in range(n_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
