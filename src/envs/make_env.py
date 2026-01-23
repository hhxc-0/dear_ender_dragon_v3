# Create env, seed reset(), optionally record stats/video in eval

import gymnasium as gym


def make_env(id: str, n_envs: int = 1, seed: int = 42):
    def make_thunk(rank:int):
        def thunk():
            env = gym.make(id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # Seed spaces (optional but good practice)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            # Note: actual env RNG is seeded on reset()
            env.reset(seed=seed + rank)
            return env
        return thunk

    return gym.vector.SyncVectorEnv([make_thunk(i) for i in range(n_envs)])
