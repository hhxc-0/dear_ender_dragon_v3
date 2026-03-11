from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import Wrapper, ObservationWrapper, ActionWrapper, Env
from gymnasium.spaces import Box


class NormalizePixelObservation(ObservationWrapper):
    """Normalize pixel observations from [0, 255] to [0, 1]."""

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        obs_space = self.observation_space
        assert isinstance(
            obs_space, Box
        ), f"Expected Box observation space, got {type(obs_space)}"
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=obs_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return observation / 255.0
