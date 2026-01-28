# Store rollout + compute GAE/returns with correct terminated/truncated handling

from __future__ import annotations

from typing import Sequence
import torch
from torch import zeros, Tensor


class RolloutBuffer:
    def __init__(
        self,
        T: int,  # maximum rollout length
        N: int,  # number of envs
        obs_shape: Sequence[int],
        obs_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
        action_shape: Sequence[int] = (),
        device: str = "cpu",
    ) -> None:
        self.T, self.N = T, N
        self.t = 0  # current rollout length

        self.obs = zeros(
            (T, N, *obs_shape), dtype=obs_dtype, device=device
        )  # [T, N, *obs_shape]: observation used to choose the action.
        self.actions = zeros(
            (T, N, *action_shape), dtype=torch.int64, device=device
        )  # [T, N, *action_shape]: sampled action.
        self.logp = zeros(
            (T, N), dtype=torch.float32, device=device
        )  # [T, N]: log-prob of the sampled action under the behavior policy (the policy that collected the rollout).
        self.values = zeros(
            (T, N), dtype=torch.float32, device=device
        )  # [T, N]: critic value estimate at collection time.
        self.rewards = zeros(
            (T, N), dtype=reward_dtype, device=device
        )  # [T, N]: reward.
        self.terminated = zeros(
            (T, N), dtype=torch.bool, device=device
        )  # [T, N]: environment termination flag (true terminal).
        self.truncated = zeros(
            (T, N), dtype=torch.bool, device=device
        )  # [T, N]: truncation flag (terminated due to time limit or other reasons).
        self.dones = zeros(
            (T, N), dtype=torch.bool, device=device
        )  # [T, N], terminated or truncated
        self.timeouts = zeros(
            (T, N), dtype=torch.bool, device=device
        )  # [T, N]: time out flag, truncations truely because of time limit (only bootstrap for these, as truncations can have other reasons).
        self.episode_start = zeros(
            (T, N), dtype=torch.bool, device=device
        )  # [T, N], true when a new episode begins at that step; helps later with RNNs/sequence packing.

        # at end of rollout
        self.last_obs = None  # [N, *obs_shape]: last_obs for clarity (or just keep the current obs after the loop).
        self.last_value = None  # [N]: for bootstrapping GAE.

        # compute and store after rollout (derived)
        self.advantages = None  # [T, N]
        self.returns = None  # [T, N], Bootstrapped λ-return targets for critic training (GAE-based), not raw episode total reward.

    def reset(self) -> None:
        self.t = 0
        self.advantages = None
        self.returns = None

    def add(
        self,
        obs: Tensor,
        action: Tensor,
        logp: Tensor,
        value: Tensor,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        done: Tensor,
        timeout: Tensor,
        episode_start: Tensor,
    ) -> None:
        assert self.t < self.T, "Rollout buffer is full"
        self.obs[self.t].copy_(obs)
        self.actions[self.t].copy_(action)
        self.logp[self.t].copy_(logp)
        self.values[self.t].copy_(value)
        self.rewards[self.t].copy_(reward)
        self.terminated[self.t].copy_(terminated)
        self.truncated[self.t].copy_(truncated)
        self.dones[self.t].copy_(done)
        self.timeouts[self.t].copy_(timeout)
        self.episode_start[self.t].copy_(episode_start)
        self.t += 1

    def add_last(
        self,
    ) -> None:
        raise NotImplementedError()

    def process(self):
        # TODO: preprocess
        # TODO: compute GAE
        raise NotImplementedError()
