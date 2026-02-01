# Store rollout + compute GAE/returns with correct terminated/truncated handling

from __future__ import annotations

from typing import Sequence, Union, Generator
import torch
from torch import zeros, Tensor
import numpy as np

from src.types import MiniBatch


class RolloutBuffer:
    def __init__(
        self,
        T: int,  # maximum rollout length
        B: int,  # number of envs
        obs_shape: Sequence[int],
        obs_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
        action_shape: Sequence[int] = (),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.T, self.B = T, B
        self.t = 0  # current rollout length
        self.device = device

        self.obs = zeros(
            (T, B, *obs_shape), dtype=obs_dtype, device=device
        )  # [T, B, *obs_shape]: observation used to choose the action.
        self.actions = zeros(
            (T, B, *action_shape), dtype=torch.int64, device=device
        )  # [T, B, *action_shape]: sampled action.
        self.logp = zeros(
            (T, B), dtype=torch.float32, device=device
        )  # [T, B]: log-prob of the sampled action under the behavior policy (the policy that collected the rollout).
        self.values = zeros(
            (T, B), dtype=torch.float32, device=device
        )  # [T, B]: value estimate by critic at collection time.
        self.next_values = zeros(
            (T, B), dtype=torch.float32, device=device
        )  # [T, B]: next value estimated by critic at collection time. partially filled during rollout and completed after rollout. additional to values because VectorEnv auto-reset will return obs after reset instead of current episode.
        self.rewards = zeros(
            (T, B), dtype=reward_dtype, device=device
        )  # [T, B]: reward.
        self.terminated = zeros(
            (T, B), dtype=torch.bool, device=device
        )  # [T, B]: environment termination flag (true terminal).
        self.truncated = zeros(
            (T, B), dtype=torch.bool, device=device
        )  # [T, B]: truncation flag (terminated due to time limit or other reasons).
        self.dones = zeros(
            (T, B), dtype=torch.bool, device=device
        )  # [T, B], terminated or truncated
        self.timeouts = zeros(
            (T, B), dtype=torch.bool, device=device
        )  # [T, B]: time out flag, truncations truely because of time limit (only bootstrap for these, as truncations can have other reasons).
        self.episode_start = zeros(
            (T, B), dtype=torch.bool, device=device
        )  # [T, B], true when a new episode begins at that step; helps later with RNNs/sequence packing.

        # at end of rollout (not needed since having next_values)
        # self.last_obs = None  # [B, *obs_shape]: last_obs for clarity (or just keep the current obs after the loop).
        # self.last_value = None  # [B]: for bootstrapping GAE.

        # compute and store after rollout (derived)
        self.advantages = None  # [T, B]
        self.returns = None  # [T, B], Bootstrapped λ-return targets for critic training (GAE-based), not raw episode total reward.
        self.advantages_norm = None  # [T, B], Normalized advantages

    def reset(self) -> None:
        self.t = 0
        self.advantages = None
        self.returns = None
        self.advantages_norm = None

    def add(
        self,
        obs: Tensor,
        action: Tensor,
        logp: Tensor,
        value: Tensor,
        next_value: Tensor,
        reward: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        done: Tensor,
        timeout: Tensor,
        episode_start: Tensor,
    ) -> None:
        assert self.t < self.T, "rollout buffer is full"
        self.obs[self.t].copy_(obs)
        self.actions[self.t].copy_(action)
        self.logp[self.t].copy_(logp)
        self.values[self.t].copy_(value)
        self.next_values[self.t].copy_(next_value)
        self.rewards[self.t].copy_(reward)
        self.terminated[self.t].copy_(terminated)
        self.truncated[self.t].copy_(truncated)
        self.dones[self.t].copy_(done)
        self.timeouts[self.t].copy_(timeout)
        self.episode_start[self.t].copy_(episode_start)
        self.t += 1

    def finalize(self, last_value: Tensor) -> None:
        last_value = last_value.to(self.device)
        assert last_value.shape == (self.B,)
        mask = ~self.dones[: self.t - 1]  # [t-1, B]
        self.next_values[: self.t - 1][mask] = self.values[1 : self.t][mask]
        mask_last = ~self.dones[self.t - 1]  # [B]
        self.next_values[self.t - 1, mask_last] = last_value[mask_last]

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, normalize_advantages: bool, eps: float
    ) -> None:
        self.advantages = zeros(
            (self.T, self.B), dtype=torch.float32, device=self.device
        )
        gae = zeros(self.B, dtype=torch.float32, device=self.device)
        next_value = self.next_values[self.t - 1]
        done = self.dones  # terminated | truncated
        timeout = self.timeouts  # true only for time-limit truncation
        # Allow bootstrap unless it's (terminal or non-timeout truncation)
        bootstrap_td = (~(done & ~timeout)).float()  # shape [T, B]
        continue_gae = (~done).float()  # shape [T, B]
        for t in reversed(range(self.t)):
            delta = (
                self.rewards[t] + gamma * next_value * bootstrap_td[t] - self.values[t]
            )
            gae = delta + gamma * gae_lambda * continue_gae[t] * gae
            self.advantages[t] = gae
            next_value = self.values[t]
        self.returns = self.advantages + self.values
        if normalize_advantages:
            adv_flatten = self.advantages.reshape(-1)
            adv_mean = adv_flatten.mean()
            adv_std = adv_flatten.std(unbiased=False)
            self.advantages_norm = (self.advantages - adv_mean) / (adv_std + eps)
        else:
            self.advantages_norm = self.advantages

    def iter_minibatches(
        self, mini_batch_size: int, shuffle: bool
    ) -> Generator[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Yields MiniBatch"""

        # Flatten [T, B] -> [N]
        def flatten_TB(x: Tensor) -> Tensor:
            # x: [T, B, ...] -> [T*B, ...]
            return torch.flatten(x, 0, 1)

        if self.returns is None or self.advantages_norm is None:
            raise RuntimeError(
                "Must call compute_returns_and_advantages() before iter_minibatches()"
            )
        b_obs = flatten_TB(self.obs)
        b_actions = flatten_TB(self.actions)
        b_logp_old = flatten_TB(self.logp)
        b_values_old = flatten_TB(self.values)
        b_returns = flatten_TB(self.returns)
        b_adv_norm = flatten_TB(self.advantages_norm)
        # Shape checking
        N = self.t * self.B
        assert b_obs.shape[0] == N
        assert b_actions.shape[0] == N
        assert b_logp_old.shape[0] == N
        assert b_values_old.shape[0] == N
        assert b_returns.shape[0] == N
        assert b_adv_norm.shape[0] == N
        # Sample indices
        if shuffle:
            b_inds = torch.randperm(N, device=self.device)
        else:
            b_inds = torch.arange(N, device=self.device)
        # Yield mini-batches
        for start in range(0, len(b_inds), mini_batch_size):
            end = start + mini_batch_size
            mb_inds = b_inds[start:end]
            yield MiniBatch(
                obs=b_obs[mb_inds],
                actions=b_actions[mb_inds],
                logp_old=b_logp_old[mb_inds],
                values_old=b_values_old[mb_inds],
                returns=b_returns[mb_inds],
                advantages_norm=b_adv_norm[mb_inds],
            )
